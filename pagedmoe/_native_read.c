/*
 * Copyright (C) 2026 Noah Clark
 * SPDX-License-Identifier: AGPL-3.0-or-later
 *
 * Native expert-read pool: one job per expert, GIL released for the latch.
 * Same bytes as ExpertCache._pread_full; only scheduling changes.
 *
 * A job is split into pieces of at most split_bytes that any idle worker can
 * claim, so one miss runs at the drive's queue depth instead of QD1. A 32 MB
 * MiniMax expert read by one worker took ~9 ms (3.5 GB/s); the drive does
 * 12-13 GB/s at 1 MB x QD16. Urgent jobs (demand misses) have their own queue
 * and are drained before speculative ones. A speculative job a layer starts
 * waiting on is moved to the urgent queue (promote).
 *
 * Keep the short-read loop. A single pread looks cleaner and fails under
 * load on APFS the same way the Python path used to.
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#ifndef NATIVE_READ_MAX_COMPONENTS
#define NATIVE_READ_MAX_COMPONENTS 32
#endif

#ifndef NATIVE_READ_MAX_THREADS
#define NATIVE_READ_MAX_THREADS 256
#endif

typedef struct {
    int fd;
    off_t offset;
    char *ptr;
    size_t len;
    PyObject *mv_ref; /* keep buffer alive until job finishes */
} Comp;

typedef struct {
    int n;
    Comp comps[NATIVE_READ_MAX_COMPONENTS];
    PyObject *latch; /* ExpertCache._SlotLatch */
    /* Claim cursor and completion count; guarded by Pool.mu. */
    int next_comp;
    size_t next_off;
    int outstanding;
    int claimed;
    int failed;
    char err[256];
} Job;

typedef struct {
    Job **buf;
    size_t cap;
    size_t head;
    size_t tail;
    size_t len;
} Queue;

typedef struct {
    pthread_t *threads;
    int n_threads;
    int stop;
    pthread_mutex_t mu;
    pthread_cond_t cv;
    Queue urgent;
    Queue normal;
    size_t split_bytes; /* 0 = one worker per job */
    unsigned char *busy; /* length n_threads; worker i writes busy[i] */
    int started;
} Pool;

static void
free_job(Job *job)
{
    int i;
    if (job == NULL) {
        return;
    }
    for (i = 0; i < job->n; i++) {
        Py_XDECREF(job->comps[i].mv_ref);
        job->comps[i].mv_ref = NULL;
    }
    Py_XDECREF(job->latch);
    job->latch = NULL;
    free(job);
}

/* F_NOCACHE only bypasses the page cache for preads up to 1 MB on macOS.
 * Bigger ones land in the file cache (16 MB reads: ~all of it), which on a
 * long prefill is tens of GB competing with the slab. Must match
 * safetensors_index.MAX_IO_BYTES. */
#define MAX_IO_BYTES ((size_t)1 << 20)

/* Same short-read loop as Python _pread_full / os.preadv. */
static int
pread_full(int fd, off_t offset, char *ptr, size_t len, char *err, size_t errlen)
{
    size_t done = 0;
    while (done < len) {
        size_t want = len - done;
        if (want > MAX_IO_BYTES) {
            want = MAX_IO_BYTES;
        }
        ssize_t n = pread(fd, ptr + done, want, offset + (off_t)done);
        if (n < 0) {
            if (errno == EINTR) {
                continue;
            }
            snprintf(err, errlen, "pread failed fd=%d off=%lld: %s",
                     fd, (long long)(offset + (off_t)done), strerror(errno));
            return -1;
        }
        if (n == 0) {
            snprintf(err, errlen, "short read at %lld (fd %d)",
                     (long long)(offset + (off_t)done), fd);
            return -1;
        }
        done += (size_t)n;
    }
    return 0;
}

static void
finish_job(Job *job, const char *err)
{
    PyGILState_STATE st = PyGILState_Ensure();
    PyObject *result = NULL;
    if (err != NULL) {
        PyObject *exc = PyObject_CallFunction(PyExc_OSError, "s", err);
        if (exc == NULL) {
            PyErr_Clear();
            exc = PyExc_OSError;
            Py_INCREF(exc);
        }
        result = PyObject_CallMethod(job->latch, "done", "O", exc);
        Py_DECREF(exc);
    } else {
        result = PyObject_CallMethod(job->latch, "done", NULL);
    }
    if (result == NULL) {
        /* Never abort the worker on a bad callback; surface and clear. */
        PyErr_WriteUnraisable(job->latch);
    } else {
        Py_DECREF(result);
    }
    free_job(job);
    PyGILState_Release(st);
}

static Job *
queue_pop(Queue *q)
{
    Job *job;
    if (q->len == 0) {
        return NULL;
    }
    job = q->buf[q->head];
    q->buf[q->head] = NULL;
    q->head = (q->head + 1) % q->cap;
    q->len--;
    return job;
}

static int
queue_push(Queue *q, Job *job)
{
    if (q->len == q->cap) {
        size_t new_cap = q->cap ? q->cap * 2 : 64;
        Job **linear = (Job **)malloc(new_cap * sizeof(Job *));
        size_t i;
        if (linear == NULL) {
            return -1;
        }
        for (i = 0; i < q->len; i++) {
            linear[i] = q->buf[(q->head + i) % q->cap];
        }
        for (i = q->len; i < new_cap; i++) {
            linear[i] = NULL;
        }
        free(q->buf);
        q->buf = linear;
        q->cap = new_cap;
        q->head = 0;
        q->tail = q->len;
    }
    q->buf[q->tail] = job;
    q->tail = (q->tail + 1) % q->cap;
    q->len++;
    return 0;
}

static int
queue_init(Queue *q)
{
    q->cap = 64;
    q->head = q->tail = q->len = 0;
    q->buf = (Job **)calloc(q->cap, sizeof(Job *));
    return q->buf == NULL ? -1 : 0;
}

/* Assumes pool->mu held and job at the head of q. Claims the next piece:
 * the rest of the current component capped at split_bytes, or every
 * remaining component when split_bytes is 0. Pops the job once nothing is
 * left to claim. */
static void
claim_piece(Pool *pool, Queue *q, Job *job, int *first, int *last,
            size_t *off, size_t *len)
{
    Comp *c = &job->comps[job->next_comp];
    if (pool->split_bytes == 0) {
        *first = job->next_comp;
        *last = job->n - 1;
        *off = job->next_off;
        *len = 0;
        job->next_comp = job->n;
    } else {
        size_t left = c->len - job->next_off;
        size_t take = left < pool->split_bytes ? left : pool->split_bytes;
        *first = *last = job->next_comp;
        *off = job->next_off;
        *len = take;
        job->next_off += take;
        if (job->next_off >= c->len) {
            job->next_comp++;
            job->next_off = 0;
        }
    }
    job->outstanding++;
    if (job->next_comp >= job->n) {
        job->claimed = 1;
        queue_pop(q);
    }
}

static void *
worker_main(void *arg)
{
    void **pack = (void **)arg;
    Pool *pool = (Pool *)pack[0];
    int idx = (int)(intptr_t)pack[1];
    free(pack);

    for (;;) {
        Job *job;
        Queue *q;
        char err[256];
        int first, last, i, done;
        size_t off, len;
        int failed;

        pthread_mutex_lock(&pool->mu);
        while (pool->urgent.len == 0 && pool->normal.len == 0 && !pool->stop) {
            pthread_cond_wait(&pool->cv, &pool->mu);
        }
        if (pool->stop && pool->urgent.len == 0 && pool->normal.len == 0) {
            pthread_mutex_unlock(&pool->mu);
            return NULL;
        }
        q = pool->urgent.len ? &pool->urgent : &pool->normal;
        job = q->buf[q->head];
        claim_piece(pool, q, job, &first, &last, &off, &len);
        pthread_mutex_unlock(&pool->mu);

        pool->busy[idx] = 1;
        failed = 0;
        err[0] = '\0';
        if (len > 0) {
            Comp *c = &job->comps[first];
            failed = pread_full(c->fd, c->offset + (off_t)off, c->ptr + off, len,
                                err, sizeof(err)) != 0;
        } else {
            for (i = first; i <= last; i++) {
                Comp *c = &job->comps[i];
                if (c->len == 0) {
                    continue;
                }
                if (pread_full(c->fd, c->offset, c->ptr, c->len, err, sizeof(err)) != 0) {
                    failed = 1;
                    break;
                }
            }
        }
        pool->busy[idx] = 0;

        pthread_mutex_lock(&pool->mu);
        if (failed && !job->failed) {
            job->failed = 1;
            memcpy(job->err, err, sizeof(job->err));
        }
        job->outstanding--;
        done = job->claimed && job->outstanding == 0;
        pthread_mutex_unlock(&pool->mu);
        if (done) {
            finish_job(job, job->failed ? job->err : NULL);
        }
    }
}

static void
pool_dealloc(Pool *pool)
{
    int i;
    if (pool == NULL) {
        return;
    }
    if (pool->started) {
        pthread_mutex_lock(&pool->mu);
        pool->stop = 1;
        pthread_cond_broadcast(&pool->cv);
        pthread_mutex_unlock(&pool->mu);
        for (i = 0; i < pool->n_threads; i++) {
            pthread_join(pool->threads[i], NULL);
        }
    }
    Queue *queues[2] = {&pool->urgent, &pool->normal};
    int qi;
    for (qi = 0; qi < 2; qi++) {
        while (queues[qi]->len > 0) {
            Job *job = queue_pop(queues[qi]);
            if (job != NULL) {
                /* Drop without calling into Python: interpreter may be finalizing. */
                for (i = 0; i < job->n; i++) {
                    /* refs may be invalid at shut down; only free the struct. */
                    job->comps[i].mv_ref = NULL;
                }
                job->latch = NULL;
                free(job);
            }
        }
        free(queues[qi]->buf);
    }
    free(pool->threads);
    free(pool->busy);
    pthread_mutex_destroy(&pool->mu);
    pthread_cond_destroy(&pool->cv);
    free(pool);
}

static void
pool_capsule_destructor(PyObject *capsule)
{
    Pool *pool = (Pool *)PyCapsule_GetPointer(capsule, "pagedmoe.NativeReadPool");
    if (pool != NULL) {
        pool_dealloc(pool);
    }
}

static PyObject *
native_create_pool(PyObject *self, PyObject *args)
{
    int n_threads;
    long long split_bytes = (long long)MAX_IO_BYTES;
    Pool *pool;
    int i;

    (void)self;
    if (!PyArg_ParseTuple(args, "i|L", &n_threads, &split_bytes)) {
        return NULL;
    }
    if (split_bytes < 0) {
        split_bytes = 0;
    }
    if (n_threads < 1) {
        n_threads = 1;
    }
    if (n_threads > NATIVE_READ_MAX_THREADS) {
        n_threads = NATIVE_READ_MAX_THREADS;
    }

    pool = (Pool *)calloc(1, sizeof(Pool));
    if (pool == NULL) {
        return PyErr_NoMemory();
    }
    pool->n_threads = n_threads;
    pool->split_bytes = (size_t)split_bytes;
    pool->threads = (pthread_t *)calloc((size_t)n_threads, sizeof(pthread_t));
    pool->busy = (unsigned char *)calloc((size_t)n_threads, 1);
    if (pool->threads == NULL || pool->busy == NULL ||
        queue_init(&pool->urgent) != 0 || queue_init(&pool->normal) != 0) {
        pool_dealloc(pool);
        return PyErr_NoMemory();
    }
    if (pthread_mutex_init(&pool->mu, NULL) != 0 ||
        pthread_cond_init(&pool->cv, NULL) != 0) {
        pool_dealloc(pool);
        PyErr_SetString(PyExc_RuntimeError, "native read pool: mutex/cond init failed");
        return NULL;
    }

    for (i = 0; i < n_threads; i++) {
        void **pack = (void **)malloc(2 * sizeof(void *));
        if (pack == NULL) {
            pool->stop = 1;
            pthread_cond_broadcast(&pool->cv);
            while (--i >= 0) {
                pthread_join(pool->threads[i], NULL);
            }
            pool->started = 0;
            pool_dealloc(pool);
            return PyErr_NoMemory();
        }
        pack[0] = pool;
        pack[1] = (void *)(intptr_t)i;
        if (pthread_create(&pool->threads[i], NULL, worker_main, pack) != 0) {
            free(pack);
            pool->stop = 1;
            pthread_cond_broadcast(&pool->cv);
            while (--i >= 0) {
                pthread_join(pool->threads[i], NULL);
            }
            pool->started = 0;
            pool_dealloc(pool);
            PyErr_SetString(PyExc_RuntimeError, "native read pool: pthread_create failed");
            return NULL;
        }
    }
    pool->started = 1;

    return PyCapsule_New(pool, "pagedmoe.NativeReadPool", pool_capsule_destructor);
}

static Pool *
pool_from_capsule(PyObject *obj)
{
    Pool *pool = (Pool *)PyCapsule_GetPointer(obj, "pagedmoe.NativeReadPool");
    if (pool == NULL) {
        PyErr_SetString(PyExc_TypeError, "expected NativeReadPool capsule");
    }
    return pool;
}

static PyObject *
native_submit_expert(PyObject *self, PyObject *args)
{
    PyObject *capsule;
    PyObject *comps;
    PyObject *latch;
    int urgent = 0;
    Pool *pool;
    Job *job;
    Py_ssize_t n;
    Py_ssize_t i;

    (void)self;
    if (!PyArg_ParseTuple(args, "OOO|p", &capsule, &comps, &latch, &urgent)) {
        return NULL;
    }
    pool = pool_from_capsule(capsule);
    if (pool == NULL) {
        return NULL;
    }
    if (!PyList_Check(comps)) {
        PyErr_SetString(PyExc_TypeError, "components must be a list of (fd, offset, memoryview)");
        return NULL;
    }
    n = PyList_GET_SIZE(comps);
    if (n <= 0 || n > NATIVE_READ_MAX_COMPONENTS) {
        PyErr_Format(PyExc_ValueError, "component count %zd out of range", (Py_ssize_t)n);
        return NULL;
    }

    job = (Job *)calloc(1, sizeof(Job));
    if (job == NULL) {
        return PyErr_NoMemory();
    }
    job->n = (int)n;
    Py_INCREF(latch);
    job->latch = latch;

    for (i = 0; i < n; i++) {
        PyObject *item = PyList_GET_ITEM(comps, i);
        int fd;
        long long offset;
        PyObject *buf_obj;
        Py_buffer view;

        if (!PyArg_ParseTuple(item, "iLO", &fd, &offset, &buf_obj)) {
            free_job(job);
            return NULL;
        }
        if (PyObject_GetBuffer(buf_obj, &view, PyBUF_SIMPLE | PyBUF_WRITABLE) != 0) {
            free_job(job);
            return NULL;
        }
        if (!PyBuffer_IsContiguous(&view, 'C')) {
            PyBuffer_Release(&view);
            free_job(job);
            PyErr_SetString(PyExc_ValueError, "component buffer must be C-contiguous");
            return NULL;
        }
        job->comps[i].fd = fd;
        job->comps[i].offset = (off_t)offset;
        job->comps[i].ptr = (char *)view.buf;
        job->comps[i].len = (size_t)view.len;
        Py_INCREF(buf_obj);
        job->comps[i].mv_ref = buf_obj;
        PyBuffer_Release(&view);
    }

    pthread_mutex_lock(&pool->mu);
    if (pool->stop) {
        pthread_mutex_unlock(&pool->mu);
        free_job(job);
        PyErr_SetString(PyExc_RuntimeError, "native read pool is shut down");
        return NULL;
    }
    if (queue_push(urgent ? &pool->urgent : &pool->normal, job) != 0) {
        pthread_mutex_unlock(&pool->mu);
        free_job(job);
        return PyErr_NoMemory();
    }
    /* A split job can keep every worker busy, so wake them all. */
    if (pool->split_bytes) {
        pthread_cond_broadcast(&pool->cv);
    } else {
        pthread_cond_signal(&pool->cv);
    }
    pthread_mutex_unlock(&pool->mu);
    Py_RETURN_NONE;
}

/* Move a queued speculative job to the urgent queue: a layer is now blocked
 * on it, and behind the other guesses it waits for all of them. A job whose
 * pieces are all claimed already is left alone. Returns True if moved. */
static PyObject *
native_promote(PyObject *self, PyObject *args)
{
    PyObject *capsule;
    PyObject *latch;
    Pool *pool;
    Queue *q;
    size_t i, j;
    int moved = 0;

    (void)self;
    if (!PyArg_ParseTuple(args, "OO", &capsule, &latch)) {
        return NULL;
    }
    pool = pool_from_capsule(capsule);
    if (pool == NULL) {
        return NULL;
    }
    pthread_mutex_lock(&pool->mu);
    q = &pool->normal;
    for (i = 0; i < q->len; i++) {
        size_t at = (q->head + i) % q->cap;
        Job *job = q->buf[at];
        if (job == NULL || job->latch != latch) {
            continue;
        }
        for (j = i; j + 1 < q->len; j++) {
            q->buf[(q->head + j) % q->cap] = q->buf[(q->head + j + 1) % q->cap];
        }
        q->tail = (q->tail + q->cap - 1) % q->cap;
        q->buf[q->tail] = NULL;
        q->len--;
        if (queue_push(&pool->urgent, job) != 0) {
            /* Out of memory: put it back at the end of the normal queue,
             * which cannot fail since we just freed a slot there. */
            queue_push(q, job);
        } else {
            moved = 1;
        }
        break;
    }
    if (moved) {
        pthread_cond_broadcast(&pool->cv);
    }
    pthread_mutex_unlock(&pool->mu);
    return PyBool_FromLong(moved);
}

static PyObject *
native_shutdown(PyObject *self, PyObject *args)
{
    PyObject *capsule;
    Pool *pool;
    int i;

    (void)self;
    if (!PyArg_ParseTuple(args, "O", &capsule)) {
        return NULL;
    }
    pool = pool_from_capsule(capsule);
    if (pool == NULL) {
        return NULL;
    }
    if (!pool->started) {
        Py_RETURN_NONE;
    }
    pthread_mutex_lock(&pool->mu);
    pool->stop = 1;
    pthread_cond_broadcast(&pool->cv);
    pthread_mutex_unlock(&pool->mu);
    Py_BEGIN_ALLOW_THREADS
    for (i = 0; i < pool->n_threads; i++) {
        pthread_join(pool->threads[i], NULL);
    }
    Py_END_ALLOW_THREADS
    pool->started = 0;
    /* Prevent destructor from joining again. */
    free(pool->threads);
    pool->threads = NULL;
    Py_RETURN_NONE;
}

static PyObject *
native_busy_view(PyObject *self, PyObject *args)
{
    PyObject *capsule;
    Pool *pool;
    PyObject *mv;

    (void)self;
    if (!PyArg_ParseTuple(args, "O", &capsule)) {
        return NULL;
    }
    pool = pool_from_capsule(capsule);
    if (pool == NULL) {
        return NULL;
    }
    mv = PyMemoryView_FromMemory((char *)pool->busy, (Py_ssize_t)pool->n_threads, PyBUF_WRITE);
    return mv;
}

static PyObject *
native_pread_full(PyObject *self, PyObject *args)
{
    int fd;
    long long offset;
    PyObject *buf_obj;
    Py_buffer view;
    char err[256];
    int rc;

    (void)self;
    if (!PyArg_ParseTuple(args, "iLO", &fd, &offset, &buf_obj)) {
        return NULL;
    }
    if (PyObject_GetBuffer(buf_obj, &view, PyBUF_SIMPLE | PyBUF_WRITABLE) != 0) {
        return NULL;
    }
    if (!PyBuffer_IsContiguous(&view, 'C')) {
        PyBuffer_Release(&view);
        PyErr_SetString(PyExc_ValueError, "buffer must be C-contiguous");
        return NULL;
    }
    Py_BEGIN_ALLOW_THREADS
    rc = pread_full(fd, (off_t)offset, (char *)view.buf, (size_t)view.len, err, sizeof(err));
    Py_END_ALLOW_THREADS
    PyBuffer_Release(&view);
    if (rc != 0) {
        PyErr_SetString(PyExc_OSError, err);
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyMethodDef NativeMethods[] = {
    {"create_pool", native_create_pool, METH_VARARGS,
     "create_pool(n_threads, split_bytes=1 MiB) -> capsule"},
    {"submit_expert", native_submit_expert, METH_VARARGS,
     "submit_expert(pool, [(fd, offset, mv), ...], latch, urgent=False)"},
    {"promote", native_promote, METH_VARARGS,
     "promote(pool, latch) -> bool: move a queued job to the urgent queue"},
    {"shutdown", native_shutdown, METH_VARARGS, "shutdown(pool)"},
    {"busy_view", native_busy_view, METH_VARARGS, "busy_view(pool) -> memoryview"},
    {"pread_full", native_pread_full, METH_VARARGS,
     "pread_full(fd, offset, writable_buffer)"},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef native_module = {
    PyModuleDef_HEAD_INIT,
    "pagedmoe._native_read",
    "GIL-free expert pread pool for PagedMoE.",
    -1,
    NativeMethods,
};

PyMODINIT_FUNC
PyInit__native_read(void)
{
    PyObject *m = PyModule_Create(&native_module);
    if (m != NULL && PyModule_AddIntConstant(m, "API_VERSION", 3) != 0) {
        Py_DECREF(m);
        return NULL;
    }
    return m;
}
