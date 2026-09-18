/*
 * Copyright (C) 2026 Noah Clark
 * SPDX-License-Identifier: AGPL-3.0-or-later
 *
 * Native expert-read pool: one job per expert, GIL released for the latch.
 * Same bytes as ExpertCache._pread_full; only scheduling changes.
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
} Job;

typedef struct {
    pthread_t *threads;
    int n_threads;
    int stop;
    pthread_mutex_t mu;
    pthread_cond_t cv;
    Job **queue;
    size_t q_cap;
    size_t q_head;
    size_t q_tail;
    size_t q_len;
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

/* Same short-read loop as Python _pread_full / os.preadv. */
static int
pread_full(int fd, off_t offset, char *ptr, size_t len, char *err, size_t errlen)
{
    size_t done = 0;
    while (done < len) {
        ssize_t n = pread(fd, ptr + done, len - done, offset + (off_t)done);
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
queue_pop(Pool *pool)
{
    Job *job;
    if (pool->q_len == 0) {
        return NULL;
    }
    job = pool->queue[pool->q_head];
    pool->queue[pool->q_head] = NULL;
    pool->q_head = (pool->q_head + 1) % pool->q_cap;
    pool->q_len--;
    return job;
}

static int
queue_push(Pool *pool, Job *job)
{
    if (pool->q_len == pool->q_cap) {
        size_t new_cap = pool->q_cap ? pool->q_cap * 2 : 64;
        Job **linear = (Job **)malloc(new_cap * sizeof(Job *));
        size_t i;
        if (linear == NULL) {
            return -1;
        }
        for (i = 0; i < pool->q_len; i++) {
            linear[i] = pool->queue[(pool->q_head + i) % pool->q_cap];
        }
        for (i = pool->q_len; i < new_cap; i++) {
            linear[i] = NULL;
        }
        free(pool->queue);
        pool->queue = linear;
        pool->q_cap = new_cap;
        pool->q_head = 0;
        pool->q_tail = pool->q_len;
    }
    pool->queue[pool->q_tail] = job;
    pool->q_tail = (pool->q_tail + 1) % pool->q_cap;
    pool->q_len++;
    return 0;
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
        char err[256];
        int i;
        int failed;

        pthread_mutex_lock(&pool->mu);
        while (pool->q_len == 0 && !pool->stop) {
            pthread_cond_wait(&pool->cv, &pool->mu);
        }
        if (pool->stop && pool->q_len == 0) {
            pthread_mutex_unlock(&pool->mu);
            return NULL;
        }
        job = queue_pop(pool);
        pthread_mutex_unlock(&pool->mu);
        if (job == NULL) {
            continue;
        }

        pool->busy[idx] = 1;
        failed = 0;
        err[0] = '\0';
        for (i = 0; i < job->n; i++) {
            Comp *c = &job->comps[i];
            if (pread_full(c->fd, c->offset, c->ptr, c->len, err, sizeof(err)) != 0) {
                failed = 1;
                break;
            }
        }
        pool->busy[idx] = 0;
        finish_job(job, failed ? err : NULL);
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
    while (pool->q_len > 0) {
        Job *job = queue_pop(pool);
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
    free(pool->queue);
    free(pool->threads);
    free(pool->busy);
    pthread_mutex_destroy(&pool->mu);
    pthread_cond_destroy(&pool->cv);
    free(pool);
}

static void
pool_capsule_destructor(PyObject *capsule)
{
    Pool *pool = (Pool *)PyCapsule_GetPointer(capsule, "expert_stream.NativeReadPool");
    if (pool != NULL) {
        pool_dealloc(pool);
    }
}

static PyObject *
native_create_pool(PyObject *self, PyObject *args)
{
    int n_threads;
    Pool *pool;
    int i;

    (void)self;
    if (!PyArg_ParseTuple(args, "i", &n_threads)) {
        return NULL;
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
    pool->threads = (pthread_t *)calloc((size_t)n_threads, sizeof(pthread_t));
    pool->busy = (unsigned char *)calloc((size_t)n_threads, 1);
    pool->q_cap = 64;
    pool->queue = (Job **)calloc(pool->q_cap, sizeof(Job *));
    if (pool->threads == NULL || pool->busy == NULL || pool->queue == NULL) {
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

    return PyCapsule_New(pool, "expert_stream.NativeReadPool", pool_capsule_destructor);
}

static Pool *
pool_from_capsule(PyObject *obj)
{
    Pool *pool = (Pool *)PyCapsule_GetPointer(obj, "expert_stream.NativeReadPool");
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
    Pool *pool;
    Job *job;
    Py_ssize_t n;
    Py_ssize_t i;

    (void)self;
    if (!PyArg_ParseTuple(args, "OOO", &capsule, &comps, &latch)) {
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
    if (queue_push(pool, job) != 0) {
        pthread_mutex_unlock(&pool->mu);
        free_job(job);
        return PyErr_NoMemory();
    }
    pthread_cond_signal(&pool->cv);
    pthread_mutex_unlock(&pool->mu);
    Py_RETURN_NONE;
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
     "create_pool(n_threads) -> capsule"},
    {"submit_expert", native_submit_expert, METH_VARARGS,
     "submit_expert(pool, [(fd, offset, mv), ...], latch)"},
    {"shutdown", native_shutdown, METH_VARARGS, "shutdown(pool)"},
    {"busy_view", native_busy_view, METH_VARARGS, "busy_view(pool) -> memoryview"},
    {"pread_full", native_pread_full, METH_VARARGS,
     "pread_full(fd, offset, writable_buffer)"},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef native_module = {
    PyModuleDef_HEAD_INIT,
    "expert_stream._native_read",
    "GIL-free expert pread pool for PagedMoE.",
    -1,
    NativeMethods,
};

PyMODINIT_FUNC
PyInit__native_read(void)
{
    return PyModule_Create(&native_module);
}
