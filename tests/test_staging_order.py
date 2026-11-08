# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Staging overflow keeps the prefetch the next layer will block on."""

from pagedmoe.cache import ExpertCache


def _cache():
    return ExpertCache(budget_bytes=1 << 30, staging_bytes=100, read_threads=1)


def test_trim_drops_furthest_tagged_guess():
    c = _cache()
    try:
        c._raw_ready[("L", 1)] = {"__nbytes__": 60, "__need_at__": 3}
        c._raw_ready[("L", 2)] = {"__nbytes__": 60, "__need_at__": 30}
        c._raw_bytes = 120
        c._trim_staging()
        assert ("L", 1) in c._raw_ready
        assert ("L", 2) not in c._raw_ready
        assert c._raw_bytes == 60
    finally:
        c._executor.shutdown(wait=False)


def test_full_staging_skips_instead_of_discarding():
    c = _cache()
    try:
        c._raw_ready[("L", 1)] = {"__nbytes__": 40, "__need_at__": 4}
        c._raw_ready[("L", 3)] = {"__nbytes__": 40, "__need_at__": 8}
        c._raw_ready[("L", 2)] = {"__nbytes__": 40, "__need_at__": 40}
        c._raw_bytes = 120
        assert c._spec_blocked(need_at=6) is True
        assert ("L", 1) in c._raw_ready
        assert ("L", 2) in c._raw_ready
        assert ("L", 3) in c._raw_ready
    finally:
        c._executor.shutdown(wait=False)


def test_later_prefetch_does_not_evict_sooner():
    c = _cache()
    try:
        c._raw_ready[("L", 1)] = {"__nbytes__": 100, "__need_at__": 3}
        c._raw_bytes = 100
        assert c._spec_blocked(need_at=40) is True
        assert ("L", 1) in c._raw_ready
        assert c._raw_bytes == 100
    finally:
        c._executor.shutdown(wait=False)


def test_untagged_overflow_still_drops_oldest():
    c = _cache()
    try:
        c._raw_ready[("L", 1)] = {"__nbytes__": 80}
        c._raw_ready[("L", 2)] = {"__nbytes__": 80}
        c._raw_bytes = 160
        c._trim_staging()
        assert ("L", 1) not in c._raw_ready
        assert ("L", 2) in c._raw_ready
    finally:
        c._executor.shutdown(wait=False)


if __name__ == "__main__":
    test_trim_drops_furthest_tagged_guess()
    test_full_staging_skips_instead_of_discarding()
    test_later_prefetch_does_not_evict_sooner()
    test_untagged_overflow_still_drops_oldest()
    print("ok")
