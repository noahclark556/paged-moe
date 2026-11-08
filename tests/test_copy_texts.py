# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Host file bodies and tool results the verify drafter is allowed to copy."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pagedmoe.copy_texts import escaped_runs  # noqa: E402


def test_host_texts_come_first_and_duplicates_drop():
    body_text = "alpha\n" + ("beta " * 30)
    body = {
        "copy_texts": [body_text, "no-newline-no-quote-" + ("z" * 80)],
        "messages": [
            {"role": "tool", "content": body_text},
            {"role": "tool", "content": "short"},
            {"role": "user", "content": "ignore"},
        ],
    }
    runs = escaped_runs(body, limit_chars=10_000, limit_runs=12)
    assert len(runs) == 1, runs
    assert "\\n" in runs[0]
    # Plain text with nothing to escape is not a copy source: the prompt
    # n-gram already matches it.
    assert "zzzz" not in runs[0]


def test_newest_tool_result_fits_when_the_host_list_is_empty():
    text = "line\n" + ("word " * 40)
    body = {"messages": [{"role": "tool", "content": text}]}
    runs = escaped_runs(body, limit_chars=10_000, limit_runs=12)
    assert len(runs) == 1 and "\\n" in runs[0]


if __name__ == "__main__":
    test_host_texts_come_first_and_duplicates_drop()
    test_newest_tool_result_fits_when_the_host_list_is_empty()
    print("OK: copy texts")
