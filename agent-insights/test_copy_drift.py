#!/usr/bin/env python3
"""
Tests for check_copy_drift.py.

A drift detector that cannot detect drift is worse than none: it turns "nobody
checked" into "CI says it's fine". So these feed it deliberately drifted copies
and assert it fails, in every direction drift can happen — the same way the
`_verify_*` rejection suite is kept non-vacuous.
"""

import os

import check_copy_drift as D


def _mk(d, files):
    os.makedirs(d, exist_ok=True)
    for name, body in files.items():
        with open(os.path.join(d, name), "w", encoding="utf-8") as fh:
            fh.write(body)
    return d


def test_identical_copies_pass(tmp_path):
    files = {"extract.py": "x = 1\n", "report.py": "y = 2\n"}
    a = _mk(str(tmp_path / "a"), files)
    b = _mk(str(tmp_path / "b"), files)
    assert D.compare(a, b) == []
    assert D.main(["check_copy_drift.py", a, b]) == 0


def test_content_drift_is_caught(tmp_path):
    a = _mk(str(tmp_path / "a"), {"extract.py": "x = 1\n"})
    b = _mk(str(tmp_path / "b"), {"extract.py": "x = 2\n"})
    problems = D.compare(a, b)
    assert problems, "a changed line was not detected"
    assert any("extract.py" in p and "differs" in p for p in problems)
    assert D.main(["check_copy_drift.py", a, b]) == 1


def test_a_file_added_to_only_one_side_is_caught_in_both_directions(tmp_path):
    a = _mk(str(tmp_path / "a"), {"extract.py": "x = 1\n", "newthing.py": "z = 3\n"})
    b = _mk(str(tmp_path / "b"), {"extract.py": "x = 1\n"})
    assert any("newthing.py" in p for p in D.compare(a, b))
    # and the reverse orientation, so neither repo is the privileged one
    assert any("newthing.py" in p for p in D.compare(b, a))
    assert D.main(["check_copy_drift.py", a, b]) == 1


def test_whitespace_only_drift_is_still_drift(tmp_path):
    """Byte-identical means byte-identical. A reformat on one side only is
    exactly the kind of silent divergence this exists to stop."""
    a = _mk(str(tmp_path / "a"), {"extract.py": "x = 1\n"})
    b = _mk(str(tmp_path / "b"), {"extract.py": "x  =  1\n"})
    assert D.compare(a, b)


def test_non_python_files_are_ignored(tmp_path):
    """The standalone repo carries CLAUDE.md, REMEDIATION.md and its own README
    that the cortex-mind copy does not. Only the tracked .py files are mirrored."""
    a = _mk(str(tmp_path / "a"), {"extract.py": "x = 1\n", "README.md": "standalone\n"})
    b = _mk(str(tmp_path / "b"), {"extract.py": "x = 1\n", "README.md": "embedded\n"})
    assert D.compare(a, b) == []
