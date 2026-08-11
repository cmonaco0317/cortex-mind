#!/usr/bin/env python3
"""
Fail if the two copies of agent-insights have drifted apart.

This code exists identically in `cmonaco0317/agent-insights` (repo root) and in
`cmonaco0317/cortex-mind` (the `agent-insights/` subdirectory). Five commits had
already been applied twice by hand and nothing enforced agreement, so the only
thing keeping them in step was somebody remembering. Drift should turn CI red,
not be discovered later.

    python3 check_copy_drift.py <dir-a> <dir-b>

Compares the tracked `.py` files by exact bytes, in both directions, so a file
added to one side and not the other is a failure too. Exits non-zero on any
difference and prints what differs.

This script is itself one of the mirrored files, so it checks itself.
"""

import os
import sys


def _py_files(d):
    return {f for f in os.listdir(d) if f.endswith(".py") and os.path.isfile(os.path.join(d, f))}


def compare(a, b):
    """Return a list of human-readable drift descriptions (empty == in step)."""
    problems = []
    fa, fb = _py_files(a), _py_files(b)

    for missing in sorted(fa - fb):
        problems.append("%s exists in %s but not in %s" % (missing, a, b))
    for missing in sorted(fb - fa):
        problems.append("%s exists in %s but not in %s" % (missing, b, a))

    for name in sorted(fa & fb):
        with open(os.path.join(a, name), "rb") as fh:
            ba = fh.read()
        with open(os.path.join(b, name), "rb") as fh:
            bb = fh.read()
        if ba != bb:
            problems.append(
                "%s differs (%d bytes vs %d bytes)" % (name, len(ba), len(bb))
            )
    return problems


def main(argv):
    if len(argv) != 3:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    a, b = argv[1], argv[2]
    for d in (a, b):
        if not os.path.isdir(d):
            print("not a directory: %s" % d, file=sys.stderr)
            return 2

    problems = compare(a, b)
    n = len(_py_files(a) & _py_files(b))
    if problems:
        print("the two agent-insights copies have DRIFTED:")
        for p in problems:
            print("  - %s" % p)
        print(
            "\nEvery change to one copy needs the same change to the other "
            "(see CLAUDE.md constraint 4)."
        )
        return 1
    print("the two agent-insights copies are byte-identical (%d .py files)" % n)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
