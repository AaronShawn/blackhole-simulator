#!/usr/bin/env python3
"""Guard the figure numbering of the paper.

LaTeX numbers floats in order of appearance, so the number that belongs under a
figure is a property of the *document* (the order of ``\\input`` files and of
the figure environments inside them), not of the script that drew it.  The five
montages built by ``make_render_figures.py`` carry their number burnt into the
image, which is where a number can silently go stale; they read it from the
single table ``FIG_NUM`` in that file.

This checker re-derives the order from ``paper.tex`` + ``parts/*.tex``, fails
when ``FIG_NUM`` disagrees, refuses a literal ``图 <n>`` written into a title
strip, and flags caption text that hard-codes a figure number instead of using
``\\ref{fig:..}``.  Run it after touching any figure environment:

    python paper/tools/check_fignum.py             # verify (exit 1 on drift)
    python paper/tools/check_fignum.py --table     # print the full mapping
"""

from __future__ import annotations

import argparse
import ast
import glob
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(HERE)

INPUT_RE = re.compile(r"\\input\{(parts/[^}]+)\}")
FIGLABEL_RE = re.compile(r"\\label\{(fig:[0-9]+)\}")
# a title strip that starts with a literal number:  "图 22  渲染代价分解…"
LITERAL_TITLE_RE = re.compile(r"[\"']\s*图\s*[0-9]")
TITLE_CALL_RE = re.compile(r"fig_title\(\s*[\"'](f[0-9]+)[\"']")
# prose/caption text such as "图 22(d)" or "插图 1--13": must go through \ref
HARDCODED_RE = re.compile(r"(?:图|插图)\s*~?\s*\d")
# an explicit range "图~\ref{fig:01}--\ref{fig:10}" claims a contiguous run of
# printed numbers; labels are not ordered, so the claim must be checked
RANGE_RE = re.compile(r"\\ref\{(fig:[0-9]+)\}\s*--\s*\\ref\{(fig:[0-9]+)\}")


def document_order() -> tuple[list[str], list[str]]:
    """Return (file parts in \\input order, figure labels in printed order)."""
    with open(os.path.join(PAPER, "paper.tex"), encoding="utf-8") as fh:
        main = fh.read()
    parts = INPUT_RE.findall(main)
    if not parts:
        raise SystemExit("paper.tex: no \\input{parts/..} found")
    labels: list[str] = []
    for rel in parts:
        path = os.path.join(PAPER, rel + ".tex")
        if not os.path.exists(path):
            raise SystemExit("missing \\input target: " + path)
        with open(path, encoding="utf-8") as fh:
            labels += FIGLABEL_RE.findall(fh.read())
    return parts, labels


def declared_numbers() -> dict[str, int]:
    """Read ``FIG_NUM`` out of make_render_figures.py without importing it."""
    src = os.path.join(HERE, "make_render_figures.py")
    with open(src, encoding="utf-8") as fh:
        tree = ast.parse(fh.read(), src)
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "FIG_NUM"
                for t in node.targets):
            return ast.literal_eval(node.value)
    raise SystemExit("FIG_NUM not found in make_render_figures.py")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", action="store_true",
                    help="print the derived mapping and exit")
    args = ap.parse_args()

    parts, labels = document_order()
    printed = {"f" + lab.split(":")[1]: n
               for n, lab in enumerate(labels, start=1)}
    declared = declared_numbers()

    if args.table:
        print("parts in order:", ", ".join(parts))
        for tag in sorted(printed, key=lambda t: printed[t]):
            print("  图 %-2d  <-  %s" % (printed[tag], tag))
        print("stem order by printed number:",
              " ".join(sorted(printed, key=lambda t: printed[t])))
        return 0

    problems: list[str] = []

    for tag, num in sorted(printed.items(), key=lambda kv: kv[1]):
        if tag not in declared:
            problems.append("FIG_NUM is missing %s (printed as 图 %d)"
                            % (tag, num))
        elif declared[tag] != num:
            problems.append("FIG_NUM[%s] = %d but LaTeX prints 图 %d"
                            % (tag, declared[tag], num))
    for tag in sorted(set(declared) - set(printed)):
        problems.append("FIG_NUM[%s] = %d has no float in the paper"
                        % (tag, declared[tag]))

    with open(os.path.join(HERE, "make_render_figures.py"),
              encoding="utf-8") as fh:
        lines = fh.read().splitlines()
    for i, line in enumerate(lines, start=1):
        if LITERAL_TITLE_RE.search(line):
            problems.append("make_render_figures.py:%d writes a literal figure"
                            " number into a title strip; use fig_title()"
                            % i)
    for i, line in enumerate(lines, start=1):
        for tag in TITLE_CALL_RE.findall(line):
            if tag not in declared:
                problems.append("make_render_figures.py:%d uses fig_title(%s)"
                                " which is not in FIG_NUM" % (i, tag))

    for path in sorted(glob.glob(os.path.join(PAPER, "parts", "*.tex"))
                       + [os.path.join(HERE, "make_tables.py")]):
        with open(path, encoding="utf-8") as fh:
            for i, line in enumerate(fh, start=1):
                if line.lstrip().startswith("%"):
                    continue
                if HARDCODED_RE.search(line):
                    problems.append("%s:%d hard-codes a figure number; use"
                                    " \\ref{fig:..}"
                                    % (os.path.relpath(path, PAPER), i))
                for a, b in RANGE_RE.findall(line):
                    ta, tb = "f" + a.split(":")[1], "f" + b.split(":")[1]
                    if ta not in printed or tb not in printed:
                        problems.append(
                            "%s:%d range %s--%s names a label with no float"
                            % (os.path.relpath(path, PAPER), i, a, b))
                    elif printed[tb] <= printed[ta]:
                        problems.append(
                            "%s:%d range %s--%s is not ascending in printed"
                            " order (图 %d -- 图 %d)"
                            % (os.path.relpath(path, PAPER), i, ta, tb,
                               printed[ta], printed[tb]))

    if problems:
        print("figure numbering: %d problem(s)" % len(problems))
        for p in problems:
            print("  -", p)
        return 1
    print("figure numbering OK: %d floats, FIG_NUM agrees with paper.tex"
          % len(labels))
    return 0


if __name__ == "__main__":
    sys.exit(main())
