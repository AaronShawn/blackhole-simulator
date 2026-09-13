#!/usr/bin/env python3
"""Extract the shader excerpts quoted in the paper appendix.

The appendix must never drift away from the code it quotes, so the excerpts are
not copy-pasted into the .tex source: they are sliced out of web/src/shaders.js
by the line ranges recorded below and written to paper/code/.  The ranges are
declared with a *content* guard (a substring that must appear on the first and
last line of the slice) so that a silent re-numbering of shaders.js makes this
script fail loudly instead of quietly quoting the wrong block.

Run from the repository root:

    python paper/tools/extract_snippets.py

Output: paper/code/*.glsl  (pure ASCII, LF, no trailing whitespace)
"""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC = ROOT / "web" / "src" / "shaders.js"
OUT = pathlib.Path(__file__).resolve().parents[1] / "code"

# name -> (first_line, last_line, guard_first, guard_last)   1-based, inclusive
SNIPPETS = {
    "geodesic_ode": (
        167, 178,
        "Schwarzschild ODE",
        "}",
    ),
    "subpixel_jitter": (
        321, 333,
        "integrator ---",
        "ndc += (jit - 0.5) * 2.0 * uJitter / uResolution;",
    ),
    "disk_source": (
        256, 318,
        "4x10^6-cell quadrature",
        "return planckRGB(Tdisp) * Irel;",
    ),
    "radiative_transfer": (
        389, 461,
        "for (int i = 0; i < 4096; i++)",
        "    }",
    ),
}


def main() -> int:
    if not SRC.exists():
        print(f"missing source: {SRC}", file=sys.stderr)
        return 1

    lines = SRC.read_text(encoding="utf-8").splitlines()
    OUT.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []

    for name, (a, b, g_first, g_last) in SNIPPETS.items():
        chunk = lines[a - 1 : b]
        head, tail = chunk[0], chunk[-1]
        if g_first not in head:
            failures.append(f"{name}: line {a} is {head!r}, expected {g_first!r}")
        if g_last not in tail:
            failures.append(f"{name}: line {b} is {tail!r}, expected {g_last!r}")
        if any(ord(ch) > 127 for ch in "".join(chunk)):
            failures.append(f"{name}: excerpt contains non-ASCII characters")
        text = "\n".join(line.rstrip() for line in chunk).strip("\n")
        (OUT / f"{name}.glsl").write_text(text + "\n", encoding="ascii", newline="\n")
        print(f"{name}.glsl  lines {a}-{b} ({b - a + 1} lines)")

    if failures:
        print("\nGUARD FAILURES:", file=sys.stderr)
        for f in failures:
            print("  " + f, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
