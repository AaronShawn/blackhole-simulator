# Contributing

Thanks for taking a look! This is a physics-first renderer, so contributions that
improve **numerical fidelity**, **derivations**, or **verification** are especially
welcome.

## Quick start

```powershell
git clone https://github.com/AaronShawn/blackhole-simulator.git
cd blackhole-simulator
pip install -r requirements-dev.txt
python launcher.py            # desktop window (WebView2)
python launcher.py --browser  # or just open web/index.html over http
```

The renderer is plain ES modules - no bundler. `three.js` is vendored under
`web/vendor/`, so nothing needs to be fetched at runtime.

## Useful commands

| Command | What it does |
| --- | --- |
| `python tools/verify.py` | Headless render check: shader/console errors + shadow-radius table |
| `python tools/ui_check.py` | Sidebar collapse/expand and centred-framing regression test |
| `python tools/shot.py --preset "俯视盘面" --out shot.png` | Render a single preset to PNG |
| `python tools/probe_dpi.py` | Inspect WebView2 window/viewport sizing under DPI scaling |
| `powershell -File build_portable.ps1` | Build `dist/BlackHoleSimulator` (portable folder) |

## Ground rules for physics changes

1. **Keep the geometry exact.** Any change to the geodesic integrator must keep the
   shadow-radius check in `README.md#shadow-radius-validation` passing within ~1 %.
2. **Do not mix up angular momenta.** The Doppler term uses the component of the
   photon angular momentum along the disk spin axis (`b_axis = (L·ŷ)/E`), not the total
   `|L|/E`. Using the total magnitude produces a fake bright/dark seam down the middle
   of the frame - see the note in `web/src/shaders.js`.
3. **Separate physics from display.** Colour/tone-mapping choices must not change
   bolometric fluxes. If you add a "cinematic" option, label it as such.
4. Document new approximations in the *Known approximations* section of the README.

## Style

- GLSL lives in `web/src/shaders.js`; keep it commented with the equation it implements.
- JavaScript: 2-space indent, ES modules, no build step.
- Python: 4-space indent, standard library first, no mandatory third-party imports
  outside `requirements-dev.txt`.

## Pull requests

Please include a short summary, the physics/UX motivation, and a screenshot or the
`tools/verify.py` output when the change is visual. Small, focused PRs are easiest to
review.
