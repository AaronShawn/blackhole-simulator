# Schwarzschild Black Hole Simulator

A real-time, general-relativistic black hole renderer. **Every pixel is a numerically
integrated null geodesic** of the Schwarzschild metric, and the accretion disk is rendered
with full relativistic radiative transfer. Written with Three.js + a custom GLSL tracer;
ships as a portable Windows folder you can just double-click.

*[中文说明 »](README.md)*

![screenshot](docs/screenshot-main.webp)

## Features

- **Exact geodesics** — per-pixel RK4 integration of `d²u/dφ² = -u + 3Mu²` (u = 1/r) with
  adaptive step size, reduced to the photon's orbital plane. Not a screen-space distortion.
- **Static-observer (FIDO) camera** — pixel directions are defined in the observer's local
  orthonormal frame and converted to Schwarzschild coordinates (radial `√(1-rₛ/r₀)` stretch).
- **Novikov–Thorne style thin disk** — `F(r) ∝ r⁻³(1-√(r_in/r))`, `T = T_max(F/F_max)^{1/4}`,
  ISCO inner edge at 6 M, Gaussian vertical profile, volumetric front-to-back transfer so the
  near side occults the lensed far side.
- **Relativistic frequency shift** — `g = 1/[u^t√(1-rₛ/r₀)(1+Ω·b_axis)]` with the photon angular
  momentum component along the disk spin axis; applied as `T_obs = g·T_emit`, `I_obs = g⁴·I_emit`.
- **Planck blackbody colour** (CIE 1931 locus → XYZ → linear sRGB) with an optional
  display-space temperature compression.
- **Gravitationally lensed procedural sky** — star field and nebula sampled along the asymptotic
  photon direction, so Einstein rings appear by construction.
- **Built-in physics self-check** — renders a capture mask and measures the apparent shadow
  radius against `b_c√(1-rₛ/r₀)/r₀`.
- **HDR pipeline** — RGBA16F → progressive jittered accumulation → 3-level bloom → ACES → sRGB.
- **Portable build** — PyInstaller + pywebview/WebView2, DPI-aware sizing, browser fallback.

## Quick start

**Portable (recommended):** grab `BlackHoleSimulator-win64.zip` from
[Releases](https://github.com/AaronShawn/blackhole-simulator/releases), unzip, run `BlackHoleSimulator.exe`.
It serves the UI on `127.0.0.1` and opens a native WebView2 window with hardware-accelerated
WebGL2 (D3D11/ANGLE). If the WebView2 runtime is missing it falls back to your default browser.
The exe is unsigned, so SmartScreen may ask for *More info → Run anyway* once.

**From source:**

```powershell
git clone https://github.com/AaronShawn/blackhole-simulator.git
cd blackhole-simulator
pip install -r requirements-dev.txt
python launcher.py            # native window
python launcher.py --browser  # or just a browser tab
```

**Rebuild the portable folder:**

```powershell
pip install pywebview pyinstaller pillow
powershell -ExecutionPolicy Bypass -File build_portable.ps1
# -> dist\BlackHoleSimulator\
```

## Controls

| Input | Action |
| --- | --- |
| Left-drag | Orbit around the hole (OrbitControls) |
| Wheel / pinch | Zoom (observer radius r₀: 4.2 M → 800 M) |
| `H` or the `◀` button | Collapse the sidebar for a full-screen preview |
| `Space` | Pause / resume the disk evolution |
| `S` | Save the current frame as PNG |
| `📐 Measure shadow radius` | Render the capture mask and compare with the analytic value |

## Physics summary

Geometric units, `G = c = 1`, `M = 1`: horizon `rₛ = 2M`, photon sphere `3M`, ISCO `6M`,
critical impact parameter `b_c = 3√3 M ≈ 5.196 M`.

```
geodesic      d²u/dφ² = -u + 3M u²                (du/dφ)² = 1/b² - u²(1-2Mu)
camera        du/dφ|₀ = -u₀ (cosψ/sinψ) √(1-2M/r₀)
disk flux     F(r) ∝ r⁻³ (1 - √(r_in/r)),   T = T_max (F/F_max)^{1/4}
transfer      dτ = κρ dl ⟨e^{-y²/2H²}⟩,  I += e^{-τ}(1-e^{-dτ}) S
shift         g = 1 / [ u^t √(1-rₛ/r₀) (1 + Ω b_axis) ],  b_axis = (L·ŷ)/E
```

> Pitfall worth knowing: the Doppler term needs the angular-momentum component **along the
> disk spin axis**, not the total `|L|/E`. Using the magnitude produces a non-physical bright
> / dark seam down the centre of the frame. See the notes in `web/src/shaders.js`.

### Shadow radius validation

Analytic apparent radius for a static observer: `sinψ = b_c√(1-rₛ/r₀)/r₀`.
Measured from the rendered capture mask at 1600×900, 58° FOV (1 px ≈ 0.9 %):

| r₀ [M] | measured [px] | analytic [px] | error |
| ---: | ---: | ---: | ---: |
| 8 | 551.50 | 552.31 | −0.15 % |
| 12 | 349.00 | 349.35 | −0.10 % |
| 20 | 206.50 | 206.46 | +0.02 % |
| 26 | 158.50 | 158.83 | −0.21 % |
| 40 | 103.50 | 103.62 | −0.12 % |
| 80 | 51.50 | 52.17 | −1.29 % |
| 150 | 27.00 | 27.95 | −3.40 % |

Sub-pixel agreement — the integrator reproduces the analytic shadow to the resolution limit.

## Performance

Pure fragment-shader workload (100–900 RK4 steps per pixel). Test machine only had an Intel
UHD iGPU: ~20 FPS at 1500×860 (~1.7 Mpx) with the default 300 steps. Discrete NVIDIA GPUs
usually leave tens of times more headroom, so 800+ steps at 1.5× render scale is practical.
WebView2 runs on D3D11/ANGLE; on hybrid laptops set *Windows Settings → System → Display →
Graphics → BlackHoleSimulator.exe → High performance* (or the NVIDIA Control Panel) to force
the discrete GPU. The overlay in the top-right corner shows the GPU actually in use.

## Known approximations

1. Schwarzschild only — no Kerr frame dragging or spin-dependent shadow deformation.
2. Disk is a geometrically thin slab with a Gaussian vertical profile; no MHD, procedural turbulence.
3. Light-travel-time delays inside the disk are ignored (irrelevant for a steady axisymmetric disk,
   minor for the turbulence pattern).
4. Disk radiation is treated as a blackbody (no synchrotron / Comptonised spectrum).
5. The default "compressed" colour mapping is a **display choice**; fluxes, beaming and shifts stay physical.
6. Canvas pixels are capped (~2.2 Mpx) on very high-DPI displays to avoid GPU device loss.

## Verification & tooling

```powershell
python tools/verify.py     # headless render + shader/console check + shadow radius table
python tools/ui_check.py   # sidebar collapse/expand + centred framing regression
powershell -File build_portable.ps1
```

## License & citation

MIT — see [LICENSE](LICENSE). See [CITATION.cff](CITATION.cff) for academic use.

Physics references: Schwarzschild (1916); Luminet (1979), *Image of a spherical black hole with
thin accretion disk*; Novikov & Thorne (1973); Cunningham & Bardeen (1973); Kim et al. (2002)
for the Planckian-locus fit.
