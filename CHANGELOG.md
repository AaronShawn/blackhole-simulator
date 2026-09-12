# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and the project adheres to [Semantic Versioning](https://semver.org/).

## [1.0.0] - 2026-09-13

### Added

- **Exact Schwarzschild null-geodesic ray tracer** in GLSL: RK4 integration of
  `d²u/dφ² = -u + 3Mu²` per pixel with adaptive step size and orbital-plane reduction.
- **Static-observer (FIDO) camera model**: pixel angles are converted from the local
  orthonormal frame, including the radial `√(1-rₛ/r₀)` stretch.
- **Novikov–Thorne style thin accretion disk**: `F(r) ∝ r⁻³(1-√(r_in/r))`,
  `T(r) = T_max·(F/F_max)^{1/4}`, ISCO inner edge at 6 M, Gaussian vertical profile.
- **Volumetric radiative transfer** with front-to-back accumulation, so the near side
  correctly occults the lensed far side and grazing rays get the full oblique path.
- **Relativistic frequency shift**: `g = 1/[u^t√(1-rₛ/r₀)(1+Ω·b_axis)]` with
  `b_axis = (L·ŷ)/E`, applied as `T_obs = g·T_emit` and `I_obs = g⁴·I_emit`.
- **Planck blackbody colour** (CIE 1931 locus → XYZ → linear sRGB) with an optional
  display-space temperature compression for readable imagery.
- **Gravitationally lensed procedural sky** (star field + fbm galactic band) sampled
  along the asymptotic photon direction.
- **OrbitControls** camera (rotate/zoom), 6 presets, collapsible sidebar for full-screen
  preview, custom slim scrollbar.
- **HDR pipeline**: RGBA16F buffer → progressive jittered accumulation → 3-level bloom
  pyramid → ACES tone map → sRGB.
- **Built-in physics self-check**: capture-mask render plus a one-click measurement of
  the apparent shadow radius against the analytic value `b_c√(1-rₛ/r₀)/r₀`.
- **Portable Windows build** (PyInstaller + pywebview/WebView2) driven by a local HTTP
  server, with DPI-aware sizing and a browser fallback when WebView2 is missing.

### Notes

- Verified against the analytic shadow radius at 7 observer radii: agreement within
  ~1 % (sub-pixel at 1600×900) — see the README table.
