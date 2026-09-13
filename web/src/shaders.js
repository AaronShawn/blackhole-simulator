/**
 * shaders.js — GLSL sources for the Schwarzschild black-hole ray tracer.
 *
 * Physical model (geometric units G = c = 1, mass M = 1):
 *   Schwarzschild radius  rs = 2M          -> horizon at r = 2
 *   Photon sphere         r  = 3M
 *   ISCO                  r  = 6M
 *   Critical impact par.  bc = 3*sqrt(3) M ~= 5.196 M
 *
 * Null geodesics are integrated in the orbital plane of each ray using the
 * exact Schwarzschild orbit equation for u = 1/r:
 *
 *      d^2u/dphi^2 = -u + 3 M u^2          (M = 1)
 *
 * which is equivalent to the standard first integral
 *
 *      (du/dphi)^2 = 1/b^2 - u^2 (1 - 2M u),   b = L/E
 *
 * Every pixel therefore is a numerically integrated null geodesic (RK4,
 * adaptive step), not a fake lensing warp.
 */

export const FULLSCREEN_VERT = /* glsl */ `
varying vec2 vUv;
void main() {
  vUv = uv;
  gl_Position = vec4(position.xy, 0.0, 1.0);
}
`;

export const GEODESIC_FRAG = /* glsl */ `
precision highp float;
precision highp int;

varying vec2 vUv;

uniform vec2  uResolution;
uniform float uSampleIndex;   // progressive-accumulation sample counter
uniform float uJitter;

uniform vec3  uCamPos;      // world position of the static observer (units of M)
uniform mat3  uCamBasis;    // columns: right, up, forward
uniform float uTanHalfFov;
uniform float uAspect;
uniform float uTime;        // coordinate time t/M

uniform int   uMaxSteps;
uniform float uTol;         // relative change of u per RK4 step
uniform float uStepScale;
uniform float uEscapeR;

uniform float uDiskInner;   // r_in / M   (ISCO = 6)
uniform float uDiskOuter;   // r_out / M
uniform float uDiskThick;   // H/r
uniform float uDiskTemp;    // peak emitted temperature [K]
uniform float uDiskBright;
uniform float uDiskOpacity;
uniform float uTurb;
uniform float uSpin;        // +1 / -1 : sense of disk rotation
uniform float uKepler;      // 1 = orbiting disk, 0 = static emitter
uniform float uDoppler;     // 1 = relativistic shifts on
uniform float uFluxModel;   // 0 = Shakura-Sunyaev (Newtonian), 1 = Page-Thorne
uniform float uShowDisk;

uniform float uShowStars;
uniform float uStarBright;

uniform float uColorMode;   // 0 = displayed colour compression, 1 = true physical
uniform float uDispTMin;
uniform float uDispTMax;
uniform float uDebugMask;   // 1 = render the capture mask (white = escapes)

const float PI = 3.141592653589793;

// ---------------------------------------------------------------- globals ---
vec3  G_E1;         // in-plane basis, e1 = r_hat at phi = 0
vec3  G_E2;
float G_B;          // conserved impact parameter b = |L|/E of this ray
float G_FOBS;       // sqrt(1 - rs/r0): gravitational factor of the camera

// ------------------------------------------------------------- hash / noise ---
float hash21(vec2 p) {
  p = fract(p * vec2(123.34, 456.21));
  p += dot(p, p + 45.32);
  return fract(p.x * p.y);
}

vec3 hash33(vec3 p) {
  p = vec3(dot(p, vec3(127.1, 311.7, 74.7)),
           dot(p, vec3(269.5, 183.3, 246.1)),
           dot(p, vec3(113.5, 271.9, 124.6)));
  return fract(sin(p) * 43758.5453123);
}

float vnoise(vec3 x) {
  vec3 i = floor(x);
  vec3 f = fract(x);
  f = f * f * (3.0 - 2.0 * f);
  float n000 = hash33(i + vec3(0.0, 0.0, 0.0)).x;
  float n100 = hash33(i + vec3(1.0, 0.0, 0.0)).x;
  float n010 = hash33(i + vec3(0.0, 1.0, 0.0)).x;
  float n110 = hash33(i + vec3(1.0, 1.0, 0.0)).x;
  float n001 = hash33(i + vec3(0.0, 0.0, 1.0)).x;
  float n101 = hash33(i + vec3(1.0, 0.0, 1.0)).x;
  float n011 = hash33(i + vec3(0.0, 1.0, 1.0)).x;
  float n111 = hash33(i + vec3(1.0, 1.0, 1.0)).x;
  return mix(mix(mix(n000, n100, f.x), mix(n010, n110, f.x), f.y),
             mix(mix(n001, n101, f.x), mix(n011, n111, f.x), f.y), f.z);
}

float fbm(vec3 p) {
  float a = 0.5, s = 0.0;
  for (int i = 0; i < 4; i++) {
    s += a * vnoise(p);
    p *= 2.03;
    a *= 0.5;
  }
  return s;
}

// ------------------------------------------------------------------- erf ---
// Abramowitz & Stegun 7.1.26, |err| < 1.5e-7
float erfApprox(float x) {
  float s = sign(x);
  float ax = abs(x);
  float t = 1.0 / (1.0 + 0.3275911 * ax);
  float y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
                    - 0.284496736) * t + 0.254829592) * t * exp(-ax * ax);
  return s * y;
}

// ------------------------------------------------------------- blackbody ---
// Chromaticity of a Planck spectrum via the CIE 1931 (x,y) locus
// (Kim et al. cubic approximation), converted to linear sRGB and normalised
// to unit luminance (Y = 1).  Radiometric intensity is handled separately.
vec3 planckRGB(float T) {
  T = clamp(T, 1000.0, 40000.0);
  float x;
  if (T < 4000.0) {
    x = -0.2661239e9 / (T * T * T) - 0.2343589e6 / (T * T) + 0.8776956e3 / T + 0.179910;
  } else {
    x = -3.0258469e9 / (T * T * T) + 2.1070379e6 / (T * T) + 0.2226347e3 / T + 0.240390;
  }
  float y;
  if (T < 2222.0) {
    y = -1.1063814 * x * x * x - 1.34811020 * x * x + 2.18555832 * x - 0.20219683;
  } else if (T < 4000.0) {
    y = -0.9549476 * x * x * x - 1.37418593 * x * x + 2.09137015 * x - 0.16748867;
  } else {
    y = 3.0817580 * x * x * x - 5.87338670 * x * x + 3.75112997 * x - 0.37001483;
  }
  y = max(y, 1e-4);
  float X = x / y;
  float Y = 1.0;
  float Z = (1.0 - x - y) / y;
  // CIE XYZ -> linear sRGB (Rec.709 primaries, D65)
  vec3 rgb;
  rgb.r =  3.2404542 * X - 1.5371385 * Y - 0.4985314 * Z;
  rgb.g = -0.9692660 * X + 1.8760108 * Y + 0.0415560 * Z;
  rgb.b =  0.0556434 * X - 0.2040259 * Y + 1.0572252 * Z;
  // gamut mapping: add white until the spectrum fits inside sRGB
  float mn = min(min(rgb.r, rgb.g), rgb.b);
  rgb -= min(mn, 0.0);
  return rgb;
}

// ------------------------------------------------------- Schwarzschild ODE ---
vec2 geoDeriv(vec2 s) {              // s = (u, du/dphi)
  return vec2(s.y, -s.x + 3.0 * s.x * s.x);
}

vec2 rk4(vec2 s, float h) {
  vec2 k1 = geoDeriv(s);
  vec2 k2 = geoDeriv(s + 0.5 * h * k1);
  vec2 k3 = geoDeriv(s + 0.5 * h * k2);
  vec2 k4 = geoDeriv(s + h * k3);
  return s + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
}

// ------------------------------------------------------------- background ---
// Procedural sky: resolved stars (cube-face cell hashing) over a faint
// fbm nebula band.  Sampled with the asymptotic photon direction, so the
// Einstein-ring distortion of the background comes from the geodesics.
vec3 starField(vec3 dir) {
  vec3 col = vec3(0.0);

  vec3 a = abs(dir);
  vec2 uv;
  float face;
  if (a.x >= a.y && a.x >= a.z) {
    uv = dir.yz / max(a.x, 1e-6); face = dir.x > 0.0 ? 0.0 : 1.0;
  } else if (a.y >= a.z) {
    uv = dir.xz / max(a.y, 1e-6); face = dir.y > 0.0 ? 2.0 : 3.0;
  } else {
    uv = dir.xy / max(a.z, 1e-6); face = dir.z > 0.0 ? 4.0 : 5.0;
  }

  for (int L = 0; L < 3; L++) {
    float sc = 34.0 * pow(2.35, float(L));
    vec2 g = uv * sc;
    vec2 cell = floor(g);
    vec2 f = fract(g);
    float lw = 1.0 / (1.0 + 2.2 * float(L));
    for (int oy = -1; oy <= 1; oy++) {
      for (int ox = -1; ox <= 1; ox++) {
        vec2 c = cell + vec2(float(ox), float(oy));
        vec3 h = hash33(vec3(c, face * 19.7 + float(L) * 57.3));
        vec2 sp = vec2(h.x, h.y) * 0.8 + 0.1;
        float d = length(f - vec2(float(ox), float(oy)) - sp);
        float mag = pow(h.z, 9.0);
        float size = 0.008 + 0.030 * h.z * h.z;
        float psf = exp(-d * d / (2.0 * size * size));
        float T = mix(2800.0, 24000.0, pow(h.z, 0.6));
        col += planckRGB(T) * (mag * psf * 18.0 * lw);
      }
    }
  }

  // faint galactic band + nebulosity
  vec3 bn = normalize(vec3(0.31, 0.87, -0.38));
  float band = exp(-pow(dot(dir, bn) / 0.30, 2.0));
  float neb = fbm(dir * 3.2) * 0.65 + 0.5 * fbm(dir * 9.0);
  col += vec3(0.30, 0.42, 0.72) * band * neb * 0.020;
  col += vec3(0.55, 0.40, 0.28) * band * band * 0.010;
  return col;
}

// ---------------------------------------------------------- accretion disk ---
float diskWindow(float r) {
  float w = smoothstep(uDiskInner * 1.0, uDiskInner * 1.15, r);
  w *= 1.0 - smoothstep(uDiskOuter * 0.88, uDiskOuter, r);
  return w;
}

// ---------------------------------------------------------- disk flux laws ---
// Both profiles are normalised to unit peak so that the disk temperature and
// brightness scales are model independent; the *shape* is what differs.
//
// (a) Shakura-Sunyaev 1973 (Newtonian, Mdot = const, zero torque at r_in):
//       F(r) = 3M/(8 pi) (1 - sqrt(r_in/r)) / r^3 ,
//     which peaks exactly at r = (49/36) r_in.
float diskFluxSS(float r) {
  float rp = 1.3611111 * uDiskInner;          // (49/36) r_in
  float F = 7.0 * (1.0 - sqrt(uDiskInner / r)) * pow(rp / r, 3.0);
  return max(F, 0.0);
}

// (b) Page-Thorne 1974 / Novikov-Thorne relativistic thin disk (Schwarzschild):
//       F(r) = Mdot/(4 pi r) (-dOmega/dr) (E - Omega L)^-2 B(r)
//     with the circular-geodesic integrals (M = 1, sqrt(1-3/r) = E - Omega L)
//       E(r) = (1-2/r) / sqrt(1-3/r),   L(r) = sqrt(r) / sqrt(1-3/r),
//       Omega(r) = r^(-3/2),            dL/dr = (r-3)/ (2 sqrt(r) (1-3/r)^(3/2)),
//       B(r) = int_{r_in}^{r} (E - Omega L) dL/dr' dr'
//            = (sqrt(r)-sqrt(r_in)) - (sqrt(3)/2) * ln( [(sqrt(r)-sqrt3)(sqrt(r_in)+sqrt3)]
//                                                     / [(sqrt(r)+sqrt3)(sqrt(r_in)-sqrt3)] ).
//     The closed form above was audited against a 4x10^6-cell quadrature to
//     2e-14 relative accuracy (paper/tools/audit_nt.py).
float ptRawFlux(float r, float rin) {
  const float SQ3 = 1.7320508075688772;
  float rc = max(r, rin);                     // profile is defined for r >= r_in
  float s = sqrt(rc);
  float si = sqrt(rin);
  float num = (s - SQ3) * (si + SQ3);
  float den = (s + SQ3) * (si - SQ3);
  float B = (s - si) - 0.5 * SQ3 * log(max(num / max(den, 1e-20), 1e-20));
  float u_red = max(1.0 - 3.0 / rc, 1e-4);    // (E - Omega L)^2 = 1 - 3M/r
  // -dOmega/dr = 1.5 r^(-5/2)
  return 1.5 * pow(rc, -2.5) * B / (4.0 * PI * rc * u_red);
}

float diskFluxPT(float r) {
  float rin = max(uDiskInner, 3.001);
  float pk = ptRawFlux(1.5918219 * rin, rin); // peak at 1.5918 r_in (r_in = 6M)
  return max(ptRawFlux(r, rin) / max(pk, 1e-30), 0.0);
}

// temperature follows  T ~ F^(1/4),  emission j ~ F (Stefan-Boltzmann).
float diskFlux(float r) {
  return (uFluxModel > 0.5) ? diskFluxPT(r) : diskFluxSS(r);
}

vec3 diskSource(float r, vec3 pm, vec3 tdir, float Fh) {
  float Te = uDiskTemp * pow(max(Fh, 1e-8), 0.25);

  // --- relativistic frequency shift ---------------------------------------
  //   g = nu_obs / nu_emit = 1 / ( u^t sqrt(1-rs/r0) (1 + Omega b_axis) )
  // with u^t = 1/sqrt(1-3M/r) for circular-geodesic (Keplerian) emitters,
  // u^t = 1/sqrt(1-2M/r) for a static emitter, and
  //   b_axis = (L . y_hat) / E   -> the photon angular momentum component along
  //                                the spin axis of the disk (NOT |L|/E).
  // The photon was traced *backwards* (camera -> sky), so p_real = -tdir.
  vec3 Lvec = -cross(pm, tdir);
  float Lmag = length(Lvec);
  float bAxis = (Lmag > 1e-9) ? G_B * (Lvec.y / Lmag) : 0.0;

  float Omega = uKepler * uSpin / pow(r, 1.5);  // dpsi/dt of the emitter
  float ut = (uKepler > 0.5)
      ? inversesqrt(max(1.0 - 3.0 / r, 1e-3))
      : inversesqrt(max(1.0 - 2.0 / r, 1e-3));
  float g = 1.0 / (ut * G_FOBS * (1.0 + Omega * bAxis));
  g = clamp(g, 0.05, 8.0);
  if (uDoppler < 0.5) g = 1.0;

  // --- spectrum ------------------------------------------------------------
  float Tdisp;
  if (uColorMode < 0.5) {
    // display mode: keep the *shape* of the temperature profile readable by
    // mapping it logarithmically onto a visible range, then apply the exact
    // relativistic shift g on top of it.
    float s = pow(clamp(Fh, 0.0, 1.0), 0.80);
    Tdisp = exp(mix(log(uDispTMin), log(uDispTMax), s)) * g;
  } else {
    Tdisp = Te * g;                             // true physical colour temperature
  }

  // bolometric beaming: I_obs = g^4 I_emit ; emissivity ~ sigma T^4 ~ F
  float Irel = uDiskBright * Fh * pow(g, 4.0);
  return planckRGB(Tdisp) * Irel;
}

// --------------------------------------------------------------- integrator ---
void main() {
  vec2 ndc = vUv * 2.0 - 1.0;
  // Sub-pixel sample position: a Cranley--Patterson rotation of the R2
  // low-discrepancy sequence.  The rotation is a per-pixel constant, so a
  // capture with a fixed sample index is bit-reproducible, while successive
  // indices are stratified inside the pixel instead of clustering like
  // independent uniform noise would.
  vec2 rot = vec2(hash21(gl_FragCoord.xy),
                  hash21(gl_FragCoord.yx * 1.371 + 17.3));
  vec2 jit = fract(rot + uSampleIndex * vec2(0.7548776662466927,
                                              0.5698402909980532));
  ndc += (jit - 0.5) * 2.0 * uJitter / uResolution;

  // local (static observer) direction of the pixel
  vec3 nLocal = normalize(uCamBasis * vec3(ndc.x * uTanHalfFov * uAspect,
                                           ndc.y * uTanHalfFov,
                                           1.0));

  vec3 ro = uCamPos;
  float r0 = length(ro);
  float u0 = 1.0 / r0;
  float fObs = sqrt(max(1.0 - 2.0 * u0, 1e-6));   // sqrt(1 - rs/r0)
  G_FOBS = fObs;
  G_E1 = ro / r0;

  vec3 color = vec3(0.0);
  float trans = 1.0;

  vec3 cr = cross(ro, nLocal);
  float cr2 = dot(cr, cr);
  float a = dot(nLocal, G_E1);

  // The orbital plane is spanned by ro and the ray direction.  For rays that
  // are (anti-)parallel to the radial direction that plane is degenerate; there
  // the trajectory is a straight radial line through the hole.
  float btan = sqrt(max(1.0 - a * a, 0.0));      // sin(psi), well conditioned
  if (btan < 1e-5) {
    if (a < 0.0) { gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0); return; }   // falls in
    if (uDebugMask > 0.5) { gl_FragColor = vec4(vec3(10.0), 1.0); return; }
    vec3 bg = (uShowStars > 0.5) ? starField(nLocal) * uStarBright : vec3(0.0);
    gl_FragColor = vec4(bg, 1.0);
    return;
  }

  vec3 cN;
  if (cr2 > 1e-12) {
    cN = cr * inversesqrt(cr2);
  } else {
    // any plane containing the (nearly) radial ray is equivalent
    vec3 axis = (abs(G_E1.y) < 0.9) ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    cN = normalize(cross(G_E1, axis));
  }
  G_E2 = normalize(cross(cN, G_E1));
  // orient the in-plane axis so that increasing phi marches *forward* along the
  // ray (the cross product alone is ambiguous at the degenerate column).
  if (dot(nLocal, G_E2) < 0.0) G_E2 = -G_E2;

  // initial slope: radial component of the local direction is stretched by
  // sqrt(1-rs/r) when converted to Schwarzschild coordinate components.
  float du = -u0 * (a / btan) * fObs;
  G_B = 1.0 / sqrt(max(du * du + u0 * u0 * (1.0 - 2.0 * u0), 1e-12));

  float uMin = 1.0 / max(uEscapeR, 1.6 * r0);
  float u = u0;
  float phi = 0.0;
  bool captured = false;

  for (int i = 0; i < 4096; i++) {
    if (i >= uMaxSteps) break;
    if (u > 0.5) { captured = true; break; }              // r < rs
    if (u < uMin && du < 0.0) break;                      // escaped to infinity

    // adaptive step: bound the relative change of u, but allow long steps far
    // away where the ODE is nearly the flat-space harmonic oscillator.
    float hmax = 0.35 - 0.30 * smoothstep(0.0, 0.30, u);
    float h = uTol * u / max(abs(du), 1e-7);
    h = clamp(h, 0.002, hmax) * uStepScale;

    vec2 s = rk4(vec2(u, du), h);
    float phi2 = phi + h;

    float c1 = cos(phi),  s1 = sin(phi);
    float c2 = cos(phi2), s2 = sin(phi2);
    vec3 p1 = (c1 * G_E1 + s1 * G_E2) / max(u, 1e-7);
    vec3 p2 = (c2 * G_E1 + s2 * G_E2) / max(s.x, 1e-7);

    // ---- volumetric accretion-disk emission (front-to-back transfer) -----
    if (uShowDisk > 0.5 && trans > 0.002) {
      float r1 = 1.0 / max(u, 1e-7);
      float r2 = 1.0 / max(s.x, 1e-7);
      float Hm = uDiskThick * max(r1, r2);
      bool nearDisk = (min(abs(p1.y), abs(p2.y)) < 7.0 * Hm) &&
                      (min(r1, r2) < uDiskOuter * 1.3) &&
                      (max(r1, r2) > uDiskInner * 0.85);
      int K = nearDisk ? 5 : 1;
      for (int k = 0; k < 5; k++) {
        if (k >= K) break;
        float t0 = float(k) / float(K);
        float t1 = float(k + 1) / float(K);
        vec3 pa = mix(p1, p2, t0);
        vec3 pb = mix(p1, p2, t1);
        vec3 pm = 0.5 * (pa + pb);
        float rm = length(pm);
        if (rm < uDiskInner * 0.9 || rm > uDiskOuter * 1.05) continue;
        float H = uDiskThick * rm;
        float ya = pa.y, yb = pb.y;
        float dy = yb - ya;
        float dl = distance(pa, pb);
        // vertical average of the Gaussian profile over this sub-segment
        float avg;
        if (abs(dy) < 1e-5 * max(1.0, abs(ya))) {
          avg = exp(-0.5 * (ya * ya) / (H * H));
        } else {
          float kk = 1.0 / (H * sqrt(2.0));
          avg = abs(sqrt(PI * 0.5) * H * (erfApprox(yb * kk) - erfApprox(ya * kk)) / dy);
        }
        float rho = pow(uDiskInner / rm, 2.0) * diskWindow(rm);
        float dtau = min(uDiskOpacity * rho * dl * avg, 6.0);
        if (dtau > 1e-5) {
          float Fh = diskFlux(rm);
          if (Fh > 1e-6) {
            vec3 tdir = normalize(pb - pa + vec3(0.0, 0.0, 1e-9));
            vec3 S = diskSource(rm, pm, tdir, Fh);
            // turbulent advection pattern in the co-rotating frame
            if (uTurb > 0.001) {
              float psi = atan(pm.z, pm.x);
              float Om = uKepler * uSpin / pow(rm, 1.5);
              float psiRel = psi - Om * uTime;
              vec3 q = vec3(rm * cos(psiRel), rm * sin(psiRel), rm * 0.30) / 2.2;
              float nz = 0.62 * vnoise(q) + 0.38 * vnoise(q * 2.3 + 5.3);
              // dissipation clumps: log-normal-ish emissivity contrast
              S *= exp(uTurb * 3.1 * (nz - 0.5)) * mix(1.0, 0.55 + 0.9 * nz, uTurb);
            }
            float ab = 1.0 - exp(-dtau);
            color += trans * ab * S;
            trans *= exp(-dtau);
          }
        }
      }
    }

    phi = phi2;
    u = s.x;
    du = s.y;
    if (u < 1e-6) { u = 1e-6; break; }          // ran off to infinity
  }

  // ---- background star field through the remaining transmittance ----------
  if (!captured && uShowStars > 0.5 && trans > 0.002) {
    float c = cos(phi), sn = sin(phi);
    vec3 er = c * G_E1 + sn * G_E2;
    vec3 ep = -sn * G_E1 + c * G_E2;
    float rr = 1.0 / max(u, 1e-7);
    float drdphi = -du / max(u * u, 1e-12);
    vec3 tdir = normalize(drdphi * er + rr * ep);
    color += trans * starField(tdir) * uStarBright;
  }

  // white = photon escapes to infinity, black = photon falls through the
  // horizon.  Used to measure the apparent shadow radius and compare it with
  // the analytic value sin(psi) = b_c sqrt(1-rs/r0)/r0, b_c = 3 sqrt(3) M.
  if (uDebugMask > 0.5) {
    gl_FragColor = vec4(vec3(captured ? 0.0 : 10.0), 1.0);
    return;
  }

  gl_FragColor = vec4(color, 1.0);
}
`;

// ------------------------------------------------------------- post passes ---

export const BRIGHT_FRAG = /* glsl */ `
precision highp float;
varying vec2 vUv;
uniform sampler2D tSrc;
uniform float uThreshold;
uniform float uKnee;
void main() {
  vec3 c = texture2D(tSrc, vUv).rgb;
  float l = dot(c, vec3(0.2126, 0.7152, 0.0722));
  float w = clamp((l - uThreshold + uKnee) / max(2.0 * uKnee, 1e-4), 0.0, 1.0);
  w = w * w * (l > 0.0 ? 1.0 : 0.0);
  gl_FragColor = vec4(c * w, 1.0);
}
`;

export const BLUR_FRAG = /* glsl */ `
precision highp float;
varying vec2 vUv;
uniform sampler2D tSrc;
uniform vec2 uDir;      // texel-sized blur direction
void main() {
  float w[5];
  w[0] = 0.2270270270; w[1] = 0.1945945946; w[2] = 0.1216216216;
  w[3] = 0.0540540541; w[4] = 0.0162162162;
  vec3 sum = texture2D(tSrc, vUv).rgb * w[0];
  for (int i = 1; i < 5; i++) {
    vec2 o = uDir * float(i) * 1.35;
    sum += texture2D(tSrc, vUv + o).rgb * w[i];
    sum += texture2D(tSrc, vUv - o).rgb * w[i];
  }
  gl_FragColor = vec4(sum, 1.0);
}
`;

export const BLEND_FRAG = /* glsl */ `
precision highp float;
varying vec2 vUv;
uniform sampler2D tPrev;
uniform sampler2D tCur;
uniform float uMix;     // 1.0 -> take current frame only
void main() {
  vec3 a = texture2D(tPrev, vUv).rgb;
  vec3 b = texture2D(tCur, vUv).rgb;
  gl_FragColor = vec4(mix(a, b, uMix), 1.0);
}
`;

export const COMPOSITE_FRAG = /* glsl */ `
precision highp float;
varying vec2 vUv;
uniform sampler2D tBase;
uniform sampler2D tB0;
uniform sampler2D tB1;
uniform sampler2D tB2;
uniform float uBloom;
uniform float uExposure;
uniform float uVignette;
uniform float uGrain;
uniform float uFrame;
uniform vec2  uResolution;

vec3 aces(vec3 x) {
  return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0);
}

void main() {
  vec3 base = texture2D(tBase, vUv).rgb;
  vec3 bloom = texture2D(tB0, vUv).rgb * 0.5 +
               texture2D(tB1, vUv).rgb * 0.32 +
               texture2D(tB2, vUv).rgb * 0.18;
  vec3 c = base + bloom * uBloom;
  c *= uExposure;
  c = aces(c);
  // linear -> sRGB
  c = mix(c * 12.92, 1.055 * pow(max(c, 1e-5), vec3(1.0 / 2.4)) - 0.055,
          step(0.0031308, c));

  vec2 q = vUv - 0.5;
  c *= 1.0 - uVignette * dot(q, q) * 1.35;

  float g = fract(sin(dot(vUv * uResolution + uFrame, vec2(12.9898, 78.233))) * 43758.5453);
  c += (g - 0.5) * uGrain;
  gl_FragColor = vec4(max(c, 0.0), 1.0);
}
`;
