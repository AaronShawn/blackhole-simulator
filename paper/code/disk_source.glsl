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
