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
