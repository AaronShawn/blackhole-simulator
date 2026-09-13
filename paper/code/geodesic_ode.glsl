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
