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
