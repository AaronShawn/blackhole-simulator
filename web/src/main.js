/**
 * main.js — Schwarzschild black hole renderer.
 *
 * Rendering pipeline:
 *   1. geodesic pass   : per-pixel null geodesic integration (HDR, half float)
 *   2. accumulation    : progressive jittered averaging while the view is static
 *   3. bloom           : bright pass + 3-level separable gaussian pyramid
 *   4. composite       : ACES tone map, sRGB encode, vignette, dither
 *
 * Camera: a static observer (FIDO) at radius r0; angles are measured in the
 * observer's local orthonormal frame.  OrbitControls drives it directly.
 */

import * as THREE from '../vendor/three.module.js';
import { OrbitControls } from '../vendor/OrbitControls.js';
import * as SH from './shaders.js';

// ------------------------------------------------------------------ params ---
const DEFAULTS = {
  // camera
  dist: 26.0,
  elevation: 13.0,      // degrees above the disk plane
  fov: 58.0,
  autoRotate: false,
  // accretion disk
  showDisk: 1,
  diskInner: 6.0,       // ISCO [M]
  diskOuter: 26.0,
  diskThick: 0.075,     // H/r
  diskTemp: 9000.0,     // peak emitted temperature [K]
  diskBright: 1.0,
  diskOpacity: 2.6,
  turb: 0.55,
  spin: 1.0,
  kepler: 1.0,
  doppler: 1.0,
  fluxModel: 1.0,       // 0 = Shakura-Sunyaev (Newtonian), 1 = Page-Thorne (GR)
  colorMode: 0.0,       // 0 = readable colour compression, 1 = true physical
  dispTMin: 2000.0,
  dispTMax: 14000.0,
  // sky
  showStars: 1.0,
  starBright: 1.0,
  // physics / integration
  maxSteps: 300,
  tol: 0.03,
  stepScale: 1.0,
  escapeR: 140.0,
  timeRate: 9.0,        // coordinate time t/M per second
  paused: false,
  debugMask: 0,
  autoQuality: true,
  // image
  renderScale: 1.0,
  exposure: 1.0,
  bloom: 0.35,
  bloomThreshold: 1.6,
  vignette: 0.35,
  grain: 0.010,
  accumulate: true,
  shadowCircle: false,
};
const P = Object.assign({}, DEFAULTS);

// ---------------------------------------------------------------- renderer ---
const canvas = document.getElementById('gl');
const appEl = document.getElementById('app');

window.__BH_ERR = [];
window.addEventListener('error', (e) => window.__BH_ERR.push('error: ' + (e.message || e)));
window.addEventListener('unhandledrejection', (e) => window.__BH_ERR.push('reject: ' + e.reason));
canvas.addEventListener('webglcontextlost', (e) => {
  e.preventDefault();
  window.__BH_ERR.push('webglcontextlost');
});
canvas.addEventListener('webglcontextrestored', () => {
  window.__BH_ERR.push('webglcontextrestored');
  allocateTargets();
});

let renderer;
try {
  renderer = new THREE.WebGLRenderer({
    canvas, antialias: false, alpha: false, depth: false, stencil: false,
    powerPreference: 'high-performance', preserveDrawingBuffer: false,
  });
} catch (e) {
  document.getElementById('fatal').style.display = 'flex';
  document.getElementById('fatal-msg').textContent = String(e);
  throw e;
}

renderer.outputColorSpace = THREE.LinearSRGBColorSpace;
renderer.toneMapping = THREE.NoToneMapping;
renderer.setPixelRatio(1);
renderer.setClearColor(0x000000, 1);

const gl = renderer.getContext();
const isWebGL2 = renderer.capabilities.isWebGL2;
const halfFloatRT = renderer.extensions.has('EXT_color_buffer_float') ||
                    renderer.extensions.has('EXT_color_buffer_half_float');
const HDR_TYPE = halfFloatRT ? THREE.HalfFloatType : THREE.UnsignedByteType;

// ------------------------------------------------------------------ camera ---
const camera = new THREE.PerspectiveCamera(P.fov, 1, 0.05, 20000);
setViewFromOrbit();

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 0, 0);
controls.enableDamping = true;
controls.dampingFactor = 0.075;
controls.rotateSpeed = 0.62;
controls.zoomSpeed = 0.9;
controls.panSpeed = 0.0;
controls.enablePan = false;
controls.minDistance = 4.2;
controls.maxDistance = 800;
controls.maxPolarAngle = Math.PI;
controls.autoRotateSpeed = 0.35;
controls.update();

function setViewFromOrbit() {
  const el = THREE.MathUtils.degToRad(P.elevation);
  const d = P.dist;
  camera.position.set(d * Math.cos(el), d * Math.sin(el), 0.0);
  camera.lookAt(0, 0, 0);
}

// ------------------------------------------------------------ fullscreen quad -
const fsScene = new THREE.Scene();
const fsCam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
const quad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2));
quad.frustumCulled = false;
fsScene.add(quad);

function drawPass(material, target) {
  quad.material = material;
  renderer.setRenderTarget(target);
  renderer.clear(true, false, false);
  renderer.render(fsScene, fsCam);
}

// --------------------------------------------------------------- materials ---
const geoUniforms = {
  uResolution: { value: new THREE.Vector2(1, 1) },
  uSampleIndex: { value: 0.0 },
  uJitter: { value: 1.0 },
  uCamPos: { value: new THREE.Vector3() },
  uCamBasis: { value: new THREE.Matrix3() },
  uTanHalfFov: { value: 0.5 },
  uAspect: { value: 1.0 },
  uTime: { value: 0.0 },
  uMaxSteps: { value: P.maxSteps },
  uTol: { value: P.tol },
  uStepScale: { value: P.stepScale },
  uEscapeR: { value: P.escapeR },
  uDiskInner: { value: P.diskInner },
  uDiskOuter: { value: P.diskOuter },
  uDiskThick: { value: P.diskThick },
  uDiskTemp: { value: P.diskTemp },
  uDiskBright: { value: P.diskBright },
  uDiskOpacity: { value: P.diskOpacity },
  uTurb: { value: P.turb },
  uSpin: { value: P.spin },
  uKepler: { value: P.kepler },
  uDoppler: { value: P.doppler },
  uFluxModel: { value: P.fluxModel },
  uShowDisk: { value: P.showDisk },
  uShowStars: { value: P.showStars },
  uStarBright: { value: P.starBright },
  uColorMode: { value: P.colorMode },
  uDispTMin: { value: P.dispTMin },
  uDispTMax: { value: P.dispTMax },
  uDebugMask: { value: 0.0 },
};

const geoMat = new THREE.ShaderMaterial({
  vertexShader: SH.FULLSCREEN_VERT,
  fragmentShader: SH.GEODESIC_FRAG,
  uniforms: geoUniforms,
  depthTest: false, depthWrite: false,
});

const brightMat = new THREE.ShaderMaterial({
  vertexShader: SH.FULLSCREEN_VERT, fragmentShader: SH.BRIGHT_FRAG,
  uniforms: { tSrc: { value: null }, uThreshold: { value: P.bloomThreshold }, uKnee: { value: 0.6 } },
  depthTest: false, depthWrite: false,
});

const blurMat = new THREE.ShaderMaterial({
  vertexShader: SH.FULLSCREEN_VERT, fragmentShader: SH.BLUR_FRAG,
  uniforms: { tSrc: { value: null }, uDir: { value: new THREE.Vector2() } },
  depthTest: false, depthWrite: false,
});

const blendMat = new THREE.ShaderMaterial({
  vertexShader: SH.FULLSCREEN_VERT, fragmentShader: SH.BLEND_FRAG,
  uniforms: { tPrev: { value: null }, tCur: { value: null }, uMix: { value: 1.0 } },
  depthTest: false, depthWrite: false,
});

const compMat = new THREE.ShaderMaterial({
  vertexShader: SH.FULLSCREEN_VERT, fragmentShader: SH.COMPOSITE_FRAG,
  uniforms: {
    tBase: { value: null }, tB0: { value: null }, tB1: { value: null }, tB2: { value: null },
    uBloom: { value: P.bloom }, uExposure: { value: P.exposure },
    uVignette: { value: P.vignette }, uGrain: { value: P.grain },
    uFrame: { value: 0 }, uResolution: { value: new THREE.Vector2(1, 1) },
  },
  depthTest: false, depthWrite: false,
});

// ---------------------------------------------------------------- targets ----
function makeRT(w, h, type) {
  const rt = new THREE.WebGLRenderTarget(Math.max(1, Math.floor(w)), Math.max(1, Math.floor(h)), {
    type: type || HDR_TYPE,
    format: THREE.RGBAFormat,
    minFilter: THREE.LinearFilter,
    magFilter: THREE.LinearFilter,
    depthBuffer: false,
    stencilBuffer: false,
    generateMipmaps: false,
  });
  rt.texture.colorSpace = THREE.NoColorSpace;
  return rt;
}

let rtScene, rtAccA, rtAccB, rtBright, rtTmp, rtB0, rtB1a, rtB1, rtB2a, rtB2;
let accFlip = false;
let SH_W = 1, SH_H = 1, CW = 1, CH = 1;

function allocateTargets() {
  let dpr = Math.min(window.devicePixelRatio || 1, 1.5);
  // The CSS box of the canvas is authoritative (100% of the viewport).  Using
  // getBoundingClientRect() instead of innerWidth/innerHeight keeps the drawing
  // buffer and the composited size identical even when WebView2 reports the
  // viewport in device pixels (high-DPI hosts).
  //
  // Desktop host (launcher.py) additionally pins the real visible viewport
  // through window.__BH_HOST_VIEWPORT, because WebView2 may be given a control
  // larger than its window on high-DPI displays.
  const hv = window.__BH_HOST_VIEWPORT;
  if (hv && hv.w > 80 && hv.h > 80) {
    appEl.style.width = Math.round(hv.w) + 'px';
    appEl.style.height = Math.round(hv.h) + 'px';
  } else {
    appEl.style.width = '';
    appEl.style.height = '';
  }
  const rect = canvas.getBoundingClientRect();
  if (rect.width < 2 || rect.height < 2) return;
  const cssW = Math.max(2, Math.round(rect.width));
  const cssH = Math.max(2, Math.round(rect.height));
  // Keep the drawing buffer inside a sane budget: the geodesic pass is heavy
  // and oversized float targets can trip a GPU device loss on weaker adapters.
  const MAX_PIXELS = 2.2e6;
  if (cssW * cssH * dpr * dpr > MAX_PIXELS) {
    dpr = Math.max(0.7, Math.sqrt(MAX_PIXELS / (cssW * cssH)));
  }
  renderer.setPixelRatio(dpr);
  renderer.setSize(cssW, cssH, false);
  CW = Math.max(2, canvas.width | 0);
  CH = Math.max(2, canvas.height | 0);
  SH_W = Math.max(2, Math.floor(CW * P.renderScale));
  SH_H = Math.max(2, Math.floor(CH * P.renderScale));

  const dispose = (rt) => { if (rt) rt.dispose(); };
  dispose(rtScene); dispose(rtAccA); dispose(rtAccB);
  dispose(rtBright); dispose(rtTmp);
  dispose(rtB0); dispose(rtB1a); dispose(rtB1); dispose(rtB2a); dispose(rtB2);

  rtScene = makeRT(SH_W, SH_H);
  rtAccA = makeRT(SH_W, SH_H);
  rtAccB = makeRT(SH_W, SH_H);
  rtBright = makeRT(SH_W / 2, SH_H / 2);
  rtTmp = makeRT(SH_W / 2, SH_H / 2);
  rtB0 = makeRT(SH_W / 2, SH_H / 2);
  rtB1a = makeRT(SH_W / 4, SH_H / 4);
  rtB1 = makeRT(SH_W / 4, SH_H / 4);
  rtB2a = makeRT(SH_W / 8, SH_H / 8);
  rtB2 = makeRT(SH_W / 8, SH_H / 8);

  geoUniforms.uResolution.value.set(SH_W, SH_H);
  geoUniforms.uAspect.value = SH_W / SH_H;
  compMat.uniforms.uResolution.value.set(CW, CH);
  camera.aspect = CW / CH;
  camera.updateProjectionMatrix();
  lastCssW = cssW; lastCssH = cssH;
  resetAccumulation();
  updateShadowCircle();
}

window.addEventListener('resize', allocateTargets);
window.addEventListener('bh-viewport', () => allocateTargets());

if (typeof ResizeObserver !== 'undefined') {
  new ResizeObserver(() => allocateTargets()).observe(canvas);
}

// ------------------------------------------------------------------- state ---
let simTime = 0.0;
let accCount = 0;
let accReset = true;
// Index of the sub-pixel sample used by the running mean.  It *must* advance
// together with ``accCount``: a frozen index would make every accumulated
// frame identical, so the progressive average would neither anti-alias the
// silhouette nor reduce the residual noise of the star field.
let sampleIndex = 0;
// Debug-only switch used by paper/tools/validate_accum.py to reproduce the
// frozen-seed behaviour that shipped before v1.0.1.  It is not part of the
// user-visible parameter set.
let jitterFrozen = false;
// Sample index override used by the sub-pixel shadow probe (measureShadow).
// The probe has to walk the sample index itself, so it cannot rely on the
// accumulation policy inside renderPipeline(); a negative value means "no
// override".
let probeSampleIndex = -1;
let frameCount = 0;
// Phase of the composite-pass film grain / dither.  It must be part of the
// accumulation epoch: ``uFrame`` used to be driven by the *global* frame
// counter, which ``resetAccumulation()`` does not touch, so two otherwise
// identical renders differed by a fresh dither pattern.  That made converged
// stills irreproducible and put a hard noise floor of ~4e-3 (luma, 8-bit) on
// every accumulation measurement (paper, section on the sample budget).
// Tying it to a counter that is reset with the accumulator makes the dither
// decorrelated across the frames of one epoch and deterministic across epochs.
let grainFrame = 0;
let lastParamsKey = '';
let lastCamKey = '';
let lastCssW = 0, lastCssH = 0;

function resetAccumulation() { accReset = true; accCount = 0; sampleIndex = 0; grainFrame = 0; }

function paramsKey() {
  return [P.maxSteps, P.tol, P.stepScale, P.escapeR, P.diskInner, P.diskOuter, P.diskThick,
          P.diskTemp, P.diskBright, P.diskOpacity, P.turb, P.spin, P.kepler, P.doppler,
          P.fluxModel, P.showDisk, P.showStars, P.starBright, P.colorMode, P.dispTMin, P.dispTMax,
          P.renderScale, P.fov, P.paused].join(',');
}

function camKey() {
  return [camera.position.x.toFixed(5), camera.position.y.toFixed(5), camera.position.z.toFixed(5)].join(',');
}

// ------------------------------------------------------------------ update ---
const tmpRight = new THREE.Vector3(), tmpUp = new THREE.Vector3(), tmpBack = new THREE.Vector3();

function updateUniforms() {
  const u = geoUniforms;
  camera.updateMatrixWorld();
  u.uCamPos.value.copy(camera.position);
  tmpRight.setFromMatrixColumn(camera.matrixWorld, 0);
  tmpUp.setFromMatrixColumn(camera.matrixWorld, 1);
  tmpBack.setFromMatrixColumn(camera.matrixWorld, 2).negate();
  u.uCamBasis.value.set(
    tmpRight.x, tmpUp.x, tmpBack.x,
    tmpRight.y, tmpUp.y, tmpBack.y,
    tmpRight.z, tmpUp.z, tmpBack.z,
  );
  const tanHalf = Math.tan(THREE.MathUtils.degToRad(camera.fov) * 0.5);
  u.uTanHalfFov.value = tanHalf;
  u.uAspect.value = CW / CH;
  u.uTime.value = simTime;
  u.uMaxSteps.value = Math.round(P.maxSteps);
  u.uTol.value = P.tol;
  u.uStepScale.value = P.stepScale;
  u.uEscapeR.value = Math.max(P.escapeR, camera.position.length() * 1.6);
  u.uDiskInner.value = P.diskInner;
  u.uDiskOuter.value = Math.max(P.diskOuter, P.diskInner * 1.2);
  u.uDiskThick.value = P.diskThick;
  u.uDiskTemp.value = P.diskTemp;
  u.uDiskBright.value = P.diskBright;
  u.uDiskOpacity.value = P.diskOpacity;
  u.uTurb.value = P.turb;
  u.uSpin.value = P.spin;
  u.uKepler.value = P.kepler;
  u.uDoppler.value = P.doppler;
  u.uFluxModel.value = P.fluxModel;
  u.uShowDisk.value = P.showDisk;
  u.uShowStars.value = P.showStars;
  u.uStarBright.value = P.starBright;
  u.uColorMode.value = P.colorMode;
  u.uDispTMin.value = P.dispTMin;
  u.uDispTMax.value = P.dispTMax;
  u.uDebugMask.value = P.debugMask ? 1.0 : 0.0;

  brightMat.uniforms.uThreshold.value = P.bloomThreshold;
  compMat.uniforms.uBloom.value = P.bloom;
  compMat.uniforms.uExposure.value = P.exposure;
  compMat.uniforms.uVignette.value = P.vignette;
  compMat.uniforms.uGrain.value = P.grain;
}

// ------------------------------------------------------------------ render ---
function renderPipeline(target) {
  const canAccum = P.accumulate && (P.paused || P.timeRate === 0.0);
  // Pin the sample index to zero when the accumulation is off: a moving index
  // without averaging is temporal dithering, i.e. visible flicker.
  // ``jitterFrozen`` reproduces the pre-1.0.1 behaviour, where the index was
  // never written, so that tools/validate_accum.py can measure the two
  // regimes side by side.
  geoUniforms.uSampleIndex.value = probeSampleIndex >= 0 ? probeSampleIndex
    : ((canAccum && !jitterFrozen) ? sampleIndex : 0.0);
  drawPass(geoMat, rtScene);

  const prev = accFlip ? rtAccB : rtAccA;
  const next = accFlip ? rtAccA : rtAccB;
  blendMat.uniforms.tPrev.value = accReset ? rtScene.texture : prev.texture;
  blendMat.uniforms.tCur.value = rtScene.texture;
  blendMat.uniforms.uMix.value = (!canAccum || accReset) ? 1.0 : 1.0 / (accCount + 1.0);
  drawPass(blendMat, next);
  accFlip = !accFlip;
  const base = next;
  accCount = Math.min(accCount + 1, 4096);
  if (canAccum) sampleIndex = Math.min(sampleIndex + 1, 4096);
  accReset = false;

  // bloom pyramid
  const px = 1.0 / Math.max(rtBright.width, 1);
  const py = 1.0 / Math.max(rtBright.height, 1);
  brightMat.uniforms.tSrc.value = base.texture;
  drawPass(brightMat, rtBright);

  blurMat.uniforms.tSrc.value = rtBright.texture;
  blurMat.uniforms.uDir.value.set(px, 0);
  drawPass(blurMat, rtTmp);
  blurMat.uniforms.tSrc.value = rtTmp.texture;
  blurMat.uniforms.uDir.value.set(0, py);
  drawPass(blurMat, rtB0);

  blurMat.uniforms.tSrc.value = rtB0.texture;
  blurMat.uniforms.uDir.value.set(px * 2.0, 0);
  drawPass(blurMat, rtB1a);
  blurMat.uniforms.tSrc.value = rtB1a.texture;
  blurMat.uniforms.uDir.value.set(0, py * 2.0);
  drawPass(blurMat, rtB1);

  blurMat.uniforms.tSrc.value = rtB1.texture;
  blurMat.uniforms.uDir.value.set(px * 4.0, 0);
  drawPass(blurMat, rtB2a);
  blurMat.uniforms.tSrc.value = rtB2a.texture;
  blurMat.uniforms.uDir.value.set(0, py * 4.0);
  drawPass(blurMat, rtB2);

  // composite
  compMat.uniforms.tBase.value = base.texture;
  compMat.uniforms.tB0.value = rtB0.texture;
  compMat.uniforms.tB1.value = rtB1.texture;
  compMat.uniforms.tB2.value = rtB2.texture;
  compMat.uniforms.uFrame.value = grainFrame % 1024;
  grainFrame++;
  drawPass(compMat, target || null);
  renderer.setRenderTarget(null);
}

// -------------------------------------------------------------------- loop ---
let lastTime = performance.now();
let fpsAccum = 0, fpsFrames = 0, fps = 0;

function tick() {
  requestAnimationFrame(tick);
  const now = performance.now();
  // Two different time steps on purpose.  ``frameMs`` is the true wall-clock
  // interval and is what the FPS read-out must use: clamping it would make
  // every frame slower than 10 Hz report exactly 10.0 fps.  ``dt`` is the
  // simulation step, clamped so that a stall (tab in background, GPU hitch)
  // cannot teleport the accretion-disk turbulence by a large amount.
  const frameMs = now - lastTime;
  const dt = Math.min(frameMs / 1000, 0.1);
  lastTime = now;
  frameCount++;

  controls.update();
  if (!P.paused) simTime += dt * P.timeRate;

  if (frameCount % 30 === 0) {
    const rc = canvas.getBoundingClientRect();
    if (Math.abs(rc.width - lastCssW) > 1 || Math.abs(rc.height - lastCssH) > 1) allocateTargets();
  }

  const pk = paramsKey();
  const ck = camKey();
  if (pk !== lastParamsKey || ck !== lastCamKey) {
    lastParamsKey = pk;
    lastCamKey = ck;
    resetAccumulation();
  }

  updateUniforms();
  renderPipeline(null);

  // HUD
  fpsAccum += frameMs / 1000; fpsFrames++;
  if (fpsAccum > 0.4) {
    fps = fpsFrames / fpsAccum;
    fpsAccum = 0; fpsFrames = 0;
    updateHud();
    if (P.autoQuality && frameCount > 90) {
      if (fps < 24 && P.renderScale > 0.45) {
        P.renderScale = Math.max(0.45, P.renderScale - 0.1);
        allocateTargets(); syncUI();
      } else if (fps > 58 && P.renderScale < 1.0) {
        P.renderScale = Math.min(1.0, P.renderScale + 0.05);
        allocateTargets(); syncUI();
      }
    }
  }
}

// --------------------------------------------------------------------- HUD ---
const hudEls = {
  fps: document.getElementById('hud-fps'),
  res: document.getElementById('hud-res'),
  gpu: document.getElementById('hud-gpu'),
  phys: document.getElementById('hud-phys'),
  acc: document.getElementById('hud-acc'),
};

function gpuName() {
  try {
    const dbg = gl.getExtension('WEBGL_debug_renderer_info');
    if (dbg) return gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL);
    return gl.getParameter(gl.RENDERER);
  } catch (e) { return 'unknown'; }
}
let GPU_NAME = gpuName();

function physicsReadout() {
  const r0 = camera.position.length();
  const bc = 3.0 * Math.sqrt(3.0);
  const f = Math.sqrt(Math.max(1.0 - 2.0 / r0, 0.0));
  const sinPsi = Math.min(1.0, bc * f / r0);
  const psi = Math.asin(sinPsi);
  const sinEl = camera.position.y / Math.max(r0, 1e-6);
  const incl = Math.asin(Math.min(1, Math.max(-1, sinEl))) * 180 / Math.PI;
  const vIsco = 0.5;                                      // sqrt(M/r)/sqrt(1-2M/r) at r=6M
  const redshiftIsco = Math.sqrt(1 - 3 / 6);              // sqrt(1-3M/r) at ISCO
  return { r0, bc, psi, incl, vIsco, redshiftIsco };
}

function updateHud() {
  const p = physicsReadout();
  hudEls.fps.textContent = fps.toFixed(1);
  hudEls.res.textContent = SH_W + 'x' + SH_H + '  (x' + P.renderScale.toFixed(2) + ')';
  hudEls.gpu.textContent = GPU_NAME;
  const diag = document.getElementById('diag');
  if (diag) {
    diag.innerHTML =
      '画布缓冲 <b>' + CW + '×' + CH + '</b> · CSS <b>' + canvas.clientWidth + '×' + canvas.clientHeight +
      '</b> · DPR <b>' + (window.devicePixelRatio || 1).toFixed(2) + '</b> · inner <b>' +
      window.innerWidth + '×' + window.innerHeight + '</b> · outer <b>' +
      window.outerWidth + '×' + window.outerHeight + '</b><br>' +
      '光线追踪 <b>' + SH_W + '×' + SH_H + '</b> · <b>' + fps.toFixed(0) + '</b> FPS · 累积 <b>' +
      ((P.accumulate && (P.paused || P.timeRate === 0)) ? accCount : 'off') + '</b>';
  }
  hudEls.acc.textContent = (P.accumulate && (P.paused || P.timeRate === 0)) ? String(accCount) : 'off';
  hudEls.phys.innerHTML =
    '<div><span>观测者半径 r₀</span><b>' + p.r0.toFixed(2) + ' M</b></div>' +
    '<div><span>视野倾角 (盘面)</span><b>' + Math.abs(p.incl).toFixed(1) + '°</b></div>' +
    '<div><span>事件视界 rₛ</span><b>2 M</b></div>' +
    '<div><span>光子球</span><b>3 M</b></div>' +
    '<div><span>最内稳定轨道 ISCO</span><b>6 M</b></div>' +
    '<div><span>临界碰撞参数 b_c</span><b>' + p.bc.toFixed(4) + ' M</b></div>' +
    '<div><span>理论阴影张角半径</span><b>' + (p.psi * 180 / Math.PI).toFixed(3) + '°</b></div>' +
    '<div><span>阴影屏上半径</span><b>' + shadowRadiusPx().toFixed(1) + ' px</b></div>' +
    '<div><span>ISCO 轨道速度</span><b>' + p.vIsco.toFixed(3) + ' c</b></div>' +
    '<div><span>ISCO 引力+运动红移</span><b>' + p.redshiftIsco.toFixed(4) + '</b></div>';
  updateShadowCircle();
}

function shadowRadiusPx() {
  return shadowRadiusPxFor(canvas.clientHeight || window.innerHeight);   // CSS px
}

// analytic apparent shadow radius:  sin(psi) = b_c sqrt(1 - rs/r0) / r0
function shadowAngle() {
  const r0 = camera.position.length();
  const bc = 3.0 * Math.sqrt(3.0);
  const f = Math.sqrt(Math.max(1.0 - 2.0 / r0, 0.0));
  return Math.asin(Math.min(1.0, bc * f / r0));
}

function shadowRadiusPxFor(heightPx) {
  const tanHalf = Math.tan(THREE.MathUtils.degToRad(camera.fov) * 0.5);
  return (heightPx * 0.5) * Math.tan(shadowAngle()) / tanHalf;
}

// Renders the capture mask and measures the black disc radius on the centre
// row.
//
// A single mask frame is binary, so the bright/dark boundary can only be
// reported to the nearest pixel (+-0.5 px, i.e. +-0.35% of the radius at a
// typical viewport) and the integer scan therefore carries a systematic
// quantisation error that is larger than every other error term in the
// verification suite (paper, section V3).  The probe instead renders
// ``SHADOW_PROBE_SAMPLES`` mask frames, each with a different sub-pixel sample
// offset taken from the same Cranley--Patterson sequence that drives the
// progressive accumulator.  Every pixel then records the fraction of those
// samples that landed inside the shadow, i.e. its coverage, and the two
// 0.5-coverage crossings on the centre row are solved by linear interpolation
// between neighbouring pixels.  That localises the edge to O(1/N) of a pixel
// at any resolution, and turns an error that was dominated by the sampling
// grid into one that is dominated by the physics.
const SHADOW_PROBE_SAMPLES = 24;
const SHADOW_DARK_LUMA = 40;    // 8-bit sRGB luma below which a pixel is "inside"

function measureShadow() {
  const saveBloom = P.bloom, saveVig = P.vignette, saveExp = P.exposure;
  const saveAccum = P.accumulate, saveMask = P.debugMask, saveGrain = P.grain;
  P.debugMask = 1; P.bloom = 0; P.vignette = 0; P.exposure = 1; P.grain = 0;
  P.accumulate = false;
  updateUniforms();

  const n = SHADOW_PROBE_SAMPLES;
  const y = CH >> 1, cx = CW >> 1;
  const rt = makeRT(CW, CH, THREE.UnsignedByteType);
  const row = new Uint8Array(CW * 4);
  const hits = new Uint16Array(CW);
  for (let k = 0; k < n; k++) {
    probeSampleIndex = k;
    renderPipeline(rt);
    // One row only: the read-back is what serialises the pipeline, and the
    // edge has to be found on the centre row anyway.
    renderer.readRenderTargetPixels(rt, 0, y, CW, 1, row);
    for (let x = 0; x < CW; x++) {
      const o = x * 4;
      const l = 0.2126 * row[o] + 0.7152 * row[o + 1] + 0.0722 * row[o + 2];
      if (l < SHADOW_DARK_LUMA) hits[x]++;
    }
  }
  probeSampleIndex = -1;
  rt.dispose();
  P.debugMask = saveMask; P.bloom = saveBloom; P.vignette = saveVig;
  P.exposure = saveExp; P.accumulate = saveAccum; P.grain = saveGrain;
  updateUniforms();
  resetAccumulation();

  // Coverage profile of the centre row, 1 = always captured, 0 = never.
  const cov = (x) => hits[x] / n;
  let l = cx, r = cx;
  while (l > 0 && hits[l - 1] > n / 2) l--;
  while (r < CW - 1 && hits[r + 1] > n / 2) r++;

  // Sub-pixel edge position.  For a straight silhouette the coverage of the
  // pixel *centres* is a ramp of unit slope (one unit of coverage per pixel),
  // so with the centre of pixel x at x + 1/2 a rising edge sits at
  // x + 1 - cov(x) and a falling edge at x + cov(x).
  //
  // That formula only holds on a *partially* covered pixel.  When the
  // outermost captured pixel is instead fully dark (cov = 1) the edge has
  // crossed into its outer neighbour, and the whole ramp lives in that
  // neighbour: the rising edge is then at l - cov(l - 1), and symmetrically
  // the falling edge at (r + 1) + cov(r + 1).  The two branches are mutually
  // exclusive - cov(x) < 1 implies the outer neighbour is completely empty -
  // so this is exact for a straight silhouette to O(1/N) in either branch.
  //
  // Using only the first branch (and clamping it into [l, l + 1]) is what
  // silently degraded the probe to an integer scan whenever the boundary
  // happened to land in the outer neighbour; the clamp is what threw the
  // sub-pixel part away, so it must not be used to pick the branch.
  const covLeft = (x) => (x >= 0 ? hits[x] / n : 0);
  const covRight = (x) => (x <= CW - 1 ? hits[x] / n : 0);
  const edgeL = hits[l] === n ? l - covLeft(l - 1) : l + 1 - cov(l);
  const edgeR = hits[r] === n ? (r + 1) + covRight(r + 1) : r + cov(r);
  const measured = (edgeR - edgeL) / 2;

  // Coverage quantisation +-1/(2N) propagates to +-1/(2N) pixels on a
  // unit-slope edge; the two edges are independent, so the radius carries
  // 1/(2*sqrt(2)*N).
  const sigmaPx = 1 / (2 * Math.SQRT2 * n);

  const analytic = shadowRadiusPxFor(CH);
  const rel = analytic > 0 ? (measured - analytic) / analytic * 100 : 0;
  return { measured, analytic, rel, sigmaPx, samples: n,
           subpixel: true, lo: l, hi: r, edgeL, edgeR, height: CH };
}

const svg = document.getElementById('overlay');
const circle = document.getElementById('shadow-circle');

function updateShadowCircle() {
  const show = P.shadowCircle;
  svg.style.display = show ? 'block' : 'none';
  if (!show) return;
  const r = shadowRadiusPx();
  circle.setAttribute('cx', (canvas.clientWidth || window.innerWidth) / 2);
  circle.setAttribute('cy', (canvas.clientHeight || window.innerHeight) / 2);
  circle.setAttribute('r', r.toFixed(2));
}

// --------------------------------------------------------------- interface ---
const CONTROLS = [
  { group: '吸积盘 (Shakura–Sunyaev / Novikov–Thorne)', open: true, items: [
    { k: 'showDisk', t: 'check', l: '显示吸积盘' },
    { k: 'fluxModel', t: 'select', l: '辐射通量模型 F(r)', num: true, opts: [
      ['1', 'Page–Thorne 广义相对论 (1974)'],
      ['0', 'Shakura–Sunyaev 牛顿近似 (1973)'],
    ]},
    { k: 'diskInner', t: 'range', l: '内缘 r<sub>in</sub> [M]', min: 6, max: 18, step: 0.1, fmt: v => v.toFixed(1) },
    { k: 'diskOuter', t: 'range', l: '外缘 r<sub>out</sub> [M]', min: 8, max: 80, step: 0.5, fmt: v => v.toFixed(1) },
    { k: 'diskThick', t: 'range', l: '厚度 H/r', min: 0.01, max: 0.25, step: 0.005, fmt: v => v.toFixed(3) },
    { k: 'diskTemp', t: 'range', l: '峰值温度 T<sub>max</sub> [K]', min: 1500, max: 40000, step: 100, fmt: v => v.toFixed(0) },
    { k: 'diskBright', t: 'range', l: '辐射强度', min: 0.05, max: 4, step: 0.05, fmt: v => v.toFixed(2) },
    { k: 'diskOpacity', t: 'range', l: '不透明度 κρ', min: 0, max: 8, step: 0.1, fmt: v => v.toFixed(1) },
    { k: 'turb', t: 'range', l: '湍流/差动旋转', min: 0, max: 1, step: 0.02, fmt: v => v.toFixed(2) },
    { k: 'spin', t: 'select', l: '盘旋转方向', opts: [['1', '逆时针 (+φ)'], ['-1', '顺时针 (−φ)']], num: true },
    { k: 'kepler', t: 'check', l: '开普勒公转 (v=√(M/r)/√(1−2M/r))' },
    { k: 'doppler', t: 'check', l: '相对论多普勒 + 引力红移 (g⁴ 增亮)' },
    { k: 'colorMode', t: 'check', l: '真实物理色温 (否则压缩显示)' },
  ]},
  { group: '背景星空 (引力透镜)', items: [
    { k: 'showStars', t: 'check', l: '显示星空' },
    { k: 'starBright', t: 'range', l: '星等亮度', min: 0, max: 3, step: 0.05, fmt: v => v.toFixed(2) },
  ]},
  { group: '相机 / 观测者', items: [
    { k: 'fov', t: 'range', l: '视场角 [°]', min: 15, max: 110, step: 1, fmt: v => v.toFixed(0), cb: () => { camera.fov = P.fov; camera.updateProjectionMatrix(); } },
    { k: 'autoRotate', t: 'check', l: '自动环绕', cb: () => { controls.autoRotate = P.autoRotate; } },
    { k: 'shadowCircle', t: 'check', l: '叠加理论阴影边界 (校验)', cb: () => updateShadowCircle() },
  ]},
  { group: '数值积分 / 性能', items: [
    { k: 'maxSteps', t: 'range', l: '每条光线最大步数', min: 60, max: 900, step: 10, fmt: v => v.toFixed(0) },
    { k: 'tol', t: 'range', l: '自适应步长容差 Δu/u', min: 0.008, max: 0.09, step: 0.002, fmt: v => v.toFixed(3) },
    { k: 'stepScale', t: 'range', l: '步长缩放', min: 0.4, max: 2.5, step: 0.05, fmt: v => v.toFixed(2) },
    { k: 'escapeR', t: 'range', l: '背景采样半径 [M]', min: 60, max: 600, step: 10, fmt: v => v.toFixed(0) },
    { k: 'renderScale', t: 'range', l: '渲染分辨率比例', min: 0.4, max: 1.5, step: 0.05, fmt: v => v.toFixed(2), cb: () => allocateTargets() },
    { k: 'autoQuality', t: 'check', l: '自适应分辨率 (按 FPS 调节)' },
    { k: 'accumulate', t: 'check', l: '静止时累积渲染 (去噪/抗锯齿)' },
    { k: 'timeRate', t: 'range', l: '时间流速 t/M 每秒', min: 0, max: 60, step: 1, fmt: v => v.toFixed(0) },
    { k: 'debugMask', t: 'check', l: '阴影掩膜 (白=逃逸, 黑=落入视界)' },
  ]},
  { group: '成像', items: [
    { k: 'exposure', t: 'range', l: '曝光', min: 0.05, max: 6, step: 0.05, fmt: v => v.toFixed(2) },
    { k: 'bloom', t: 'range', l: '泛光强度', min: 0, max: 2, step: 0.05, fmt: v => v.toFixed(2) },
    { k: 'bloomThreshold', t: 'range', l: '泛光阈值', min: 0.2, max: 4, step: 0.05, fmt: v => v.toFixed(2) },
    { k: 'vignette', t: 'range', l: '暗角', min: 0, max: 1, step: 0.05, fmt: v => v.toFixed(2) },
    { k: 'dispTMin', t: 'range', l: '显示色温下限 [K]', min: 800, max: 4000, step: 50, fmt: v => v.toFixed(0) },
    { k: 'dispTMax', t: 'range', l: '显示色温上限 [K]', min: 5000, max: 30000, step: 500, fmt: v => v.toFixed(0) },
  ]},
];

// Applying a preset must not depend on what the user was looking at a moment
// ago, so every preset is expanded from this fully explicit base instead of
// being a bare partial override.  (With a bare partial override, choosing
// "pure lensing" and then "X-ray disk" silently inherited showDisk = 0 and
// rendered a disk-less image -- see the paper, chapter on the UI layer.)
const PRESET_BASE = {
  dist: 26.0, elevation: 13.0, fov: 58.0, autoRotate: false,
  showDisk: 1, diskInner: 6.0, diskOuter: 26.0, diskThick: 0.075,
  diskTemp: 9000.0, diskBright: 1.0, diskOpacity: 2.6, turb: 0.55,
  spin: 1.0, kepler: 1.0, doppler: 1.0, fluxModel: 1.0,
  colorMode: 0.0, dispTMin: 2000.0, dispTMax: 14000.0,
  showStars: 1.0, starBright: 1.0,
  maxSteps: 300, tol: 0.03, stepScale: 1.0, escapeR: 140.0,
  timeRate: 9.0, paused: false, debugMask: 0, autoQuality: true,
  renderScale: 1.0, exposure: 1.0, bloom: 0.38, bloomThreshold: 1.6,
  vignette: 0.35, grain: 0.010, accumulate: true, shadowCircle: false,
};

const PRESETS = {
  '经典视界 (推荐)': { ...PRESET_BASE, dist: 26, elevation: 13, timeRate: 9 },
  '近观光子环': { ...PRESET_BASE, dist: 12, elevation: 7, diskOuter: 22, diskTemp: 11000, exposure: 0.9, bloom: 0.42, fov: 42, maxSteps: 520, tol: 0.02, turb: 0.5, paused: true },
  '俯视盘面': { ...PRESET_BASE, dist: 40, elevation: 58, diskOuter: 30, diskTemp: 8000, bloom: 0.28, fov: 50, maxSteps: 320, paused: true },
  '纯引力透镜 (无盘)': { ...PRESET_BASE, dist: 20, elevation: 6, showDisk: 0, starBright: 1.3, bloom: 0.25, fov: 55, maxSteps: 420, tol: 0.025, paused: true },
  'X 射线盘 (T=10⁷ K)': { ...PRESET_BASE, dist: 26, elevation: 11, diskTemp: 1.0e7, colorMode: 1, dispTMin: 2000, dispTMax: 14000, exposure: 0.75, bloom: 0.35, maxSteps: 340, paused: true },
  '牛顿盘对照 (SS 1973)': { ...PRESET_BASE, dist: 26, elevation: 13, fluxModel: 0, paused: true },
  '高分辨率静帧': { ...PRESET_BASE, dist: 24, elevation: 10, renderScale: 1.5, maxSteps: 800, tol: 0.015, accumulate: true, paused: true, fov: 50 },
};

const uiEls = {};   // key -> { input, out, item }

function buildUI() {
  const host = document.getElementById('controls');
  for (const grp of CONTROLS) {
    const sec = document.createElement('section');
    sec.className = 'grp';
    const h = document.createElement('h3');
    h.textContent = grp.group;
    sec.appendChild(h);
    const body = document.createElement('div');
    body.className = 'grp-body';
    for (const it of grp.items) {
      const row = document.createElement('label');
      row.className = 'row ' + it.t;
      const name = document.createElement('span');
      name.className = 'lbl';
      name.innerHTML = it.l;
      row.appendChild(name);

      let input = null, out = null;
      if (it.t === 'check') {
        input = document.createElement('input');
        input.type = 'checkbox';
        input.checked = !!P[it.k];
        input.addEventListener('change', () => {
          P[it.k] = input.checked ? 1 : 0;
          if (it.cb) it.cb();
          resetAccumulation();
        });
        row.appendChild(input);
      } else if (it.t === 'range') {
        const ctl = document.createElement('span');
        ctl.className = 'ctl';
        input = document.createElement('input');
        input.type = 'range';
        input.min = it.min; input.max = it.max; input.step = it.step;
        input.value = P[it.k];
        out = document.createElement('output');
        out.textContent = it.fmt ? it.fmt(P[it.k]) : P[it.k];
        input.addEventListener('input', () => {
          P[it.k] = parseFloat(input.value);
          out.textContent = it.fmt ? it.fmt(P[it.k]) : P[it.k];
          if (it.cb) it.cb();
          resetAccumulation();
        });
        ctl.appendChild(input);
        ctl.appendChild(out);
        row.appendChild(ctl);
      } else if (it.t === 'select') {
        input = document.createElement('select');
        for (const [v, t] of it.opts) {
          const o = document.createElement('option');
          o.value = v; o.textContent = t;
          if (parseFloat(v) === P[it.k]) o.selected = true;
          input.appendChild(o);
        }
        input.addEventListener('change', () => {
          P[it.k] = it.num ? parseFloat(input.value) : input.value;
          resetAccumulation();
        });
        row.appendChild(input);
      }
      if (input) uiEls[it.k] = { input, out, item: it };
      body.appendChild(row);
    }
    sec.appendChild(body);
    host.appendChild(sec);
  }
}

function syncUI() {
  for (const k of Object.keys(uiEls)) {
    const { input, out, item } = uiEls[k];
    const v = P[k];
    if (item.t === 'check') input.checked = !!v;
    else input.value = v;
    if (out && item.fmt) out.textContent = item.fmt(v);
  }
  const pb = document.getElementById('btn-pause');
  if (pb) pb.textContent = P.paused ? '▶ 继续时间' : '⏸ 暂停时间';
}

function applyPreset(name) {
  Object.assign(P, PRESETS[name]);
  controls.autoRotate = P.autoRotate;
  camera.fov = P.fov;
  camera.updateProjectionMatrix();
  setViewFromOrbit();
  controls.update();
  lastCamKey = '';
  resetAccumulation();
  syncUI();
  updateShadowCircle();
}

function buildPresets() {
  const host = document.getElementById('presets');
  for (const name of Object.keys(PRESETS)) {
    const b = document.createElement('button');
    b.textContent = name;
    b.addEventListener('click', () => applyPreset(name));
    host.appendChild(b);
  }
}

// ------------------------------------------------------------------- misc ----
function snapshot() {
  const rt = makeRT(CW, CH, THREE.UnsignedByteType);
  renderPipeline(rt);
  const buf = new Uint8Array(CW * CH * 4);
  renderer.readRenderTargetPixels(rt, 0, 0, CW, CH, buf);
  const cv = document.createElement('canvas');
  cv.width = CW; cv.height = CH;
  const ctx = cv.getContext('2d');
  const img = ctx.createImageData(CW, CH);
  for (let y = 0; y < CH; y++) {
    const src = (CH - 1 - y) * CW * 4;
    img.data.set(buf.subarray(src, src + CW * 4), y * CW * 4);
  }
  ctx.putImageData(img, 0, 0);
  rt.dispose();
  cv.toBlob((blob) => {
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'blackhole_' + Date.now() + '.png';
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 4000);
  }, 'image/png');
}

function wireButtons() {
  document.getElementById('btn-shot').addEventListener('click', snapshot);
  document.getElementById('btn-measure').addEventListener('click', () => {
    const m = measureShadow();
    const el = document.getElementById('measure-out');
    el.innerHTML =
      '实测阴影半径 <b>' + m.measured.toFixed(2) + '</b> px &nbsp;|&nbsp; ' +
      '解析值 <b>' + m.analytic.toFixed(2) + '</b> px &nbsp;|&nbsp; ' +
      '相对误差 <b>' + (m.rel >= 0 ? '+' : '') + m.rel.toFixed(2) + '%</b><br>' +
      '<span>（沿画面中线反解捕获掩膜，' + m.samples + ' 次分层子像素采样，' +
      '边缘定位 ±' + m.sigmaPx.toFixed(3) + ' px；b<sub>c</sub>=3√3 M）</span>';
  });
  document.getElementById('btn-reset').addEventListener('click', () => { applyPreset('经典视界 (推荐)'); });
  document.getElementById('btn-default').addEventListener('click', () => {
    Object.assign(P, DEFAULTS);
    controls.autoRotate = P.autoRotate;
    camera.fov = P.fov; camera.updateProjectionMatrix();
    setViewFromOrbit(); controls.update();
    allocateTargets(); syncUI();
  });
  const p = document.getElementById('btn-pause');
  p.addEventListener('click', () => { P.paused = !P.paused; resetAccumulation(); syncUI(); });
  // The render surface is a fixed, full-window layer and the sidebar floats on
  // top of it, so collapsing the panel only changes the *viewport* the camera
  // reports; the camera itself is positioned by OrbitControls from (r0, i, phi)
  // and therefore keeps looking at the origin.  That is what makes the shadow
  // stay pinned to the centre of the frame with the panel open or closed.
  const PANEL_KEY = 'bh.panelCollapsed';
  const panelCollapsed = () =>
    document.body.classList.contains('nopanel');
  const setPanel = (collapsed, remember) => {
    document.body.classList.toggle('nopanel', !!collapsed);
    if (remember !== false) {
      try { localStorage.setItem(PANEL_KEY, collapsed ? '1' : '0'); } catch (e) { /* private mode */ }
    }
    updateShadowCircle();
    requestAnimationFrame(allocateTargets);
  };
  const togglePanel = () => setPanel(!panelCollapsed());
  try {
    if (localStorage.getItem(PANEL_KEY) === '1') setPanel(true, false);
  } catch (e) { /* private mode */ }
  document.getElementById('btn-hide').addEventListener('click', togglePanel);
  document.getElementById('btn-collapse').addEventListener('click', togglePanel);
  document.getElementById('btn-expand').addEventListener('click', togglePanel);

  window.addEventListener('keydown', (e) => {
    if (e.key === 'h' || e.key === 'H' || e.key === 'f' || e.key === 'F') togglePanel();
    // Esc is the conventional "leave full screen" key: it only ever restores
    // the sidebar, so it cannot be confused with the F/H toggle.
    if (e.key === 'Escape' && panelCollapsed()) setPanel(false);
    if (e.key === ' ') { P.paused = !P.paused; resetAccumulation(); syncUI(); e.preventDefault(); }
    if (e.key === 's' || e.key === 'S') snapshot();
    if (e.key === 'r' || e.key === 'R') applyPreset('经典视界 (推荐)');
  });
}

// ------------------------------------------------------------------- boot ----
buildUI();
buildPresets();
wireButtons();
allocateTargets();

document.getElementById('hud-webgl').textContent =
  (isWebGL2 ? 'WebGL2' : 'WebGL1') + (halfFloatRT ? ' · RGBA16F' : ' · LDR');

// simple capability probe: measure one frame so the user can spot a software fallback
let probeRT = null;
function ldrPixel(x, y) {
  if (!probeRT || probeRT.width !== CW || probeRT.height !== CH) {
    if (probeRT) probeRT.dispose();
    probeRT = makeRT(CW, CH, THREE.UnsignedByteType);
  }
  renderPipeline(probeRT);
  const buf = new Uint8Array(4);
  renderer.readRenderTargetPixels(probeRT, Math.max(0, Math.min(CW - 1, x | 0)),
                                  Math.max(0, Math.min(CH - 1, y | 0)), 1, 1, buf);
  return [buf[0], buf[1], buf[2]];
}

function ldrColumn() {
  if (!probeRT || probeRT.width !== CW || probeRT.height !== CH) {
    if (probeRT) probeRT.dispose();
    probeRT = makeRT(CW, CH, THREE.UnsignedByteType);
  }
  renderPipeline(probeRT);
  const buf = new Uint8Array(CW * CH * 4);
  renderer.readRenderTargetPixels(probeRT, 0, 0, CW, CH, buf);
  const prof = [];
  for (let i = 0; i < 32; i++) {
    const y = Math.floor((i + 0.5) / 32 * CH);
    const x = CW >> 1;
    const o = (y * CW + x) * 4;
    prof.push([buf[o], buf[o + 1], buf[o + 2]]);
  }
  return prof;
}

window.__bh = {
  P, DEFAULTS, PRESETS, applyPreset, camera, controls, renderer,
  // ``renderScale`` changes the size of every render target, so it has to go
  // through allocateTargets() rather than only touching the parameter object.
  set: (k, v) => {
    P[k] = v;
    resetAccumulation();
    if (k === 'renderScale') allocateTargets();
    syncUI();
  },
  pixel: ldrPixel,
  column: ldrColumn,
  measure: measureShadow,
  // Synchronised single-frame cost.  A frame is pushed into an off-screen
  // target and a one-pixel read-back forces the driver to finish every GPU
  // command, so the samples measure real GPU work and not the
  // requestAnimationFrame rate limit of the host.  All three order statistics
  // are reported; the paper uses ``min`` as its estimator because contention,
  // thermal throttling and scheduler jitter can only add time, never remove
  // it, whereas ``median`` on a soft-rasterised machine is frequently
  // quantised to a ~2x-coarser level (see paper/tools/diag_taucap.json).
  cost: (n) => {
    const reps = Math.max(1, n | 0);
    const rt = makeRT(CW, CH, THREE.UnsignedByteType);
    const px = new Uint8Array(4);
    const ts = [];
    for (let i = 0; i < reps; i++) {
      const t0 = performance.now();
      renderPipeline(rt);
      renderer.readRenderTargetPixels(rt, 0, 0, 1, 1, px);
      ts.push(performance.now() - t0);
    }
    rt.dispose();
    ts.sort((a, b) => a - b);
    return { median: ts[ts.length >> 1], min: ts[0], max: ts[ts.length - 1], n: reps };
  },
  freezeJitter: (on) => { jitterFrozen = !!on; resetAccumulation(); return jitterFrozen; },
  // Synchronous progressive-accumulation probe used by
  // paper/tools/validate_accum.py.  It renders ``n`` frames back to back with
  // the current sample-index policy and reads the LDR result back, so the
  // convergence of the running mean can be measured without screenshots
  // (a screenshot keeps accumulating while the capture is taken and would
  // therefore not report a well-defined sample count).
  accumulateTo: (n) => {
    const rep = Math.max(1, n | 0);
    if (!probeRT || probeRT.width !== CW || probeRT.height !== CH) {
      if (probeRT) probeRT.dispose();
      probeRT = makeRT(CW, CH, THREE.UnsignedByteType);
    }
    const keepPaused = P.paused, keepRate = P.timeRate;
    P.paused = true; P.timeRate = 0.0;
    resetAccumulation();
    for (let i = 0; i < rep; i++) renderPipeline(probeRT);
    P.paused = keepPaused; P.timeRate = keepRate;
    const buf = new Uint8Array(CW * CH * 4);
    renderer.readRenderTargetPixels(probeRT, 0, 0, CW, CH, buf);
    const luma = (o) => (0.2126 * buf[o] + 0.7152 * buf[o + 1] + 0.0722 * buf[o + 2]) / 255;
    const sx = Math.max(1, Math.round(CW / 128));
    const sy = Math.max(1, Math.round(CH / 96));
    const grid = [];
    for (let y = 0; y < CH; y += sy) {
      for (let x = 0; x < CW; x += sx) grid.push(luma((y * CW + x) * 4));
    }
    const rowY = CH >> 1;
    const row = [];
    for (let x = 0; x < CW; x++) row.push(luma((rowY * CW + x) * 4));
    return { n: rep, w: CW, h: CH, sx, sy, rowY, grid, row };
  },
  // Deterministic playback hook: the paper's figures freeze the coordinate
  // time so that every re-render of a figure reproduces the same turbulence
  // pattern.  ``time`` returns the current coordinate time in units of M.
  time: () => simTime,
  setTime: (t) => { simTime = t; resetAccumulation(); },
  info: () => ({ gpu: GPU_NAME, webgl2: isWebGL2, hdr: halfFloatRT, fps, accCount,
                 shadowPx: shadowRadiusPx(), dist: camera.position.length(),
                 simTime, traced: SH_W + 'x' + SH_H, scale: P.renderScale,
                 steps: P.maxSteps, tol: P.tol }),
};

tick();
