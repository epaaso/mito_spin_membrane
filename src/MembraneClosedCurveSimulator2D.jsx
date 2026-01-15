import React, { useEffect, useMemo, useRef, useState } from "react";

/**
 * 2D Interactive simulator where the membrane is a SINGLE CLOSED CURVE.
 *
 * Key requirements you asked for (implemented):
 *  1) Only ONE membrane (single vesicle) should be treated as “the membrane”.
 *     - We extract the largest closed contour of φ=0.
 *     - We build a narrow-band mask around that contour.
 *     - All “membrane-only” fields (g, coupling, and membrane-local driving of m) are multiplied by that mask.
 *     => Even if φ develops other interfaces (numerical artifacts), g will NEVER appear on them.
 *
 *  2) g must live only on the membrane.
 *     - g_base is a template (angle-based) but effective g is:
 *         g_on_membrane(x,y) = g_base(x,y) * I(φ) * mask_main
 *       where I(φ)=max(0,1-φ^2)^p localizes to φ≈0 and mask_main localizes to the MAIN membrane only.
 *
 *  3) The membrane was drifting/wrapping due to periodic boundaries.
 *     - Optional: keep the vesicle centered by circularly shifting φ (and m) each frame.
 *
 * Dynamics (parsimonious & stable): mass-conserving Allen–Cahn
 *   μ = (φ^3-φ) - ε² ∇²φ - ρg*(g_base-0.5)*I_main - ρm*(m*I_main)
 *   ∂t φ = - ( μ - ⟨μ⟩ )
 * This conserves ⟨φ⟩ (so the vesicle doesn’t vanish) but is less prone than Cahn–Hilliard
 * to generating many small domains from noise.
 */

// ---------- RNG + utils ----------
function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function randn(rng) {
  let u = 0,
    v = 0;
  while (u === 0) u = rng();
  while (v === 0) v = rng();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

function idx2(i, j, N) {
  return i * N + j;
}

function clamp01(x) {
  return Math.max(0, Math.min(1, x));
}

function meanField(a) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i];
  return s / a.length;
}

function circularShift2D(src, N, shiftI, shiftJ) {
  // returns a new array shifted (periodic)
  const dst = new Float32Array(src.length);
  const si = ((shiftI % N) + N) % N;
  const sj = ((shiftJ % N) + N) % N;
  for (let i = 0; i < N; i++) {
    const ii = (i + si) % N;
    for (let j = 0; j < N; j++) {
      const jj = (j + sj) % N;
      dst[idx2(ii, jj, N)] = src[idx2(i, j, N)];
    }
  }
  return dst;
}

function centerOfMassPositive(phi, N) {
  // COM of inside region (phi>0)
  let sx = 0,
    sy = 0,
    w = 0;
  for (let i = 0; i < N; i++) {
    for (let j = 0; j < N; j++) {
      const v = phi[idx2(i, j, N)];
      if (v > 0) {
        sx += i;
        sy += j;
        w += 1;
      }
    }
  }
  if (w < 1) return { cx: N / 2, cy: N / 2, w: 0 };
  return { cx: sx / w, cy: sy / w, w };
}

// ---------- Periodic Laplacian (5-point stencil) ----------
function laplacianPeriodic(src, dst, N, invDx2) {
  for (let i = 0; i < N; i++) {
    const im1 = (i - 1 + N) % N;
    const ip1 = (i + 1) % N;
    const row = i * N;
    const rowIm1 = im1 * N;
    const rowIp1 = ip1 * N;
    for (let j = 0; j < N; j++) {
      const jm1 = (j - 1 + N) % N;
      const jp1 = (j + 1) % N;
      const c = src[row + j];
      const up = src[rowIm1 + j];
      const dn = src[rowIp1 + j];
      const lf = src[row + jm1];
      const rt = src[row + jp1];
      dst[row + j] = (up + dn + lf + rt - 4 * c) * invDx2;
    }
  }
}

// ---------- Initial φ: a closed vesicle (circle) + optional perturbations ----------
function makeInitialPhi({ N, L, mode, radiusFrac, noiseAmp, seed }) {
  const rng = mulberry32(seed);
  const dx = L / N;
  const phi = new Float32Array(N * N);

  const cx = 0.5 * L;
  const cy = 0.5 * L;
  const R = radiusFrac * L;

  for (let i = 0; i < N; i++) {
    const x = i * dx;
    let dxp = Math.abs(x - cx);
    dxp = Math.min(dxp, L - dxp);
    for (let j = 0; j < N; j++) {
      const y = j * dx;
      let dyp = Math.abs(y - cy);
      dyp = Math.min(dyp, L - dyp);
      const r = Math.sqrt(dxp * dxp + dyp * dyp);

      let sdist = r - R;
      if (mode === "wobbly") {
        const theta = Math.atan2(y - cy, x - cx);
        const wob = 0.10 * R * Math.sin(3 * theta) + 0.07 * R * Math.sin(5 * theta);
        sdist = r - (R + wob);
      }

      const base = Math.tanh(-sdist / (0.7 * dx));
      phi[idx2(i, j, N)] = base;
    }
  }

  if (mode === "noisy" || mode === "wobbly") {
    for (let k = 0; k < phi.length; k++) phi[k] += noiseAmp * randn(rng);
  }

  for (let k = 0; k < phi.length; k++) phi[k] = Math.max(-1.25, Math.min(1.25, phi[k]));
  return phi;
}

// ---------- g_base defined by angle around center ----------
function wrapAngle(a) {
  while (a > Math.PI) a -= 2 * Math.PI;
  while (a < -Math.PI) a += 2 * Math.PI;
  return a;
}

function makeGBaseAngle({ N, L, mode, nPatches, sigmaTheta, seed }) {
  const rng = mulberry32(seed);
  const dx = L / N;
  const g = new Float32Array(N * N);
  const cx = 0.5 * L;
  const cy = 0.5 * L;

  let centers = [];
  if (mode === "patches") {
    const K = Math.max(1, nPatches);
    centers = new Array(K).fill(0).map(() => rng() * 2 * Math.PI);
  }

  for (let i = 0; i < N; i++) {
    const x = i * dx;
    for (let j = 0; j < N; j++) {
      const y = j * dx;
      const theta = Math.atan2(y - cy, x - cx);

      let val = 0.5;
      if (mode === "uniform") {
        val = 0.5;
      } else if (mode === "sin") {
        const k = Math.max(1, nPatches);
        val = 0.5 + 0.5 * Math.sin(k * theta);
      } else {
        let s = 0;
        const s2 = sigmaTheta * sigmaTheta;
        for (let c = 0; c < centers.length; c++) {
          const d = wrapAngle(theta - centers[c]);
          s += Math.exp(-(d * d) / (2 * s2));
        }
        val = s;
      }

      g[idx2(i, j, N)] = val;
    }
  }

  let mn = Infinity,
    mx = -Infinity;
  for (let k = 0; k < g.length; k++) {
    mn = Math.min(mn, g[k]);
    mx = Math.max(mx, g[k]);
  }
  const denom = mx - mn || 1;
  for (let k = 0; k < g.length; k++) g[k] = (g[k] - mn) / denom;
  return g;
}

// ---------- Color map ----------
function colorMap(t) {
  const x = clamp01(t);
  if (x < 0.5) {
    const u = x / 0.5;
    const r = 10 + u * (80 - 10);
    const g = 30 + u * (230 - 30);
    const b = 120 + u * (255 - 120);
    return [r, g, b];
  } else {
    const u = (x - 0.5) / 0.5;
    const r = 80 + u * (255 - 80);
    const g = 230 + u * (220 - 230);
    const b = 255 + u * (60 - 255);
    return [r, g, b];
  }
}

function renderHeatmap(canvas, field, N, title, lockScale, scaleRef) {
  const ctx = canvas.getContext("2d");
  const W = canvas.width;
  const H = canvas.height;

  let mn = Infinity,
    mx = -Infinity;
  if (!lockScale || !scaleRef.current) {
    for (let i = 0; i < field.length; i++) {
      const v = field[i];
      mn = Math.min(mn, v);
      mx = Math.max(mx, v);
    }
    if (!isFinite(mn) || !isFinite(mx) || mn === mx) {
      mn = -1;
      mx = 1;
    }
    const pad = 0.08 * (mx - mn);
    mn -= pad;
    mx += pad;
    if (lockScale) scaleRef.current = { mn, mx };
  } else {
    mn = scaleRef.current.mn;
    mx = scaleRef.current.mx;
  }

  const img = ctx.createImageData(N, N);
  const data = img.data;
  const denom = mx - mn || 1;
  for (let i = 0; i < field.length; i++) {
    const t = (field[i] - mn) / denom;
    const [r, g, b] = colorMap(t);
    const k = 4 * i;
    data[k + 0] = r;
    data[k + 1] = g;
    data[k + 2] = b;
    data[k + 3] = 255;
  }

  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = "#0b1220";
  ctx.fillRect(0, 0, W, H);

  const off = document.createElement("canvas");
  off.width = N;
  off.height = N;
  const offCtx = off.getContext("2d");
  offCtx.putImageData(img, 0, 0);

  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(off, 0, 0, W, H);

  ctx.fillStyle = "rgba(255,255,255,0.9)";
  ctx.font = "600 13px ui-sans-serif, system-ui";
  ctx.fillText(title, 10, 18);
}

// ---------- Marching Squares ----------
function marchingSquaresLargestLoop(field, N, level = 0) {
  const segs = [];
  function interp(p1, p2, v1, v2) {
    const t = (level - v1) / (v2 - v1 || 1e-12);
    return { x: p1.x + t * (p2.x - p1.x), y: p1.y + t * (p2.y - p1.y) };
  }

  for (let j = 0; j < N - 1; j++) {
    for (let i = 0; i < N - 1; i++) {
      const v0 = field[idx2(i, j, N)];
      const v1 = field[idx2(i + 1, j, N)];
      const v2 = field[idx2(i + 1, j + 1, N)];
      const v3 = field[idx2(i, j + 1, N)];

      const b0 = v0 >= level ? 1 : 0;
      const b1 = v1 >= level ? 1 : 0;
      const b2 = v2 >= level ? 1 : 0;
      const b3 = v3 >= level ? 1 : 0;
      const code = (b0 << 3) | (b1 << 2) | (b2 << 1) | b3;
      if (code === 0 || code === 15) continue;

      const p0 = { x: i, y: j };
      const p1p = { x: i + 1, y: j };
      const p2p = { x: i + 1, y: j + 1 };
      const p3p = { x: i, y: j + 1 };

      const top = interp(p0, p1p, v0, v1);
      const right = interp(p1p, p2p, v1, v2);
      const bottom = interp(p3p, p2p, v3, v2);
      const left = interp(p0, p3p, v0, v3);

      switch (code) {
        case 1:
        case 14:
          segs.push([left, bottom]);
          break;
        case 2:
        case 13:
          segs.push([bottom, right]);
          break;
        case 3:
        case 12:
          segs.push([left, right]);
          break;
        case 4:
        case 11:
          segs.push([top, right]);
          break;
        case 5:
          segs.push([top, left]);
          segs.push([bottom, right]);
          break;
        case 6:
        case 9:
          segs.push([top, bottom]);
          break;
        case 7:
        case 8:
          segs.push([top, left]);
          break;
        case 10:
          segs.push([top, right]);
          segs.push([bottom, left]);
          break;
        default:
          break;
      }
    }
  }

  if (segs.length === 0) return null;

  const keyOf = (p) => `${Math.round(p.x * 1000)}/${Math.round(p.y * 1000)}`;
  const adj = new Map();
  const addAdj = (a, b) => {
    const ka = keyOf(a);
    if (!adj.has(ka)) adj.set(ka, []);
    adj.get(ka).push(b);
  };
  for (const [a, b] of segs) {
    addAdj(a, b);
    addAdj(b, a);
  }

  const visited = new Set();
  const edgeKey = (a, b) => {
    const ka = keyOf(a);
    const kb = keyOf(b);
    return ka < kb ? `${ka}|${kb}` : `${kb}|${ka}`;
  };

  const loops = [];
  for (const [a0, b0] of segs) {
    const ek0 = edgeKey(a0, b0);
    if (visited.has(ek0)) continue;

    const chain = [a0];
    let prev = a0;
    let curr = b0;
    visited.add(ek0);

    for (let guard = 0; guard < 200000; guard++) {
      chain.push(curr);
      const kc = keyOf(curr);
      const nbrs = adj.get(kc) || [];
      let next = null;
      for (const cand of nbrs) {
        const ek = edgeKey(curr, cand);
        if (visited.has(ek)) continue;
        if (keyOf(cand) === keyOf(prev) && nbrs.length > 1) continue;
        next = cand;
        visited.add(ek);
        break;
      }
      if (!next) break;
      prev = curr;
      curr = next;
      if (keyOf(curr) === keyOf(chain[0])) {
        chain.push(chain[0]);
        break;
      }
    }

    if (chain.length > 20 && keyOf(chain[0]) === keyOf(chain[chain.length - 1])) loops.push(chain);
  }

  if (loops.length === 0) return null;
  loops.sort((A, B) => B.length - A.length);
  return loops[0];
}

function drawMembraneOverlay(ctx, loopPts, canvasW, canvasH, N) {
  if (!loopPts || loopPts.length < 2) return;
  const sx = canvasW / N;
  const sy = canvasH / N;

  ctx.save();
  ctx.lineWidth = 2.4;
  ctx.strokeStyle = "rgba(255,255,255,0.95)";
  ctx.shadowColor = "rgba(0,0,0,0.35)";
  ctx.shadowBlur = 6;

  ctx.beginPath();
  for (let k = 0; k < loopPts.length; k++) {
    const p = loopPts[k];
    const x = p.x * sx;
    const y = p.y * sy;
    if (k === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();
  ctx.restore();
}

export default function MembraneClosedCurveSimulator2D() {
  // -------- UI state --------
  const [caseMode, setCaseMode] = useState("B");
  const [bgView, setBgView] = useState("g"); // phi | g | m

  const [initMode, setInitMode] = useState("noisy");
  const [seed, setSeed] = useState(3);

  const [N, setN] = useState(96);
  const [L, setL] = useState(64);

  // phase-field params
  const [epsilon, setEpsilon] = useState(1.25);
  const [radiusFrac, setRadiusFrac] = useState(0.22);
  const [noiseAmp, setNoiseAmp] = useState(0.03);

  // membrane localization
  const [iClamp, setIClamp] = useState(3.0); // sharpen I(φ)
  const [bandPx, setBandPx] = useState(4); // mask width around the MAIN contour (in grid pixels)

  // couplings
  const [rhoG, setRhoG] = useState(1.0);
  const [rhoM, setRhoM] = useState(1.0);

  // alignment params (Case B)
  const [chi, setChi] = useState(1.0);
  const [xi, setXi] = useState(1.1);
  const [mu, setMu] = useState(1.2);
  const [lam, setLam] = useState(0.9);
  const [gammaOut, setGammaOut] = useState(2.0);

  // g params
  const [gMode, setGMode] = useState("patches");
  const [nPatches, setNPatches] = useState(6);
  const [sigmaTheta, setSigmaTheta] = useState(0.45);

  // integration
  const [dt, setDt] = useState(0.01);
  const [substeps, setSubsteps] = useState(8);
  const [running, setRunning] = useState(false);

  // visuals
  const [lockScale, setLockScale] = useState(false);
  const [showMembrane, setShowMembrane] = useState(true);
  const [keepCentered, setKeepCentered] = useState(true);

  const heatRef = useRef(null);
  const rafRef = useRef(null);
  const scaleRef = useRef(null);

  // -------- Simulation buffers --------
  const phiRef = useRef(null);
  const mRef = useRef(null);
  const gBaseRef = useRef(null);

  // scratch
  const lapPhiRef = useRef(null);
  const muPhiRef = useRef(null);
  const lapMRef = useRef(null);

  const gMemRef = useRef(null);
  const maskMainRef = useRef(null);
  const offMaskCanvasRef = useRef(null);
  const latestLoopRef = useRef(null);

  const invDx2 = useMemo(() => {
    const dx = L / N;
    return 1 / (dx * dx);
  }, [L, N]);

  function IofPhi(phiVal) {
    let I0 = 1 - phiVal * phiVal;
    if (I0 < 0) I0 = 0;
    if (iClamp !== 1) I0 = Math.pow(I0, iClamp);
    return I0;
  }

  function ensureOffMaskCanvas() {
    if (offMaskCanvasRef.current && offMaskCanvasRef.current.width === N) return;
    const c = document.createElement("canvas");
    c.width = N;
    c.height = N;
    offMaskCanvasRef.current = c;
  }

  function updateMainMaskFromLoop(loopPts) {
    // maskMain = 1 in a narrow band around the MAIN membrane only
    const mask = maskMainRef.current;
    if (!mask) return;

    // default: zero
    mask.fill(0);

    if (!loopPts || loopPts.length < 2) return;

    ensureOffMaskCanvas();
    const c = offMaskCanvasRef.current;
    const ctx = c.getContext("2d");

    ctx.clearRect(0, 0, N, N);
    ctx.lineWidth = Math.max(1, bandPx);
    ctx.lineJoin = "round";
    ctx.lineCap = "round";
    ctx.strokeStyle = "rgba(255,255,255,1)";

    ctx.beginPath();
    for (let k = 0; k < loopPts.length; k++) {
      const p = loopPts[k];
      if (k === 0) ctx.moveTo(p.x, p.y);
      else ctx.lineTo(p.x, p.y);
    }
    ctx.stroke();

    const img = ctx.getImageData(0, 0, N, N).data;
    for (let i = 0; i < N * N; i++) {
      // alpha channel
      const a = img[4 * i + 3] / 255;
      mask[i] = a > 0.05 ? 1 : 0;
    }
  }

  const reset = () => {
    setRunning(false);
    scaleRef.current = null;

    const phi = makeInitialPhi({ N, L, mode: initMode, radiusFrac, noiseAmp, seed: seed + 100 });
    const gBase = makeGBaseAngle({ N, L, mode: gMode, nPatches, sigmaTheta, seed });

    const size = N * N;
    phiRef.current = phi;
    gBaseRef.current = gBase;

    lapPhiRef.current = new Float32Array(size);
    muPhiRef.current = new Float32Array(size);
    lapMRef.current = new Float32Array(size);

    gMemRef.current = new Float32Array(size);
    maskMainRef.current = new Float32Array(size);

    if (caseMode === "B") {
      const rng = mulberry32(seed + 200);
      const m = new Float32Array(size);
      for (let i = 0; i < size; i++) m[i] = 0.02 * randn(rng);
      mRef.current = m;
    } else {
      mRef.current = null;
    }

    // initialize mask
    const loopPts = marchingSquaresLargestLoop(phi, N, 0);
    latestLoopRef.current = loopPts;
    updateMainMaskFromLoop(loopPts);

    draw();
  };

  useEffect(() => {
    reset();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [caseMode, bgView, initMode, seed, N, L, epsilon, radiusFrac, noiseAmp, iClamp, bandPx, gMode, nPatches, sigmaTheta]);

  const step = () => {
    const phi0 = phiRef.current;
    const m0 = mRef.current;
    const gBase = gBaseRef.current;

    const lapPhi = lapPhiRef.current;
    const muPhi = muPhiRef.current;
    const lapM = lapMRef.current;
    const mask = maskMainRef.current;

    if (!phi0 || !gBase || !lapPhi || !muPhi || !mask) return;

    // Optionally re-center vesicle (prevents wrapping off-screen)
    let phi = phi0;
    let m = m0;
    if (keepCentered) {
      const { cx, cy, w } = centerOfMassPositive(phi0, N);
      if (w > 50) {
        const si = Math.round(N / 2 - cx);
        const sj = Math.round(N / 2 - cy);
        if (Math.abs(si) >= 1 || Math.abs(sj) >= 1) {
          phi = circularShift2D(phi0, N, si, sj);
          phiRef.current = phi;
          if (caseMode === "B" && m0) {
            m = circularShift2D(m0, N, si, sj);
            mRef.current = m;
          }
        }
      }
    }

    // Update MAIN membrane loop + mask ONCE per frame
    const loopPts = marchingSquaresLargestLoop(phi, N, 0);
    latestLoopRef.current = loopPts;
    updateMainMaskFromLoop(loopPts);

    const size = N * N;

    for (let s = 0; s < substeps; s++) {
      // μ = (φ^3-φ) - ε²∇²φ - ρg*(gBase-0.5)*I_main - ρm*(m*I_main)
      laplacianPeriodic(phi, lapPhi, N, invDx2);

      for (let i = 0; i < size; i++) {
        const I = IofPhi(phi[i]) * mask[i]; // MAIN membrane only
        const gMem = (gBase[i] - 0.5) * I;
        const mTerm = caseMode === "B" && m ? rhoM * (m[i] * I) : 0;
        muPhi[i] = (phi[i] * phi[i] * phi[i] - phi[i]) - (epsilon * epsilon) * lapPhi[i] - rhoG * gMem - mTerm;
      }

      // mass-conserving Allen–Cahn: dφ/dt = -(μ - <μ>)
      const muMean = meanField(muPhi);
      for (let i = 0; i < size; i++) {
        phi[i] += -dt * (muPhi[i] - muMean);
        if (phi[i] > 1.25) phi[i] = 1.25;
        if (phi[i] < -1.25) phi[i] = -1.25;
      }

      // m dynamics (Case B): keep it near MAIN membrane
      if (caseMode === "B" && m) {
        laplacianPeriodic(m, lapM, N, invDx2);
        for (let i = 0; i < size; i++) {
          const I = IofPhi(phi[i]) * mask[i];
          const out = (1 - I) * gammaOut;
          const gMem = (gBase[i] - 0.5) * I;
          const rhsM = (chi + out) * m[i] - xi * lapM[i] - mu * gMem - lam * I;
          m[i] += -dt * rhsM;
        }
      }
    }
  };

  const draw = () => {
    const canvas = heatRef.current;
    if (!canvas) return;

    const phi = phiRef.current;
    const m = mRef.current;
    const gBase = gBaseRef.current;
    const gMem = gMemRef.current;
    const mask = maskMainRef.current;

    if (!phi || !gBase || !gMem || !mask) return;

    const size = N * N;
    for (let i = 0; i < size; i++) {
      const I = IofPhi(phi[i]) * mask[i];
      gMem[i] = gBase[i] * I; // strictly: ONLY on the MAIN membrane
    }

    let field = gMem;
    let title = "g_on_main_membrane";
    if (bgView === "phi") {
      field = phi;
      title = "φ";
    } else if (bgView === "m") {
      field = m || new Float32Array(size);
      title = "m";
    }

    renderHeatmap(canvas, field, N, `${caseMode === "A" ? "Case A" : "Case B"} — background: ${title}`, lockScale, scaleRef);

    if (showMembrane) {
      const ctx = canvas.getContext("2d");
      const loopPts = latestLoopRef.current || marchingSquaresLargestLoop(phi, N, 0);
      drawMembraneOverlay(ctx, loopPts, canvas.width, canvas.height, N);
    }
  };

  // animation loop
  useEffect(() => {
    const loop = () => {
      if (running) {
        step();
        draw();
        rafRef.current = requestAnimationFrame(loop);
      }
    };
    if (running) rafRef.current = requestAnimationFrame(loop);
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [running, dt, substeps, epsilon, rhoG, rhoM, chi, xi, mu, lam, gammaOut, N, L, invDx2, lockScale, showMembrane, bgView, caseMode, iClamp, bandPx, keepCentered]);

  // -------- UI components --------
  // Tooltips: each control shows help on hover.
  // Edit text here if you want to expand explanations.
  const HELP_MAP = {
    Case: "A: no extra ordering. B: adds field m (alignment/order) coupled to g.",
    Background: "Which field is shown: g on membrane, φ (inside/outside), or m (order).",
    Initial: "Initial vesicle shape (always a closed curve).",
    "g pattern (along membrane)": "How g is distributed along the main membrane (angle around center).",
    Seed: "Reproducible seed (affects initial noise and g patches).",
    N: "Grid size N×N (higher N = more detail, slower).",
    L: "Physical domain size. dx=L/N.",
    "ε": "Interface width/smoothness. Lower ε = more flexible but needs smaller dt.",
    "initial radius": "Initial radius (fraction of L).",
    "initial noise": "Initial noise in φ (breaks symmetry).",
    "I sharp": "Exponent p in I(φ)=max(0,1−φ²)^p. Larger = thinner localization near φ≈0.",
    bandPx: "Band width (pixels) around the main contour where g and couplings live.",
    "ρg": "Strength of g→φ coupling (membrane deformation by g).",
    "ρm": "Strength of m→φ coupling (Case B only).",
    "# patches / frequency": "Number of g patches or sine frequency.",
    "σθ": "Angular width of each g patch. Smaller = tighter patches.",
    dt: "Time step (lower if unstable).",
    substeps: "Sub-iterations per frame (higher = more stable).",
    "χ": "m relaxation/gain (how fast it responds).",
    "ξ": "m diffusion (spatial smoothing).",
    "μ": "g→m coupling strength (how much m follows g).",
    "λ": "Bias keeping m active on the membrane.",
    "γ_out": "Damping of m outside the membrane.",
  };

  const helpFor = (label, hint) => {
    const k = String(label);
    return (HELP_MAP[k] || hint || "").trim();
  };

  const HelpIcon = ({ text }) => (
    <span
      className="ml-2 inline-flex h-5 w-5 items-center justify-center rounded-full border border-slate-700 bg-slate-900 text-[11px] font-bold text-slate-200"
      title={text}
      aria-label={text}
    >
      ?
    </span>
  );

  const Slider = ({ label, value, setValue, min, max, step, hint }) => {
    const helpText = helpFor(label, hint);
    return (
      <div className="space-y-1" title={helpText}>
        <div className="flex items-center justify-between gap-3">
          <div className="text-sm text-slate-200">
            <span className="font-medium">{label}</span>
            {helpText ? <HelpIcon text={helpText} /> : null}
            {hint ? <span className="ml-2 text-xs text-slate-400">{hint}</span> : null}
          </div>
          <div className="text-xs tabular-nums text-slate-300">{typeof value === "number" ? value.toFixed(3) : value}</div>
        </div>
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={value}
          onChange={(e) => setValue(parseFloat(e.target.value))}
          className="w-full accent-sky-400"
          title={helpText}
          aria-label={helpText || label}
        />
      </div>
    );
  };

  const IntSlider = ({ label, value, setValue, min, max, step, hint }) => {
    const helpText = helpFor(label, hint);
    return (
      <div className="space-y-1" title={helpText}>
        <div className="flex items-center justify-between gap-3">
          <div className="text-sm text-slate-200">
            <span className="font-medium">{label}</span>
            {helpText ? <HelpIcon text={helpText} /> : null}
            {hint ? <span className="ml-2 text-xs text-slate-400">{hint}</span> : null}
          </div>
          <div className="text-xs tabular-nums text-slate-300">{value}</div>
        </div>
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={value}
          onChange={(e) => setValue(parseInt(e.target.value, 10))}
          className="w-full accent-sky-400"
          title={helpText}
          aria-label={helpText || label}
        />
      </div>
    );
  };

  const Select = ({ label, value, setValue, options }) => {
    const helpText = helpFor(label);
    return (
      <label className="block" title={helpText}>
        <div className="mb-1 text-sm font-medium text-slate-200">
          {label}
          {helpText ? <HelpIcon text={helpText} /> : null}
        </div>
        <select
          value={value}
          onChange={(e) => setValue(e.target.value)}
          className="w-full rounded-xl border border-slate-700 bg-slate-900 px-3 py-2 text-sm text-slate-100 shadow"
          title={helpText}
          aria-label={helpText || label}
        >
          {options.map((o) => (
            <option key={o.value} value={o.value}>
              {o.label}
            </option>
          ))}
        </select>
      </label>
    );
  };

  return (
    <div className="min-h-screen w-full bg-gradient-to-b from-slate-950 to-slate-900 p-3 text-slate-100 app-shell">
      <div className="mx-auto max-w-7xl space-y-3 app-container">
        <header className="rounded-2xl border border-slate-800 bg-slate-950/60 p-3 shadow">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div className="min-w-0">
              <h1 className="text-lg font-semibold tracking-tight">Single membrane (closed φ=0 curve) — g ONLY on that membrane</h1>
              <p className="mt-1 text-xs text-slate-300">
                We extract the largest contour and build a narrow band around it: that is where <span className="font-semibold">g_on_main_membrane</span> lives.
              </p>
            </div>
            <div className="flex items-center gap-2">
              <button onClick={() => setRunning((r) => !r)} className={`rounded-xl px-4 py-2 text-sm font-semibold shadow ${running ? "bg-rose-500/90 hover:bg-rose-500" : "bg-emerald-500/90 hover:bg-emerald-500"}`}>
                {running ? "Pause" : "Start"}
              </button>
              <button
                onClick={() => {
                  step();
                  draw();
                }}
                className="rounded-xl border border-slate-700 bg-slate-900 px-4 py-2 text-sm font-semibold text-slate-100 shadow hover:bg-slate-800"
              >
                Step
              </button>
              <button onClick={reset} className="rounded-xl border border-slate-700 bg-slate-900 px-4 py-2 text-sm font-semibold text-slate-100 shadow hover:bg-slate-800">
                Reset
              </button>
            </div>
          </div>
        </header>

        <div className="grid grid-cols-1 gap-3 md:grid-cols-[minmax(0,1fr)_360px] md:items-start app-main">
          <div className="flex flex-col gap-2 rounded-2xl border border-slate-800 bg-slate-950/60 p-3 shadow canvas-panel">
            <div className="flex flex-wrap items-center justify-between gap-2 px-1">
              <div className="flex items-center gap-2">
                <div className="text-sm font-semibold text-slate-200">View</div>
                <div className="text-xs text-slate-400">(white line = main membrane; g appears only there)</div>
              </div>
              <div className="flex flex-wrap items-center gap-3">
                <label className="flex items-center gap-2 text-xs text-slate-300">
                  <input type="checkbox" checked={showMembrane} onChange={(e) => setShowMembrane(e.target.checked)} className="accent-sky-400" />
                  show membrane
                </label>
                <label className="flex items-center gap-2 text-xs text-slate-300">
                  <input
                    type="checkbox"
                    checked={keepCentered}
                    onChange={(e) => setKeepCentered(e.target.checked)}
                    className="accent-sky-400"
                  />
                  center vesicle
                </label>
                <label className="flex items-center gap-2 text-xs text-slate-300">
                  <input
                    type="checkbox"
                    checked={lockScale}
                    onChange={(e) => {
                      setLockScale(e.target.checked);
                      scaleRef.current = null;
                    }}
                    className="accent-sky-400"
                  />
                  lock scale
                </label>
              </div>
            </div>

            <canvas
              ref={heatRef}
              width={860}
              height={520}
              className="h-[40vh] min-h-[280px] max-h-[520px] w-full rounded-2xl border border-slate-800 bg-slate-950"
            />

            <div className="rounded-xl border border-slate-800 bg-slate-950/50 p-3 text-xs text-slate-300">
              <div className="font-semibold text-slate-200">Stability tips</div>
              <ul className="mt-2 list-disc space-y-1 pl-5">
                <li>If you see artifacts: lower <span className="font-semibold">dt</span> or raise <span className="font-semibold">substeps</span>.</li>
                <li>If g looks too thick: lower <span className="font-semibold">bandPx</span> or raise <span className="font-semibold">I sharp</span>.</li>
              </ul>
            </div>
          </div>

          <div className="space-y-3 controls-panel">
            <div className="grid grid-cols-1 gap-3 md:grid-cols-2 controls-grid">
              <div className="space-y-3 controls-col">
                <div className="rounded-2xl border border-slate-800 bg-slate-950/60 p-3 shadow">
                  <div className="grid grid-cols-2 gap-3">
                    <Select
                      label="Case"
                      value={caseMode}
                      setValue={(v) => {
                        setRunning(false);
                        setCaseMode(v);
                      }}
                      options={[
                        { value: "A", label: "Case A (no m)" },
                        { value: "B", label: "Case B (+ m)" },
                      ]}
                    />

                    <Select
                      label="Background"
                      value={bgView}
                      setValue={(v) => {
                        setRunning(false);
                        setBgView(v);
                        scaleRef.current = null;
                      }}
                      options={[
                        { value: "g", label: "g_on_main_membrane" },
                        { value: "phi", label: "φ" },
                        { value: "m", label: "m" },
                      ]}
                    />

                    <Select
                      label="Initial"
                      value={initMode}
                      setValue={(v) => {
                        setRunning(false);
                        setInitMode(v);
                      }}
                      options={[
                        { value: "circle", label: "Circle" },
                        { value: "noisy", label: "Circle + noise" },
                        { value: "wobbly", label: "Wobbly circle" },
                      ]}
                    />

                    <Select
                      label="g pattern (along membrane)"
                      value={gMode}
                      setValue={(v) => {
                        setRunning(false);
                        setGMode(v);
                      }}
                      options={[
                        { value: "patches", label: "Patches (angle)" },
                        { value: "sin", label: "Sine (angle)" },
                        { value: "uniform", label: "Uniform" },
                      ]}
                    />

                    <label className="col-span-2 block">
                      <div className="mb-1 text-sm font-medium text-slate-200">Seed</div>
                      <div className="flex gap-2">
                        <input
                          type="number"
                          value={seed}
                          onChange={(e) => {
                            setRunning(false);
                            setSeed(parseInt(e.target.value || "0", 10));
                          }}
                          className="w-full rounded-xl border border-slate-700 bg-slate-900 px-3 py-2 text-sm text-slate-100 shadow"
                        />
                        <button
                          onClick={() => {
                            setRunning(false);
                            setSeed((s) => (s + 1) % 100000);
                          }}
                          className="rounded-xl border border-slate-700 bg-slate-900 px-3 py-2 text-sm font-semibold text-slate-100 shadow hover:bg-slate-800"
                        >
                          +
                        </button>
                      </div>
                    </label>
                  </div>
                </div>

                <div className="rounded-2xl border border-slate-800 bg-slate-950/60 p-3 shadow">
                  <div className="space-y-3">
                    <div className="text-sm font-semibold text-slate-200">Resolution</div>
                    <IntSlider label="N" value={N} setValue={(v) => { setRunning(false); setN(v); }} min={64} max={256} step={16} />
                    <Slider label="L" value={L} setValue={(v) => { setRunning(false); setL(v); }} min={48} max={160} step={1} />

                    <div className="mt-2 text-sm font-semibold text-slate-200">Membrane</div>
                    <Slider label="ε" value={epsilon} setValue={(v) => { setRunning(false); setEpsilon(v); }} min={0.9} max={3.0} step={0.05} />
                    <Slider label="initial radius" value={radiusFrac} setValue={(v) => { setRunning(false); setRadiusFrac(v); }} min={0.12} max={0.35} step={0.005} />
                    <Slider label="initial noise" value={noiseAmp} setValue={(v) => { setRunning(false); setNoiseAmp(v); }} min={0.0} max={0.2} step={0.005} />

                    <div className="mt-2 text-sm font-semibold text-slate-200">g on membrane only</div>
                    <Slider label="I sharp" value={iClamp} setValue={(v) => { setRunning(false); setIClamp(v); }} min={1.0} max={8.0} step={0.25} />
                    <IntSlider label="bandPx" value={bandPx} setValue={(v) => { setRunning(false); setBandPx(v); }} min={1} max={12} step={1} hint="band width" />

                    <div className="mt-2 text-sm font-semibold text-slate-200">Couplings</div>
                    <Slider label="ρg" value={rhoG} setValue={setRhoG} min={0.0} max={3.0} step={0.02} />
                    <Slider label="ρm" value={rhoM} setValue={setRhoM} min={0.0} max={3.0} step={0.02} />
                  </div>
                </div>
              </div>

              <div className="space-y-3 controls-col">
                <div className="rounded-2xl border border-slate-800 bg-slate-950/60 p-3 shadow">
                  <div className="space-y-3">
                    <div className="text-sm font-semibold text-slate-200">g (angle)</div>
                    <IntSlider label="# patches / frequency" value={nPatches} setValue={(v) => { setRunning(false); setNPatches(v); }} min={1} max={16} step={1} />
                    <Slider label="σθ" value={sigmaTheta} setValue={(v) => { setRunning(false); setSigmaTheta(v); }} min={0.12} max={1.2} step={0.02} />

                    <div className="mt-2 text-sm font-semibold text-slate-200">Integration</div>
                    <Slider label="dt" value={dt} setValue={setDt} min={0.002} max={0.03} step={0.001} />
                    <IntSlider label="substeps" value={substeps} setValue={setSubsteps} min={1} max={30} step={1} />
                  </div>
                </div>

                <div className={`rounded-2xl border border-slate-800 bg-slate-950/60 p-3 shadow ${caseMode !== "B" ? "opacity-50" : ""}`}>
                  <div className="text-sm font-semibold text-slate-200">m (Case B only)</div>
                  <div className="mt-3 space-y-3">
                    <Slider label="χ" value={chi} setValue={setChi} min={0.2} max={3.0} step={0.02} />
                    <Slider label="ξ" value={xi} setValue={setXi} min={0.0} max={2.5} step={0.02} />
                    <Slider label="μ" value={mu} setValue={setMu} min={0.0} max={2.5} step={0.02} />
                    <Slider label="λ" value={lam} setValue={setLam} min={0.0} max={2.5} step={0.02} />
                    <Slider label="γ_out" value={gammaOut} setValue={setGammaOut} min={0.0} max={4.0} step={0.05} />
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        <footer className="rounded-2xl border border-slate-800 bg-slate-950/60 p-3 text-xs text-slate-400 shadow">
          <span className="font-semibold text-slate-300">Note:</span> if φ creates other interfaces, <span className="font-semibold">maskMain</span> keeps g and couplings only on the main membrane.
        </footer>
      </div>
    </div>
  );
}
