/* p2-12 — A neural layer: transformation plus a fold (Part 2 capstone).
   Width-3 tanh network on circles120, so the hidden space is exactly 3D. The storyboard's hand-verified weights are
   the primary weights. Step 9 replays a hard-coded gradient-descent run (seed 12, 1200 steps); Play → Retrain reruns
   the same deterministic trainer live. Spirals: 2 → 8 → 8 → 1 tanh, weights trained offline, shown as a labeled PCA-3
   shadow of the 8-D hidden space. */
(function () {
  'use strict';
  const N = LA.num, lerpV = LA.lerpV, lerpM = LA.lerpM, seg = LA.seg, C = LA.color, lerp = LA.lerp;
  const I2 = () => [[1, 0], [0, 1]];

  /* ---------- data (identical to the offline trainer) ---------- */
  function circles() {
    const r = LA.rng(12), X = [], y = [];
    for (let i = 0; i < 120; i++) { const inner = i < 60; const a = 2 * Math.PI * r(); const rad = (inner ? 0.5 : 1.5) + 0.1 * LA.rng.gauss(r); X.push([rad * Math.cos(a), rad * Math.sin(a)]); y.push(inner ? 1 : -1); }
    return { X, y };
  }
  function spirals() {
    const r = LA.rng(13), X = [], y = [];
    for (let k = 0; k < 2; k++) for (let i = 0; i < 100; i++) { const t = i / 99; const th = 1.5 * 2 * Math.PI * t + k * Math.PI; const rad = 0.25 + 2.0 * t; X.push([rad * Math.cos(th) + 0.08 * LA.rng.gauss(r), rad * Math.sin(th) + 0.08 * LA.rng.gauss(r)]); y.push(k === 0 ? 1 : -1); }
    return { X, y };
  }
  const CIRC = circles(), SPIR = spirals();
  const cls = y => (y > 0 ? 'data' : 'dataB');
  const lineAcc = (th, off) => { const n = [Math.cos(th), Math.sin(th)]; let a = 0; CIRC.X.forEach((x, i) => { if ((n[0] * x[0] + n[1] * x[1] > off ? 1 : -1) === CIRC.y[i]) a++; }); return Math.max(a, 120 - a) / 120; };
  const BEST = (function () { let b = { th: 0, off: 0, acc: 0 }; for (let k = 0; k < 90; k++) for (let q = 0; q <= 72; q++) { const th = Math.PI * k / 90, off = -1.8 + 0.05 * q, a = lineAcc(th, off); if (a > b.acc + 1e-9) b = { th, off, acc: a }; } return b; })();   // best straight cut ≈ 76%

  /* ---------- weights ---------- */
  const NETS = {   // hand-verified (storyboard): positive score = inner class
    tanh: { W1: [[2, 0], [-1, 1.732], [-1, -1.732]], b1: [1.5, 1.5, 1.5], w2: [1, 1, 1], b2: -1.5 },
    relu: { W1: [[1, 0], [-0.5, 0.866], [-0.5, -0.866]], b1: [0.3, 0.3, 0.3], w2: [-1, -1, -1], b2: 1.45 },
  };
  const ACTF = { tanh: Math.tanh, relu: x => (x > 0 ? x : 0), none: x => x };
  /* replay of trainCircles(1200, 0.5, 12): [step, W1 (6), b1 (3), w2 (3), b2, loss] every 25 steps.
     (The storyboard's lr 0.1 stalls at 94% on this data; lr 0.5 reaches 100%.) */
  const HIST = [[0,-0.424,-0.849,0.16,0.883,-0.694,-0.973,0.392,0.28,0.11,0.473,-0.939,0.856,-0.347,1.759],[25,-0.31,-0.76,-0.718,0.291,-0.953,-0.786,0.667,-0.57,1.051,-0.142,-0.448,0.448,-0.328,0.876],[50,-0.218,-0.445,-1.794,0.046,-1.83,-0.489,0.48,-1.055,1.164,-0.332,-1.413,1.359,-1.1,0.45],[75,-0.184,-0.22,-2.099,0.04,-2.056,0.091,0.428,-1.251,1.205,-0.414,-1.734,1.635,-1.393,0.372],[100,-0.165,-0.06,-2.281,0.055,-1.976,0.857,0.428,-1.29,1.284,-0.44,-1.988,1.909,-1.494,0.296],[125,-0.1,0.083,-2.353,0.24,-1.81,1.366,0.439,-1.316,1.161,-0.442,-2.181,2.207,-1.526,0.254],[150,-0.025,0.2,-2.415,0.371,-1.832,1.529,0.402,-1.307,1.066,-0.471,-2.44,2.47,-1.499,0.236],[175,0.032,0.269,-2.471,0.395,-1.875,1.57,0.336,-1.283,1.021,-0.531,-2.669,2.68,-1.489,0.226],[200,0.088,0.333,-2.51,0.403,-1.901,1.594,0.238,-1.258,0.988,-0.611,-2.863,2.866,-1.502,0.218],[225,0.153,0.419,-2.539,0.402,-1.91,1.613,0.096,-1.234,0.957,-0.71,-3.033,3.03,-1.55,0.209],[250,0.237,0.553,-2.558,0.396,-1.899,1.625,-0.093,-1.217,0.929,-0.816,-3.181,3.17,-1.661,0.199],[275,0.339,0.749,-2.56,0.388,-1.87,1.629,-0.312,-1.214,0.913,-0.917,-3.3,3.287,-1.839,0.186],[300,0.431,0.967,-2.546,0.377,-1.83,1.627,-0.51,-1.223,0.908,-1.024,-3.385,3.381,-2.039,0.172],[325,0.5,1.158,-2.524,0.363,-1.788,1.621,-0.661,-1.235,0.908,-1.134,-3.443,3.452,-2.227,0.163],[350,0.551,1.314,-2.498,0.347,-1.746,1.614,-0.772,-1.248,0.909,-1.24,-3.483,3.5,-2.397,0.156],[375,0.59,1.44,-2.469,0.332,-1.705,1.605,-0.856,-1.259,0.911,-1.341,-3.505,3.529,-2.553,0.151],[400,0.622,1.544,-2.436,0.317,-1.663,1.597,-0.922,-1.267,0.914,-1.441,-3.51,3.54,-2.701,0.147],[425,0.651,1.635,-2.397,0.307,-1.616,1.589,-0.977,-1.273,0.917,-1.55,-3.491,3.524,-2.853,0.142],[450,0.684,1.724,-2.334,0.334,-1.55,1.587,-1.026,-1.255,0.92,-1.709,-3.415,3.454,-3.043,0.134],[475,0.786,1.875,-2.121,0.557,-1.329,1.675,-1.132,-1.158,0.882,-2.23,-3.072,3.153,-3.473,0.055],[500,0.846,1.951,-2.024,0.445,-1.102,1.745,-1.263,-1.26,0.955,-2.551,-2.861,2.943,-3.824,0.014],[525,0.865,1.963,-2.014,0.374,-1.03,1.771,-1.33,-1.336,1.012,-2.615,-2.824,2.902,-3.992,0.009],[550,0.886,1.964,-2.019,0.326,-1.006,1.782,-1.375,-1.387,1.053,-2.645,-2.816,2.888,-4.107,0.007],[575,0.907,1.962,-2.029,0.29,-0.999,1.788,-1.409,-1.425,1.084,-2.666,-2.82,2.884,-4.195,0.006],[600,0.927,1.96,-2.041,0.261,-0.999,1.793,-1.436,-1.455,1.111,-2.683,-2.829,2.884,-4.267,0.005],[625,0.944,1.958,-2.053,0.237,-1.003,1.798,-1.46,-1.48,1.133,-2.698,-2.841,2.887,-4.328,0.005],[650,0.96,1.957,-2.066,0.218,-1.009,1.802,-1.48,-1.501,1.153,-2.712,-2.853,2.891,-4.382,0.004],[675,0.973,1.957,-2.078,0.201,-1.015,1.806,-1.497,-1.519,1.17,-2.725,-2.866,2.895,-4.429,0.004],[700,0.984,1.958,-2.091,0.188,-1.022,1.811,-1.513,-1.535,1.186,-2.738,-2.879,2.901,-4.472,0.004],[725,0.993,1.96,-2.103,0.177,-1.029,1.816,-1.526,-1.55,1.2,-2.75,-2.891,2.907,-4.511,0.003],[750,1.001,1.962,-2.114,0.168,-1.036,1.821,-1.538,-1.563,1.212,-2.761,-2.904,2.914,-4.547,0.003],[775,1.008,1.965,-2.126,0.16,-1.043,1.826,-1.549,-1.575,1.223,-2.773,-2.916,2.921,-4.58,0.003],[800,1.013,1.969,-2.136,0.154,-1.049,1.831,-1.559,-1.586,1.233,-2.784,-2.928,2.928,-4.611,0.003],[825,1.018,1.973,-2.147,0.149,-1.056,1.836,-1.568,-1.597,1.242,-2.795,-2.939,2.936,-4.64,0.003],[850,1.022,1.977,-2.157,0.144,-1.062,1.841,-1.576,-1.607,1.25,-2.806,-2.95,2.944,-4.667,0.003],[875,1.025,1.982,-2.167,0.14,-1.068,1.846,-1.583,-1.616,1.258,-2.816,-2.961,2.952,-4.693,0.003],[900,1.029,1.986,-2.176,0.137,-1.074,1.851,-1.59,-1.625,1.265,-2.827,-2.972,2.96,-4.717,0.002],[925,1.031,1.991,-2.185,0.134,-1.08,1.856,-1.597,-1.634,1.271,-2.837,-2.982,2.968,-4.74,0.002],[950,1.034,1.996,-2.194,0.131,-1.085,1.861,-1.603,-1.642,1.277,-2.847,-2.992,2.976,-4.762,0.002],[975,1.036,2,-2.203,0.129,-1.091,1.866,-1.609,-1.65,1.283,-2.856,-3.002,2.984,-4.783,0.002],[1000,1.038,2.005,-2.211,0.127,-1.096,1.871,-1.615,-1.657,1.288,-2.866,-3.011,2.992,-4.803,0.002],[1025,1.039,2.01,-2.219,0.125,-1.101,1.875,-1.62,-1.664,1.293,-2.875,-3.021,3,-4.822,0.002],[1050,1.041,2.014,-2.227,0.123,-1.106,1.88,-1.625,-1.671,1.298,-2.884,-3.03,3.007,-4.841,0.002],[1075,1.042,2.019,-2.234,0.122,-1.11,1.884,-1.63,-1.677,1.302,-2.893,-3.038,3.015,-4.859,0.002],[1100,1.044,2.024,-2.241,0.12,-1.115,1.889,-1.635,-1.684,1.306,-2.901,-3.047,3.022,-4.876,0.002],[1125,1.045,2.028,-2.248,0.119,-1.119,1.893,-1.639,-1.69,1.31,-2.91,-3.055,3.03,-4.893,0.002],[1150,1.046,2.032,-2.255,0.117,-1.124,1.897,-1.643,-1.696,1.314,-2.918,-3.063,3.037,-4.909,0.002],[1175,1.047,2.037,-2.262,0.116,-1.128,1.901,-1.648,-1.701,1.318,-2.926,-3.071,3.044,-4.925,0.002],[1200,1.048,2.041,-2.268,0.115,-1.132,1.905,-1.652,-1.707,1.321,-2.934,-3.079,3.051,-4.94,0.002]];
  const histNet = k => { const h = HIST[Math.max(0, Math.min(HIST.length - 1, k))]; return { W1: [[h[1], h[2]], [h[3], h[4]], [h[5], h[6]]], b1: [h[7], h[8], h[9]], w2: [h[10], h[11], h[12]], b2: h[13] }; };
  /* spirals 2 → 8 → 8 → 1 tanh, trained offline (full-batch Adam, 6000 steps, lr 0.02, init LA.rng(13) uniform ±1; 100% on spirals200) */
  const SP = {"W1":[[0.646,-0.075],[-0.578,-3.59],[-2.102,-2.601],[-3.271,-0.661],[-0.704,1.531],[-0.562,0.645],[-1.136,-1.886],[0.202,-0.296]],"b1":[-0.061,-0.161,-1.442,1.138,-1.914,1.174,-3.24,-1.474],"W2":[[2.86,-1.918,-3.811,1.888,-4.168,-3.522,-1.47,1.9],[2.777,-0.521,-1.211,1.466,-4.334,-1.991,-2.682,0.541],[-4.339,-1.514,1.292,-2.369,-1.377,-3.387,1.338,-1.059],[-5.467,0.971,-0.753,-4.204,2.165,2.803,1.751,0.177],[-2.841,0.617,-1.295,1.458,0.4,2.861,-1.82,-1.425],[1.93,0.667,2.85,-2.398,2.014,0.872,0.387,-1.37],[2.954,1.382,-0.825,4.139,-0.61,-2.207,-0.856,2.134],[-0.051,-3.071,0.221,-0.232,-1.027,-0.201,0.831,-0.859]],"b2":[0.926,1.167,1.09,1.322,0.325,-0.072,-0.637,-1.332],"w3":[-2.848,-2.748,-5.764,-5.378,1.261,5.797,-3.55,2.753],"b3":[-0.147]};
  function spH2(x) {
    const h1 = SP.W1.map((w, j) => Math.tanh(w[0] * x[0] + w[1] * x[1] + SP.b1[j]));
    return SP.W2.map((w, j) => { let a = SP.b2[j]; for (let k = 0; k < 8; k++) a += w[k] * h1[k]; return Math.tanh(a); });
  }
  const spScore = x => N.dot(SP.w3, spH2(x)) + SP.b3[0];
  /* PCA-3 of the 200 second-layer hidden vectors (computed once) */
  const SPCA = (function () {
    const H = SPIR.X.map(spH2), p = N.pca(H, 3);
    const proj = h => p.components.map(c => N.dot(c, N.sub(h, p.mean)));
    const kept = p.explained[0] + p.explained[1] + p.explained[2];
    const Z = H.map(proj);
    let R = 0; Z.forEach(z => { R = Math.max(R, Math.hypot(z[0], z[1], z[2])); });
    const nz = p.components.map(c => N.dot(c, SP.w3)), off = N.dot(SP.w3, p.mean) + SP.b3[0];
    const lines = [];
    for (let i = -4; i <= 4; i++) { const a = [], b = []; for (let k = 0; k <= 36; k++) { const u = -2.4 + 4.8 * k / 36; a.push(proj(spH2([i * 0.6, u]))); b.push(proj(spH2([u, i * 0.6]))); } lines.push(a, b); }
    let acc = 0; SPIR.X.forEach((x, i) => { if (Math.sign(spScore(x)) === SPIR.y[i]) acc++; });
    // a view direction 50° off the cut's normal, so the gold shadow plane is seen at an angle (not edge-on)
    const nh = N.normalize(nz); let u = N.cross(nh, [0, 0, 1]); if (N.norm(u) < 0.2) u = N.cross(nh, [1, 0, 0]); u = N.normalize(N.cross(u, nh));
    let e = N.normalize(N.add(N.scale(nh, Math.cos(0.87)), N.scale(u, Math.sin(0.87)))); if (e[2] < 0) e = N.scale(e, -1);
    const nKept = N.dot(nz, nz) / N.dot(SP.w3, SP.w3);   // share of the cut's normal that lies in the 3 shown directions (≈ 3%)
    return { Z, kept, nKept, R, nz, off, lines, acc: acc / SPIR.X.length, yaw: Math.atan2(e[1], e[0]), pitch: Math.min(1.2, Math.asin(e[2])) };
  })();

  /** the spirals' input-plane picture never changes: sample it once (heat grid + zero-contour segments) */
  let SPV = null;
  function spView() {
    if (SPV) return SPV;
    const BH = { x0: -3.4, x1: 3.4, y0: -7, y1: 7 }, B = { x0: -3.4, x1: 3.4, y0: -3.4, y1: 3.4 };
    const sample = (b, nx, ny, f) => { const G = []; for (let i = 0; i < ny; i++) { const y = b.y1 - (b.y1 - b.y0) * (i + 0.5) / ny, row = []; for (let j = 0; j < nx; j++) row.push(f([b.x0 + (b.x1 - b.x0) * (j + 0.5) / nx, y])); G.push(row); } return G; };
    const heat = sample(BH, 48, 96, p => Math.tanh(spScore(p)));
    const n = 96, fine = sample(B, n, n, spScore);   // bilinear lookup keeps g.contours cheap every frame
    const look = p => { const fx = (p[0] - B.x0) / (B.x1 - B.x0) * n - 0.5, fy = (B.y1 - p[1]) / (B.y1 - B.y0) * n - 0.5; const j = Math.max(0, Math.min(n - 2, Math.floor(fx))), i = Math.max(0, Math.min(n - 2, Math.floor(fy))); const a = fx - j, b = fy - i; return (fine[i][j] * (1 - a) + fine[i][j + 1] * a) * (1 - b) + (fine[i + 1][j] * (1 - a) + fine[i + 1][j + 1] * a) * b; };
    SPV = { BH, B, heat, look };
    return SPV;
  }

  /* ---------- the width-3 network ---------- */
  const netOf = s => ({ W1: s.W1, b1: s.b1, w2: s.w2, b2: s.b2 });
  function hid(net, act, x, fr, fa) {
    const p = [0, 1, 2].map(i => net.W1[i][0] * x[0] + net.W1[i][1] * x[1] + net.b1[i]);
    const f = ACTF[act] || ACTF.tanh, h = p.map(f);
    if (fr !== null && fr !== undefined && fa > 0) { p[fr] = lerp(p[fr], 0, fa); h[fr] = lerp(h[fr], 0, fa); }
    return { p, h };
  }
  const score = (net, act, x, fr, fa) => { const h = hid(net, act, x, fr, fa).h; return net.w2[0] * h[0] + net.w2[1] * h[1] + net.w2[2] * h[2] + net.b2; };
  function accuracy(net, act, fr, fa) { let a = 0; CIRC.X.forEach((x, i) => { if (Math.sign(score(net, act, x, fr, fa)) === CIRC.y[i]) a++; }); return a / CIRC.X.length; }
  const lossOf = net => { let l = 0; CIRC.X.forEach((x, i) => { l += (Math.tanh(score(net, 'tanh', x)) - CIRC.y[i]) ** 2; }); return l / CIRC.X.length; };
  /** position of input x at stage st (0 input plane at h₃ = 0, 1 affine, 2 fold, 3 score axis) */
  function stagePos(net, act, x, st, fr, fa) {
    const { p, h } = hid(net, act, x, fr, fa);
    if (st <= 1) return lerpV([x[0], x[1], 0], p, Math.max(0, st));
    if (st <= 2) return lerpV(p, h, st - 1);
    const nn = N.norm(net.w2) || 1, n = N.scale(net.w2, 1 / nn), foot = N.scale(n, -net.b2 / nn);
    const sc = (N.dot(net.w2, h) + net.b2) / nn;
    return lerpV(h, N.add(foot, N.scale(n, sc)), Math.min(1, st - 2));
  }
  /** radius/target per stage so the lifted sheet and the folded cube both fit */
  function stageView(net, act, st) {
    const s1 = N.svd(net.W1).S[0];
    const k0 = { r: 2.3, t: [0, 0, 0] }, k1 = { r: 1.95 * s1 + 0.6, t: N.scale(net.b1, 0.8) };
    const k2 = act === 'tanh' ? { r: 1.8, t: [0.1, 0.1, 0.05] } : act === 'relu' ? { r: 1.7, t: [0.55, 0.55, 0.45] } : k1;
    const a = st <= 1 ? [k0, k1, Math.max(0, st)] : st <= 2 ? [k1, k2, st - 1] : [k2, k2, 0];
    return { r: lerp(a[0].r, a[1].r, a[2]), t: lerpV(a[0].t, a[1].t, a[2]), k1, k2 };
  }
  /** a 3D camera (orbit angles from base) that fits a ball of the given radius in a viewport fw × fh of the canvas */
  function camFit3(g, fw, fh, radius, target, base) {
    const W = Math.max(50, g.size.w * fw), H = Math.max(50, g.size.h * fh), fov = (base && base.fov) || 36;
    const zoom = base && base.dist ? base.dist / 7 : 1;
    return { target, dist: 1.15 * radius / Math.tan(fov * Math.PI / 360) * Math.max(1, H / W) * zoom, yaw: base && base.yaw !== undefined ? base.yaw : 0.6, pitch: base && base.pitch !== undefined ? base.pitch : 0.38, fov, ortho: false, halfH: 4 };
  }
  function hcam(g, s, net, act, st, fw, fh, mul) { const v = stageView(net, act, st); return camFit3(g, fw, fh, v.r * (mul || 1), v.t, s.cam); }
  /** deterministic trainer (Play → Retrain): full-batch GD on mean squared error of tanh(score) vs ±1 */
  function trainCircles(steps, lr, seed) {
    const r = LA.rng(seed), U = () => 2 * r() - 1;
    const W = [U(), U(), U(), U(), U(), U()], b = [U(), U(), U()], w = [U(), U(), U()]; let c = U();
    const X = CIRC.X, Y = CIRC.y, n = X.length, curve = [];
    const g = new Float64Array(13), h = new Float64Array(3);
    for (let s = 0; s <= steps; s++) {
      let loss = 0; g.fill(0);
      for (let i = 0; i < n; i++) {
        const x0 = X[i][0], x1 = X[i][1];
        h[0] = Math.tanh(W[0] * x0 + W[1] * x1 + b[0]); h[1] = Math.tanh(W[2] * x0 + W[3] * x1 + b[1]); h[2] = Math.tanh(W[4] * x0 + W[5] * x1 + b[2]);
        const o = Math.tanh(w[0] * h[0] + w[1] * h[1] + w[2] * h[2] + c), e = o - Y[i]; loss += e * e;
        const d = 2 * e * (1 - o * o) / n; g[12] += d;
        for (let j = 0; j < 3; j++) { g[9 + j] += d * h[j]; const dh = d * w[j] * (1 - h[j] * h[j]); g[6 + j] += dh; g[2 * j] += dh * x0; g[2 * j + 1] += dh * x1; }
      }
      if (s % 25 === 0) curve.push(Math.round(loss / n * 1000) / 1000);
      if (s === steps) break;
      for (let j = 0; j < 6; j++) W[j] -= lr * g[j];
      for (let j = 0; j < 3; j++) { b[j] -= lr * g[6 + j]; w[j] -= lr * g[9 + j]; }
      c -= lr * g[12];
    }
    const R3 = v => Math.round(v * 1000) / 1000;
    return { net: { W1: [[W[0], W[1]], [W[2], W[3]], [W[4], W[5]]].map(r => r.map(R3)), b1: b.map(R3), w2: w.map(R3), b2: R3(c) }, curve };
  }

  /* ---------- small helpers ---------- */
  const pct = a => Math.round(a * 100) + '%';
  const f2 = x => LA.fmt(x);
  function fit(g, fw, fh, x0, x1, y0, y1) {
    const W = Math.max(50, g.size.w * fw), H = Math.max(50, g.size.h * fh), m = 1.06;
    return { center: [(x0 + x1) / 2, (y0 + y1) / 2], halfH: m * Math.max((y1 - y0) / 2, (x1 - x0) / 2 * H / W) };
  }
  const hiddenBase = () => Object.assign(LA.cam.defaults3d(), { target: [0, 0, 0], dist: 7, yaw: -0.3, pitch: 0.38 });   // ~55° off the lifted sheet's normal [1, 1, 1], so the sheet reads as tilted
  const cam2 = () => Object.assign(LA.cam.defaults2d(), { center: [0, 0], halfH: 2.6 });
  const STAGES = ['input plane', 'affine lift: W₁x + b₁', 'fold: tanh(W₁x + b₁)', 'score: w₂·h + b₂'];
  const ACTS = ['none', 'tanh', 'relu'];

  /** an on-canvas slider (works in learn steps marked interactive, and in Play/Check modes) */
  function slider(g, o) {
    const R = o.rect, W = g.size.w * R.w, H = g.size.h * R.h, halfH = 0.6 * H / W;
    g.viewport(R, { view: '2d', cam: { center: [0.5, 0.18 * halfH], halfH }, bg: 'panel' }, (g, r) => {
      g.line([0, 0], [1, 0], { color: 'line', width: 5 });
      g.line([0, 0], [o.u, 0], { color: o.color || 'gold', width: 5, alpha: 0.7 });
      (o.ticks || []).forEach(tk => { g.point([tk.u, 0], { r: 3, color: 'muted' }); g.label(tk.label, [tk.u, 0], { color: tk.on ? (o.color || 'gold') : 'muted', size: 12, font: 'ui', anchor: 'below', dy: 12, bg: false }); });
      g.point([o.u, 0], { r: 7.5, color: o.color || 'gold' });
      g.hud(o.title, r.x + 8, r.y + 13, { size: 12, color: 'hud' });
      g.handle(o.id, [o.u, 0], (p, st) => o.onSet(st, LA.clamp(p[0], 0, 1)), { color: o.color || 'gold', r: 18 });
    });
  }
  function stageSlider(g, s) {
    slider(g, { id: 'stage', rect: { x: 0.06, y: 0.86, w: 0.88, h: 0.12 }, u: s.stage / 3, title: 'layer stage (drag)', ticks: STAGES.map((nm, k) => ({ u: k / 3, label: ['input', 'affine', 'fold', 'score'][k], on: Math.abs(s.stage - k) < 0.15 })), onSet: (st, u) => { st.stage = 3 * u; } });
  }
  function actSlider(g, s) {
    const k = Math.max(0, ACTS.indexOf(s.act));
    slider(g, { id: 'act', rect: { x: 0.2, y: 0.86, w: 0.6, h: 0.12 }, u: k / 2, title: 'activation (drag)', ticks: ['none', 'tanh', 'ReLU'].map((nm, i) => ({ u: i / 2, label: nm, on: i === k })), onSet: (st, u) => { const a = ACTS[Math.round(u * 2)]; if (a !== st.act) setAct(st, a); } });
  }
  function freezeSlider(g, s) {
    const k = s.frozenUnit === null || s.frozenUnit === undefined || s.freezeAmt < 0.5 ? 0 : s.frozenUnit + 1;
    slider(g, { id: 'freeze', rect: { x: 0.2, y: 0.86, w: 0.6, h: 0.12 }, u: k / 3, title: 'freeze a hidden unit (drag)', ticks: ['none', 'h₁', 'h₂', 'h₃'].map((nm, i) => ({ u: i / 3, label: nm, on: i === k })), onSet: (st, u) => { const i = Math.round(u * 3); st.frozenUnit = i === 0 ? null : i - 1; st.freezeAmt = i === 0 ? 0 : 1; } });
  }
  function b2Slider(g, s) {
    const nn = N.norm(s.w2) || 1, d = -s.b2 / nn;   // plane: n̂·h = d
    slider(g, { id: 'b2', rect: { x: 0.2, y: 0.86, w: 0.6, h: 0.12 }, u: (d + 1.7) / 3.4, title: 'slide the gold plane (b₂)', ticks: [{ u: 0, label: '−1.7' }, { u: 0.5, label: '0' }, { u: 1, label: '1.7' }], onSet: (st, u) => { const n2 = N.norm(st.w2) || 1; st.b2 = -(u * 3.4 - 1.7) * n2; } });
  }
  function setAct(s, a) {
    s.act = a; const W = NETS[a === 'relu' ? 'relu' : 'tanh'];
    s.W1 = LA.clone(W.W1); s.b1 = LA.clone(W.b1); s.w2 = LA.clone(W.w2); s.b2 = W.b2;
  }

  /* ---------- drawing ---------- */
  function hiddenAxes(g, L, fr, fa, mini) {
    ['i', 'j', 'k'].forEach((c, k) => {
      const d = [0, 0, 0]; d[k] = L; const al = fr === k ? 1 - 0.8 * fa : 0.9;
      g.vector(N.scale(d, -1), d, { color: c, width: 1.6, head: 8, alpha: al });
      if (!mini) g.label(['h₁', 'h₂', 'h₃'][k], N.scale(d, 1.1), { color: c, font: 'math', alpha: al, size: 15 });
    });
  }
  /** the hidden space: sheet, points, cube, gold plane (draw inside a 3D viewport) */
  function hiddenScene(g, s, o) {
    o = o || {};
    const net = o.net || netOf(s), act = o.act || s.act, st = o.stage === undefined ? s.stage : o.stage;
    const fr = o.frozen === undefined ? s.frozenUnit : o.frozen, fa = o.freezeAmt === undefined ? s.freezeAmt : o.freezeAmt;
    const P = x => stagePos(net, act, x, st, fr, fa);
    const pa = o.planeAlpha !== undefined ? o.planeAlpha : (s.plane === null || s.plane === undefined ? seg(st, 1.8, 2) : s.plane);
    const v = stageView(net, act, st);
    const L1 = 0.75 * v.k1.r, L2 = act === 'tanh' ? 1.45 : act === 'relu' ? 2.2 : L1;
    const L = st <= 1 ? lerp(2.2, L1, Math.max(0, st)) : lerp(L1, L2, Math.min(1, st - 1));
    hiddenAxes(g, L, fr, fa, o.mini);
    if (act === 'tanh' && st > 1.3) g.box(null, { origin: [-1, -1, -1], size: 2, faces: false, stroke: 'muted', strokeAlpha: 0.5 * seg(st, 1.3, 1.8), width: 1 });
    const colorAt = pa > 0.3 ? (u, w) => cls(score(net, act, [u, w], fr, fa)) : null;
    g.paramSurface((u, w) => P([u, w]), { u0: -1.9, u1: 1.9, v0: -1.9, v1: 1.9 }, { res: o.res || 20, mode: 'both', color: 'teal', alpha: colorAt ? 0.2 : 0.14, wireAlpha: 0.32, colorAt });
    g.points(CIRC.X.map(P), { r: o.r || 3.6, colors: i => cls(CIRC.y[i]) });
    if (pa > 0.01) {
      const nn = N.norm(net.w2) || 1, n = N.scale(net.w2, 1 / nn), foot = N.scale(n, -net.b2 / nn);
      g.planes([{ point: foot, normal: n, extent: act === 'relu' ? 1.7 : 1.2, color: 'gold', alpha: 0.26 * pa }]);
      if (!o.mini) g.vector(foot, N.add(foot, N.scale(n, 0.8)), { color: 'gold', label: 'w₂', width: 2.6, alpha: pa });
      if (o.normalHandle) g.handle('normal', N.add(foot, N.scale(n, 0.8)), (p, stt) => { const dd = -stt.b2 / (N.norm(stt.w2) || 1); const ft = N.scale(N.normalize(stt.w2), dd); const nv = N.sub(p, ft); if (N.norm(nv) > 0.05) { stt.w2 = N.scale(N.normalize(nv), Math.sqrt(3)); stt.b2 = -dd * Math.sqrt(3); } }, { color: 'gold', plane: 'view' });
    }
  }
  function actInset(g, act, rect) {
    g.viewport(rect, { view: '2d', cam: { center: [0, act === 'relu' ? 0.9 : 0], halfH: act === 'relu' ? 1.6 : 1.5 }, bg: 'panel' }, (g, r) => {
      g.line([-3, 0], [3, 0], { color: 'gridAxis', width: 1 }); g.line([0, -2], [0, 3], { color: 'gridAxis', width: 1 });
      const pts = []; for (let k = 0; k <= 40; k++) { const x = -2.5 + 5 * k / 40; pts.push([x, ACTF[act](x)]); }
      g.polyline(pts, { color: 'gold', width: 2.2 });
      if (act === 'tanh') { g.line([-3, 1], [3, 1], { color: 'muted', dashed: true, width: 1 }); g.line([-3, -1], [3, -1], { color: 'muted', dashed: true, width: 1 }); }
      g.hud(act === 'relu' ? 'ReLU: max(0, t)' : 'tanh(t)', r.x + 6, r.y + 12, { size: 11, color: 'gold' });
    });
  }
  function boundary2d(g, s, o) {
    o = o || {};
    const net = o.net || netOf(s), act = o.act || s.act, fr = o.frozen === undefined ? s.frozenUnit : o.frozen, fa = o.freezeAmt === undefined ? s.freezeAmt : o.freezeAmt;
    const B = { x0: -3.4, x1: 3.4, y0: -3.4, y1: 3.4 }, F = p => score(net, act, p, fr, fa);
    if (o.heat > 0) g.heat2d(p => Math.tanh(2 * F(p)), B, { res: 60, range: [-1, 1], ramp: t => (t < 0.5 ? C.mix(C.dataB, C.bg, t * 2) : C.mix(C.bg, C.data, (t - 0.5) * 2)), alpha: 0.32 * o.heat });
    else g.ghostGrid();
    if (o.contour !== false) g.contours(F, [0], B, { res: o.res || 64, color: 'gold', width: 2.6, alpha: o.contourAlpha === undefined ? 1 : o.contourAlpha });
    g.points(CIRC.X, { r: o.r || 3.6, colors: i => cls(CIRC.y[i]) });
  }

  /* ---------- scenes ---------- */
  function sceneLine(g, s, ctx) {
    g.ghostGrid(); g.axes({ labels: false });
    g.points(CIRC.X, { r: 4, colors: i => cls(CIRC.y[i]) });
    const n = [Math.cos(s.th), Math.sin(s.th)], d = [-n[1], n[0]];
    g.lineThrough(N.scale(n, s.off), d, { color: 'gold', width: 2.6, len: 12 });
    const a = lineAcc(s.th, s.off);
    g.hud((s.atBest ? 'best possible line: ' : 'this line: ') + pct(a) + ' correct', 12, 22, { size: 15, color: 'gold' });
    g.hud('synthetic rings · 60 + 60 points', 12, 44, { size: 12, color: 'muted' });
    if (s.cross) { g.hud('straight boundary', 12, ctx.size.h - 22, { size: 15, color: 'bad' }); g.marker.cross([160, ctx.size.h - 22], { screen: true, size: 22 }); }
  }
  function sceneStack(g, s, ctx) {
    g.ghostGrid(); g.grid(s.M, { alpha: 0.85 });
    g.points(CIRC.X.map(x => N.matVec(s.M, x)), { r: 3.6, colors: i => cls(CIRC.y[i]) });
    const a = N.matVec(s.M, [0.35, -4]), b = N.matVec(s.M, [0.35, 4]);
    g.line(a, b, { color: 'gold', width: 2.6 });
    g.vector(N.col(s.M, 0), { color: 'i', label: 'i', hat: true, width: 2.6 });
    g.vector(N.col(s.M, 1), { color: 'j', label: 'j', hat: true, width: 2.6 });
    const names = ['W₁ · shear', 'W₂ · rotate', 'W₃ · scale', 'W₃W₂W₁ · one matrix'];
    g.hud(names[s.phase], 12, 22, { size: 15, color: s.phase === 3 ? 'gold' : 'hud' });
    g.hud('the gold cut stays a straight line · this one: ' + pct(lineAcc(0, 0.35)) + ' correct', 12, 44, { size: 12, color: 'muted' });
    if (s.cross) { g.hud('W₂W₁ bends the boundary', 12, ctx.size.h - 22, { size: 15, color: 'bad' }); g.marker.cross([206, ctx.size.h - 22], { screen: true, size: 22 }); }
  }
  function sceneHidden(g, s, ctx) {
    const net = netOf(s), wid = s.widget;
    const h = wid ? 0.84 : 1;
    g.viewport({ x: 0, y: 0, w: 1, h }, { view: '3d', cam: hcam(g, s, net, s.act, s.stage, 1, h), border: false }, (g, r) => {
      hiddenScene(g, s, { normalHandle: !!s.normalHandle });
    });
    const st = Math.min(3, Math.max(0, s.stage)), k = Math.round(st);
    const kk = Math.min(3, Math.floor(st + 0.5)), stName = kk === 2 ? (s.act === 'relu' ? 'fold: ReLU(W₁x + b₁)' : s.act === 'none' ? 'no fold: still W₁x + b₁' : STAGES[2]) : STAGES[kk];
    g.hud('stage: ' + stName, 12, 22, { size: 14, color: 'hud' });
    if (st < 0.5) g.hud('input plane drawn at h₃ = 0', 12, 42, { size: 12, color: 'muted' });
    const paNow = s.plane === null || s.plane === undefined ? seg(st, 1.8, 2) : s.plane; if (s.showAcc || paNow > 0.5) { const a = accuracy(net, s.act, s.frozenUnit, s.freezeAmt); const shown = s.accFade === undefined ? a : lerp(0.5, a, s.accFade); g.hud('accuracy ' + pct(shown), 12, st < 0.5 ? 62 : 42, { size: 14, color: shown >= 0.95 ? 'ok' : 'gold' }); }
    if (s.inset) actInset(g, s.act === 'relu' ? 'relu' : 'tanh', { x: 0.02, y: wid ? 0.62 : 0.76, w: 0.24, h: 0.2 });
    if (s.rowsInset) {
      g.viewport({ x: 0.68, y: 0.03, w: 0.3, h: 0.3 }, { view: '2d', cam: fit(g, 0.3, 0.3, -2.4, 2.4, -2.4, 2.4), bg: 'panel' }, (g, r) => {
        g.points(CIRC.X, { r: 2, colors: i => cls(CIRC.y[i]), alpha: 0.6 });
        net.W1.forEach((w, k2) => { const fr = s.frozenUnit === k2 && s.freezeAmt > 0.5; g.vector([0, 0], N.scale(w, 0.85), { color: ['i', 'j', 'k'][k2], label: 'r' + ['₁', '₂', '₃'][k2], width: 2.4, alpha: fr ? 0.3 : 1 }); });
        g.hud('rows r₁, r₂, r₃ of W₁ in the input plane', r.x + 6, r.y + 12, { size: 11 });
      });
    }
    if (wid === 'stage') stageSlider(g, s);
    else if (wid === 'act') actSlider(g, s);
    else if (wid === 'freeze') freezeSlider(g, s);
    else if (wid === 'b2') b2Slider(g, s);
  }
  function sceneBoundary(g, s, ctx) {
    boundary2d(g, s, { heat: s.heat, contourAlpha: s.contourAlpha });
    g.hud('back in the input plane', 12, 22, { size: 14 });
    g.hud('accuracy ' + pct(accuracy(netOf(s), s.act)), 12, 44, { size: 14, color: 'ok' });
    if (s.heat > 0.3) g.hud('shading: sign of the score (blue +, rose −)', 12, ctx.size.h - 16, { size: 12, color: 'muted' });
  }
  function sceneLearn(g, s, ctx) {
    const k = Math.round(s.trainIdx), net = histNet(k);
    g.split((g, r) => {
      hiddenScene(g, s, { net, act: 'tanh', stage: 2, frozen: null, freezeAmt: 0, planeAlpha: 1, res: 18 });
      g.hud('hidden space after the fold', r.x + 8, r.y + 16, { size: 12 });
    }, (g, r) => {
      boundary2d(g, s, { net, act: 'tanh', frozen: null, freezeAmt: 0, res: 48, r: 3 });
      g.hud('step ' + HIST[k][0] + ' · accuracy ' + pct(accuracy(net, 'tanh')), r.x + 8, r.y + 16, { size: 13, color: 'gold' });
      g.hud('replay · seed 12 · 1200 steps · lr 0.5', r.x + 8, r.y + 34, { size: 11, color: 'muted' });
      const w = Math.min(200, r.w - 40);
      g.lineChart([{ points: HIST.slice(0, k + 1).map(hh => [hh[0], hh[14]]), color: 'teal' }], { x: r.x + 28, y: r.y + r.h - 96, w, h: 70, xRange: [0, 1200], yRange: [0, 1.8], title: 'loss', xLabel: 'step' });
    }, { ratio: 0.5, left: { view: '3d', cam: hcam(g, s, NETS.tanh, 'tanh', 2, 0.5, 1) }, right: { view: '2d', cam: fit(g, 0.5, 1, -2.2, 2.2, -2.6, 2.2) } });
  }
  function sceneSpirals(g, s, ctx) {
    const c = s.cam;
    g.split((g, r) => {
      SPCA.lines.forEach(l => g.polyline(l, { color: 'grid', width: 0.9, alpha: 0.5 }));
      g.points(SPCA.Z, { r: 3, colors: i => cls(SPIR.y[i]) });
      // no gold plane here: the cut's normal lies almost entirely in the 5 hidden directions, so in this shadow the classes overlap
      g.hud('PCA projection of 8-D hidden space · ' + pct(SPCA.kept) + ' variance', r.x + 8, r.y + 16, { size: 12, color: 'gold' });
      g.hud('a shadow: 3 of 8 directions · the classes overlap here,', r.x + 8, r.y + 34, { size: 11, color: 'muted' });
      g.hud('because only ' + pct(SPCA.nKept) + ' of the cut' + String.fromCharCode(8217) + 's direction lies in this shadow', r.x + 8, r.y + 50, { size: 11, color: 'muted' });
      g.hud('The picture stops at 3. The formula does not.', r.x + 8, r.y + r.h - 14, { size: 12, color: 'hud' });
    }, (g, r) => {
      const V = spView();
      g.heat2d(V.heat, V.BH, { range: [-1, 1], ramp: t => (t < 0.5 ? C.mix(C.dataB, C.bg, t * 2) : C.mix(C.bg, C.data, (t - 0.5) * 2)), alpha: 0.28 });
      g.contours(V.look, [0], V.B, { res: 80, color: 'gold', width: 2.4 });
      g.points(SPIR.X, { r: 3, colors: i => cls(SPIR.y[i]) });
      g.hud('input plane · accuracy ' + pct(SPCA.acc), r.x + 8, r.y + 16, { size: 12, color: 'ok' });
      g.hud('synthetic spirals · 2 → 8 → 8 → 1', r.x + 8, r.y + 34, { size: 11, color: 'muted' });
    }, { ratio: 0.5, left: { view: '3d', cam: camFit3(g, 0.5, 1, SPCA.R * 0.85, [0, 0, 0], c) }, right: { view: '2d', cam: fit(g, 0.5, 1, -2.6, 2.6, -2.7, 2.7) } });
  }
  function sceneRelu(g, s, ctx) {
    g.split((g, r) => {
      hiddenScene(g, s, { net: NETS.relu, act: 'relu', stage: s.stage, frozen: null, freezeAmt: 0, res: 24 });
      g.hud('ReLU: the sheet is creased, not bent', r.x + 8, r.y + 16, { size: 12 });
    }, (g, r) => {
      boundary2d(g, s, { net: NETS.relu, act: 'relu', frozen: null, freezeAmt: 0, res: 72, contourAlpha: s.contourAlpha });
      g.hud('boundary: straight pieces · accuracy ' + pct(accuracy(NETS.relu, 'relu')), r.x + 8, r.y + 16, { size: 12, color: 'ok' });
    }, { ratio: 0.55, left: { view: '3d', cam: hcam(g, s, NETS.relu, 'relu', s.stage, 0.55, 1) }, right: { view: '2d', cam: fit(g, 0.45, 1, -2.2, 2.2, -2.4, 2.2) } });
    actInset(g, 'relu', { x: 0.02, y: 0.76, w: 0.22, h: 0.2 });
  }
  function sceneSizes(g, s, ctx) {
    const top = 0.56;
    g.viewport({ x: 0, y: 0, w: 1, h: top }, { view: '2d', cam: fit(g, 1, top, -4.4, 4.4, -1.6, 1.5), border: false }, (g, r) => {
      const dig = s.digit, n = 8, px = 0.13;
      g.image(dig, { x0: -4.3, x1: -4.3 + n * px, y0: -0.52, y1: -0.52 + n * px }, { cells: false });
      g.label('synthetic digit', [-4.3 + n * px / 2, -0.75], { color: 'muted', size: 11, font: 'ui' });
      const tiles = [['784', -2.6], ['128', -0.9], ['128', 0.8], ['10', 2.3]];
      tiles.forEach(([t, x], k) => {
        const lit = s.flow * 4 > k;
        g.polygon([[x - 0.42, -0.35], [x + 0.42, -0.35], [x + 0.42, 0.35], [x - 0.42, 0.35]], { fill: lit ? 'panel2' : 'panel', alpha: 0.95, stroke: lit ? 'gold' : 'line', width: 1.6 });
        g.label(t, [x, 0], { color: lit ? 'text' : 'muted', size: 16, font: 'ui', bold: true, bg: false });
      });
      for (let k = 0; k < 3; k++) {
        const x0 = tiles[k][1] + 0.45, x1 = tiles[k + 1][1] - 0.45, xm = (x0 + x1) / 2;
        g.vector([x0, 0], [x1, 0], { color: 'v', label: k < 2 ? (k ? 'W₂h + b₂, fold' : 'W₁x + b₁, fold') : 'W₃', width: 2, head: 9, labelAt: 0.5, dy: -20, labelSize: 11 });
        if (k < 2) { const pts = []; for (let q = 0; q <= 12; q++) { const u = -1 + 2 * q / 12; pts.push([xm - 0.2 + 0.4 * (q / 12), -0.55 + 0.14 * Math.tanh(3 * u)]); } g.polyline(pts, { color: 'gold', width: 1.8 }); }
      }
      const logits = [0.3, 0.2, 0.4, 2.9, 0.1, 0.8, 0.2, 0.5, 0.6, 0.3].map(v => v * seg(s.flow, 0.75, 1));
      const sm = N.softmax(logits);
      g.bars(sm.map(v => v + 0.001), { origin: [3.0, -0.6], width: 0.1, gap: 0.03, height: 1.4, max: 1, colors: sm.map((v, k) => (k === 3 ? 'gold' : 'muted')) });
      g.label('10 scores (synthetic)', [3.62, -0.85], { color: 'muted', size: 11, font: 'ui' });
      g.label('depth = composition · width = dimension of the hidden room', [0, 1.15], { color: 'text', size: 13, font: 'ui' });
    });
    g.viewport({ x: 0.2, y: top, w: 0.6, h: 1 - top - 0.02 }, { view: '3d', cam: camFit3(g, 0.6, 1 - top - 0.02, 1.7, [0.1, 0.1, 0.05], Object.assign(hiddenBase(), { yaw: 0.7 })), bg: 'panel' }, (g, r) => {
      hiddenScene(g, s, { net: NETS.tanh, act: 'tanh', stage: 2, frozen: null, freezeAmt: 0, planeAlpha: 1, mini: true, res: 16, r: 2.6 });
      g.hud('same mechanism, width 3', r.x + 8, r.y + 16, { size: 12, color: 'gold' });
    });
    g.hud('Part 5: backprop · Part 6: watch a classifier learn', 12, ctx.size.h - 12, { size: 12, color: 'gold' });
  }
  function sceneKernel(g, s, ctx) {
    g.viewport({ x: 0, y: 0, w: 1, h: 1 }, { view: '3d', cam: camFit3(g, 1, 1, 2.3, [0, 0, 1.1], s.cam), border: false }, (g) => {
      g.paramSurface((r, a) => [r * Math.cos(a), r * Math.sin(a), r * r * s.liftK], { u0: 0, u1: 1.9, v0: 0, v1: 2 * Math.PI }, { res: 18, mode: 'wire', color: 'teal', wireAlpha: 0.3 });
      g.axes({ length: 2.2, names: ['x', 'y', 'x² + y²'] });
      const u = s.liftK;
      g.points(CIRC.X.map(x => [x[0], x[1], u * (x[0] * x[0] + x[1] * x[1])]), { r: 3.6, colors: i => cls(CIRC.y[i]) });
      if (s.planeK > 0) g.planes([{ point: [0, 0, 1], normal: [0, 0, 1], extent: 2, color: 'gold', alpha: 0.25 * s.planeK }]);
    });
    g.hud('a fixed lift φ(x) = [x; y; x² + y²]', 12, 22, { size: 14 });
    g.hud('Part 6: kernel trick', 12, 44, { size: 13, color: 'gold' });
  }
  function sceneRecap(g, s) {
    const h = 1 / 3, cam = Object.assign(hiddenBase(), { yaw: 0.7 }), camL = Object.assign(hiddenBase(), { yaw: -0.5 });
    g.viewport({ x: 0, y: 0, w: 1, h }, { view: '3d', cam: hcam(g, { cam: camL }, NETS.tanh, 'tanh', 1, 1, h, 0.95) }, (g, r) => {
      hiddenScene(g, s, { net: NETS.tanh, act: 'tanh', stage: 1, frozen: null, freezeAmt: 0, planeAlpha: 0, mini: true, res: 14, r: 2.4 });
      g.hud('1 · affine lift: a tilted sheet in the hidden room', r.x + 8, r.y + 16, { size: 12 });
    });
    g.viewport({ x: 0, y: h, w: 1, h }, { view: '3d', cam: hcam(g, { cam }, NETS.tanh, 'tanh', 2, 1, h, 0.9) }, (g, r) => {
      hiddenScene(g, s, { net: NETS.tanh, act: 'tanh', stage: 2, frozen: null, freezeAmt: 0, planeAlpha: 1, mini: true, res: 14, r: 2.4 });
      g.hud('2 · fold, then a flat gold cut', r.x + 8, r.y + 16, { size: 12 });
    });
    g.viewport({ x: 0, y: 2 * h, w: 1, h }, { view: '2d', cam: fit(g, 1, h, -4.2, 2.6, -2.1, 2.1) }, (g, r) => {
      boundary2d(g, s, { net: NETS.tanh, act: 'tanh', frozen: null, freezeAmt: 0, res: 48, r: 2.6 });
      g.label('W₂W₁ alone', [-3.1, 0.3], { color: 'bad', size: 13, font: 'ui' }); g.marker.cross([-3.1, -0.3], { size: 18 });
      g.hud('3 · the preimage of the cut: a closed curve', r.x + 8, r.y + 16, { size: 12 });
    });
  }
  function scenePlay(g, s, ctx) {
    if (s.dataset === 'spirals') { sceneSpirals(g, s, ctx); return; }
    const net = netOf(s);
    g.viewport({ x: 0, y: 0, w: 1, h: 1 }, { view: '3d', cam: hcam(g, s, net, s.act, s.stage, 1, 1), border: false }, () => hiddenScene(g, s, {}));
    g.viewport({ x: 0.66, y: 0.02, w: 0.32, h: 0.34 }, { view: '2d', cam: fit(g, 0.32, 0.34, -2.2, 2.2, -2.2, 2.2), bg: 'panel' }, (g, r) => {
      boundary2d(g, s, { res: 40, r: 2 });
      g.hud('input plane · ' + pct(accuracy(net, s.act, s.frozenUnit, s.freezeAmt)), r.x + 6, r.y + 12, { size: 11, color: 'gold' });
    });
    g.hud('stage: ' + STAGES[Math.min(3, Math.round(s.stage))] + (s.act === 'none' ? ' · no fold' : ''), 12, 22, { size: 14 });
    if (s.trained) g.hud('weights: retrained in your browser', 12, 42, { size: 12, color: 'ok' });
    if (s.curve && s.curve.length) g.lineChart([{ points: s.curve.map((l, k) => [k * 25, l]), color: 'teal' }], { x: 30, y: ctx.size.h - 100, w: 170, h: 64, xRange: [0, 1200], yRange: [0, 1.8], title: 'loss of the retrain', xLabel: 'step' });
  }

  /* ---------- math panel ---------- */
  function mathPanel(s, tex) {
    const T = (x, o) => tex.text(typeof x === 'number' ? f2(x) : x, o);
    const rows = { colors: 'rows' };
    const p = [];
    const W1m = () => tex.mat(s.W1, rows), b1v = () => tex.vec(s.b1, { colors: 'rows' });
    switch (s.scene) {
      case 'line': p.push(tex.eq(T('best possible line: accuracy ' + pct(BEST.acc))), tex.eq(T('no line separates a ring from its surrounding ring', { color: 'muted' }))); break;
      case 'stack': p.push(tex.eq(tex.sym('W', { sub: '3' }), tex.sym('W', { sub: '2' }), tex.sym('W', { sub: '1' }), tex.sym('x', { color: 'v' }), tex.op('='), tex.op('('), tex.sym('W', { sub: '3' }), tex.sym('W', { sub: '2' }), tex.sym('W', { sub: '1' }), tex.op(')'), tex.sym('x', { color: 'v' })), tex.eq(T('= '), tex.mat(s.M.map(r => r.map(x => Math.round(x * 100) / 100)), { colors: 'cols' }), tex.sym('x', { color: 'v' }), T('   a straight cut: at most ' + pct(BEST.acc)))); break;
      case 'hidden': {
        const fold = s.act === 'relu' ? 'max(0, ' : s.act === 'none' ? '(' : 'tanh(';
        if (s.mathMode === 'lift') p.push(tex.eq(tex.sym('h', { sub: 'pre' }), tex.op('='), tex.sym('W', { sub: '1' }), tex.sym('x', { color: 'v' }), tex.op('+'), tex.sym('b', { sub: '1', color: 'b' })), tex.eq(tex.sym('W', { sub: '1' }), tex.op('='), W1m(), tex.dim(3, 2), tex.sym('b', { sub: '1', color: 'b' }), tex.op('='), b1v()));
        else if (s.mathMode === 'fold') p.push(tex.eq(tex.sym('h'), tex.op('='), T(fold), tex.sym('W', { sub: '1' }), tex.sym('x', { color: 'v' }), tex.op('+'), tex.sym('b', { sub: '1', color: 'b' }), T(')')), tex.eq(T('each coordinate squashed into (−1, 1) separately', { color: 'muted' })));
        else if (s.mathMode === 'units') p.push(tex.eq(tex.sym('h', { sub: 'i' }), tex.op('='), T('tanh('), tex.sym('r', { sub: 'i' }), tex.op('·'), tex.sym('x', { color: 'v' }), tex.op('+'), tex.sym('b', { sub: '1,i', color: 'b' }), T(')'), T('  i = 1, 2, 3')), tex.eq(T('rᵢ = row i of W₁ · b₁,ᵢ = entry i of b₁', { color: 'muted' })), tex.eq(T(s.frozenUnit === null || s.frozenUnit === undefined || s.freezeAmt < 0.5 ? 'no unit frozen' : 'h' + '₁₂₃'[s.frozenUnit] + ' frozen at 0', { color: 'muted' })));
        else if (s.mathMode === 'cut' || s.mathMode === 'check2') {
          const terms = s.w2.map((w, k) => (k ? (w < 0 ? ' − ' : ' + ') : (w < 0 ? '−' : '')) + f2(Math.abs(w)) + 'h' + '₁₂₃'[k]).join('');
          p.push(tex.eq(T('score'), tex.op('='), tex.sym('w', { sub: '2', color: 'gold' }), tex.op('·'), tex.sym('h'), tex.op('+'), tex.sym('b', { sub: '2' }), tex.op('='), T(terms + (s.b2 < 0 ? ' − ' : ' + ') + f2(Math.abs(s.b2)))));
          p.push(tex.eq(T('accuracy ' + pct(accuracy(netOf(s), s.act)), { color: 'gold' })));
        } else {
          const k = Math.min(3, Math.round(s.stage));
          const forms = ['x', 'W₁x + b₁', fold + 'W₁x + b₁)', 'w₂·h + b₂'];
          p.push(tex.eq(T('stage ' + f2(Math.round(s.stage * 100) / 100) + ': ' + STAGES[k].split(':')[0]), tex.op('  '), T(forms[k])));
          if (s.mathMode === 'check1') p.push(tex.eq(T('activation: ' + (s.act === 'relu' ? 'ReLU' : s.act) + ' · accuracy ' + pct(accuracy(netOf(s), s.act)), { color: 'gold' })));
        }
        break; }
      case 'boundary': p.push(tex.eq(T('boundary'), tex.op('='), T('{ x : w₂·tanh(W₁x + b₁) + b₂ = 0 }', { color: 'gold' })), tex.eq(T('accuracy ' + pct(accuracy(netOf(s), s.act))))); break;
      case 'learn': { const k = Math.round(s.trainIdx), net = histNet(k); p.push(tex.eq(T('step ' + HIST[k][0]), tex.op('·'), T('loss ' + LA.fmt.fixed(HIST[k][14], 3)), tex.op('·'), T('accuracy ' + pct(accuracy(net, 'tanh')), { color: 'gold' }))); p.push(tex.eq(tex.sym('W', { sub: '1' }), tex.op('='), tex.mat(net.W1, rows))); break; }
      case 'spirals': p.push(tex.eq(tex.sym('h', { sub: '2' }), tex.op('='), T('tanh(W₂ tanh(W₁x + b₁) + b₂)'), tex.op('∈'), T('ℝ⁸')), tex.eq(T('shown: 3 of 8 directions · ' + pct(SPCA.kept) + ' of the variance', { color: 'gold' }))); break;
      case 'relu': p.push(tex.eq(tex.sym('h'), tex.op('='), T('max(0, '), tex.sym('W', { sub: '1' }), tex.sym('x', { color: 'v' }), tex.op('+'), tex.sym('b', { sub: '1', color: 'b' }), T(')')), tex.eq(tex.sym('W', { sub: '1' }), tex.op('='), tex.mat(NETS.relu.W1, rows), tex.sym('w', { sub: '2', color: 'gold' }), tex.op('='), tex.vec(NETS.relu.w2, { color: 'gold' }), T('  b₂ = 1.45'))); break;
      case 'sizes': p.push(tex.eq(tex.sym('ŷ'), tex.op('='), T('W₃ f(W₂ f(W₁x + b₁) + b₂) + b₃')), tex.eq(T('784 → 128 → 128 → 10', { color: 'muted' }))); break;
      case 'kernel': p.push(tex.eq(tex.sym('φ'), tex.op('('), tex.sym('x', { color: 'v' }), tex.op(')'), tex.op('='), tex.vec(['x', 'y', 'x² + y²']))); break;
      case 'play': if (s.dataset === 'spirals') p.push(tex.eq(T('2 → 8 → 8 → 1 · PCA-3 shadow keeps ' + pct(SPCA.kept) + ' of the variance'))); else { p.push(tex.eq(tex.sym('W', { sub: '1' }), tex.op('='), W1m(), tex.sym('b', { sub: '1', color: 'b' }), tex.op('='), b1v())); p.push(tex.eq(tex.sym('w', { sub: '2', color: 'gold' }), tex.op('='), tex.vec(s.w2, { color: 'gold' }), T('  b₂ = ' + f2(s.b2) + ' · accuracy ' + pct(accuracy(netOf(s), s.act, s.frozenUnit, s.freezeAmt))))); }
        break;
      default: return '';
    }
    return tex.block.apply(null, p);
  }

  /* shared step setup */
  function hiddenSetup(s, o) {
    s.scene = 'hidden'; s.view = '3d'; s.widget = null; s.normalHandle = false; s.inset = false; s.rowsInset = false; s.showAcc = false; s.accFade = undefined;
    s.frozenUnit = null; s.freezeAmt = 0; s.mathMode = ''; s.plane = 0;
    if (!s.cam || s.cam.dist === undefined || s._prevView !== '3d') s.cam = hiddenBase();
    s._prevView = '3d';
    Object.assign(s, o || {});
  }
  function flatSetup(s, scene) { s.scene = scene; s.view = '2d'; s.cam = cam2(); s._prevView = '2d'; s.widget = null; s.cross = false; }

  LA.registerLesson({
    id: 'p2-12-neural-layer',
    part: 2,
    title: 'A neural layer: transformation plus a fold',
    subtitle: 'Lift the plane into a 3D hidden space, bend it with tanh, cut it flat.',
    minutes: 14,
    prereqs: ['p2-09-non-square', 'p2-11-affine-bias', 'p1-09-first-model-neuron', 'p2-04-composition'],
    tags: ['capstone'],
    badge: { drawnIn: '2D input, an exactly 3D hidden space (width 3), 2D output; the spirals’ 8-D hidden space as a labeled PCA-3 projection', realSize: 'hidden widths 256–4096 and dozens of layers; W₁ is 128×784 in a digit classifier', transfers: '“affine lift, elementwise fold, flat cut” is the exact mechanism at every width and depth; only the hidden space stops being drawable above 3' },
    cantShow: ['the 8-dimensional hidden space of the spirals network (we show a PCA-3 shadow and print the variance kept)', 'that in very high dimensions almost every labeled point set is already linearly separable (Cover’s result); the lift is still the mechanism, but “the rings need a lift” does not mean everything does'],
    terms: [
      { term: 'hidden layer', def: 'The intermediate vector h = fold(W₁x + b₁); its size is the width.' },
      { term: 'width', def: 'The number of hidden units: the dimension of the hidden space.' },
      { term: 'activation', def: 'A fixed nonlinear function (the fold) applied to each coordinate separately, such as tanh or ReLU.' },
      { term: 'tanh', def: 'A smooth function that squashes any number into the interval from −1 to 1.' },
      { term: 'ReLU', def: 'Replaces negative numbers by 0 and leaves positive ones alone: a fold along each axis.' },
      { term: 'decision boundary', def: 'The input points whose output score is exactly 0: the preimage of the gold plane.' },
      { term: 'preimage', def: 'All inputs that a map sends into a given set of outputs.' },
      { term: 'depth', def: 'The number of layers composed one after another.' },
    ],
    atlas: ['mlp', 'backprop', 'kernels'],
    thumbnail: 5,

    state() {
      const T = NETS.tanh;
      return {
        view: '2d', cam: cam2(), scene: 'line', _prevView: '2d',
        dataset: 'circles', act: 'tanh', W1: LA.clone(T.W1), b1: LA.clone(T.b1), w2: LA.clone(T.w2), b2: T.b2,
        stage: 0, frozenUnit: null, freezeAmt: 0, widget: null, normalHandle: false, inset: false, rowsInset: false, showAcc: false,
        th: 0.3, off: 0, cross: false, M: I2(), phase: 0, heat: 0, contourAlpha: 1, trainIdx: 0, flow: 0, liftK: 0, planeK: 0,
        mathMode: '', trained: false, curve: [], plane: null,
        digit: (function () { const d = LA.rng.syntheticDigits(5, 4, { size: 8 })[3].pixels, g = []; for (let r = 7; r >= 0; r--) g.push(d.slice(r * 8, r * 8 + 8)); return g; })(),
      };
    },

    render(g, s, ctx) {
      switch (s.scene) {
        case 'line': sceneLine(g, s, ctx); break;
        case 'stack': sceneStack(g, s, ctx); break;
        case 'hidden': sceneHidden(g, s, ctx); break;
        case 'boundary': sceneBoundary(g, s, ctx); break;
        case 'learn': sceneLearn(g, s, ctx); break;
        case 'spirals': sceneSpirals(g, s, ctx); break;
        case 'relu': sceneRelu(g, s, ctx); break;
        case 'sizes': sceneSizes(g, s, ctx); break;
        case 'kernel': sceneKernel(g, s, ctx); break;
        case 'recap': sceneRecap(g, s); break;
        case 'play': scenePlay(g, s, ctx); break;
        default: sceneLine(g, s, ctx);
      }
    },

    math(s, tex) { return mathPanel(s, tex); },

    steps: [
      { title: 'A line cannot do it', duration: 4,
        narration: 'Two classes: a blue inner ring and a rose outer ring. A neuron from Part 1 draws a straight boundary; sweep the gold line anywhere you like: even the best line gets only about 76% right, because no line separates a ring from the ring around it. We need a map that bends space before the cut.',
        setup(s) { flatSetup(s, 'line'); s.th = 0.3; s.off = 0; s.atBest = false; },
        at(s, t) { const u = seg(t, 0, 0.65, 'linear'), w = seg(t, 0.68, 0.82); s.th = lerp(0.3 + 2.6 * u, BEST.th, w); s.off = lerp(0.9 * Math.sin(2 * Math.PI * 1.5 * u), BEST.off, w); s.atBest = w >= 1; s.cross = t >= 0.85; } },

      { title: 'Stacking linear layers does not help', duration: 5, misconception: true,
        narration: 'Belief: “stack several layers of W and the boundary will curve.” From p2-04, W₃W₂W₁ is a single matrix, and a straight boundary stays straight through it. Three grid morphs collapse into one, and the gold cut is still a line. Depth alone adds nothing; the fold is what matters.',
        setup(s) { flatSetup(s, 'stack'); s.cam = Object.assign(cam2(), { halfH: 3.0 }); s.M = I2(); s.phase = 0; s.cross = false; },
        at(s, t) {
          const W1 = [[1, 0.6], [0, 1]], W2 = N.rot2(40 * Math.PI / 180), W3 = [[1.3, 0], [0, 0.75]];
          const P3 = N.matMul(W3, N.matMul(W2, W1));
          if (t < 0.5) {
            const u1 = seg(t, 0.02, 0.16), u2 = seg(t, 0.18, 0.32), u3 = seg(t, 0.34, 0.48);
            s.M = N.matMul(LA.lerpM(I2(), W3, u3), N.matMul(N.rot2(40 * Math.PI / 180 * u2), LA.lerpM(I2(), W1, u1)));
            s.phase = t < 0.17 ? 0 : t < 0.33 ? 1 : 2;
          } else { s.M = LA.lerpM(I2(), P3, seg(t, 0.55, 0.8)); s.phase = 3; }
          s.cross = t >= 0.85;
        } },

      { title: 'Layer 1: lift the plane into 3D', duration: 6,
        narration: 'The first layer is affine with three outputs: W₁ is 3×2 and b₁ has three entries. From p2-09 and p2-11, that lifts the plane into 3D as a tilted sheet, and the 120 points ride along on it. This is the <b>hidden layer</b>, of <b>width</b> 3, so the hidden space is exactly three-dimensional. Orbit.',
        setup(s) { setAct(s, 'tanh'); hiddenSetup(s, { stage: 0, mathMode: 'lift' }); s.cam = hiddenBase(); },
        at(s, t) { s.stage = seg(t, 0.1, 0.7); LA.cam.tween(s, { yaw: -0.2 }, { yaw: -0.7 }, t); } },

      { title: 'The fold: tanh bends the sheet', duration: 6,
        narration: 'Now the <b>activation</b>: <b>tanh</b> applied to each of the three coordinates separately. Numbers far from zero are squashed toward ±1, so the flat sheet bends into a curved surface inside the cube [−1, 1]³. Watch the inner ring gather on one side of the bend while the outer ring is pushed toward the cube’s walls.',
        setup(s) { hiddenSetup(s, { stage: 1, inset: true, mathMode: 'fold' }); },
        at(s, t) { s.stage = 1 + seg(t, 0.1, 0.75); } },

      { title: 'Three neurons, three directions', duration: 5, interactive: true,
        narration: 'Each hidden unit is a neuron from p1-09: a weight row of W₁ with a bias, a dot product and a fold. Its weight row is one direction in the input plane and one axis of the hidden cube. Freeze a unit, setting its output to a constant, and the sheet flattens along that axis. Drag the selector to try.',
        setup(s) { hiddenSetup(s, { stage: 2, rowsInset: true, widget: 'freeze', mathMode: 'units' }); },
        at(s, t) { s.freezeAmt = seg(t, 0.3, 0.4) - seg(t, 0.55, 0.65); s.frozenUnit = s.freezeAmt > 0 ? 0 : null; } },

      { title: 'Layer 2: a flat cut in hidden space', duration: 5,
        narration: 'The output layer is a single neuron on the hidden vector: score = w₂·h + b₂, with w₂ = [1; 1; 1] and b₂ = −1.5. Its boundary is the gold plane h₁ + h₂ + h₃ = 1.5 in the hidden cube, flat as every neuron’s boundary is. It slices the bent sheet: inner points on one side, outer on the other. 100% correct.',
        setup(s) { hiddenSetup(s, { stage: 2, showAcc: true, accFade: 0, mathMode: 'cut' }); },
        at(s, t) { s.plane = seg(t, 0.05, 0.4); s.accFade = seg(t, 0.3, 0.8); s.stage = 2; } },

      { title: 'Back in the plane: a curved boundary', duration: 5,
        narration: 'Pull the cut back to the input plane. The points of the plane that land exactly on the gold plane form a closed curve around the inner ring: the <b>decision boundary</b>, the <b>preimage</b> of a flat cut. The network never drew a curve; it bent space until a flat cut did the work. The shading is the sign of the score.',
        setup(s) { flatSetup(s, 'boundary'); s.heat = 0; s.contourAlpha = 0; },
        at(s, t) { s.contourAlpha = seg(t, 0.05, 0.45); s.heat = seg(t, 0.55, 0.9); } },

      { title: 'Step through the layers', duration: 6, interactive: true,
        narration: 'Drag the stage slider through input plane, affine lift, fold and score, and watch which stage first makes a flat cut possible. The lift alone leaves the points on a flat sheet, still not separable by a plane through that sheet. The fold is the moment the geometry changes.',
        setup(s) { hiddenSetup(s, { stage: 0, widget: 'stage', mathMode: 'stage', plane: null }); },
        at(s, t) { s.stage = t < 0.7 ? 3 * seg(t, 0.05, 0.65, 'inOut') : 3 - seg(t, 0.75, 0.95); } },

      { title: 'Watch it learn', duration: 6,
        narration: 'These weights came from plain gradient descent: 1200 steps from random numbers, seed 12. This replays that run; Retrain in Play reruns it in your browser in milliseconds. Early on the cut is wrong; the sheet folds, tilts and settles until the boundary closes around the inner ring and the loss curve flattens.',
        setup(s) { hiddenSetup(s, { stage: 2, plane: null }); s.scene = 'learn'; s.trainIdx = 0; },
        at(s, t) { s.trainIdx = Math.round(48 * seg(t, 0.03, 0.97, 'linear')); } },

      { title: 'Spirals: width 8, a labeled shadow', duration: 6,
        narration: 'Two spirals need more room: a network 2 → 8 → 8 → 1. Its hidden space has eight dimensions, which cannot be drawn. Left, its PCA-3 shadow, labeled with the variance kept; the classes overlap there, because the cut uses the dropped directions. Right, honestly, the input plane, where the gold boundary winds between the arms. The picture stops at 3. The formula does not.',
        setup(s) { hiddenSetup(s, {}); s.scene = 'spirals'; },
        at(s, t) { LA.cam.tween(s, { yaw: SPCA.yaw - 0.3, pitch: SPCA.pitch }, { yaw: SPCA.yaw + 0.3, pitch: SPCA.pitch }, t); } },

      { title: 'ReLU folds along axes', duration: 4,
        narration: 'Swap tanh for <b>ReLU</b>, which sets negatives to zero. The sheet is no longer smoothly bent; it is creased flat along each hidden axis, and the cut’s preimage is a polygon of straight pieces. Different fold, same mechanism. Any fixed nonlinearity applied coordinate by coordinate turns the lift into a bend.',
        setup(s) { hiddenSetup(s, { stage: 1, plane: null }); s.scene = 'relu'; s.contourAlpha = 0; },
        at(s, t) { s.stage = 1 + seg(t, 0.05, 0.6); s.contourAlpha = seg(t, 0.6, 0.85); } },

      { title: 'Real sizes: depth is composition, width is dimension', duration: 5, mlTie: true,
        narration: 'A digit network lifts 784 pixels to 128 hidden directions, folds, lifts again, and cuts ten ways. Every layer is this lesson: an affine map from p2-01 and p2-11, a fold, and a flat cut at the end. <b>Depth</b> is composition from p2-04; width is the dimension of the hidden room. Part 5 trains it.',
        setup(s) { flatSetup(s, 'sizes'); s.flow = 0; },
        at(s, t) { s.flow = seg(t, 0.05, 0.9, 'linear'); } },

      { title: 'A learned lift versus a fixed one', duration: 4,
        narration: 'The network learned its lift. Part 6’s kernel trick fixes the lift in advance: for example, send [x; y] to [x; y; x² + y²] so the rings land on a paraboloid where a horizontal gold plane separates them. Learned or fixed, the story is the same: lift, then cut flat.',
        setup(s) { s.scene = 'kernel'; s.view = '3d'; s._prevView = '3d'; s.widget = null; s.cam = Object.assign(LA.cam.defaults3d(), { target: [0, 0, 1], dist: 8.5, yaw: 0.6, pitch: 0.32 }); s.liftK = 0; s.planeK = 0; },
        at(s, t) { s.liftK = seg(t, 0.1, 0.6); s.planeK = seg(t, 0.65, 0.85); } },

      { title: 'Recap', duration: 3, recap: true,
        narration: 'A layer is an affine map plus a fold. The affine part lifts the input into a hidden space whose dimension is the width; the fold bends the lifted sheet; a final neuron cuts it with a flat plane, and the preimage of that cut is a curved boundary in the input. Linear layers alone stay flat. Beyond width 3 we show labeled shadows.',
        cards: ['<b>Affine lift:</b> W₁x + b₁ raises the plane into the hidden room as a tilted sheet.', '<b>Fold, then cut:</b> tanh bends the sheet; a flat gold plane separates the classes.', '<b>Preimage of the cut = curved boundary.</b> Linear layers alone (W₂W₁) stay flat.'],
        setup(s) { s.scene = 'recap'; s.view = '2d'; s._prevView = '2d'; s.cam = cam2(); s.widget = null; },
        at(s, t) { } },
    ],

    playground: {
      intro: '<p>Pick a dataset and an activation, slide through the layer stages, freeze a hidden unit, or retrain the width-3 network in your browser (deterministic, seed 12). Orbit the hidden space; the inset shows the boundary in the input plane.</p>',
      setup(s) { setAct(s, 'tanh'); s.scene = 'play'; s.view = '3d'; s._prevView = '3d'; s.cam = hiddenBase(); s.dataset = 'circles'; s.stage = 2; s.frozenUnit = null; s.freezeAmt = 0; s.widget = null; s.trained = false; s.curve = []; s.plane = null; },
      controls: [
        { type: 'preset', label: 'Dataset', items: [
          { label: 'Circles (width 3)', apply(s) { s.dataset = 'circles'; } },
          { label: 'Spirals (width 8)', apply(s) { s.dataset = 'spirals'; } },
        ] },
        { type: 'preset', label: 'Activation', items: [
          { label: 'tanh', apply(s) { setAct(s, 'tanh'); s.trained = false; s.curve = []; } },
          { label: 'ReLU', apply(s) { setAct(s, 'relu'); s.trained = false; s.curve = []; } },
          { label: 'none (fails)', apply(s) { setAct(s, 'none'); s.trained = false; s.curve = []; } },
        ] },
        { type: 'slider', bind: 'stage', min: 0, max: 3, step: 0.01, label: 'Layer stage', color: 'gold', fmt: x => ['input', 'affine', 'fold', 'score'][Math.min(3, Math.round(x))] + ' ' + LA.fmt(x) },
        { type: 'preset', label: 'Freeze a hidden unit', items: [
          { label: 'none', apply(s) { s.frozenUnit = null; s.freezeAmt = 0; } },
          { label: 'h₁', apply(s) { s.frozenUnit = 0; s.freezeAmt = 1; } },
          { label: 'h₂', apply(s) { s.frozenUnit = 1; s.freezeAmt = 1; } },
          { label: 'h₃', apply(s) { s.frozenUnit = 2; s.freezeAmt = 1; } },
        ] },
        { type: 'button', label: 'Retrain in browser (1200 steps)', primary: true, onClick(s) { const r = trainCircles(1200, 0.5, 12); s.act = 'tanh'; s.dataset = 'circles'; s.W1 = r.net.W1; s.b1 = r.net.b1; s.w2 = r.net.w2; s.b2 = r.net.b2; s.curve = r.curve; s.trained = true; } },
      ],
    },

    checks: [
      { kind: 'manipulate',
        prompt: 'Drag the activation selector to a fold that lets the gold plane separate the two rings (accuracy ≥ 95%).',
        setup(s) { setAct(s, 'none'); hiddenSetup(s, { stage: 2, widget: 'act', showAcc: true, mathMode: 'check1', plane: null }); s.cam = hiddenBase(); },
        judge(s) { const a = accuracy(netOf(s), s.act); return { ok: a >= 0.95, progress: s.act === 'none' ? 0.2 : a }; },
        hint: 'Without a fold the sheet stays flat, and a flat cut of a flat sheet is still a straight line.',
        solution(s) { setAct(s, 'tanh'); } },
      { kind: 'manipulate',
        prompt: 'Turn the gold plane (drag the tip of its normal w₂) and slide it (bottom slider) until it separates the classes: accuracy ≥ 95%.',
        setup(s) { setAct(s, 'tanh'); hiddenSetup(s, { stage: 2, widget: 'b2', normalHandle: true, showAcc: true, mathMode: 'check2', plane: null }); s.cam = hiddenBase(); s.w2 = [1, -1, 0.2]; s.b2 = 0; },
        judge(s) { const a = accuracy(netOf(s), s.act); return { ok: a >= 0.95, progress: Math.max(0, (a - 0.5) / 0.45) }; },
        hint: 'Point the normal toward the corner where the blue inner points gather (all three coordinates large), then slide the plane between the two groups.',
        solution(s) { s.w2 = [1, 1, 1]; s.b2 = -1.5; } },
      { kind: 'manipulate',
        prompt: 'Slide the layer stage to the first stage at which a flat cut can separate the two classes.',
        setup(s) { setAct(s, 'tanh'); hiddenSetup(s, { stage: 0, widget: 'stage', mathMode: 'stage', plane: null }); s.cam = hiddenBase(); },
        judge(s) { const d = Math.abs(s.stage - 2); return { ok: d < 0.15, progress: Math.max(0, 1 - d / 2) }; },
        hint: 'A flat sheet cut by a plane gives a straight line in the input; look for the stage where the sheet bends.',
        solution(s) { s.stage = 2; } },
    ],
  });
})();
