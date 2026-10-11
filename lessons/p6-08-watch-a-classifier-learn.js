/* p6-08 — Grand capstone: watch a classifier learn. A boundary turning in data space is a point sliding down a loss
   bowl in weight space; the bowl's shape (Hessian) comes from the data; a hidden layer is shown as a labeled PCA shadow. */
(function () {
  'use strict';
  const N = LA.num, seg = LA.seg, fmt = LA.fmt, lerp = LA.lerp, lerpV = LA.lerpV;
  const wS = '<span class="c-w">w</span>', gS = '<span class="c-v">∇L</span>';
  const LAM = 0.01, ETA = 1.5, W0 = [-1.5, 2.0], NSTEP = 30;
  const WB = { x0: -3, x1: 3, y0: -3, y1: 3 }, RES = 60;

  // ---- data: 80 points, two blobs (synthetic), centered so that b = 0 is reasonable ----
  const BL = LA.rng.blobs(608, 80, { centers: [[1.2, 0.8], [-1.0, -0.6]], spread: 0.7 });
  const MU = N.mean(BL.X);
  const XR = BL.X.map(p => N.sub(p, MU));
  const T = BL.y.map(c => (c === 0 ? 1 : 0));                         // 1 = the blue blob, 0 = the rose blob
  const SD = [0, 1].map(j => Math.sqrt(XR.reduce((a, p) => a + p[j] * p[j], 0) / XR.length));
  const DATA = { raw: XR, wide: XR.map(p => [3 * p[0], p[1]]), std: XR.map(p => [p[0] / SD[0], p[1] / SD[1]]) };
  const sig = N.sigmoid;
  function loss(w, X) { let s = 0; for (let i = 0; i < X.length; i++) { const p = sig(w[0] * X[i][0] + w[1] * X[i][1]); s -= T[i] ? Math.log(Math.max(p, 1e-12)) : Math.log(Math.max(1 - p, 1e-12)); } return s / X.length + LAM / 2 * (w[0] * w[0] + w[1] * w[1]); }
  function grad(w, X) { const g = [LAM * w[0], LAM * w[1]]; for (let i = 0; i < X.length; i++) { const p = sig(w[0] * X[i][0] + w[1] * X[i][1]); g[0] += (p - T[i]) * X[i][0] / X.length; g[1] += (p - T[i]) * X[i][1] / X.length; } return g; }
  function hess(w, X) { const h = [[LAM, 0], [0, LAM]]; X.forEach(x => { const p = sig(w[0] * x[0] + w[1] * x[1]), c = p * (1 - p) / X.length; h[0][0] += c * x[0] * x[0]; h[0][1] += c * x[0] * x[1]; h[1][1] += c * x[1] * x[1]; }); h[1][0] = h[0][1]; return h; }
  function acc(w, X) { let ok = 0; X.forEach((x, i) => { if ((w[0] * x[0] + w[1] * x[1] > 0 ? 1 : 0) === T[i]) ok++; }); return ok / X.length; }
  const GD_MEMO = { key: null, X: null, P: null };
  function gdPath(w0, eta, n, X) { const mk = w0[0] + ',' + w0[1] + ',' + eta + ',' + n; if (GD_MEMO.key === mk && GD_MEMO.X === X) return GD_MEMO.P; const P = gdRun(w0, eta, n, X); GD_MEMO.key = mk; GD_MEMO.X = X; GD_MEMO.P = P; return P; }
  function gdRun(w0, eta, n, X) { const P = [w0.slice()]; let w = w0.slice(); for (let k = 0; k < n; k++) { const g = grad(w, X); w = [w[0] - eta * g[0], w[1] - eta * g[1]]; P.push(w); } return P; }
  const INFO = {};
  function info(key) {
    if (INFO[key]) return INFO[key];
    const X = DATA[key]; let w = [0, 0];
    for (let k = 0; k < 20; k++) w = N.sub(w, N.solve(hess(w, X), grad(w, X)));          // Newton, 20 iterations
    const H = hess(w, X), e = N.eigSym(H);
    const grid = [];
    for (let i = 0; i < RES; i++) { const row = []; const y = WB.y1 - (WB.y1 - WB.y0) * (i + 0.5) / RES; for (let j = 0; j < RES; j++) { const x = WB.x0 + (WB.x1 - WB.x0) * (j + 0.5) / RES; const v = loss([x, y], X); row.push(v); } grid.push(row); }
    const Lmin = loss(w, X), levels = []; for (let k = 0; k < 8; k++) levels.push(Lmin + 0.012 * Math.pow(2, k));   // nested around the minimum
    const pc = N.eigSym(N.cov(X));
    INFO[key] = { ws: w, Ls: loss(w, X), H, lmax: e.values[0], lmin: e.values[1], qmax: e.vectors[0], qmin: e.vectors[1], grid, levels, pc };
    return INFO[key];
  }
  const lookup = key => p => { const G = info(key).grid; const j = LA.clamp(Math.round((p[0] - WB.x0) / (WB.x1 - WB.x0) * RES - 0.5), 0, RES - 1), i = LA.clamp(Math.round((WB.y1 - p[1]) / (WB.y1 - WB.y0) * RES - 0.5), 0, RES - 1); return G[i][j]; };
  const ACC0 = acc(W0, XR), PATH_R = gdPath(W0, ETA, NSTEP, XR), ACC30 = acc(PATH_R[NSTEP], XR);
  const angDeg = (a, b) => Math.acos(Math.min(1, Math.abs(N.dot(N.normalize(a), N.normalize(b))))) * 180 / Math.PI;
  const ANG_RAW = angDeg(info('raw').qmax, info('raw').pc.vectors[0]);
  const ETA_MAX = 2 / info('raw').lmax;
  const KAP = { raw: info('raw').lmax / info('raw').lmin, wide: info('wide').lmax / info('wide').lmin, std: info('std').lmax / info('std').lmin };
  const PUSH = XR.map((x, i) => N.norm(N.scale(x, sig(N.dot(W0, x)) - T[i])));
  const ARGPUSH = PUSH.indexOf(Math.max.apply(null, PUSH));

  // ---- the width-4 tanh network on moons (hard-coded snapshots of the deterministic procedure below) ----
  const MX = LA.rng.moons(609, 120, { noise: 0.1 }), XM = MX.X, YM = MX.y;
  const SNAP_STEPS = [0, 100, 300, 1000, 2000];
  // init LA.rng(609) uniform(−0.5, 0.5) in the order W1 (row-major), b1, w2, b2; full-batch GD on mean log-loss, η = 0.5, 2000 steps; weights rounded to 4 decimals
  const NET0 = [{ W1: [[-0.462, -0.4316], [0.4916, -0.1086], [-0.1347, -0.452], [-0.2245, -0.1031]], b1: [-0.2483, -0.0218, -0.0187, 0.0495], w2: [-0.0447, 0.4067, -0.3029, 0.2713], b2: -0.4267 }, { W1: [[-0.9493, -0.8245], [0.2702, -1.4025], [-0.2817, -0.2935], [0.2451, -1.1842]], b1: [-0.104, 0.182, -0.0731, 0.1046], w2: [0.8564, 2.3972, 0.114, 1.7218], b2: -0.1093 }, { W1: [[-2.4052, -1.2242], [0.3834, -0.9666], [-0.3318, -0.2807], [0.3772, -0.867]], b1: [-0.0505, 0.3805, -0.071, -0.2006], w2: [2.5349, 3.3132, -0.2186, 2.6679], b2: -0.268 }, { W1: [[-2.9584, -2.0948], [1.4074, -1.3977], [-0.3009, 0.485], [1.538, -1.2893]], b1: [0.0194, 2.2948, -0.0144, -2.3724], w2: [4.965, 5.0755, -1.0655, 4.6874], b2: -0.2351 }, { W1: [[-3.0529, -2.3702], [1.7491, -1.5136], [-0.2487, 0.7761], [1.8785, -1.4027]], b1: [0.0103, 2.8295, -0.0347, -2.9073], w2: [5.7794, 5.7379, -1.4084, 5.4475], b2: -0.1959 }];
  function fwd(n, x) { const h = n.W1.map((r, k) => Math.tanh(r[0] * x[0] + r[1] * x[1] + n.b1[k])); return { h, p: sig(N.dot(n.w2, h) + n.b2) }; }
  function netAcc(n) { let ok = 0; XM.forEach((x, i) => { if ((fwd(n, x).p > 0.5 ? 1 : 0) === YM[i]) ok++; }); return ok / XM.length; }
  const lossStr = v => (v < 0.005 ? fmt.fixed(v, 3) : fmt(v));   // the trained loss is 0.0035: never print it as 0
  function netLoss(n) { let s = 0; XM.forEach((x, i) => { const p = fwd(n, x).p; s -= YM[i] ? Math.log(Math.max(p, 1e-12)) : Math.log(Math.max(1 - p, 1e-12)); }); return s / XM.length; }
  function train() {
    const r = LA.rng(609), u = () => r() - 0.5;
    const n = { W1: [[u(), u()], [u(), u()], [u(), u()], [u(), u()]], b1: [u(), u(), u(), u()], w2: [u(), u(), u(), u()], b2: u() };
    const rnd = m => ({ W1: m.W1.map(row => row.map(v => +v.toFixed(4))), b1: m.b1.map(v => +v.toFixed(4)), w2: m.w2.map(v => +v.toFixed(4)), b2: +m.b2.toFixed(4) });
    const snaps = [], M = XM.length;
    for (let k = 0; k <= 2000; k++) {
      if (SNAP_STEPS.indexOf(k) >= 0) snaps.push(rnd(n));
      if (k === 2000) break;
      const gW = [[0, 0], [0, 0], [0, 0], [0, 0]], gb = [0, 0, 0, 0], gw = [0, 0, 0, 0]; let gb2 = 0;
      for (let i = 0; i < M; i++) { const x = XM[i], f = fwd(n, x), d2 = f.p - YM[i]; gb2 += d2 / M; for (let q = 0; q < 4; q++) { gw[q] += d2 * f.h[q] / M; const d1 = d2 * n.w2[q] * (1 - f.h[q] * f.h[q]); gW[q][0] += d1 * x[0] / M; gW[q][1] += d1 * x[1] / M; gb[q] += d1 / M; } }
      for (let q = 0; q < 4; q++) { n.W1[q][0] -= 0.5 * gW[q][0]; n.W1[q][1] -= 0.5 * gW[q][1]; n.b1[q] -= 0.5 * gb[q]; n.w2[q] -= 0.5 * gw[q]; } n.b2 -= 0.5 * gb2;
    }
    return snaps;
  }
  const flatNet = n => [].concat(n.W1[0], n.W1[1], n.W1[2], n.W1[3], n.b1, n.w2, [n.b2]);
  const unflat = v => ({ W1: [[v[0], v[1]], [v[2], v[3]], [v[4], v[5]], [v[6], v[7]]], b1: v.slice(8, 12), w2: v.slice(12, 16), b2: v[16] });
  function netAt(s, snap) { const S = s.net || NET0, k = Math.min(3, Math.floor(snap)), f = snap - k; return k >= 4 || f <= 0 ? S[Math.min(4, Math.round(snap))] : unflat(lerpV(flatNet(S[k]), flatNet(S[k + 1]), f)); }
  function shadow(n) {
    const H = XM.map(x => fwd(n, x).h);
    const P = N.pca(H, 3);
    const varKept = P.explained.slice(0, 3).reduce((a, b) => a + b, 0);
    const res = H.map((h, i) => { const hc = N.sub(h, P.mean); const back = [0, 0, 0, 0]; for (let c = 0; c < 3; c++) for (let q = 0; q < 4; q++) back[q] += P.Z[i][c] * P.components[c][q]; return N.norm(N.sub(hc, back)); });
    const nrm = P.components.map(c => N.dot(c, n.w2)), off = N.dot(n.w2, P.mean) + n.b2;  // readout hyperplane sliced by the shadow space
    return { Z: P.Z, varKept, res, nrm, off };
  }

  // ---- cameras ----
  const camD = () => ({ center: [0, 0], halfH: 3.8, halfW: 3.8 });   // the farthest point sits at x₁ ≈ −2.92; room for its push arrow
  const camW = () => ({ center: [0.7, 0.5], halfH: 3.25, halfW: 3.25 });
  const camS = () => Object.assign(LA.cam.defaults3d(), { target: [0, 0, 0], dist: 7.5, yaw: 0.6, pitch: 0.35 });
  const camM = () => ({ center: [0, 0], halfH: 2.1, halfW: 2.7 });
  function fit2(cam, w, h) { const c = Object.assign({}, cam); if (c.halfW) c.halfH = Math.max(c.halfH, c.halfW * h / w); return c; }
  function fit3(cam, w, h) { return Object.assign({}, cam, { dist: cam.dist * Math.max(1, 1.1 / (w / h)) }); }

  const RESET = s => { s.scene = 'two'; s.view = '2d'; s.dmix = 0; s.std = false; s.showLoss = false; s.lossFade = 0; s.showStar = false; s.showPath = false; s.k = NSTEP; s.gradPhase = 0; s.fly = 0; s.eig = 0; s.etaDemo = false; s.jiggle = 0; s.cross = false; s.link = 0; s.rings = true; s.pick = null; s.handleW = false; s.handleW0 = false; s.rightAxes = 1; };

  LA.registerLesson({
    id: 'p6-08-watch-a-classifier-learn',
    part: 6,
    title: 'Grand capstone: watch a classifier learn',
    subtitle: 'A boundary turning in data space is a point sliding down a bowl in weight space.',
    minutes: 15,
    prereqs: ['p5-03-gradient-descent', 'p5-06-chain-rule-backprop', 'p6-04-softmax-classifier', 'p4-10-pca', 'p2-12-neural-layer', 'p1-09-first-model-neuron', 'p4-09-covariance'],
    tags: ['core', 'capstone'],
    badge: { drawnIn: '2D data space + 2D weight space (bias fixed at 0) + 3D PCA shadow of ℝ⁴', realSize: 'a modern model has 10⁹–10¹² weights and the same step rule w ← w − η∇L', transfers: 'boundary ↔ point duality, loss bowl from the data, H ≈ weighted covariance, the GD step, PCA shadows — exact in any n; only the bowl’s dimension grows' },
    cantShow: ['Weight spaces with more than two parameters (we fix the bias at 0 so the bowl is drawable).', 'The 17-dimensional non-convex loss of the small network — no contours are drawn for it.', 'Mini-batch noise; the hidden layer in ℝ⁴ itself (shadow only, variance labeled).'],
    terms: [
      { term: 'data space', def: 'The plane of inputs, where the points and the boundary live.' },
      { term: 'weight space', def: 'The plane of (w₁, w₂), where one whole model is one point.' },
      { term: 'log-loss', def: 'Cross-entropy, −[y log p + (1 − y) log(1 − p)], averaged over the points.' },
      { term: 'training trajectory', def: 'The sequence of weight points visited by gradient descent.' },
      { term: 'hidden representation', def: 'The vector h a network computes for each input before its readout.' },
      { term: 'PCA shadow', def: 'The projection of many vectors onto their top principal directions, labeled with the share of variance kept.' },
    ],
    atlas: ['logistic-regression', 'optimization', 'pca', 'mlp', 'backprop'],
    thumbnail: 4,

    state() {
      const s = { view: '2d', cam: camD(), cam2: camW(), cam3: camS(), panel: 'model', w: W0.slice(), w0: W0.slice(), eta: ETA, steps: NSTEP,
        path: PATH_R.map(p => [+p[0].toFixed(5), +p[1].toFixed(5)]), model: 'logistic', snap: 4, net: null, retrained: false, retrainDiff: 0, hiPt: -1, backPhase: 0, shadow3d: true };
      RESET(s);
      return s;
    },

    render(g, s, ctx) {
      if (s.scene === 'net' || (s.scene === 'play' && s.model === 'network')) return renderNet(g, s, ctx);
      if (s.scene === 'back') return renderBack(g, s);
      if (s.scene === 'ml') return renderML(g, s);
      if (s.scene === 'recap') return renderRecap(g, s);
      renderTwo(g, s, ctx);
    },

    math(s, tex) {
      const H = str => tex.text(str, { html: true }), Tx = (v, c) => tex.text(typeof v === 'number' ? fmt(v) : v, { color: c });
      const key = keyOf(s), X = dataOf(s), I = info(key), w = curW(s);
      switch (s.panel) {
        case 'model': return tex.lines(H('p = σ(<span class="c-w">w</span>·x), b = 0 (fixed)'), tex.eq(tex.text('accuracy'), Tx(acc(w, X), 'gold')));
        case 'dual': return tex.lines(H('<span class="c-w">w</span> ↦ boundary {x : w·x = 0}'), tex.eq(tex.sym('w', { color: 'w' }), tex.op('='), Tx('(' + fmt(w[0]) + ', ' + fmt(w[1]) + ')', 'w')), tex.eq(tex.text('accuracy'), Tx(acc(w, X))));
        case 'loss': return tex.lines(H('L(w) = (1/n)Σ<sub>i</sub> −[y<sub>i</sub> log p<sub>i</sub> + (1 − y<sub>i</sub>) log(1 − p<sub>i</sub>)] + (λ/2)‖w‖²'), tex.eq(tex.text('λ = 0.01 · w* ='), Tx('(' + fmt(I.ws[0]) + ', ' + fmt(I.ws[1]) + ')', 'gold'), tex.text('L* ='), Tx(I.Ls)));
        case 'grad': { const gr = grad(w, X); return tex.lines(H('<span class="c-v">∇L</span> = (1/n)Σ<sub>i</sub> (p<sub>i</sub> − y<sub>i</sub>)x<sub>i</sub> + λw'), tex.eq(tex.sym('∇L', { color: 'v' }), tex.op('='), Tx('(' + fmt(gr[0]) + ', ' + fmt(gr[1]) + ')', 'v'))); }
        case 'step': { const k = Math.round(s.k), wk = s.path[k] || s.path[s.path.length - 1]; return tex.lines(H('w ← w − η<span class="c-v">∇L</span>, η = 1.5'), tex.eq(tex.text('k ='), Tx(String(k), 'b'), tex.text('L ='), Tx(loss(wk, XR)), tex.text('accuracy'), Tx(acc(wk, XR), 'gold'))); }
        case 'hess': return tex.lines(H('H = (1/n)Σ<sub>i</sub> p<sub>i</sub>(1 − p<sub>i</sub>)x<sub>i</sub>x<sub>i</sub><sup>ᵀ</sup> + λI'), tex.eq(tex.text('λmax ='), Tx(I.lmax, 'gold'), tex.text('λmin ='), Tx(I.lmin, 'gold')), tex.eq(tex.text('angle to the data’s PC1:'), Tx(fmt(angDeg(I.qmax, I.pc.vectors[0])) + '°')));
        case 'eta': return tex.lines(H('stable near w* ⇔ η < 2/λmax'), tex.eq(tex.text('η ='), Tx(s.eta, s.eta < ETA_MAX ? 'ok' : 'bad'), tex.text('2/λmax ='), Tx(ETA_MAX, 'bad')));
        case 'std': return tex.lines(H('κ = λmax/λmin'), tex.eq(tex.text(key === 'raw' ? 'original units' : key === 'wide' ? 'x₁ recorded ×3' : 'standardized'), tex.text('κ ='), Tx(I.lmax / I.lmin, 'gold')));
        case 'mis': return tex.lines(H('x fixed · w ← w − η<span class="c-v">∇L</span>'));
        case 'net': { const n = netAt(s, s.snap); return tex.lines(H('h = tanh(W₁x + b₁) ∈ ℝ⁴ · p = σ(w₂·h + b₂)'), tex.eq(tex.text('step'), Tx(String(snapStep(s.snap)), 'b'), tex.text('L ='), Tx(lossStr(netLoss(n))), tex.text('accuracy'), Tx(netAcc(n), 'gold'))); }
        case 'back': return tex.lines(H('∂L/∂W₁ = Σ<sub>i</sub> δ<sup>(i)</sup> x<sup>(i)ᵀ</sup>'), H('δ<sup>(i)</sup> = (1 − h²) ⊙ w₂ (p<sub>i</sub> − y<sub>i</sub>)'));
        case 'ml': return tex.lines(H('w ← w − η∇L · w ∈ ℝ<sup>10⁹</sup>'));
        case 'play': return s.model === 'network' ? tex.lines(H('h = tanh(W₁x + b₁) ∈ ℝ⁴'), tex.eq(tex.text('snapshot step'), Tx(String(snapStep(s.snap)), 'b'), tex.text('accuracy'), Tx(netAcc(netAt(s, s.snap)), 'gold'))) : tex.lines(tex.eq(tex.text('η ='), Tx(s.eta, s.eta < 2 / I.lmax ? 'ok' : 'bad'), tex.text('2/λmax ='), Tx(2 / I.lmax, 'bad'), tex.text('steps'), Tx(String(Math.round(s.steps)))), tex.eq(tex.text('w ='), Tx('(' + fmt(w[0]) + ', ' + fmt(w[1]) + ')', 'w'), tex.text('accuracy'), Tx(acc(w, X), 'gold')));
        case 'check': return tex.lines(tex.eq(tex.text('accuracy'), Tx(acc(s.w, XR), acc(s.w, XR) >= 0.95 ? 'ok' : 'text')));
        case 'check2': { const d = N.sub(s.w0, I.ws); return tex.lines(tex.eq(tex.text('angle(w₀ − w*, gentle axis) ='), Tx(fmt(angDeg(d, I.qmin)) + '°'))); }
        case 'check3': return tex.lines(H('push of point i: ‖(p<sub>i</sub> − y<sub>i</sub>)x<sub>i</sub>‖'));
        default: return '';
      }
    },

    steps: [
      { title: 'The data and the model', duration: 3,
        narration: 'Eighty synthetic points, two classes, centered (p1-07). The model is p1-09’s neuron with the bias fixed at zero so we can draw its weight space: p = σ(' + wS + '·x). Its boundary is the line through the origin perpendicular to w. Right now w is wrong: only ' + Math.round(ACC0 * 100) + '% of the points are classified correctly.',
        setup(s) { RESET(s); s.panel = 'model'; s.cam = camD(); s.cam2 = camW(); s.w = W0.slice(); s.w0 = W0.slice(); s.rightAxes = 0; },
        at(s, t) { s.rightAxes = seg(t, 0.4, 1); } },

      { title: 'Two spaces: a line here is a point there', duration: 4.5, interactive: true,
        narration: 'Every candidate boundary is one point in <b>weight space</b> (p3-06 showed this flip for regression). Drag the purple point on the right: the boundary on the left, in <b>data space</b>, turns. One model, two pictures: a line in data space, a point in weight space.',
        setup(s) { s.panel = 'dual'; s.rightAxes = 1; s.handleW = true; s.w = W0.slice(); },
        at(s, t) { const a = 2 * Math.PI * seg(t, 0.05, 0.9, 'inOut'); s.w = [W0[0] + 0.9 * Math.sin(a), W0[1] - 0.9 * (1 - Math.cos(a)) * 0.5 + 0.3 * Math.sin(a)]; s.link = 0.5 + 0.5 * Math.sin(8 * Math.PI * t); } },

      { title: 'The loss landscape', duration: 4,
        narration: 'Score every point of weight space by the average <b>log-loss</b> over the 80 data points (p6-04) plus a tiny ridge term. The contours of that bowl appear on the right; the gold minimum w* is the best boundary, drawn faint gold on the left. Training is walking down this bowl.',
        setup(s) { s.panel = 'loss'; s.handleW = false; s.link = 0; s.w = W0.slice(); s.showLoss = true; s.lossFade = 0; s.showStar = false; },
        at(s, t) { s.lossFade = seg(t, 0, 0.6); s.showStar = t > 0.55; } },

      { title: 'The gradient is a sum of data arrows', duration: 4.5,
        narration: '∇L = (1/n)Σ (pᵢ − yᵢ)xᵢ + λw: each data point pushes w by its own arrow, scaled by how wrong the model is about it (p6-04’s (p − y)x). Misclassified points push hardest. On the right the pushes add up, tip to tail, to one yellow gradient arrow ' + gS + '.',
        setup(s) { s.panel = 'grad'; s.lossFade = 1; s.showStar = true; s.gradPhase = 0; s.fly = 0; },
        at(s, t) { s.gradPhase = seg(t, 0, 0.4); s.fly = seg(t, 0.5, 0.9); } },

      { title: 'Step, step, step', duration: 5,
        narration: 'w ← w − η∇L with η = 1.5 (p5-03). Thirty steps: on the right the orange <b>training trajectory</b> slides down the bowl toward the gold minimum; on the left the boundary turns, nearly all the red rings (misclassified points) vanish, and accuracy climbs to ' + Math.round(ACC30 * 100) + '%. Same motion, two pictures.',
        setup(s) { s.panel = 'step'; s.gradPhase = 0; s.fly = 0; s.showPath = true; s.k = 0; },
        at(s, t) { s.k = NSTEP * seg(t, 0.05, 0.95, 'linear'); const k = Math.floor(s.k); s.w = (s.path[k] || s.path[NSTEP]).slice(); } },

      { title: 'The bowl’s shape comes from the data', duration: 4.5,
        narration: 'At the minimum the Hessian is H = (1/n)Σ pᵢ(1 − pᵢ)xᵢxᵢᵀ + λI (p5-02): a weighted covariance of the data (p4-09). Its gold eigenvectors on the right line up with the data’s spread directions on the left (p4-10), here within ' + Math.round(ANG_RAW) + '°. The bowl is steepest (q₁, the larger λ) along the cloud’s long direction PC1.',
        setup(s) { s.panel = 'hess'; s.showPath = true; s.k = NSTEP; s.w = s.path[NSTEP].slice(); s.eig = 0; },
        at(s, t) { s.eig = seg(t, 0.1, 0.7); } },

      { title: 'Learning rate and the red line', duration: 5.5,
        narration: 'Watch η grow. The red line sits at 2/λmax ≈ ' + Math.round(ETA_MAX) + ', computed from this Hessian (the log-loss bowl is shallow). Start near the minimum: below the line the path settles; above it the path bounces around w* and never comes to rest. Log-loss gradients are bounded, so it does not fly away. p5-03’s rule, on a real loss.',
        setup(s) { s.panel = 'eta'; s.eig = 0; s.showPath = false; s.etaDemo = true; s.eta = 0.2 * ETA_MAX; },
        at(s, t) { s.eta = ETA_MAX * lerp(0.2, 1.1, seg(t, 0.05, 0.95, 'linear')); } },   // at 1.1·(2/λmax) the bounce stays inside the panel

      { title: 'Standardize the data', duration: 5,
        narration: 'Suppose x₁ had been recorded in units three times smaller: the cloud stretches sideways and the bowl narrows (κ = λmax/λmin goes from ' + fmt(KAP.raw) + ' to ' + fmt(KAP.wide) + '). Now divide each feature by its spread (p5-03’s toggle, p6-07’s BatchNorm on the input): every feature gets unit spread, the bowl rounds back (κ = ' + fmt(KAP.std) + '), and one learning rate again suits every direction.',
        setup(s) { s.panel = 'std'; s.etaDemo = false; s.eta = ETA; s.showPath = true; s.k = NSTEP; s.dmix = 0; },
        at(s, t) { s.dmix = t < 0.45 ? seg(t, 0.05, 0.38) : 1 + seg(t, 0.5, 0.85); const hw = s.dmix <= 1 ? lerp(3.8, 9.2, s.dmix) : lerp(9.2, 3.8, s.dmix - 1); /* ×3 data reach x₁ ≈ ±8.75 */s.cam.halfW = hw; s.cam.halfH = hw; s.w = curW(Object.assign({}, s, { showPath: true })).slice(); } },

      { title: 'Training does not move the data', duration: 4, misconception: true,
        narration: 'It looks as if the points were being pushed around. They never move. Only w moves, in a different space, and the boundary it defines sweeps across the fixed data. Keep the two spaces apart in your head: the arrows of step 4 live in data space but act on w.',
        setup(s) { s.panel = 'mis'; s.dmix = 0; s.cam = camD(); s.showPath = true; s.k = 0; s.jiggle = 0; s.cross = false; s.w = W0.slice(); },
        at(s, t) { s.jiggle = LA.seg.pulse(t, 0.05, 0.4); s.cross = t > 0.05; s.k = NSTEP * seg(t, 0.45, 0.95, 'linear'); s.w = (s.path[Math.floor(s.k)] || s.path[NSTEP]).slice(); } },

      { title: 'A network: the lift, seen as a shadow', duration: 5,
        narration: 'Now moons (p2-12’s harder data) and a width-4 tanh layer: the <b>hidden representation</b> h = tanh(W₁x + b₁) lives in ℝ⁴, which we cannot draw. We show its <b>PCA shadow</b> (p4-10) on three directions, labeled with the variance kept, plus what it drops. The readout is a plane cutting this space.',
        setup(s) { RESET(s); s.scene = 'net'; s.panel = 'net'; s.view = '3d'; s.cam = camS(); s.cam2 = camM(); s.snap = 4; },
        at(s, t) { LA.cam.tween(s, { yaw: 0.35 }, { yaw: 0.85 }, seg(t, 0, 1, 'linear')); } },

      { title: 'Watch the hidden layer learn', duration: 6,
        narration: 'Five snapshots of training: steps 0, 100, 300, 1000, 2000 (weights stored; Retrain in Play reproduces them). At step 0 the shadow is a tangled blob; by step 300 the two moons pull apart; by 2000 a plane separates them and the data-space boundary has bent into a curve. The lift is learned, not chosen (p6-05).',
        setup(s) { s.scene = 'net'; s.panel = 'net'; s.view = '3d'; s.snap = 0; },
        at(s, t) { s.snap = 4 * seg(t, 0.05, 0.95, 'linear'); } },

      { title: 'Backprop inside it', duration: 4.5,
        narration: 'The gradient for W₁ is a sum of rank-1 sheets δxᵀ (p5-06), one per point: the 4×2 heatmap for a single point has every row a multiple of xᵀ; the batch sum does not. Every number steering this network is an outer product pulled back through a transpose.',
        setup(s) { s.scene = 'back'; s.panel = 'back'; s.view = '2d'; s.cam = camM(); s.backPhase = 0; },
        at(s, t) { s.backPhase = seg(t, 0.1, 0.9); } },

      { title: 'Why ML cares: the whole course in one loop', duration: 4.5, mlTie: true,
        narration: 'Data are vectors (Part 1); a model is a map (Part 2); fitting is projection or descent (Parts 3, 5); the bowl’s shape is a matrix with eigenvectors (Part 4); gradients are outer products pulled back (Part 5); hidden layers are lifts we view as shadows. At 10⁹ weights the bowl has 10⁹ axes. The picture stops at 3. The formula does not.',
        setup(s) { s.scene = 'ml'; s.panel = 'ml'; s.view = '2d'; s.cam = LA.cam.defaults2d(); },
        at(s, t) { } },

      { title: 'Recap', duration: 2, recap: true,
        narration: 'Rows are points; models are maps; training is stepping down a bowl whose shape is a matrix built from the data. A boundary turning in data space is a point sliding in weight space, and what you cannot draw you measure, or show as a labeled shadow.',
        cards: ['<b>Boundary ↔ point.</b> One model is a line in data space and a point in weight space.', '<b>Descend the bowl the data built.</b> H = Σ pᵢ(1 − pᵢ)xᵢxᵢᵀ/n + λI; stay below η = 2/λmax.', '<b>Shadows, labeled.</b> A hidden layer in ℝ⁴ is shown as its PCA shadow with the variance kept.'],
        setup(s) { s.scene = 'recap'; s.panel = 'none'; s.view = '2d'; s.cam = LA.cam.defaults2d(); },
        at(s, t) { } },
    ],

    playground: {
      intro: '<p>Drag the start point w₀ in weight space and watch the path and the boundary. Push η past the red line, standardize, or switch to the network and its hidden-layer shadow.</p>',
      setup(s) { RESET(s); s.scene = 'play'; s.panel = 'play'; s.model = 'logistic'; s.cam = camD(); s.cam2 = camW(); s.cam3 = camS(); s.showLoss = true; s.lossFade = 1; s.showStar = true; s.showPath = true; s.handleW0 = true; s.w0 = W0.slice(); s.eta = ETA; s.steps = NSTEP; s.snap = 4; s.shadow3d = true; },
      controls: [
        { type: 'slider', bind: 'eta', min: 0.1, max: 70, step: 0.1, label: 'Learning rate η (red line at 2/λmax)', color: 'b' },
        { type: 'slider', bind: 'steps', min: 0, max: 60, step: 1, label: 'Steps' },
        { type: 'toggle', bind: 'std', label: 'Standardize data' },
        { type: 'preset', label: 'Model', items: [
          { label: 'Logistic (default)', apply(s) { s.model = 'logistic'; s.view = '2d'; s.std = false; s.dmix = 0; s.eta = ETA; s.steps = NSTEP; s.w0 = W0.slice(); s.cam = camD(); } },
          { label: 'x₁ recorded ×3', apply(s) { s.model = 'logistic'; s.view = '2d'; s.std = false; s.dmix = 1; s.cam = Object.assign(camD(), { halfH: 9.2, halfW: 9.2 }); } },
          { label: 'Network snapshots', apply(s) { s.model = 'network'; s.view = '3d'; s.cam = camS(); s.cam2 = camM(); s.snap = 4; } },
        ] },
        { type: 'button', label: 'Retrain network (2000 steps)', onClick(s) { const snaps = train(); let d = 0; snaps.forEach((n, i) => { const a = flatNet(n), b = flatNet(NET0[i]); a.forEach((v, j) => { d = Math.max(d, Math.abs(v - b[j])); }); }); s.net = snaps; s.retrained = true; s.retrainDiff = d; s.model = 'network'; s.view = '3d'; s.cam = camS(); s.cam2 = camM(); } },
        { type: 'slider', bind: 'snap', min: 0, max: 4, step: 0.01, label: 'Network snapshot (0 → 2000 steps)' },
      ],
    },

    checks: [
      { kind: 'manipulate',
        prompt: 'Drag <span class="c-w">w</span> in weight space so the boundary classifies at least 95% of the points correctly.',
        setup(s) { RESET(s); s.panel = 'check'; s.cam = camD(); s.cam2 = camW(); s.w = W0.slice(); s.handleW = true; s.showLoss = true; s.lossFade = 0.6; },
        judge(s) { const a = acc(s.w, XR); return { ok: a >= 0.95, progress: Math.max(0, (a - 0.4) / 0.55) }; },
        hint: 'Aim w across the gap between the two clouds, roughly toward the blue blob.',
        solution(s) { s.w = info('raw').ws.slice(); } },
      { kind: 'manipulate',
        prompt: 'Drag the start point w₀ onto the gentle eigen-direction of the bowl (the gold axis with the smaller λ) so the path runs straight to the minimum.',
        setup(s) { RESET(s); s.panel = 'check2'; s.cam = camD(); s.cam2 = camW(); s.w0 = W0.slice(); s.handleW0 = true; s.showLoss = true; s.lossFade = 1; s.showStar = true; s.showPath = true; s.eig = 1; s.eta = ETA; s.steps = NSTEP; },
        judge(s) { const I = info('raw'), d = N.sub(s.w0, I.ws), a = angDeg(d, I.qmin), far = N.norm(d) > 0.5; return { ok: a < LA.TOL.angleDeg && far, progress: far ? Math.max(0, 1 - a / 60) : 0.2, hint: far ? undefined : 'Start farther than 0.5 from the minimum.' }; },
        hint: 'Start on the gold axis with the smaller λ.',
        solution(s) { const I = info('raw'); s.w0 = N.add(I.ws, N.scale(I.qmin, 1.5)); } },
      { kind: 'manipulate',
        prompt: 'Drag the gold marker onto the data point that pushes w hardest right now: the one with the largest arrow (pᵢ − yᵢ)xᵢ.',
        setup(s) { RESET(s); s.panel = 'check3'; s.cam = camD(); s.cam2 = camW(); s.w = W0.slice(); s.gradPhase = 1; s.pick = [0, 0]; },
        judge(s) { const i = nearest(s.pick), d = N.dist(s.pick, XR[ARGPUSH]); const ok = i === ARGPUSH && N.dist(s.pick, XR[i]) < 0.25; return { ok, progress: ok ? 1 : Math.max(0, 1 - d / 4) }; },
        hint: 'Far from the origin and on the wrong side.',
        solution(s) { s.pick = XR[ARGPUSH].slice(); } },
    ],
  });

  /* ---------------- helpers ---------------- */
  function snapStep(snap) { const k = Math.min(3, Math.floor(snap)), f = snap - k; return Math.round(lerp(SNAP_STEPS[k], SNAP_STEPS[Math.min(4, k + 1)], f)); }
  function nearest(p) { let b = -1, bd = Infinity; XR.forEach((x, i) => { const d = N.dist(p, x); if (d < bd) { bd = d; b = i; } }); return b; }
  function keyOf(s) { if (s.scene === 'play') return s.std ? 'std' : (s.dmix >= 0.5 ? 'wide' : 'raw'); return s.dmix < 0.5 ? 'raw' : s.dmix < 1.5 ? 'wide' : 'std'; }
  function dataOf(s) {
    if (s.scene === 'play') return DATA[keyOf(s)];
    if (s.dmix <= 0) return XR;
    if (s.dmix <= 1) return XR.map((p, i) => lerpV(p, DATA.wide[i], s.dmix));
    return DATA.wide.map((p, i) => lerpV(p, DATA.std[i], Math.min(1, s.dmix - 1)));
  }
  function curW(s) {
    if (s.scene === 'play' && s.model === 'logistic') { const P = gdPath(s.w0, s.eta, Math.round(s.steps), DATA[keyOf(s)]); return P[P.length - 1]; }
    if (s.panel === 'std') { const k = keyOf(s); return gdPath(W0, ETA, NSTEP, DATA[k])[NSTEP]; }
    if (s.panel === 'check2') return gdPath(s.w0, ETA, NSTEP, XR)[NSTEP];
    return s.w;
  }
  const classColor = i => (T[i] ? 'data' : 'dataB');
  const sideRamp = t => LA.color.mix(LA.color.dataB, LA.color.data, t);

  /* ---------------- the two spaces ---------------- */
  function renderTwo(g, s, ctx) {
    const key = keyOf(s), X = dataOf(s), I = info(key), w = curW(s);
    const W = g.size.w, H = g.size.h;
    const camL = fit2(s.cam, W / 2, H), camR = fit2(s.cam2, W / 2, H);   // panels are full height: keep the requested half-width visible
    let PL = null, PR = null;
    const pathData = s.etaDemo ? gdPath(N.add(info('raw').ws, N.scale(info('raw').qmax, -0.8)), s.eta, NSTEP, XR)
      : (s.scene === 'play' ? gdPath(s.w0, s.eta, Math.round(s.steps), DATA[key]) : (s.panel === 'std' ? gdPath(W0, ETA, NSTEP, DATA[key]) : (s.panel === 'check2' ? gdPath(s.w0, ETA, NSTEP, XR) : s.path)));
    const wNow = s.etaDemo ? pathData[pathData.length - 1] : w;
    g.split((g2, r) => {
      PL = g2.P;
      g2.grid(null, { range: 10, step: 1, alpha: 0.25 });
      g2.axes({ labels: false, names: ['x₁', 'x₂'] });
      if (s.panel !== 'mis' || !(s.jiggle > 0.01)) g2.heat2d(p => sig(wNow[0] * p[0] + wNow[1] * p[1]), g2.P.bounds(), { res: 48, range: [0, 1], ramp: sideRamp, alpha: 0.14 });
      // the data (jiggling only in the misconception step)
      const n = N.normalize([wNow[0], wNow[1]]);
      const P = X.map(x => { if (!s.jiggle) return x; const d = n[0] * x[0] + n[1] * x[1]; return [x[0] - 0.35 * s.jiggle * d * n[0], x[1] - 0.35 * s.jiggle * d * n[1]]; });
      g2.points(P, { r: 3.6, colors: classColor });
      if (s.rings) P.forEach((x, i) => { if ((wNow[0] * X[i][0] + wNow[1] * X[i][1] > 0 ? 1 : 0) !== T[i]) g2.point(x, { r: 6.5, color: 'rgba(0,0,0,0)', stroke: 'bad', strokeWidth: 1.3 }); });
      if (s.showStar && key === 'raw') g2.lineThrough([0, 0], [-I.ws[1], I.ws[0]], { color: 'gold', width: 1.4, alpha: 0.55, len: 12 });
      g2.lineThrough([0, 0], [-wNow[1], wNow[0]], { color: 'w', width: 2.4, len: 12 });
      g2.vector([0, 0], wNow, { color: 'w', width: 2.6, label: 'w' });
      // gradient pushes (p − y)x at each point
      if (s.gradPhase > 0 && s.fly < 0.02) X.forEach((x, i) => { const c = sig(N.dot(w, x)) - T[i]; const v = N.scale(x, c * 0.35 * s.gradPhase); if (N.norm(v) > 0.02) g2.vector(x, N.add(x, v), { color: 'v', width: 1.4, head: 6, alpha: 0.85 }); });
      // the data's principal axes (Hessian step)
      if (s.eig > 0.01) { const pc = I.pc; pc.vectors.forEach((q, k) => { const L = 1.6 * Math.sqrt(pc.values[k]); g2.line(N.scale(q, -L), N.scale(q, L), { color: 'gold', width: 2.2, alpha: s.eig }); g2.label('PC' + (k + 1), N.scale(q, L), { color: 'gold', size: 12, dx: 12, dy: -8, alpha: s.eig }); }); }
      // gold marker for check 3
      if (s.pick) { g2.point(s.pick, { r: 7, color: 'rgba(0,0,0,0)', stroke: 'gold', strokeWidth: 2.2 }); g2.handle('pick', s.pick, (p, st) => { st.pick = [p[0], p[1]]; }, { color: 'gold' }); }
      if (s.cross) { g2.label('“the data moves”', [r.x + 40, r.y + 44], { screen: true, anchor: 'left', size: 14, color: 'bad', font: 'ui' }); g2.marker.cross([r.x + 22, r.y + 44], { screen: true, size: 18 }); }
      g2.label('data space', [r.x + 12, r.y + 18], { screen: true, anchor: 'left', size: 13.5, color: 'text', font: 'ui', bold: true });
      g2.label('accuracy ' + fmt.pct(acc(wNow, X)), [r.x + r.w - 10, r.y + 18], { screen: true, anchor: 'right', size: 12.5, color: acc(wNow, X) >= 0.95 ? 'ok' : 'hud', font: 'ui' });
      if (s.panel === 'std') g2.label(key === 'raw' ? 'original units' : key === 'wide' ? 'x₁ recorded ×3' : 'standardized: unit spread per feature', [r.x + 12, r.y + r.h - 16], { screen: true, anchor: 'left', size: 12.5, color: 'gold', font: 'ui' });
      if (s.panel === 'model' || s.panel === 'dual') g2.label('blue: y = 1 · rose: y = 0 · ringed: misclassified', [r.x + 12, r.y + r.h - 16], { screen: true, anchor: 'left', size: 11.5, color: 'muted', font: 'ui' });
    }, (g2, r) => {
      PR = g2.P;
      const ra = s.rightAxes;
      if (ra > 0.01) { g2.grid(null, { range: 3, step: 1, alpha: 0.3 * ra }); g2.axes({ labels: ra > 0.5, names: ['w₁', 'w₂'] }); }
      if (s.showLoss && s.lossFade > 0.01) {
        if (s.panel === 'std' && s.dmix > 0 && s.dmix < 2) {   // crossfade between bowls while the data morphs
          const fr = s.dmix <= 1 ? s.dmix : s.dmix - 1, a = s.dmix <= 1 ? 'raw' : 'wide', b = s.dmix <= 1 ? 'wide' : 'std';
          g2.contours(lookup(a), info(a).levels, WB, { res: RES, color: 'residual', width: 1.2, alpha: 0.75 * (1 - fr) });
          g2.contours(lookup(b), info(b).levels, WB, { res: RES, color: 'residual', width: 1.2, alpha: 0.75 * fr });
        } else g2.contours(lookup(key), I.levels, WB, { res: RES, color: 'residual', width: 1.2, alpha: 0.75 * s.lossFade });
      }
      if (s.showStar) { g2.point(I.ws, { r: 5.5, color: 'gold' }); g2.label('w*', I.ws, { color: 'gold', dx: 12, dy: 12, size: 13 }); }
      // Hessian eigenvectors at w*
      if (s.eig > 0.01) [[I.qmax, I.lmax, 'q₁'], [I.qmin, I.lmin, 'q₂']].forEach(([q, l, nm]) => { g2.vector(I.ws, N.add(I.ws, N.scale(q, 1.1)), { color: 'gold', width: 2.4, alpha: s.eig, label: nm + ' (λ = ' + fmt.fixed(l, 3) + ')', labelAt: nm === 'q₁' ? 0.8 : 1, labelOffset: nm === 'q₁' ? 26 : 16 }); g2.line(I.ws, N.add(I.ws, N.scale(q, -1.1)), { color: 'gold', width: 1.2, dashed: true, alpha: s.eig }); });
      // the training trajectory
      if (s.showPath || s.etaDemo || s.scene === 'play') {
        const kk = s.etaDemo || s.scene === 'play' || s.panel === 'std' || s.panel === 'check2' ? pathData.length - 1 : Math.floor(s.k);
        const P = pathData.slice(0, kk + 1);
        const bounce = s.etaDemo && s.eta > ETA_MAX;   // the rule itself; just below the line the path still settles, only slowly
        if (P.length > 1) {
          const cut = bounce ? Math.max(1, P.length - 11) : P.length;
          g2.polyline(P.slice(0, cut), { color: 'b', width: 2 }); g2.points(P.slice(0, cut), { r: 2.6, color: 'b' });
          if (bounce) { g2.polyline(P.slice(cut - 1), { color: 'bad', width: 2 }); g2.points(P.slice(cut), { r: 2.6, color: 'bad' }); }
        }
        if (s.etaDemo) g2.label(bounce ? 'η > 2/λmax: never comes to rest' : 'η < 2/λmax: settles', [r.x + 12, r.y + r.h - 16], { screen: true, anchor: 'left', size: 12.5, color: bounce ? 'bad' : 'ok', font: 'ui' });
      }
      // gradient as a sum of pushes, tip to tail at w
      if (s.fly > 0.98 || (s.gradPhase > 0 && s.panel === 'check3')) gradSum(g2, w, XR, 1);
      // current w
      if (!s.etaDemo) { g2.point(wNow, { r: 6, color: 'w' }); g2.label('w', wNow, { color: 'w', dx: -12, dy: -12, size: 14 }); }
      if (s.handleW) g2.handle('w', s.w, (p, st) => { st.w = [LA.clamp(p[0], -3, 3), LA.clamp(p[1], -3, 3)]; }, { color: 'w' });
      if (s.handleW0) { g2.point(s.w0, { r: 5, color: 'b' }); g2.label('w₀', s.w0, { color: 'b', dx: 12, dy: -10, size: 13 }); g2.handle('w0', s.w0, (p, st) => { st.w0 = [LA.clamp(p[0], -3, 3), LA.clamp(p[1], -3, 3)]; }, { color: 'b' }); }
      if (s.etaDemo || s.scene === 'play') etaBar(g2, r, s.eta, s.scene === 'play' ? 2 / I.lmax : ETA_MAX);
      g2.label('weight space', [r.x + 12, r.y + 18], { screen: true, anchor: 'left', size: 13.5, color: 'text', font: 'ui', bold: true });
      if (s.showLoss) g2.label('L(w) = ' + fmt(loss(wNow, X)), [r.x + r.w - 10, r.y + 18], { screen: true, anchor: 'right', size: 12.5, color: 'hud', font: 'ui' });
      if (s.panel === 'std' || (s.scene === 'play' && s.model === 'logistic')) g2.label('κ = λmax/λmin = ' + fmt(I.lmax / I.lmin), [r.x + r.w - 10, r.y + 40], { screen: true, anchor: 'right', size: 12.5, color: 'gold', font: 'ui' });
    }, { left: { view: '2d', cam: camL }, right: { view: '2d', cam: camR } });
    // ---- cross-panel drawing in the outer (full-canvas) frame ----
    const toOuter = sp => g.P.toWorld(sp[0], sp[1]);
    if (s.link > 0 && PL && PR) {
      const a = PR.toScreen(s.w), b = PL.toScreen(s.w);
      g.line(toOuter(a), toOuter(b), { color: 'w', width: 1.4, dashed: true, alpha: 0.3 + 0.5 * s.link });
    }
    if (s.fly > 0.01 && s.fly <= 0.98 && PL && PR) {   // the pushes fly from the data points to the weight panel and line up tip to tail
      let o = w.slice(); const n = XR.length;
      XR.forEach((x, i) => {
        const c = sig(N.dot(w, x)) - T[i], v = N.scale(x, c);
        const fromA = PL.toScreen(x), fromB = PL.toScreen(N.add(x, N.scale(v, 0.35)));
        const toA = PR.toScreen(o), o2 = N.add(o, N.scale(v, 1 / n)), toB = PR.toScreen(o2);
        o = o2;
        const A = lerpV(fromA.slice(0, 2), toA.slice(0, 2), s.fly), B = lerpV(fromB.slice(0, 2), toB.slice(0, 2), s.fly);
        if (Math.hypot(B[0] - A[0], B[1] - A[1]) > 1) g.vector(toOuter(A), toOuter(B), { color: 'v', width: 1.3, head: 5, alpha: 0.85 });
      });
    }
  }
  function gradSum(g2, w, X, u) {
    const n = X.length; let o = w.slice();
    X.forEach((x, i) => { const v = N.scale(x, (sig(N.dot(w, x)) - T[i]) / n); const e = N.add(o, v); g2.line(o, e, { color: 'v', width: 1, alpha: 0.5 }); o = e; });
    const lw = N.scale(w, LAM); g2.line(o, N.add(o, lw), { color: 'v', width: 1, alpha: 0.5 });
    const gr = grad(w, X);
    g2.vector(w, N.add(w, gr), { color: 'v', width: 3, label: '∇L' });
    g2.vector(w, N.sub(w, gr), { color: 'v', width: 1.6, dashed: true, alpha: 0.8, label: '−∇L' });
  }
  function etaBar(g2, r, eta, emax) {
    const x0 = r.x + 16, x1 = r.x + r.w - 16, y = r.y + r.h - 44, vmax = 70;
    const X = v => x0 + (x1 - x0) * Math.min(1, v / vmax);
    const P = g2.P, W = (sx, sy) => P.toWorld(sx, sy);
    g2.line(W(x0, y), W(x1, y), { color: 'gridAxis', width: 2 });
    g2.line(W(X(emax), y - 9), W(X(emax), y + 9), { color: 'bad', width: 2.6 });
    g2.label('2/λmax = ' + fmt(emax), [X(emax), y - 18], { screen: true, size: 11.5, color: 'bad', font: 'ui' });
    g2.label('η = ' + fmt(eta), [X(eta), y + 18], { screen: true, size: 11.5, color: 'b', font: 'ui' });
    g2.marker.ring(W(X(eta), y), { r: 5, color: 'b' });
  }

  /* ---------------- the network: data inset + 3D PCA shadow ---------------- */
  function renderNet(g, s, ctx) {
    const n = netAt(s, s.snap), sh = shadow(n), W = g.size.w, H = g.size.h;
    const side = W >= 1.2 * H;
    const rA = side ? { x: 0, y: 0, w: 0.4, h: 1 } : { x: 0, y: 0, w: 1, h: 0.36 }, rB = side ? { x: 0.4, y: 0, w: 0.6, h: 1 } : { x: 0, y: 0.36, w: 1, h: 0.64 };
    const cam3 = s.scene === 'play' ? s.cam : s.cam;
    g.viewport(rA, { view: '2d', cam: fit2(s.cam2, rA.w * W, rA.h * H), bg: 'panel' }, (g2, r) => {
      const logit = p => { const f = fwd(n, p); return Math.log(Math.max(f.p, 1e-9) / Math.max(1 - f.p, 1e-9)); };
      g2.heat2d(p => fwd(n, p).p, g2.P.bounds(), { res: 44, range: [0, 1], ramp: t => LA.color.mix(LA.color.data, LA.color.dataB, t), alpha: 0.3 });
      g2.contours(logit, [0], g2.P.bounds(), { res: 44, color: 'gold', width: 2.2 });
      g2.points(XM, { r: 3, colors: i => (YM[i] ? 'dataB' : 'data') });
      g2.label('data space: moons (synthetic)', [r.x + 10, r.y + 16], { screen: true, anchor: 'left', size: 12.5, color: 'text', font: 'ui', bold: true });
      g2.label('accuracy ' + fmt.pct(netAcc(n)) + ' · L = ' + lossStr(netLoss(n)), [r.x + 10, r.y + r.h - 14], { screen: true, anchor: 'left', size: 12, color: 'hud', font: 'ui' });
    });
    g.viewport(rB, { view: '3d', cam: fit3(cam3, rB.w * W, rB.h * H), border: false }, (g3, r) => {
      g3.axes({ length: 2.2, names: ['PC1', 'PC2', 'PC3'] });
      const Z = sh.Z;
      // the readout hyperplane sliced by the shadow space: (C w₂)·z + (w₂·μ + b₂) = 0
      const nn = N.norm(sh.nrm);
      if (nn > 0.05) { const p0 = N.scale(sh.nrm, -sh.off / (nn * nn)); if (N.norm(p0) < 2.5) g3.planes([{ point: p0, normal: sh.nrm, extent: 1.9, color: 'gold', alpha: 0.18 }]); }
      g3.points(Z, { r: 3.2, colors: i => (YM[i] ? 'dataB' : 'data') });
      g3.label('shadow of ℝ⁴ on 3 directions · ' + Math.round(sh.varKept * 100) + '% variance', [r.x + 12, r.y + 18], { screen: true, anchor: 'left', size: 13, color: 'mag', font: 'ui', bold: true });
      g3.label('gold: the readout plane, sliced by the shadow space', [r.x + 12, r.y + 38], { screen: true, anchor: 'left', size: 11.5, color: 'muted', font: 'ui' });
      g3.label('step ' + snapStep(s.snap) + ' · ' + (s.retrained ? 'trained in your browser: 2000 steps (max diff ' + fmt.fixed(s.retrainDiff, 6) + ')' : 'hard-coded weights (deterministic)'), [r.x + 12, r.y + r.h - 16], { screen: true, anchor: 'left', size: 11.5, color: s.retrained ? 'ok' : 'muted', font: 'ui' });
      const hw = 120;
      g3.hist(sh.res, 10, { x: r.x + r.w - hw - 14, y: r.y + r.h - 86, w: hw, h: 46, range: [0, Math.max(0.05, Math.max.apply(null, sh.res))], color: 'residual', title: 'what the shadow drops' });
    });
  }

  /* ---------------- backprop: δxᵀ for one point and for the batch ---------------- */
  function renderBack(g, s) {
    const n = (s.net || NET0)[2];                       // the step-300 snapshot: still learning
    const per = XM.map((x, i) => { const f = fwd(n, x), d2 = f.p - YM[i]; const delta = f.h.map((h, q) => (1 - h * h) * n.w2[q] * d2); return { x, delta, f, d2 }; });
    let hi = 0; per.forEach((o, i) => { if (Math.abs(o.d2) > Math.abs(per[hi].d2)) hi = i; });
    const one = per[hi], G1 = one.delta.map(d => [d * one.x[0], d * one.x[1]]);
    const GB = [[0, 0], [0, 0], [0, 0], [0, 0]]; per.forEach(o => o.delta.forEach((d, q) => { GB[q][0] += d * o.x[0]; GB[q][1] += d * o.x[1]; }));
    const W = g.size.w, H = g.size.h, side = W >= 1.2 * H;
    const rA = side ? { x: 0, y: 0, w: 0.5, h: 1 } : { x: 0, y: 0, w: 1, h: 0.45 }, rB = side ? { x: 0.5, y: 0, w: 0.5, h: 1 } : { x: 0, y: 0.45, w: 1, h: 0.55 };
    g.viewport(rA, { view: '2d', cam: fit2(camM(), rA.w * W, rA.h * H), bg: 'panel' }, (g2, r) => {
      g2.points(XM, { r: 2.8, colors: i => (YM[i] ? 'dataB' : 'data'), alpha: 0.6 });
      g2.point(one.x, { r: 7, color: 'rgba(0,0,0,0)', stroke: 'gold', strokeWidth: 2 });
      const back = [0, 0]; one.delta.forEach((d, q) => { back[0] += n.W1[q][0] * d; back[1] += n.W1[q][1] * d; });   // ∂L/∂x = W₁ᵀδ
      const L = N.norm(back) || 1, k = 0.9 / L;
      if (s.backPhase > 0.6) g2.vector(one.x, N.add(one.x, N.scale(back, k * seg(s.backPhase, 0.6, 1))), { color: 'v', width: 2.6, label: 'W₁ᵀδ' });
      g2.label('the highlighted point (largest |p − y| at step 300)', [r.x + 10, r.y + 16], { screen: true, anchor: 'left', size: 12, color: 'text', font: 'ui', bold: true });
      g2.label('yellow: δ pulled back through W₁ᵀ (direction; length rescaled)', [r.x + 10, r.y + r.h - 14], { screen: true, anchor: 'left', size: 11, color: 'muted', font: 'ui' });
    });
    g.viewport(rB, { view: '2d', cam: fit2({ center: [0, -0.1], halfH: 2.4, halfW: 2.25 }, rB.w * W, rB.h * H), border: false }, (g2, r) => {
      const heat = (M, cx, title, alpha) => {
        const m = Math.max(1e-6, ...M.map(row => Math.max(Math.abs(row[0]), Math.abs(row[1]))));
        const b = { x0: cx - 0.8, x1: cx + 0.8, y0: -1.7, y1: 1.5 };
        g2.heat2d(M.map(row => row.map(v => v / m)), b, { range: [-1, 1], ramp: t => LA.color.diverge(2 * t - 1), smooth: false, alpha });
        for (let q = 0; q < 4; q++) for (let j = 0; j < 2; j++) g2.label(fmt(M[q][j]), [cx - 0.4 + 0.8 * j, 1.5 - 0.4 - 0.8 * q], { size: 11.5, color: 'text', bg: false, font: 'ui', alpha });
        g2.label(title, [cx, 1.5], { anchor: 'above', dy: -6, size: 12.5, color: 'text', font: 'ui', alpha });
      };
      const xT = 'xᵀ = (' + fmt(one.x[0]) + ', ' + fmt(one.x[1]) + ')';
      heat(G1, -1.1, 'δ⁽ⁱ⁾x⁽ⁱ⁾ᵀ (one point)', 1);
      g2.label('every row ∝ ' + xT, [-1.1, -1.7], { anchor: 'below', dy: 6, size: 11.5, color: 'gold', font: 'ui' });
      if (s.backPhase > 0.3) { heat(GB, 1.1, 'batch sum Σᵢ δ⁽ⁱ⁾x⁽ⁱ⁾ᵀ', seg(s.backPhase, 0.3, 0.6)); g2.label('rows no longer parallel', [1.1, -1.7], { anchor: 'below', dy: 6, size: 11.5, color: 'muted', font: 'ui' }); }
    });
  }

  /* ---------------- ML tie-in ---------------- */
  function renderML(g, s) {
    const W = g.size.w, H = g.size.h, r6 = LA.rng(610);
    const wts = [], grs = []; for (let i = 0; i < 12; i++) { wts.push(LA.rng.gauss(r6)); grs.push(LA.rng.gauss(r6) * 0.4); }
    g.strip(wts, { x: 20, y: 40, w: W * 0.58, h: 22, range: [-2.5, 2.5], ramp: t => LA.color.diverge(2 * t - 1), title: '12 of 10⁹ weights w' });
    g.strip(grs, { x: 20, y: 92, w: W * 0.58, h: 22, range: [-1, 1], ramp: t => LA.color.diverge(2 * t - 1), title: 'their gradient ∇L' });
    // a 6×6 Hessian sketch (random symmetric positive definite)
    const A = []; for (let i = 0; i < 6; i++) { A.push([]); for (let j = 0; j < 6; j++) A[i].push(LA.rng.gauss(r6)); }
    const Hm = N.addM(N.scaleM(N.matMul(A, N.transpose(A)), 1 / 6), N.scaleM(N.identity(6), 0.2));
    const hx = W * 0.62 + 20, hs = Math.min(W - hx - 20, 120);
    g.viewport({ x: hx, y: 22, w: hs, h: hs + 18 }, { view: '2d', cam: { center: [0, 0.15], halfH: 1.25 }, bg: 'panel' }, (g2, r) => {
      g2.heat2d(Hm, { x0: -1, x1: 1, y0: -1, y1: 1 }, { range: [-2, 2], ramp: t => LA.color.diverge(2 * t - 1), smooth: false, alpha: 1 });
      g2.label('H (6 of 10⁹ axes)', [r.x + r.w / 2, r.y + 10], { screen: true, size: 10.5, color: 'muted', font: 'ui' });
    });
    const items = [['Part 1', 'points'], ['Part 2', 'grid morph'], ['Part 3', 'projection'], ['Part 4', 'eigen-axes'], ['Part 5', 'zig-zag path'], ['Part 6', 'PCA shadow']];
    const cols = 3, tw = (W - 40 - (cols - 1) * 10) / cols, th = Math.min(230, (H - 250) / 2);
    items.forEach((it, k) => {
      const cx = 20 + (k % cols) * (tw + 10), cy = 170 + Math.floor(k / cols) * (th + 12);
      g.viewport({ x: cx, y: cy, w: tw, h: th }, { view: '2d', cam: { center: [0, 0], halfH: 1.4 }, bg: 'panel' }, (g2, r) => { thumb(g2, k); g2.label(it[0] + ' · ' + it[1], [r.x + 8, r.y + 12], { screen: true, anchor: 'left', size: 11, color: 'text', font: 'ui', bold: true }); });
    });
    g.hud('The picture stops at 3. The formula does not.', 20, H - 24, { color: 'gold', size: 14 });
  }
  function thumb(g2, k) {
    const rr = LA.rng(620 + k);
    if (k === 0) { const P = XR.slice(0, 40).map(p => N.scale(p, 0.45)); g2.points(P, { r: 2.2, colors: i => classColor(i) }); }
    else if (k === 1) { g2.grid([[1, 0.5], [0.2, 0.9]], { range: 3, alpha: 0.7 }); g2.vector([1, 0.2], { color: 'i', width: 2, label: 'i', hat: true }); g2.vector([0.5, 0.9], { color: 'j', width: 2, label: 'j', hat: true }); }
    else if (k === 2) { g2.lineThrough([0, 0], [1, 0.3], { color: 'teal', width: 1.6, len: 3 }); g2.vector([0.4, 0.9], { color: 'v', width: 2, label: 'v' }); const f = N.proj([0.4, 0.9], [1, 0.3]); g2.dropLine([0.4, 0.9], f, { color: 'residual' }); }
    else if (k === 3) { g2.ellipse([[1.0, -0.3], [0.5, 0.5]], [0, 0], { stroke: 'teal', width: 1.4 }); const e = N.eigSym([[1.09, 0.35], [0.35, 0.5]]); e.vectors.forEach((q, i) => g2.line(N.scale(q, -1.1 * Math.sqrt(e.values[i])), N.scale(q, 1.1 * Math.sqrt(e.values[i])), { color: 'gold', width: 2 })); }
    else if (k === 4) { g2.contours(p => p[0] * p[0] / 0.25 + p[1] * p[1] * 1.2, [0.3, 0.8, 1.5, 2.4], { x0: -2, x1: 2, y0: -1.4, y1: 1.4 }, { res: 32, color: 'residual', width: 1 }); const P = []; let q = [-0.6, 0.9]; for (let i = 0; i < 10; i++) { P.push(q.slice()); q = [q[0] - 0.22 * 8 * q[0], q[1] - 0.22 * 2.4 * q[1]]; } g2.polyline(P, { color: 'b', width: 1.6 }); }
    else { for (let i = 0; i < 40; i++) { const a = rr() * 6.28, cl = i % 2; g2.point([(cl ? 0.6 : -0.6) + 0.35 * Math.cos(a) * rr(), 0.3 * Math.sin(a) * rr() + (cl ? -0.2 : 0.2)], { r: 2.2, color: cl ? 'dataB' : 'data' }); } g2.label('shadow · % variance', [0, -1.05], { size: 10, color: 'mag', font: 'ui' }); }
  }

  /* ---------------- recap ---------------- */
  function renderRecap(g, s) {
    const card = (i, fn, text) => g.viewport({ x: 0.02, y: 0.02 + i * 0.327, w: 0.96, h: 0.31 }, { view: '2d', cam: { center: [0, 0], halfH: 1 }, bg: 'panel' }, (g2, r) => { fn(g2, r); g2.label(text, [r.x + 12, r.y + 16], { screen: true, anchor: 'left', size: 13.5, color: 'text', font: 'ui', bold: true }); });
    const I = info('raw'), w = PATH_R[NSTEP];
    card(0, (g0, r) => {
      const hw = r.w / 2 - 12;
      g.viewport({ x: r.x + 6, y: r.y + 28, w: hw, h: r.h - 34 }, { view: '2d', cam: fit2({ center: [0, 0], halfH: 2.6, halfW: 2.6 }, hw, r.h - 34), border: false }, (g2) => {
        g2.points(XR, { r: 2.2, colors: classColor }); g2.lineThrough([0, 0], [-w[1], w[0]], { color: 'w', width: 2, len: 8 }); g2.vector([0, 0], w, { color: 'w', width: 2, label: 'w' });
      });
      g.viewport({ x: r.x + r.w / 2 + 6, y: r.y + 28, w: hw, h: r.h - 34 }, { view: '2d', cam: fit2({ center: [0, 0], halfH: 3.1, halfW: 3.1 }, hw, r.h - 34), border: false }, (g2) => {
        g2.axes({ labels: false, ticks: false }); g2.contours(lookup('raw'), I.levels, WB, { res: RES, color: 'residual', width: 1, alpha: 0.6 }); g2.point(w, { r: 5, color: 'w' }); g2.label('w', w, { color: 'w', dx: -10, dy: -10, size: 13 });
      });
    }, '1 · boundary ↔ point');
    card(1, (g0, r) => {
      g.viewport({ x: r.x + 6, y: r.y + 26, w: r.w - 12, h: r.h - 30 }, { view: '2d', cam: fit2({ center: [0.4, 0.6], halfH: 2.4, halfW: 2.4 }, r.w - 12, r.h - 30), border: false }, (g2) => {
        g2.contours(lookup('raw'), I.levels, WB, { res: RES, color: 'residual', width: 1, alpha: 0.6 });
        g2.polyline(PATH_R, { color: 'b', width: 2 }); g2.points(PATH_R, { r: 2.2, color: 'b' });
        [I.qmax, I.qmin].forEach(q => g2.line(N.add(I.ws, N.scale(q, -1)), N.add(I.ws, N.scale(q, 1)), { color: 'gold', width: 2 }));
        g2.point(I.ws, { r: 4.5, color: 'gold' });
      });
    }, '2 · descend the bowl the data built');
    card(2, (g0, r) => {
      const n = (s.net || NET0)[4], sh = shadow(n);
      g.viewport({ x: r.x + 6, y: r.y + 26, w: r.w - 12, h: r.h - 30 }, { view: '3d', cam: fit3(Object.assign(camS(), { dist: 6.2 }), r.w - 12, r.h - 30), border: false }, (g3, rr) => {
        g3.points(sh.Z, { r: 2.4, colors: i => (YM[i] ? 'dataB' : 'data') });
        g3.label('shadow of ℝ⁴ · ' + Math.round(sh.varKept * 100) + '% variance', [rr.x + rr.w - 10, rr.y + rr.h / 2], { screen: true, anchor: 'right', size: 12.5, color: 'mag', font: 'ui' });
      });
    }, '3 · shadows, labeled');
  }
})();
