/* p6-05 — The kernel trick lifts 2D into 3D: lift the rings onto a bowl, cut flat, read the cut back as a circle,
   and compute lifted dot products (the Gram matrix) without lifting. */
(function () {
  'use strict';
  const N = LA.num, seg = LA.seg, fmt = LA.fmt, lerp = LA.lerp;
  const PI = Math.PI;
  const vS = '<span class="c-v">a</span>', wS = '<span class="c-w">b</span>';

  // ---- data (synthetic, seeded) ----
  const CIRC = LA.rng.circles(605, 100, { r1: 0.6, r2: 1.4, noise: 0.08 });           // y: 0 inner (blue), 1 outer (rose)
  const XOR = (function () { const b = LA.rng.blobs(606, 100, { centers: [[1, 1], [-1, -1], [1, -1], [-1, 1]], spread: 0.3 }); return { X: b.X, y: b.y.map(c => (c < 2 ? 1 : 0)) }; })();
  const SUB = [0, 2, 4, 6, 8, 10, 1, 3, 5, 7, 9, 11];                                  // 6 inner then 6 outer
  const ALPHA = SUB.map((_, k) => (k < 6 ? -1 / 6 : 1 / 6));                            // inner −, outer + (so w points up)
  const DS = s => (s.preset === 'xor' ? XOR : CIRC);
  const liftZ = (p, preset) => (preset === 'xor' ? p[0] * p[1] : p[0] * p[0] + p[1] * p[1]);
  const phi = (p, preset) => [p[0], p[1], liftZ(p, preset)];
  const dot3 = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const kPoly = (a, b) => a[0] * b[0] + a[1] * b[1] + (a[0] * a[0] + a[1] * a[1]) * (b[0] * b[0] + b[1] * b[1]);
  const kRbf = (a, b, sg) => Math.exp(-((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) / (2 * sg * sg));
  const W_SUM = (function () { let w = [0, 0, 0]; SUB.forEach((i, k) => { w = N.add(w, N.scale(phi(CIRC.X[i], 'circles'), ALPHA[k])); }); return w; })();
  const B_POLY = -W_SUM[2];                                                              // f = 0 on r = 1 (on average)
  const fPoly = x => { let f = B_POLY; SUB.forEach((i, k) => { f += ALPHA[k] * kPoly(CIRC.X[i], x); }); return f; };
  function bRbf(sg) { let m = 0; for (let q = 0; q < 64; q++) { const u = [Math.cos(q / 64 * 2 * PI), Math.sin(q / 64 * 2 * PI)]; SUB.forEach((i, k) => { m += ALPHA[k] * kRbf(CIRC.X[i], u, sg); }); } return -m / 64; }
  const fRbf = (x, sg, b) => { let f = b; SUB.forEach((i, k) => { f += ALPHA[k] * kRbf(CIRC.X[i], x, sg); }); return f; };
  const BR05 = bRbf(0.5);
  function gram(kind, sg, preset) {
    const X = SUB.map(i => DS({ preset }).X[i]);
    return X.map(a => X.map(b => (kind === 'rbf' ? kRbf(a, b, sg) : dot3(phi(a, preset), phi(b, preset)))));
  }
  const K_POLY = gram('poly', 0.5, 'circles'), K_RBF = gram('rbf', 0.5, 'circles');
  // the same 12×12 table cell by cell with the kernel, and the largest difference between the two (computed, not assumed)
  const K_KER = SUB.map(i => SUB.map(j => kPoly(CIRC.X[i], CIRC.X[j])));
  const K_DIFF = Math.max.apply(null, K_POLY.map((row, i) => Math.max.apply(null, row.map((v, j) => Math.abs(v - K_KER[i][j])))));
  // accuracy of the kernel classifiers on all 100 points
  const accOf = f => CIRC.X.filter((p, i) => (f(p) > 0 ? 1 : 0) === CIRC.y[i]).length / CIRC.X.length;
  const ACC_POLY = accOf(fPoly), ACC_RBF = accOf(p => fRbf(p, 0.5, BR05));
  // best accuracy of a line through the origin (misconception step), sampled every degree
  function lineAcc(th) {
    const n = [-Math.sin(th), Math.cos(th)]; let ok = 0;
    CIRC.X.forEach((p, i) => { if ((N.dot(n, p) > 0 ? 1 : 0) === CIRC.y[i]) ok++; });
    return Math.max(ok, 100 - ok) / 100;
  }
  const LINE_ACC = []; for (let d = 0; d <= 180; d++) LINE_ACC.push(lineAcc(d * PI / 180));
  const LINE_BEST = Math.max.apply(null, LINE_ACC);
  // plane ax + by + cz = d
  const side = (pl, p, preset) => pl.a * p[0] + pl.b * p[1] + pl.c * liftZ(p, preset) - pl.d;
  function planeAcc(pl, preset) { const D = DS({ preset }); let ok = 0; D.X.forEach((p, i) => { if ((side(pl, p, preset) > 0 ? 1 : 0) === D.y[i]) ok++; }); return ok / D.X.length; }
  function circleOf(pl) { const c = pl.c || 1e-6; const cx = -pl.a / (2 * c), cy = -pl.b / (2 * c); const r2 = pl.d / c + (pl.a * pl.a + pl.b * pl.b) / (4 * c * c); return { cx, cy, r: Math.sqrt(Math.max(r2, 0)), r2 }; }

  // ---- ramps ----
  const panelRamp = t => (t < 0 ? LA.color.panel2 : LA.color.heat(t));
  const divRamp = t => LA.color.diverge(2 * t - 1);
  // ---- cameras ----
  const cam2 = () => ({ center: [0, 0], halfH: 2.2 });
  const cam3 = o => Object.assign(LA.cam.defaults3d(), { target: [0, 0, 0.9], dist: 8.6, yaw: 0.52, pitch: 0.44, fov: 36 }, o || {});
  const camD = o => cam3(Object.assign({ dist: 6.6, target: [0, 0, 0.8] }, o || {}));    // 3D camera for a half panel
  const camR = () => ({ center: [0.2, 0], halfH: 1.95, halfW: 2.0 });
  const TOPDOWN = { yaw: -PI / 2, pitch: 1.55, target: [0, 0, 0] };
  // ---- adaptive two-panel layout: side by side when the canvas is wide, stacked when it is tall ----
  function fit2(cam, w, h) { const c = Object.assign({}, cam); if (c.halfW) c.halfH = Math.max(c.halfH, c.halfW * h / w); return c; }
  function fit3(cam, w, h) { return Object.assign({}, cam, { dist: cam.dist * Math.max(1, 1.25 / (w / h)) }); }
  function duo(g, fnA, fnB, A, B) {
    const W = g.size.w, H = g.size.h, side = W >= 1.2 * H;
    const camFor = (V, w, h) => (V.view === '3d' ? fit3(V.cam, w, h) : fit2(V.cam, w, h));
    if (side) g.split(fnA, fnB, { left: { view: A.view, cam: camFor(A, W / 2, H) }, right: { view: B.view, cam: camFor(B, W / 2, H) } });
    else {
      g.viewport({ x: 0, y: 0, w: 1, h: 0.5 }, { view: A.view, cam: camFor(A, W, H / 2) }, fnA);
      g.viewport({ x: 0, y: 0.5, w: 1, h: 0.5 }, { view: B.view, cam: camFor(B, W, H / 2) }, fnB);
    }
  }

  LA.registerLesson({
    id: 'p6-05-kernel-trick',
    part: 6,
    title: 'The kernel trick lifts 2D into 3D',
    subtitle: 'Lift the data, cut it with a plane, and the flat cut comes home as a curve.',
    minutes: 12,
    prereqs: ['p2-12-neural-layer', 'p1-09-first-model-neuron', 'p3-07-gram-matrix', 'p2-09-non-square', 'p6-04-softmax-classifier', 'p1-03-linear-combinations-span'],
    tags: ['core', 'capstone'],
    badge: { drawnIn: '2D → 3D exact lift φ(x, y) = (x, y, x² + y²)', realSize: 'the RBF kernel’s feature space is infinite-dimensional; an SVM on 10⁴ points uses a 10⁴×10⁴ Gram matrix', transfers: 'lift + flat cut = curved boundary; K = ΦΦᵀ; f(x) = Σ αᵢ k(xᵢ, x) + b — exact for any lift' },
    cantShow: ['The infinite-dimensional RBF feature space; only its Gram heatmap.', 'Degree-2 polynomial lifts in 2D already have 6 features; we draw the 3-feature version.', 'The max-margin optimization that picks the plane (we place it by hand).'],
    terms: [
      { term: 'feature map', def: 'A rule φ that computes new features from the old ones, here (x, y, x² + y²).' },
      { term: 'lift', def: 'Drawing φ(x) as a point one dimension up.' },
      { term: 'kernel', def: 'A function k(a, b) that equals the lifted dot product φ(a)·φ(b) but is computed directly from a and b.' },
      { term: 'Gram matrix', def: 'The table K of all pairwise kernel values between training points.' },
      { term: 'kernel classifier', def: 'f(x) = Σ αᵢ k(xᵢ, x) + b: a weighted sum of similarities to training points.' },
      { term: 'RBF kernel', def: 'k(a, b) = exp(−‖a − b‖²/2σ²), a kernel whose feature space cannot be drawn.' },
      { term: 'support vectors', def: 'The training points with αᵢ ≠ 0, the ones the boundary actually depends on.' },
    ],
    atlas: ['kernels', 'logistic-regression'],
    thumbnail: 2,

    state() {
      return {
        view: '2d', cam: cam2(), cam2: camR(), scene: 'line', panel: 'line',
        lineAngle: 0, lift: 0, showPlane: false, showCut: false, showW: false, planeFade: 0,
        plane: { a: 0, b: 0, c: 1, d: 1 }, kernel: 'poly', sigma: 0.5, preset: 'circles',
        fill: 0, hover: false, chain: 0, field: 0, morph: 0, dotPhase: 0, guessK: [0],
      };
    },

    render(g, s, ctx) {
      switch (s.scene) {
        case 'line': return renderLine(g, s);
        case 'bowl': return renderBowl(g, s, ctx);
        case 'tilt': case 'play': return renderTilt(g, s, ctx);
        case 'dot': case 'entry': return renderDot(g, s);
        case 'gram': return renderGram(g, s, ctx);
        case 'score': return renderScore(g, s);
        case 'rbf': return renderRbf(g, s);
        case 'xor': return renderXor(g, s);
        case 'ml': return renderML(g, s);
        case 'recap': return renderRecap(g, s);
      }
    },

    math(s, tex) {
      const H = str => tex.text(str, { html: true });
      const pl = s.plane;
      switch (s.panel) {
        case 'line': { const k = Math.round(s.lineAngle * 180 / PI); const best = Math.max.apply(null, LINE_ACC.slice(0, k + 1)); return tex.block(tex.eq(tex.text('line accuracy'), tex.op('='), tex.num(LINE_ACC[k], { color: 'gold' })), tex.eq(tex.text('best so far'), tex.op('='), tex.num(best, { color: 'bad' }))); }
        case 'phi': return tex.block(tex.eq(tex.sym('φ'), H('(<span class="c-i">x</span>, <span class="c-j">y</span>)'), tex.op('='), H('(<span class="c-i">x</span>, <span class="c-j">y</span>, <span class="c-k">x² + y²</span>)')));
        case 'cut': return tex.block(tex.eq(tex.sym('w', { color: 'gold' }), tex.op('·'), tex.sym('φ'), H('(x)'), tex.op('+'), tex.sym('b'), tex.op('='), H('z − 1')), tex.eq(tex.sym('w', { color: 'gold' }), tex.op('='), tex.vec([0, 0, 1], { color: 'gold' }), tex.op(','), tex.sym('b'), tex.op('='), tex.text('−1')), tex.eq(tex.text('accuracy'), tex.op('='), tex.text(fmt.pct(planeAcc(pl, 'circles')), { color: 'ok' })));
        case 'circle': return tex.block(tex.eq(H('x² + y²'), tex.op('='), tex.num(1, { color: 'teal' })));
        case 'tilt': case 'play': {
          const c = circleOf(pl);
          const eqs = [tex.eq(tex.num(pl.a), H('x +'), tex.num(pl.b), H('y +'), tex.num(pl.c), H('z ='), tex.num(pl.d, { color: 'gold' }))];
          if (s.preset === 'xor') eqs.push(tex.eq(tex.text('boundary'), tex.num(pl.a), H('x +'), tex.num(pl.b), H('y +'), tex.num(pl.c), H('xy ='), tex.num(pl.d)));
          else eqs.push(tex.eq(tex.text('center'), H('(−a/2c, −b/2c) ='), tex.paren(tex.num(c.cx, { color: 'teal' }) + tex.op(',') + tex.num(c.cy, { color: 'teal' })), tex.op(','), tex.text('radius'), tex.num(c.r, { color: 'teal' })));
          eqs.push(tex.eq(tex.text('accuracy'), tex.op('='), tex.num(planeAcc(pl, s.preset), { color: planeAcc(pl, s.preset) === 1 ? 'ok' : 'text' })));
          return tex.block.apply(null, eqs);
        }
        case 'dot': {
          const a = [1, 0], b = [0.6, 0.8], pa = phi(a), pb = phi(b);
          const L1 = tex.eq(tex.sym('φ(a)', { color: 'v' }), tex.op('·'), tex.sym('φ(b)', { color: 'w' }), tex.op('='), tex.num(pa[0] * pb[0] + pa[1] * pb[1], { color: 'i' }), tex.op('+'), tex.num(pa[2] * pb[2], { color: 'k' }), tex.op('='), tex.num(dot3(pa, pb), { color: 'gold' }));
          const L2 = tex.eq(tex.sym('a', { color: 'v' }), tex.op('·'), tex.sym('b', { color: 'w' }), tex.op('+'), H('‖a‖²‖b‖²'), tex.op('='), tex.num(N.dot(a, b), { color: 'i' }), tex.op('+'), tex.num(1, { color: 'k' }), tex.op('='), tex.num(kPoly(a, b), { color: 'gold' }));
          return s.dotPhase < 0.5 ? tex.lines(L1) : tex.lines(L1, L2);
        }
        case 'gram': return tex.block(H('K<sub>ij</sub> = φ(x<sub>i</sub>)·φ(x<sub>j</sub>) = k(x<sub>i</sub>, x<sub>j</sub>)'), tex.eq(tex.text('largest difference'), tex.op('='), tex.num(K_DIFF)));
        case 'score': return tex.block(tex.eq(tex.sym('w', { color: 'gold' }), tex.op('='), H('Σ<sub>i</sub> α<sub>i</sub> φ(x<sub>i</sub>)'), tex.op('='), tex.vec(W_SUM, { color: 'gold' })), tex.eq(tex.sym('f'), H('(x) = Σ<sub>i</sub> α<sub>i</sub> k(x<sub>i</sub>, x) + b')));
        case 'rbf': return tex.block(tex.eq(H('k(a, b) = exp(−‖a − b‖²/2σ²)'), tex.op(','), tex.sym('σ'), tex.op('='), tex.num(0.5)));
        case 'xor': return tex.block(tex.eq(tex.sym('φ'), H('(x, y) = (x, y, xy)')), tex.eq(tex.text('boundary'), H('xy = 0')), tex.eq(tex.text('accuracy'), tex.op('='), tex.num(planeAcc({ a: 0, b: 0, c: 1, d: 0 }, 'xor'), { color: 'ok' })));
        case 'ml': return tex.block(H('f(x) = Σ α<sub>i</sub> k(x<sub>i</sub>, x) + b'), tex.text('·'), H('learned lift: φ = tanh(Wx + b)'));
        case 'entry': return tex.block(tex.eq(tex.text('your φ(a)·φ(b)'), tex.op('='), tex.num(s.guessK[0], { color: 'b' })), tex.eq(tex.sym('φ(a)', { color: 'v' }), tex.op('='), tex.vec([1, 0, 1], { color: 'v' }), tex.sym('φ(b)', { color: 'w' }), tex.op('='), tex.vec([1, 1, 2], { color: 'w' })));
        default: return '';
      }
    },

    steps: [
      { title: 'A line will do it eventually', duration: 4, misconception: true,
        narration: 'Blue points inside, rose points outside. “Train a line long enough and it will separate them.” Watch a gold line through the center sweep through every angle: at best it gets about ' + Math.round(LINE_BEST * 100) + '% right, barely better than a coin. No line separates a ring from its center. The arrows of p6-04 cannot bend.',
        setup(s) { s.scene = 'line'; s.panel = 'line'; s.view = '2d'; s.cam = cam2(); s.lineAngle = 0; },
        at(s, t) { s.lineAngle = PI * seg(t, 0, 0.92, 'linear'); } },

      { title: 'Compute a new feature', duration: 4,
        narration: 'Add a feature you compute yourself: z = x² + y², the squared distance from the center. Each point rises to its own height; the flat plane of data becomes a bowl. This is a <b>lift</b> (p2-09), but a curved one, given by a <b>feature map</b> φ.',
        setup(s) { s.scene = 'bowl'; s.panel = 'phi'; s.view = '3d'; s.cam = cam3({ yaw: 0.2, pitch: 0.3 }); s.lift = 0; s.showPlane = false; s.showCut = false; s.showW = false; s.planeFade = 0; s.plane = { a: 0, b: 0, c: 1, d: 1 }; },
        at(s, t) { s.lift = seg(t, 0.1, 0.8); LA.cam.tween(s, { yaw: 0.2, pitch: 0.3 }, { yaw: 0.52, pitch: 0.44 }, seg(t, 0, 1)); } },

      { title: 'A flat cut works', duration: 3,
        narration: 'Now a plane separates them: z = 1, gold. Inner points sit below (their r² ≈ 0.36), outer points above (≈ 1.96). A plane in the lifted space is just p1-09 with three features: <span class="c-gold">w</span> = (0, 0, 1), b = −1.',
        setup(s) { s.scene = 'bowl'; s.panel = 'cut'; s.view = '3d'; s.lift = 1; s.showPlane = true; s.showW = false; s.planeFade = 0; },
        at(s, t) { s.planeFade = seg(t, 0, 0.5); s.showW = t > 0.5; } },

      { title: 'The cut, seen from below', duration: 3.5,
        narration: 'Where the plane meets the bowl is a circle, drawn teal. Look straight down: in the original 2D plane the boundary is the circle x² + y² = 1. A straight cut up there is a curve down here — that is the whole trick.',
        setup(s) { s.scene = 'bowl'; s.panel = 'circle'; s.lift = 1; s.showPlane = true; s.planeFade = 1; s.showW = false; s.showCut = true; s._c0 = { yaw: s.cam.yaw, pitch: s.cam.pitch, target: s.cam.target.slice() }; },
        at(s, t) {
          if (t < 0.6) { s.view = '3d'; LA.cam.tween(s, s._c0, TOPDOWN, seg(t, 0, 0.6)); s.planeFade = 1 - 0.7 * seg(t, 0.2, 0.6); }
          else { s.view = '2d'; s.cam.center = [0, 0]; s.cam.halfH = 2.4; s.planeFade = 0.3; }
        } },

      { title: 'Tilt the plane', duration: 4, interactive: true,
        narration: 'Tilt the plane: ax + by + cz = d. Its cut moves: still a circle, now off-center, with center (−a/2c, −b/2c). Every plane in the lifted space is some circle in the data plane. Drag the teal center or the rim on the right and watch both pictures follow.',
        setup(s) { s.scene = 'tilt'; s.panel = 'tilt'; s.view = '3d'; s.cam = camD(); s.cam2 = camR(); s.preset = 'circles'; s.lift = 1; s.plane = { a: 0, b: 0, c: 1, d: 1 }; },
        at(s, t) { s.plane.a = -seg(t, 0.1, 0.9); } },

      { title: 'Dot products up there, computed down here', duration: 4,
        narration: 'Take two points ' + vS + ' and ' + wS + '. Their lifted dot product is φ(a)·φ(b) = a·b + ‖a‖²‖b‖². You can lift both and multiply, or compute it straight from a and b without lifting. Same number. A formula that does this is a <b>kernel</b>.',
        setup(s) { s.scene = 'dot'; s.panel = 'dot'; s.view = '3d'; s.cam = camD({ target: [0.4, 0.3, 0.6], dist: 5.6 }); s.cam2 = { center: [0.5, 0.45], halfH: 1.1, halfW: 1.3 }; s.dotPhase = 0; },
        at(s, t) { s.dotPhase = seg(t, 0.15, 0.9, 'linear'); } },

      { title: 'The Gram matrix, two ways', duration: 4,
        narration: 'For twelve points, all pairwise lifted dot products fill a 12×12 <b>Gram matrix</b> (p3-07): K = ΦΦᵀ. Compute it by lifting, or cell by cell with the kernel: the two heatmaps are identical. Hover a cell (in Play) to see its pair; here cell (2, 9) lights up.',
        setup(s) { s.scene = 'gram'; s.panel = 'gram'; s.view = '2d'; s.cam = { center: [0, 0], halfH: 3.6 }; s.fill = 0; s.hover = false; },
        at(s, t) { s.fill = seg(t, 0, 0.75, 'linear'); s.hover = t > 0.8; } },

      { title: 'A classifier that only needs K', duration: 4.5,
        narration: 'Build the gold normal from the lifted points (p1-03): w = Σ αᵢφ(xᵢ), outer points +, inner −. So a new point scores f(x) = Σ αᵢ k(xᵢ, x) + b, a <b>kernel classifier</b>: a weighted sum of similarities. Only kernel values appear; the lift is never visited.',
        setup(s) { s.scene = 'score'; s.panel = 'score'; s.view = '3d'; s.cam = camD({ target: [0, 0, 0.7], dist: 5.8 }); s.cam2 = camR(); s.chain = 0; s.field = 0; },
        at(s, t) { s.chain = seg(t, 0.05, 0.6, 'linear'); s.field = seg(t, 0.55, 0.95); } },

      { title: 'A kernel you cannot lift', duration: 4,
        narration: 'The <b>RBF kernel</b> k(a, b) = exp(−‖a − b‖²/2σ²) is also a dot product, in a space with infinitely many features. Nobody can draw that space, but its Gram matrix is an ordinary 12×12 heatmap, and the classifier works unchanged. The picture stops at 3. The formula does not.',
        setup(s) { s.scene = 'rbf'; s.panel = 'rbf'; s.view = '2d'; s.cam = { center: [0, 0.3], halfH: 2.9 }; s.cam2 = camR(); s.morph = 0; },
        at(s, t) { s.morph = seg(t, 0.15, 0.75); } },

      { title: 'A second lift: XOR', duration: 3.5,
        narration: 'Four blobs in a checkerboard: no line, again. Lift with φ = (x, y, xy): the flat plane bends into a saddle (p4-06) and the horizontal gold plane z = 0 separates the quadrants. Different data, different lift, same trick.',
        setup(s) { s.scene = 'xor'; s.panel = 'xor'; s.view = '3d'; s.cam = cam3({ target: [0, 0, 0], dist: 11, yaw: 0.75, pitch: 0.42 }); s.lift = 0; },
        at(s, t) { s.lift = seg(t, 0.1, 0.7); LA.cam.tween(s, { yaw: 0.75 }, { yaw: 1.05 }, seg(t, 0, 1, 'linear')); } },

      { title: 'Why ML cares: lifts everywhere', duration: 4, mlTie: true,
        narration: 'An SVM is this plane chosen with the widest margin; the points on the margin are its <b>support vectors</b>, and it needs only K. Attention’s QKᵀ (p6-03) is a table of dot products too. A neural network learns its own lift (p2-12) where a kernel fixes it in advance. p6-08 trains the learned one.',
        setup(s) { s.scene = 'ml'; s.panel = 'ml'; s.view = '2d'; s.cam = cam2(); },
        at(s, t) { } },

      { title: 'Recap', duration: 2, recap: true,
        narration: 'Lift the data with a feature map, cut it with a plane, and the cut is a curve back home. Dot products in the lifted space are kernels you can compute without lifting, so the Gram matrix is all a kernel method needs — even when the lift is infinite.',
        cards: ['<b>Lift, then cut flat.</b> z = x² + y² turns rings into heights; the plane z = 1 separates them.', '<b>The cut comes home as a curve.</b> Plane ∩ bowl = the circle x² + y² = 1 in the data plane.', '<b>Kernel = lifted dot product without lifting.</b> K = ΦΦᵀ is all f(x) = Σ αᵢ k(xᵢ, x) + b needs.'],
        setup(s) { s.scene = 'recap'; s.panel = 'none'; s.view = '2d'; s.cam = cam2(); },
        at(s, t) { } },
    ],

    playground: {
      intro: '<p>Tilt and raise the gold plane with the sliders, or drag the teal center and rim of the 2D boundary. Switch the data, and the kernel that fills the Gram inset.</p>',
      setup(s) { s.scene = 'play'; s.panel = 'play'; s.view = '3d'; s.cam = camD(); s.cam2 = camR(); s.lift = 1; s.preset = 'circles'; s.plane = { a: 0, b: 0, c: 1, d: 1 }; s.kernel = 'poly'; s.sigma = 0.5; },
      controls: [
        { type: 'slider', bind: 'plane.a', min: -2, max: 2, step: 0.01, label: 'Tilt x (a)', color: 'gold' },
        { type: 'slider', bind: 'plane.b', min: -2, max: 2, step: 0.01, label: 'Tilt y (b)', color: 'gold' },
        { type: 'slider', bind: 'plane.d', min: -1, max: 2.5, step: 0.01, label: 'Height d', color: 'gold' },
        { type: 'preset', label: 'Data', items: [
          { label: 'Flat cut', apply(s) { s.preset = 'circles'; s.plane = { a: 0, b: 0, c: 1, d: 1 }; } },
          { label: 'Tilted cut', apply(s) { s.preset = 'circles'; s.plane = { a: -1, b: 0.4, c: 1, d: 1 }; } },
          { label: 'XOR saddle', apply(s) { s.preset = 'xor'; s.plane = { a: 0, b: 0, c: 1, d: 0 }; } },
        ] },
        { type: 'preset', label: 'Kernel for K', items: [
          { label: 'lifted dot (poly)', values: { kernel: 'poly' } },
          { label: 'RBF', values: { kernel: 'rbf' } },
        ] },
        { type: 'slider', bind: 'sigma', min: 0.2, max: 1.5, step: 0.01, label: 'RBF σ' },
      ],
    },

    checks: [
      { kind: 'manipulate',
        prompt: 'Set the plane height so it separates the two rings perfectly (drag the gold handle in 3D, or the teal rim in 2D).',
        setup(s) { s.scene = 'play'; s.panel = 'play'; s.view = '3d'; s.cam = camD(); s.cam2 = camR(); s.lift = 1; s.preset = 'circles'; s.plane = { a: 0, b: 0, c: 1, d: 0.2 }; },
        judge(s) { const acc = planeAcc(s.plane, 'circles'); return { ok: acc === 1, progress: Math.max(0, (acc - 0.5) * 2) }; },
        hint: 'Inner heights are about 0.36, outer about 1.96 — go between.',
        solution(s) { s.plane = { a: 0, b: 0, c: 1, d: 1 }; } },
      { kind: 'manipulate',
        prompt: 'Tilt the plane so the 2D boundary is a circle centered at (0.5, 0). Drag the teal center handle.',
        setup(s) { s.scene = 'play'; s.panel = 'play'; s.view = '3d'; s.cam = camD(); s.cam2 = camR(); s.lift = 1; s.preset = 'circles'; s.plane = { a: 0, b: 0, c: 1, d: 1 }; },
        judge(s) { const c = circleOf(s.plane); return LA.judge.near([c.cx, c.cy], [0.5, 0], LA.TOL.pos); },
        hint: 'The center is (−a/2c, −b/2c): tilt along x only, a = −1.',
        solution(s) { s.plane = { a: -1, b: 0, c: 1, d: 1 }; } },
      { kind: 'entry',
        prompt: 'Type φ(a)·φ(b) for <span class="c-v">a</span> = (1, 0) and <span class="c-w">b</span> = (1, 1) with φ = (x, y, x² + y²). Your number is drawn as the orange bar.',
        bind: 'guessK', answer: [3], tol: LA.TOL.entry, label: 'φ(a)·φ(b)', colors: 'b',
        hint: 'a·b = 1; ‖a‖² = 1, ‖b‖² = 2.',
        setup(s) { s.scene = 'entry'; s.panel = 'entry'; s.view = '3d'; s.cam = camD({ target: [0.4, 0.3, 0.8], dist: 6 }); s.cam2 = { center: [0.7, 0.5], halfH: 1.2, halfW: 1.5 }; s.guessK = [0]; s.dotPhase = 1; } },
    ],
  });

  /* ---------------- drawing helpers ---------------- */
  function pts2d(g, s, D, opts) {
    opts = opts || {};
    g.points(D.X, { r: opts.r || 4, colors: i => (D.y[i] ? 'dataB' : 'data'), alpha: opts.alpha === undefined ? 0.95 : opts.alpha });
  }
  function floor2d(g, range) { g.grid(null, { range: range || 3, step: 0.5, alpha: 0.35 }); g.axes({ labels: false, ticks: false }); }
  function bowlSheet(g, preset, alpha) {
    const wa = alpha === undefined ? 0.5 : alpha;
    if (preset === 'xor') g.paramSurface((u, v) => [u, v, u * v], { u0: -1.6, u1: 1.6, v0: -1.6, v1: 1.6 }, { res: 16, mode: 'wire', color: 'grid', wireAlpha: wa });
    else g.paramSurface((u, v) => { const x = u * Math.cos(v), y = u * Math.sin(v); return [x, y, x * x + y * y]; }, { u0: 0, u1: 1.6, v0: 0, v1: 2 * PI }, { res: 16, mode: 'wire', color: 'grid', wireAlpha: wa });
  }
  function lifted3d(g, D, preset, lift, sticks) {
    const P = D.X.map(p => [p[0], p[1], lift * liftZ(p, preset)]);
    if (sticks) P.forEach(p => { if (Math.abs(p[2]) > 0.03) g.line([p[0], p[1], 0], p, { color: 'residual', dashed: [3, 4], width: 1, alpha: 0.45 }); });
    g.points(P, { r: 3.6, colors: i => (D.y[i] ? 'dataB' : 'data') });
    return P;
  }
  function planeOf(g, pl, alpha, extent) {
    const c = circleOf(pl);
    const cx = LA.clamp(c.cx, -1.5, 1.5), cy = LA.clamp(c.cy, -1.5, 1.5);
    const z = (pl.d - pl.a * cx - pl.b * cy) / (pl.c || 1e-6);
    g.planes([{ point: [cx, cy, z], normal: [pl.a, pl.b, pl.c], extent: extent || 1.8, color: 'gold', alpha }]);
    return [cx, cy, z];
  }
  function cutCurve3d(g, pl, preset, opts) {
    opts = opts || {};
    if (preset === 'xor') { hyperbola(pl).forEach(br => g.polyline(br.map(p => [p[0], p[1], p[0] * p[1]]), { color: 'teal', width: opts.width || 2.6, segmented: true })); return; }
    const c = circleOf(pl); if (c.r < 1e-3) return;
    const pts = []; for (let i = 0; i <= 64; i++) { const a = i / 64 * 2 * PI, x = c.cx + c.r * Math.cos(a), y = c.cy + c.r * Math.sin(a); pts.push([x, y, x * x + y * y]); }
    g.polyline(pts, { color: 'teal', width: opts.width || 2.6 });
  }
  function boundary2d(g, pl, preset, opts) {
    opts = opts || {};
    if (preset === 'xor') { hyperbola(pl).forEach(br => g.polyline(br, { color: 'teal', width: 2.4 })); return null; }
    const c = circleOf(pl); g.circle([c.cx, c.cy], c.r, { stroke: 'teal', width: opts.width || 2.4, segments: 96 });
    return c;
  }
  /** x·(a) + b·y + c·xy = d  ⇔  (c x + b)(c y + a) = c d + a b; two branches clipped to |coord| ≤ 2. */
  function hyperbola(pl) {
    const out = []; const c = pl.c || 1e-6;
    let cur = [];
    for (let i = 0; i <= 160; i++) {
      const x = -2 + 4 * i / 160, den = pl.b + c * x;
      const y = Math.abs(den) < 1e-6 ? NaN : (pl.d - pl.a * x) / den;
      if (Number.isFinite(y) && Math.abs(y) <= 2) cur.push([x, y]); else { if (cur.length > 1) out.push(cur); cur = []; }
    }
    if (cur.length > 1) out.push(cur);
    return out;
  }
  function heatCells(g, K, b, opts) {
    opts = opts || {};
    const n = K.length, lo = opts.range[0], hi = opts.range[1];
    const filled = opts.fill === undefined ? n * n : opts.fill;
    const grid = K.map((row, i) => row.map((v, j) => (i * n + j < filled ? v : lo - (hi - lo))));
    g.heat2d(grid, b, { range: opts.range, ramp: opts.ramp || panelRamp, smooth: false, alpha: opts.alpha === undefined ? 1 : opts.alpha });
    g.polyline([[b.x0, b.y0], [b.x1, b.y0], [b.x1, b.y1], [b.x0, b.y1]], { color: 'line', width: 1, closed: true });
  }
  function cellRect(b, n, i, j) { const cw = (b.x1 - b.x0) / n, ch = (b.y1 - b.y0) / n; const x0 = b.x0 + j * cw, y1 = b.y1 - i * ch; return [[x0, y1 - ch], [x0 + cw, y1 - ch], [x0 + cw, y1], [x0, y1]]; }
  function rangeOf(K) { let lo = Infinity, hi = -Infinity; K.forEach(r => r.forEach(v => { lo = Math.min(lo, v); hi = Math.max(hi, v); })); return [lo, hi]; }
  function title(g, r, text, opts) { g.label(text, [r.x + 12, r.y + 18], Object.assign({ screen: true, anchor: 'left', size: 13.5, color: 'text', font: 'ui', bold: true }, opts || {})); }

  /* ---------------- scenes ---------------- */
  function renderLine(g, s) {
    floor2d(g);
    const th = s.lineAngle, n = [-Math.sin(th), Math.cos(th)], d = [Math.cos(th), Math.sin(th)];
    let ok = 0; CIRC.X.forEach((p, i) => { if ((N.dot(n, p) > 0 ? 1 : 0) === CIRC.y[i]) ok++; });
    const flip = ok < 50;                                           // use the better labeling of the two sides
    pts2d(g, s, CIRC);
    CIRC.X.forEach((p, i) => { const pred = (N.dot(n, p) > 0 ? 1 : 0) ^ (flip ? 1 : 0); if (pred !== CIRC.y[i]) g.point(p, { r: 6.5, color: 'rgba(0,0,0,0)', stroke: 'bad', strokeWidth: 1.4, alpha: 0.85 }); });
    g.lineThrough([0, 0], d, { color: 'gold', width: 2.6, len: 6 });
    const k = Math.round(th * 180 / PI), best = Math.max.apply(null, LINE_ACC.slice(0, k + 1));
    const S = g.size;
    g.hud('accuracy ' + fmt.pct(LINE_ACC[k]) + '   ·   best so far ' + fmt.pct(best), 16, 24, { color: 'hud', size: 14 });
    g.hud('misclassified points are ringed', 16, 46, { color: 'muted', size: 12 });
    g.label('“a line will separate them”', [S.w - 40, 30], { screen: true, anchor: 'right', size: 15, color: 'bad', font: 'ui' });
    g.marker.cross([S.w - 24, 30], { screen: true, size: 20 });
  }

  function renderBowl(g, s) {
    if (s.view === '2d') {   // seen from straight above: the data plane with the teal circle
      floor2d(g);
      pts2d(g, s, CIRC);
      if (s.showCut) boundary2d(g, s.plane, 'circles', { width: 2.8 });
      g.label('x² + y² = 1', [0.75, 0.95], { color: 'teal', size: 14 });
      g.hud('the plane’s cut, seen from straight above: a circle', 16, 24, { color: 'hud' });
      return;
    }
    g.grid(null, { range: 2, step: 0.5, alpha: 0.4 });
    g.axes({ length: 2.2, names: ['x', 'y', 'z'] });
    bowlSheet(g, 'circles', 0.35 * Math.max(0.15, s.lift));
    lifted3d(g, CIRC, 'circles', s.lift, true);
    if (s.showPlane && s.planeFade > 0.01) planeOf(g, s.plane, 0.3 * s.planeFade);
    if (s.showCut) cutCurve3d(g, s.plane, 'circles');
    if (s.showW) { g.vector([0, 0, 1], [0, 0, 2.1], { color: 'gold', width: 3, label: 'w' }); }
    if (s.showCut) g.hud('plane ∩ bowl: the teal circle at height z = 1', 16, 24, { color: 'teal' });
    else if (s.showPlane && s.planeFade > 0.5) g.hud('accuracy ' + fmt.pct(planeAcc(s.plane, 'circles')), 16, 24, { color: 'ok', size: 14 });
    else g.hud('z = x² + y²: each point rises to its squared distance', 16, 24, { color: 'hud' });
  }

  function renderTilt(g, s, ctx) {
    const pl = s.plane, preset = s.preset;
    const D = DS(s);
    let P0 = null;
    duo(g, (g3, r) => {
      g3.grid(null, { range: 2, step: 0.5, alpha: 0.35 });
      g3.axes({ length: 2.1, names: ['x', 'y', 'z'] });
      bowlSheet(g3, preset, 0.3);
      lifted3d(g3, D, preset, 1, false);
      P0 = planeOf(g3, pl, 0.28);
      cutCurve3d(g3, pl, preset);
      title(g3, r, 'lifted: plane ∩ ' + (preset === 'xor' ? 'saddle' : 'bowl'));
      g3.handle('height', P0, (p, st) => { st.plane.d = LA.clamp(st.plane.a * P0[0] + st.plane.b * P0[1] + st.plane.c * p[2], -1, 4); }, { color: 'gold', plane: 'view' });
    }, (g2, r) => {
      floor2d(g2, 4);
      pts2d(g2, s, D, { r: 3.5 });
      const c = boundary2d(g2, pl, preset);
      const acc = planeAcc(pl, preset);
      title(g2, r, 'data plane: the cut comes home');
      g2.label('accuracy ' + fmt.pct(acc), [r.x + 12, r.y + 40], { screen: true, anchor: 'left', size: 13, color: acc === 1 ? 'ok' : 'hud', font: 'ui' });
      if (c) {
        g2.point([c.cx, c.cy], { r: 4, color: 'teal' });
        g2.handle('center', [c.cx, c.cy], (p, st) => { const c0 = circleOf(st.plane); st.plane.a = -2 * st.plane.c * p[0]; st.plane.b = -2 * st.plane.c * p[1]; st.plane.d = st.plane.c * (c0.r2 - (st.plane.a ** 2 + st.plane.b ** 2) / (4 * st.plane.c ** 2)); }, { color: 'teal' });
        g2.handle('rim', [c.cx + c.r, c.cy], (p, st) => { const c0 = circleOf(st.plane); const rr = Math.max(0.05, Math.hypot(p[0] - c0.cx, p[1] - c0.cy)); st.plane.d = st.plane.c * (rr * rr - (st.plane.a ** 2 + st.plane.b ** 2) / (4 * st.plane.c ** 2)); }, { color: 'teal' });
      }
      if (s.scene === 'play') {   // Gram inset with the chosen kernel
        const K = gram(s.kernel, s.sigma, preset);
        const P = g2.P, w = Math.min(110, r.h * 0.4);
        const tl = P.toWorld(r.x + r.w - w - 12, r.y + r.h - w - 30), br = P.toWorld(r.x + r.w - 12, r.y + r.h - 30);
        heatCells(g2, K, { x0: tl[0], x1: br[0], y0: br[1], y1: tl[1] }, { range: rangeOf(K) });
        g2.label((s.kernel === 'rbf' ? 'RBF K, σ = ' + fmt(s.sigma) : 'K = ΦΦᵀ') + ' (12 points)', [r.x + r.w - 12, r.y + r.h - 16], { screen: true, anchor: 'right', size: 11.5, color: 'muted', font: 'ui' });
      }
    }, { view: '3d', cam: s.cam }, { view: '2d', cam: s.cam2 });
  }

  function renderDot(g, s) {
    const entry = s.scene === 'entry';
    const a = [1, 0], b = entry ? [1, 1] : [0.6, 0.8];
    const pa = phi(a), pb = phi(b);
    const u = s.dotPhase;
    duo(g, (g3, r) => {
      g3.grid(null, { range: 2, step: 0.5, alpha: 0.35 });
      g3.axes({ length: 2.2, names: ['x', 'y', 'z'] });
      bowlSheet(g3, 'circles', 0.22);
      g3.vector([0, 0, 0], pa, { color: 'v', width: 3, label: 'φ(a)' });
      g3.vector([0, 0, 0], pb, { color: 'w', width: 3, label: 'φ(b)' });
      g3.line(a, pa, { color: 'residual', dashed: true, width: 1 }); g3.line(b, pb, { color: 'residual', dashed: true, width: 1 });
      g3.point(a, { r: 4, color: 'v' }); g3.point(b, { r: 4, color: 'w' });
      if (entry) { g3.label('(1, 0, 1)', pa, { color: 'v', size: 12.5, dy: 18 }); g3.label('(1, 1, 2)', pb, { color: 'w', size: 12.5, dy: -20 }); }
      title(g3, r, 'lift both, then multiply');
      if (!entry && u > 0.3) g3.label('φ(a)·φ(b) = 0.6 + 1 = ' + fmt(dot3(pa, pb)), [r.x + 12, r.y + 42], { screen: true, anchor: 'left', size: 13.5, color: 'gold', font: 'ui' });
    }, (g2, r) => {
      g2.grid(null, { range: 3, step: 0.5, alpha: 0.35 }); g2.axes({ labels: false, ticks: false });
      g2.vector(a, { color: 'v', width: 3, label: 'a' });
      g2.vector(b, { color: 'w', width: 3, label: 'b' });
      title(g2, r, 'stay down here: use the kernel');
      if (!entry && u > 0.6) {
        g2.label('a·b + ‖a‖²‖b‖² = 0.6 + 1 = ' + fmt(kPoly(a, b)), [r.x + 12, r.y + 42], { screen: true, anchor: 'left', size: 13.5, color: 'gold', font: 'ui' });
        if (u > 0.85) { g2.label('same number', [r.x + 34, r.y + 66], { screen: true, anchor: 'left', size: 13, color: 'ok', font: 'ui' }); g2.marker.check([r.x + 20, r.y + 66], { screen: true, size: 16 }); }
      }
      if (entry) {
        const v = s.guessK[0] || 0, ok = Math.abs(v - 3) <= LA.TOL.entry;
        g2.bars([Math.max(0, v)], { screen: true, x: r.x + r.w - 80, y: r.y + 70, w: 40, h: 150, max: 5, colors: [ok ? 'gold' : 'b'], values: true, gap: 0 });
        g2.label('your φ(a)·φ(b)', [r.x + r.w - 60, r.y + 238], { screen: true, size: 11.5, color: 'muted', font: 'ui' });
        g2.label('scale 0 … 5', [r.x + r.w - 60, r.y + 56], { screen: true, size: 11, color: 'muted', font: 'ui' });
        if (ok) g2.marker.check([r.x + r.w - 60, r.y + 258], { screen: true, size: 18 });
      }
    }, { view: '3d', cam: s.cam }, { view: '2d', cam: s.cam2 });
  }

  function renderGram(g, s, ctx) {
    const n = 12, B = { x0: -1.8, x1: 1.8, y0: -1.8, y1: 1.8 };
    const rg = rangeOf(K_POLY), fill = Math.round(s.fill * n * n);
    const K2 = K_KER;
    // which cell is highlighted: scripted (2, 9) or the hovered one in Play
    let hi = s.hover ? [1, 8] : null;
    if (ctx.hot && /^cell-/.test(ctx.hot)) { const m = ctx.hot.split('-'); hi = [+m[1], +m[2]]; }
    const W = g.size.w, H = g.size.h, topH = 0.58;
    const camH = fit2({ center: [0, 0], halfH: 2.35, halfW: 2.15 }, W / 2, H * topH);
    const half = (x, K, text, foot, footColor, handles) => g.viewport({ x, y: 0, w: 0.5, h: topH }, { view: '2d', cam: camH, border: false }, (g2) => {
      heatCells(g2, K, B, { range: rg, fill });
      g2.label(text, [0, B.y1], { anchor: 'above', dy: -6, size: 13.5, color: 'text', font: 'ui' });
      g2.label(foot, [0, B.y0], { anchor: 'below', dy: 6, size: 11.5, color: footColor, font: 'ui' });
      if (handles) [0, 6].forEach(i => g2.label(String(i + 1), [B.x0, B.y1 - (i + 0.5) * 3.6 / n], { anchor: 'right', dx: -4, size: 10.5, color: 'muted', font: 'ui' }));
      if (hi) g2.polyline(cellRect(B, n, hi[0], hi[1]), { color: 'gold', width: 2.2, closed: true });
      if (handles && ctx.mode !== 'learn') for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) { const q = cellRect(B, n, i, j); g2.handle('cell-' + i + '-' + j, [(q[0][0] + q[1][0]) / 2, (q[0][1] + q[2][1]) / 2], () => { }, { show: false, r: 8 }); }
    });
    half(0, K_POLY, 'ΦΦᵀ: lift, then dot', 'rows: inner 1–6, outer 7–12', 'muted', true);
    half(0.5, K2, 'k(xᵢ, xⱼ): never lift', 'largest difference: ' + fmt.fixed(K_DIFF, 2), K_DIFF < 0.005 ? 'ok' : 'bad', false);
    // the twelve sample points below, with the highlighted pair
    g.viewport({ x: 0.04, y: topH + 0.01, w: 0.92, h: 0.98 - topH - 0.01 }, { view: '2d', cam: { center: [0, 0], halfH: 1.75 }, bg: 'panel' }, (g2, r) => {
      g2.circle([0, 0], 1, { stroke: 'teal', width: 1, dashed: true, strokeAlpha: 0.4 });
      SUB.forEach((idx, k) => { const p = CIRC.X[idx]; g2.point(p, { r: 4.5, color: k < 6 ? 'data' : 'dataB' }); g2.label(String(k + 1), p, { dx: 10, dy: -9, size: 11, color: 'muted', font: 'ui' }); });
      if (hi) {
        const p = CIRC.X[SUB[hi[0]]], q = CIRC.X[SUB[hi[1]]];
        g2.point(p, { r: 7, color: 'rgba(0,0,0,0)', stroke: 'gold' }); g2.point(q, { r: 7, color: 'rgba(0,0,0,0)', stroke: 'gold' });
        g2.arc([0, 0], p, q, { r: 0.45, color: 'gold' });
        g2.line([0, 0], p, { color: 'gold', width: 1, dashed: true, alpha: 0.6 }); g2.line([0, 0], q, { color: 'gold', width: 1, dashed: true, alpha: 0.6 });
        g2.label('K(' + (hi[0] + 1) + ', ' + (hi[1] + 1) + ') = ' + fmt(K_POLY[hi[0]][hi[1]]), [r.x + 12, r.y + 18], { screen: true, anchor: 'left', size: 13, color: 'gold', font: 'ui' });
      } else g2.label('the twelve sample points', [r.x + 12, r.y + 18], { screen: true, anchor: 'left', size: 12.5, color: 'muted', font: 'ui' });
    });
  }

  function chainArrows(g3, u) {
    const nShow = Math.round(u * 12);
    let o = [0, 0, 0];
    for (let k = 0; k < nShow; k++) { const v = N.scale(phi(CIRC.X[SUB[k]], 'circles'), ALPHA[k]); const e = N.add(o, v); g3.vector(o, e, { color: k < 6 ? 'data' : 'dataB', width: 1.8, head: 7, alpha: 0.9 }); o = e; }
    if (nShow > 0 && nShow < 12) g3.label('αᵢφ(xᵢ)', o, { color: 'muted', size: 12, dx: 26, dy: -6 });
    if (nShow >= 12) g3.vector([0, 0, 0], W_SUM, { color: 'gold', width: 3.2, label: 'w' });
  }
  function scoreField(g2, fn, alpha) {
    const m = 1.8;
    g2.heat2d(p => LA.clamp(fn(p) / m, -1, 1), g2.P.bounds(), { res: 60, range: [-1, 1], ramp: divRamp, alpha });
  }
  function renderScore(g, s) {
    duo(g, (g3, r) => {
      g3.grid(null, { range: 2, step: 0.5, alpha: 0.35 });
      g3.axes({ length: 2.1, names: ['x', 'y', 'z'] });
      bowlSheet(g3, 'circles', 0.18);
      SUB.forEach((i, k) => { const p = phi(CIRC.X[i], 'circles'); g3.point(p, { r: 4, color: k < 6 ? 'data' : 'dataB' }); });
      chainArrows(g3, s.chain);
      title(g3, r, 'w = Σ αᵢ φ(xᵢ), tip to tail');
    }, (g2, r) => {
      floor2d(g2, 4);
      if (s.field > 0.01) {
        scoreField(g2, fPoly, 0.65 * s.field);
        g2.contours(fPoly, [0], { x0: -2.4, x1: 2.4, y0: -2.4, y1: 2.4 }, { res: 60, color: 'teal', width: 2.6, alpha: s.field });
      }
      pts2d(g2, s, CIRC, { r: 3.2, alpha: 0.6 });
      SUB.forEach((i, k) => g2.point(CIRC.X[i], { r: 5, color: k < 6 ? 'data' : 'dataB', stroke: 'white', strokeWidth: 1 }));
      title(g2, r, 'f(x) = Σ αᵢ k(xᵢ, x) + b');
      if (s.field > 0.5) {
        g2.label('teal curve: f = 0 · teal shade f > 0 · magenta f < 0', [r.x + 12, r.y + 40], { screen: true, anchor: 'left', size: 11.5, color: 'muted', font: 'ui' });
        g2.label('accuracy on all 100 points: ' + fmt.pct(ACC_POLY), [r.x + 12, r.y + 60], { screen: true, anchor: 'left', size: 12, color: ACC_POLY === 1 ? 'ok' : 'hud', font: 'ui' });
      }
    }, { view: '3d', cam: s.cam }, { view: '2d', cam: s.cam2 });
  }

  function renderRbf(g, s) {
    const u = s.morph;
    duo(g, (g2, r) => {
      const b = { x0: -1.65, x1: 1.65, y0: -1.45, y1: 1.85 };
      heatCells(g2, K_POLY, b, { range: rangeOf(K_POLY), alpha: 1 - u });
      if (u > 0.01) heatCells(g2, K_RBF, b, { range: [0, 1], alpha: u });
      g2.label(u < 0.5 ? 'K from the lift φ (12×12)' : 'RBF K, σ = 0.5: ∞ features, no lift to draw', [0, b.y1], { anchor: 'above', dy: -4, size: 13.5, color: 'text', font: 'ui' });
      g2.label('The picture stops at 3. The formula does not.', [r.x + r.w / 2, r.y + r.h - 22], { screen: true, size: 13, color: 'gold', font: 'ui' });
    }, (g2, r) => {
      floor2d(g2, 4);
      const fR = p => fRbf(p, 0.5, BR05) * 4.5;    // rescaled for the shared color range
      scoreField(g2, fPoly, 0.65 * (1 - u));
      if (u > 0.01) scoreField(g2, fR, 0.65 * u);
      g2.contours(u < 0.5 ? fPoly : fR, [0], { x0: -2.4, x1: 2.4, y0: -2.4, y1: 2.4 }, { res: 60, color: 'teal', width: 2.6 });
      pts2d(g2, s, CIRC, { r: 3.2, alpha: 0.6 });
      SUB.forEach((i, k) => g2.point(CIRC.X[i], { r: 5, color: k < 6 ? 'data' : 'dataB', stroke: 'white', strokeWidth: 1 }));
      title(g2, r, u < 0.5 ? 'f with the lifted-dot kernel' : 'same α, RBF kernel: bumpier boundary');
      g2.label('accuracy on all 100 points: ' + fmt.pct(u < 0.5 ? ACC_POLY : ACC_RBF), [r.x + 12, r.y + 40], { screen: true, anchor: 'left', size: 12, color: (u < 0.5 ? ACC_POLY : ACC_RBF) === 1 ? 'ok' : 'hud', font: 'ui' });
    }, { view: '2d', cam: { center: [0, 0.1], halfH: 2.2, halfW: 2.0 } }, { view: '2d', cam: s.cam2 });
  }

  function renderXor(g, s) {
    g.grid(null, { range: 2, step: 0.5, alpha: 0.3 });
    g.axes({ length: 2.3, names: ['x', 'y', 'z'] });
    if (s.lift > 0.02) g.paramSurface((u, v) => [u, v, s.lift * u * v], { u0: -1.6, u1: 1.6, v0: -1.6, v1: 1.6 }, { res: 16, mode: 'wire', color: 'grid', wireAlpha: 0.45 });
    lifted3d(g, XOR, 'xor', s.lift, true);
    if (s.lift > 0.6) {
      g.planes([{ point: [0, 0, 0], normal: [0, 0, 1], extent: 2, color: 'gold', alpha: 0.22 * seg(s.lift, 0.6, 1) }]);
      g.line([-2, 0, 0], [2, 0, 0], { color: 'teal', width: 2.6 }); g.line([0, -2, 0], [0, 2, 0], { color: 'teal', width: 2.6 });
      g.label('cut: xy = 0 (the two axes)', [1.9, 0, 0], { color: 'teal', size: 12.5, dy: 16 });
    }
    g.hud('XOR: rose where xy > 0, blue where xy < 0 (synthetic blobs)', 16, 24, { color: 'hud' });
    if (s.lift > 0.95) g.hud('accuracy ' + fmt.pct(planeAcc({ a: 0, b: 0, c: 1, d: 0 }, 'xor')), 16, 46, { color: 'ok' });
  }

  function renderML(g, s) {
    const card = (i, vopts, fn, text) => g.viewport({ x: 0.02, y: 0.02 + i * 0.327, w: 0.96, h: 0.31 }, Object.assign({ bg: 'panel' }, vopts), (g2, r) => { fn(g2, r); title(g2, r, text); });
    // 1. SVM: margin planes from the data
    const zIn = Math.max.apply(null, CIRC.X.filter((_, i) => !CIRC.y[i]).map(p => liftZ(p))), zOut = Math.min.apply(null, CIRC.X.filter((_, i) => CIRC.y[i]).map(p => liftZ(p)));
    card(0, { view: '3d', cam: Object.assign(LA.cam.defaults3d(), { target: [0, 0, 1.25], dist: 5.6, yaw: 0.5, pitch: 0.2 }) }, (g3) => {
      bowlSheet(g3, 'circles', 0.25);
      lifted3d(g3, CIRC, 'circles', 1, false);
      const zm = (zIn + zOut) / 2;
      g3.planes([{ point: [0, 0, zm], normal: [0, 0, 1], extent: 1.7, color: 'gold', alpha: 0.25 }]);
      [zIn, zOut].forEach(z => g3.polyline([[-1.7, -1.7, z], [1.7, -1.7, z], [1.7, 1.7, z], [-1.7, 1.7, z]], { color: 'gold', dashed: true, closed: true, width: 1.2 }));
      CIRC.X.forEach(p => { const z = liftZ(p); if (Math.abs(z - zIn) < 1e-9 || Math.abs(z - zOut) < 1e-9) g3.marker.ring([p[0], p[1], z], { r: 9, color: 'gold' }); });
    }, 'SVM idea: widest margin (shown among level planes); ringed = support vectors');
    // 2. attention scores as a Gram matrix
    card(1, { view: '2d', cam: { center: [0, 0], halfH: 1.3 } }, (g2, r) => {
      const T = LA.rng.tokenVectors(611, 6, 4); const G = T.map(a => T.map(b => N.dot(a, b)));
      const x0 = g2.P.toWorld(r.x + 24, 0)[0];
      heatCells(g2, G, { x0, x1: x0 + 2.0, y0: -1.15, y1: 0.85 }, { range: [-1, 1], ramp: t => LA.color.heat(t) });
      g2.label('QKᵀ: a 6×6 table of token dot products (p6-03)', [r.x + r.w - 14, r.y + r.h / 2], { screen: true, anchor: 'right', size: 13, color: 'muted', font: 'ui' });
    }, 'attention scores');
    // 3. a learned lift: a tanh layer bends the plane into a sheet in 3D
    card(2, { view: '3d', cam: Object.assign(LA.cam.defaults3d(), { target: [0, 0, 0], dist: 5.2, yaw: 0.8, pitch: 0.35 }) }, (g3, r) => {
      const L = (u, v) => [Math.tanh(1.4 * u), Math.tanh(1.4 * v), Math.tanh(1.1 * (u * 0.9 + v * 0.9) - 0.4)];
      g3.paramSurface(L, { u0: -1.8, u1: 1.8, v0: -1.8, v1: 1.8 }, { res: 14, mode: 'wire', color: 'teal', wireAlpha: 0.55 });
      g3.planes([{ point: [0, 0, 0], normal: [0, 0, 1], extent: 1.1, color: 'gold', alpha: 0.22 }]);
      g3.label('see p6-08', [r.x + r.w - 14, r.y + r.h - 18], { screen: true, anchor: 'right', size: 13, color: 'gold', font: 'ui' });
    }, 'a network learns its lift: φ = tanh(Wx + b), then a gold cut');
  }

  function renderRecap(g, s) {
    const card = (i, vopts, fn, text) => g.viewport({ x: 0.02, y: 0.02 + i * 0.327, w: 0.96, h: 0.31 }, Object.assign({ bg: 'panel' }, vopts), (g2, r) => { fn(g2, r); title(g2, r, text); });
    card(0, { view: '3d', cam: Object.assign(LA.cam.defaults3d(), { target: [0, 0, 1.2], dist: 5.8, yaw: 0.5, pitch: 0.28 }) }, (g3) => {
      bowlSheet(g3, 'circles', 0.3); lifted3d(g3, CIRC, 'circles', 1, false);
      g3.planes([{ point: [0, 0, 1], normal: [0, 0, 1], extent: 1.7, color: 'gold', alpha: 0.28 }]);
      cutCurve3d(g3, { a: 0, b: 0, c: 1, d: 1 }, 'circles');
    }, '1 · lift, then cut flat');
    card(1, { view: '2d', cam: { center: [0, 0], halfH: 1.9 } }, (g2) => {
      pts2d(g2, s, CIRC, { r: 3 }); g2.circle([0, 0], 1, { stroke: 'teal', width: 2.6 });
      g2.label('x² + y² = 1', [1.3, 1.0], { color: 'teal', size: 13 });
    }, '2 · the cut comes home as a curve');
    card(2, { view: '2d', cam: { center: [0, 0], halfH: 1.3 } }, (g2, r) => {
      const x0 = g2.P.toWorld(r.x + 24, 0)[0];
      heatCells(g2, K_POLY, { x0, x1: x0 + 2.0, y0: -1.15, y1: 0.85 }, { range: rangeOf(K_POLY) });
      g2.label('K = ΦΦᵀ = [k(xᵢ, xⱼ)], 12×12', [r.x + r.w - 14, r.y + r.h / 2], { screen: true, anchor: 'right', size: 13.5, color: 'muted', font: 'ui' });
    }, '3 · kernel = lifted dot product; K is all you need');
  }
})();
