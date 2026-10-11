/* p1-09 — Your first model: a neuron and its boundary. */
(function () {
  'use strict';
  const N = LA.num, seg = LA.seg, lerp = LA.lerp, lerpV = LA.lerpV, C = LA.color;
  const HONEST = 'The picture stops at 3. The formula does not.';
  const W = '<span class="c-w">w</span>';
  const W0 = [0.2, 0.8], LR = 2;     // step size η: with the off-centre data, 8 steps go from 10/40 to 38/40
  const sig = z => 1 / (1 + Math.exp(-z));
  const f2 = x => LA.fmt.fixed(x, 2);
  const SUBS = '₀₁₂₃₄₅₆₇₈₉';

  // ---------------- data ----------------
  // LA.rng(9): A (blue, 0) mean [−0.5, 0]; B (rose, 1) mean [2.5, −2]; sd 0.6. The clouds sit off-centre on purpose:
  // with w = [1, −1] the line through the origin gets 34/40, and only a bias (b ≈ −1.5) separates all 40.
  function twoBlobs40() {
    const r = LA.rng(9), X = [], y = [];
    for (let i = 0; i < 20; i++) { X.push([-0.5 + 0.6 * LA.rng.gauss(r), 0 + 0.6 * LA.rng.gauss(r)]); y.push(0); }
    for (let i = 0; i < 20; i++) { X.push([2.5 + 0.6 * LA.rng.gauss(r), -2 + 0.6 * LA.rng.gauss(r)]); y.push(1); }
    return { X, y };
  }
  function blobs3() {                              // LA.rng(19): means [−1.5, 1, 0.5] and [1.5, −1, −0.5], sd 0.6
    const r = LA.rng(19), X = [], y = [];
    for (let c = 0; c < 2; c++) for (let i = 0; i < 20; i++) { const m = c ? [1.5, -1, -0.5] : [-1.5, 1, 0.5]; X.push(m.map(v => v + 0.6 * LA.rng.gauss(r))); y.push(c); }
    return { X, y };
  }
  const score = (w, b, x) => N.dot(w, x) + b;
  function accuracy(X, y, w, b) { let c = 0; for (let i = 0; i < X.length; i++) if ((score(w, b, X[i]) > 0 ? 1 : 0) === y[i]) c++; return c; }
  /** one full-batch gradient step of the mean logistic loss */
  function gradStep(X, y, w, b, lr) {
    const g = w.map(() => 0); let gb = 0;
    for (let i = 0; i < X.length; i++) { const e = sig(score(w, b, X[i])) - y[i]; for (let k = 0; k < w.length; k++) g[k] += e * X[i][k]; gb += e; }
    return { w: w.map((v, k) => v - lr * g[k] / X.length), b: b - lr * gb / X.length };
  }
  function trainPath(X, y) { let w = W0.slice(), b = 0; const out = [[w[0], w[1], b]]; for (let k = 0; k < 8; k++) { const r = gradStep(X, y, w, b, LR); w = r.w; b = r.b; out.push([w[0], w[1], b]); } return out; }

  // ten neurons reading a synthetic "7" (memoized; 784-component strips are not stored in the state)
  let NEU = null;
  function neurons() {
    if (NEU) return NEU;
    const x = LA.rng.syntheticDigits(2, 8, { size: 28 })[7].pixels;
    const T = LA.rng.syntheticDigits(99, 10, { size: 28 }).map(d => d.pixels);
    const m = T[0].map((_, k) => T.reduce((a, v) => a + v[k], 0) / 10);
    const r = LA.rng(99);
    const Wt = T.map(v => v.map((p, k) => p - m[k] + 0.04 * LA.rng.gauss(r)));
    NEU = { x, W: Wt, scores: Wt.map(w => N.dot(w, x)) };
    return NEU;
  }

  // ---------------- geometry helpers ----------------
  const foot = (w, b) => { const n2 = N.dot(w, w) || 1e-9; return N.scale(w, -b / n2); };      // closest boundary point to the origin
  const along = w => [-w[1], w[0]];
  function clipHalf(poly, w, b) {                  // Sutherland–Hodgman: keep w·x + b ≥ 0
    const out = [];
    for (let i = 0; i < poly.length; i++) {
      const a = poly[i], c = poly[(i + 1) % poly.length], sa = score(w, b, a), sc = score(w, b, c);
      if (sa >= 0) out.push(a);
      if ((sa >= 0) !== (sc >= 0)) { const t = sa / (sa - sc); out.push([a[0] + (c[0] - a[0]) * t, a[1] + (c[1] - a[1]) * t]); }
    }
    return out;
  }
  function rightMark(g, at, u, v, r, color) {
    const nu = N.normalize(u), nv = N.normalize(v);
    g.polyline([N.add(at, N.scale(nu, r)), N.add(at, N.add(N.scale(nu, r), N.scale(nv, r))), N.add(at, N.scale(nv, r))], { color, width: 1.4 });
  }

  // ---------------- cameras ----------------
  const CAM2 = () => ({ center: [0.6, -0.9], halfH: 4.0 });
  const CAMSIG = () => Object.assign(LA.cam.defaults3d(), { target: [0, 0, 0.9], dist: 18.5, yaw: 0.52, pitch: 0.61 });
  const CAMPL = () => Object.assign(LA.cam.defaults3d(), { target: [0, 0, 0], dist: 13.5, yaw: 0.6, pitch: 0.42 });
  const CAMN = () => ({ center: [0, 0], halfH: 4 });

  // ---------------- 2D scene ----------------
  function render2d(g, s, ctx) {
    g.ghostGrid({ range: 6 });
    g.axes({ ticks: true });
    const w = s.w, b = s.b, f = foot(w, b), d = along(w), wn = N.norm(w);
    // half-plane shading: rose side (score > 0) and blue side
    if (s.shade > 0 && wn > 1e-6) {
      const box = [[-8, -8], [8, -8], [8, 8], [-8, 8]];
      const pos = clipHalf(box, w, b), neg = clipHalf(box, N.scale(w, -1), -b);
      if (pos.length > 2) g.polygon(pos, { fill: 'dataB', alpha: 0.07 * s.shade });
      if (neg.length > 2) g.polygon(neg, { fill: 'data', alpha: 0.07 * s.shade });
    }
    if (s.ghostB !== null && s.ghostB !== undefined) g.lineThrough(foot(w, s.ghostB), d, { color: 'gold', width: 1.5, dashed: true, alpha: 0.55, len: 14 });
    if (s.trainK > 0) {                            // ghosts of earlier training states + the w-tip trail
      const P = s.trainPath, k = Math.floor(s.trainK + 1e-9);
      for (let i = 0; i < Math.min(k, 8); i++) g.lineThrough(foot([P[i][0], P[i][1]], P[i][2]), along([P[i][0], P[i][1]]), { color: 'gold', width: 1, alpha: 0.2, len: 14 });
      const tips = []; for (let i = 0; i <= Math.min(k, 8); i++) tips.push([P[i][0], P[i][1]]); tips.push(w.slice());
      if (tips.length > 1) g.polyline(tips, { color: 'gold', width: 1.6, dash: [2, 4] });
      tips.slice(0, -1).forEach(p => g.point(p, { r: 2.5, color: 'gold' }));
    }
    if (s.showB > 0 && wn > 1e-6) g.lineThrough(f, d, { color: 'gold', width: 2.6, alpha: s.showB, len: 14 });
    if (s.showB > 0 && s.showRight) rightMark(g, f, w, d, 0.32, 'gold');
    if (s.dimB) g.dim([0, 0], f, { label: '−b/‖w‖ = ' + f2(-b / wn), color: 'gold', offset: 18 });
    // shadows of six sample points on w's line (step 2)
    if (s.shadows > 0) {
      g.lineThrough([0, 0], w, { color: 'w', width: 1, dashed: true, alpha: 0.45, len: 14 });
      [0, 3, 7, 21, 25, 33].forEach(i => {
        const p = s.X[i], u = N.normalize(w), ft = N.scale(u, N.dot(p, u));
        g.line(p, ft, { color: 'residual', width: 1, dashed: true, alpha: s.shadows });
        g.line([0, 0], ft, { color: 'mag', width: 3, alpha: 0.55 * s.shadows });
        g.point(ft, { r: 3.5, color: 'mag', alpha: s.shadows });
      });
    }
    // points
    for (let i = 0; i < s.X.length; i++) {
      const p = s.X[i], pred = score(w, b, p) > 0 ? 1 : 0, col = s.y[i] ? 'dataB' : 'data';   // true class; the shading shows the prediction
      g.point(p, { r: 4.6, color: col, alpha: s.appear });
      if (s.rings && pred !== s.y[i]) g.marker.ring(p, { r: 8, color: 'residual' });
    }
    if (s.showQ) { g.point([0.3, 0.2], { r: 6, color: 'muted', stroke: 'white' }); g.label('?', [0.3, 0.2], { dx: 12, dy: -12, color: 'hud', size: 16, bold: true }); }
    // misconception: a gray "w?" along the boundary
    if (s.wq > 0) {
      const q = N.scale([1, 1], s.wq), p = [1.5, 1.5];
      g.vector([0, 0], q, { color: 'muted', width: 3, label: 'w?', labelColor: 'muted' });
      if (s.wq >= 1) { g.line([0, 0], p, { color: 'mag', width: 3, alpha: 0.6 }); g.point(p, { r: 6, color: 'v', stroke: 'white', label: 'p (on the line)', labelColor: 'v' }); }
      if (s.cross) g.marker.cross(q, { dx: 34, dy: 16, size: 20 });
    }
    // the weight vector
    const pw = s.pulse ? 1 + 0.6 * s.pulse : 1;
    if (wn > 1e-6) g.vector(w, { color: 'w', label: 'w', width: 3 * pw, head: 12 + 4 * (s.pulse || 0) });
    // probe (check 3)
    if (s.showProbe) {
      const sc = score(w, b, s.probe);
      g.point(s.probe, { r: 7, color: 'gold', stroke: 'white' });
      g.label('probe · score ' + LA.fmt.signed(sc), s.probe, { anchor: 'left', dx: 12, dy: -14, size: 12.5, color: 'gold', font: 'ui' });
      g.handle('probe', s.probe, (p, st) => { st.probe = [p[0], p[1]]; }, { color: 'gold' });
    }
    // handles: w tip; a gold dot on the boundary that slides it (b = −w·p)
    g.handle('w', w, (p, st) => { st.w = [p[0], p[1]]; }, { color: 'w' });
    if (s.bHandle && wn > 1e-6) {
      g.point(f, { r: 6, color: 'gold', stroke: 'white' });
      g.handle('b', f, (p, st) => { st.b = -N.dot(st.w, p); }, { color: 'gold' });
    }
    // HUD
    const acc = accuracy(s.X, s.y, w, b);
    if (s.showAcc) g.hud('accuracy  ' + acc + ' / 40', 14, 24, { size: 16, color: 'gold', bold: true });
    if (s.showAcc && s.rings) g.hud('rule: score > 0 means rose · gray rings: wrong side', 14, 46, { size: 11.5, color: 'muted' });
    if (s.trainK > 0) g.hud('training step ' + Math.min(8, Math.round(s.trainK)) + ' / 8', 14, 68, { size: 12.5, color: 'hud' });
    if (s.showScores) scoreBars(g, s);
    if (s.wq > 0) misBars(g, s);
  }

  function scoreBars(g, s) {
    g.viewport({ x: 0.02, y: 0.795, w: 0.96, h: 0.19 }, { view: '2d', cam: { center: [0, 0], halfH: 1 }, bg: 'bg', border: true }, (gg, r) => {
      const sc = s.X.map(p => score(s.w, s.b, p)), m = Math.max(1e-6, ...sc.map(Math.abs)), n = sc.length, half = Math.min(4.2, r.w / 2 / (r.h / 2) - 0.3);
      const bw = 2 * half / n, y0 = -0.12;
      gg.line([-half, y0], [half, y0], { color: 'line', width: 1 });
      for (let i = 0; i < n; i++) {
        const h = sc[i] / m * 0.62 * s.barsT, x0 = -half + i * bw + bw * 0.12, x1 = -half + (i + 1) * bw - bw * 0.12;
        if (Math.abs(h) > 1e-4) gg.polygon([[x0, y0], [x1, y0], [x1, y0 + h], [x0, y0 + h]], { fill: s.y[i] ? 'dataB' : 'data', alpha: 0.85 });
      }
      gg.hud('scores w·x' + (s.b ? ' + b' : '') + ' of all 40 points (blue, then rose) · above the line = positive', r.x + 8, r.y + 13, { size: 11.5, color: 'muted' });
    });
  }
  function misBars(g, s) {
    g.viewport({ x: 0.02, y: 0.795, w: 0.96, h: 0.19 }, { view: '2d', cam: { center: [0, 0], halfH: 1 }, bg: 'bg', border: true }, (gg, r) => {
      const p = [1.5, 1.5], a = N.dot(s.w, p), q = N.dot([1, 1], p) * s.wq;
      gg.line([-3, -0.6], [3, -0.6], { color: 'line', width: 1 });
      if (Math.abs(a) > 1e-3) gg.polygon([[-1.6, -0.6], [-0.6, -0.6], [-0.6, -0.6 + a / 3 * 1.1], [-1.6, -0.6 + a / 3 * 1.1]], { fill: 'w', alpha: 0.85 });
      gg.polygon([[0.6, -0.6], [1.6, -0.6], [1.6, -0.6 + q / 3 * 1.1], [0.6, -0.6 + q / 3 * 1.1]], { fill: 'muted', alpha: 0.85 });
      gg.label('w·p = ' + f2(a), [-1.1, -0.6], { anchor: 'above', dy: -6, size: 12, color: 'w', font: 'ui' });
      gg.label('w?·p = ' + f2(q), [1.1, -0.6 + q / 3 * 1.1], { anchor: 'above', dy: -4, size: 12, color: 'hud', font: 'ui' });
      gg.hud('score of the point p that lies on the boundary', r.x + 8, r.y + 13, { size: 11.5, color: 'muted' });
    });
  }

  // ---------------- 3D scenes ----------------
  function renderSig(g, s, ctx, compact) {
    const w = s.w, b = s.b, Z = 3;
    g.grid(null, { range: 4, alpha: 0.5 });
    g.surface(p => sig(score(w, b, p)), { x0: -4, x1: 4, y0: -4, y1: 4 }, { res: compact ? 16 : 30, zScale: Z, zRange: [0, Z], mode: compact ? 'quads' : 'both', wireStep: 3, alpha: 0.5 * s.sigA, ramp: t => C.mix(C.data, C.dataB, t), wireColor: 'muted', wireAlpha: 0.35 * s.sigA });
    // boundary lifted to height 0.5 (×3) on the surface
    const wn = N.norm(w);
    if (wn > 1e-6) {
      const f = foot(w, b), d = N.normalize(along(w)), pts = [];
      for (let k = -40; k <= 40; k++) { const p = N.add(f, N.scale(d, k * 0.15)); if (Math.abs(p[0]) <= 4 && Math.abs(p[1]) <= 4) pts.push([p[0], p[1], 0.5 * Z]); }
      if (pts.length > 1) g.polyline(pts, { color: 'gold', width: 2.6 });
      if (!compact) g.vector([f[0], f[1], 0], [f[0] + w[0] / wn * 1.4, f[1] + w[1] / wn * 1.4, 0], { color: 'w', label: 'w', width: 2.6 });
    }
    if (!compact) for (let i = 0; i < s.X.length; i++) { const p = s.X[i]; g.point([p[0], p[1], Z * sig(score(w, b, p)) * s.lift], { r: 4, color: s.y[i] ? 'dataB' : 'data' }); }
    if (compact) return;
    g.hud('height = σ(w·x + b): probability of “rose”', 14, 24, { size: 12.5, color: 'muted' });
    g.viewport({ x: 0.02, y: 0.72, w: 0.36, h: 0.25 }, { view: '2d', cam: { center: [0, 0.5], halfH: 0.78 }, bg: 'bg', border: true }, (gg, r) => {
      const pts = []; for (let k = 0; k <= 60; k++) { const z = -6 + k * 0.2; pts.push([z / 7.5, sig(z)]); }
      gg.line([-0.85, 0], [0.85, 0], { color: 'gridAxis', width: 1 }); gg.line([0, -0.05], [0, 1.05], { color: 'gridAxis', width: 1 });
      gg.line([-0.85, 0.5], [0.85, 0.5], { color: 'gold', width: 1, dashed: true });
      gg.polyline(pts, { color: 'w', width: 2.2 });
      gg.label('σ(s)', [0.05, 1.0], { anchor: 'left', size: 11.5, color: 'w', font: 'ui', bg: false });
      gg.label('0.5', [-0.85, 0.5], { anchor: 'left', dy: -8, size: 10.5, color: 'gold', font: 'ui', bg: false });
      gg.label('score s', [0.85, 0], { anchor: 'right', dy: 9, size: 10.5, color: 'muted', font: 'ui', bg: false });
    });
  }

  function renderPlane(g, s, ctx) {
    const w = s.w3, b = s.b3, wn = N.norm(w);
    g.grid(null, { range: 3, alpha: 0.45 });
    g.axes({ length: 3, names: ['x₁', 'x₂', 'x₃'] });
    const pt = N.scale(w, -b / (wn * wn));
    g.planes([{ point: pt, normal: w, extent: 2.8, color: 'gold', alpha: 0.22 * s.planeA }]);
    for (let i = 0; i < s.X3.length; i++) {
      const p = s.X3[i], pred = score(w, b, p) > 0 ? 1 : 0;
      g.point(p, { r: 4.2, color: pred ? 'dataB' : 'data' });
      if (pred !== s.y3[i]) g.marker.ring(p, { r: 8, color: 'residual' });
    }
    g.vector(pt, N.add(pt, N.scale(w, 1.6 / wn)), { color: 'w', label: 'w', width: 3 });
    g.hud('3 features: the boundary w·x + b = 0 is a plane, w its normal · ' + accuracy(s.X3, s.y3, w, b) + ' / 40 correct', 14, 24, { size: 12, color: 'muted' });
    g.hud(HONEST, 14, g.size.h - 18, { size: 12.5, color: 'hud', bold: true });
  }

  // ---------------- ten neurons ----------------
  function renderNeurons(g, s, ctx, L) {
    const D = neurons(), u = s.neuT;
    L = L || { sx0: -2.4, sx1: 3.2, xY: 3.15, rowTop: 2.45, rowH: 0.3, bars: { x0: -2.4, x1: 3.2, y0: -3.4, y1: -0.8 }, labels: true };
    const strip = (vals, y0, h, ramp, range, a) => { g.heat2d([vals], { x0: L.sx0, x1: L.sx1, y0, y1: y0 + h }, { range, ramp, alpha: a, smooth: false }); g.polyline([[L.sx0, y0], [L.sx1, y0], [L.sx1, y0 + h], [L.sx0, y0 + h]], { closed: true, color: 'line', width: 1, alpha: a }); };
    strip(D.x, L.xY, 0.3, t => { const v = Math.round(255 * t); return 'rgb(' + v + ',' + v + ',' + v + ')'; }, [0, 1], 1);
    g.label('x', [L.sx0, L.xY + 0.15], { anchor: 'right', dx: -8, size: 14, color: 'v', font: 'math' });
    if (L.labels) g.label('synthetic “7”, flattened: 784 pixels', [L.sx0, L.xY + 0.3], { anchor: 'left', dy: -10, size: 11.5, color: 'muted', font: 'ui', bg: false });
    const active = u > 0 && u < 1 ? Math.min(9, Math.floor(u * 10)) : -1;
    for (let k = 0; k < 10; k++) {
      const y0 = L.rowTop - k * L.rowH;
      strip(D.W[k], y0, L.rowH * 0.8, C.diverge, [-1, 1], 0.95);
      g.label('w' + SUBS[k], [L.sx0, y0 + L.rowH * 0.4], { anchor: 'right', dx: -6, size: 11.5, color: 'w', font: 'ui', bg: false });
      if (k === active) g.polyline([[L.sx0, y0], [L.sx1, y0], [L.sx1, y0 + L.rowH * 0.8], [L.sx0, y0 + L.rowH * 0.8]], { closed: true, color: 'v', width: 2 });
    }
    const B = L.bars, m = Math.max(...D.scores.map(Math.abs)), zeroY = (B.y0 + B.y1) / 2, half = (B.y1 - B.y0) / 2 * 0.9, bw = (B.x1 - B.x0) / 10;
    g.line([B.x0, zeroY], [B.x1, zeroY], { color: 'gridAxis', width: 1.2 });
    let best = 0; D.scores.forEach((v, k) => { if (v > D.scores[best]) best = k; });
    for (let k = 0; k < 10; k++) {
      const grow = LA.clamp01(u * 10 - k), h = D.scores[k] / m * half * grow, x0 = B.x0 + k * bw + bw * 0.15, x1 = B.x0 + (k + 1) * bw - bw * 0.15;
      const col = (u >= 1 && k === best) ? 'gold' : 'w';
      if (Math.abs(h) > 1e-4) g.polygon([[x0, zeroY], [x1, zeroY], [x1, zeroY + h], [x0, zeroY + h]], { fill: col, alpha: 0.85 });
      g.label(String(k), [(x0 + x1) / 2, B.y0], { anchor: 'below', dy: 2, size: 11, color: (u >= 1 && k === best) ? 'gold' : 'muted', font: 'ui', bg: false });
      if (L.labels && grow >= 1) g.label(LA.fmt.fixed(D.scores[k], 0), [(x0 + x1) / 2, zeroY + h], { anchor: D.scores[k] >= 0 ? 'above' : 'below', dy: D.scores[k] >= 0 ? -3 : 3, size: 10, color: 'hud', font: 'ui', bg: false });
    }
    if (L.labels) {
      g.label('scoreₖ = wₖ·x  (biases 0 here)', [B.x0, B.y1], { anchor: 'left', dy: -6, size: 11.5, color: 'muted', font: 'ui', bg: false });
      if (u >= 1) g.label('tallest: 7', [B.x0 + (best + 0.5) * bw, zeroY + D.scores[best] / m * half], { anchor: 'above', dy: -16, size: 12, color: 'gold', font: 'ui' });
    }
  }

  // ---------------- recap ----------------
  function renderRecap(g, s, ctx) {
    const W1 = [1.3, -0.9], B1 = -1.75;
    g.viewport({ x: 0, y: 0.02, w: 1, h: 0.31 }, { view: '2d', cam: { center: [0, 0], halfH: 3.1 } }, (gg, r) => {
      const box = [[-12, -4], [12, -4], [12, 4], [-12, 4]];
      gg.polygon(clipHalf(box, W1, B1), { fill: 'dataB', alpha: 0.08 }); gg.polygon(clipHalf(box, N.scale(W1, -1), -B1), { fill: 'data', alpha: 0.08 });
      gg.lineThrough(foot(W1, B1), along(W1), { color: 'gold', width: 2.2, len: 12 });
      s.X.forEach((p, i) => gg.point(p, { r: 3.4, color: s.y[i] ? 'dataB' : 'data' }));
      gg.vector(foot(W1, B1), N.add(foot(W1, B1), W1), { color: 'w', label: 'w', width: 2.6 });
      rightMark(gg, foot(W1, B1), W1, along(W1), 0.3, 'gold');
      gg.hud('score = w·x + b · boundary ⟂ w · b slides it', r.x + 10, r.y + 16, { size: 12.5, color: 'gold' });
    });
    g.viewport({ x: 0, y: 0.345, w: 1, h: 0.31 }, { view: '3d', cam: Object.assign(CAMSIG(), { dist: 15 }) }, (gg, r) => {
      renderSig(gg, Object.assign({}, s, { w: W1, b: B1, sigA: 1, lift: 1 }), ctx, true);
      gg.hud('sigmoid: score → probability', r.x + 10, r.y + 16, { size: 12.5, color: 'gold' });
    });
    g.viewport({ x: 0, y: 0.67, w: 1, h: 0.31 }, { view: '2d', cam: { center: [0, 0.2], halfH: 4.4 } }, (gg, r) => {
      renderNeurons(gg, Object.assign({}, s, { neuT: 1 }), ctx, { sx0: -5.4, sx1: -0.6, xY: 3.0, rowTop: 2.2, rowH: 0.48, bars: { x0: 0.6, x1: 5.4, y0: -3.6, y1: 3.2 }, labels: false });
      gg.hud('ten neurons = ten dot products = a classifier', r.x + 10, r.y + 16, { size: 12.5, color: 'gold' });
    });
  }

  // ---------------- state helpers ----------------
  function base(s) {
    s.scene = '2d'; s.view = '2d'; s.cam = CAM2();
    s.appear = 1; s.shade = 0; s.showB = 0; s.showRight = false; s.dimB = false; s.ghostB = null; s.shadows = 0;
    s.colorPred = false; s.rings = false; s.showQ = false; s.wq = 0; s.cross = false; s.pulse = 0; s.showProbe = false;
    s.bHandle = false; s.showAcc = false; s.showScores = false; s.barsT = 1; s.trainK = 0; s.sigA = 1; s.lift = 1; s.planeA = 1; s.neuT = 1;
  }
  const fullRule = s => { s.shade = 1; s.showB = 1; s.colorPred = true; s.rings = true; s.showAcc = true; };

  LA.registerLesson({
    id: 'p1-09-first-model-neuron',
    part: 1,
    title: 'Your first model: a neuron and its boundary',
    subtitle: 'A weight vector, a dot product and a bias split the plane; a sigmoid turns the score into a probability.',
    minutes: 10,
    prereqs: ['p1-06-dot-product', 'p1-07-data-as-vectors'],
    tags: ['core'],
    badge: { drawnIn: '2D and 3D', realSize: 'a digit classifier’s last layer is 10 neurons, each with a 784-component w; a boundary in ℝ⁷⁸⁴ is a 783-dimensional hyperplane', transfers: 'w·x + b, the perpendicularity of w to the boundary, and the sigmoid are identical in any dimension; only the drawing of the boundary stops at a plane' },
    cantShow: ['a hyperplane in ℝ⁷⁸⁴ (we draw the plane in 3D and say the rule is the same)'],
    terms: [
      { term: 'weight vector', def: 'The direction a neuron compares inputs against (written w).' },
      { term: 'score', def: 'The number w·x + b; positive on one side of the boundary, negative on the other.' },
      { term: 'decision boundary', def: 'The set of points with score 0: a line perpendicular to w.' },
      { term: 'bias', def: 'A number b added to the score that slides the boundary without turning it.' },
      { term: 'sigmoid', def: 'A smooth S-shaped function squashing any score into a probability between 0 and 1.' },
      { term: 'hyperplane', def: 'The boundary in n dimensions: a line in 2D, a plane in 3D, flat in general.' },
      { term: 'training', def: 'Adjusting w and b, step by step, to lower the number of mistakes.' },
    ],
    atlas: ['logistic-regression', 'mlp', 'softmax', 'optimization'],
    thumbnail: 6,

    state() {
      const d = twoBlobs40(), d3 = blobs3();
      neurons();
      const s = { view: '2d', cam: CAM2(), X: d.X, y: d.y, X3: d3.X, y3: d3.y, w: [1, -1], b: 0, w3: [1, -1, -0.5], b3: -0.5, trainPath: trainPath(d.X, d.y), probe: [2, 2] };
      base(s);
      return s;
    },

    render(g, s, ctx) {
      if (s.scene === 'sig') return renderSig(g, s, ctx, false);
      if (s.scene === 'plane') return renderPlane(g, s, ctx);
      if (s.scene === 'neurons') { renderNeurons(g, s, ctx); if (s.neuT >= 1) g.hud('Part 6: softmax classifier — the bars become probabilities', 14, g.size.h - 20, { size: 12.5, color: 'gold' }); return; }
      if (s.scene === 'recap') return renderRecap(g, s, ctx);
      render2d(g, s, ctx);
    },

    math(s, tex) {
      // numbers are written with tex.text: the panel's in-place update mixes up tex.num and matrix cells
      const T = tex.text, eq = tex.eq, num = (x, c) => T(LA.fmt(x), { color: c || 'gold' }), wv = tex.vec(s.w, { color: 'w' });
      const acc = accuracy(s.X, s.y, s.w, s.b);
      switch (s.scene) {
        case 'sig': return tex.block(eq(T('p(rose) = σ(w·x + b) = 1 / (1 + e<sup>−(w·x + b)</sup>)', { html: true })), eq(tex.sym('w'), tex.op('='), wv, tex.sym('b'), tex.op('='), num(s.b, 'b')));
        case 'plane': return tex.block(eq(T('w·x + b = 0 in ℝ³: a plane · in ℝⁿ: a hyperplane')), eq(tex.sym('w'), tex.op('='), tex.vec(s.w3, { color: 'w' }), tex.sym('b'), tex.op('='), num(s.b3, 'b')));
        case 'neurons': { const D = neurons(); return tex.block(eq(T('scoreₖ = wₖ·x + bₖ,  k = 0 … 9')), eq(T('score₇ ='), num(D.scores[7]), T('· next best ='), num(Math.max(...D.scores.filter((_, k) => k !== 7)), 'text'))); }
        case 'recap': return tex.block(eq(T('score = w·x + b · boundary: score = 0 · p = σ(score)')));
      }
      const parts = [eq(tex.sym('w'), tex.op('='), wv, tex.sym('b'), tex.op('='), num(s.b, 'b'))];
      if (s.shadows > 0 && !s.showB) { const ex = [0, 21, 33].map(i => score(s.w, 0, s.X[i])); parts.push(eq(T('score(x) = w·x:'), num(ex[0], 'data'), T(','), num(ex[1], 'dataB'), T(','), num(ex[2], 'dataB'), T('…'))); }
      if (s.wq > 0) parts.push(eq(T('w ⟂ boundary: w·p ='), num(N.dot(s.w, [1.5, 1.5]), 'w'), T('but w?·p ='), num(3 * s.wq, 'text')));
      else if (s.showB) parts.push(eq(T('boundary: w·x + b = 0 · rule: w·x + b > 0 means rose')));
      if (s.dimB) parts.push(eq(T('slide: −b/‖w‖ ='), num(-s.b / N.norm(s.w))));
      if (s.trainK > 0) parts.push(eq(T('w ← w − η∇L,  η = ' + LR + ' · step'), num(Math.min(8, Math.round(s.trainK)), 'text')));
      if (s.showProbe) parts.push(eq(T('probe distance to boundary = |w·p + b| / ‖w‖ ='), num(Math.abs(score(s.w, s.b, s.probe)) / (N.norm(s.w) || 1))));
      if (s.showAcc || s.trainK > 0) parts.push(eq(T('accuracy ='), num(acc), T('/ 40')));
      return tex.block.apply(null, parts);
    },

    steps: [
      { title: 'Two kinds of points', duration: 3,
        narration: 'Forty synthetic points, two kinds: blue and rose, like the flowers of the last lesson. We want a rule that, given a new point such as the gray one, says which kind it is. The rule will be built from one vector and one number — your first model.',
        setup(s) { base(s); s.w = [1, -1]; s.b = 0; s.appear = 0; s.showQ = true; },
        at(s, t) { s.appear = seg(t, 0.05, 0.6); } },

      { title: 'A weight vector scores every point', duration: 4,
        narration: 'Pick a purple <b>weight vector</b> ' + W + ' = [1; −1]. Give every point a <b>score</b>: the dot product w·x, its magenta shadow on ' + W + ' times ‖w‖. Points on the side ' + W + ' points to score positive; points behind score negative. The bars below show all forty scores.',
        setup(s) { base(s); s.w = [1, -1]; s.b = 0; s.shadows = 0; s.showScores = true; s.barsT = 0; },
        at(s, t) { s.shadows = seg(t, 0.05, 0.45); s.barsT = seg(t, 0.5, 0.95); } },

      { title: 'The boundary is perpendicular to w', duration: 4,
        narration: (s) => 'Where is the score exactly 0? On the line through the origin perpendicular to ' + W + ' — every point there has no shadow. That gold line is the <b>decision boundary</b>. Read the sign of each score: the rule is “positive means rose”. Here it gets ' + accuracy(s.X, s.y, [1, -1], 0) + ' of 40 right: the line is stuck at the origin, too close to the blue cloud.',
        setup(s) { base(s); s.w = [1, -1]; s.b = 0; s.showRight = true; },
        at(s, t) { s.showB = seg(t, 0, 0.4); s.shade = seg(t, 0.4, 0.8); s.colorPred = t > 0.4; s.rings = t > 0.4; s.showAcc = t > 0.6; } },

      { title: 'w does not lie along the boundary', duration: 4, misconception: true,
        narration: 'Belief: “the weight vector points along the boundary line.” Try it: a gray arrow w? laid along the line gives the yellow point p, which sits on the line, a full-strength score instead of zero. The weight vector is the <i>normal</i> — it points across the boundary, toward the side it calls positive.',
        setup(s) { base(s); fullRule(s); s.showAcc = false; s.rings = false; s.w = [1, -1]; s.b = 0; s.showRight = true; s.wq = 0; },
        at(s, t) { s.wq = seg(t, 0.05, 0.4); s.cross = t > 0.7; s.pulse = seg.pulse(t, 0.75, 1); } },

      { title: 'The bias slides the boundary', duration: 4,
        narration: (s) => 'Add a number b to every score: w·x + b. The boundary slides along ' + W + ' by −b/‖w‖ without turning. With b = −1.5 it moves 1.06 toward the rose side, and ' + accuracy(s.X, s.y, [1, -1], 0) + ' of 40 correct becomes ' + accuracy(s.X, s.y, [1, -1], -1.5) + ' of 40. The <b>bias</b> lets the boundary leave the origin.',
        setup(s) { base(s); fullRule(s); s.w = [1, -1]; s.b = 0; s.ghostB = 0; s.showRight = true; },
        at(s, t) { s.b = -1.5 * seg(t, 0.1, 0.8); s.dimB = t > 0.8; } },

      { title: 'Separate them yourself', duration: 6, interactive: true,
        narration: 'Drag the purple tip of ' + W + ' to turn the boundary and the gold dot to slide it. Watch the accuracy counter. A direction that points from the blue cloud toward the rose cloud, with b near −1.5, separates all forty. There are many correct answers.',
        setup(s) { base(s); fullRule(s); s.bHandle = true; s.showRight = true; s._w0 = s.w.slice(); s._b0 = s.b; },
        at(s, t) { const u = seg(t, 0.1, 0.8); s.w = lerpV(s._w0, [1.3, -0.9], u); s.b = lerp(s._b0, -1.6, u); } },

      { title: 'From score to probability: the sigmoid', duration: 5,
        narration: 'A score of 7 and a score of 0.2 are both “rose”, but not equally sure. The <b>sigmoid</b> σ(s) = 1/(1 + e<sup>−s</sup>) squashes any score into a probability between 0 and 1 (e ≈ 2.72 is a fixed number). Over the plane it is a smooth ramp, steep along ' + W + ' and flat along the boundary, crossing 0.5 exactly on the gold line.',
        setup(s) { base(s); s.scene = 'sig'; s.view = '3d'; s.cam = CAMSIG(); s.sigA = 0; s.lift = 0; },
        at(s, t) { s.sigA = seg(t, 0.05, 0.5); s.lift = seg(t, 0.3, 0.8); LA.cam.tween(s, { yaw: 0.52 }, { yaw: 0.85 }, seg(t, 0, 1, 'linear')); } },

      { title: 'In 3D the boundary is a plane; in general, a hyperplane', duration: 5,
        narration: 'With three features the same rule gives a plane: all points with w·x + b = 0, perpendicular to ' + W + '. With 784 features it is a <b>hyperplane</b> we cannot draw, but ' + W + ' is still the normal and the score is still one dot product. The picture stops at 3. The formula does not.',
        setup(s) { base(s); s.scene = 'plane'; s.view = '3d'; s.cam = CAMPL(); s.planeA = 0; },
        at(s, t) { s.planeA = seg(t, 0.05, 0.4); LA.cam.tween(s, { yaw: 0.6 }, { yaw: 1.47 }, seg(t, 0, 1, 'linear')); } },

      { title: 'Training is moving w', duration: 5,
        narration: (s) => { const P = s.trainPath, a = k => accuracy(s.X, s.y, [P[k][0], P[k][1]], P[k][2]); return 'Start with a bad guess and let the data push. Each step changes ' + W + ' and b a little in the direction that lowers the mistakes — the scaled subtraction from p1-02. Eight steps swing the boundary from ' + a(0) + ' to ' + a(8) + ' of 40 correct; this is <b>training</b>. Part 5 explains where the push comes from.'; },
        setup(s) { base(s); fullRule(s); s.showRight = false; s.trainK = 0; const P = s.trainPath; s.w = [P[0][0], P[0][1]]; s.b = P[0][2]; },
        at(s, t) {
          const P = s.trainPath, k = 8 * seg(t, 0.05, 0.95, 'linear'), i = Math.min(7, Math.floor(k)), u = k - i;
          s.trainK = Math.max(1e-6, k); s.w = [lerp(P[i][0], P[i + 1][0], u), lerp(P[i][1], P[i + 1][1], u)]; s.b = lerp(P[i][2], P[i + 1][2], u);
        } },

      { title: 'Ten neurons read a digit', duration: 5, mlTie: true,
        narration: 'A digit classifier’s last layer is ten of these neurons, each with a 784-component ' + W + ', scoring the same flattened image. The ten scores become ten bars; the tallest wins. Logistic regression, the perceptron and the output layer of every network are this picture. Part 6 turns the bars into probabilities that sum to 1.',
        setup(s) { base(s); s.scene = 'neurons'; s.view = '2d'; s.cam = CAMN(); s.neuT = 0; },
        at(s, t) { s.neuT = seg(t, 0.05, 0.9, 'linear'); } },

      { title: 'Recap', duration: 3, recap: true,
        narration: 'A neuron is a weight vector, a dot product and a bias: score = w·x + b. The boundary is the zero set, perpendicular to ' + W + '; b slides it. The sign labels points; the sigmoid turns the score into a probability. Training moves ' + W + '. In ℝⁿ the boundary is a hyperplane and the formula is unchanged.',
        cards: ['<b>score = w·x + b.</b> The boundary (score 0) is perpendicular to w; b slides it without turning.', '<b>Sigmoid: score → probability.</b> σ(w·x + b) is a smooth ramp that crosses 0.5 on the boundary.', '<b>Ten neurons = ten dot products = a classifier.</b> The tallest score wins; training moves each w.'],
        setup(s) { base(s); s.scene = 'recap'; s.view = '2d'; s.cam = CAMN(); },
        at(s, t) { } },
    ],

    playground: {
      intro: '<p>Drag the purple tip of <span class="c-w">w</span> and the gold dot on the boundary, or edit the numbers. Train one gradient step at a time, or switch to the 3D sigmoid surface.</p>',
      setup(s) { base(s); fullRule(s); s.bHandle = true; s.showRight = true; s.showScores = true; s.w = [1, -1]; s.b = 0; },
      controls: [
        { type: 'matrix', bind: 'w', label: 'w (drag tip)', colors: 'w', step: 0.1 },
        { type: 'slider', bind: 'b', min: -4, max: 4, step: 0.05, label: 'bias b', color: 'b', fmt: x => LA.fmt(x) },
        { type: 'toggle', bind: 'showScores', label: 'Score bars' },
        { type: 'button', label: 'Train 1 step', onClick(s) { const r = gradStep(s.X, s.y, s.w, s.b, LR); s.w = r.w; s.b = r.b; } },
        { type: 'preset', label: 'Presets', items: [
          { label: 'Untrained', values: { w: W0, b: 0 } },
          { label: 'Trained', apply(s) { const P = s.trainPath[8]; s.w = [P[0], P[1]]; s.b = P[2]; } },
          { label: 'Vertical boundary', values: { w: [1.4, 0], b: 0 } },
          { label: 'Sigmoid surface (3D)', apply(s) { s.scene = 'sig'; s.view = '3d'; s.cam = CAMSIG(); s.sigA = 1; s.lift = 1; } },
          { label: '2D view', apply(s) { s.scene = '2d'; s.view = '2d'; s.cam = CAM2(); } },
        ] },
      ],
    },

    checks: [
      { kind: 'manipulate',
        prompt: 'Turn and slide the boundary (purple tip, gold dot) until all 40 points are on the correct side.',
        setup(s) { base(s); fullRule(s); s.bHandle = true; s.showRight = true; s.w = W0.slice(); s.b = 0; },
        judge(s) { const a = accuracy(s.X, s.y, s.w, s.b); return { ok: a === 40, progress: 0.95 * a / 40 }; },
        hint: 'Point ' + W + ' from the blue cloud toward the rose cloud, then drag the gold dot to slide the line into the gap (b near −1.5).',
        solution(s) { s.w = [1.3, -0.9]; s.b = -1.75; } },
      { kind: 'manipulate',
        prompt: 'Make the boundary vertical by dragging the purple tip of ' + W + '.',
        setup(s) { base(s); fullRule(s); s.showRight = true; s.w = [1, -1]; s.b = 0; },
        judge(s) { if (N.norm(s.w) < 0.2) return { ok: false, progress: 0, hint: 'Keep ' + W + ' longer than a stub; only its direction matters.' }; const a = LA.judge.angleDeg(s.w, [1, 0], 0), c = LA.judge.angleDeg(s.w, [-1, 0], 0); return a.dist < c.dist ? a : c; },
        hint: 'A vertical line has a horizontal normal: ' + W + ' along the green axis.',
        solution(s) { s.w = [1.4, 0]; } },
      { kind: 'manipulate',
        prompt: 'Drag the gold probe point onto the decision boundary.',
        setup(s) { base(s); fullRule(s); s.rings = false; s.showAcc = false; s.w = [1, -1]; s.b = -1; s.probe = [2, 2]; s.showProbe = true; },
        judge(s) { const d = Math.abs(score(s.w, s.b, s.probe)) / (N.norm(s.w) || 1); return { ok: d < LA.TOL.pos, progress: Math.max(0, 1 - d / 1.5) }; },
        hint: 'Move it until its score reads 0.',
        solution(s) { const n2 = N.dot(s.w, s.w), sc = score(s.w, s.b, s.probe); s.probe = N.sub(s.probe, N.scale(s.w, sc / n2)); } },
    ],
  });
})();
