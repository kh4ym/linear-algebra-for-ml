/* p3-02 — Gaussian elimination, visually. Every row operation rotates one line about the fixed crossing. */
(function () {
  'use strict';
  const N = LA.num, lerpM = LA.lerpM, seg = LA.seg, fmt = LA.fmt, C = LA.color;
  const I2 = () => [[1, 0], [0, 1]];
  const A0 = [[1, 2], [3, 4]], B0 = [5, 11], X0 = [1, 2];              // the lesson's system, x = [1; 2]
  const E1 = [[1, 0], [-3, 1]], E2 = [[1, 1], [0, 1]], DS = [[1, 0], [0, -0.5]];
  const E21 = N.matMul(E2, E1), AINV = N.matMul(DS, E21);              // D E₂ E₁ = A⁻¹ = [[−2,1],[1.5,−0.5]]
  const SWAP_A = [[0, 2], [3, 1]], SWAP_B = [4, 5];
  const SING_A = [[1, 2], [2, 4]], SING_B = [3, 1];
  const CAM = () => ({ center: [2.0, 1.3], halfH: 3.7 });
  const CAMW = () => ({ center: [3.1, 1.0], halfH: 4.6 });
  const bb = '<span class="c-b">b</span>';

  /* ---------- helpers ---------- */
  /* Engine workaround: LA.tex.reconcile maps tex.num spans onto matrix cells when only numbers change; emit live
     numbers as plain colored spans so any change rebuilds the panel cleanly. */
  function safeTex(tex) { return Object.assign({}, tex, { num: (x, o) => { o = o || {}; return '<span class="tx-n ' + tex.colorClass(o.color) + '">' + (typeof x === 'number' ? fmt(x, o.digits) : x) + '</span>'; } }); }
  const det2 = A => A[0][0] * A[1][1] - A[0][1] * A[1][0];
  /** handle multipliers m = [m21, m12] as a matrix: row₂ −= m21·row₁, then row₁ −= m12·row₂ */
  function H(m) { return N.matMul([[1, -m[1]], [0, 1]], [[1, 0], [-m[0], 1]]); }
  function effE(s) { return N.matMul(H(s.m), s.E); }
  function sys(s, E) { E = E || effE(s); return { A: N.matMul(E, s.A0), b: N.matVec(E, s.b0) }; }
  /** the line {p : a·p = c}: closest point to the origin and a unit direction (null for a zero row) */
  function rowLine(a, c) { const nn = a[0] * a[0] + a[1] * a[1]; if (nn < 1e-9) return null; const n = Math.sqrt(nn); return { p0: [a[0] * c / nn, a[1] * c / nn], d: [-a[1] / n, a[0] / n] }; }
  function drawRow(g, a, c, o) {
    const L = rowLine(a, c); if (!L) return null;
    g.lineThrough(L.p0, L.d, { len: 40, color: o.color || 'hud', width: o.width || 2.2, alpha: o.alpha === undefined ? 1 : o.alpha, dashed: o.dashed });
    return L;
  }
  /** a point on L inside the view, about `dist` from `from` (or the view centre) */
  function spotOn(g, L, from, dist) {
    const b = g.P.bounds(), mx = (b.x1 - b.x0) * 0.08, my = (b.y1 - b.y0) * 0.06;
    const inside = p => p[0] > b.x0 + mx && p[0] < b.x1 - mx && p[1] > b.y0 + my && p[1] < b.y1 - my;
    const base = from || [(b.x0 + b.x1) / 2, (b.y0 + b.y1) / 2];
    const t0 = N.dot(N.sub(base, L.p0), L.d), c0 = N.add(L.p0, N.scale(L.d, t0));
    for (const k of [dist, -dist, dist * 0.6, -dist * 0.6, dist * 0.35, -dist * 0.35]) { const p = N.add(c0, N.scale(L.d, k)); if (inside(p)) return p; }
    return c0;
  }
  /** a point on L at `dist` from x, on the side facing the view centre (for handles) */
  function handleSpot(g, L, x, dist) {
    const b = g.P.bounds(), c = [(b.x0 + b.x1) / 2, (b.y0 + b.y1) / 2];
    const sg = N.dot(N.sub(c, x), L.d) >= 0 ? 1 : -1;
    return N.add(x, N.scale(L.d, sg * dist));
  }
  const snap = m => { const r = Math.round(m * 4) / 4; return Math.abs(m - r) < 0.07 ? r : Math.round(m * 100) / 100; };
  function rowStr(a, c) {
    const term = (k, v, first) => { if (Math.abs(k) < 1e-9) return first ? '0' + v : ''; const sgn = k < 0 ? '− ' : (first ? '' : '+ '); const ak = Math.abs(k); return sgn + (Math.abs(ak - 1) < 1e-9 ? '' : fmt(ak)) + v; };
    const t1 = term(a[0], 'x', true), t2 = term(a[1], 'y', !t1 || t1 === '0x');
    return ((t1 && t1 !== '0x' ? t1 + ' ' : '') + (t2 || '')).trim() + ' = ' + fmt(c);
  }

  /** small matrix tile (world coords of a 2D viewport) */
  function tile(g, c, M, name, o) {
    o = o || {}; const a = o.alpha === undefined ? 1 : o.alpha; if (a < 0.02) return;
    const w = o.w || 0.78, h = o.h || 0.56, col = o.color || 'hud';
    g.polygon([[c[0] - w, c[1] - h], [c[0] + w, c[1] - h], [c[0] + w, c[1] + h], [c[0] - w, c[1] + h]], { fill: 'panel2', alpha: 0.95 * a, stroke: col, width: 1.4 });
    const cell = v => typeof v === 'number' ? fmt(v) : v;
    for (let i = 0; i < 2; i++) for (let j = 0; j < 2; j++) g.label(cell(M[i][j]), [c[0] + (j ? 0.5 : -0.5) * w, c[1] + (i ? -0.45 : 0.45) * h], { color: o.cellColor || 'text', size: o.size || 14, font: 'math', alpha: a, bg: false });
    if (name) g.label(name, [c[0], c[1] + h], { anchor: 'above', dy: -6, color: col, size: o.nameSize || 14, font: 'math', alpha: a, bg: false });
  }

  /** 6×6 synthetic SPD matrix and its elimination stages (computed once in state()) */
  function mlStages() {
    const rand = LA.rng(41), n = 6, B = [];
    for (let i = 0; i < n; i++) B.push(LA.rng.gaussV(rand, n));
    let M = N.addM(N.scaleM(N.matMul(N.transpose(B), B), 1 / n), N.scaleM(N.identity(n), 0.6));
    M = M.map(r => r.map(x => Math.round(x * 100) / 100));
    const stages = [LA.clone(M)]; let U = LA.clone(M);
    for (let k = 0; k < n; k++) {
      for (let i = k + 1; i < n; i++) { const f = U[i][k] / U[k][k]; for (let j = k; j < n; j++) U[i][j] -= f * U[k][j]; U[i][k] = 0; }
      stages.push(U.map(r => r.map(x => Math.round(x * 100) / 100)));
    }
    let R = 0; stages.forEach(S => S.forEach(r => r.forEach(x => { R = Math.max(R, Math.abs(x)); })));
    return { stages, R };
  }

  /* ---------- main 2D scene ---------- */
  function drawMain(g, s, ctx) {
    g.ghostGrid(); g.axes();
    const E = effE(s), S = sys(s, E), A = S.A, b = S.b;
    const dt = det2(A), x = Math.abs(dt) > 1e-6 ? N.solve(A, b) : null;
    // ghost lines: either set by a step, or the previous playground state
    let ghosts = s.ghost || [];
    if (s.showPrev && s.hist.length) { const P = sys(s, s.hist[s.hist.length - 1]); ghosts = [0, 1].map(i => ({ a: P.A[i], c: P.b[i] })); }
    ghosts.forEach(gh => drawRow(g, gh.a, gh.c, { color: 'hud', width: 1.4, alpha: 0.4, dashed: true }));
    const Ls = [0, 1].map(i => {
      const gold = s.goldRow[i], flash = i === 1 ? s.flash : 0;
      return drawRow(g, A[i], b[i], { color: gold ? 'gold' : 'hud', width: 2.2 + 3 * flash, alpha: 1 });
    });
    Ls.forEach((L, i) => { if (!L) return; const p = spotOn(g, L, x, 2.6); g.label('row ' + (i + 1), p, { color: s.goldRow[i] ? 'gold' : 'hud', size: 13, dy: -12 }); });
    // a zero row: 0 = c
    let msgY = 30;
    [0, 1].forEach(i => {
      if (Ls[i]) return;
      const bad = Math.abs(b[i]) > 0.05;
      g.hud('row ' + (i + 1) + ': 0 = ' + fmt(b[i]) + (bad ? '  →  no solution' : '  (no information)'), 14, msgY, { color: bad ? 'bad' : 'muted', size: 14 });
      if (bad) g.marker.cross([g.size.w * 0.5, msgY + 34], { screen: true, size: 26 });
      msgY += 22;
    });
    if (s.dimY && x) g.dim([-0.7, 0], [-0.7, x[1]], { label: 'y = ' + fmt(x[1]), color: 'gold', offset: 0 });
    if (x && s.showX) g.point(x, { r: 5.5 + 3 * s.pulse, color: 'gold', label: s.ptLabel || ('[' + fmt(x[0]) + '; ' + fmt(x[1]) + ']'), dx: 14, dy: -16 });
    // handles: drag a line to apply a row operation (it pivots about the crossing)
    if (s.handles && x) {
      const R = sys(s, s.E);                     // committed rows
      if (Ls[1]) g.handle('row2', handleSpot(g, Ls[1], x, 2), (q, st) => {
        const den = N.dot(R.A[0], q) - R.b[0]; if (Math.abs(den) < 1e-4) return;
        st.m[0] = LA.clamp(snap((N.dot(R.A[1], q) - R.b[1]) / den), -20, 20);
      }, { color: 'gold' });
      if (Ls[0]) g.handle('row1', handleSpot(g, Ls[0], x, 2.9), (q, st) => {
        const r2 = N.sub(R.A[1], N.scale(R.A[0], st.m[0])), c2 = R.b[1] - st.m[0] * R.b[0];
        const den = N.dot(r2, q) - c2; if (Math.abs(den) < 1e-4) return;
        st.m[1] = LA.clamp(snap((N.dot(R.A[0], q) - R.b[0]) / den), -20, 20);
      }, { color: 'gold' });
    }
    if (s.status) {
      const a21 = A[1][0], piv = A[0][0];
      const txt = Math.abs(piv) < 1e-6 ? 'pivot a₁₁ = 0: it cannot clear the entry below it' : (Math.abs(a21) < 0.05 ? 'a₂₁ = 0: row 2 is flat' : 'a₂₁ = ' + fmt(a21));
      g.hud(txt, 14, g.size.h - 22, { color: Math.abs(piv) < 1e-6 ? 'bad' : 'hud', size: 13 });
    }
  }

  /* ---------- misconception: E applied to A only ---------- */
  function drawWrong(g, s) {
    g.ghostGrid(); g.axes();
    drawRow(g, A0[0], B0[0], { color: 'hud' });
    g.label('row 1', [-0.3, 2.65], { color: 'hud', size: 13, dy: -12 });
    drawRow(g, A0[1], B0[1], { color: 'hud', width: 1.4, alpha: 0.4, dashed: true });
    const w = s.wU, r = s.rU;
    if (w > 0) {
      const aw = [3 - w, 4 - 2 * w], Lw = drawRow(g, aw, 11, { color: 'bad', width: 2.4, alpha: r > 0 ? 0.55 : 1 });
      const xw = N.solve([A0[0], aw], [5, 11]);
      if (Lw) g.label('E·A with b unchanged', spotOn(g, Lw, [3.2, 4.6], 0.01), { color: 'bad', size: 13, dx: 70 });
      if (w > 0.98 && xw) { g.marker.cross(xw, { size: 24 }); g.label('wrong system', xw, { color: 'bad', size: 13, dy: 26 }); }
    }
    if (r > 0) {
      const ar = [3 - r, 4 - 2 * r], Lr = drawRow(g, ar, 11 - 5 * r, { color: 'hud', width: 2.6 });
      if (Lr) g.label('E·A with E·b', spotOn(g, Lr, [3.6, -0.6], 0.01), { color: 'ok', size: 13, dx: -60 });
    }
    g.point(X0, { r: 5.5, color: 'gold', alpha: w > 0 && r === 0 ? 0.45 : 1, label: w > 0 && r === 0 ? 'old crossing' : '[1; 2]', dx: -14, dy: -16 });
    if (r > 0.98) g.marker.check(X0, { dx: 22, dy: 18, size: 22 });
  }

  /* ---------- inverse assembled ---------- */
  function drawInverse(g, s) {
    const u = s.invT;
    g.viewport({ x: 0, y: 0, w: 1, h: 0.42 }, { view: '2d', cam: { center: [0, 0.15], halfH: 1.25 } }, (g2, r) => {
      const slide = seg(u, 0.05, 0.38), fade = 1 - seg(u, 0.4, 0.52), show = seg(u, 0.46, 0.62);
      tile(g2, [LA.lerp(-2.3, -1.62, slide), 0], DS.map(row => row.map(v => v === -0.5 ? '−½' : v)), 'D', { alpha: fade });
      tile(g2, [0, 0], E2, 'E₂', { alpha: fade });
      tile(g2, [LA.lerp(2.3, 1.62, slide), 0], E1, 'E₁', { alpha: fade });
      if (slide < 1 && fade > 0.5) { g2.label('·', [LA.lerp(-1.15, -0.81, slide), 0], { size: 22, color: 'hud', bg: false }); g2.label('·', [LA.lerp(1.15, 0.81, slide), 0], { size: 22, color: 'hud', bg: false }); }
      tile(g2, [0, 0], AINV, 'A⁻¹ = D E₂ E₁', { alpha: show, color: 'gold', cellColor: 'gold', w: 0.95 });
      g2.label('row operations, each a matrix, multiplied together', [r.x + 12, r.y + 18], { screen: true, anchor: 'left', size: 12.5, color: 'muted', font: 'ui' });
    });
    const M = u < 0.62 ? I2() : u < 0.8 ? lerpM(I2(), A0, seg(u, 0.62, 0.78)) : lerpM(A0, I2(), seg(u, 0.82, 0.98));
    g.viewport({ x: 0, y: 0.42, w: 1, h: 0.58 }, { view: '2d', cam: { center: [0.8, 1.2], halfH: 3.4 } }, (g2, r) => {
      g2.ghostGrid(); g2.grid(M, { alpha: 0.8 });
      g2.vector(N.col(M, 0), { color: 'i', label: 'i', hat: true, width: 3 });
      g2.vector(N.col(M, 1), { color: 'j', label: 'j', hat: true, width: 3 });
      const cap = u < 0.62 ? 'the grid, untouched' : u < 0.8 ? 'A bends the grid' : 'A⁻¹ = D E₂ E₁ bends it back';
      g2.label(cap, [r.x + 12, r.y + 18], { screen: true, anchor: 'left', size: 13, color: u >= 0.8 ? 'gold' : 'hud', font: 'ui' });
    });
  }

  /* ---------- ML tie-in: 6×6 elimination heatmap ---------- */
  function drawML(g, s) {
    const st = s.ml.stages, n = 6, u = s.mlT * n, k = Math.min(n - 1, Math.floor(u)), f = seg(u - k, 0, 0.8);
    const M = u >= n ? st[n] : lerpM(st[k], st[k + 1], f);
    const R = s.ml.R, bnd = { x0: -2.7, x1: 2.7, y0: -2.9, y1: 2.5 };
    g.image(M, bnd, { cells: true, range: [-R, R], ramp: t => C.diverge(2 * t - 1) });
    const done = u >= n ? n : k + (f >= 1 ? 1 : 0);
    const cw = (bnd.x1 - bnd.x0) / n, ch = (bnd.y1 - bnd.y0) / n;
    if (u < n) { const x0 = bnd.x0 + k * cw, y1 = bnd.y1 - k * ch; g.polyline([[x0, y1], [x0 + cw, y1], [x0 + cw, y1 - ch], [x0, y1 - ch]], { color: 'gold', width: 2.4, closed: true }); }
    g.label('pivots: ' + done + '/6', [0, bnd.y1], { anchor: 'above', dy: -10, size: 15, color: 'gold', font: 'ui' });
    g.label('synthetic 6×6 symmetric matrix, cleared column by column', [0, bnd.y0], { anchor: 'below', dy: 8, size: 12.5, color: 'muted', font: 'ui' });
    g.label('teal = positive, magenta = negative, dark = 0', [0, bnd.y0], { anchor: 'below', dy: 26, size: 12, color: 'muted', font: 'ui' });
  }

  /* ---------- recap ---------- */
  function drawRecap(g) {
    const band = (i, cam, text, fn) => g.viewport({ x: 0, y: i / 3, w: 1, h: 1 / 3 }, { view: '2d', cam }, (g2, r) => { fn(g2); g2.label(text, [r.x + 12, r.y + 18], { screen: true, anchor: 'left', size: 13.5, color: 'text', bold: true }); });
    band(0, { center: [0, 0], halfH: 1.9 }, 'pivot about the answer', g2 => {
      g2.ghostGrid({ alpha: 0.6 });
      g2.lineThrough([0, 0], [1, -0.75], { len: 9, color: 'hud', width: 1.4, alpha: 0.4, dashed: true });
      g2.lineThrough([0, 0], [1, -0.2], { len: 9, color: 'hud', width: 1.6, alpha: 0.6 });
      g2.lineThrough([0, 0], [1, 0], { len: 9, color: 'gold', width: 2.4 });
      g2.lineThrough([0, 0], [1, 0.9], { len: 9, color: 'hud', width: 2 });
      g2.arc([0, 0], [1, -0.75], [1, 0], { r: 1.2, color: 'gold' });
      g2.point([0, 0], { r: 6, color: 'gold' });
    });
    band(1, { center: [1, 2], halfH: 1.9 }, 'zeros = axis-aligned lines', g2 => {
      g2.ghostGrid({ alpha: 0.6 }); g2.axes({ labels: false });
      g2.lineThrough([1, 2], [1, 0], { len: 9, color: 'gold', width: 2.4 }); g2.lineThrough([1, 2], [0, 1], { len: 9, color: 'gold', width: 2.4 });
      g2.point([1, 2], { r: 6, color: 'gold', label: '[1; 2]', dx: 30, dy: -14 });
    });
    band(2, { center: [0, -0.1], halfH: 1.25 }, 'the chain of E’s is A⁻¹', g2 => {
      const o = { w: 0.45, h: 0.4, size: 12.5, nameSize: 13 };
      tile(g2, [-2.65, -0.1], DS.map(row => row.map(v => v === -0.5 ? '−½' : v)), 'D', o);
      tile(g2, [-1.6, -0.1], E2, 'E₂', o);
      tile(g2, [-0.55, -0.1], E1, 'E₁', o);
      tile(g2, [0.5, -0.1], A0, 'A', o);
      g2.label('=', [1.35, -0.1], { size: 20, color: 'hud', font: 'math' });
      tile(g2, [2.2, -0.1], I2(), 'I', Object.assign({ color: 'gold', cellColor: 'gold' }, o));
    });
  }

  /* ---------- playground operations ---------- */
  function commit(s) { const E = effE(s); s.hist.push(LA.clone(s.E)); if (s.hist.length > 30) s.hist.shift(); s.E = E; s.m = [0, 0]; }
  function applyOp(s, Op) { commit(s); s.E = N.matMul(Op, s.E); s.ops++; }
  function autoSolve(s) {
    commit(s); s.hist.pop();
    for (let k = 0; k < 2; k++) {
      let A = sys(s, s.E).A;
      if (Math.abs(A[k][k]) < 1e-9) { const o = 1 - k; if (Math.abs(A[o][k]) > 1e-9 && o > k) { s.hist.push(LA.clone(s.E)); s.E = N.matMul([[0, 1], [1, 0]], s.E); s.ops++; A = sys(s, s.E).A; } else continue; }
      for (let i = 0; i < 2; i++) { if (i === k || Math.abs(A[i][k]) < 1e-9) continue; const f = A[i][k] / A[k][k]; const Op = I2(); Op[i][k] = -f; s.hist.push(LA.clone(s.E)); s.E = N.matMul(Op, s.E); s.ops++; A = sys(s, s.E).A; }
    }
    const A = sys(s, s.E).A;
    if (Math.abs(det2(A)) > 1e-9) { s.hist.push(LA.clone(s.E)); s.E = N.matMul([[1 / A[0][0], 0], [0, 1 / A[1][1]]], s.E); s.ops++; }
  }
  function load(s, A, b) { s.A0 = LA.clone(A); s.b0 = b.slice(); s.E = I2(); s.m = [0, 0]; s.hist = []; s.ops = 0; }
  function base(s) {
    s.scene = 'main'; s.view = '2d'; s.cam = CAM(); load(s, A0, B0); s.ghost = []; s.goldRow = [false, false]; s.flash = 0; s.pulse = 0;
    s.showX = true; s.ptLabel = null; s.dimY = false; s.handles = true; s.showPrev = true; s.status = false; s.panel = 'live';
  }

  LA.registerLesson({
    id: 'p3-02-elimination',
    part: 3,
    title: 'Gaussian elimination, visually',
    subtitle: 'Each row operation rotates one line about the crossing, so the answer never moves.',
    minutes: 8,
    prereqs: ['p3-01-systems-row-column', 'p2-04-composition'],
    tags: ['extension'],
    badge: { drawnIn: '2D', realSize: 'LU solvers run elimination on 10⁴×10⁴ systems; np.linalg.solve is this', transfers: 'a row operation is always “replace one equation by itself plus a multiple of another”; the solution set never changes, in any size' },
    cantShow: ['pivoting for numerical stability (swapping rows to avoid tiny pivots): only its geometry, a swap relabels the lines, is shown', 'cost: elimination takes about n³/3 operations; the picture has no sense of cost'],
    terms: [
      { term: 'row operation', def: 'Add a multiple of one equation to another, scale an equation, or swap two equations.' },
      { term: 'pivot', def: 'The entry used to clear the entries below (or above) it in its column.' },
      { term: 'back substitution', def: 'Solving a triangular system from the bottom equation up, one unknown at a time.' },
      { term: 'elimination matrix', def: 'The matrix that performs one row operation when it multiplies A (and b) from the left.' },
    ],
    atlas: ['linear-regression'],
    thumbnail: 1,

    state() {
      return {
        view: '2d', cam: CAM(), scene: 'main', panel: 'sys',
        A0: LA.clone(A0), b0: B0.slice(), E: I2(), m: [0, 0], hist: [], ops: 0,
        ghost: [], goldRow: [false, false], flash: 0, pulse: 0, showX: true, ptLabel: '?', dimY: false,
        handles: false, showPrev: false, status: false, mOp: 1, hlPivot: false,
        wU: 0, rU: 0, invT: 0, mlT: 0, ml: mlStages(),
      };
    },

    render(g, s, ctx) {
      if (s.scene === 'recap') { drawRecap(g); return; }
      if (s.scene === 'inverse') { drawInverse(g, s); return; }
      if (s.scene === 'ml') { drawML(g, s); return; }
      if (s.scene === 'wrong') { drawWrong(g, s); return; }
      drawMain(g, s, ctx);
    },

    math(s, tex0) {
      const tex = safeTex(tex0), P = s.panel;
      const xy = tex.vec(['x', 'y']);
      if (P === 'recap') return tex.block(tex.eq(tex.sym('E', { sub: 'k' }), tex.op(' ⋯ '), tex.sym('E', { sub: '1' }), tex.sym('A'), tex.op(' = '), tex.sym('I')));
      if (P === 'ml') return tex.block(tex.eq(tex.T(tex.sym('X')), tex.sym('X'), tex.sym('w'), tex.op(' = '), tex.T(tex.sym('X')), tex.sym('y', { color: 'b' })), tex.eq(tex.op('→ '), tex.text('solve: eliminate, then back-substitute')));
      if (P === 'wrong') return tex.lines(
        tex.eq(tex.sym('E'), tex.op('·'), tex.sym('A'), tex.op(', '), tex.sym('b', { color: 'b' }), tex.text('  ✕ wrong system', { color: 'bad' })),
        tex.eq(tex.sym('E'), tex.op('·'), tex.sym('A'), tex.op(', '), tex.sym('E'), tex.op('·'), tex.sym('b', { color: 'b' }), tex.text('  ✓ same crossing', { color: 'ok' })));
      if (P === 'inverse') return tex.block(
        tex.eq(tex.inv(tex.sym('A')), tex.op(' = '), tex.sym('D'), tex.sym('E', { sub: '2' }), tex.sym('E', { sub: '1' }), tex.op(' = '),tex.mat(AINV, { colors: 'cols' })),
        tex.eq(tex.inv(tex.sym('A')), tex.sym('A'), tex.op(' = '), tex.sym('I')));
      const E = effE(s), S = sys(s, E);
      const colorFn = (i, j) => Math.abs(S.A[i][j]) < 0.01 && i !== j ? 'muted' : (j === 0 ? 'i' : 'j');
      const hl = s.hlPivot ? [0, 0] : undefined;
      if (P === 'sys') return tex.block(tex.eq(tex.mat(S.A, { colors: 'cols' }), xy, tex.op('='), tex.vec(S.b, { color: 'b' })));
      if (P === 'E1' || P === 'E2' || P === 'live') {
        const name = P === 'E1' ? [tex.sym('E', { sub: '1' })] : P === 'E2' ? [tex.sym('E', { sub: '2' }), tex.sym('E', { sub: '1' })] : [tex.sym('E')];
        const parts = [tex.eq.apply(null, name.concat([tex.op(' = '), tex.mat(E, { digits: 2 })])),
          tex.eq.apply(null, name.concat([tex.sym('A'), tex.op(' = '), tex.mat(S.A, { colors: colorFn, highlight: hl })])),
          tex.eq.apply(null, name.concat([tex.sym('b', { color: 'b' }), tex.op(' = '), tex.vec(S.b, { color: 'b' })]))];
        if (P === 'E2' && s.solved) parts.push(tex.eq(tex.text('so'), tex.sym('x'), tex.op(' = '), tex.num(1, { color: 'gold' }), tex.op(', '), tex.sym('y'), tex.op(' = '), tex.num(2, { color: 'gold' })));
        if (P === 'live') parts.push(tex.eq(tex.text('row operations used: '), tex.num(s.ops + (s.m[0] || s.m[1] ? 1 : 0))));
        return tex.block.apply(null, parts);
      }
      if (P === 'readY') return tex.block(tex.eq(tex.mat(S.A, { colors: colorFn }), xy, tex.op('='), tex.vec(S.b, { color: 'b' })), tex.eq(tex.op('−'), tex.num(2), tex.sym('y'), tex.op(' = '), tex.num(-4, { color: 'b' }), tex.op(' ⇒ '), tex.sym('y'), tex.op(' = '), tex.num(2, { color: 'gold' })));
      if (P === 'scale') return tex.block(
        tex.eq(tex.sym('D'), tex.op(' = '), tex.mat([[1, 0], [0, '−½']])),
        tex.eq(tex.sym('D'), tex.sym('E', { sub: '2' }), tex.sym('E', { sub: '1' }),tex.sym('A'), tex.op(' = '), tex.mat(S.A, { colors: colorFn })),
        tex.eq(tex.sym('D'), tex.sym('E', { sub: '2' }), tex.sym('E', { sub: '1' }),tex.sym('b', { color: 'b' }), tex.op(' = '), tex.vec(S.b, { color: 'b' })));
      return '';
    },

    steps: [
      { title: 'Start: two lines, one crossing', duration: 3,
        narration: 'Take x + 2y = 5 and 3x + 4y = 11. The row picture shows two lines crossing at the gold point. Elimination is a way to <i>find</i> that point without guessing: tilt the lines, one at a time, until the answer can be read straight off the axes.',
        setup(s) { s.scene = 'main'; s.view = '2d'; s.cam = CAM(); load(s, A0, B0); s.ghost = []; s.goldRow = [false, false]; s.flash = 0; s.pulse = 0; s.showX = true; s.ptLabel = '?'; s.dimY = false; s.handles = false; s.showPrev = false; s.status = false; s.panel = 'sys'; s.hlPivot = false; },
        at(s, t) { s.pulse = seg.pulse(t, 0.4, 0.9); } },

      { title: 'One row operation: the line rotates about the answer', duration: 4.5,
        narration: 'A <b>row operation</b>: replace row 2 by row 2 − 3·row 1. Watch line 2 <i>rotate</i> about the gold point: any point on both old lines satisfies the new equation too, so the crossing cannot move. The 1 used to clear the 3 below it is the <b>pivot</b>. Row 2 becomes 0x − 2y = −4.',
        setup(s) { s.E = I2(); s.ghost = [{ a: A0[1], c: B0[1] }]; s.panel = 'E1'; s.pulse = 0; s.ptLabel = '?'; },
        at(s, t) { s.E = lerpM(I2(), E1, seg(t, 0.1, 0.9)); s.hlPivot = t > 0.05 && t < 0.95 && Math.floor(t * 6) % 2 === 0; } },

      { title: 'The new line is horizontal: read y', duration: 3,
        narration: '0x − 2y = −4 has no x in it, so its line is horizontal: y = 2. One coordinate of the answer is now visible by eye. That is the whole point of making an entry zero: a zero below the pivot lays the line flat.',
        setup(s) { s.E = LA.clone(E1); s.ghost = []; s.hlPivot = false; s.panel = 'readY'; },
        at(s, t) { s.goldRow = [false, t > 0.2]; s.dimY = t > 0.35; s.ptLabel = t > 0.6 ? '[?; 2]' : '?'; s.pulse = seg.pulse(t, 0.6, 0.95); } },

      { title: 'Second operation: make line 1 vertical', duration: 4,
        narration: 'You could stop and use <b>back substitution</b>: put y = 2 into row 1 to get x = 1. Or clear the 2 above the second pivot: row 1 ← row 1 + row 2. Line 1 rotates about the same gold point until it is vertical, x = 1. The answer is written on the axes.',
        setup(s) { s.E = LA.clone(E1); s.goldRow = [false, true]; s.dimY = false; s.ghost = [{ a: A0[0], c: B0[0] }]; s.panel = 'E2'; s.solved = false; s.ptLabel = '[?; 2]'; },
        at(s, t) { s.E = lerpM(E1, E21, seg(t, 0.1, 0.8)); s.goldRow = [t > 0.82, true]; s.solved = t > 0.85; s.ptLabel = t > 0.85 ? '[1; 2]' : '[?; 2]'; s.pulse = seg.pulse(t, 0.82, 1); } },

      { title: 'Scaling a row changes nothing you can see', duration: 2.5,
        narration: 'Multiply row 2 by −½: the equation −2y = −4 becomes y = 2. The line does not even move: scaling an equation rescales its numbers, not its points. Now the matrix is the identity and the orange ' + bb + ' has become the answer [1; 2].',
        setup(s) { s.E = LA.clone(E21); s.ghost = []; s.goldRow = [true, true]; s.solved = true; s.ptLabel = '[1; 2]'; s.panel = 'scale'; s.pulse = 0; },
        at(s, t) { s.E = t < 0.45 ? LA.clone(E21) : LA.clone(AINV); s.flash = seg.pulse(t, 0.35, 0.75); } },

      { title: 'The operations are a matrix: the inverse, assembled', duration: 4,
        narration: 'Every step was a left-multiplication by an <b>elimination matrix</b>. Chain them: (D E₂ E₁) A = I, so D E₂ E₁ is exactly A⁻¹. Elimination never computes the inverse first; it builds it as a side effect of tilting lines. Real solvers stop at the triangular stage and back-substitute — cheaper.',
        setup(s) { s.scene = 'inverse'; s.flash = 0; s.panel = 'inverse'; s.invT = 0; },
        at(s, t) { s.invT = t; } },

      { title: 'Misconception: “row operations change the solution”', duration: 4, misconception: true,
        narration: '✕ Someone replaces row 2 by row 2 − row 1 on the left side only, leaving ' + bb + ' alone. Now the red line really moves, and the crossing jumps away from the gold point. A row operation acts on the whole row, A and ' + bb + ' together, or it is not a row operation.',
        setup(s) { s.scene = 'wrong'; s.cam = CAMW(); s.panel = 'wrong'; s.wU = 0; s.rU = 0; },
        at(s, t) { s.wU = seg(t, 0.05, 0.45); s.rU = seg(t, 0.6, 0.95); } },

      { title: 'Why ML cares: solve, don’t invert', duration: 3.5, mlTie: true,
        narration: '<i>solve(A, b)</i> in every numerics library is elimination with row swaps: column by column, the entries under each pivot are cleared, as in this synthetic 6×6 matrix. It is cheaper and more accurate than forming A⁻¹. When you fit a linear regression by the normal equations, the library eliminates; it never inverts.',
        setup(s) { s.scene = 'ml'; s.cam = { center: [0, -0.2], halfH: 3.7 }; s.panel = 'ml'; s.mlT = 0; },
        at(s, t) { s.mlT = seg(t, 0.05, 0.9, 'linear'); } },

      { title: 'Recap', duration: 2.5, recap: true,
        narration: 'A row operation rotates one line about the common crossing, so the answer never moves. Zero entries make lines axis-aligned and the answer readable. Each operation is a matrix E; their product is A⁻¹, built on the way.',
        cards: ['<b>Pivot about the answer.</b> Row₂ ← row₂ − m·row₁ tilts line 2 about the crossing; the solution never moves.', '<b>Zeros = axis-aligned lines.</b> Clearing an entry lays a line flat (or upright), so a coordinate can be read off.', '<b>The E-chain is A⁻¹.</b> Each operation is a matrix; D E₂ E₁ A = I.'],
        setup(s) { s.scene = 'recap'; s.cam = CAM(); s.panel = 'recap'; },
        at(s, t) { } },
    ],

    playground: {
      intro: '<p>Drag the gold handle on a line to apply a row operation live: the line pivots about the crossing. Or set the multiplier <b>m</b> and press an operation. Dashed lines show the previous state.</p>',
      setup(s) { base(s); },
      controls: [
        { type: 'slider', bind: 'mOp', min: -4, max: 4, step: 0.25, label: 'multiplier m', color: 'gold' },
        { type: 'preset', label: 'Row operation', items: [
          { label: 'row₂ ← row₂ − m·row₁', apply(s) { const Op = I2(); Op[1][0] = -s.mOp; applyOp(s, Op); } },
          { label: 'row₁ ← row₁ − m·row₂', apply(s) { const Op = I2(); Op[0][1] = -s.mOp; applyOp(s, Op); } },
          { label: 'Swap rows', apply(s) { applyOp(s, [[0, 1], [1, 0]]); } },
          { label: 'Scale row 2 by m', apply(s) { if (Math.abs(s.mOp) > 1e-9) applyOp(s, [[1, 0], [0, s.mOp]]); } },
        ] },
        { type: 'button', label: 'Undo', onClick(s) { if (s.m[0] || s.m[1]) { s.m = [0, 0]; return; } if (s.hist.length) { s.E = s.hist.pop(); s.ops = Math.max(0, s.ops - 1); } } },
        { type: 'button', label: 'Auto-solve', primary: true, onClick(s) { autoSolve(s); } },
        { type: 'preset', label: 'Systems', items: [
          { label: 'Step-2 system', apply(s) { load(s, A0, B0); } },
          { label: 'Needs a swap', apply(s) { load(s, SWAP_A, SWAP_B); } },
          { label: 'Singular', apply(s) { load(s, SING_A, SING_B); } },
        ] },
      ],
    },

    checks: [
      { kind: 'manipulate',
        prompt: 'Drag the gold handles (each applies a row operation) until one line is horizontal and the other vertical.',
        setup(s) { base(s); s.showPrev = false; s.ghost = [{ a: A0[0], c: B0[0] }, { a: A0[1], c: B0[1] }]; },
        judge(s) { const A = sys(s).A, e = Math.abs(A[0][1]) + Math.abs(A[1][0]); const ok = Math.abs(A[0][1]) < LA.TOL.scalar && Math.abs(A[1][0]) < LA.TOL.scalar && Math.abs(det2(A)) > 0.05; return { ok, progress: Math.max(0, 1 - e / 5) }; },
        hint: 'Clear the entry below the first pivot first (drag line 2 until it is flat), then the one above the second (drag line 1 upright).',
        solution(s) { s.m = [3, -1]; } },
      { kind: 'entry',
        prompt: 'For the system with <b>A</b> = [[2, 1], [1, 3]], type the elimination matrix E that zeroes the entry under the first pivot. Line 2 follows E·A live.',
        bind: 'E', answer: [[1, 0], [-0.5, 1]], tol: LA.TOL.entry, label: 'E', colors: 'text',
        setup(s) { base(s); load(s, [[2, 1], [1, 3]], [4, 7]); s.handles = false; s.showPrev = false; s.ghost = [{ a: [1, 3], c: 7 }]; s.panel = 'live'; } },
      { kind: 'entry',
        prompt: 'In this system the first pivot is 0, so it cannot clear anything. Type the row-operation matrix E that <i>swaps</i> the two rows (watch the labels and the pivot).',
        bind: 'E', answer: [[0, 1], [1, 0]], tol: LA.TOL.entry, label: 'E', colors: 'text',
        setup(s) { base(s); load(s, SWAP_A, SWAP_B); s.handles = false; s.showPrev = false; s.status = true; s.panel = 'live'; } },
    ],
  });
})();
