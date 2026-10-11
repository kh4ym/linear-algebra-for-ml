/* p2-04 — Matrix multiplication is composition. */
(function () {
  'use strict';
  const N = LA.num, lerpM = LA.lerpM, lerpV = LA.lerpV, seg = LA.seg, fmt = LA.fmt;
  const I2 = () => [[1, 0], [0, 1]];
  const R = [[0, -1], [1, 0]], S = [[1, 1], [0, 1]], C3 = [[1.5, 0], [0, 1]];
  const SR = N.matMul(S, R), RS = N.matMul(R, S);
  const W1 = [[1, 0.5], [0, 1]], W2 = N.rot2(Math.PI / 6), W3 = [[1.2, 0], [0, 0.8]];
  const W = N.matMul(W3, N.matMul(W2, W1));
  const FLIPX = [[1, 0], [0, -1]];
  const ihat = '<span class="c-i">î</span>', jhat = '<span class="c-j">ĵ</span>', vv = '<span class="c-v">v</span>';
  const SUB = ['₁', '₂', '₃'];

  /* numbers in the math panel as plain colored spans (tex.num's in-place update can write into the wrong matrix cell) */
  const tn = (tex, x, color) => '<span class="' + tex.colorClass(color) + '">' + (typeof x === 'number' ? fmt(x) : x) + '</span>';
  const r2 = M => M.map(r => r.map(x => Math.round(x * 100) / 100));

  /** tween between two matrices: through rotations/stretches when both keep orientation, entrywise otherwise */
  function tw(A, B, t) { return N.det(A) * N.det(B) > 1e-6 ? N.matPolar(A, B, t) : lerpM(A, B, t); }
  /** grid matrix after progress m ∈ [0, 2] of "First, then Second" */
  function stage(First, Second, m) { return m <= 1 ? tw(I2(), First, Math.max(0, m)) : N.matMul(tw(I2(), Second, Math.min(1, m - 1)), First); }

  function fitCam(g, fw, fh, c) { const aspect = (g.size.w * fw) / (g.size.h * fh); return { center: c.center.slice(), halfH: Math.max(c.halfH, c.halfW / aspect) }; }
  function sline(g, a, b, o) { g.line(g.P.toWorld(a[0], a[1]), g.P.toWorld(b[0], b[1]), o); }
  function srect(g, x, y, w, h, o) { const W = (a, b) => g.P.toWorld(a, b); g.polygon([W(x, y), W(x + w, y), W(x + w, y + h), W(x, y + h)], o); }
  /** bracketed matrix at screen px; cells[i][j] = {t, c, hl, a}; returns its width */
  function hudMat(g, cells, x, y, o) {
    o = o || {}; const cw = o.cw || 46, ch = o.ch || 28, size = o.size || 15, col = o.color || 'hud';
    const rows = cells.length, cols = cells[0].length, w = cols * cw, h = rows * ch, st = { color: col, width: 1.5, alpha: o.alpha };
    if (o.panel !== false) {   // opaque backing so grid lines never run through the numbers
      const nameW = o.name ? o.name.length * (size + 1) * 0.55 + 12 : 0, pa = o.alpha === undefined ? 1 : o.alpha;
      if (pa > 0.02) g.rect(x - nameW - 8, y - 6, w + nameW + 24, h + 12, { fill: 'panel', alpha: 0.88 * pa });
    }
    const br = (bx, d) => { sline(g, [bx + 6 * d, y], [bx, y], st); sline(g, [bx, y], [bx, y + h], st); sline(g, [bx, y + h], [bx + 6 * d, y + h], st); };
    br(x, 1); br(x + w + 8, -1);
    for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) {
      const c = cells[i][j]; if (!c) continue;
      g.label(c.t, [x + 4 + cw * (j + 0.5), y + ch * (i + 0.5)], { screen: true, color: c.c || col, size: c.hl ? size + 2 : size, font: 'math', bold: !!c.hl, alpha: o.alpha === undefined ? c.a : o.alpha, bg: false });
    }
    if (o.name) g.label(o.name, [x - 8, y + h / 2], { screen: true, color: o.nameColor || 'hud', size: size + 1, anchor: 'right', font: 'math', alpha: o.alpha, bg: false });
    return w + 8;
  }
  const cellsOf = (M, known) => M.map((r, i) => r.map((x, j) => ({ t: known && !known(i, j) ? '?' : fmt(Math.round(x * 100) / 100), c: j === 0 ? 'i' : 'j' })));

  /** the staged picture: ghost grid, dotted intermediate grid, moving grid, î, ĵ and optionally v's hops */
  function drawStaged(g, First, Second, m, o) {
    o = o || {};
    const M = stage(First, Second, m);
    g.ghostGrid();
    if (m >= 1 && o.midGhost !== false) g.grid(First, { color: 'muted', axisColor: 'muted', dash: [2, 5], alpha: 0.5, range: 8 });
    g.grid(M, { alpha: o.gridAlpha === undefined ? 0.9 : o.gridAlpha, range: 8 });
    if (o.v) {
      const v0 = o.v, v1 = N.matVec(First, v0);
      if (m > 0.02) g.vector(v0, { color: 'v', dashed: true, width: 1.8, alpha: 0.45, label: 'v', labelSize: 12 });
      if (m >= 1) g.vector(v1, { color: 'v', dashed: true, width: 1.8, alpha: 0.55, label: o.v1Label || 'Av', labelSize: 12 });
      // label the moving arrow only at rest, so it never sits on top of the dashed "v" / "Av" labels
      g.vector(N.matVec(M, v0), { color: 'v', label: m >= 1.98 ? (o.v2Label || 'BAv') : m <= 0.02 ? 'v' : undefined, width: 3 });
    }
    if (o.midArrows && m >= 1)   // dashed î where the first map left it (a 90° turn maps the grid onto itself, so the grid ghost is invisible)
      g.vector(N.col(First, 0), { color: 'i', dashed: true, width: 2, alpha: 0.55, label: 'î after R', labelSize: 12, dx: -44, dy: -4 });
    if (o.showIJ !== false) {
      g.vector(N.col(M, 0), { color: 'i', label: 'i', hat: true, width: 3, alpha: o.dimJ === 'i' ? 0.25 : 1 });
      g.vector(N.col(M, 1), { color: 'j', label: 'j', hat: true, width: 3, alpha: o.dimJ === 'j' ? 0.25 : 1 });
    }
    return M;
  }

  /* ---------------- shape tiles (px layout, as in p2-02) ---------------- */
  function tile(g, x, y, parts, color, k) {
    const w = 108 * k, h = 56 * k, size = Math.max(11, 19 * k), W = (a, b) => g.P.toWorld(a, b);
    g.polygon([W(x - w / 2, y - h / 2), W(x + w / 2, y - h / 2), W(x + w / 2, y + h / 2), W(x - w / 2, y + h / 2)], { fill: color, alpha: 0.16, stroke: color, width: 1.4 });
    parts.forEach((p, i) => g.label(p.t, [x + (i - 1) * 28 * k, y], { screen: true, color: p.c || color, size, bold: true, font: 'ui', alpha: p.a, bg: false }));
  }
  function renderTiles(g, u, rect, k) {
    const TW = 108 * k, cx = rect.x + rect.w / 2, pc = cx - 60 * k, cy = rect.y + rect.h / 2;
    const row = (dy, L, Rt, res, t0, t1, mismatch) => {
      const y = cy + dy * 95 * k, a = seg(u, t0, t1, 'inOut');
      let xl = pc - TW / 2 - LA.lerp(90, 0, a) * k, xr = pc + TW / 2 + LA.lerp(90, 0, a) * k;
      if (mismatch) { const back = seg(u, t1, t1 + 0.1, 'out'); xl -= (10 + 50 * back) * k; xr += (10 + 50 * back) * k; }
      const innerA = mismatch ? 1 : 1 - seg(u, t1, t1 + 0.08, 'linear'), badC = mismatch && u > t1 ? 'bad' : undefined;
      tile(g, xl, y, [{ t: L[0] }, { t: '×' }, { t: L[1], a: innerA, c: badC }], 'teal', k);
      tile(g, xr, y, [{ t: Rt[0], a: innerA, c: badC }, { t: '×' }, { t: Rt[1] }], 'w', k);
      if (!mismatch && u > t1 + 0.08) { g.label('→', [pc + TW + 22 * k, y], { screen: true, size: 20 * k + 4, color: 'hud', bg: false }); tile(g, pc + 1.5 * TW + 44 * k, y, [{ t: res[0] }, { t: '×' }, { t: res[1] }], 'ok', k); }
      if (mismatch && u > t1 + 0.02) { g.marker.cross([pc, y], { screen: true, size: 26 * k }); g.label(L[1] + ' ≠ ' + Rt[0] + ': the maps do not chain', [pc, y + 30 * k + 12], { screen: true, color: 'bad', size: Math.max(11, 13 * k) }); }
    };
    row(-0.6, ['2', '3'], ['3', '2'], ['2', '2'], 0.02, 0.4, false);
    row(0.8, ['2', '3'], ['2', '3'], null, 0.6, 0.75, true);
    if (k >= 1) g.hud('first map’s output size must equal the next map’s input size', cx, rect.y + rect.h - 24, { anchor: 'center', color: 'muted' });
  }

  /* ---------------- render pieces ---------------- */
  function renderMain(g, s, ctx) {
    const st = s.st;
    if (s.mode === 'stage') {
      const M = drawStaged(g, s.First, s.Second, s.m, { v: s.showV ? s.v : null, v1Label: 'Rv', v2Label: 'S(Rv)', dimJ: s.dimJ, gridAlpha: s.gridAlpha, midGhost: s.midGhost, showIJ: !s.showV, midArrows: s.mathMode === 'stage' });
      if (s.track) drawTrack(g, s);
      if (s.hud) drawHud(g, s, M);
      return;
    }
    if (s.mode === 'direct') {
      g.ghostGrid();
      const M = lerpM(I2(), s.Target, s.m);
      g.grid(M, { range: 8 });
      g.vector(N.col(M, 0), { color: 'i', label: 'i', hat: true, width: 3 });
      g.vector(N.col(M, 1), { color: 'j', label: 'j', hat: true, width: 3 });
      if (s.hud) drawHud(g, s, M);
      return;
    }
    if (s.mode === 'columns') return renderColumns(g, s);
    if (s.mode === 'entries') return renderEntries(g, s);
    if (s.mode === 'assoc') return renderAssoc(g, s);
    if (s.mode === 'tiles') return renderTiles(g, s.tileT, { x: 0, y: 0, w: g.size.w, h: g.size.h }, 1);
    if (s.mode === 'layers') return renderLayers(g, s);
    if (s.mode === 'split') return renderSplit(g, s, ctx);
    if (s.mode === 'recap') return renderRecap(g, s);
    if (s.mode === 'check1') return renderCheck1(g, s);
    if (s.mode === 'check2') return renderCheck2(g, s);
    if (s.mode === 'check3') return renderCheck3(g, s);
  }

  /** the path of î or ĵ through both hops, with a ghost at the intermediate landing spot */
  function drawTrack(g, s) {
    const j = s.track === 'i' ? 0 : 1, color = s.track === 'i' ? 'i' : 'j';
    const e = j === 0 ? [1, 0] : [0, 1], mid = N.matVec(R, e), end = N.matVec(S, mid);
    const pts = []; for (let k = 0; k <= 20; k++) pts.push(N.matVec(N.rot2((Math.PI / 2) * k / 20), e));
    g.polyline(pts, { color, dashed: true, width: 1.4, alpha: 0.7 });
    if (N.dist(mid, end) > 1e-6) g.line(mid, end, { color, dashed: true, width: 1.4, alpha: 0.7 });
    if (s.m >= 1) {
      g.vector(mid, { color, dashed: true, width: 2, alpha: 0.55 });
      g.label(j === 0 ? 'after R: [0; 1]' : 'after R: [−1; 0]', mid, { color, size: 12.5, dx: j === 0 ? -58 : -72, dy: j === 0 ? -6 : -16 });
    }
    if (s.m >= 1.98) g.label(j === 0 ? 'after S: [1; 1]' : 'S leaves it: stays [−1; 0]', end, { color, size: 12.5, dx: j === 0 ? 62 : -20, dy: j === 0 ? -12 : 22 });
  }

  /** on-canvas product bracket (top right) */
  function drawHud(g, s, M) {
    const x = g.size.w - 130, y = 18;
    hudMat(g, cellsOf(s.hud.M, (i, j) => s.hud.known[j]), x, y, { cw: 48, name: s.hud.name });
  }

  /* step 5's equation on the canvas: S(R v) with "first" under R and "second" under S */
  function drawRTL(g, s) {
    const cx = g.size.w / 2, y = 34, sz = 24;
    const parts = [['S', 'hud'], ['(', 'muted'], ['R', 'hud'], ['v', 'v'], [')', 'muted'], ['=', 'muted'], ['(', 'muted'], ['S', 'hud'], ['R', 'hud'], [')', 'muted'], ['v', 'v']];
    const dx = [0, 22, 40, 64, 84, 116, 146, 166, 190, 212, 234], x0 = cx - 117;
    parts.forEach((p, i) => g.label(p[0], [x0 + dx[i], y], { screen: true, size: sz, color: p[1], font: 'math', bg: false }));
    const a1 = seg(s.m, 0.3, 1), a2 = seg(s.m, 1.3, 2);
    if (a1 > 0) g.label('↑ first', [x0 + 46, y + 30], { screen: true, size: 13, color: 'gold', alpha: a1, bg: false });
    if (a2 > 0) g.label('second ↑', [x0 - 14, y + 30], { screen: true, size: 13, color: 'teal', alpha: a2, bg: false });
  }

  /* step 6: column j of SR = S · (column j of R), drawn as the column-view recipe on S's grid */
  function renderColumns(g, s) {
    const u = s.colT;
    g.ghostGrid();
    g.grid(S, { alpha: 0.55, range: 8 });
    const s1 = N.col(S, 0), s2 = N.col(S, 1);
    g.vector(s1, { color: 'i', width: 1.8, head: 8, alpha: 0.6, label: 's₁', labelSize: 12 });
    g.vector(s2, { color: 'j', width: 1.8, head: 8, alpha: 0.6, label: 's₂', labelSize: 12 });
    g.hud('S’s columns s₁, s₂ (thin) · R’s columns pushed through S (thick)', 14, g.size.h - 22, { color: 'muted' });
    const one = (j, t, color) => {
      if (t <= 0) return;
      const r = N.col(R, j);   // recipe: r[0]·s₁ + r[1]·s₂
      const p1 = N.scale(s1, r[0] * seg(t, 0, 0.45)), p2 = N.add(N.scale(s1, r[0]), N.scale(s2, r[1] * seg(t, 0.45, 0.9)));
      if (Math.abs(r[0]) > 0 && t > 0) g.vector([0, 0], p1, { color: 'i', width: 2.2, head: 9, alpha: 0.8, label: t > 0.45 ? fmt(r[0]) + '·s₁' : undefined, labelAt: 0.5, labelSize: 12 });
      if (Math.abs(r[1]) > 0 && t > 0.45) g.vector(N.scale(s1, r[0]), p2, { color: 'j', width: 2.2, head: 9, alpha: 0.8, label: t > 0.9 ? fmt(r[1]) + '·s₂' : undefined, labelAt: 0.5, labelSize: 12 });
      if (t > 0.9) { const res = N.matVec(S, r); g.vector(res, { color, width: 3.4, label: 'col' + SUB[j] + '(SR)', labelSize: 13, labelOffset: 26 }); g.marker.ring(res, { r: 12, color }); }
    };
    one(0, seg(u, 0, 0.48, 'linear'), 'i');
    one(1, seg(u, 0.5, 0.98, 'linear'), 'j');
    hudMat(g, cellsOf(SR, (i, j) => (j === 0 ? u > 0.44 : u > 0.94)), g.size.w - 130, 18, { cw: 48, name: 'SR =' });
  }

  /* step 7: entry (i, j) = row i of S · column j of R */
  function renderEntries(g, s) {
    const u = s.entT, k = Math.min(3, Math.floor(seg(u, 0.03, 0.97, 'linear') * 4)), i = k >> 1, j = k & 1, done = u >= 0.97;
    g.ghostGrid(); g.grid(SR, { alpha: 0.25, range: 8 });
    const cx = g.size.w / 2, y = 60, cw = 44, ch = 30;
    const x0 = cx - 190;
    // highlight bars behind row i of S and column j of R
    if (!done) {
      srect(g, x0 - 2, y + ch * i + 2, 2 * cw + 12, ch - 4, { fill: 'gold', alpha: 0.18 });
      srect(g, x0 + 2 * cw + 50 + cw * j + 2, y - 2, cw, 2 * ch + 4, { fill: 'gold', alpha: 0.18 });
    }
    hudMat(g, cellsOf(S), x0, y, { cw, ch, name: 'S', panel: false });
    hudMat(g, cellsOf(R), x0 + 2 * cw + 48, y, { cw, ch, name: 'R', panel: false });
    g.label('=', [x0 + 4 * cw + 92, y + ch], { screen: true, size: 20, color: 'hud', bg: false });
    const filled = (a, b) => done || a * 2 + b < k || (a * 2 + b === k && seg(u, 0.03, 0.97, 'linear') * 4 - k > 0.55);
    hudMat(g, SR.map((r, a) => r.map((x, b) => filled(a, b) ? { t: fmt(x), c: b === 0 ? 'i' : 'j', hl: a === i && b === j && !done } : { t: '·', c: 'muted' })), x0 + 4 * cw + 112, y, { cw, ch, panel: false });
    const row = S[i], col = N.col(R, j), p = (x) => (x < 0 ? '(' + fmt(x) + ')' : fmt(x));
    const text = done ? 'four dot products fill SR = [[1, −1],[1, 0]]' : 'row ' + (i + 1) + ' of S · column ' + (j + 1) + ' of R = ' + p(row[0]) + '·' + p(col[0]) + ' + ' + p(row[1]) + '·' + p(col[1]) + ' = ' + fmt(N.dot(row, col));
    g.hud(text, cx, y + 2 * ch + 34, { anchor: 'center', color: 'gold', size: 15 });
  }

  /* step 9: C, then R, then S; the brackets slide from (SR)C to S(RC) */
  function renderAssoc(g, s) {
    const m = s.m3;   // 0..3
    const M = m <= 1 ? lerpM(I2(), C3, m) : m <= 2 ? N.matMul(tw(I2(), R, m - 1), C3) : N.matMul(lerpM(I2(), S, m - 2), N.matMul(R, C3));
    g.ghostGrid();
    if (m >= 1) g.grid(C3, { color: 'muted', axisColor: 'muted', dash: [2, 5], alpha: 0.4, range: 8 });
    if (m >= 2) g.grid(N.matMul(R, C3), { color: 'muted', axisColor: 'muted', dash: [2, 5], alpha: 0.4, range: 8 });
    g.grid(M, { range: 8 });
    g.vector(N.col(M, 0), { color: 'i', label: 'i', hat: true, width: 3 });
    g.vector(N.col(M, 1), { color: 'j', label: 'j', hat: true, width: 3 });
    const cx = g.size.w / 2, y = 34, b = s.brk, x0 = cx - 50;
    [['S', 0], ['R', 40], ['C', 80]].forEach(p => g.label(p[0], [x0 + p[1], y], { screen: true, size: 26, color: 'hud', font: 'math', bg: false }));
    g.label('(', [x0 + LA.lerp(-20, 20, b), y], { screen: true, size: 28, color: 'gold', font: 'math', bg: false });
    g.label(')', [x0 + LA.lerp(60, 100, b), y], { screen: true, size: 28, color: 'gold', font: 'math', bg: false });
    const lbl = ['C first', 'then R', 'then S'];
    for (let k = 0; k < 3; k++) if (m > k + 0.05) g.hud((k + 1) + '. ' + lbl[k], 14, 22 + 20 * k, { color: m < k + 1 ? 'gold' : 'muted' });
    if (s.brk > 0) g.hud('same motions, same end grid', cx, y + 34, { anchor: 'center', color: 'teal' });
  }

  /* step 11: three linear layers collapse into one matrix */
  function renderLayers(g, s) {
    const m = s.m3, d = s.direct;
    const stages = [W1, N.matMul(W2, W1), W];
    let M;
    if (d > 0) M = lerpM(I2(), W, d);
    else M = m <= 1 ? lerpM(I2(), W1, m) : m <= 2 ? N.matMul(tw(I2(), W2, m - 1), W1) : N.matMul(lerpM(I2(), W3, m - 2), stages[1]);
    g.ghostGrid();
    const ga = d > 0 ? 1 - seg(d, 0, 0.3) : 1;
    if (ga > 0) for (let k = 0; k < 2; k++) if (m >= k + 1) g.grid(stages[k], { color: 'muted', axisColor: 'muted', dash: [2, 5], alpha: 0.4 * ga, range: 8 });
    g.grid(M, { range: 8 });
    const x = [1, 0.6];
    g.vector(N.matVec(M, x), { color: 'v', label: 'x', width: 3 });
    g.vector(N.col(M, 0), { color: 'i', label: 'i', hat: true, width: 3 });
    g.vector(N.col(M, 1), { color: 'j', label: 'j', hat: true, width: 3 });
    // three matrix panels that slide together and merge into one
    const mg = s.merge, cxs = [g.size.w / 2 - 210, g.size.w / 2 - 60, g.size.w / 2 + 90], y = 22;
    const Ws = [W3, W2, W1], names = ['W₃', 'W₂', 'W₁'];
    if (mg < 1) Ws.forEach((A, k) => hudMat(g, cellsOf(r2(A)), LA.lerp(cxs[k], g.size.w / 2 - 40, mg) + 20, y, { cw: 40, ch: 22, size: 13, name: names[k], alpha: 1 - mg }));
    if (mg > 0) hudMat(g, cellsOf(r2(W)), g.size.w / 2 - 20, y, { cw: 44, ch: 24, size: 14, name: 'W₃W₂W₁ =', alpha: mg });
    g.hud(d > 0 ? 'one matrix, one straight motion: no bending' : 'layer ' + Math.min(3, Math.floor(m) + 1) + ' of 3', 14, g.size.h - 44, { color: d > 0 ? 'teal' : 'hud' });
    g.hud('p2-12: a neural layer', 14, g.size.h - 22, { color: 'muted' });
  }

  /* step 8 and the playground: the two orders side by side */
  function renderSplit(g, s, ctx) {
    const A = s.live ? s.A : R, B = s.live ? s.B : S, m = s.live ? s.morph : s.m;
    const camA = fitCam(g, 0.5, 1, { center: [0.1, 0.4], halfH: 2.4, halfW: 2.1 });
    const pane = (First, Second, title, g2, r) => {
      const M = drawStaged(g2, First, Second, m, { v: s.showV ? s.v : null, v1Label: title.v1, v2Label: title.v2 });
      g2.hud(title.t, r.x + 12, r.y + 20, { color: 'hud' });
      const P = N.matMul(Second, First);
      hudMat(g2, cellsOf(r2(P)), r.x + r.w - 116, r.y + r.h - 82, { cw: 44, ch: 26, size: 14, name: title.p + ' =' });
      return M;
    };
    g.split((g2, r) => pane(A, B, s.live ? { t: 'A first, then B', p: 'BA', v1: 'Av', v2: 'BAv' } : { t: 'R first, then S', p: 'SR', v1: 'Rv', v2: 'SRv' }, g2, r),
      (g2, r) => {
        pane(B, A, s.live ? { t: 'B first, then A', p: 'AB', v1: 'Bv', v2: 'ABv' } : { t: 'S first, then R', p: 'RS', v1: 'Sv', v2: 'RSv' }, g2, r);
        if (s.cross) { g2.label('SR = RS ?', [r.x + r.w / 2 - 14, r.y + 64], { screen: true, color: 'bad', size: 17 }); g2.marker.cross([r.x + r.w / 2 + 52, r.y + 64], { screen: true, size: 24 }); }
        if (s.live) { const AB = N.matMul(A, B), BA = N.matMul(B, A), same = N.approxEqM ? N.approxEqM(AB, BA, 1e-6) : false; g2.hud(same ? 'these two commute: AB = BA' : 'AB ≠ BA', r.x + 12, r.y + 40, { color: same ? 'ok' : 'bad' }); }
      }, { left: { view: '2d', cam: camA }, right: { view: '2d', cam: camA } });
  }

  function renderRecap(g, s) {
    const camRow = (fw, fh, c) => fitCam(g, fw, fh, c);
    g.viewport({ x: 0, y: 0, w: 1, h: 0.34 }, { view: '2d', cam: camRow(1, 0.34, { center: [0, 0.5], halfH: 1.35, halfW: 2 }) }, (g2, r) => {
      g2.ghostGrid(); g2.grid(SR, { alpha: 0.5, range: 6 });
      drawTrack(g2, { track: 'i', m: 2 });
      g2.vector([1, 1], { color: 'i', label: 'i', hat: true, width: 3 });
      g2.hud('1 · î hops twice: R, then S → column 1 of SR', r.x + 10, r.y + 16, { color: 'muted' });
    });
    const cam2 = camRow(0.5, 0.33, { center: [0, 0.4], halfH: 1.4, halfW: 1.5 });
    g.viewport({ x: 0, y: 0.34, w: 0.5, h: 0.33 }, { view: '2d', cam: cam2 }, (g2, r) => { drawStaged(g2, R, S, 2, { midGhost: false }); g2.hud('2 · SR', r.x + 10, r.y + 16, { color: 'muted' }); });
    g.viewport({ x: 0.5, y: 0.34, w: 0.5, h: 0.33 }, { view: '2d', cam: cam2 }, (g2, r) => { drawStaged(g2, S, R, 2, { midGhost: false }); g2.hud('RS ≠ SR', r.x + 10, r.y + 16, { color: 'bad' }); });
    g.viewport({ x: 0, y: 0.67, w: 1, h: 0.33 }, { view: '2d', cam: LA.cam.defaults2d() }, (g2, r) => {
      const cx = r.x + r.w / 2, y = r.y + r.h / 2 - 22;
      hudMat(g2, cellsOf(r2(W3)), cx - 250, y, { cw: 34, ch: 22, size: 12.5, name: 'W₃' });
      hudMat(g2, cellsOf(r2(W2)), cx - 140, y, { cw: 34, ch: 22, size: 12.5, name: 'W₂' });
      hudMat(g2, cellsOf(r2(W1)), cx - 30, y, { cw: 34, ch: 22, size: 12.5, name: 'W₁' });
      g2.label('=', [cx + 66, y + 22], { screen: true, size: 18, color: 'hud', bg: false });
      hudMat(g2, cellsOf(r2(W)), cx + 84, y, { cw: 44, ch: 22, size: 13 });
      g2.hud('3 · three linear layers are one matrix; (AB)C = A(BC), shapes chain', r.x + 10, r.y + 16, { color: 'muted' });
    });
  }

  /* checks */
  function renderCheck1(g, s) {   // B draggable, A fixed, composed grid AB shown
    const AB = N.matMul(s.A, s.B);
    g.ghostGrid();
    g.grid(AB, { range: 8 });
    const b1 = N.col(s.B, 0), b2 = N.col(s.B, 1);
    g.vector(b1, { color: 'i', dashed: true, width: 2, alpha: 0.6, label: 'b₁', labelSize: 12 });
    g.vector(b2, { color: 'j', dashed: true, width: 2, alpha: 0.6, label: 'b₂', labelSize: 12 });
    g.vector(N.col(AB, 0), { color: 'i', label: 'i', hat: true, width: 3 });
    g.vector(N.col(AB, 1), { color: 'j', label: 'j', hat: true, width: 3 });
    g.hud('dashed: B’s columns (drag them) · solid: î, ĵ after B, then A', 14, 22, { color: 'muted' });
    hudMat(g, cellsOf(r2(AB)), g.size.w - 130, 18, { cw: 48, name: 'AB =' });
    g.handle('b1', b1, (p, st) => { st.B[0][0] = p[0]; st.B[1][0] = p[1]; }, { color: 'i' });
    g.handle('b2', b2, (p, st) => { st.B[0][1] = p[0]; st.B[1][1] = p[1]; }, { color: 'j' });
  }
  const CARD = { rot: { name: 'rotate 90°', M: R }, ref: { name: 'reflect in x-axis', M: FLIPX } };
  function renderCheck2(g, s) {   // two cards in slots "first" and "second"; î's two hops for the current order
    const first = CARD[s.cards[0]].M, second = CARD[s.cards[1]].M, e = [1, 0], mid = N.matVec(first, e), end = N.matVec(second, mid);
    g.ghostGrid();
    g.grid(N.matMul(second, first), { alpha: 0.5, range: 8 });
    g.marker.ring([0, -1], { r: 10, color: 'hud' }); g.label('target', [0, -1], { color: 'muted', size: 12, anchor: 'below', dy: 14 });
    g.vector(mid, { color: 'i', dashed: true, width: 2, alpha: 0.55 });
    g.line([1, 0], mid, { color: 'i', dashed: true, alpha: 0.5, width: 1.3 }); g.line(mid, end, { color: 'i', dashed: true, alpha: 0.5, width: 1.3 });
    g.vector(end, { color: 'i', label: 'i', hat: true, width: 3 });
    const slots = [[-1.0, 1.9], [1.0, 1.9]];
    ['first', 'second'].forEach((nm, k) => {
      const c = slots[k], w = 0.9, h = 0.3;
      g.polygon([[c[0] - w, c[1] - h], [c[0] + w, c[1] - h], [c[0] + w, c[1] + h], [c[0] - w, c[1] + h]], { fill: 'teal', alpha: 0.14, stroke: 'teal', width: 1.4 });
      g.label(CARD[s.cards[k]].name, [c[0] - 0.12, c[1]], { color: 'text', size: 14, bg: false });
      g.label(nm, [c[0], c[1] + h], { color: 'muted', size: 12, anchor: 'above', dy: -4 });
      const name = s.cards[k];   // the handle follows the card, so a swap is idempotent while dragging; it sits at the card's right end, clear of the text
      g.handle('card-' + name, [c[0] + w - 0.17, c[1]], (p, st) => { const want = p[0] < 0 ? 0 : 1; if (st.cards[want] !== name) st.cards = want === 0 ? [name, st.cards[0]] : [st.cards[1], name]; }, { color: 'teal', r: 18 });
    });
    g.hud('drag a card onto the other slot to swap the order', 14, g.size.h - 22, { color: 'muted' });
  }
  function renderCheck3(g, s) {   // v draggable; A(Bv) gold, B(Av) gray
    const A = [[2, 0], [0, 1]], B = S, v = s.v, ABv = N.matVec(A, N.matVec(B, v)), BAv = N.matVec(B, N.matVec(A, v));
    g.ghostGrid(); g.grid(null, { alpha: 0.4, range: 8 });
    g.vector(ABv, { color: 'gold', label: 'A(Bv)', width: 3 });
    g.vector(BAv, { color: 'muted', label: 'B(Av)', width: 3, dx: 14, dy: 16 });
    if (N.dist(ABv, BAv) > 0.05) g.line(ABv, BAv, { color: 'bad', dashed: true, width: 1.4 });
    g.vector(v, { color: 'v', label: 'v', width: 3 });
    g.handle('v', v, (p, st) => { st.v = [p[0], p[1]]; }, { color: 'v' });
  }

  LA.registerLesson({
    id: 'p2-04-composition',
    part: 2,
    title: 'Matrix multiplication is composition',
    subtitle: 'Do one map, then another: the product matrix records the whole trip, read right to left.',
    minutes: 11,
    prereqs: ['p2-03-transformation-zoo'],
    tags: ['core'],
    badge: { drawnIn: '2D', realSize: 'a 3-layer network with no nonlinearity, W₃W₂W₁, collapses to a single 10×784 matrix', transfers: 'composition = multiplication, right-to-left order, AB ≠ BA and (AB)C = A(BC) hold for all shapes; only the number of columns tracked changes' },
    cantShow: ['a product of a 128×784 and a 10×128 matrix as grid motion (we track two columns and say “one per input direction”)'],
    terms: [
      { term: 'composition', def: 'Doing one transformation after another.' },
      { term: 'matrix product', def: 'AB is the matrix of “B first, then A”; its columns are B’s columns pushed through A.' },
      { term: 'right to left', def: 'In AB v the matrix nearest v acts first.' },
      { term: 'commute', def: 'Two maps commute if doing them in either order gives the same result; most pairs do not.' },
      { term: 'associative', def: '(AB)C = A(BC): how you group a product does not matter, only the order of the factors.' },
      { term: 'identity matrix', def: 'The do-nothing matrix I, with 1s on the diagonal and 0s elsewhere.' },
    ],
    atlas: ['mlp', 'backprop'],
    thumbnail: 7,

    state() {
      return {
        view: '2d', cam: { center: [0.1, 0.45], halfH: 2.5 },
        mode: 'stage', First: LA.clone(R), Second: LA.clone(S), Target: LA.clone(SR), m: 0, showV: false, v: [1, 1],
        track: null, dimJ: null, gridAlpha: 0.9, midGhost: true, hud: null, rtl: false, colT: 0, entT: 0, m3: 0, brk: 0, tileT: 0,
        direct: 0, merge: 0, cross: false, mathMode: null,
        live: false, A: LA.clone(R), B: LA.clone(S), morph: 2, cards: ['ref', 'rot'],
      };
    },

    render(g, s, ctx) {
      renderMain(g, s, ctx);
      if (s.rtl) drawRTL(g, s);
    },

    math(s, tex) {
      const eq = tex.op('='), mat = (X, o) => tex.mat(r2(X), Object.assign({ colors: 'cols' }, o || {}));
      const Q = [['?', '?'], ['?', '?']];
      const lit = on => (on ? () => true : undefined);
      switch (s.mathMode) {
        case 'stage': return tex.block(tex.eq(tex.sym('R'), eq, mat(R, { highlight: lit(s.m > 0 && s.m < 1.05) })), tex.eq(tex.sym('S'), eq, mat(S, { highlight: lit(s.m >= 1.05) })), tex.text(s.m < 1 ? 'first R acts' : 'then S acts'));
        case 'unknown': return tex.block(tex.eq(tex.sym('S') + tex.sym('R'), eq, tex.mat(Q)));
        case 'trackI': return tex.block(tex.eq(tex.ihat(), tex.op('→'), s.m >= 1 ? tex.vec([0, 1], { color: 'i' }) : tex.text('?'), tex.op('→'), s.m >= 1.98 ? tex.vec([1, 1], { color: 'i' }) : tex.text('?')), tex.eq(tex.sym('S') + tex.sym('R'), eq, tex.mat(s.m >= 1.98 ? [[1, '?'], [1, '?']] : Q, { colors: 'cols' })));
        case 'trackJ': return tex.block(tex.eq(tex.jhat(), tex.op('→'), s.m >= 1 ? tex.vec([-1, 0], { color: 'j' }) : tex.text('?'), tex.op('→'), s.m >= 1.98 ? tex.vec([-1, 0], { color: 'j' }) : tex.text('?')), tex.eq(tex.sym('S') + tex.sym('R'), eq, tex.mat(s.m >= 1.98 ? SR : [[1, '?'], [1, '?']], { colors: 'cols' })));
        case 'rtl': return tex.block(tex.eq(tex.sym('S') + tex.paren(tex.sym('R') + tex.sym('v', { color: 'v' })), eq, tex.paren(tex.sym('S') + tex.sym('R')) + tex.sym('v', { color: 'v' })),
          tex.eq(tex.sym('v', { color: 'v' }), eq, tex.vec([1, 1], { color: 'v' }), tex.op('→'), tex.vec([-1, 1], { color: 'v' }), tex.op('→'), tex.vec([0, 1], { color: 'v' })));
        case 'columns': return tex.block(tex.eq(tex.sym('S') + tex.sym('R'), eq, tex.op('['), tex.sym('S') + tex.op('·') + tex.text('col₁(R)'), tex.text('  '), tex.sym('S') + tex.op('·') + tex.text('col₂(R)'), tex.op(']'), eq, mat(SR)));
        case 'entries': return tex.block(tex.eq(tex.paren(tex.sym('S') + tex.sym('R')) + '<sub>ij</sub>', eq, tex.text('row<sub>i</sub>(S) · col<sub>j</sub>(R)', { html: true })), tex.eq(tex.sym('S') + tex.sym('R'), eq, mat(SR)));
        case 'order': return tex.block(tex.eq(tex.sym('S') + tex.sym('R'), eq, mat(SR), tex.op('≠'), tex.sym('R') + tex.sym('S'), eq, mat(RS)));
        case 'assoc': return tex.block(tex.eq(tex.paren(tex.sym('S') + tex.sym('R')) + tex.sym('C'), eq, tex.sym('S') + tex.paren(tex.sym('R') + tex.sym('C')), eq, mat(N.matMul(SR, C3))), tex.eq(tex.sym('C'), eq, mat(C3)));
        case 'tiles': return tex.block(tex.eq(tex.paren(tex.dim(2, 3)), tex.paren(tex.dim(3, 2)), tex.op('→'), tex.dim(2, 2)), tex.eq(tex.paren(tex.dim(2, 3)), tex.paren(tex.dim(2, 3)), tex.op('→'), tex.text('✕ no product', { color: 'bad' })));
        case 'layers': return tex.block(tex.eq(tex.sym('W', { sub: 3 }) + tex.sym('W', { sub: 2 }) + tex.sym('W', { sub: 1 }) + tex.sym('x', { color: 'v' }), eq, tex.sym('W') + tex.sym('x', { color: 'v' })), tex.eq(tex.sym('W'), eq, mat(W)));
        case 'recap': return '';
        case 'check1': return tex.block(tex.eq(tex.sym('A'), eq, mat(s.A), tex.sym('B'), eq, mat(s.B)), tex.eq(tex.sym('A') + tex.sym('B'), eq, mat(N.matMul(s.A, s.B))));
        case 'check2': { const P = N.matMul(CARD[s.cards[1]].M, CARD[s.cards[0]].M); return tex.block(tex.eq(tex.text('second · first'), eq, mat(P)), tex.eq(tex.ihat(), tex.op('→'), tex.vec(N.col(P, 0), { color: 'i' }))); }
        case 'check3': { const A = [[2, 0], [0, 1]], ABv = N.matVec(A, N.matVec(S, s.v)), BAv = N.matVec(S, N.matVec(A, s.v)); return tex.block(tex.eq(tex.sym('A') + tex.paren(tex.sym('B') + tex.sym('v', { color: 'v' })), eq, tex.vec(r2([ABv])[0], { color: 'gold' })), tex.eq(tex.sym('B') + tex.paren(tex.sym('A') + tex.sym('v', { color: 'v' })), eq, tex.vec(r2([BAv])[0]))); }
        default: {
          if (!s.live) return '';
          const BA = N.matMul(s.B, s.A), AB = N.matMul(s.A, s.B);
          return tex.block(tex.eq(tex.sym('A'), eq, mat(s.A), tex.sym('B'), eq, mat(s.B)), tex.eq(tex.sym('B') + tex.sym('A'), eq, mat(BA), tex.sym('A') + tex.sym('B'), eq, mat(AB)));
        }
      }
    },

    steps: [
      { title: 'Rotate, then shear', duration: 5,
        narration: 'Two transformations, one after another. First a quarter-turn rotation R = [[0, −1],[1, 0]]. Pause. Then the shear S = [[1, 1],[0, 1]]. A dashed green arrow marks where the rotation left <span class="c-i">î</span>, so you can see the stop in between. The whole trip is called the <b>composition</b> of the two maps.',
        setup(s) { s.mode = 'stage'; s.mathMode = 'stage'; s.First = LA.clone(R); s.Second = LA.clone(S); s.m = 0; s.showV = false; s.track = null; s.dimJ = null; s.hud = null; s.rtl = false; s.gridAlpha = 0.9; s.midGhost = true; s.cam = { center: [0.1, 0.45], halfH: 2.5 }; },
        at(s, t) { s.m = seg(t, 0, 0.4) + seg(t, 0.55, 0.95); } },

      { title: 'The result is linear, so it has a matrix', duration: 3.5,
        narration: 'Look at the end state: lines are still straight, parallel and evenly spaced, and the origin has not moved. The trip is itself a linear transformation — so it must have its own matrix. Which one? Watch the grid go directly from start to finish; the bracket shows the unknown entries as question marks.',
        setup(s) { s.mode = 'direct'; s.mathMode = 'unknown'; s.Target = LA.clone(SR); s.m = 0; s.hud = { M: SR, known: [false, false], name: 'SR =' }; },
        at(s, t) { s.m = seg(t, 0.1, 0.8); } },

      { title: 'Track î through both hops', duration: 4.5,
        narration: 'Follow the green ' + ihat + '. The rotation sends it to [0; 1]. The shear then sends [0; 1] to [1; 1]. So the whole trip sends ' + ihat + ' to [1; 1] — that is the first column of the product. Two hops, one landing spot.',
        setup(s) { s.mode = 'stage'; s.mathMode = 'trackI'; s.m = 0; s.track = 'i'; s.dimJ = 'j'; s.gridAlpha = 0.4; s.hud = { M: SR, known: [false, false], name: 'SR =' }; },
        at(s, t) { s.m = seg(t, 0, 0.4) + seg(t, 0.42, 0.8); s.hud = { M: SR, known: [t > 0.82, false], name: 'SR =' }; } },

      { title: 'Track ĵ', duration: 4.5,
        narration: 'Now the red ' + jhat + '. The rotation sends it to [−1; 0]. The shear leaves [−1; 0] alone, because it only leans things that point up or down. So ' + jhat + ' lands on [−1; 0]: the second column. The product is complete: SR = [[1, −1],[1, 0]].',
        setup(s) { s.mathMode = 'trackJ'; s.m = 0; s.track = 'j'; s.dimJ = 'i'; s.hud = { M: SR, known: [true, false], name: 'SR =' }; },
        at(s, t) { s.m = seg(t, 0, 0.4) + seg(t, 0.42, 0.8); s.hud = { M: SR, known: [true, t > 0.82], name: 'SR =' }; } },

      { title: 'Read right to left', duration: 4.5,
        narration: 'In symbols, “rotate then shear” is S(R' + vv + ') = (SR)' + vv + '. Read it <b>right to left</b>, like f(g(x)): the matrix nearest the vector acts first. Writing SR for “R first” feels backwards at first; the picture of the yellow v hopping twice makes it natural.',
        setup(s) { s.mathMode = 'rtl'; s.m = 0; s.track = null; s.dimJ = null; s.gridAlpha = 0.6; s.hud = null; s.showV = true; s.v = [1, 1]; s.rtl = true; s.cam = { center: [0.1, 0.2], halfH: 2.5 }; },
        at(s, t) { s.m = seg(t, 0.05, 0.42) + seg(t, 0.55, 0.92); } },

      { title: 'The product, column by column', duration: 5.5,
        narration: 'Matrix multiplication is defined so that this works. Column j of the <b>matrix product</b> SR is column j of R pushed through S — the recipe from p2-02. Column 1 of R is [0; 1]: S·[0; 1] = 0·[1; 0] + 1·[1; 1] = [1; 1]. Column 2 of R is [−1; 0]: S·[−1; 0] = [−1; 0].',
        setup(s) { s.mode = 'columns'; s.mathMode = 'columns'; s.showV = false; s.rtl = false; s.colT = 0; s.cam = { center: [0.1, 0.45], halfH: 2.5 }; },
        at(s, t) { s.colT = t; } },

      { title: 'Entry by entry: row dot column', duration: 5,
        narration: 'Entry by entry, each number of the product is a dot product: row i of the left matrix with column j of the right. Row 1 of S is [1, 1]; column 1 of R is [0; 1]; their dot product is 1 — the top-left entry. Four such dot products fill the product.',
        setup(s) { s.mode = 'entries'; s.mathMode = 'entries'; s.entT = 0; },
        at(s, t) { s.entT = t; } },

      { title: 'Order matters', duration: 5.5, misconception: true,
        narration: 'Belief: “AB = BA.” Try the other order: shear first, then rotate. The grids end in different places, and the matrices differ: RS = [[0, −1],[1, 1]], not SR = [[1, −1],[1, 0]]. Transformations rarely <b>commute</b>. The order you write is the reverse of the order you do.',
        setup(s) { s.mode = 'split'; s.mathMode = 'order'; s.live = false; s.m = 0; s.cross = false; s.showV = false; },
        at(s, t) { s.m = seg(t, 0, 0.38) + seg(t, 0.42, 0.8); s.cross = t > 0.85; } },

      { title: 'Grouping does not matter', duration: 5,
        narration: 'Three maps in a row: C, then R, then S. Whether you think of it as (SR)C or S(RC), it is the same sequence of motions — nothing about the trip changes when you bracket it differently. Matrix multiplication is <b>associative</b>; the order of the factors is fixed, the grouping is free.',
        setup(s) { s.mode = 'assoc'; s.mathMode = 'assoc'; s.cross = false; s.m3 = 0; s.brk = 0; },
        at(s, t) { s.m3 = seg(t, 0, 0.18) + seg(t, 0.2, 0.38) + seg(t, 0.4, 0.58); s.brk = seg(t, 0.65, 0.92); } },

      { title: 'Shapes must chain', duration: 3.5,
        narration: 'A product only makes sense if the first map’s output size is the next map’s input size. Shape tiles: (2×3)(3×2) chains, the inner 3s meet, and the result is 2×2. (2×3)(2×3) does not chain: the inner numbers differ and the tiles bounce apart.',
        setup(s) { s.mode = 'tiles'; s.mathMode = 'tiles'; s.tileT = 0; },
        at(s, t) { s.tileT = t; } },

      { title: 'A stack of linear layers is one matrix', duration: 6.5, mlTie: true,
        narration: 'A network with three linear layers and nothing between them computes W₃W₂W₁<span class="c-v">x</span>. That product is a single matrix: three grid motions collapse into one. However many layers you stack, it is still one linear map and cannot bend anything. That is why real layers put a fold between the matrices — the capstone of this part.',
        setup(s) { s.mode = 'layers'; s.mathMode = 'layers'; s.m3 = 0; s.direct = 0; s.merge = 0; s.cam = { center: [0.2, 0.3], halfH: 2.5 }; },
        at(s, t) { s.m3 = seg(t, 0, 0.18) + seg(t, 0.2, 0.38) + seg(t, 0.4, 0.58); s.direct = seg(t, 0.64, 0.92); s.merge = seg(t, 0.62, 0.8); } },

      { title: 'Recap', duration: 2, recap: true,
        narration: 'Multiplying matrices composes transformations; read right to left. The columns of AB are B’s columns pushed through A; each entry is a row dotted with a column. Order matters, grouping does not, and shapes must chain. A stack of linear layers is one matrix.',
        cards: ['<b>AB = do B, then A.</b> Read right to left: the matrix nearest v acts first.', '<b>Columns of AB</b> are B’s columns pushed through A; each entry is row · column.', '<b>AB ≠ BA</b>, but (AB)C = A(BC); shapes must chain; stacked linear layers are one matrix.'],
        setup(s) { s.mode = 'recap'; s.mathMode = 'recap'; },
        at(s, t) { } },
    ],

    playground: {
      intro: '<p>A is the first map, B the second. The left pane does A then B (the product BA), the right pane B then A (AB). Slide “Step through” from 0 to 2 to watch the two hops; the identity matrix is the do-nothing starting grid.</p>',
      setup(s) { s.mode = 'split'; s.mathMode = null; s.live = true; s.A = LA.clone(R); s.B = LA.clone(S); s.morph = 2; s.showV = true; s.v = [1, 1]; s.cross = false; },
      controls: [
        { type: 'matrix', bind: 'A', label: 'A (first)', colors: 'cols', step: 0.1, min: -3, max: 3 },
        { type: 'matrix', bind: 'B', label: 'B (second)', colors: 'cols', step: 0.1, min: -3, max: 3 },
        { type: 'slider', bind: 'morph', min: 0, max: 2, step: 0.01, label: 'Step through', color: 'gold' },
        { type: 'matrix', bind: 'v', label: 'v', colors: 'v', step: 0.1, min: -3, max: 3 },
        { type: 'button', label: 'Swap order', onClick(s) { const t = s.A; s.A = s.B; s.B = t; } },
        { type: 'preset', label: 'Presets', items: [
          { label: 'Rotate then shear', apply(s) { s.A = LA.clone(R); s.B = LA.clone(S); } },
          { label: 'Two rotations (commute)', apply(s) { s.A = r2(N.rot2(Math.PI / 6)); s.B = r2(N.rot2(Math.PI / 3)); } },
          { label: 'Two scalings (commute)', apply(s) { s.A = [[2, 0], [0, 0.5]]; s.B = [[0.75, 0], [0, 1.5]]; } },
          { label: 'Shear then reflect', apply(s) { s.A = LA.clone(S); s.B = [[0, 1], [1, 0]]; } },
        ] },
      ],
    },

    checks: [
      { kind: 'manipulate',
        prompt: 'A = [[2, 1],[1, 1]] is fixed. Drag the dashed landing spots of B so that AB = I: B undoes A, and the composed grid lands back on the ghost.',
        setup(s) { s.mode = 'check1'; s.mathMode = 'check1'; s.live = false; s.A = [[2, 1], [1, 1]]; s.B = I2(); s.cam = { center: [0.2, 0.3], halfH: 2.8 }; },
        // A stretches errors in B by up to 2.6×, so judge AB at 2× the position tolerance (STYLE allows ≤ 2×)
        judge(s) { return LA.judge.matrix(N.matMul(s.A, s.B), I2(), 2 * LA.TOL.pos); },
        hint: 'The composed grid must land back on the dashed ghost: the solid ' + ihat + ' on [1; 0] and the solid ' + jhat + ' on [0; 1].',
        solution(s) { s.B = [[1, -1], [-1, 2]]; } },
      { kind: 'manipulate',
        prompt: 'Order the two cards “rotate 90°” and “reflect in the x-axis” so that ' + ihat + ' ends at the target [0; −1].',
        setup(s) { s.mode = 'check2'; s.mathMode = 'check2'; s.live = false; s.cards = ['ref', 'rot']; s.cam = { center: [0, 0.4], halfH: 2.4 }; },
        judge(s) { const ok = s.cards[0] === 'rot' && s.cards[1] === 'ref'; return { ok, progress: ok ? 1 : 0 }; },
        hint: 'Rotating first sends ' + ihat + ' to [0; 1]; which card then flips it down?',
        solution(s) { s.cards = ['rot', 'ref']; } },
      { kind: 'manipulate',
        prompt: 'With A = diag(2, 1) and B = the shear [[1, 1],[0, 1]], drag ' + vv + ' so that A(Bv) = B(Av).',
        setup(s) { s.mode = 'check3'; s.mathMode = 'check3'; s.live = false; s.v = [1, 1]; s.cam = { center: [1.8, 0.7], halfH: 3 }; },
        judge(s) { const A = [[2, 0], [0, 1]]; const d = N.dist(N.matVec(A, N.matVec(S, s.v)), N.matVec(S, N.matVec(A, s.v))), ok = d < LA.TOL.pos && N.norm(s.v) > 0.5; return { ok, progress: Math.max(0, 1 - d / 2), hint: N.norm(s.v) <= 0.5 ? 'Keep v away from the origin.' : undefined }; },
        hint: 'Put ' + vv + ' where the shear does nothing to it.',
        solution(s) { s.v = [2, 0]; } },
    ],
  });
})();
