/* p2-10 — Transpose, xᵀy, outer products, symmetric matrices. */
(function () {
  'use strict';
  const N = LA.num, lerpM = LA.lerpM, lerpV = LA.lerpV, seg = LA.seg, C = LA.color, lerp = LA.lerp;
  const I2 = () => [[1, 0], [0, 1]];
  const A0 = [[2, 1], [0, 3]], B0 = [[1, 2, 3], [4, 5, 6]], S0 = [[2, 1], [1, 2]], H0 = [[1, 1], [0, 1]];
  const ihat = '<span class="c-i">î</span>', jhat = '<span class="c-j">ĵ</span>';
  const T = M => N.transpose(M);
  const DEG = Math.PI / 180;
  const BASIS = ['i', 'j', 'k'];

  /* ---------- seeded data (small) ---------- */
  function tokens() {
    const rand = LA.rng(10); const out = [];
    for (let i = 0; i < 4; i++) { const a = (i / 4) * Math.PI * 2 + 0.5 + (rand() - 0.5) * 0.8; const r = 0.75 + 0.35 * rand(); out.push([r * Math.cos(a), r * Math.sin(a)]); }
    return out;
  }
  function featureGram() {          // XᵀX of a small synthetic 8×3 feature table
    const rand = LA.rng(11); const X = [];
    for (let i = 0; i < 8; i++) { const a = LA.rng.gauss(rand), b = LA.rng.gauss(rand), c = LA.rng.gauss(rand); X.push([a, 0.6 * a + 0.8 * b, -0.5 * a + 0.3 * b + 0.8 * c]); }
    return N.matMul(T(X), X).map(r => r.map(x => Math.round(x * 10) / 10));
  }

  /* ---------- drawing helpers ---------- */
  /** A matrix as a block of cells. o: x0, y0 (top-left, world), cs, fill(i,j,v), textAt(i,j,v), strokeAt(i,j), pos(i,j)→[cx,cy], digits, size, bracket, brColor, alpha */
  function cells(g, M, o) {
    const rows = M.length, cols = M[0].length, cs = o.cs || 0.8, h = cs / 2 - 0.035;
    for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) {
      const v = M[i][j];
      const c = o.pos ? o.pos(i, j) : [o.x0 + (j + 0.5) * cs, o.y0 - (i + 0.5) * cs];
      const f = o.fill ? o.fill(i, j, v) : 'panel2';
      const st = o.strokeAt ? o.strokeAt(i, j) : 'line';
      g.polygon([[c[0] - h, c[1] - h], [c[0] + h, c[1] - h], [c[0] + h, c[1] + h], [c[0] - h, c[1] + h]], { fill: f, alpha: o.alpha === undefined ? 0.92 : o.alpha, stroke: st || null, width: st === 'line' ? 1 : 1.8 });
      if (o.text !== false) g.label(typeof v === 'number' ? LA.fmt(v, o.digits) : String(v), c, { color: o.textAt ? o.textAt(i, j, v) : (o.textColor || 'text'), size: o.size || 15, font: 'math', bg: false });
    }
    if (o.bracket) bracket(g, o.x0, o.y0, rows * cs, cols * cs, o.brColor || 'hud');
  }
  function bracket(g, x0, y0, h, w, color) {
    const p = 0.07, k = 0.14, st = { color: color || 'hud', width: 1.6 };
    g.polyline([[x0 - p + k, y0 + p], [x0 - p, y0 + p], [x0 - p, y0 - h - p], [x0 - p + k, y0 - h - p]], st);
    g.polyline([[x0 + w + p - k, y0 + p], [x0 + w + p, y0 + p], [x0 + w + p, y0 - h - p], [x0 + w + p - k, y0 - h - p]], st);
  }
  const heatFill = (lo, hi) => (i, j, v) => C.heat((v - lo) / ((hi - lo) || 1));
  const heatText = (lo, hi) => (i, j, v) => ((v - lo) / ((hi - lo) || 1) > 0.62 ? 'bg' : 'text');
  const rangeOf = M => { let lo = Infinity, hi = -Infinity; M.forEach(r => r.forEach(x => { lo = Math.min(lo, x); hi = Math.max(hi, x); })); return [Math.min(0, lo), Math.max(hi, lo + 1)]; };
  /** Column of cells rotated CCW by ang about pivot (ang = 90° → a row); the brackets stay upright and reshape. */
  function rotCells(g, vals, pivot, ang, cs, color) {
    const n = vals.length, h = cs / 2 - 0.035, ca = Math.cos(ang), sa = Math.sin(ang);
    const R = p => [pivot[0] + p[0] * ca - p[1] * sa, pivot[1] + p[0] * sa + p[1] * ca];
    let hw = 0, hh = 0;
    for (let i = 0; i < n; i++) {
      const cy = (n / 2 - i - 0.5) * cs;
      const q = [[-h, cy - h], [h, cy - h], [h, cy + h], [-h, cy + h]].map(R);
      q.forEach(p => { hw = Math.max(hw, Math.abs(p[0] - pivot[0])); hh = Math.max(hh, Math.abs(p[1] - pivot[1])); });
      g.polygon(q, { fill: 'panel2', alpha: 0.92, stroke: 'line', width: 1 });
      g.label(LA.fmt(vals[i]), R([0, cy]), { color, size: 15, font: 'math', bg: false });
    }
    bracket(g, pivot[0] - hw, pivot[1] + hh, 2 * hh, 2 * hw, 'hud');
  }
  function tile(g, c, w, h, name, shape, o) {
    o = o || {};
    const pts = [[c[0] - w / 2, c[1] - h / 2], [c[0] + w / 2, c[1] - h / 2], [c[0] + w / 2, c[1] + h / 2], [c[0] - w / 2, c[1] + h / 2]];
    g.polygon(pts, { fill: o.fill || 'panel2', alpha: o.alpha === undefined ? 0.92 : o.alpha, stroke: o.stroke || 'hud', width: 1.4 });
    if (name) g.label(name, [c[0], c[1] + h * 0.22], { color: o.nameColor || 'hud', size: 14, font: 'math', bg: false, alpha: o.alpha });
    if (shape) g.label(shape, [c[0], c[1] - (name ? h * 0.2 : 0)], { color: o.color || 'text', size: o.size || 16, font: 'ui', bold: true, bg: false, alpha: o.alpha });
  }
  /** 2D camera that shows the world box [x0,x1]×[y0,y1] inside a viewport of fractional width fw / height fh. */
  function fit(g, fw, fh, x0, x1, y0, y1) {
    const W = Math.max(50, g.size.w * fw), H = Math.max(50, g.size.h * fh), m = 1.06;
    return { center: [(x0 + x1) / 2, (y0 + y1) / 2], halfH: m * Math.max((y1 - y0) / 2, (x1 - x0) / 2 * H / W) };
  }
  const proj = (v, onto) => { const k = N.dot(v, onto) / (N.dot(onto, onto) || 1); return N.scale(onto, k); };
  const sym = M => Math.abs(M[0][1] - M[1][0]);

  /* ---------- scenes ---------- */
  function sceneFlip(g, s, o) {
    o = o || {};
    const cs = 0.9, x0 = o.x0 === undefined ? -1.55 : o.x0, y0 = o.y0 === undefined ? 1.75 : o.y0, f = s.flip;
    const B = s.B;
    g.line([x0, y0], [x0 + 3 * cs, y0 - 3 * cs], { color: 'gold', dashed: true, width: 1.2, alpha: 0.8 });
    g.label('diagonal', [x0 + 3 * cs, y0 - 3 * cs], { color: 'gold', size: 12, anchor: 'below', dy: 6 });
    cells(g, B, {
      cs,
      pos: (i, j) => lerpV([x0 + (j + 0.5) * cs, y0 - (i + 0.5) * cs], [x0 + (i + 0.5) * cs, y0 - (j + 0.5) * cs], f),
      textAt: (i, j) => BASIS[j],
      strokeAt: (i, j) => (i === j ? 'gold' : 'line'),
      size: 17,
    });
    bracket(g, x0, y0, lerp(2, 3, f) * cs, lerp(3, 2, f) * cs, 'hud');
    const name = f < 0.5 ? 'B · 2×3' : 'Bᵀ · 3×2';
    g.label(name, [x0 + lerp(3, 2, f) * cs / 2, y0 + 0.12], { color: 'text', size: 15, anchor: 'above', font: 'ui', bg: false });
    if (f < 0.5) { for (let i = 0; i < 2; i++) g.label('row ' + (i + 1), [x0 - 0.2, y0 - (i + 0.5) * cs], { color: 'muted', size: 12, anchor: 'right', font: 'ui' }); }
    else { for (let j = 0; j < 2; j++) g.label('col ' + (j + 1), [x0 + (j + 0.5) * cs, y0 - 3 * cs - 0.12], { color: 'muted', size: 12, anchor: 'below', font: 'ui' }); }
    if (s.pulse > 0.02) for (let k = 0; k < 2; k++) g.marker.ring([x0 + (k + 0.5) * cs, y0 - (k + 0.5) * cs], { r: 20 + 10 * s.pulse, color: 'gold', width: 1.5 });
    if (o.entry) {
      // check 1: the learner's Bᵀ beside B; matching cells light their mirror
      const X0 = x0 + 3 * cs + 1.1, Y0 = y0;
      g.label('your Bᵀ', [X0 + cs, Y0 + 0.12], { color: 'text', size: 15, anchor: 'above', font: 'ui', bg: false });
      const ok = (i, j) => Math.abs(s.Bt[i][j] - B[j][i]) <= LA.TOL.entry;
      cells(g, s.Bt, { x0: X0, y0: Y0, cs, bracket: true, textAt: (i, j) => (ok(i, j) ? BASIS[i] : 'text'), strokeAt: (i, j) => (ok(i, j) ? 'ok' : 'line') });
      for (let i = 0; i < 3; i++) for (let j = 0; j < 2; j++) if (ok(i, j)) g.marker.ring([x0 + (i + 0.5) * cs, y0 - (j + 0.5) * cs], { r: 17, color: 'ok', width: 1.4 });
      g.label('type the entries; a cell lights when it matches its mirror in B', [x0 + 2.4, y0 - 3 * cs - 0.9], { color: 'muted', size: 12, font: 'ui' });
    }
  }

  function sceneRowvec(g, s) {
    g.ghostGrid(); g.axes({ labels: false });
    g.vector(s.x, { color: 'v', label: 'x', width: 3, components: 'labels' });
    const ang = s.ang;
    rotCells(g, s.x, [-1.9, 1.9], ang, 0.8, 'v');
    g.label(ang < Math.PI / 4 ? 'x · a column' : 'xᵀ · a row', [-1.9, 2.95], { color: 'v', size: 14, font: 'ui' });
    if (ang > 0.01 && ang < Math.PI / 2 - 0.01) g.arc([-1.9, 1.9], [0, 1], [-Math.sin(ang), Math.cos(ang)], { r: 1.0, color: 'muted' });
  }

  function sceneDot(g, s) {
    g.ghostGrid(); g.axes({ labels: false });
    const x = s.x, y = s.y;
    g.vector(x, { color: 'v', label: 'x', width: 3 });
    g.vector(y, { color: 'w', label: 'y', width: 3 });
    if (s.shadow > 0) {
      const foot = proj(y, x);
      g.line(y, lerpV(y, foot, s.shadow), { color: 'residual', dashed: true, width: 1.2 });
      if (s.shadow >= 1) { g.point(foot, { r: 4, color: 'mag' }); g.line([0, 0], foot, { color: 'mag', width: 3.5, alpha: 0.9 }); g.label('shadow ' + LA.fmt(N.norm(foot)) + ' × length of x ' + LA.fmt(N.norm(x)) + ' = ' + LA.fmt(N.dot(x, y)), [foot[0] + 0.15, foot[1] - 0.25], { color: 'mag', size: 12.5, anchor: 'left', font: 'ui' }); }
    }
    // the equation row on top: [1 1] slides to meet [1; 0]
    const cs = 0.7, yr = 2.55;
    const rx = lerp(-2.9, -2.25, s.slide);
    cells(g, [x], { x0: rx, y0: yr + cs / 2, cs, bracket: true, textColor: 'v', brColor: 'v' });
    g.label('xᵀ', [rx + cs, yr + cs / 2 + 0.1], { color: 'v', size: 14, anchor: 'above', bg: false });
    const cx0 = -0.6;
    cells(g, y.map(v => [v]), { x0: cx0, y0: yr + cs, cs, bracket: true, textColor: 'w', brColor: 'w' });
    g.label('y', [cx0 + cs / 2, yr + cs + 0.1], { color: 'w', size: 14, anchor: 'above', bg: false });
    if (s.lit > 0 && s.lit < 1) { const k = s.lit < 0.5 ? 0 : 1; g.marker.ring([rx + (k + 0.5) * cs, yr], { r: 16, color: 'gold', width: 1.6 }); g.marker.ring([cx0 + cs / 2, yr + cs - (k + 0.5) * cs], { r: 16, color: 'gold', width: 1.6 }); }
    if (s.sum > 0) {
      g.label('=', [0.45, yr], { color: 'hud', size: 18, font: 'math', bg: false });
      cells(g, [[N.dot(x, y)]], { x0: 0.7, y0: yr + cs / 2, cs, bracket: true, textColor: 'gold', brColor: 'gold' });
      g.label('1×1', [0.7 + cs / 2, yr - cs / 2 - 0.08], { color: 'muted', size: 12, anchor: 'below', font: 'ui' });
      g.label(LA.fmt(x[0]) + '·' + LA.fmt(y[0]) + ' + ' + LA.fmt(x[1]) + '·' + LA.fmt(y[1]) + ' = ' + LA.fmt(N.dot(x, y)), [-0.8, 1.55], { color: 'gold', size: 14, font: 'ui', alpha: s.sum });
    }
  }

  function sceneAdjoint(g, s) {
    const A = s.A, x = s.x, y = s.y, Ax = N.matVec(A, x), Aty = N.matVec(T(A), y);
    const cam = fit(g, 0.5, 1, -0.4, 3.6, -0.9, 3.5);
    const left = (g, r) => {
      g.ghostGrid(); g.axes({ labels: false });
      g.vector(x, { color: 'v', label: 'x', width: 2, dashed: true, alpha: 0.55 });
      const ax = lerpV(x, Ax, s.morph);
      g.vector(y, { color: 'w', label: 'y', width: 3 });
      g.vector(ax, { color: 'v', label: s.morph > 0.5 ? 'Ax' : 'x', width: 3 });
      if (s.shadow > 0) {
        const foot = proj(Ax, y);
        g.line(Ax, lerpV(Ax, foot, s.shadow), { color: 'residual', dashed: true, width: 1.2 });
        if (s.shadow >= 1) { g.line([0, 0], foot, { color: 'mag', width: 4, alpha: 0.85 }); g.point(foot, { r: 4, color: 'mag' }); g.dim([0, 0], foot, { label: '(Ax)·y = ' + LA.fmt(N.dot(Ax, y)), offset: -22, color: 'mag' }); if (s.pulse > 0) g.marker.ring(foot, { r: 14 + 10 * s.pulse, color: 'gold' }); g.hud('shadow ' + LA.fmt(N.norm(foot)) + ' × length of y ' + LA.fmt(N.norm(y)) + ' = ' + LA.fmt(N.dot(Ax, y)), r.x + 10, r.y + r.h - 16, { color: 'mag', size: 13 }); }
      }
      g.hud('transform x by A, measure on y', r.x + 10, r.y + 18, { color: 'hud' });
    };
    const right = (g, r) => {
      g.ghostGrid(); g.axes({ labels: false });
      g.vector(y, { color: 'w', label: 'y', width: 2, dashed: true, alpha: 0.55 });
      const ay = lerpV(y, Aty, s.morph);
      g.vector(x, { color: 'v', label: 'x', width: 3 });
      g.vector(ay, { color: 'w', label: s.morph > 0.5 ? 'Aᵀy' : 'y', width: 3 });
      if (s.shadow > 0) {
        const foot = proj(x, Aty);
        g.line(x, lerpV(x, foot, s.shadow), { color: 'residual', dashed: true, width: 1.2 });
        if (s.shadow >= 1) { g.line([0, 0], foot, { color: 'mag', width: 4, alpha: 0.85 }); g.point(foot, { r: 4, color: 'mag' }); g.dim([0, 0], foot, { label: 'x·(Aᵀy) = ' + LA.fmt(N.dot(x, Aty)), offset: 22, color: 'mag' }); if (s.pulse > 0) g.marker.ring(foot, { r: 14 + 10 * s.pulse, color: 'gold' }); g.hud('shadow ' + LA.fmt(N.norm(foot)) + ' × length of Aᵀy ' + LA.fmt(N.norm(Aty)) + ' = ' + LA.fmt(N.dot(x, Aty)), r.x + 10, r.y + r.h - 16, { color: 'mag', size: 13 }); }
      }
      g.hud('transform y by Aᵀ, measure on x', r.x + 10, r.y + 18, { color: 'hud' });
    };
    g.split(left, right, { left: { view: '2d', cam: LA.clone(cam) }, right: { view: '2d', cam: LA.clone(cam) } });
  }

  function sceneNotInv(g, s, ctx) {
    g.ghostGrid();
    g.grid(s.M);
    g.vector(N.col(s.M, 0), { color: 'i', label: 'i', hat: true, width: 3 });
    g.vector(N.col(s.M, 1), { color: 'j', label: 'j', hat: true, width: 3 });
    const h = ctx.size.h, w = ctx.size.w;
    g.hud(s.stage === 0 ? 'apply A …' : '… then apply Aᵀ', 12, 20, { color: 'hud', size: 14 });
    if (s.cross) { g.hud('Aᵀ undoes A', 14, h - 34, { color: 'bad', size: 15 }); g.marker.cross([118, h - 34], { screen: true, size: 22 }); g.hud('the grid is far from the dashed ghost', 14, h - 14, { color: 'muted', size: 12 }); }
    if (s.insetOn) {
      g.viewport({ x: 0.01, y: 0.075, w: 0.37, h: 0.37 }, { view: '2d', cam: { center: [0, 0], halfH: 2.3 }, bg: 'panel' }, (g, r) => {
        g.ghostGrid(); const R = N.rot2(s.th * DEG); g.grid(R);
        g.vector(N.col(R, 0), { color: 'i', label: 'i', hat: true, width: 2.4 }); g.vector(N.col(R, 1), { color: 'j', label: 'j', hat: true, width: 2.4 });
        g.hud(s.back < 1 ? 'rotation R(40°), then Rᵀ' : 'RᵀR = I: back on the ghost', r.x + 8, r.y + 14, { color: s.back < 1 ? 'hud' : 'ok', size: 12 });
        if (s.check) g.marker.check([r.x + r.w - 22, r.y + 22], { screen: true, size: 22 });
      });
    }
  }

  function sceneProdT(g, s) {
    const slotL = [-1.75, 1.35], slotR = [1.75, 1.35], tL = [-0.8, 1.35], tR = [0.8, 1.35], w = 1.5, h = 0.95;
    const u1 = s.merge, u2 = s.swap;
    const arc = Math.sin(Math.PI * u2);
    const pa = N.add(lerpV(lerpV(slotL, tL, u1), tR, u2), [0, 1.0 * arc]);
    const pb = N.add(lerpV(lerpV(slotR, tR, u1), tL, u2), [0, -1.0 * arc]);
    const flipped = u2 >= 0.5;
    tile(g, pa, w, h, flipped ? 'Aᵀ' : 'A', flipped ? '3×2' : '2×3', { color: 'i' });
    tile(g, pb, w, h, flipped ? 'Bᵀ' : 'B', flipped ? '4×3' : '3×4', { color: 'j' });
    const touching = (u1 >= 1 && u2 <= 0) || u2 >= 1;
    if (touching) { g.marker.ring([0, 1.35], { r: 16, color: 'gold' }); g.label(flipped ? 'inner 3s meet: Bᵀ then Aᵀ chains' : 'inner 3s meet: the product chains', [0, 2.15], { color: 'gold', size: 13, font: 'ui' }); }
    if (s.result > 0) {
      const name = flipped && u2 >= 1 ? '(AB)ᵀ = BᵀAᵀ' : 'AB';
      tile(g, [0, -0.55], 2.6, 1.0, name, flipped && u2 >= 1 ? '4×2' : '2×4', { color: 'gold', alpha: s.result * 0.92 });
      g.label(flipped && u2 >= 1 ? 'transpose of the product: inputs and outputs trade places, so the order reverses' : 'A then B written as AB: (2×3)(3×4) → 2×4', [0, -1.5], { color: 'muted', size: 12.5, font: 'ui' });
    }
    if (u2 > 0 && u2 < 1) g.label('flip each factor and swap their order', [0, 2.15], { color: 'hud', size: 13, font: 'ui' });
  }

  function outerLeft(g, s, r) {
    const u = s.u, v = s.v, M = N.outer(u, v), rg = rangeOf(M), cs = 0.85;
    const x0 = -0.55, y0 = 0.75, n = Math.max(0, Math.min(4, Math.floor(s.fill + 1e-9)));
    cells(g, u.map(a => [a]), { x0: x0 - cs - 0.35, y0, cs, bracket: true, textColor: 'v', brColor: 'v' });
    g.label('u', [x0 - cs / 2 - 0.35, y0 + 0.12], { color: 'v', size: 15, anchor: 'above', bg: false });
    cells(g, [v], { x0, y0: y0 + cs + 0.35, cs, bracket: true, textColor: 'w', brColor: 'w' });
    g.label('vᵀ', [x0 - 0.3, y0 + cs / 2 + 0.35], { color: 'w', size: 15, anchor: 'right', bg: false });
    const shown = (i, j) => i * 2 + j < n;
    cells(g, M, { x0, y0, cs, bracket: true, fill: (i, j, x) => (shown(i, j) ? heatFill(rg[0], rg[1])(i, j, x) : 'panel'), textAt: (i, j, x) => (shown(i, j) ? heatText(rg[0], rg[1])(i, j, x) : 'panel'), size: 17 });
    for (let i = 0; i < 2; i++) for (let j = 0; j < 2; j++) if (shown(i, j)) g.label(LA.fmt(u[i]) + '·' + LA.fmt(v[j]), [x0 + (j + 0.5) * cs, y0 - (i + 0.5) * cs - 0.27], { color: 'muted', size: 10.5, font: 'ui', bg: false });
    g.label('uvᵀ', [x0 + cs, y0 - 2 * cs - 0.3], { color: 'text', size: 15, font: 'math', anchor: 'below' });
    if (n === 4 || s.fill >= 4) g.label('each column is a multiple of u', [x0 + cs, y0 - 2 * cs - 0.75], { color: 'text', size: 12.5, font: 'ui', anchor: 'below' });
    else g.label('cell (i, j) = uᵢ · vⱼ', [x0 + cs, y0 - 2 * cs - 0.75], { color: 'muted', size: 12.5, font: 'ui', anchor: 'below' });
    g.hud('(2×1)(1×2) → 2×2', r.x + 10, r.y + 18, { color: 'hud' });
  }
  function outerRight(g, s, r, handles) {
    const u = s.u, v = s.v, M = lerpM(I2(), N.outer(u, v), s.morph);
    g.ghostGrid(); g.grid(M, { alpha: 0.9 });
    if (s.target) { g.lineThrough([0, 0], [1, 2], { color: 'gold', dashed: true, width: 1.6, alpha: 0.9, len: 12 }); g.label('target: y = 2x', [1.1, 2.2], { color: 'gold', size: 13, font: 'ui', anchor: 'left', dx: 8 }); }
    if (s.morph > 0.6 && N.norm(u) > 1e-6) { g.lineThrough([0, 0], u, { color: 'teal', width: 2.2, alpha: 0.5 + 0.5 * s.morph, len: 12 }); }
    g.vector(v, { color: 'w', label: 'v', width: 2.4 });
    g.vector(u, { color: 'v', label: 'u', width: 3 });
    if (s.rank) { g.label('rank 1: the whole plane', [0.6, -2.3], { color: 'teal', size: 13, font: 'ui' }); g.label('lands on the teal line of u', [0.6, -2.75], { color: 'teal', size: 13, font: 'ui' }); }
    if (handles) {
      g.handle('u', u, (p, st) => { st.u = [p[0], p[1]]; }, { color: 'v' });
      g.handle('v', v, (p, st) => { st.v = [p[0], p[1]]; }, { color: 'w' });
    }
    if (r) g.hud('the map uvᵀ', r.x + 10, r.y + 18, { color: 'hud' });
  }
  function sceneOuter(g, s, handles) {
    g.split((g, r) => outerLeft(g, s, r), (g, r) => outerRight(g, s, r, handles), { ratio: 0.46, left: { view: '2d', cam: fit(g, 0.46, 1, -2.3, 1.7, -2.2, 2.4) }, right: { view: '2d', cam: fit(g, 0.54, 1, -1.4, 3.6, -3.0, 3.3) } });
  }

  function goldAxes(g, S, alpha) {
    const e = N.eigSym(S);
    for (let k = 0; k < 2; k++) {
      const d = e.vectors[k], lam = e.values[k];
      g.lineThrough([0, 0], d, { color: 'gold', dashed: true, width: 1, alpha: 0.35 * alpha, len: 10 });
      g.vector(N.scale(d, lam), { color: 'gold', label: 'e' + (k + 1) + ' · ' + LA.fmt(lam) + '×', width: 2.6, alpha });
    }
    g.arc([0, 0], e.vectors[0], e.vectors[1], { r: 0.45, color: 'gold', right: true });
    return e;
  }
  function symLeft(g, s, r, M, showAxes, handles) {
    g.ghostGrid(); g.grid(M, { alpha: 0.55 });
    g.circle([0, 0], 1, { stroke: 'ghostAxis', dashed: true, width: 1.2 });
    g.ellipse(M, null, { stroke: 'hud', width: 2, fill: 'v', alpha: 0.08 });
    g.vector(N.col(M, 0), { color: 'i', label: 'i', hat: true, width: 2.4, alpha: handles ? 1 : 0.7 });
    g.vector(N.col(M, 1), { color: 'j', label: 'j', hat: true, width: 2.4, alpha: handles ? 1 : 0.7 });
    if (showAxes > 0.02) goldAxes(g, [[M[0][0], (M[0][1] + M[1][0]) / 2], [(M[0][1] + M[1][0]) / 2, M[1][1]]], showAxes);
    if (handles) {
      g.handle('i', N.col(M, 0), (p, st) => { st.A[0][0] = p[0]; st.A[1][0] = p[1]; if (st.lockSym) st.A[0][1] = p[1]; }, { color: 'i' });
      g.handle('j', N.col(M, 1), (p, st) => { st.A[0][1] = p[0]; st.A[1][1] = p[1]; if (st.lockSym) st.A[1][0] = p[0]; }, { color: 'j' });
    }
    if (r) g.hud(showAxes > 0.5 ? 'symmetric: two perpendicular stretch axes' : 'S = Sᵀ', r.x + 10, r.y + 18, { color: showAxes > 0.5 ? 'gold' : 'hud' });
  }
  function symRight(g, s, r, M) {
    g.ghostGrid(); g.grid(M, { alpha: 0.55 });
    g.circle([0, 0], 1, { stroke: 'ghostAxis', dashed: true, width: 1.2 });
    g.ellipse(M, null, { stroke: 'hud', width: 2, fill: 'v', alpha: 0.08 });
    const ci = N.col(M, 0), cj = N.col(M, 1);
    g.vector(ci, { color: 'i', label: 'i', hat: true, width: 2.6 });
    g.vector(cj, { color: 'j', label: 'j', hat: true, width: 2.6 });
    if (s.morph2 > 0.02) {
      // the shear's only direction that stays on its own line: the x-axis (no perpendicular partner)
      g.lineThrough([0, 0], [1, 0], { color: 'gold', dashed: true, width: 1.6, alpha: 0.8 * s.morph2, len: 10 });
      g.label('stays on its line', [-2.35, 0.3], { color: 'gold', size: 12, font: 'ui', anchor: 'left', alpha: s.morph2 });
    }
    g.hud(s.morph2 >= 1 ? 'shear: one direction stays on its line, no partner' : 'shear H = [[1, 1],[0, 1]]', r.x + 10, r.y + 18, { color: s.morph2 >= 1 ? 'bad' : 'hud' });
  }
  function sceneSym(g, s) {
    const cam = fit(g, 0.5, 1, -2.4, 3.0, -2.4, 2.6);
    if (s.single) { const M = s.A; const near = Math.max(0, 1 - sym(M) / LA.TOL.pos); symLeft(g, s, null, M, near >= 0.999 ? 1 : 0, true); g.hud(sym(M) <= LA.TOL.pos ? 'symmetric: a₁₂ = a₂₁' : 'a₁₂ = ' + LA.fmt(M[0][1]) + ', a₂₁ = ' + LA.fmt(M[1][0]), 12, 20, { color: sym(M) <= LA.TOL.pos ? 'ok' : 'hud', size: 14 }); return; }
    const M = lerpM(I2(), s.S, s.morph), H = lerpM(I2(), H0, s.morph2);
    g.split((g, r) => symLeft(g, s, r, M, seg(s.morph, 0.6, 1), false), (g, r) => symRight(g, s, r, H), { left: { view: '2d', cam: LA.clone(cam) }, right: { view: '2d', cam: LA.clone(cam) } });
  }

  function sceneGram(g, s) {
    const B = s.B, cols = N.cols(B), BtB = N.matMul(T(B), B), BBt = N.matMul(B, T(B));
    const n = Math.max(0, Math.min(9, Math.floor(s.fill + 1e-9))), m = Math.max(0, Math.min(4, Math.floor(s.fillB + 1e-9)));
    const cur = n < 9 && s.fill > 0 ? [Math.floor(n / 3), n % 3] : null;      // the pair being measured
    const left = (g, r) => {
      g.ghostGrid(); g.axes({ labels: false });
      cols.forEach((c, k) => { const on = !cur || cur[0] === k || cur[1] === k; g.vector(c, { color: BASIS[k], label: 'b' + (k + 1), width: on ? 3 : 1.8, alpha: on ? 1 : 0.3 }); });
      if (cur) { const a = cols[cur[0]], b = cols[cur[1]]; const foot = proj(b, a); g.line(b, foot, { color: 'residual', dashed: true, width: 1.2 }); g.line([0, 0], foot, { color: 'mag', width: 4, alpha: 0.8 }); g.point(foot, { r: 3.5, color: 'mag' }); g.label('b' + (cur[0] + 1) + '·b' + (cur[1] + 1) + ' = ' + LA.fmt(N.dot(a, b)), [0.3, -0.5], { color: 'mag', size: 13, font: 'ui', anchor: 'left' }); }
      g.hud('the three columns of B as arrows', r.x + 10, r.y + 18, { color: 'hud' });
    };
    const right = (g, r) => {
      const cs = 0.74, rg = [0, 80];
      const x0 = -1.11, y0 = 2.75;
      g.label('BᵀB · 3×3 · columns compared', [x0 + 1.5 * cs, y0 + 0.14], { color: 'text', size: 13, anchor: 'above', font: 'ui', bg: false });
      cells(g, BtB, { x0, y0, cs, bracket: true, fill: (i, j, x) => (i * 3 + j < n ? heatFill(rg[0], rg[1])(i, j, x) : 'panel'), textAt: (i, j, x) => (i * 3 + j < n ? heatText(rg[0], rg[1])(i, j, x) : 'panel'), strokeAt: (i, j) => (i === j && n >= 9 ? 'gold' : 'line') });
      if (n >= 9) { g.line([x0, y0], [x0 + 3 * cs, y0 - 3 * cs], { color: 'gold', dashed: true, width: 1, alpha: 0.7 }); g.label('cell (i, j) = cell (j, i): symmetric', [x0 + 1.5 * cs, y0 - 3 * cs - 0.1], { color: 'gold', size: 12, anchor: 'below', font: 'ui' }); }
      const x1 = -0.74, y1 = -0.95;
      if (s.fillB > 0) {
        g.label('BBᵀ · 2×2 · rows compared', [x1 + cs, y1 + 0.14], { color: 'text', size: 13, anchor: 'above', font: 'ui', bg: false });
        cells(g, BBt, { x0: x1, y0: y1, cs, bracket: true, fill: (i, j, x) => (i * 2 + j < m ? heatFill(rg[0], rg[1])(i, j, x) : 'panel'), textAt: (i, j, x) => (i * 2 + j < m ? heatText(rg[0], rg[1])(i, j, x) : 'panel'), strokeAt: (i, j) => (i === j && m >= 4 ? 'gold' : 'line') });
        if (m >= 4) { g.line([x1, y1], [x1 + 2 * cs, y1 - 2 * cs], { color: 'gold', dashed: true, width: 1, alpha: 0.7 }); g.label('also symmetric', [x1 + cs, y1 - 2 * cs - 0.1], { color: 'gold', size: 12, anchor: 'below', font: 'ui' }); }
      }
      g.hud('Gram matrices', r.x + 10, r.y + 18, { color: 'hud' });
    };
    g.split(left, right, { ratio: 0.52, left: { view: '2d', cam: { center: [1.6, 2.9], halfH: 4.1 } }, right: { view: '2d', cam: { center: [0, 0.1], halfH: 3.5 } } });
  }

  function sceneML(g, s) {
    const tk = s.tokens, rows = Math.max(0, Math.min(4, Math.floor(s.fill + 1e-9)));
    const left = (g, r) => {
      const o = [0, 1.75];
      for (let k = -2.5; k <= 2.5; k += 1) { g.line([o[0] + k * 0.5, o[1] - 1.25], [o[0] + k * 0.5, o[1] + 1.25], { color: 'gridGhost', width: 1, alpha: 0.6 }); g.line([o[0] - 1.25, o[1] + k * 0.5], [o[0] + 1.25, o[1] + k * 0.5], { color: 'gridGhost', width: 1, alpha: 0.6 }); }
      tk.forEach((t, k) => { const on = s.fill < 4 && k === rows; g.vector(o, N.add(o, t), { color: on ? 'gold' : 'teal', label: 't' + (k + 1), width: on ? 3.2 : 2.2, alpha: on || s.fill >= 4 ? 1 : 0.75 }); });
      g.point(o, { r: 2.5, color: 'muted' });
      const cs = 0.55, x0 = -1.1, y0 = -0.4, rg = [-1.4, 1.4];
      const QK = tk.map(q => tk.map(k => N.dot(q, k)));
      g.label('QKᵀ · 4×4 scores', [x0 + 2 * cs, y0 + 0.12], { color: 'text', size: 13, anchor: 'above', font: 'ui', bg: false });
      cells(g, QK, { x0, y0, cs, bracket: true, size: 12, fill: (i, j, x) => (i < rows ? heatFill(rg[0], rg[1])(i, j, x) : 'panel'), textAt: (i, j, x) => (i < rows ? heatText(rg[0], rg[1])(i, j, x) : 'panel') });
      for (let i = 0; i < 4; i++) g.label('q' + (i + 1), [x0 - 0.18, y0 - (i + 0.5) * cs], { color: i === rows && s.fill < 4 ? 'gold' : 'muted', size: 11, anchor: 'right', bg: false });
      for (let j = 0; j < 4; j++) g.label('k' + (j + 1), [x0 + (j + 0.5) * cs, y0 - 4 * cs - 0.12], { color: 'muted', size: 11, anchor: 'below', bg: false });
      g.label(s.fill < 4 ? 'row ' + (rows + 1) + ': query ' + (rows + 1) + ' dotted with every key' : 'every query · every key (here qᵢ = kᵢ = tᵢ)', [x0 + 2 * cs, y0 - 4 * cs - 0.5], { color: s.fill < 4 ? 'gold' : 'muted', size: 12, font: 'ui', anchor: 'below' });
      g.hud('attention · Part 6', r.x + 10, r.y + 18, { color: 'hud' });
    };
    const right = (g, r) => {
      const cs = 0.66, x0 = -0.99, y0 = 2.75, G = s.XtX, rg = rangeOf(G);
      g.label('XᵀX · 3×3 · features compared', [x0 + 1.5 * cs, y0 + 0.12], { color: 'text', size: 13, anchor: 'above', font: 'ui', bg: false });
      cells(g, G, { x0, y0, cs, bracket: true, size: 12, digits: 1, fill: heatFill(rg[0], rg[1]), textAt: heatText(rg[0], rg[1]), strokeAt: (i, j) => (i === j ? 'gold' : 'line') });
      g.label('Gram matrix of 3 synthetic features · Part 3', [x0 + 1.5 * cs, y0 - 3 * cs - 0.1], { color: 'muted', size: 12, anchor: 'below', font: 'ui' });
      // the layer tile: forward by W, backward by Wᵀ
      const yc = -1.3;
      tile(g, [0, yc], 1.0, 0.8, null, 'W', { color: 'text', size: 18 });
      g.vector([-1.35, yc + 0.22], [-0.55, yc + 0.22], { color: 'v', label: 'x', width: 2.2, labelAt: 0, dx: -10, dy: 0 });
      g.vector([0.55, yc + 0.22], [1.35, yc + 0.22], { color: 'v', label: 'Wx', width: 2.2, dx: 0, dy: -14 });
      g.vector([1.35, yc - 0.22], [0.55, yc - 0.22], { color: 'b', label: 'δ', width: 2.2, labelAt: 0, dx: 10, dy: 0 });
      g.vector([-0.55, yc - 0.22], [-1.35, yc - 0.22], { color: 'b', label: 'Wᵀδ', width: 2.2, dx: 0, dy: 14 });
      g.label('forward: W · backward: Wᵀ', [0, yc + 0.75], { color: 'text', size: 13, font: 'ui', bg: false });
      g.label('the gradient crosses the layer by Wᵀ (step 4) · Part 5', [0, yc - 0.75], { color: 'muted', size: 12, font: 'ui' });
      g.hud('Gram · backprop', r.x + 10, r.y + 18, { color: 'hud' });
    };
    g.split(left, right, { ratio: 0.5, left: { view: '2d', cam: fit(g, 0.5, 1, -1.6, 1.4, -3.35, 3.1) }, right: { view: '2d', cam: fit(g, 0.5, 1, -2.05, 2.05, -2.3, 3.3) } });
  }

  function sceneRecap(g, s) {
    const h = 1 / 3;
    g.viewport({ x: 0, y: 0, w: 1, h }, { view: '2d', cam: fit(g, 1, h, -2.2, 2.6, -1.75, 2.2) }, (g, r) => { sceneFlip(g, Object.assign({}, s, { flip: 1, pulse: 0 }), { x0: -0.9, y0: 1.5 }); g.hud('1 · Aᵀ flips across the diagonal', r.x + 8, r.y + 16, { size: 12 }); });
    g.viewport({ x: 0, y: h, w: 1, h }, { view: '2d', cam: fit(g, 1, h, -3.2, 3.6, -2.3, 2.6) }, (g, r) => { outerRight(g, Object.assign({}, s, { morph: 1, rank: false, target: false }), null, false); g.hud('2 · uvᵀ: rank 1, the plane lands on the line of u', r.x + 8, r.y + 16, { size: 12 }); });
    g.viewport({ x: 0, y: 2 * h, w: 1, h }, { view: '2d', cam: fit(g, 1, h, -3.0, 3.0, -2.3, 2.3) }, (g, r) => { symLeft(g, s, null, s.S, 1, false); g.hud('3 · symmetric: perpendicular stretch axes', r.x + 8, r.y + 16, { size: 12 }); });
  }

  function scenePlay(g, s, ctx) {
    if (s.showOuter) { sceneOuter(g, Object.assign({}, s, { morph: 1, rank: true, target: false, fill: 4 }), true); return; }
    const A = s.A, x = s.x, y = s.y, Ax = N.matVec(A, x), Aty = N.matVec(T(A), y);
    g.ghostGrid(); g.grid(A, { alpha: 0.85 });
    if (s.showAt) g.grid(T(A), { color: 'teal', axisColor: 'teal', alpha: 0.45, dash: [5, 5] });
    const isSym = sym(A) <= 0.02;
    if (isSym) { g.circle([0, 0], 1, { stroke: 'ghostAxis', dashed: true, width: 1.2 }); g.ellipse(A, null, { stroke: 'hud', width: 1.8, fill: 'v', alpha: 0.07 }); goldAxes(g, A, 1); }
    g.vector(N.col(A, 0), { color: 'i', label: 'i', hat: true, width: 2.6 });
    g.vector(N.col(A, 1), { color: 'j', label: 'j', hat: true, width: 2.6 });
    g.vector(x, { color: 'v', label: 'x', width: 3 });
    g.vector(y, { color: 'w', label: 'y', width: 3 });
    g.vector(Ax, { color: 'v', label: 'Ax', width: 2, dashed: true, alpha: 0.8 });
    const f1 = proj(Ax, y); g.line(Ax, f1, { color: 'residual', dashed: true, width: 1 }); g.point(f1, { r: 3.5, color: 'mag' });
    if (s.showAt) { g.vector(Aty, { color: 'w', label: 'Aᵀy', width: 2, dashed: true, alpha: 0.8 }); const f2 = proj(x, Aty); g.line(x, f2, { color: 'residual', dashed: true, width: 1 }); g.point(f2, { r: 3.5, color: 'mag' }); }
    g.handle('i', N.col(A, 0), (p, st) => { st.A[0][0] = p[0]; st.A[1][0] = p[1]; if (st.lockSym) st.A[0][1] = p[1]; }, { color: 'i' });
    g.handle('j', N.col(A, 1), (p, st) => { st.A[0][1] = p[0]; st.A[1][1] = p[1]; if (st.lockSym) st.A[1][0] = p[0]; }, { color: 'j' });
    g.handle('x', x, (p, st) => { st.x = [p[0], p[1]]; }, { color: 'v' });
    g.handle('y', y, (p, st) => { st.y = [p[0], p[1]]; }, { color: 'w' });
    const h = ctx.size.h;
    g.hud('(Ax)·y = ' + LA.fmt(N.dot(Ax, y)) + '   x·(Aᵀy) = ' + LA.fmt(N.dot(x, Aty)) + '   (always equal)', 12, h - 34, { color: 'mag', size: 14 });
    g.hud(isSym ? 'A = Aᵀ: symmetric, gold stretch axes are perpendicular' : 'a₁₂ = ' + LA.fmt(A[0][1]) + ', a₂₁ = ' + LA.fmt(A[1][0]) + (s.lockSym ? ' (locked symmetric: drag a tip)' : ''), 12, h - 14, { color: isSym ? 'gold' : 'muted', size: 12.5 });
  }

  LA.registerLesson({
    id: 'p2-10-transpose-outer-symmetric',
    part: 2,
    title: 'Transpose, xᵀy, outer products, symmetric matrices',
    subtitle: 'Flip across the diagonal; a row times a column; a column times a row.',
    minutes: 10,
    prereqs: ['p2-02-matrix-vector', 'p1-06-dot-product'],
    tags: ['core'],
    badge: { drawnIn: '2D', realSize: 'Wᵀ (784×128) carries gradients backward through a 128×784 layer; QKᵀ is a 1000×1000 score table in attention; XᵀX is d×d', transfers: '(Ax)·y = x·(Aᵀy), rank(uvᵀ) = 1 and “symmetric ⇒ perpendicular stretch axes” are exact in every dimension' },
    cantShow: ['a 1000×1000 QKᵀ (we show 4×4 and say every cell is a dot product)'],
    terms: [
      { term: 'transpose', def: 'The matrix flipped across its diagonal, written Aᵀ: rows become columns and columns become rows.' },
      { term: 'row vector', def: 'A vector written lying down, xᵀ: the same numbers side by side instead of stacked.' },
      { term: 'outer product', def: 'A column times a row, uvᵀ: a matrix whose every column is a multiple of u, so it has rank 1.' },
      { term: 'symmetric matrix', def: 'A matrix equal to its own transpose, so the entry in row i, column j equals the entry in row j, column i.' },
      { term: 'Gram matrix', def: 'XᵀX or XXᵀ: a table of all pairwise dot products of the columns (or rows) of X; it is always symmetric.' },
    ],
    atlas: ['attention', 'backprop', 'linear-regression'],
    thumbnail: 7,

    state() {
      return {
        view: '2d', cam: LA.cam.defaults2d(), scene: 'flip',
        A: LA.clone(A0), B: LA.clone(B0), Bt: [[0, 0], [0, 0], [0, 0]], S: LA.clone(S0), M: I2(),
        x: [1, 1], y: [1, 0], u: [1, 2], v: [3, 1],
        tokens: tokens(), XtX: featureGram(),
        flip: 0, pulse: 0, ang: 0, slide: 0, lit: 0, sum: 0, shadow: 0, morph: 0, morph2: 0,
        stage: 0, cross: false, insetOn: false, th: 0, back: 0, check: false,
        merge: 0, swap: 0, result: 0, fill: 0, fillB: 0, rank: false, target: false, single: false,
        showAt: false, lockSym: false, showOuter: false,
      };
    },

    render(g, s, ctx) {
      switch (s.scene) {
        case 'flip': sceneFlip(g, s); break;
        case 'entry': sceneFlip(g, s, { entry: true, x0: -2.7, y0: 1.75 }); break;
        case 'rowvec': sceneRowvec(g, s); break;
        case 'dot': sceneDot(g, s); break;
        case 'adjoint': sceneAdjoint(g, s); break;
        case 'notinv': sceneNotInv(g, s, ctx); break;
        case 'prodT': sceneProdT(g, s); break;
        case 'outer': sceneOuter(g, s, ctx.mode !== 'learn'); break;
        case 'sym': sceneSym(g, s); break;
        case 'gram': sceneGram(g, s); break;
        case 'ml': sceneML(g, s); break;
        case 'recap': sceneRecap(g, s); break;
        case 'play': scenePlay(g, s, ctx); break;
        default: sceneFlip(g, s);
      }
    },

    math(s, tex) {
      const p = [];
      const AT = (name) => tex.T(tex.sym(name));
      // numbers as text spans: the engine's in-place reconcile mis-assigns tx-v nums when a matrix is also shown
      const num = (x, o) => tex.text(LA.fmt(x, o && o.digits), o);
      switch (s.scene) {
        case 'flip': case 'entry':
          p.push(tex.eq(tex.sym('B'), tex.op('='), tex.mat(s.B, { colors: 'cols' }), tex.dim(2, 3)));
          if (s.scene === 'flip') p.push(tex.eq(AT('B'), tex.op('='), tex.mat(T(s.B), { colors: 'rows' }), tex.dim(3, 2)));   // hidden in the check: it is the answer
          if (s.scene === 'entry') p.push(tex.eq(tex.text('your '), AT('B'), tex.op('='), tex.mat(s.Bt, { colors: 'rows' })));
          break;
        case 'rowvec':
          p.push(tex.eq(tex.sym('x', { color: 'v' }), tex.op('='), tex.vec(s.x, { color: 'v' })));
          p.push(tex.eq(AT('x'), tex.op('='), tex.rowvec(s.x, { color: 'v' })));
          break;
        case 'dot':
          p.push(tex.eq(AT('x'), tex.sym('y', { color: 'w' }), tex.op('='), tex.rowvec(s.x, { color: 'v' }), tex.vec(s.y, { color: 'w' }), tex.op('='), num(N.dot(s.x, s.y), { color: 'gold' }), tex.op('='), tex.sym('x', { color: 'v' }), tex.op('·'), tex.sym('y', { color: 'w' })));
          break;
        case 'adjoint': {
          const Ax = N.matVec(s.A, s.x), Aty = N.matVec(T(s.A), s.y);
          p.push(tex.eq(tex.sym('A'), tex.op('='), tex.mat(s.A, { colors: 'cols' }), tex.op('  '), AT('A'), tex.op('='), tex.mat(T(s.A), { colors: 'rows' })));
          p.push(tex.eq(tex.paren(tex.sym('A') + tex.sym('x', { color: 'v' })), tex.op('·'), tex.sym('y', { color: 'w' }), tex.op('='), tex.vec(Ax, { color: 'v' }), tex.op('·'), tex.vec(s.y, { color: 'w' }), tex.op('='), num(N.dot(Ax, s.y), { color: 'mag' })));
          p.push(tex.eq(tex.sym('x', { color: 'v' }), tex.op('·'), tex.paren(AT('A') + tex.sym('y', { color: 'w' })), tex.op('='), tex.vec(s.x, { color: 'v' }), tex.op('·'), tex.vec(Aty, { color: 'w' }), tex.op('='), num(N.dot(s.x, Aty), { color: 'mag' })));
          break; }
        case 'notinv':
          p.push(tex.eq(AT('A'), tex.sym('A'), tex.op('='), tex.mat(N.matMul(T(s.A), s.A)), tex.op('≠'), tex.sym('I')));
          if (s.insetOn) p.push(tex.eq(AT('R'), tex.sym('R'), tex.op('='), tex.sym('I'), tex.text(' for a rotation', { color: 'ok' })));
          break;
        case 'prodT':
          p.push(tex.eq(tex.T(tex.paren(tex.sym('A') + tex.sym('B'))), tex.op('='), AT('B'), AT('A')));
          p.push(tex.eq(tex.dim(2, 3), tex.dim(3, 4), tex.op('→'), tex.dim(2, 4), tex.op('   '), tex.dim(4, 3), tex.dim(3, 2), tex.op('→'), tex.dim(4, 2)));
          break;
        case 'outer':
          p.push(tex.eq(tex.sym('u', { color: 'v' }), AT('v').replace('tx-sym', 'tx-sym c-w'), tex.op('='), tex.vec(s.u, { color: 'v' }), tex.rowvec(s.v, { color: 'w' }), tex.op('='), tex.mat(N.outer(s.u, s.v))));
          p.push(tex.text('rank 1: every column is a multiple of u', { color: 'teal' }));
          break;
        case 'sym': {
          const M = s.single ? s.A : s.S;
          p.push(tex.eq(tex.sym(s.single ? 'A' : 'S'), tex.op('='), tex.mat(M, { colors: 'cols' }), tex.op('  '), AT(s.single ? 'A' : 'S'), tex.op('='), tex.mat(T(M), { colors: 'rows' })));
          if (sym(M) <= LA.TOL.pos) { const e = N.eigSym(M); p.push(tex.eq(tex.text('symmetric: ', { color: 'gold' }), tex.sym('S'), tex.vec(e.vectors[0].map(x => x * Math.SQRT2), { color: 'gold' }), tex.op('='), num(e.values[0], { color: 'gold' }), tex.vec(e.vectors[0].map(x => x * Math.SQRT2), { color: 'gold' }), tex.op(' ⟂ '), tex.vec(e.vectors[1].map(x => x * Math.SQRT2), { color: 'gold' }))); }
          else p.push(tex.text('a₁₂ ≠ a₂₁: not symmetric', { color: 'bad' }));
          break; }
        case 'gram':
          p.push(tex.eq(AT('B'), tex.sym('B'), tex.op('='), tex.mat(N.matMul(T(s.B), s.B)), tex.dim(3, 3)));
          p.push(tex.eq(tex.sym('B'), AT('B'), tex.op('='), tex.mat(N.matMul(s.B, T(s.B))), tex.dim(2, 2)));
          break;
        case 'ml':
          p.push(tex.eq(tex.paren(tex.sym('Q') + AT('K')), tex.sym('', { sub: 'ij' }), tex.op('='), tex.sym('q', { sub: 'i' }), tex.op('·'), tex.sym('k', { sub: 'j' })));
          p.push(tex.eq(AT('X'), tex.sym('X'), tex.op('='), tex.mat(s.XtX, { digits: 1 })));
          p.push(tex.eq(tex.sym('δ', { sub: 'in', color: 'b' }), tex.op('='), AT('W'), tex.sym('δ', { sub: 'out', color: 'b' })));
          break;
        case 'play': {
          const Ax = N.matVec(s.A, s.x), Aty = N.matVec(T(s.A), s.y);
          if (s.showOuter) { p.push(tex.eq(tex.sym('u', { color: 'v' }), AT('v'), tex.op('='), tex.vec(s.u, { color: 'v' }), tex.rowvec(s.v, { color: 'w' }), tex.op('='), tex.mat(N.outer(s.u, s.v)))); break; }
          p.push(tex.eq(tex.sym('A'), tex.op('='), tex.mat(s.A, { colors: 'cols' }), tex.op('  '), AT('A'), tex.op('='), tex.mat(T(s.A), { colors: 'rows' })));
          p.push(tex.eq(tex.paren(tex.sym('A') + tex.sym('x', { color: 'v' })), tex.op('·'), tex.sym('y', { color: 'w' }), tex.op('='), num(N.dot(Ax, s.y), { color: 'mag' }), tex.op('='), tex.sym('x', { color: 'v' }), tex.op('·'), tex.paren(AT('A') + tex.sym('y', { color: 'w' }))));
          if (sym(s.A) <= 0.02) { const e = N.eigSym(s.A); p.push(tex.eq(tex.text('symmetric: stretch ', { color: 'gold' }), num(e.values[0], { color: 'gold' }), tex.text('× along '), tex.vec(e.vectors[0], { color: 'gold' }), tex.text(' and '), num(e.values[1], { color: 'gold' }), tex.text('× along '), tex.vec(e.vectors[1], { color: 'gold' }))); }
          break; }
        default: return '';
      }
      return tex.block.apply(null, p);
    },

    steps: [
      { title: 'Flip across the diagonal', duration: 4,
        narration: 'The <b>transpose</b> Bᵀ flips a matrix across its diagonal: rows become columns. For B = [[1, 2, 3],[4, 5, 6]], a 2×3, the transpose is 3×2: the first row [1, 2, 3] stands up as the first column. The gold diagonal entries do not move; every other entry trades places with its mirror.',
        setup(s) { s.scene = 'flip'; s.flip = 0; s.pulse = 0; },
        at(s, t) { s.flip = seg(t, 0.12, 0.8); s.pulse = seg.pulse(t, 0.8, 1); } },

      { title: 'A vector lying down', duration: 3,
        narration: 'A column vector transposed is a <b>row vector</b>, written lying down: xᵀ = [1 1]. Same numbers, sideways. The yellow <span class="c-v">x</span> on the grid does not change; only the way it is written does. We will need lying-down vectors for exactly one thing: putting a row next to a column.',
        setup(s) { s.scene = 'rowvec'; s.x = [1, 1]; s.ang = 0; },
        at(s, t) { s.ang = seg(t, 0.2, 0.7) * Math.PI / 2; } },

      { title: 'Row times column is the dot product', duration: 4.5,
        narration: 'Put the row xᵀ in front of the column y and multiply as a (1×2)(2×1) product: one row dotted with one column, a 1×1 result. That is the dot product x·y = 1·1 + 1·0 = 1, the purple <span class="c-w">y</span>’s shadow on the yellow <span class="c-v">x</span>. Writing x·y as xᵀy is the habit of every ML paper, and the atom of every matrix product.',
        setup(s) { s.scene = 'dot'; s.x = [1, 1]; s.y = [1, 0]; s.slide = 0; s.lit = 0; s.sum = 0; s.shadow = 0; },
        at(s, t) { s.slide = seg(t, 0.1, 0.4); s.lit = seg(t, 0.45, 0.7, 'linear'); s.sum = seg(t, 0.7, 0.8); s.shadow = seg(t, 0.78, 0.95); } },

      { title: 'Aᵀ moves the matrix to the other side', duration: 5,
        narration: 'Why the transpose matters: (A<span class="c-v">x</span>)·<span class="c-w">y</span> = <span class="c-v">x</span>·(Aᵀ<span class="c-w">y</span>). Transform x and measure against y, or transform y by Aᵀ and measure against x: the same number. With A = [[2, 1],[0, 3]], x = [1; 1], y = [1; 0]: Ax = [3; 3], (Ax)·y = 3; Aᵀy = [2; 1], x·(Aᵀy) = 3. Each magenta shadow times its arrow’s length gives 3.',
        setup(s) { s.scene = 'adjoint'; s.A = LA.clone(A0); s.x = [1, 1]; s.y = [1, 0]; s.morph = 0; s.shadow = 0; s.pulse = 0; },
        at(s, t) { s.morph = seg(t, 0.05, 0.4); s.shadow = seg(t, 0.5, 0.75); s.pulse = seg.pulse(t, 0.82, 1); } },

      { title: 'The transpose is not the inverse', duration: 5, misconception: true,
        narration: 'Belief: “Aᵀ undoes A.” For A = [[2, 1],[0, 3]], apply A, then Aᵀ: the grid lands on AᵀA = [[4, 2],[2, 10]], nowhere near the dashed ghost. Red ✕. It is true for rotations and reflections, whose columns are perpendicular unit vectors: in the inset, R turns by 40° and Rᵀ brings the grid back exactly. Part 3 calls those matrices orthogonal.',
        setup(s) { s.scene = 'notinv'; s.cam = Object.assign(LA.cam.defaults2d(), { center: [1.2, 4.2], halfH: 7.2 }); s.A = LA.clone(A0); s.M = I2(); s.stage = 0; s.cross = false; s.insetOn = false; s.th = 0; s.back = 0; s.check = false; },
        at(s, t) {
          const AtA = N.matMul(T(A0), A0);
          const u1 = seg(t, 0.02, 0.3), u2 = seg(t, 0.36, 0.62);
          s.M = u2 > 0 ? lerpM(A0, AtA, u2) : lerpM(I2(), A0, u1);
          s.stage = t < 0.34 ? 0 : 1; s.cross = t >= 0.66;
          s.insetOn = t >= 0.7; s.back = seg(t, 0.86, 0.96); s.th = 40 * seg(t, 0.72, 0.84) - 40 * s.back; s.check = t >= 0.97;
        } },

      { title: 'Transposing a product reverses the order', duration: 4,
        narration: '(AB)ᵀ = BᵀAᵀ: flip a product and the factors swap order, each transposed. The shape tiles show why: (2×3)(3×4) is 2×4; its transpose is 4×2, which only chains as (4×3)(3×2) = BᵀAᵀ. Order reverses because inputs and outputs trade places.',
        setup(s) { s.scene = 'prodT'; s.cam = LA.cam.defaults2d(); s.merge = 0; s.swap = 0; s.result = 0; },
        at(s, t) { s.merge = seg(t, 0.05, 0.3); s.result = seg(t, 0.32, 0.42); s.swap = seg(t, 0.5, 0.9); } },

      { title: 'Column times row: the outer product', duration: 5,
        narration: 'Reverse the dot product’s shape: a column times a row, (2×1)(1×2), is a 2×2 matrix, the <b>outer product</b> uvᵀ. With <span class="c-v">u</span> = [1; 2], <span class="c-w">v</span> = [3; 1]: [[3, 1],[6, 2]]. Each cell is uᵢvⱼ; each column is a multiple of u. As a map it collapses the whole plane onto the teal line of u: rank 1.',
        setup(s) { s.scene = 'outer'; s.u = [1, 2]; s.v = [3, 1]; s.fill = 0; s.morph = 0; s.rank = false; s.target = false; },
        at(s, t) { s.fill = 5 * seg(t, 0.05, 0.45, 'linear'); s.morph = seg(t, 0.5, 0.88); s.rank = t >= 0.9; } },

      { title: 'Symmetric matrices: perpendicular stretch axes', duration: 5,
        narration: 'A matrix equal to its own transpose, aᵢⱼ = aⱼᵢ, is a <b>symmetric matrix</b>. S = [[2, 1],[1, 2]] stretches along two perpendicular gold directions: [1; 1] three times, [1; −1] not at all, so the circle becomes an ellipse on those axes. The shear has no such pair: only one direction, along the green <span class="c-i">î</span>, stays on its line. Part 4 proves symmetric always means perpendicular.',
        setup(s) { s.scene = 'sym'; s.S = LA.clone(S0); s.single = false; s.morph = 0; s.morph2 = 0; },
        at(s, t) { s.morph = seg(t, 0.05, 0.55); s.morph2 = seg(t, 0.62, 0.95); } },

      { title: 'XᵀX and XXᵀ: tables of dot products', duration: 5,
        narration: 'Multiply any matrix by its transpose and you get a symmetric one: (XᵀX)ᵀ = XᵀX. Its cell (i, j) is column i dotted with column j, a table of all pairwise dot products, a <b>Gram matrix</b>. For B: BᵀB is 3×3 (columns compared), BBᵀ is 2×2 (rows compared). A covariance matrix is a Gram matrix too.',
        setup(s) { s.scene = 'gram'; s.fill = 0; s.fillB = 0; },
        at(s, t) { s.fill = 10 * seg(t, 0.03, 0.6, 'linear'); s.fillB = 5 * seg(t, 0.68, 0.92, 'linear'); } },

      { title: 'Where ᵀ lives in ML', duration: 5, mlTie: true,
        narration: 'Three places. Attention: QKᵀ is every query dotted with every key, a 4×4 score table here, each cell a dot product. Regression: XᵀX is the Gram matrix of the features. Backprop: a gradient flows backward through a layer by Wᵀ, the transpose moving the matrix to the other side of the dot product, exactly as in step 4.',
        setup(s) { s.scene = 'ml'; s.fill = 0; },
        at(s, t) { s.fill = 5 * seg(t, 0.05, 0.6, 'linear'); } },

      { title: 'Recap', duration: 2, recap: true,
        narration: 'Transpose flips rows and columns and moves a matrix across a dot product: (Ax)·y = x·(Aᵀy). Row times column is the dot product; column times row is the rank-1 outer product. Symmetric matrices stretch along perpendicular axes, and XᵀX is always symmetric: a table of pairwise dot products.',
        cards: ['<b>Aᵀ flips rows and columns.</b> It moves across a dot product: (Ax)·y = x·(Aᵀy).', '<b>xᵀy is the dot product; uvᵀ is a rank-1 matrix.</b> Row over column, column over row.', '<b>Symmetric ⇒ perpendicular stretch axes.</b> XᵀX is always symmetric: pairwise dot products.'],
        setup(s) { s.scene = 'recap'; s.flip = 1; s.u = [1, 2]; s.v = [3, 1]; s.S = LA.clone(S0); },
        at(s, t) { } },
    ],

    playground: {
      intro: '<p>Drag the tips of <span class="c-i">î</span>, <span class="c-j">ĵ</span>, <span class="c-v">x</span> and <span class="c-w">y</span>. The two dot products stay equal whatever you do. Make A symmetric to see its gold axes; switch to the outer product to drag u and v.</p>',
      setup(s) { s.scene = 'play'; s.A = LA.clone(A0); s.x = [1, 1]; s.y = [1, 0]; s.u = [1, 2]; s.v = [3, 1]; s.showAt = true; s.lockSym = false; s.showOuter = false; s.cam = LA.cam.defaults2d(); },
      controls: [
        { type: 'matrix', bind: 'A', label: 'A', colors: 'cols', step: 0.1 },
        { type: 'toggle', bind: 'showAt', label: 'Show the map Aᵀ (teal grid)' },
        { type: 'toggle', bind: 'lockSym', label: 'Keep A symmetric while dragging' },
        { type: 'toggle', bind: 'showOuter', label: 'Outer product uvᵀ (drag u, v)' },
        { type: 'matrix', bind: 'x', label: 'x', colors: 'v', step: 0.1 },
        { type: 'preset', label: 'Presets', items: [
          { label: 'Symmetric', values: { A: S0, showOuter: false } },
          { label: 'Shear', values: { A: H0, showOuter: false } },
          { label: 'Rotation (Aᵀ = A⁻¹)', values: { A: [[0.77, -0.64], [0.64, 0.77]], showOuter: false, showAt: true } },
          { label: 'Outer product', values: { showOuter: true, u: [1, 2], v: [3, 1] } },
        ] },
      ],
    },

    checks: [
      { kind: 'entry',
        prompt: 'Type Bᵀ for B = [[1, 2, 3],[4, 5, 6]]. Each cell lights up when it matches its mirror in B.',
        bind: 'Bt', answer: [[1, 4], [2, 5], [3, 6]], tol: LA.TOL.entry, label: 'Bᵀ', colors: 'rows',
        setup(s) { s.scene = 'entry'; s.flip = 0; s.pulse = 0; s.Bt = [[0, 0], [0, 0], [0, 0]]; s.cam = { center: [-0.4, -0.3], halfH: 3.6 }; } },
      { kind: 'manipulate',
        prompt: 'Drag <span class="c-v">u</span> and <span class="c-w">v</span> so that the column space of uvᵀ (the teal line) is the gold line y = 2x.',
        setup(s) { s.scene = 'outer'; s.u = [1, 0]; s.v = [1, 1]; s.fill = 4; s.morph = 1; s.rank = false; s.target = true; },
        judge(s) {
          const nu = N.norm(s.u), nv = N.norm(s.v);
          const a = LA.judge.angleDeg(s.u, [1, 2], 0, LA.TOL.angleDeg).angle, d = Math.min(a, 180 - a);
          const ok = d <= LA.TOL.angleDeg && nu > 0.3 && nv > 0.3;
          return { ok, progress: Math.max(0, 1 - d / 90), hint: nu <= 0.3 || nv <= 0.3 ? 'Keep both u and v longer than zero.' : undefined };
        },
        hint: 'Only <span class="c-v">u</span> decides the output line; <span class="c-w">v</span> only decides how much each input contributes. Point u along [1; 2].',
        solution(s) { s.u = [1, 2]; } },
      { kind: 'manipulate',
        prompt: 'Make A symmetric by dragging the landing spots of <span class="c-i">î</span> and <span class="c-j">ĵ</span>; the gold stretch axes appear when a₁₂ = a₂₁.',
        setup(s) { s.scene = 'sym'; s.single = true; s.A = LA.clone(A0); s.lockSym = false; s.cam = { center: [0.4, 0.4], halfH: 3.6 }; },
        judge(s) { const d = sym(s.A); return { ok: d <= LA.TOL.pos, progress: Math.max(0, 1 - d / 1.5) }; },
        hint: 'The top-right and bottom-left entries must agree: lift the second coordinate of ' + ihat + ' to 1, or lower the first coordinate of ' + jhat + ' to 0.',
        solution(s) { s.A = [[2, 1], [1, 3]]; } },
    ],
  });
})();
