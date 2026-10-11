/* p4-07 — SVD: rotate, stretch, rotate. */
(function () {
  'use strict';
  const N = LA.num, seg = LA.seg, lerpM = LA.lerpM, lerp = LA.lerp;
  const I2 = () => [[1, 0], [0, 1]];
  const A0 = [[2, 1], [0, 1.5]];
  const SHEAR = [[1, 1], [0, 1]];
  const ROT60 = N.rot2(Math.PI / 3);
  const FLIP = [[3, 0], [0, -1]];
  const A32 = [[1, 0], [0, 1], [1, 1]];
  const HONEST = 'The picture stops at 3. The formula does not.';
  const gold = x => '<span class="c-gold">' + x + '</span>';
  const ihat = '<span class="c-i">î</span>', jhat = '<span class="c-j">ĵ</span>';

  /* ---------- SVD helpers (2×2) ---------- */
  /** rotate–stretch–rotate factors; U2 has det ±1 so that both σ ≥ 0 */
  function parts(A) {
    const r = N.svd2rot(A);
    const flip = r.S[1] < 0;
    const RU = N.rot2(r.thetaU), RV = N.rot2(r.thetaV);
    const U = flip ? N.matMul(RU, [[1, 0], [0, -1]]) : RU;
    return { thetaU: r.thetaU, thetaV: r.thetaV, S: r.S, s1: r.S[0], s2: Math.abs(r.S[1]), flip, RU, RV, U, V: RV,
      u1: N.col(U, 0), u2: N.col(U, 1), v1: N.col(RV, 0), v2: N.col(RV, 1) };
  }
  /** A with σ₂ scaled */
  function scaled(A, k) { const p = parts(A); return N.matMul(N.matMul(p.RU, N.diag([p.S[0], p.S[1] * k])), N.transpose(p.RV)); }
  /** the three moves: stage 0..3 → matrix (tween angles, not entries) */
  function stageM(A, st) {
    const p = parts(A);
    if (st <= 1) return N.rot2(-p.thetaV * st);
    const Rv = N.rot2(-p.thetaV);
    if (st <= 2) { const u = st - 1; return N.matMul(N.diag([lerp(1, p.S[0], u), lerp(1, p.S[1], u)]), Rv); }
    const u = st - 2; return N.matMul(N.rot2(p.thetaU * u), N.matMul(N.diag(p.S), Rv));
  }

  /* ---------- the 32×32 image (computed once, outside the lesson state) ---------- */
  let IMG = null;
  function img() {
    if (!IMG) {
      const A = LA.rng.smallImage(48, { size: 32 }); const sv = N.svd(A);
      IMG = { A, U: N.cols(sv.U), S: sv.S, V: N.cols(sv.V), rec: {} };
    }
    return IMG;
  }
  function recon(k) {
    const D = img(); if (D.rec[k]) return D.rec[k];
    let R = N.zerosM(32, 32);
    for (let i = 0; i < k; i++) R = N.addM(R, N.scaleM(N.outer(D.U[i], D.V[i]), D.S[i]));
    D.rec[k] = R; return R;
  }
  /** a planted decaying spectrum of a synthetic 784×128 weight matrix (W = Q₁ diag(σ) Q₂ᵀ has exactly these σ) */
  function spectrum128() {
    const rand = LA.rng(47), out = [];
    for (let i = 0; i < 128; i++) out.push(6 * Math.exp(-i / 9) + 0.35 + 0.04 * Math.abs(LA.rng.gauss(rand)));
    return out.sort((a, b) => b - a).map(x => Math.round(x * 1000) / 1000);
  }

  /* ---------- drawing ---------- */
  function radii(g, M, alpha) {
    for (let k = 0; k < 8; k++) { const a = k * Math.PI / 4 + 0.2; const p = [Math.cos(a), Math.sin(a)]; g.line([0, 0], N.matVec(M, p), { color: 'muted', width: 1, alpha: alpha === undefined ? 0.6 : alpha }); }
  }
  function axesSU(g, A, o) {
    o = o || {};
    const p = parts(A);
    const a1 = N.scale(p.u1, p.s1), a2 = N.scale(p.u2, p.s2);
    g.vector(a1, { color: 'gold', width: 3, label: o.plain ? 'long axis' : 'σ₁u₁', labelColor: 'gold' });
    if (p.s2 > 0.02) g.vector(a2, { color: 'gold', width: 3, label: o.plain ? 'short axis' : 'σ₂u₂', labelColor: 'gold' });
    if (p.s2 > 0.05) g.arc([0, 0], p.u1, p.u2, { r: 0.3, color: 'gold', right: true });
  }
  function vsDashed(g, A, o) {
    o = o || {};
    const p = parts(A);
    g.vector(p.v1, { color: 'gold', width: 2, dashed: true, head: 9, label: 'v₁', labelColor: 'gold' });
    g.vector(p.v2, { color: 'gold', width: 2, dashed: true, head: 9, label: 'v₂', labelColor: 'gold' });
  }
  function tiles(g, names, active, W) {
    names.forEach((nm, i) => g.hud(nm, 14 + i * (W - 20) / names.length, 22, { color: active === i ? 'gold' : (active > i ? 'hud' : 'muted'), size: 13.5, bold: active === i }));
  }
  function colHandles(g, s, key) {
    const A = s[key || 'A'], c1 = N.col(A, 0), c2 = N.col(A, 1);
    g.vector(c1, { color: 'i', width: 1.8, label: 'i', hat: true, alpha: 0.9 });
    g.vector(c2, { color: 'j', width: 1.8, label: 'j', hat: true, alpha: 0.9 });
    g.handle('c1', c1, (q, st) => { st[key || 'A'][0][0] = r2(q[0]); st[key || 'A'][1][0] = r2(q[1]); }, { color: 'i' });
    g.handle('c2', c2, (q, st) => { st[key || 'A'][0][1] = r2(q[0]); st[key || 'A'][1][1] = r2(q[1]); }, { color: 'j' });
  }
  const r2 = x => Math.round(LA.clamp(x, -4, 4) * 100) / 100;
  function slider(g, s, rect, bind, label, lo, hi) {
    g.viewport(rect, { view: '2d', cam: { center: [0, 0], halfH: 0.32 } }, (g2) => {
      const X = f => -2 + 4 * f;
      g2.line([X(0), -0.06], [X(1), -0.06], { color: 'gridAxis', width: 2 });
      [0, 0.5, 1].forEach(f => g2.label(LA.fmt(lerp(lo, hi, f)), [X(f), -0.1], { anchor: 'below', dy: 4, size: 10.5, color: 'muted', font: 'ui' }));
      const x = X((s[bind] - lo) / (hi - lo));
      g2.label(label + ' = ' + LA.fmt(s[bind]) + '  (drag)', [X(0), 0.17], { anchor: 'left', size: 12, color: 'v', font: 'ui' });
      g2.point([x, -0.06], { r: 6, color: 'v' });
      g2.handle(bind, [x, -0.06], (q, st) => { st[bind] = Math.round(lerp(lo, hi, LA.clamp01((q[0] + 2) / 4)) * 100) / 100; }, { color: 'v' });
    });
  }
  function ticksLine(g, rows, y0, o) {
    o = o || {};
    const lo = -2, hi = 4, x0 = -0.9, x1 = 2.2, X = l => x0 + (l - lo) / (hi - lo) * (x1 - x0);
    rows.forEach((row, k) => {
      const y = y0 - k * 0.42;
      g.line([x0, y], [x1, y], { color: 'gridAxis', width: 1 });
      for (let l = lo; l <= hi; l++) g.line([X(l), y - 0.04], [X(l), y + 0.04], { color: l === 0 ? 'text' : 'gridAxis', width: l === 0 ? 2 : 1 });
      g.label(row.name, [x0, y], { anchor: 'right', dx: -6, size: 11.5, color: row.color, font: 'ui' });
      row.vals.forEach(v => { g.point([X(v), y], { r: 5, color: row.color }); g.label(LA.fmt(v), [X(v), y], { anchor: 'above', dy: -6, size: 11, color: row.color, font: 'ui' }); });
    });
    for (let l = lo; l <= hi; l++) g.label(l < 0 ? '−' + (-l) : String(l), [X(l), y0 - (rows.length - 1) * 0.42], { anchor: 'below', dy: 6, size: 10, color: 'muted', font: 'ui' });
  }

  /* ---------- scenes ---------- */
  function renderMain(g, s, ctx) {
    const W = ctx.size.w;
    const At = s.sigma2Scale < 1 ? scaled(s.A, s.sigma2Scale) : s.A;
    const staged = s.mode === 'stage' || s.mode === 'reverse' || (s.mode === 'play' && s.stage < 2.999);
    const M = s.mode === 'morph' ? lerpM(I2(), At, s.morph) : staged ? stageM(At, s.stage) : At;
    g.ghostGrid();
    g.grid(M, { alpha: 0.55 });
    g.circle([0, 0], 1, { stroke: 'hud', dashed: true, width: 1.2, strokeAlpha: 0.6 });
    const p = parts(At);
    g.ellipse(M, [0, 0], { stroke: 'teal', fill: N.det(M) < 0 ? 'neg' : 'teal', alpha: 0.13, width: 2.2 });
    if (s.showRadii) radii(g, M);
    if (staged) {
      // the two special input directions ride along as gold diameters
      [p.v1, p.v2].forEach((v, i) => { const a = N.matVec(M, v); g.line(N.scale(a, -1), a, { color: 'gold', width: 2.6 }); g.label(['v₁', 'v₂'][i] + (s.stage > 0.02 ? ' (moved)' : ''), a, { color: 'gold', size: 12.5, dx: 10, dy: -10 }); });
      if (s.mode !== 'reverse') { const active = s.stage <= 0 ? -1 : s.stage < 1 ? 0 : s.stage < 2 ? 1 : s.stage < 3 ? 2 : 3; tiles(g, ['Vᵀ: rotate', 'Σ: stretch σ₁, σ₂', 'U: rotate'], active, W); }
      else { const active = s.stage >= 3 ? -1 : s.stage > 2 ? 0 : s.stage > 1 ? 1 : s.stage > 0 ? 2 : 3; tiles(g, ['Uᵀ: un-rotate', 'Σ⁻¹: 1/σ', 'V: un-rotate'], active, W); }
    } else if (s.showSU) {
      axesSU(g, At, { plain: s.panel === 'A' });   // step 1: the names σu come in step 3
      if (s.showV) {
        vsDashed(g, At);
        [[p.v1, p.u1, 0.75], [p.v2, p.u2, 0.55]].forEach(([v, u, r]) => g.arc([0, 0], v, u, { r, color: 'gold', width: 1 }));
      }
    }
    if (s.nullArrow && p.s2 < 0.05) {
      g.vector(N.scale(p.v2, 1.6), { color: 'mag', width: 2.6, label: 'v₂ ↦ 0', labelColor: 'mag' });
      g.lineThrough([0, 0], p.u1, { color: 'gold', width: 1, alpha: 0.4 });
    }
    if (s.showRank) g.hud('σ₁ = ' + LA.fmt(p.s1) + '   σ₂ = ' + LA.fmt(p.s2) + '   rank = ' + (p.s2 > 0.05 ? 2 : (p.s1 > 0.05 ? 1 : 0)), 14, 22, { color: p.s2 > 0.05 ? 'gold' : 'mag', size: 14, bold: true });
    if (s.showSlider) slider(g, s, { x: 0.04, y: 0.87, w: 0.92, h: 0.11 }, 'sigma2Scale', 'σ₂ scale', 0, 1);
    if (s.kappaInset > 0) {
      g.viewport({ x: 0.6, y: 0.72, w: 0.38, h: 0.26 }, { view: '2d', cam: { center: [0, 0], halfH: 1.1 }, bg: 'bg' }, (g2, r) => {
        const T = N.matMul(N.rot2(0.4), N.diag([1.4, 0.12]));
        g2.ellipse(T, [0, 0], { stroke: 'teal', fill: 'teal', alpha: 0.15, width: 2 });
        g2.hud('a thinner ellipse: κ = σ₁/σ₂ = 12 (this A: 1.89)', r.x + 6, r.y + 14, { size: 11.5, color: 'gold' });
        g2.hud('the inverse amplifies errors up to 12×', r.x + 6, r.y + r.h - 12, { size: 11, color: 'muted' });
      });
    }
    if (s.mode === 'play') colHandles(g, s);
  }

  function renderNames(g, s, ctx) { renderMain(g, s, ctx); }

  function renderCompare(g, s, ctx) {
    const top = { x: 0, y: 0, w: 1, h: 0.5 }, bot = { x: 0, y: 0.5, w: 1, h: 0.5 };
    g.viewport(top, { view: '2d', cam: { center: [0, -0.1], halfH: 1.75 } }, (g2, r) => {
      g2.ghostGrid(); g2.circle([0, 0], 1, { stroke: 'hud', dashed: true, width: 1.1, strokeAlpha: 0.5 });
      g2.ellipse(SHEAR, [0, 0], { stroke: 'teal', fill: 'teal', alpha: 0.12, width: 2 });
      g2.lineThrough([0, 0], [1, 0], { color: 'b', width: 1.6, dashed: true, alpha: 0.9 });
      g2.label('only eigenline (λ = 1, 1)', [-2.9, 0], { anchor: 'left', dy: -12, color: 'b', size: 12, font: 'ui' });
      vsDashed(g2, SHEAR); axesSU(g2, SHEAR);
      g2.hud('shear [[1,1],[0,1]]: two perpendicular pairs, σ = 1.62, 0.62', r.x + 10, r.y + 18, { color: 'text', size: 13 });
    });
    g.viewport(bot, { view: '2d', cam: { center: [0, 0], halfH: 1.9 } }, (g2, r) => {
      g2.ghostGrid(); g2.circle([0, 0], 1, { stroke: 'teal', width: 2 });
      const p = parts(ROT60);
      g2.vector(p.v1, { color: 'gold', width: 2, dashed: true, head: 9, label: 'v₁', labelColor: 'gold' });
      g2.vector(p.v2, { color: 'gold', width: 2, dashed: true, head: 9, label: 'v₂', labelColor: 'gold' });
      g2.vector(p.u1, { color: 'gold', width: 3, label: 'u₁', labelColor: 'gold' });
      g2.vector(p.u2, { color: 'gold', width: 3, label: 'u₂', labelColor: 'gold' });
      g2.arc([0, 0], p.v1, p.u1, { r: 0.55, color: 'gold', label: '60°' });
      g2.hud('rotation 60°: no real eigenvectors, yet σ₁ = σ₂ = 1', r.x + 10, r.y + 18, { color: 'text', size: 13 });
      g2.hud('the circle stays a circle; v’s turn into u’s', r.x + 10, r.y + 38, { color: 'gold', size: 12.5 });
    });
  }

  function renderMis(g, s, ctx) {
    const H = ctx.size.h;
    g.viewport({ x: 0, y: 0, w: 1, h: 0.66 }, { view: '2d', cam: { center: [0, 0], halfH: 2.7 }, border: false }, (g2) => {
      const M = lerpM(I2(), FLIP, s.morph);
      g2.ghostGrid(); g2.grid(M, { alpha: 0.5 });
      g2.circle([0, 0], 1, { stroke: 'hud', dashed: true, width: 1.1, strokeAlpha: 0.5 });
      g2.ellipse(M, [0, 0], { stroke: 'neg', fill: N.det(M) < 0 ? 'neg' : 'teal', alpha: 0.2, width: 2 });
      g2.vector(N.col(M, 0), { color: 'i', width: 2.6, label: 'i', hat: true });
      g2.vector(N.col(M, 1), { color: 'j', width: 2.6, label: 'j', hat: true });
      if (s.morph >= 1) g2.label('flipped (det < 0)', [1.6, -1.2], { color: 'neg', size: 12.5, font: 'ui' });
    });
    g.viewport({ x: 0, y: 0.66, w: 1, h: 0.34 }, { view: '2d', cam: { center: [0.3, -0.2], halfH: 0.9 } }, (g2, r) => {
      ticksLine(g2, [{ name: 'λ (eigen)', vals: [3, -1], color: 'white' }, { name: 'σ (singular)', vals: [3, 1], color: 'gold' }], 0.3);
      g2.hud('σ = λ ?', r.x + 18, r.y + 22, { color: 'bad', size: 15, bold: true });
      if (s.cross) g2.marker.cross([r.x + 100, r.y + 22], { screen: true, size: 24 });
      g2.hud('σ ≥ 0 always: the flip lives in U', r.x + r.w - 250, r.y + 22, { color: 'gold', size: 12.5 });
    });
  }

  function render3x2(g, s, ctx) {
    const sv = N.svd(A32), U = N.cols(sv.U), S = sv.S;
    const u3 = N.normalize(N.cross(U[0], U[1]));
    const E = [[1, 0, 0], [0, 1, 0], [0, 0, 0]], Af = [[1, 0, 0], [0, 1, 0], [1, 1, 0]];
    const M = lerpM(E, Af, s.morph);
    g.cage({ range: 2, walls: false, alpha: 0.4 });
    g.axes({ length: 2.4 });
    g.plane({ point: [0, 0, 0], span: [[1, 0, 1], [0, 1, 1]], extent: 2, color: 'teal', alpha: 0.18 * s.morph });
    g.circle([0, 0, 0], 1, { stroke: 'hud', dashed: true, width: 1.2 });
    g.ellipse(M, [0, 0, 0], { stroke: 'teal', width: 2.6 });
    if (s.morph > 0.95) {
      g.vector([0, 0, 0], N.scale(U[0], S[0]), { color: 'gold', width: 3, label: 'σ₁u₁', labelColor: 'gold' });
      g.vector([0, 0, 0], N.scale(U[1], S[1]), { color: 'gold', width: 3, label: 'σ₂u₂', labelColor: 'gold' });
      g.vector([0, 0, 0], N.scale(u3, 1.1), { color: 'muted', width: 1.6, dashed: true, label: 'u₃ unused', labelColor: 'muted' });
    }
    g.hud('dashed: the input circle (the 2D input room, on the floor)', 12, 20, { color: 'hud', size: 12.5 });
    if (s.morph > 0.5) g.hud('teal plane: the column space of A inside 3D (output room)', 12, 40, { color: 'teal', size: 12.5 });
  }

  function renderSheets(g, s, ctx) {
    const p = parts(A0);
    const sh1 = N.scaleM(N.outer(p.u1, p.v1), p.s1), sh2 = N.scaleM(N.outer(p.u2, p.v2), p.s2);
    const ramp = t => LA.color.diverge(2 * t - 1);
    const tile = (M, x0, y0, lbl, alpha) => {
      const b = { x0, x1: x0 + 1.1, y0, y1: y0 + 1.1 };
      g.image(M, b, { ramp, range: [-2.5, 2.5], cells: true, alpha });
      for (let i = 0; i < 2; i++) for (let j = 0; j < 2; j++) g.label(LA.fmt(M[i][j]), [x0 + 0.275 + 0.55 * j, y0 + 0.825 - 0.55 * i], { size: 11.5, color: 'text', bg: false, alpha });
      g.label(lbl, [x0 + 0.55, y0 + 1.1], { anchor: 'above', dy: -4, size: 13, color: 'gold', alpha });
    };
    const u = s.sum;
    tile(sh1, -2.85, 1.9, 'σ₁u₁v₁ᵀ', 1);
    g.label('+', [-1.38, 2.45], { size: 20, color: 'hud' });
    tile(sh2, -1.1, 1.9, 'σ₂u₂v₂ᵀ', 1);
    g.label('=', [0.38, 2.45], { size: 20, color: 'hud', alpha: u });
    tile(N.addM(sh1, sh2), 0.65, 1.9, 'A', u);
    g.label('rank 1 + rank 1 = rank 2', [1.9, 1.6], { size: 11.5, color: 'muted', font: 'ui', alpha: u });
    // the grid under the first sheet alone collapses onto the u₁ line
    g.viewport({ x: 0.03, y: 0.33, w: 0.45, h: 0.34 }, { view: '2d', cam: { center: [0, 0], halfH: 2.2 } }, (g2, r) => {
      const M = lerpM(I2(), sh1, s.collapse);
      g2.ghostGrid(); g2.grid(M, { alpha: 0.6 });
      g2.lineThrough([0, 0], p.u1, { color: 'gold', width: 1.4, alpha: 0.7 });
      g2.hud('grid under σ₁u₁v₁ᵀ alone', r.x + 6, r.y + 14, { size: 11.5, color: 'text' });
      g2.hud('everything lands on the u₁ line', r.x + 6, r.y + r.h - 12, { size: 11.5, color: 'gold' });
    });
    g.hud('keep only the biggest sheet:', ctx.size.w * 0.52, ctx.size.h * 0.42, { size: 13, color: 'text' });
    g.hud('the best rank-1 approximation', ctx.size.w * 0.52, ctx.size.h * 0.46, { size: 13, color: 'gold' });
    g.hud('next lesson: a 32×32 synthetic image', ctx.size.w * 0.52, ctx.size.h * 0.52, { size: 12, color: 'muted' });
    [1, 4, 16].forEach((k, i) => {
      const x0 = -3.0 + i * 2.1;
      g.image(recon(k), { x0, x1: x0 + 1.8, y0: -3.45, y1: -1.65 }, { range: [0, 1] });
      g.label('k = ' + k, [x0 + 0.9, -1.65], { anchor: 'above', dy: -4, size: 12.5, color: 'gold', font: 'ui' });
    });
  }

  function renderML(g, s, ctx) {
    const W = ctx.size.w, H = ctx.size.h, sp = s.sp128, n = Math.ceil(128 * s.mlT);
    g.bars(sp.map((x, i) => i < n ? x : 0), { screen: true, x: W * 0.06, y: H * 0.1, w: W * 0.88, h: H * 0.3, gap: 0.5, color: 'gold', max: sp[0], title: '128 singular values of a synthetic 784×128 weight matrix (planted spectrum)' });
    g.hud('σ₁ … σ₁₂₈, decaying fast — illustrative: planted, not taken from a trained network', W * 0.06, H * 0.44, { color: 'gold', size: 12.5 });
    g.hud(HONEST, W * 0.06, H * 0.5, { color: 'text', size: 15 });
    const boxes = [{ x: 0.03, y: 0.56, w: 0.3, h: 0.3 }, { x: 0.35, y: 0.56, w: 0.3, h: 0.3 }, { x: 0.67, y: 0.56, w: 0.3, h: 0.3 }];
    g.viewport(boxes[0], { view: '2d', cam: { center: [0, 0], halfH: 1.6 } }, (g2) => {
      const rand = LA.rng(71); const pts = [];
      for (let i = 0; i < 40; i++) { const a = LA.rng.gauss(rand) * 1.1, b = LA.rng.gauss(rand) * 0.3; pts.push([0.8 * a - 0.6 * b, 0.6 * a + 0.8 * b]); }
      g2.points(pts, { r: 2.5, color: 'data' });
      g2.vector([0, 0], [1.2, 0.9], { color: 'gold', width: 2.4, head: 8 });
    });
    g.viewport(boxes[1], { view: '2d', cam: { center: [0, 0], halfH: 1.1 } }, (g2) => { g2.image(recon(8), { x0: -1, x1: 1, y0: -1, y1: 1 }, { range: [0, 1] }); });
    g.viewport(boxes[2], { view: '2d', cam: { center: [0, 0], halfH: 1.2 } }, (g2) => {
      const box = (x0, y0, w, h, c, l) => { g2.polygon([[x0, y0], [x0 + w, y0], [x0 + w, y0 + h], [x0, y0 + h]], { fill: c, alpha: 0.35, stroke: c }); g2.label(l, [x0 + w / 2, y0 + h / 2], { size: 12, color: 'text', bg: false }); };
      box(-0.95, -0.6, 0.9, 1.2, 'grid', 'W');
      g2.label('+', [0.1, 0], { size: 16, color: 'hud' });
      box(0.25, -0.6, 0.14, 1.2, 'gold', 'B'); box(0.45, 0.46, 0.5, 0.14, 'gold', 'A');
    });
    ['PCA = SVD of centered data', 'compression: keep big σ', 'LoRA: W + low-rank BA'].forEach((c, i) => g.hud(c, (boxes[i].x + 0.005) * W, (boxes[i].y + boxes[i].h + 0.03) * H, { size: 12, color: 'text' }));
  }

  const ROWS = [{ x: 0.03, y: 0.03, w: 0.55, h: 0.3 }, { x: 0.03, y: 0.35, w: 0.55, h: 0.3 }, { x: 0.03, y: 0.67, w: 0.55, h: 0.3 }];
  function renderRecap(g, s, ctx) {
    const W = ctx.size.w, H = ctx.size.h;
    g.viewport(ROWS[0], { view: '2d', cam: { center: [0, 0.1], halfH: 2.3 } }, (g2) => {
      g2.circle([0, 0], 1, { stroke: 'hud', dashed: true, width: 1.1, strokeAlpha: 0.5 });
      g2.ellipse(A0, [0, 0], { stroke: 'teal', fill: 'teal', alpha: 0.12, width: 2 });
      vsDashed(g2, A0); axesSU(g2, A0);
    });
    g.viewport(ROWS[1], { view: '2d', cam: { center: [0, 0], halfH: 1.3 } }, (g2) => {
      [[-1.5, 1], [0, 2], [1.5, 3]].forEach(([x, st], i) => {
        const M = N.scaleM(stageM(A0, st), 0.27);
        g2.ellipse(M, [x, 0.1], { stroke: 'teal', fill: 'teal', alpha: 0.12, width: 1.6 });
        const p = parts(A0); const a = N.matVec(M, p.v1);
        g2.line([x - a[0], 0.1 - a[1]], [x + a[0], 0.1 + a[1]], { color: 'gold', width: 2 });
        g2.label(['Vᵀ', 'Σ', 'U'][i], [x, -0.95], { color: 'gold', size: 14 });
      });
      g2.label('→', [-0.75, 0.1], { color: 'hud', size: 18 }); g2.label('→', [0.75, 0.1], { color: 'hud', size: 18 });
    });
    g.viewport(ROWS[2], { view: '2d', cam: { center: [0, 0], halfH: 1.3 } }, (g2) => {
      const p = parts(A0); const a = N.scale(p.u1, p.s1 * 0.45), c = [-0.7, 0];
      g2.line(N.sub(c, a), N.add(c, a), { color: 'gold', width: 4 });
      g2.label('rank 1: a flat ellipse', [-0.7, 0.85], { color: 'mag', size: 12.5, font: 'ui' });
      g2.bars([2.38, 0], { origin: [0.85, -0.9], width: 0.3, gap: 0.15, height: 1.6, max: 2.4, colors: ['gold', 'muted'], labels: ['σ₁', 'σ₂ = 0'] });
    });
    const caps = [['Rotate–stretch–rotate', 'v’s (dashed) → σu’s (solid)'], ['A = U Σ Vᵀ', 'σ ≥ 0, not eigenvalues'], ['Rank', '= number of nonzero σ']];
    ROWS.forEach((b, i) => { const y = (b.y + b.h / 2) * H, x = (b.x + b.w + 0.03) * W; g.hud(caps[i][0], x, y - 10, { color: 'text', size: 14, bold: true }); g.hud(caps[i][1], x, y + 12, { color: 'muted', size: 12.5 }); });
  }

  function renderEntry(g, s, ctx) {
    const S = s.Sigma;
    g.ghostGrid();
    g.circle([0, 0], 1, { stroke: 'hud', dashed: true, width: 1.1, strokeAlpha: 0.5 });
    g.ellipse(FLIP, [0, 0], { stroke: 'neg', fill: 'neg', alpha: 0.15, width: 2 });
    g.ellipse(S, [0, 0], { stroke: 'gold', dashed: true, width: 2.2 });
    g.label('violet: the ellipse of A = [[3,0],[0,−1]]', [-2.9, 2.6], { anchor: 'left', color: 'neg', size: 12.5, font: 'ui' });
    g.label('dashed gold: the stretch Σ you typed', [-2.9, 2.25], { anchor: 'left', color: 'gold', size: 12.5, font: 'ui' });
    const neg = S[0][0] < 0 || S[1][1] < 0;
    if (neg) g.label('a negative σ? singular values are never negative', [-2.9, -2.6], { anchor: 'left', color: 'bad', size: 12.5, font: 'ui' });
  }

  LA.registerLesson({
    id: 'p4-07-svd',
    part: 4,
    title: 'SVD: rotate, stretch, rotate',
    subtitle: 'Every matrix, any shape, is three simple moves: Vᵀ, Σ, U.',
    minutes: 11,
    prereqs: ['p4-06-symmetric-quadratic-forms', 'p3-04-orthonormal-gram-schmidt', 'p2-09-non-square'],
    tags: ['core'],
    badge: { drawnIn: '2D (+ one 3D step)', realSize: 'a 784×128 weight matrix has 128 singular values; a 4096×4096 attention projection has 4096', transfers: 'every matrix of any shape is UΣVᵀ; the circle→ellipse picture is exact for each pair (vᵢ, uᵢ)' },
    cantShow: ['more than two or three singular directions as arrows (we show a bar strip for 128)', 'the ellipsoid image of a 4×4 or larger matrix', 'why the SVD always exists (we lean on p4-06: AᵀA is symmetric PSD)'],
    terms: [
      { term: 'singular value decomposition', def: 'A = UΣVᵀ: an orthogonal Vᵀ (rotate), a diagonal stretch Σ, an orthogonal U (rotate).' },
      { term: 'singular values', def: 'The stretch factors σ₁ ≥ σ₂ ≥ … ≥ 0: the lengths of the ellipse’s semi-axes.' },
      { term: 'right singular vectors', def: 'The input directions v that stay perpendicular after the map.' },
      { term: 'left singular vectors', def: 'The unit directions u where the v’s land: Av = σu.' },
      { term: 'rank', def: 'The number of nonzero singular values: how many independent directions survive.' },
    ],
    atlas: ['svd', 'pca', 'regularization'],
    thumbnail: 2,

    state() {
      return {
        view: '2d', cam: Object.assign(LA.cam.defaults2d(), { halfH: 3.1 }),
        scene: 'main', panel: 'A', mode: 'morph',
        A: LA.clone(A0), morph: 1, stage: 3, sigma2Scale: 1, showRadii: false, showSU: true, showV: false, showRank: false, showSlider: false, nullArrow: false, kappaInset: 0,
        cross: false, sum: 1, collapse: 1, mlT: 1, sp128: spectrum128(), Sigma: [[1, 0], [0, 1]], v: [0.4, -0.9],
      };
    },

    render(g, s, ctx) {
      switch (s.scene) {
        case 'compare': return renderCompare(g, s, ctx);
        case 'mis': return renderMis(g, s, ctx);
        case '3x2': return render3x2(g, s, ctx);
        case 'sheets': return renderSheets(g, s, ctx);
        case 'ml': return renderML(g, s, ctx);
        case 'recap': return renderRecap(g, s, ctx);
        case 'entry': return renderEntry(g, s, ctx);
        default:
          renderMain(g, s, ctx);
          if (s.mode === 'play' && ctx.mode === 'play') {
            const Av = N.matVec(s.A, s.v);
            g.vector(s.v, { color: 'v', width: 1.6, dashed: true, label: 'v', head: 8 });
            g.vector(Av, { color: 'v', width: 2.6, label: 'Av' });
            g.handle('v', s.v, (q, st) => { st.v = [r2(q[0]), r2(q[1])]; }, { color: 'v' });
          }
      }
    },

    math(s, tex) {
      const T = (x, o) => tex.text(x, o);
      const A = s.scene === 'entry' ? FLIP : (s.sigma2Scale < 1 ? scaled(s.A, s.sigma2Scale) : s.A), p = parts(A);
      const Sg = [[p.s1, 0], [0, p.s2]];
      const usvt = () => tex.eq(tex.sym('A'), tex.op('='), tex.mat(p.U, { colors: 'gold' }), tex.mat(Sg), tex.mat(N.transpose(p.V), { colors: 'gold' }));
      switch (s.panel) {
        case 'A': return tex.block(tex.eq(tex.sym('A'), tex.op('='), tex.mat(A, { colors: 'cols' })));
        case 'moves': {
          const st = s.stage, lit = k => st > k && st <= k + 1.0001;
          return tex.block(tex.eq(tex.sym('A'), tex.op('='), tex.sym('U', { color: lit(2) ? 'gold' : undefined }), tex.op('·'), tex.sym('Σ', { color: lit(1) ? 'gold' : undefined }), tex.op('·'), tex.sym('V', { sup: 'T', color: lit(0) ? 'gold' : undefined })),
            tex.eq(T('Vᵀ: turn ' + LA.fmt.deg(-p.thetaV) + ' · Σ: σ = ' + LA.fmt(p.s1) + ', ' + LA.fmt(p.s2) + ' · U: turn ' + LA.fmt.deg(p.thetaU))));
        }
        case 'names': return tex.block(usvt(), tex.eq(tex.sym('A'), tex.sym('v', { sub: 'i', color: 'gold' }), tex.op('='), tex.sym('σ', { sub: 'i' }), tex.sym('u', { sub: 'i', color: 'gold' })), tex.eq(T('σ₁ = ' + LA.fmt(p.s1) + ', σ₂ = ' + LA.fmt(p.s2), { color: 'gold' })));
        case 'compare': return tex.block(tex.eq(T('shear: λ = 1, 1; σ = 1.62, 0.62')), tex.eq(T('rotation: no real λ; σ = 1, 1', { color: 'gold' })));
        case 'mis': return tex.block(tex.eq(T('σᵢ = √(eigenvalues of AᵀA)')), tex.eq(T('σ ≥ 0', { color: 'gold' })), tex.eq(T('[[3,0],[0,−1]]: λ = 3, −1 but σ = 3, 1')));
        case 'rank': return tex.block(usvt(), tex.eq(T('rank = #{σᵢ > 0} = ' + (p.s2 > 0.05 ? 2 : 1), { color: p.s2 > 0.05 ? 'gold' : 'mag' })));
        case '3x2': return tex.block(tex.eq(tex.sym('A'), T(' (3×2) = U (3×3) · Σ (3×2) · Vᵀ (2×2)')), tex.eq(tex.sym('A'), tex.op('='), tex.mat(A32, { colors: 'cols' })), tex.eq(T('σ = 1.73, 1', { color: 'gold' })));
        case 'sheets': return tex.block(tex.eq(tex.sym('A'), tex.op('='), tex.sum(tex.sym('σ', { sub: 'i' }) + tex.sym('u', { sub: 'i', color: 'gold' }) + tex.sym('v', { sub: 'i', sup: 'T', color: 'gold' }), 'i')));
        case 'inverse': return tex.block(tex.eq(tex.sym('A', { sup: '+' }), tex.op('='), tex.sym('V'), tex.sym('Σ', { sup: '+' }), tex.sym('U', { sup: 'T' })), tex.eq(tex.sym('κ'), tex.op('='), T('σ₁/σ₂ = ' + LA.fmt(p.s1 / p.s2))));
        case 'ml': return tex.block(tex.eq(tex.sym('W'), tex.op('='), tex.sym('U'), tex.sym('Σ'), tex.sym('V', { sup: 'T' }), T(', 128 σ’s')));
        case 'entry': return tex.block(tex.eq(tex.sym('A'), tex.op('='), tex.mat(FLIP, { colors: 'cols' })), tex.eq(tex.sym('Σ'), tex.op('='), tex.mat(s.Sigma, { colors: 'gold' })));
        case 'check': return tex.block(tex.eq(tex.sym('A'), tex.op('='), tex.mat(A, { colors: 'cols' })), tex.eq(T('σ₁ = ' + LA.fmt(p.s1) + ', σ₂ = ' + LA.fmt(p.s2), { color: 'gold' })));
        case 'play': return tex.block(usvt(), tex.eq(T('σ₁ = ' + LA.fmt(p.s1) + ', σ₂ = ' + LA.fmt(p.s2) + ', rank ' + (p.s2 > 0.05 ? 2 : p.s1 > 0.05 ? 1 : 0) + ', κ = ' + (p.s2 > 1e-6 ? LA.fmt(p.s1 / p.s2) : '∞'), { color: 'gold' })), tex.eq(tex.sym('A'), tex.sym('v', { color: 'v' }), tex.op('='), tex.vec(N.matVec(s.A, s.v), { color: 'v' })));
        default: return tex.block(tex.eq(tex.sym('A'), tex.op('='), tex.sym('U'), tex.sym('Σ'), tex.sym('V', { sup: 'T' })));
      }
    },

    steps: [
      { title: 'Every matrix makes an ellipse', duration: 3.5,
        narration: 'Any 2×2 matrix, rotation, shear or squash, sends the unit circle to an ellipse, possibly a flat one. A = [[2,1],[0,1.5]]: not symmetric, not a rotation, yet the circle becomes a tidy tilted teal ellipse with two perpendicular gold axes. Where did perpendicular axes come from?',
        setup(s) { s.scene = 'main'; s.view = '2d'; s.cam = Object.assign(LA.cam.defaults2d(), { halfH: 3.3 }); s.A = LA.clone(A0); s.mode = 'morph'; s.morph = 0; s.showRadii = true; s.showSU = false; s.showV = false; s.sigma2Scale = 1; s.showRank = false; s.showSlider = false; s.nullArrow = false; s.kappaInset = 0; s.panel = 'A'; },
        at(s, t) { s.morph = seg(t, 0.05, 0.8); s.showSU = t > 0.82; } },

      { title: 'Three moves: rotate, stretch, rotate', duration: 5,
        narration: 'Here is the secret: any matrix can be done in three simple moves. First <i>rotate</i> the plane (Vᵀ) so two special input directions, the gold diameters, line up with the axes; then <i>stretch</i> along the axes (Σ) by σ₁ and σ₂; then <i>rotate</i> again (U) into the final position. No shear is ever needed.',
        setup(s) { s.mode = 'stage'; s.stage = 0; s.showRadii = false; s.showSU = false; s.panel = 'moves'; },
        at(s, t) { s.stage = 3 * seg(t, 0.04, 0.96, 'linear'); } },

      { title: 'The names on the picture', duration: 4,
        narration: 'The two special input directions are the <b>right singular vectors</b> ' + gold('v₁') + ', ' + gold('v₂') + ' (dashed): perpendicular on the circle, still perpendicular after the map. The stretch factors σ₁ ≥ σ₂ are the <b>singular values</b>. Where they land, σ₁' + gold('u₁') + ' and σ₂' + gold('u₂') + ', are the ellipse’s axes: <b>left singular vectors</b>. A = UΣVᵀ is the <b>singular value decomposition</b>.',
        setup(s) { s.mode = 'final'; s.showSU = true; s.showV = true; s.panel = 'names'; },
        at(s, t) { } },

      { title: 'Not eigenvectors: pairs, always perpendicular, always there', duration: 4,
        narration: 'Eigenvectors keep their <i>own</i> direction; singular vectors come in input/output pairs that may differ. The shear [[1,1],[0,1]] has one eigenline but a full set of two perpendicular singular pairs. Even a rotation, with no real eigenvectors at all, has an SVD: σ₁ = σ₂ = 1, the circle stays a circle.',
        setup(s) { s.scene = 'compare'; s.panel = 'compare'; },
        at(s, t) { } },

      { title: 'Misconception: singular values are eigenvalues', duration: 3, misconception: true,
        narration: '✕ “Singular values are just eigenvalues.” For the shear the eigenvalues are 1 and 1 but the singular values are 1.62 and 0.62. For [[3,0],[0,−1]] the eigenvalue −1 becomes σ = 1: singular values are never negative, a flip is absorbed into U. They agree only for symmetric positive semi-definite matrices.',
        setup(s) { s.scene = 'mis'; s.morph = 0; s.cross = false; s.panel = 'mis'; },
        at(s, t) { s.morph = seg(t, 0.05, 0.5); s.cross = t > 0.6; } },

      { title: 'σ₂ → 0: the ellipse goes flat, the rank drops', duration: 3.5, interactive: true,
        narration: 'Shrink σ₂ to zero. The ellipse flattens into a segment along u₁: every input lands on one line, rank 1. The <b>rank</b> is the number of nonzero singular values, and a tiny σ means “almost rank-deficient”: a direction, the magenta v₂, that the matrix nearly ignores. Drag the slider.',
        setup(s) { s.scene = 'main'; s.A = LA.clone(A0); s.mode = 'final'; s.showSU = true; s.showV = false; s.sigma2Scale = 1; s.showRank = true; s.showSlider = true; s.nullArrow = true; s.panel = 'rank'; },
        at(s, t) { s.sigma2Scale = 1 - seg(t, 0.1, 0.85); } },

      { title: 'Non-square: same three moves, two different rooms', duration: 5,
        narration: 'A 3×2 matrix takes the plane into space. Its SVD still rotates the input circle (Vᵀ in 2D), stretches by σ₁, σ₂, and then U, now a rotation of 3D, places the ellipse on a tilted teal plane. The u’s live in the output room, the v’s in the input room. A 2×3 does the reverse.',
        setup(s) { s.scene = '3x2'; s.view = '3d'; s.cam = Object.assign(LA.cam.defaults3d(), { dist: 9, yaw: 0.55, pitch: 0.42, target: [0, 0, 0.3] }); s.morph = 0; s.showRank = false; s.showSlider = false; s.nullArrow = false; s.sigma2Scale = 1; s.panel = '3x2'; },
        at(s, t) { s.morph = seg(t, 0.1, 0.7); LA.cam.tween(s, { yaw: 0.55 }, { yaw: 0.95 }, seg(t, 0, 1, 'linear')); } },

      { title: 'Preview: a sum of rank-1 sheets', duration: 3,
        narration: 'Read A = UΣVᵀ column by column and it becomes a sum: A = σ₁' + gold('u₁v₁ᵀ') + ' + σ₂' + gold('u₂v₂ᵀ') + ', two rank-1 pieces, the big one first. Keep only the first and you have the best rank-1 approximation. The next lesson compresses an image with exactly this.',
        setup(s) { s.scene = 'sheets'; s.view = '2d'; s.cam = Object.assign(LA.cam.defaults2d(), { halfH: 3.6 }); s.sum = 0; s.collapse = 0; s.panel = 'sheets'; },
        at(s, t) { s.sum = seg(t, 0.15, 0.45); s.collapse = seg(t, 0.4, 0.9); } },

      { title: 'Preview: undo what can be undone', duration: 3,
        narration: 'To invert A, run the three moves backwards: U⁻¹ = Uᵀ, stretch by 1/σᵢ, then V. A zero σ cannot be inverted, so skip it: that is the pseudo-inverse A⁺ = VΣ⁺Uᵀ. And σ₁/σ₂, how squashed the ellipse is, measures how much the inverse amplifies errors. Both in p4-12.',
        setup(s) { s.scene = 'main'; s.view = '2d'; s.cam = Object.assign(LA.cam.defaults2d(), { halfH: 3.1 }); s.A = LA.clone(A0); s.mode = 'reverse'; s.stage = 3; s.kappaInset = 0; s.panel = 'inverse'; },
        at(s, t) { s.stage = 3 - 3 * seg(t, 0.03, 0.8, 'linear'); s.kappaInset = seg(t, 0.75, 0.9); } },

      { title: 'Why ML cares: 128 bars', duration: 3.5, mlTie: true,
        narration: 'PCA is the SVD of the centered data matrix. Compression, recommenders, LoRA, whitening, the condition number: all read off U, Σ, V. A 784×128 weight matrix cannot be drawn, but its 128 singular values can: a strip of bars, here a planted synthetic spectrum that decays fast, as trained weights often do. The picture stops at 3. The formula does not.',
        setup(s) { s.scene = 'ml'; s.kappaInset = 0; s.mlT = 0; s.panel = 'ml'; },
        at(s, t) { s.mlT = seg(t, 0.05, 0.7, 'linear'); } },

      { title: 'Recap', duration: 2.5, recap: true,
        narration: 'Rotate (Vᵀ), stretch (Σ), rotate (U): every matrix, any shape. The v’s are perpendicular inputs, the σu’s are the ellipse axes, σ ≥ 0 and not eigenvalues. Rank = nonzero σ’s. A is a sum of rank-1 sheets; inverting means 1/σ for the σ’s you can.',
        cards: ['<b>Rotate–stretch–rotate.</b> A = UΣVᵀ for every matrix, square or not.', '<b>σ ≥ 0, not λ.</b> Singular pairs Avᵢ = σᵢuᵢ always exist and stay perpendicular.', '<b>Rank = nonzero σ’s.</b> A = Σ σᵢuᵢvᵢᵀ; a tiny σ is a nearly ignored direction.'],
        setup(s) { s.scene = 'recap'; s.panel = 'recap'; },
        at(s, t) { } },
    ],

    playground: {
      intro: '<p>Drag the landing spots of <span class="c-i">î</span> and <span class="c-j">ĵ</span> (the columns of A) or the yellow v. Scrub the three moves, flatten σ₂, or switch to the 3×2 matrix in 3D.</p>',
      setup(s) { s.scene = 'main'; s.view = '2d'; s.cam = Object.assign(LA.cam.defaults2d(), { halfH: 3.1 }); s.A = LA.clone(A0); s.mode = 'play'; s.stage = 3; s.sigma2Scale = 1; s.showSU = true; s.showV = true; s.panel = 'play'; },
      controls: [
        { type: 'matrix', bind: 'A', label: 'A', colors: 'cols', step: 0.1 },
        { type: 'slider', bind: 'stage', min: 0, max: 3, step: 0.01, label: 'three moves (0 → Vᵀ → Σ → U)', color: 'gold' },
        { type: 'slider', bind: 'sigma2Scale', min: 0, max: 1, step: 0.01, label: 'σ₂ scale', color: 'mag' },
        { type: 'preset', label: 'Presets', items: [
          { label: 'Default', values: { A: A0 } },
          { label: 'Shear', values: { A: SHEAR } },
          { label: 'Rotation', values: { A: ROT60 } },
          { label: 'Symmetric', values: { A: [[2, 1], [1, 2]] } },
          { label: 'Flip [[3,0],[0,−1]]', values: { A: FLIP } },
        ] },
        { type: 'preset', label: 'View', items: [
          { label: '2×2 in 2D', apply(s) { s.scene = 'main'; s.view = '2d'; s.cam = Object.assign(LA.cam.defaults2d(), { halfH: 3.1 }); s.panel = 'play'; } },
          { label: '3×2 in 3D', apply(s) { s.scene = '3x2'; s.view = '3d'; s.cam = Object.assign(LA.cam.defaults3d(), { dist: 9, yaw: 0.8, pitch: 0.42, target: [0, 0, 0.3] }); s.morph = 1; s.panel = '3x2'; } },
        ] },
      ],
    },

    checks: [
      { kind: 'manipulate',
        prompt: 'Drag the landing spots of î and ĵ so that σ₂ = 0.',
        setup(s) { s.scene = 'main'; s.view = '2d'; s.cam = Object.assign(LA.cam.defaults2d(), { halfH: 3.1 }); s.A = LA.clone(A0); s.mode = 'play'; s.v = [0, 0]; s.showSU = true; s.showV = false; s.showRank = true; s.panel = 'check'; },
        judge(s) { const p = parts(s.A); return { ok: p.s2 < 0.05 && p.s1 > 0.5, progress: Math.max(0, 1 - p.s2 / 1.3), hint: p.s1 <= 0.5 ? 'Keep the arrows long; only line them up.' : undefined }; },
        hint: 'Put ĵ’s landing spot on the line through î’s.',
        solution(s) { s.A = [[2, 1], [1, 0.5]]; } },
      { kind: 'manipulate',
        prompt: 'Drag the landing spots of î and ĵ to make a matrix whose singular values are both 2 but which is not 2·I.',
        setup(s) { s.scene = 'main'; s.view = '2d'; s.cam = Object.assign(LA.cam.defaults2d(), { halfH: 3.1 }); s.A = I2(); s.mode = 'play'; s.v = [0, 0]; s.showSU = true; s.showV = false; s.showRank = true; s.panel = 'check'; },
        judge(s) { const p = parts(s.A); const d = Math.abs(p.s1 - 2) + Math.abs(p.s2 - 2); const far = N.frobenius(N.subM(s.A, [[2, 0], [0, 2]])); return { ok: Math.abs(p.s1 - 2) < 0.1 && Math.abs(p.s2 - 2) < 0.1 && far > 0.5, progress: Math.max(0, 1 - d / 2) * (far > 0.5 ? 1 : 0.6), hint: far <= 0.5 && d < 0.3 ? 'That is 2·I. Turn or flip the picture as well.' : undefined }; },
        hint: 'A rotation or reflection keeps the circle round; then scale it by 2.',
        solution(s) { s.A = N.scaleM(N.rot2(Math.PI / 3), 2); } },
      { kind: 'entry',
        prompt: 'Type Σ for A = [[3, 0], [0, −1]].',
        bind: 'Sigma', answer: [[3, 0], [0, 1]], tol: LA.TOL.entry, label: 'Σ', colors: 'gold',
        hint: 'Singular values are never negative: the flip is in U.',
        setup(s) { s.scene = 'entry'; s.view = '2d'; s.cam = Object.assign(LA.cam.defaults2d(), { halfH: 3.1 }); s.Sigma = [[1, 0], [0, 1]]; s.panel = 'entry'; } },
    ],
  });
})();
