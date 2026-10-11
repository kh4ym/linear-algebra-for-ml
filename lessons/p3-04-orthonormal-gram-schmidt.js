/* p3-04 — Orthonormal bases, orthogonal matrices, Gram–Schmidt, QR. */
(function () {
  'use strict';
  const N = LA.num, lerpV = LA.lerpV, lerpM = LA.lerpM, seg = LA.seg, fmt = LA.fmt;
  const I2 = () => [[1, 0], [0, 1]];
  const DEG = Math.PI / 180, TH = 30 * DEG;
  const A_ = [3, 1], W_ = [1, 2], V_ = [2, 1];                     // slanted pair (shared with p3-03) and test vector
  const SHEAR = [[1, 1], [0, 1]], FLIP = [[0, 1], [1, 0]];
  const V3 = [[1, 1, 0], [1, 0, 1], [0, 1, 1]];
  const CAM2 = () => ({ center: [1.0, 0.8], halfH: 2.7 });
  const CAM3 = () => Object.assign(LA.cam.defaults3d(), { target: [0.2, 0.4, 0.5], dist: 7.2, yaw: 0.85, pitch: 0.85 });
  const q1s = '<span class="c-i">q₁</span>', q2s = '<span class="c-j">q₂</span>', q3s = '<span class="c-k">q₃</span>';
  const av = '<span class="c-v">a</span>', wv = '<span class="c-w">w</span>';

  /* ---------- helpers ---------- */
  /* Engine workaround: LA.tex.reconcile maps tex.num spans onto matrix cells when only numbers change; emit live
     numbers as plain colored spans so any change rebuilds the panel cleanly. */
  function safeTex(tex) { return Object.assign({}, tex, { num: (x, o) => { o = o || {}; return '<span class="tx-n ' + tex.colorClass(o.color) + '">' + (typeof x === 'number' ? fmt(x, o.digits) : x) + '</span>'; } }); }
  const GS2 = N.gramSchmidt([A_, W_]);                            // q₁ = [0.95; 0.32], q₂ = [−0.32; 0.95]
  const Q2 = N.fromCols(GS2.basis), R2 = N.matMul(N.transpose(Q2), N.fromCols([A_, W_]));
  const GS3 = N.gramSchmidt(V3);
  /** stage list for the playground: 'n0', then per later vector 's'(hadows), 'l'(ift), 'n'(ormalize) */
  function stageList(n) { const L = ['n0']; for (let k = 1; k < n; k++) L.push('s' + k, 'l' + k, 'n' + k); return L; }
  function tile(g, c, M, name, o) {
    o = o || {}; const a = o.alpha === undefined ? 1 : o.alpha; if (a < 0.02) return;
    const rows = M.length, cols = M[0].length, cw = o.cw || 0.5, ch = o.ch || 0.36, w = cols * cw / 2 + 0.12, h = rows * ch / 2 + 0.1, col = o.color || 'hud';
    g.polygon([[c[0] - w, c[1] - h], [c[0] + w, c[1] - h], [c[0] + w, c[1] + h], [c[0] - w, c[1] + h]], { fill: 'panel2', alpha: 0.95 * a, stroke: col, width: 1.4 });
    for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) { const v = M[i][j]; g.label(typeof v === 'number' ? fmt(v) : v, [c[0] + (j - (cols - 1) / 2) * cw, c[1] - (i - (rows - 1) / 2) * ch], { color: o.cellColor ? (typeof o.cellColor === 'function' ? o.cellColor(i, j) : o.cellColor) : 'text', size: o.size || 13.5, font: 'math', alpha: a, bg: false }); }
    if (name) g.label(name, [c[0], c[1] + h], { anchor: 'above', dy: -6, color: col, size: 14, font: 'math', alpha: a, bg: false });
  }
  function unitCircle(g, o) { g.circle([0, 0], 1, Object.assign({ stroke: 'hud', width: 1.4, strokeAlpha: 0.7 }, o || {})); }

  /* ---------- 2D main scene ---------- */
  function draw2d(g, s, ctx) {
    g.ghostGrid();
    const R = N.rot2(s.theta), q1 = N.col(R, 0), q2 = N.col(R, 1);
    if (s.showRotGrid) g.grid(R, { alpha: 0.8 });
    g.axes({ ticks: true });
    if (s.circle) unitCircle(g);
    if (s.scene === 'rot') {
      const glow = s.glow || [1, 1];
      g.vector(q1, { color: 'i', label: 'q₁', width: 2.6 + 2.4 * (glow[0] > 1 ? 1 : 0), dx: 8, dy: 16 });
      g.vector(q2, { color: 'j', label: 'q₂', width: 2.6 + 2.4 * (glow[1] > 1 ? 1 : 0) });
      if (s.rightMark) g.arc([0, 0], q1, q2, { r: 0.28, color: 'hud', right: true });
      if (s.coords > 0) {
        const v = s.v, c1 = N.dot(v, q1), c2 = N.dot(v, q2), f1 = N.scale(q1, c1), f2 = N.scale(q2, c2), u = s.coords;
        g.lineThrough([0, 0], q1, { len: 6, color: 'i', width: 1, alpha: 0.45 }); g.lineThrough([0, 0], q2, { len: 6, color: 'j', width: 1, alpha: 0.45 });
        g.dropLine(v, lerpV(v, f1, seg(u, 0, 0.45)), { r: 4 }); g.dropLine(v, lerpV(v, f2, seg(u, 0.2, 0.65)), { r: 4 });
        if (u > 0.5) { g.label('v·q₁ = ' + fmt(c1), N.add(f1, [0.4, -0.06]), { color: 'i', size: 13, anchor: 'left', dx: 6 }); g.label('v·q₂ = ' + fmt(c2), f2, { color: 'j', size: 13, anchor: 'right', dx: -12, dy: -12 }); }
        if (u > 0.7) { g.vector([0, 0], f1, { color: 'i', width: 2, alpha: 0.55, head: 9 }); g.vector(f1, N.add(f1, f2), { color: 'j', width: 2, alpha: 0.55, head: 9 }); }
        if (u > 0.85) g.hud('coordinates in Q:  [' + fmt(c1) + '; ' + fmt(c2) + ']', 14, 30, { color: 'gold', size: 14 });
      }
      if (s.showV) g.vector(s.v, { color: 'v', label: 'v', width: 3 });
      if (s.fillCell !== null && s.fillCell !== undefined && s.fillCell >= 0) {
        const pairs = [['q₁·q₁', 1], ['q₁·q₂', 0], ['q₂·q₁', 0], ['q₂·q₂', 1]];
        if (s.fillCell < 4) { const pc = pairs[s.fillCell]; g.hud(pc[0] + ' = ' + pc[1], 14, 30, { color: 'gold', size: 15 }); }
        else g.hud(s.theta < 0.01 ? 'Qᵀ undid Q: back on the ghost grid' : 'now apply Qᵀ …', 14, 30, { color: 'gold', size: 15 });
      }
    }
    if (s.scene === 'mis') drawMis(g, s);
    if (s.scene === 'gs') drawGS2(g, s, ctx);
    if (s.scene === 'chk') drawChk(g, s, ctx);
  }

  function drawMis(g, s) {
    const a = A_, w = W_, v = V_, u = s.misU, r = s.fixU;
    g.vector(a, { color: 'v', label: 'a', width: 2.6, alpha: r > 0 ? 0.35 : 1 });
    g.vector(w, { color: 'w', label: 'w', width: 2.6, alpha: r > 0 ? 0.35 : 1 });
    g.vector(v, { color: 'hud', label: 'v', width: 3 });
    const pa = N.proj(v, a), pw = N.proj(v, w);
    if (u > 0) {
      const al = r > 0 ? 0.45 : 1;
      g.dropLine(v, lerpV(v, pa, seg(u, 0, 0.35)), { r: 4 }); g.dropLine(v, lerpV(v, pw, seg(u, 0.15, 0.5)), { r: 4 });
      if (u > 0.55) {
        g.vector([0, 0], pa, { color: 'mag', width: 2.2, head: 9, alpha: al });
        const e = lerpV(pa, N.add(pa, pw), seg(u, 0.55, 0.85));
        g.vector(pa, e, { color: 'mag', width: 2.2, head: 9, alpha: al });
        if (u > 0.9) { g.marker.cross(N.add(pa, pw), { dx: 18, dy: -16, size: 22 }); g.label('shadows add to [' + fmt(pa[0] + pw[0]) + '; ' + fmt(pa[1] + pw[1]) + '], not v', N.add(pa, pw), { color: 'bad', size: 13, anchor: 'right', dx: -10, dy: -22 }); }
      }
    }
    if (r > 0) {
      const R = N.rot2(TH), q1 = N.col(R, 0), q2 = N.col(R, 1), f1 = N.scale(q1, N.dot(v, q1)), f2 = N.scale(q2, N.dot(v, q2));
      g.vector(q1, { color: 'i', label: 'q₁', width: 2.6, alpha: r }); g.vector(q2, { color: 'j', label: 'q₂', width: 2.6, alpha: r });
      g.vector([0, 0], f1, { color: 'i', width: 2, alpha: 0.6 * r, head: 9 });
      g.vector(f1, lerpV(f1, N.add(f1, f2), seg(r, 0.3, 0.9)), { color: 'j', width: 2, alpha: 0.6 * r, head: 9 });
      if (r > 0.95) g.marker.check(v, { dx: 22, dy: -6, size: 22 });
    }
  }

  /** 2D Gram–Schmidt scene, driven by s.gs = { n1 (0..1 normalize a), sh (shadow), lift, n2, rebuildA, rebuildW } */
  function drawGS2(g, s, ctx) {
    const vs = s.vs2, G = N.gramSchmidt(vs), st = G.steps, gs = s.gs;
    const a = vs[0], w = vs[1], na = N.norm(a);
    const q1 = st[0].normalized || [1, 0];
    if (gs.ghosts) { g.vector(a, { color: 'hud', width: 2, alpha: 0.35, label: 'a' }); g.vector(w, { color: 'hud', width: 2, alpha: 0.35, label: 'w' }); }
    // first vector: a → q₁
    if (!gs.ghosts) {
      const len = LA.lerp(na, 1, gs.n1), done = gs.n1 >= 0.999;
      g.vector(N.scale(q1, len), { color: done ? 'i' : 'v', label: done ? 'q₁' : 'a', width: 3, dx: done ? 6 : undefined, dy: done ? 16 : undefined });
      if (gs.n1 > 0 && gs.n1 < 0.9) g.label('|a| = ' + fmt(na), N.scale(q1, len), { color: 'hud', size: 13, dx: 10, dy: 22, alpha: 1 - gs.n1 });
    } else g.vector(q1, { color: 'i', label: 'q₁', width: 3, dx: 6, dy: 16 });
    // second vector
    const p = st[1].projections[0], u2 = st[1].residual, q2 = st[1].normalized;
    if (!gs.ghosts && gs.n2 < 0.999) g.vector(w, { color: 'w', label: 'w', width: 3, alpha: gs.lift > 0.5 ? 0.35 : 1 });
    if (gs.sh > 0 && gs.n2 < 0.999 && !gs.ghosts) {
      g.lineThrough([0, 0], q1, { len: 6, color: 'i', width: 1, alpha: 0.4 });
      g.dropLine(w, lerpV(w, p, seg(gs.sh, 0, 0.6)), { r: 4 });
      if (gs.sh > 0.6) { g.vector(p, { color: 'mag', width: 2.6, label: fmt(N.dot(w, q1)) + '·q₁', dx: 10, dy: 18 }); }
    }
    if (gs.lift > 0 && gs.n2 < 0.999 && !gs.ghosts) {
      const from = lerpV(p, [0, 0], gs.lift);
      g.vector(from, N.add(from, u2), { color: 'residual', width: 2.8, label: gs.lift > 0.98 ? 'w − (w·q₁)q₁' : '', labelAt: 0.6, labelOffset: -16 });
    }
    if (gs.n2 > 0 && q2) {
      const done = gs.n2 >= 0.999 || gs.ghosts, len = LA.lerp(N.norm(u2), 1, gs.n2);
      g.vector(N.scale(q2, done ? 1 : len), { color: done ? 'j' : 'residual', label: done ? 'q₂' : '', width: 3 });
      if (done) g.arc([0, 0], q1, q2, { r: 0.26, color: 'hud', right: true });
    }
    // rebuild a and w from Q with the recipe R (tip to tail)
    if (gs.ghosts && q2) {
      const r11 = N.dot(a, q1), r12 = N.dot(w, q1), r22 = N.dot(w, q2);
      if (gs.rebA > 0) { const e = N.scale(q1, r11 * gs.rebA); g.vector([0, 0], e, { color: 'i', width: 2.2, alpha: 0.75, head: 9 }); if (gs.rebA > 0.999) { g.marker.ring(a, { r: 13 }); g.label(fmt(r11) + ' q₁', a, { color: 'gold', size: 13, dx: 8, dy: 26 }); } }
      if (gs.rebW > 0) {
        const t1 = seg(gs.rebW, 0, 0.5), t2 = seg(gs.rebW, 0.5, 1), e1 = N.scale(q1, r12 * t1);
        g.vector([0, 0], e1, { color: 'i', width: 2.2, alpha: 0.75, head: 9 });
        if (t2 > 0) g.vector(e1, N.add(e1, N.scale(q2, r22 * t2)), { color: 'j', width: 2.2, alpha: 0.75, head: 9 });
        if (gs.rebW > 0.999) { g.marker.ring(w, { r: 13 }); g.label(fmt(r12) + ' q₁ + ' + fmt(r22) + ' q₂', w, { color: 'gold', size: 13, anchor: 'right', dx: -14, dy: -10 }); }
      }
    }
    if (ctx.mode === 'play') {
      g.handle('v0', a, (q, st2) => { if (N.norm(q) > 0.2) st2.vs2[0] = [q[0], q[1]]; }, { color: 'v' });
      g.handle('v1', w, (q, st2) => { st2.vs2[1] = [q[0], q[1]]; }, { color: 'w' });
      if (s.showCircle) { const A = N.fromCols(vs); g.ellipse(A, [0, 0], { stroke: 'hud', dashed: true, width: 1.3 }); if (q2) g.ellipse(N.fromCols([q1, q2]), [0, 0], { stroke: 'gold', width: 1.6 }); }
    }
  }

  /** playground: Gram–Schmidt stage machine in 2D or 3D */
  function drawStages(g, s, is3) {
    const vs = is3 ? s.vs3 : s.vs2, G = N.gramSchmidt(vs), L = stageList(vs.length), done = L.slice(0, s.stage);
    const has = k => done.indexOf(k) >= 0, cols = ['i', 'j', 'k'], lab = ['q₁', 'q₂', 'q₃'], inLab = is3 ? ['v₁', 'v₂', 'v₃'] : ['v₁', 'v₂'];
    if (is3 && G.basis.length >= 2) g.planes([{ point: [0, 0, 0], span: [G.basis[0], G.basis[1]], extent: 1.8, color: 'teal', alpha: 0.16 }]);
    vs.forEach((v, k) => {
      const stp = G.steps[k], nk = has('n' + k);
      g.vector(v, { color: k === 0 ? 'v' : k === 1 ? 'w' : 'b', label: inLab[k], width: 2.4, alpha: nk ? 0.3 : 0.9 });
      if (k > 0 && has('s' + k) && !has('l' + k)) stp.projections.forEach(p => { g.dropLine(v, p, { r: 3.5 }); g.vector(p, { color: 'mag', width: 2.2, head: 9 }); });
      if (k > 0 && has('l' + k) && !nk) g.vector(stp.residual, { color: 'residual', width: 2.8, label: 'u' + ['₁', '₂', '₃'][k] });
      if (nk && stp.normalized) g.vector(stp.normalized, { color: cols[k], label: lab[k], width: 3 });
      if (nk && !stp.normalized) g.label('dependent: nothing left', v, { color: 'bad', size: 13, dy: 18 });
    });
    if (!is3 && s.showCircle) { unitCircle(g); if (G.basis.length === 2) g.ellipse(N.fromCols(vs), [0, 0], { stroke: 'hud', dashed: true, width: 1.3 }); }
    vs.forEach((v, k) => g.handle('v' + k, v, (q, st) => { const arr = is3 ? st.vs3 : st.vs2; arr[k] = is3 ? [q[0], q[1], q[2]] : [q[0], q[1]]; }, { color: k === 0 ? 'v' : k === 1 ? 'w' : 'b', plane: is3 ? 'view' : undefined }));
  }

  /** checks 1–3 scene */
  function drawChk(g, s, ctx) {
    if (s.chk === 'orth' || s.chk === 'onb') {
      if (s.chk === 'onb') unitCircle(g, { stroke: 'gold', width: 1.6 });
      g.vector(s.cv, { color: 'v', label: 'v', width: 3 });
      g.vector(s.cw, { color: 'w', label: 'w', width: 3 });
      const ang = Math.acos(LA.clamp(N.dot(s.cv, s.cw) / ((N.norm(s.cv) * N.norm(s.cw)) || 1), -1, 1)) / DEG;
      g.arc([0, 0], s.cv, s.cw, { r: 0.4, color: Math.abs(ang - 90) < 4 ? 'ok' : 'hud', right: true });
      g.hud('angle = ' + fmt(ang) + '°   v·w = ' + fmt(N.dot(s.cv, s.cw)) + (s.chk === 'onb' ? '   |v| = ' + fmt(N.norm(s.cv)) + '   |w| = ' + fmt(N.norm(s.cw)) : ''), 14, 30, { size: 13.5 });
      if (s.chk === 'onb') g.handle('cv', s.cv, (q, st) => { st.cv = [q[0], q[1]]; }, { color: 'v' });
      g.handle('cw', s.cw, (q, st) => { st.cw = [q[0], q[1]]; }, { color: 'w' });
    } else if (s.chk === 'qr') {
      const q1 = N.col(Q2, 0), q2 = N.col(Q2, 1), Re = s.Re;
      g.vector(A_, { color: 'hud', width: 2, alpha: 0.35, label: 'a' }); g.vector(W_, { color: 'hud', width: 2, alpha: 0.35, label: 'w' });
      g.vector(q1, { color: 'i', label: 'q₁', width: 3, dx: 6, dy: 16 }); g.vector(q2, { color: 'j', label: 'q₂', width: 3 });
      const c1 = N.add(N.scale(q1, Re[0][0]), N.scale(q2, Re[1][0])), c2 = N.add(N.scale(q1, Re[0][1]), N.scale(q2, Re[1][1]));
      g.vector(c1, { color: 'residual', width: 2.4, label: 'Q·(R col 1)', labelAt: 0.6 }); g.vector(c2, { color: 'residual', width: 2.4, label: 'Q·(R col 2)', labelAt: 0.6 });
      if (N.dist(c1, A_) < 0.12) g.marker.ring(A_, { r: 13 }); if (N.dist(c2, W_) < 0.12) g.marker.ring(W_, { r: 13 });
    }
  }

  /* ---------- step 4: orthogonal vs shear (split) ---------- */
  function drawOrth(g, s) {
    const cam = { center: [0.35, 0.3], halfH: 3.8 };
    const side = (g2, r, M, text, fillAlpha) => {
      g2.ghostGrid(); g2.grid(M, { alpha: 0.7 }); g2.axes({ ticks: false });
      g2.parallelogram(N.col(M, 0), N.col(M, 1), { orient: true, alpha: 0.28 * fillAlpha, stroke: null });
      g2.circle([0, 0], 1, { stroke: 'hud', dashed: true, width: 1, strokeAlpha: 0.5 });
      g2.ellipse(M, [0, 0], { stroke: 'gold', width: 2.2, strokeAlpha: fillAlpha });
      g2.vector(N.col(M, 0), { color: 'i', width: 2.6, alpha: fillAlpha, label: 'î', dx: 8, dy: 14 }); g2.vector(N.col(M, 1), { color: 'j', width: 2.6, alpha: fillAlpha, label: 'ĵ' });
      if (text) g2.label(text, [r.x + 10, r.y + 18], { screen: true, anchor: 'left', size: 12.5, color: 'hud', font: 'ui' });
    };
    const u = s.orthU, f = s.flipU;
    g.split((g2, r) => {
      if (f < 1) side(g2, r, N.rot2(TH * u), f < 0.5 ? 'Q = rotation 30°: circle stays a circle' : '', 1 - f);
      if (f > 0) { const F = lerpM(I2(), FLIP, 1); side(g2, r, F, f >= 0.5 ? 'F = reflection: det −1, still a circle' : '', f); }
      if (u > 0.98 && f < 0.05) g2.marker.check([1.15, 1.25], { size: 22 });
      if (f > 0.95) g2.marker.check([1.15, 1.25], { size: 22 });
    }, (g2, r) => {
      const S = lerpM(I2(), SHEAR, u);
      side(g2, r, S, 'S = shear: the circle becomes an ellipse', 1);
    }, { left: { view: '2d', cam }, right: { view: '2d', cam } });
  }

  /* ---------- 3D Gram–Schmidt ---------- */
  function draw3d(g, s, ctx) {
    g.cage({ range: 2, walls: false, alpha: 0.5 });
    g.axes({ length: 2.2 });
    if (s.scene === 'gs3play') { drawStages(g, s, true); return; }
    const st = GS3.steps, q1 = st[0].normalized, q2 = st[1].normalized, q3 = st[2].normalized, u = s.gs3;
    const v1 = V3[0], v2 = V3[1], v3 = V3[2];
    g.planes([{ point: [0, 0, 0], span: [v1, v2], extent: 1.9, color: 'teal', alpha: 0.18 }]);
    // q₁, q₂: quick replays of steps 6–7
    const a1 = seg(u, 0, 0.2), a2 = seg(u, 0.2, 0.5);
    g.vector(v1, { color: 'v', width: 2, alpha: 0.35, label: 'v₁' });
    g.vector(v2, { color: 'w', width: 2, alpha: 0.35, label: 'v₂' });
    g.vector(N.scale(q1, LA.lerp(N.norm(v1), 1, a1)), { color: a1 > 0.99 ? 'i' : 'v', width: 3, label: a1 > 0.99 ? 'q₁' : '' });
    if (a2 > 0) {
      const p = st[1].projections[0], r2 = st[1].residual;
      if (a2 < 0.5) { g.dropLine(v2, lerpV(v2, p, seg(a2, 0, 0.5)), { r: 3.5 }); }
      const len = LA.lerp(N.norm(r2), 1, seg(a2, 0.5, 1));
      if (a2 >= 0.5) g.vector(N.scale(q2, len), { color: a2 > 0.99 ? 'j' : 'residual', width: 3, label: a2 > 0.99 ? 'q₂' : '' });
    }
    // v₃ loses two shadows, then normalizes
    g.vector(v3, { color: 'hud', width: 2.4, label: 'v₃', alpha: u > 0.85 ? 0.35 : 1 });
    if (u > 0.5) {
      const pr = st[2].projections, l1 = seg(u, 0.5, 0.67), l2 = seg(u, 0.68, 0.85);
      const cur = N.sub(N.sub(v3, N.scale(pr[0], l1)), N.scale(pr[1], l2));
      if (l1 < 1) g.vector(N.scale(pr[0], 1), { color: 'mag', width: 2.2, head: 9, alpha: 0.9 });
      if (l1 >= 1 && l2 < 1) g.vector(N.scale(pr[1], 1), { color: 'mag', width: 2.2, head: 9, alpha: 0.9 });
      const n3 = seg(u, 0.86, 1), len = LA.lerp(N.norm(st[2].residual), 1, n3);
      if (u < 0.86) g.vector(cur, { color: 'residual', width: 2.8 });
      else g.vector(N.scale(q3, len), { color: n3 > 0.99 ? 'k' : 'residual', width: 3, label: n3 > 0.99 ? 'q₃' : '' });
      if (n3 > 0.99) { g.arc([0, 0, 0], q1, q3, { r: 0.3, color: 'hud', right: true }); g.arc([0, 0, 0], q2, q3, { r: 0.3, color: 'hud', right: true }); }
    }
  }

  /* ---------- ML tie-in ---------- */
  function drawML(g, s) {
    const u = s.mlT, k = Math.min(8, Math.floor(seg(u, 0.05, 0.7, 'linear') * 8.999));
    const Wm = N.matMul(N.matMul(N.rot2(0.35), [[1.25, 0], [0, 0.75]]), N.transpose(N.rot2(0.35)));
    const Wk = N.matPow(Wm, k), Qk = N.matPow(N.rot2(TH), k);
    g.viewport({ x: 0, y: 0, w: 0.5, h: 0.5 }, { view: '2d', cam: { center: [0, 0], halfH: 2.4 } }, (g2, r) => {
      g2.ghostGrid({ alpha: 0.7 }); unitCircle(g2, { dashed: true }); g2.ellipse(Wk, [0, 0], { stroke: 'bad', width: 2.2 });
      g2.label('generic W, applied ' + k + '×', [r.x + 10, r.y + 18], { screen: true, anchor: 'left', size: 12.5, color: 'hud', font: 'ui' });
    });
    g.viewport({ x: 0, y: 0.5, w: 0.5, h: 0.5 }, { view: '2d', cam: { center: [0, 0], halfH: 2.4 } }, (g2, r) => {
      g2.ghostGrid({ alpha: 0.7 }); g2.ellipse(Qk, [0, 0], { stroke: 'gold', width: 2.2 });
      g2.vector(N.col(Qk, 0), { color: 'i', width: 2.2, head: 9 }); g2.vector(N.col(Qk, 1), { color: 'j', width: 2.2, head: 9 });
      g2.label('orthogonal Q, applied ' + k + '×', [r.x + 10, r.y + 18], { screen: true, anchor: 'left', size: 12.5, color: 'hud', font: 'ui' });
      const big = N.svd(Wk).S[0];
      g2.bars([big, 1], { screen: true, x: r.x + r.w - 96, y: r.y + r.h - 96, w: 70, h: 60, max: 1.6 * Math.max(6, big), colors: ['bad', 'gold'], labels: ['W', 'Q'], values: true, title: 'longest stretch' });
    });
    g.viewport({ x: 0.5, y: 0, w: 0.5, h: 1 }, { view: '2d', cam: { center: [-0.4, 0.6], halfH: 6.2 } }, (g2, r) => {
      const X = s.mlX, qr = N.qr(X), Q = qr.Q, R = qr.R, split = seg(u, 0.35, 0.65);
      const ramp = t => LA.color.diverge(2 * t - 1);
      const H = X.length * 0.55, yc = 1.0;
      const bx = cx => ({ x0: cx - 0.45, x1: cx + 0.45, y0: yc - H / 2, y1: yc + H / 2 });
      if (split < 1) { const b = bx(LA.lerp(-0.4, -1.1, split)); g2.image(X, b, { cells: true, range: [-3, 3], ramp, alpha: 1 - split }); g2.label('X (12×2)', [(b.x0 + b.x1) / 2, b.y1], { anchor: 'above', dy: -6, size: 13, color: 'text', alpha: 1 - split }); }
      if (split > 0) {
        const bq = bx(-1.1); g2.image(Q, bq, { cells: true, range: [-1, 1], ramp, alpha: split });
        g2.label('Q (12×2)', [-1.1, bq.y1], { anchor: 'above', dy: -6, size: 13, color: 'text', alpha: split });
        g2.label('X =', [-2.25, yc], { size: 17, color: 'text', alpha: split, font: 'math' });
        g2.label('·', [-0.35, yc], { size: 26, color: 'hud', alpha: split });
        tile(g2, [0.95, yc], R.map(row => row.map(x => Math.round(x * 100) / 100)), 'R (2×2)', { alpha: split, cw: 0.85, ch: 0.5 });
      }
      g2.label('synthetic 12-row design matrix: ones | x', [r.x + 10, r.y + 18], { screen: true, anchor: 'left', size: 12, color: 'muted', font: 'ui' });
      g2.label('solve R w = Qᵀy: no XᵀX', [r.x + 10, r.y + r.h - 16], { screen: true, anchor: 'left', size: 13, color: 'gold', font: 'ui', alpha: seg(u, 0.7, 0.85) });
    });
  }

  /* ---------- recap ---------- */
  function drawRecap(g) {
    const band = (i, cam, text, fn) => g.viewport({ x: 0, y: i / 3, w: 1, h: 1 / 3 }, { view: '2d', cam }, (g2, r) => { fn(g2); g2.label(text, [r.x + 12, r.y + 18], { screen: true, anchor: 'left', size: 13.5, color: 'text', bold: true }); });
    band(0, { center: [0, 0.1], halfH: 1.4 }, 'coordinates = shadows', g2 => {
      const R = N.rot2(TH); g2.ghostGrid({ alpha: 0.6 }); g2.grid(R, { alpha: 0.7 }); unitCircle(g2);
      g2.vector(N.col(R, 0), { color: 'i', label: 'q₁', width: 2.6, dx: 8, dy: 14 }); g2.vector(N.col(R, 1), { color: 'j', label: 'q₂', width: 2.6 });
    });
    band(1, { center: [0.6, 1.0], halfH: 1.25 }, 'subtract the shadow, normalize', g2 => {
      g2.ghostGrid({ alpha: 0.6 });
      const q1 = N.col(Q2, 0), q2 = N.col(Q2, 1), p = N.scale(q1, N.dot(W_, q1));
      g2.vector(W_, { color: 'w', label: 'w', width: 2.4, alpha: 0.6 }); g2.vector(p, { color: 'mag', width: 2.4, head: 9 });
      g2.vector(p, W_, { color: 'residual', width: 2, dashed: true, head: 8 });
      g2.vector(N.sub(W_, p), { color: 'residual', width: 2.6 });
      g2.vector(q1, { color: 'i', label: 'q₁', width: 2.6, dx: 6, dy: 14 }); g2.vector(q2, { color: 'j', label: 'q₂', width: 2.6 });
      g2.arc([0, 0], q1, q2, { r: 0.2, color: 'hud', right: true });
    });
    band(2, { center: [0, -0.15], halfH: 1.2 }, 'A = QR', g2 => {
      tile(g2, [-2.2, -0.15], [[3, 1], [1, 2]], 'A', { cw: 0.5, ch: 0.38 });
      g2.label('=', [-1.45, -0.15], { size: 20, color: 'hud' });
      tile(g2, [-0.55, -0.15], Q2.map(r => r.map(x => Math.round(x * 100) / 100)), 'Q', { cw: 0.62, ch: 0.38, cellColor: (i, j) => j === 0 ? 'i' : 'j' });
      g2.label('·', [0.35, -0.15], { size: 24, color: 'hud' });
      tile(g2, [1.3, -0.15], R2.map(r => r.map(x => Math.abs(x) < 1e-9 ? 0 : Math.round(x * 100) / 100)), 'R', { cw: 0.62, ch: 0.38, color: 'gold', cellColor: 'gold' });
    });
  }

  function baseGS(s) { s.scene = 'gs'; s.view = '2d'; s.cam = CAM2(); s.vs2 = [A_.slice(), W_.slice()]; s.circle = true; s.showRotGrid = false; s.theta = 0; s.gs = { n1: 1, sh: 1, lift: 1, n2: 1, ghosts: false, rebA: 0, rebW: 0 }; }

  LA.registerLesson({
    id: 'p3-04-orthonormal-gram-schmidt',
    part: 3,
    title: 'Orthonormal bases, orthogonal matrices, Gram–Schmidt, QR',
    subtitle: 'Perpendicular unit vectors make coordinates into dot products; Gram–Schmidt builds them; QR records how.',
    minutes: 11,
    prereqs: ['p3-03-projections', 'p1-04-independence-basis', 'p2-03-transformation-zoo'],
    tags: ['core'],
    badge: { drawnIn: '2D and 3D', realSize: 'QR of a 10⁵×10³ design matrix is how lstsq is computed; orthogonal 4096×4096 matrices initialize transformer weights', transfers: 'QᵀQ = I, length preservation and “subtract the shadows, then normalize” are identical in every dimension; only the count of vectors grows' },
    cantShow: ['reflections in n-D (det Q = −1) beyond a 2D mirror; only one 2D reflection is shown', 'loss of orthogonality from rounding in classical Gram–Schmidt (why libraries use Householder): a numerical fact with no picture'],
    terms: [
      { term: 'orthonormal', def: 'Perpendicular to each other and each of length 1.' },
      { term: 'orthogonal matrix', def: 'A square matrix whose columns are orthonormal, so QᵀQ = I and Q⁻¹ = Qᵀ.' },
      { term: 'Gram–Schmidt', def: 'Straighten a basis one vector at a time: subtract its shadows on the vectors already built, then normalize.' },
      { term: 'QR decomposition', def: 'A = QR with Q having orthonormal columns and R upper-triangular: the recipe for rebuilding A from Q.' },
    ],
    atlas: ['linear-regression', 'embeddings', 'mlp'],
    thumbnail: 6,

    state() {
      const d = LA.rng.linearData(17, 12, { w: [0.7], b: 0.4, noise: 0.3, xRange: [-2, 2] });
      return {
        view: '2d', cam: CAM2(), scene: 'rot', panel: 'Q',
        theta: 0, v: V_.slice(), showRotGrid: true, circle: true, showV: false, coords: 0, rightMark: false, glow: [1, 1], fillCell: null,
        misU: 0, fixU: 0, orthU: 0, flipU: 0,
        vs2: [A_.slice(), W_.slice()], gs: { n1: 0, sh: 0, lift: 0, n2: 0, ghosts: false, rebA: 0, rebW: 0 },
        gs3: 0, vs3: LA.clone(V3), stage: 0, showCircle: true,
        mlT: 0, mlX: d.x.map(x => [1, Math.round(x * 100) / 100]),
        chk: null, cv: [3, 1], cw: [1, 2], Re: I2(),
      };
    },

    render(g, s, ctx) {
      if (s.scene === 'recap') { drawRecap(g); return; }
      if (s.scene === 'orth') { drawOrth(g, s); return; }
      if (s.scene === 'ml') { drawML(g, s); return; }
      if (s.view === '3d') { draw3d(g, s, ctx); return; }
      if (s.scene === 'gsplay') { g.ghostGrid(); g.axes(); drawStages(g, s, false); return; }
      draw2d(g, s, ctx);
    },

    math(s, tex0) {
      const tex = safeTex(tex0), P = s.panel;
      const R = N.rot2(s.theta);
      const qsym = (k, c) => tex.sym('q', { sub: String(k), color: c });
      if (P === 'recap') return tex.block(tex.eq(tex.sym('A'), tex.op(' = '), tex.sym('Q'), tex.sym('R')));
      if (P === 'Q') return tex.block(tex.eq(tex.sym('Q'), tex.op(' = '), tex.mat([['cos θ', '−sin θ'], ['sin θ', 'cos θ']], { colors: 'cols' }), tex.op(' = '), tex.mat(R, { colors: 'cols' })), tex.eq(tex.sym('θ'), tex.op(' = '), tex.num(s.theta / DEG, { digits: 0 }), tex.op('°')));
      if (P === 'coords') { const q1 = N.col(R, 0), q2 = N.col(R, 1); return tex.block(tex.eq(tex.sym('v', { color: 'v' }), tex.op(' = '), tex.paren(tex.sym('v') + tex.op('·') + qsym(1, 'i')), qsym(1, 'i'), tex.op(' + '), tex.paren(tex.sym('v') + tex.op('·') + qsym(2, 'j')), qsym(2, 'j')), tex.eq(tex.op(' = '), tex.num(N.dot(s.v, q1), { color: 'i' }), qsym(1, 'i'), tex.op(' + '), tex.paren(tex.num(N.dot(s.v, q2), { color: 'j' })), qsym(2, 'j'))); }
      if (P === 'mis') { const pa = N.proj(V_, A_), pw = N.proj(V_, W_); return tex.block(tex.eq(tex.frac(tex.paren(tex.sym('v') + tex.op('·') + tex.sym('a', { color: 'v' })), tex.op('|') + tex.sym('a') + tex.op('|²')), tex.sym('a', { color: 'v' }), tex.op(' + '), tex.frac(tex.paren(tex.sym('v') + tex.op('·') + tex.sym('w', { color: 'w' })), tex.op('|') + tex.sym('w') + tex.op('|²')), tex.sym('w', { color: 'w' }), tex.op(' = '), tex.vec(N.add(pa, pw), { color: 'mag' }), tex.op(' ≠ '), tex.vec(V_))); }
      if (P === 'orth') return tex.block(tex.eq(tex.op('|'), tex.sym('Q'), tex.sym('v'), tex.op('| = |'), tex.sym('v'), tex.op('|')), tex.eq(tex.text('det '), tex.sym('Q'), tex.op(' = ±1')), tex.eq(tex.text('det '), tex.sym('S'), tex.op(' = 1, but '), tex.op('|'), tex.sym('S'), tex.sym('v'), tex.op('| ≠ |'), tex.sym('v'), tex.op('|')));
      if (P === 'QtQ') {
        const f = s.fillCell === null || s.fillCell === undefined ? 4 : s.fillCell + 1, vals = [[1, 0], [0, 1]];
        const cells = vals.map((row, i) => row.map((x, j) => (i * 2 + j) < f ? x : '?'));
        return tex.block(tex.eq(tex.T(tex.sym('Q')), tex.sym('Q'), tex.op(' = '), tex.mat([['q₁·q₁', 'q₁·q₂'], ['q₂·q₁', 'q₂·q₂']]), tex.op(' = '), tex.mat(cells, { colors: 'cols', highlight: s.fillCell >= 0 && s.fillCell < 4 ? [Math.floor(s.fillCell / 2), s.fillCell % 2] : undefined })), tex.eq(tex.inv(tex.sym('Q')), tex.op(' = '), tex.T(tex.sym('Q'))));
      }
      if (P === 'gs1') { const a = s.vs2[0]; return tex.block(tex.eq(qsym(1, 'i'), tex.op(' = '), tex.frac(tex.sym('a', { color: 'v' }), tex.op('|') + tex.sym('a') + tex.op('|')), tex.op(' = '), tex.vec(N.normalize(a), { color: 'i' })), tex.eq(tex.op('|'), tex.sym('a', { color: 'v' }), tex.op('| = '), tex.num(N.norm(a), { color: 'gold' }))); }
      if (P === 'gs2') {
        const G = N.gramSchmidt(s.vs2), st = G.steps[1], q1 = G.steps[0].normalized;
        return tex.block(
          tex.eq(tex.sym('u', { sub: '2' }), tex.op(' = '), tex.sym('w', { color: 'w' }), tex.op(' − '), tex.paren(tex.sym('w') + tex.op('·') + qsym(1, 'i')), qsym(1, 'i'), tex.op(' = '), tex.vec(st.residual)),
          tex.eq(qsym(2, 'j'), tex.op(' = '), tex.frac(tex.sym('u', { sub: '2' }), tex.op('|') + tex.sym('u', { sub: '2' }) + tex.op('|')), tex.op(' = '), tex.vec(st.normalized || [0, 0], { color: 'j' })),
          tex.eq(tex.sym('w'), tex.op('·'), qsym(1, 'i'), tex.op(' = '), tex.num(N.dot(s.vs2[1], q1), { color: 'gold' }), tex.op(',  |'), tex.sym('u', { sub: '2' }), tex.op('| = '), tex.num(N.norm(st.residual), { color: 'gold' })));
      }
      if (P === 'QR') return tex.block(tex.eq(tex.sym('A'), tex.op(' = '), tex.sym('Q'), tex.sym('R')), tex.eq(tex.sym('Q'), tex.op(' = '), tex.mat(Q2, { colors: 'cols' })), tex.eq(tex.sym('R'), tex.op(' = '), tex.mat(R2.map(r => r.map(x => Math.abs(x) < 1e-9 ? 0 : x)), { colors: () => 'gold' })));
      if (P === 'gs3') { const Q3 = N.fromCols(GS3.basis); return tex.block(tex.eq(qsym(3, 'k'), tex.op(' = '), tex.frac(tex.sym('v', { sub: '3' }) + tex.op(' − (') + tex.sym('v', { sub: '3' }) + tex.op('·') + qsym(1, 'i') + tex.op(')') + qsym(1, 'i') + tex.op(' − (') + tex.sym('v', { sub: '3' }) + tex.op('·') + qsym(2, 'j') + tex.op(')') + qsym(2, 'j'), tex.op('|…|'))), tex.eq(tex.sym('Q'), tex.op(' = '), tex.mat(Q3, { colors: 'cols' })), tex.eq(tex.T(tex.sym('Q')), tex.sym('Q'), tex.op(' = '), tex.sym('I', { sub: '3' }))); }
      if (P === 'ml') return tex.block(tex.eq(tex.sym('X'), tex.op(' = '), tex.sym('Q'), tex.sym('R')), tex.eq(tex.sym('R'), tex.sym('w'), tex.op(' = '), tex.T(tex.sym('Q')), tex.sym('y', { color: 'b' })));
      if (P === 'play' || P === 'play3') {
        const vs = P === 'play3' ? s.vs3 : s.vs2, G = N.gramSchmidt(vs);
        if (G.basis.length < vs.length) return tex.block(tex.eq(tex.text('the vectors are dependent: Gram–Schmidt finds only ' + G.basis.length + ' directions', { color: 'bad' })));
        const Q = N.fromCols(G.basis), Rm = N.matMul(N.transpose(Q), N.fromCols(vs)).map((r, i) => r.map((x, j) => j < i ? 0 : x));
        return tex.block(tex.eq(tex.sym('Q'), tex.op(' = '), tex.mat(Q, { colors: 'cols' })), tex.eq(tex.sym('R'), tex.op(' = '), tex.mat(Rm, { colors: () => 'gold' })), tex.eq(tex.T(tex.sym('Q')), tex.sym('Q'), tex.op(' = '), tex.mat(N.matMul(N.transpose(Q), Q).map(r => r.map(x => Math.abs(x) < 5e-3 ? 0 : x)))));
      }
      if (P === 'chk') return tex.block(tex.eq(tex.sym('v', { color: 'v' }), tex.op(' = '), tex.vec(s.cv, { color: 'v' })), tex.eq(tex.sym('w', { color: 'w' }), tex.op(' = '), tex.vec(s.cw, { color: 'w' })));
      if (P === 'chkqr') return tex.block(tex.eq(tex.sym('Q'), tex.op(' = '), tex.mat(Q2, { colors: 'cols' })), tex.eq(tex.sym('Q'), tex.sym('R'), tex.op(' = '), tex.mat(N.matMul(Q2, s.Re))), tex.eq(tex.text('target '), tex.mat([[3, 1], [1, 2]])));
      return '';
    },

    steps: [
      { title: 'A rotated grid is still a perfect grid', duration: 3.5,
        narration: 'Rotate î and ĵ by 30°. The two arrows, the green ' + q1s + ' and the red ' + q2s + ', are still perpendicular and still length 1: an <b>orthonormal</b> basis. Its grid is the ordinary grid, turned. Nothing is stretched, nothing is sheared, and the unit circle stays a circle.',
        setup(s) { s.scene = 'rot'; s.view = '2d'; s.cam = CAM2(); s.theta = 0; s.showRotGrid = true; s.circle = true; s.showV = false; s.coords = 0; s.rightMark = false; s.glow = [1, 1]; s.fillCell = null; s.panel = 'Q'; },
        at(s, t) { s.theta = TH * seg(t, 0.1, 0.85); s.rightMark = t > 0.88; } },

      { title: 'Coordinates by dot products', duration: 4,
        narration: 'How much of the yellow <span class="c-v">v</span> = [2; 1] lies along ' + q1s + '? Its shadow, and since q₁ has length 1, the shadow length is just the dot product v·q₁ = 2.23. Same along ' + q2s + ': v·q₂ = −0.13. In an orthonormal basis, coordinates are shadows.',
        setup(s) { s.theta = TH; s.showV = true; s.coords = 0; s.rightMark = true; s.panel = 'coords'; },
        at(s, t) { s.coords = seg(t, 0.05, 0.95, 'linear'); } },

      { title: 'Misconception: dot products do not read coordinates in a slanted basis', duration: 4, misconception: true,
        narration: '✕ Try the same trick with the slanted basis, yellow ' + av + ' = [3; 1] and purple ' + wv + ' = [1; 2]: the magenta shadows of v on a and on w, added tip to tail, miss v. Shadows only add up when the basis vectors do not overlap, when they are perpendicular. That is why we want orthonormal bases.',
        setup(s) { s.scene = 'mis'; s.showRotGrid = false; s.coords = 0; s.showV = false; s.misU = 0; s.fixU = 0; s.panel = 'mis'; },
        at(s, t) { s.misU = seg(t, 0, 0.6, 'linear'); s.fixU = seg(t, 0.68, 1, 'linear'); } },

      { title: 'Orthogonal matrices keep lengths and angles', duration: 4.5,
        narration: 'A square matrix with orthonormal columns is an <b>orthogonal matrix</b>. It moves every vector without changing any length or angle: the gold circle stays a circle, the square stays a square. Compare the shear on the right, which turns the circle into an ellipse. Rotations and reflections are the only 2D examples.',
        setup(s) { s.scene = 'orth'; s.orthU = 0; s.flipU = 0; s.panel = 'orth'; },
        at(s, t) { s.orthU = seg(t, 0.05, 0.45); s.flipU = seg(t, 0.6, 0.8); } },

      { title: 'QᵀQ = I: four dot products', duration: 4,
        narration: 'Why is Qᵀ the inverse of Q? The entries of QᵀQ are the dot products of the columns: q₁·q₁ = 1, q₁·q₂ = 0, q₂·q₁ = 0, q₂·q₂ = 1. The identity. Undoing a rotation costs a transpose, not an inversion: watch Qᵀ turn the rotated grid straight back.',
        setup(s) { s.scene = 'rot'; s.theta = TH; s.showRotGrid = true; s.showV = false; s.coords = 0; s.rightMark = false; s.fillCell = -1; s.glow = [1, 1]; s.panel = 'QtQ'; },
        at(s, t) {
          const c = t < 0.08 ? -1 : Math.min(3, Math.floor(seg(t, 0.08, 0.52, 'linear') * 3.999));
          s.fillCell = t < 0.56 ? c : 4;
          s.glow = c === 0 ? [2, 1] : c === 3 ? [1, 2] : c > 0 ? [2, 2] : [1, 1];
          s.rightMark = c === 1 || c === 2;
          s.theta = TH * (1 - seg(t, 0.6, 0.92));
        } },

      { title: 'Gram–Schmidt, step 1: normalize the first', duration: 3,
        narration: 'Take the slanted pair, yellow ' + av + ' = [3; 1] and purple ' + wv + ' = [1; 2], and straighten it with <b>Gram–Schmidt</b>. Step 1: keep a’s direction, fix its length. Divide by |a| = 3.16 and a shrinks onto the unit circle: the green q₁ = a/|a| = [0.95; 0.32].',
        setup(s) { baseGS(s); s.gs = { n1: 0, sh: 0, lift: 0, n2: 0, ghosts: false, rebA: 0, rebW: 0 }; s.glow = [1, 1]; s.fillCell = null; s.panel = 'gs1'; },
        at(s, t) { s.gs.n1 = seg(t, 0.15, 0.85); } },

      { title: 'Step 2: subtract the shadow, then normalize', duration: 5,
        narration: 'Step 2: w’s shadow on q₁ is (w·q₁)q₁ = 1.58·q₁, in magenta. Lift it off: w − 1.58 q₁ = [−0.5; 1.5] is perpendicular to q₁, the residual from the projections lesson. Normalize it: the red q₂ = [−0.32; 0.95]. Two perpendicular unit vectors, built from two slanted ones.',
        setup(s) { s.gs = { n1: 1, sh: 0, lift: 0, n2: 0, ghosts: false, rebA: 0, rebW: 0 }; s.panel = 'gs2'; },
        at(s, t) { s.gs.sh = seg(t, 0, 0.3, 'linear'); s.gs.lift = seg(t, 0.32, 0.62); s.gs.n2 = seg(t, 0.68, 0.92); } },

      { title: 'R records what was removed: A = QR', duration: 4,
        narration: 'Nothing was lost. a = 3.16 q₁ and w = 1.58 q₁ + 1.58 q₂. Put those recipes in columns: that is <b>R</b>, upper-triangular, and A = [a w] = QR, the <b>QR decomposition</b>. Q is the straightened frame; R is the recipe that rebuilds the gray originals from it.',
        setup(s) { s.gs = { n1: 1, sh: 1, lift: 1, n2: 1, ghosts: true, rebA: 0, rebW: 0 }; s.panel = 'QR'; },
        at(s, t) { s.gs.rebA = seg(t, 0.05, 0.4); s.gs.rebW = seg(t, 0.5, 0.95, 'linear'); } },

      { title: 'Three vectors in 3D', duration: 6,
        narration: 'In 3D the third vector has two shadows to lose, one on the green ' + q1s + ' and one on the red ' + q2s + ', and what remains points straight out of their teal plane. Start from [1; 1; 0], [1; 0; 1], [0; 1; 1]: the blue ' + q3s + ' ends up along [−1; 1; 1]/√3, the normal of the first two.',
        // low camera, nearly edge-on to the teal plane, so q₃ visibly sticks out of it
        setup(s) { s.scene = 'gs3'; s.view = '3d'; s.cam = Object.assign(CAM3(), { yaw: 0.75, pitch: 0.38 }); s.gs3 = 0; s.panel = 'gs3'; },
        at(s, t) { s.gs3 = t; LA.cam.tween(s, { yaw: 0.75 }, { yaw: 1.05 }, seg(t, 0, 1, 'linear')); } },

      { title: 'Why ML cares: safe matrices and the real least-squares solver', duration: 4.5, mlTie: true,
        narration: 'Orthogonal matrices neither explode nor shrink anything, so deep networks are often <i>initialized</i> with them: apply a generic matrix eight times and the circle becomes a spike; apply Q and it stays a circle. And least-squares solvers factor X = QR, then solve Rw = Qᵀy: no XᵀX, numerically safer.',
        setup(s) { s.scene = 'ml'; s.view = '2d'; s.cam = CAM2(); s.mlT = 0; s.panel = 'ml'; },
        at(s, t) { s.mlT = t; } },

      { title: 'Recap', duration: 2.5, recap: true,
        narration: 'Orthonormal: perpendicular unit vectors, so coordinates are dot products and QᵀQ = I. Orthogonal matrices preserve lengths and angles: circles stay circles. Gram–Schmidt: subtract the shadows, normalize; R remembers what you removed, A = QR.',
        cards: ['<b>Coordinates = shadows.</b> In an orthonormal basis, the coordinate along q is just q·v.', '<b>Circles stay circles.</b> An orthogonal matrix has QᵀQ = I, so Q⁻¹ = Qᵀ and lengths never change.', '<b>Subtract shadows, normalize.</b> Gram–Schmidt builds Q; R records what was removed: A = QR.'],
        setup(s) { s.scene = 'recap'; s.view = '2d'; s.cam = CAM2(); s.panel = 'recap'; },
        at(s, t) { } },
    ],

    playground: {
      intro: '<p>Drag the input vectors, then press <b>Next step</b> to run Gram–Schmidt one move at a time (shadows → lift → normalize). Try the nearly dependent preset: q₂ swings wildly as you move v₂.</p>',
      setup(s) { s.scene = 'gsplay'; s.view = '2d'; s.cam = CAM2(); s.vs2 = [A_.slice(), W_.slice()]; s.stage = 0; s.showCircle = true; s.panel = 'play'; },
      controls: [
        { type: 'button', label: 'Next step', primary: true, onClick(s) { const n = s.view === '3d' ? 3 : 2; s.stage = (s.stage + 1) % (stageList(n).length + 1); } },
        { type: 'toggle', bind: 'showCircle', label: 'Unit circle under Q vs A (2D)' },
        { type: 'preset', label: 'Presets', items: [
          { label: 'Slanted pair', apply(s) { s.scene = 'gsplay'; s.view = '2d'; s.cam = CAM2(); s.vs2 = [A_.slice(), W_.slice()]; s.stage = 0; s.panel = 'play'; } },
          { label: 'Nearly dependent', apply(s) { s.scene = 'gsplay'; s.view = '2d'; s.cam = CAM2(); s.vs2 = [[3, 1], [3.1, 1]]; s.stage = 4; s.panel = 'play'; } },
          { label: 'Already orthonormal', apply(s) { s.scene = 'gsplay'; s.view = '2d'; s.cam = CAM2(); const R = N.rot2(TH); s.vs2 = [N.col(R, 0), N.col(R, 1)]; s.stage = 0; s.panel = 'play'; } },
          { label: '3D triple', apply(s) { s.scene = 'gs3play'; s.view = '3d'; s.cam = CAM3(); s.vs3 = LA.clone(V3); s.stage = 0; s.panel = 'play3'; } },
        ] },
        { type: 'preset', label: 'View', items: [
          { label: '2D', apply(s) { s.scene = 'gsplay'; s.view = '2d'; s.cam = CAM2(); s.stage = Math.min(s.stage, 4); s.panel = 'play'; } },
          { label: '3D', apply(s) { s.scene = 'gs3play'; s.view = '3d'; s.cam = CAM3(); s.panel = 'play3'; } },
        ] },
      ],
    },

    checks: [
      { kind: 'manipulate',
        prompt: 'Drag the purple ' + wv + ' so that it is orthogonal to the yellow <span class="c-v">v</span> (dot product zero).',
        setup(s) { s.scene = 'chk'; s.view = '2d'; s.cam = CAM2(); s.chk = 'orth'; s.cv = [3, 1]; s.cw = [1, 2]; s.showRotGrid = false; s.circle = false; s.panel = 'chk'; },
        judge(s) { const r = LA.judge.angleDeg(s.cv, s.cw, 90, LA.TOL.angleDeg); return { ok: r.ok && N.norm(s.cw) > 0.3, progress: r.progress }; },
        hint: 'Perpendicular to [3; 1] means along [−1; 3] (or [1; −3]).',
        solution(s) { s.cw = [-1, 3]; } },
      { kind: 'manipulate',
        prompt: 'Make {v, w} orthonormal: perpendicular, and both of length 1.',
        setup(s) { s.scene = 'chk'; s.view = '2d'; s.cam = CAM2(); s.chk = 'onb'; s.cv = [3, 1]; s.cw = [1, 2]; s.showRotGrid = false; s.circle = false; s.panel = 'chk'; },
        judge(s) {
          const a = Math.abs(N.norm(s.cv) - 1), b = Math.abs(N.norm(s.cw) - 1), d = Math.abs(N.dot(s.cv, s.cw));
          return { ok: a < LA.TOL.unitNorm && b < LA.TOL.unitNorm && d < 0.05, progress: Math.max(0, 1 - (a + b + d) / 3) };
        },
        hint: 'Both tips must sit on the gold circle, a quarter turn apart.',
        solution(s) { const q = N.normalize([3, 1]); s.cv = q; s.cw = [-q[1], q[0]]; } },
      { kind: 'entry',
        prompt: 'Type R so that [a w] = [[3, 1], [1, 2]] = Q·R for the drawn Q (two decimals). The gray arrows Q·(columns of R) should land on the ghosts of a and w.',
        bind: 'Re', answer: [[3.16, 1.58], [0, 1.58]], tol: 0.06, label: 'R', colors: 'text',
        setup(s) { s.scene = 'chk'; s.view = '2d'; s.cam = CAM2(); s.chk = 'qr'; s.Re = I2(); s.showRotGrid = false; s.circle = true; s.panel = 'chkqr'; } },
    ],
  });
})();
