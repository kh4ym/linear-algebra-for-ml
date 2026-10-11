/* p2-05 — 3×3 matrices, 3D rotations, rotation order. */
(function () {
  'use strict';
  const N = LA.num, lerpM = LA.lerpM, seg = LA.seg, fmt = LA.fmt;
  const DEG = Math.PI / 180;
  const I3 = () => [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  const TALL = [[1, 0, 0], [0, 1, 0], [0, 0, 2]];
  const SH3 = [[1, 0, 0.5], [0, 1, 0], [0, 0, 1]], SC3 = [[1.5, 0, 0], [0, 0.7, 0], [0, 0, 1.2]];
  const GEN = [[1.3, 0.3, 0.5], [0.2, 0.9, 0], [0, 0.25, 1.2]];
  const TILT = N.normalize([1, 1, 1]);
  const ihat = '<span class="c-i">î</span>', jhat = '<span class="c-j">ĵ</span>', khat = '<span class="c-k">k̂</span>';
  const BASIS = ['i', 'j', 'k'];
  // ten hand-set word vectors with planted clusters (synthetic, for the ML step)
  const WORDS = [
    ['cat', [1.5, 0.45, 0.3], 'data'], ['kitten', [1.4, 0.62, 0.42], 'data'], ['dog', [1.3, 0.3, 0.72], 'data'],
    ['car', [0.3, 1.5, 0.15], 'dataB'], ['truck', [0.15, 1.4, 0.5], 'dataB'], ['road', [0.5, 1.25, -0.2], 'dataB'],
    ['apple', [0.45, -0.35, 1.5], 'dataC'], ['pear', [0.3, -0.5, 1.4], 'dataC'],
    ['invoice', [-1.0, 0.45, 0.85], 'w'], ['tax', [-1.15, 0.3, 0.65], 'w'],
  ];
  const cosM = vs => vs.map(a => vs.map(b => N.dot(a, b) / (N.norm(a) * N.norm(b))));
  const COS0 = cosM(WORDS.map(w => w[1]));

  /* numbers in the math panel as plain colored spans (tex.num's in-place update can write into the wrong matrix cell) */
  const tn = (tex, x, color) => '<span class="' + tex.colorClass(color) + '">' + (typeof x === 'number' ? fmt(x) : x) + '</span>';
  const r2 = M => M.map(r => r.map(x => { const y = Math.round(x * 100) / 100; return y === 0 ? 0 : y; }));
  const rotAbout = (ax, th) => (ax === 'x' ? N.rotX(th) : ax === 'y' ? N.rotY(th) : ax === 'z' ? N.rotZ(th) : N.rot3(TILT, th));
  const cam0 = () => Object.assign(LA.cam.defaults3d(), { yaw: 35 * DEG, pitch: 25 * DEG, dist: 8, target: [0.35, 0.35, 0.45] });

  /** a 3×3 matrix drawn as text at screen px, entries colored by column (works in 2D and 3D viewports) */
  function hudMat3(g, M, x, y, o) {
    // (x, y) = top-left of the bracketed block; drawn on an opaque panel so cage lines never run through the numbers
    o = o || {}; const cw = o.cw || 46, ch = o.ch || 24, size = o.size || 16, a = o.alpha === undefined ? 1 : o.alpha;
    const n = M.length, m = M[0].length, w = cw * m + 12, h = ch * n;
    const nameW = o.name ? o.name.length * (size + 1) * 0.56 + 12 : 0;
    g.rect(x - nameW - 10, y - 8, w + nameW + 20, h + 16, { fill: 'panel', alpha: 0.92 * a, stroke: 'line', width: 1 });
    const bar = (bx, d) => { g.rect(bx, y - 2, 2, h + 4, { fill: 'hud', alpha: a }); g.rect(d > 0 ? bx : bx - 5, y - 2, 7, 2, { fill: 'hud', alpha: a }); g.rect(d > 0 ? bx : bx - 5, y + h, 7, 2, { fill: 'hud', alpha: a }); };
    bar(x, 1); bar(x + w, -1);
    for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) {
      const v = M[i][j];
      g.label(typeof v === 'number' ? fmt(Math.round(v * 100) / 100 || 0) : v, [x + 6 + cw * (j + 0.5), y + ch * (i + 0.5)], { screen: true, color: o.mono ? 'hud' : BASIS[j], size, font: 'math', bold: true, bg: false, alpha: a });
    }
    if (o.name) g.label(o.name, [x - 8, y + h / 2], { screen: true, color: o.nameColor || 'hud', size: size + 1, anchor: 'right', font: 'math', bg: false, alpha: a });
  }

  /** the shared 3D picture: cage, optional lattice, the box under M, the three landing spots */
  function scene(g, M, o) {
    o = o || {};
    if (o.cage !== false) g.cage({ range: 2, alpha: 0.5 });
    if (o.lattice) g.lattice(M, { range: 1, alpha: 0.3 });
    if (o.floor) g.grid(M, { range: 2, alpha: 0.75, color: 'grid' });
    g.box(M, { fill: 'v', alpha: o.boxAlpha === undefined ? 0.13 : o.boxAlpha, stroke: 'v', strokeAlpha: o.edgeAlpha === undefined ? 0.9 : o.edgeAlpha });
    if (o.sticker) {   // a red sticker on the +y face makes the end position readable
      const f = [[0.15, 1, 0.15], [0.85, 1, 0.15], [0.85, 1, 0.85], [0.15, 1, 0.85]].map(p => N.matVec(M, p));
      g.polygon(f, { fill: 'j', alpha: 0.75, stroke: 'j', subdiv: 0 });
    }
    for (let j = 0; j < 3; j++) {
      const c = N.col(M, j), w = o.thick ? 4.2 : 3, pa = o.pulse === undefined || o.pulse === null ? 1 : (o.pulse === j ? 1 : 0.35);
      g.vector([0, 0, 0], c, { color: BASIS[j], label: BASIS[j], hat: true, width: w, alpha: pa });
      if (o.coords && (o.pulse === j || o.pulse === 'all')) {
        // put the coordinates beside the tip, perpendicular to the arrow on screen, so they never sit on the shaft
        const s0 = g.P.toScreen([0, 0, 0]), s1 = g.P.toScreen(c), L = Math.hypot(s1[0] - s0[0], s1[1] - s0[1]) || 1;
        let px = -(s1[1] - s0[1]) / L, py = (s1[0] - s0[0]) / L; if (px < 0 || (Math.abs(px) < 0.2 && py < 0)) { px = -px; py = -py; }
        const k = 18 + 40 * Math.abs(px);
        g.label('[' + c.map(x => fmt(Math.round(x * 100) / 100 || 0)).join('; ') + ']', c, { color: BASIS[j], size: 12.5, dx: px * k, dy: py * k });
      }
    }
    if (o.halo !== undefined && o.halo !== null) { const c = N.col(M, o.halo); g.marker.ring(c, { r: 15, color: BASIS[o.halo] }); g.lineThrough([0, 0, 0], c, { color: BASIS[o.halo], alpha: 0.45, dashed: true, len: 3 }); }
    if (o.axes !== false) g.axes({ length: 2.4, labels: true });
  }

  /** the path of î through two 90° rotations (about a, then about b) */
  function hopPath(g, a, b, alpha) {
    const pts = [], e = [1, 0, 0];
    for (let k = 0; k <= 16; k++) pts.push(N.matVec(rotAbout(a, (Math.PI / 2) * k / 16), e));
    const mid = pts[pts.length - 1];
    for (let k = 1; k <= 16; k++) pts.push(N.matVec(rotAbout(b, (Math.PI / 2) * k / 16), mid));
    g.polyline(pts, { color: 'i', dashed: true, width: 1.6, alpha: alpha === undefined ? 0.8 : alpha });
    return pts[pts.length - 1];
  }

  function orderM(first, second, u1, u2) { return N.matMul(rotAbout(second, (Math.PI / 2) * u2), rotAbout(first, (Math.PI / 2) * u1)); }

  /* ---------------- render modes ---------------- */
  function renderMain(g, s, ctx) {
    const M = s.M3;
    if (s.mode === 'tilted') {
      scene(g, M, { lattice: false });
      g.lineThrough([0, 0, 0], TILT, { color: 'gold', width: 2.6, len: 3.2 });
      const P = N.scale(TILT, 1.25); g.point(P, { r: 6, color: 'gold' }); g.label('fixed', P, { color: 'gold', size: 12.5, dx: 26, dy: -6 });
      ['i', 'j', 'k'].forEach((b, j) => g.vector([0, 0, 0], N.basis(j, 3), { color: b, dashed: true, width: 1.6, alpha: 0.4 }));
      g.hud('gold axis through [1; 1; 1]', 14, 22, { color: 'gold' });
      g.hud('3D only: every rotation has an axis (in n-D it fixes a plane)', 14, g.size.h - 22, { color: 'muted' });
      return;
    }
    if (s.mode === 'yz') {   // rotation about î: draw the yz wall carried by M
      scene(g, M, { halo: 0 });
      for (let i = -2; i <= 2; i++) {
        g.line(N.matVec(M, [0, i, -2]), N.matVec(M, [0, i, 2]), { color: 'grid', alpha: 0.7, width: i === 0 ? 1.6 : 1 });
        g.line(N.matVec(M, [0, -2, i]), N.matVec(M, [0, 2, i]), { color: 'grid', alpha: 0.7, width: i === 0 ? 1.6 : 1 });
      }
      return;
    }
    if (s.mode === 'compose') {
      if (s.ghostBox) g.box(s.ghostBox, { faces: false, stroke: 'muted', strokeAlpha: 0.5 });
      scene(g, M, {});
      return;
    }
    if (s.mode === 'check3') return renderCheck3(g, s);
    if (s.mode === 'play') return renderPlay(g, s, ctx);
    if (s.mode === 'check1') {
      const th = ctx.instant ? 60 * DEG : 120 * DEG * (0.5 - 0.5 * Math.cos(ctx.time * 1.3));
      scene(g, N.rot3(TILT, th), {});
      g.lineThrough([0, 0, 0], TILT, { color: 'gold', width: 2.6, len: 3.2 });
      g.point(N.scale(TILT, 1.25), { r: 6, color: 'gold' });
      return;
    }
    scene(g, M, { floor: s.floor, halo: s.halo, thick: s.thick, pulse: s.pulse, coords: s.coords, boxAlpha: s.boxAlpha, edgeAlpha: s.edgeAlpha, lattice: s.lattice });
  }

  function renderOrder(g, s) {
    const u1 = seg(s.m, 0, 1), u2 = seg(s.m, 1, 2);
    const L = orderM('x', 'z', u1, u2), Rr = orderM('z', 'x', u1, u2);
    const pane = (M, first, second, title, g2, r) => {
      scene(g2, M, { sticker: true, axes: true });
      if (s.m >= 1.99) hopPath(g2, first, second, 0.6);
      g2.hud(title, r.x + 12, r.y + 20, { color: 'hud' });
      hudMat3(g2, r2(M), r.x + r.w / 2 - 35, r.y + r.h - 98, { name: first === 'x' ? 'Rz·Rx =' : 'Rx·Rz =' });
    };
    g.split((g2, r) => pane(L, 'x', 'z', 'about î first, then k̂', g2, r), (g2, r) => {
      pane(Rr, 'z', 'x', 'about k̂ first, then î', g2, r);
      if (s.cross) { g2.label('same result?', [r.x + r.w / 2 - 20, r.y + 58], { screen: true, color: 'bad', size: 17 }); g2.marker.cross([r.x + r.w / 2 + 52, r.y + 58], { screen: true, size: 24 }); }
    }, { left: { view: '3d', cam: s.cam }, right: { view: '3d', cam: s.cam } });
  }

  function renderML(g, s) {
    const R = N.rot3(TILT, 60 * DEG * s.mlT), vs = WORDS.map(w => N.matVec(R, w[1]));
    const C = cosM(vs); let delta = 0; for (let i = 0; i < 10; i++) for (let j = 0; j < 10; j++) delta = Math.max(delta, Math.abs(C[i][j] - COS0[i][j]));
    g.viewport({ x: 0, y: 0, w: 0.6, h: 1 }, { view: '3d', cam: s.cam }, (g2, r) => {
      g2.cage({ range: 2, alpha: 0.45 });
      g2.lineThrough([0, 0, 0], TILT, { color: 'gold', width: 2, len: 3, alpha: 0.8 });
      // apple/pear and invoice/tax nearly coincide: nudge one label of each pair so both stay readable
      const nudge = { pear: [-6, 18], tax: [-6, 18] };
      WORDS.forEach((w, i) => { const n = nudge[w[0]]; g2.vector([0, 0, 0], vs[i], { color: w[2], label: w[0], width: 2.2, head: 9, labelSize: 12, dx: n ? n[0] : undefined, dy: n ? n[1] : undefined }); });
      g2.hud('synthetic word vectors', r.x + 12, r.y + 20, { color: 'muted' });
      g2.hud('rotating the whole space by ' + Math.round(60 * s.mlT) + '° about the gold axis', r.x + 12, r.y + 40, { color: 'hud' });
      g2.hud('Part 6: embeddings', r.x + 12, r.y + r.h - 22, { color: 'muted' });
    });
    const aspect = (g.size.w * 0.4) / g.size.h;
    g.viewport({ x: 0.6, y: 0, w: 0.4, h: 1 }, { view: '2d', cam: { center: [-0.12, 0], halfH: 0.73 / aspect } }, (g2, r) => {
      const b = { x0: -0.48, x1: 0.52, y0: -0.5, y1: 0.5 };
      g2.image(C, b, { ramp: t => LA.color.diverge(2 * t - 1), range: [-1, 1], cells: true });
      WORDS.forEach((w, i) => g2.label(w[0], [b.x0, b.y1 - (b.y1 - b.y0) * (i + 0.5) / 10], { anchor: 'right', dx: -4, size: 10.5, color: w[2], font: 'ui', bg: false }));
      g2.hud('cosine of every pair', r.x + 12, r.y + 20, { color: 'muted' });
      g2.hud('Δ = ' + LA.fmt.fixed(delta, 2), r.x + 12, r.y + 40, { color: 'ok', size: 15 });
      g2.hud('cat–kitten: ' + LA.fmt.fixed(C[0][1], 2), r.x + 12, r.y + r.h - 22, { color: 'data' });
    });
  }

  function renderRecap(g, s) {
    const c = (y) => Object.assign(cam0(), { dist: 7.2 });
    g.viewport({ x: 0, y: 0, w: 1, h: 1 / 3 }, { view: '3d', cam: Object.assign(cam0(), { dist: 4.6, target: [0.8, 0.6, 0.65] }) }, (g2, r) => { scene(g2, GEN, { thick: true, coords: true, pulse: 'all', axes: false }); g2.hud('1 · three edges from the origin = three columns', r.x + 10, r.y + 16, { color: 'muted' }); });
    g.viewport({ x: 0, y: 1 / 3, w: 1, h: 1 / 3 }, { view: '3d', cam: Object.assign(c(), { yaw: -30 * DEG }) }, (g2, r) => {
      scene(g2, N.rot3(TILT, 80 * DEG), { axes: false });
      g2.lineThrough([0, 0, 0], TILT, { color: 'gold', width: 2.4, len: 3 });
      g2.hud('2 · the axis is the line that does not move', r.x + 10, r.y + 16, { color: 'muted' });
    });
    g.viewport({ x: 0, y: 2 / 3, w: 0.5, h: 1 / 3 }, { view: '3d', cam: c() }, (g2, r) => { scene(g2, orderM('x', 'z', 1, 1), { sticker: true, axes: false }); g2.hud('3 · î first, then k̂', r.x + 10, r.y + 16, { color: 'muted' }); });
    g.viewport({ x: 0.5, y: 2 / 3, w: 0.5, h: 1 / 3 }, { view: '3d', cam: c() }, (g2, r) => { scene(g2, orderM('z', 'x', 1, 1), { sticker: true, axes: false }); g2.hud('k̂ first, then î: different', r.x + 10, r.y + 16, { color: 'bad' }); });
  }

  function playM(s) {
    const Rx = N.rotX(s.thetaX), Ry = N.rotY(s.thetaY), Rz = N.rotZ(s.thetaZ);
    const R = s.order === 'xyz' ? N.matMul(Rz, N.matMul(Ry, Rx)) : N.matMul(Rx, N.matMul(Ry, Rz));
    return N.matMul(R, s.base);
  }
  function renderPlay(g, s) {
    const M = playM(s);
    scene(g, M, { lattice: s.showLattice });
    const RtR = N.matMul(N.transpose(M), M), isRot = N.frobenius(N.subM(RtR, I3())) < 1e-6 && N.det(M) > 0;
    const ax = isRot ? N.rotationAxis(M) : null;
    if (ax) { g.lineThrough([0, 0, 0], ax, { color: 'gold', width: 2.4, len: 3.2 }); g.hud('gold: the axis of this rotation (rotationAxis)', 14, 22, { color: 'gold' }); }
    else g.hud(isRot ? 'identity: nothing moves' : 'not a rotation: no axis to show', 14, 22, { color: 'muted' });
  }

  const CARD = { x: { name: 'rotate 90° about î', ax: 'x' }, z: { name: 'rotate 90° about k̂', ax: 'z' } };
  function renderCheck3(g, s) {
    const a = CARD[s.cards[0]].ax, b = CARD[s.cards[1]].ax;
    const M = orderM(a, b, 1, 1);
    scene(g, M, { boxAlpha: 0.08, axes: true });
    hopPath(g, a, b, 0.8);
    g.marker.ring([0, 0, 1], { r: 11, color: 'hud' }); g.label('target', [0, 0, 1], { color: 'muted', size: 12, dx: -44, dy: 2 });
    g.viewport({ x: 0, y: 0, w: 1, h: 0.17 }, { view: '2d', cam: { center: [0, 0], halfH: 1 }, bg: 'bg' }, (g2, r) => {
      [[-1.9, 'first'], [1.9, 'second']].forEach((sl, k) => {
        const c = [sl[0], -0.1], w = 1.7, h = 0.42, name = s.cards[k];
        g2.polygon([[c[0] - w, c[1] - h], [c[0] + w, c[1] - h], [c[0] + w, c[1] + h], [c[0] - w, c[1] + h]], { fill: 'teal', alpha: 0.14, stroke: 'teal', width: 1.4 });
        g2.label(CARD[name].name, [c[0] - 0.22, c[1]], { color: 'text', size: 14, bg: false });
        g2.label(sl[1], [c[0], c[1] + h], { color: 'muted', size: 11.5, anchor: 'above', dy: -2 });
        g2.handle('card-' + name, [c[0] + w - 0.3, c[1]], (p, st) => { const want = p[0] < 0 ? 0 : 1; if (st.cards[want] !== name) st.cards = want === 0 ? [name, st.cards[0]] : [st.cards[1], name]; }, { color: 'teal', r: 18 });
      });
    });
    g.hud('drag a card onto the other slot to swap the order', 14, g.size.h - 22, { color: 'muted' });
  }

  LA.registerLesson({
    id: 'p2-05-3d-transformations',
    part: 2,
    title: '3×3 matrices, 3D rotations, rotation order',
    subtitle: 'Three landing spots make a 3×3 matrix; 3D rotations have an axis and do not commute.',
    minutes: 9,
    prereqs: ['p2-04-composition'],
    tags: ['core'],
    badge: { drawnIn: '3D', realSize: 'a 3D rotation is a 3×3 matrix in graphics and robotics; orthogonal 768×768 matrices rotate embedding spaces', transfers: 'columns = landing spots and composition = product hold for every size; “a rotation has an axis” is true only in 3D — in n-D a rotation fixes a plane, not a line' },
    cantShow: ['a rotation in ℝ⁴ or higher (it has no axis; step 4 says so)'],
    terms: [
      { term: '3×3 matrix', def: 'Three columns: the landing spots of î, ĵ and k̂.' },
      { term: 'axis of rotation', def: 'The line left fixed by a 3D rotation; everything turns around it.' },
      { term: 'rotation order', def: 'Which rotation is applied first; in 3D the result depends on it.' },
      { term: 'parallelepiped', def: 'The slanted box that the unit cube becomes under a 3×3 matrix.' },
    ],
    atlas: ['embeddings', 'tensors'],
    thumbnail: 3,

    state() {
      return {
        view: '3d', cam: cam0(),
        mode: 'main', mathMode: null, M3: I3(), theta: 0, floor: false, halo: null, thick: false, pulse: null, coords: false,
        boxAlpha: 0.13, edgeAlpha: 0.9, lattice: true, m: 0, cross: false, ghostBox: null, mlT: 0,
        thetaX: 0, thetaY: 0, thetaZ: 0, order: 'xyz', base: I3(), showLattice: true, cards: ['x', 'z'],
      };
    },

    render(g, s, ctx) {
      if (s.mode === 'order') return renderOrder(g, s);
      if (s.mode === 'ml') return renderML(g, s);
      if (s.mode === 'recap') return renderRecap(g, s);
      renderMain(g, s, ctx);
    },

    math(s, tex) {
      const eq = tex.op('='), mat = (X, o) => tex.mat(r2(X), Object.assign({ colors: 'cols' }, o || {}));
      const sym = (n, sub) => tex.sym(n, sub ? { sub } : undefined);
      switch (s.mathMode) {
        case 'tall': return tex.block(tex.eq(sym('M'), eq, mat(s.M3, { highlight: (i, j) => j === 2 })), tex.eq(tex.khat(), tex.op('↦'), tex.vec(r2([N.col(s.M3, 2)])[0], { color: 'k' })));
        case 'rz': return tex.block(tex.eq(sym('R', 'z') + tex.paren(tex.sym('θ')), eq, tex.mat([['cos θ', '−sin θ', '0'], ['sin θ', 'cos θ', '0'], ['0', '0', '1']], { colors: 'cols', highlight: (i, j) => i < 2 && j < 2 })),
          tex.eq(tex.sym('θ'), eq, tn(tex, Math.round(s.theta / DEG) + '°', 'gold'), tex.op(':'), mat(s.M3, { highlight: (i, j) => i < 2 && j < 2 })));
        case 'rx': return tex.block(tex.eq(sym('R', 'x') + tex.paren(tex.sym('θ')), eq, tex.mat([['1', '0', '0'], ['0', 'cos θ', '−sin θ'], ['0', 'sin θ', 'cos θ']], { colors: 'cols', highlight: (i, j) => i > 0 && j > 0 })),
          tex.eq(tex.sym('θ'), eq, tn(tex, Math.round(s.theta / DEG) + '°', 'gold'), tex.op(':'), mat(s.M3, { highlight: (i, j) => i > 0 && j > 0 })));
        case 'tilt': return tex.block(tex.eq(sym('R', 'axis') + tex.paren(tn(tex, Math.round(s.theta / DEG) + '°', 'gold')), eq, mat(s.M3)), tex.text(s.theta > 119 * DEG ? 'a permutation of the columns' : 'axis [1; 1; 1] stays fixed'));
        case 'order': return tex.block(tex.eq(sym('R', 'z') + sym('R', 'x'), eq, mat(orderM('x', 'z', 1, 1))), tex.op('≠'), tex.eq(sym('R', 'x') + sym('R', 'z'), eq, mat(orderM('z', 'x', 1, 1))));
        case 'shsc': return tex.block(tex.eq(sym('M'), eq, mat(s.M3)), tex.text(s.M3[0][2] > 0.01 && Math.abs(s.M3[0][0] - 1) < 0.01 ? 'shear: k̂ leans to [0.5; 0; 1]' : 'scaling: a diagonal'));
        case 'edges': return tex.block(tex.eq(sym('M'), eq, tex.op('['), tex.text('col₁'), tex.text('col₂'), tex.text('col₃'), tex.op(']'), eq, mat(s.M3)));
        case 'compose': { const P = N.matMul(SH3, N.rotZ(Math.PI / 4)); return tex.block(tex.eq(sym('M', '2') + sym('M', '1'), eq, sym('H') + tex.op('·') + sym('R', 'z') + tex.paren(tn(tex, '45°')), eq, mat(P))); }
        case 'ml': { const R = N.rot3(TILT, 60 * DEG * s.mlT), a = N.matVec(R, WORDS[0][1]), b = N.matVec(R, WORDS[1][1]); return tex.block(tex.eq(tex.paren(tex.sym('R') + tex.sym('u')) + tex.op('·') + tex.paren(tex.sym('R') + tex.sym('v')), eq, tex.sym('u') + tex.op('·') + tex.sym('v')), tex.eq(tex.text('cos(cat, kitten)'), eq, tn(tex, LA.fmt.fixed(COS0[0][1], 2), 'data'), tex.op('→'), tn(tex, LA.fmt.fixed(N.dot(a, b) / (N.norm(a) * N.norm(b)), 2), 'data'))); }
        case 'recap': return '';
        case 'check3': { const P = orderM(CARD[s.cards[0]].ax, CARD[s.cards[1]].ax, 1, 1); return tex.block(tex.eq(tex.text('second · first'), eq, mat(P)), tex.eq(tex.ihat(), tex.op('→'), tex.vec(r2([N.col(P, 0)])[0], { color: 'i' }))); }
        case 'play': { const M = playM(s); return tex.block(tex.eq(sym('M'), eq, mat(M)), tex.text(s.order === 'xyz' ? 'order: about x, then y, then z' : 'order: about z, then y, then x')); }
        default: return tex.block(tex.eq(sym('M'), eq, mat(s.M3)));
      }
    },

    steps: [
      { title: 'Three landing spots', duration: 4,
        narration: 'The unit cube with the green ' + ihat + ', red ' + jhat + ' and blue ' + khat + '. A <b>3×3 matrix</b> records three landing spots. Here only ' + khat + ' moves: it stretches to [0; 0; 2], so the cube becomes a tall box and the blue third column reads [0; 0; 2]. Everything else from 2D carries over unchanged.',
        setup(s) { s.mode = 'main'; s.mathMode = 'tall'; s.view = '3d'; s.cam = cam0(); s.M3 = I3(); s.floor = false; s.halo = null; s.thick = false; s.pulse = null; s.coords = false; s.boxAlpha = 0.13; s.edgeAlpha = 0.9; s.lattice = true; s.cross = false; s.ghostBox = null; },
        at(s, t) { s.M3 = lerpM(I3(), TALL, seg(t, 0.1, 0.7)); s.pulse = t > 0.72 ? 2 : null; s.coords = t > 0.72; } },

      { title: 'Rotation about k̂', duration: 4.5,
        narration: 'Rotate about the blue axis by θ. The green ' + ihat + ' and red ' + jhat + ' turn in the floor exactly as in 2D — the 2D rotation block fills the top-left corner — while ' + khat + ' stays put, giving the third column [0; 0; 1]. The blue line is the <b>axis of rotation</b>: the one line that does not move.',
        setup(s) { s.mathMode = 'rz'; s.M3 = I3(); s.theta = 0; s.pulse = null; s.coords = false; s.halo = 2; s.floor = true; s.lattice = false; },
        at(s, t) { s.theta = 90 * DEG * seg(t, 0.05, 0.8); s.M3 = N.rotZ(s.theta); } },

      { title: 'Rotation about î', duration: 4.5,
        narration: 'Rotate about the green axis instead: now the red ' + jhat + ' and blue ' + khat + ' turn in the wall plane while ' + ihat + ' stays. The 2D block moves to the lower-right corner, and column one is [1; 0; 0]. Same picture, different fixed line.',
        setup(s) { s.mode = 'yz'; s.mathMode = 'rx'; s.M3 = I3(); s.theta = 0; s.floor = false; s.halo = null; s.cam = cam0(); },
        at(s, t) { s.theta = 90 * DEG * seg(t, 0.1, 0.85); s.M3 = N.rotX(s.theta); LA.cam.tween(s, { yaw: 35 * DEG }, { yaw: 15 * DEG }, seg(t, 0, 1, 'inOut')); } },

      { title: 'Any line can be the axis', duration: 5,
        narration: 'The axis need not be a coordinate axis. Rotate by 120° about the gold line through [1; 1; 1]: ' + ihat + ' lands where ' + jhat + ' was, ' + jhat + ' where ' + khat + ' was, ' + khat + ' where ' + ihat + ' was. The gold line itself does not move. Only in 3D does every rotation have an axis; in higher dimensions a rotation fixes a plane instead.',
        setup(s) { s.mode = 'tilted'; s.mathMode = 'tilt'; s.M3 = I3(); s.theta = 0; s.cam = Object.assign(cam0(), { yaw: -5 * DEG }); },
        // view the gold axis side-on (from yaw ≈ 45° it points almost straight at the camera and looks like a stub)
        at(s, t) { s.theta = 120 * DEG * seg(t, 0.05, 0.85); s.M3 = N.rot3(TILT, s.theta); LA.cam.tween(s, { yaw: -5 * DEG }, { yaw: -30 * DEG }, seg(t, 0, 1, 'inOut')); } },

      { title: '3D rotations do not commute', duration: 6.5, misconception: true,
        narration: 'Belief: “rotations can be done in any order.” Left: rotate 90° about ' + ihat + ', then 90° about ' + khat + '. Right: the same two rotations in the other order. The cubes end up in different positions — the red sticker faces different ways, and ' + ihat + ' lands in different places. In 3D, <b>rotation order</b> matters.',
        setup(s) { s.mode = 'order'; s.mathMode = 'order'; s.m = 0; s.cross = false; s.cam = Object.assign(cam0(), { dist: 11.5, target: [0.1, 0.1, 0.25] }); },
        at(s, t) { s.m = seg(t, 0.02, 0.4) + seg(t, 0.44, 0.82); s.cross = t > 0.86; } },

      { title: 'Shear and scale in 3D', duration: 4.5,
        narration: 'Other animals have 3D versions. A shear leans the blue ' + khat + ' over to [0.5; 0; 1] and the cube becomes a slanted box — a <b>parallelepiped</b>. A scaling stretches each axis by its own factor. Columns still tell you everything: read them off the three edges leaving the origin.',
        setup(s) { s.mode = 'main'; s.mathMode = 'shsc'; s.cross = false; s.M3 = I3(); s.cam = cam0(); s.halo = null; s.floor = false; s.thick = true; s.lattice = false; s.boxAlpha = 0.2; },
        at(s, t) { s.M3 = t < 0.5 ? lerpM(I3(), SH3, seg(t, 0.05, 0.45)) : lerpM(SH3, SC3, seg(t, 0.52, 0.9)); } },

      { title: 'The three edges are the columns', duration: 4,
        narration: 'Look at a slanted box: its three edges leaving the origin are exactly the three landed basis vectors — the columns. To read any 3×3 matrix from a picture, measure those three edges. To build a box from a matrix, draw its columns and complete the parallelogram faces.',
        setup(s) { s.mathMode = 'edges'; s.M3 = LA.clone(SC3); s.thick = true; s.pulse = null; s.coords = false; s.edgeAlpha = 0.9; },
        at(s, t) { s.M3 = lerpM(SC3, GEN, seg(t, 0, 0.22)); s.edgeAlpha = LA.lerp(0.9, 0.3, seg(t, 0.2, 0.3)); s.boxAlpha = LA.lerp(0.2, 0.08, seg(t, 0.2, 0.3)); s.pulse = t < 0.28 ? null : t < 0.5 ? 0 : t < 0.72 ? 1 : t < 0.92 ? 2 : 'all'; s.coords = t >= 0.28; } },

      { title: 'Composition in 3D is the same product', duration: 5,
        narration: 'Two 3D maps in sequence multiply exactly as in 2D: first M₁, then M₂ is M₂M₁, read right to left, the columns of M₁ pushed through M₂. Here a 45° rotation about the blue ' + khat + ' followed by the shear H gives a tilted, turned box — one matrix.',
        setup(s) { s.mode = 'compose'; s.mathMode = 'compose'; s.M3 = I3(); s.thick = false; s.pulse = null; s.coords = false; s.edgeAlpha = 0.9; s.boxAlpha = 0.13; s.ghostBox = null; },
        at(s, t) {
          const R45 = N.rotZ(Math.PI / 4 * seg(t, 0, 0.36)), P = N.matMul(SH3, N.rotZ(Math.PI / 4));
          if (t < 0.4) s.M3 = R45; else if (t < 0.8) s.M3 = N.matMul(lerpM(I3(), SH3, seg(t, 0.4, 0.76)), N.rotZ(Math.PI / 4)); else s.M3 = lerpM(I3(), P, seg(t, 0.82, 1));
          s.ghostBox = t >= 0.8 ? P : (t >= 0.4 ? N.rotZ(Math.PI / 4) : null);
        } },

      { title: 'Rotating a whole embedding space changes nothing', duration: 5.5, mlTie: true,
        narration: 'Ten synthetic word vectors in 3D. Rotate the entire space by a 3×3 rotation: every dot product, every length, every angle between words is unchanged — cat is still closest to kitten. Embedding spaces are meaningful only up to such rotations, which is why two models’ embeddings can look different yet agree. Part 6 uses this.',
        setup(s) { s.mode = 'ml'; s.mathMode = 'ml'; s.ghostBox = null; s.mlT = 0; s.cam = Object.assign(cam0(), { dist: 11, yaw: 20 * DEG, target: [0.15, 0.35, 0.45] }); },
        at(s, t) { s.mlT = seg(t, 0.03, 0.7); } },

      { title: 'Recap', duration: 2, recap: true,
        narration: 'A 3×3 matrix is three landing spots; the box’s edges from the origin are its columns. Rotations about ' + ihat + ', ' + jhat + ', ' + khat + ' put the 2D block in different corners; any line can be an axis, and the axis is what stays fixed. Order of rotations matters. Composition is the same right-to-left product.',
        cards: ['<b>3 columns = 3 edges</b> of the box leaving the origin.', '<b>The axis</b> is the line a 3D rotation leaves fixed (only 3D has one).', '<b>Order matters in 3D:</b> R<sub>z</sub>R<sub>x</sub> ≠ R<sub>x</sub>R<sub>z</sub>.'],
        setup(s) { s.mode = 'recap'; s.mathMode = 'recap'; },
        at(s, t) { } },
    ],

    playground: {
      intro: '<p>Turn the cube about each axis. The order control decides which turn happens first. The gold line is the axis of the combined rotation, found with <i>rotationAxis</i>. Drag to orbit.</p>',
      setup(s) { s.mode = 'play'; s.mathMode = 'play'; s.view = '3d'; s.cam = cam0(); s.thetaX = 0.6; s.thetaY = 0; s.thetaZ = 0.8; s.order = 'xyz'; s.base = I3(); s.showLattice = true; },
      controls: [
        { type: 'slider', bind: 'thetaX', min: 0, max: 6.283, step: 0.01, label: 'rotate about î', color: 'i', fmt: x => LA.fmt.deg(x) },
        { type: 'slider', bind: 'thetaY', min: 0, max: 6.283, step: 0.01, label: 'rotate about ĵ', color: 'j', fmt: x => LA.fmt.deg(x) },
        { type: 'slider', bind: 'thetaZ', min: 0, max: 6.283, step: 0.01, label: 'rotate about k̂', color: 'k', fmt: x => LA.fmt.deg(x) },
        { type: 'preset', label: 'Order', items: [
          { label: 'x → y → z', apply(s) { s.order = 'xyz'; } },
          { label: 'z → y → x', apply(s) { s.order = 'zyx'; } },
        ] },
        { type: 'toggle', bind: 'showLattice', label: 'Lattice' },
        { type: 'preset', label: 'Presets', items: [
          { label: 'Identity', apply(s) { s.base = I3(); s.thetaX = 0; s.thetaY = 0; s.thetaZ = 0; } },
          { label: 'Tilted axis 120°', apply(s) { s.base = r2(N.rot3(TILT, 120 * DEG)); s.thetaX = 0; s.thetaY = 0; s.thetaZ = 0; } },
          { label: 'Shear k̂', apply(s) { s.base = LA.clone(SH3); s.thetaX = 0; s.thetaY = 0; s.thetaZ = 0; } },
          { label: 'Two orders', apply(s) { s.base = I3(); s.thetaX = Math.PI / 2; s.thetaY = 0; s.thetaZ = Math.PI / 2; } },
        ] },
      ],
    },

    checks: [
      { kind: 'manipulate',
        prompt: 'Orbit until you look straight down the gold axis of this rotation.',
        setup(s) { s.mode = 'check1'; s.mathMode = 'recap';s.M3 = r2(N.rot3(TILT, 120 * DEG)); s.view = '3d'; s.cam = Object.assign(cam0(), { yaw: 20 * DEG, pitch: 10 * DEG }); },
        judge(s) { return LA.judge.lookAlong(s.cam, TILT); },
        hint: 'From the right viewpoint the gold line shrinks to a single point and the cube spins like a wheel.',
        solution(s) { s.cam = LA.cam.lookAlong(s.cam, TILT); } },
      { kind: 'entry',
        prompt: 'Type the matrix that sends ' + khat + ' to −' + khat + ' while ' + ihat + ' and ' + jhat + ' stay.',
        bind: 'M3', answer: [[1, 0, 0], [0, 1, 0], [0, 0, -1]], tol: LA.TOL.entry, label: 'M', colors: 'cols',
        setup(s) { s.mode = 'main'; s.mathMode = null; s.M3 = I3(); s.view = '3d'; s.cam = cam0(); s.halo = null; s.floor = false; s.thick = false; s.pulse = null; s.coords = false; s.lattice = false; } },
      { kind: 'manipulate',
        prompt: 'Order the two cards so that ' + ihat + ' ends at the marked spot [0; 0; 1].',
        setup(s) { s.mode = 'check3'; s.mathMode = 'check3'; s.cards = ['x', 'z']; s.view = '3d'; s.cam = Object.assign(cam0(), { dist: 9, target: [0.3, 0.3, 0.95] }); },
        judge(s) { const ok = s.cards[0] === 'z' && s.cards[1] === 'x'; return { ok, progress: ok ? 1 : 0 }; },
        hint: 'The first rotation must move ' + ihat + ' off the green axis; then the second can lift it.',
        solution(s) { s.cards = ['z', 'x']; } },
    ],
  });
})();
