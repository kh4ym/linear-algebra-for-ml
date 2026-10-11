/* p2-09 — Non-square matrices change dimension. */
(function () {
  'use strict';
  const N = LA.num, lerpM = LA.lerpM, lerpV = LA.lerpV, seg = LA.seg;
  const I2 = () => [[1, 0], [0, 1]];
  const I3 = () => [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  const B0 = [[1, 0, 0.5], [0, 1, 0.5]], PZ = [[1, 0, 0], [0, 1, 0]], L0 = [[1, 0], [0, 1], [0.5, 0.5]];
  const BL = N.matMul(B0, L0);
  const NULLB = [-1, -1, 2];
  const ihat = '<span class="c-i">î</span>', jhat = '<span class="c-j">ĵ</span>', khat = '<span class="c-k">k̂</span>';
  const cam2 = (cx, cy, hh) => Object.assign(LA.cam.defaults2d(), { center: [cx, cy], halfH: hh });
  const cam3 = (o) => Object.assign(LA.cam.defaults3d(), o);
  const LEFT3 = () => cam3({ dist: 7.2, yaw: 0.6, pitch: 0.42, target: [0.3, 0.3, 0.35] });
  const RIGHT2 = () => cam2(0.6, 0.5, 2.7);
  const CUBE = []; for (let a = 0; a <= 1; a++) for (let b = 0; b <= 1; b++) for (let c = 0; c <= 1; c++) CUBE.push([a, b, c]);
  const LAT3 = []; for (let a = 0; a <= 1; a += 0.5) for (let b = 0; b <= 1; b += 0.5) for (let c = 0; c <= 1; c += 0.5) LAT3.push([a, b, c]);
  const LAT2 = []; for (let a = -2; a <= 2; a++) for (let b = -2; b <= 2; b++) LAT2.push([a, b]);
  const EDGES = [[0, 1], [2, 3], [4, 5], [6, 7], [0, 2], [1, 3], [4, 6], [5, 7], [0, 4], [1, 5], [2, 6], [3, 7]];
  const OUT10 = [0.02, 0.01, 0.03, 0.05, 0.02, 0.04, 0.01, 0.74, 0.05, 0.03];
  const TILES = [{ r: 3, c: 2 }, { r: 2, c: 4 }, { r: 4, c: 1 }];

  function makeData() {
    const dig = LA.rng.syntheticDigits(9, 8, { size: 28 })[7].pixels.map(x => Math.round(x * 100) / 100);
    const r = LA.rng(64), h = [];
    for (let i = 0; i < 128; i++) { let sum = 0; for (let j = 0; j < 784; j += 1) { const w = (r() - 0.5) * 0.18; sum += w * dig[j]; } h.push(Math.round(sum * 100) / 100); }
    const r2 = LA.rng(65), heat = [];
    for (let i = 0; i < 8; i++) { heat.push([]); for (let j = 0; j < 24; j++) heat[i].push(Math.round(LA.rng.gauss(r2) * 100) / 100); }
    return { dig, h, heat };
  }

  /* ---------- small drawing helpers (2D world coordinates) ---------- */
  // r×c tile of cells centred at (cx, cy)
  function tile(g, r, c, cx, cy, cell, o) {
    o = o || {};
    const w = c * cell, h = r * cell, x0 = cx - w / 2, y0 = cy - h / 2, al = o.alpha === undefined ? 1 : o.alpha;
    if (al <= 0) return;
    const col = o.color || 'teal';
    g.polygon([[x0, y0], [x0 + w, y0], [x0 + w, y0 + h], [x0, y0 + h]], { fill: col, alpha: 0.22 * al, stroke: col, width: 1.6 });
    for (let i = 1; i < r; i++) g.line([x0, y0 + i * cell], [x0 + w, y0 + i * cell], { color: col, width: 0.8, alpha: 0.6 * al });
    for (let j = 1; j < c; j++) g.line([x0 + j * cell, y0], [x0 + j * cell, y0 + h], { color: col, width: 0.8, alpha: 0.6 * al });
    if (o.label !== false) g.label(o.label || (r + '×' + c), [cx, y0], { anchor: 'below', dy: 8, color: o.labelColor || 'text', size: o.size || 15, font: 'ui', alpha: al });
  }
  // a column vector drawn as n stacked cells centred at (cx, cy)
  function strip(g, n, cx, cy, cell, color, label, al) {
    if (al === undefined) al = 1; if (al <= 0) return;
    const h = n * cell, y0 = cy - h / 2;
    for (let i = 0; i < n; i++) g.polygon([[cx - cell / 2, y0 + i * cell], [cx + cell / 2, y0 + i * cell], [cx + cell / 2, y0 + (i + 1) * cell], [cx - cell / 2, y0 + (i + 1) * cell]], { fill: color, alpha: 0.45 * al, stroke: color, width: 1.2 });
    if (label) g.label(label, [cx, y0], { anchor: 'below', dy: 8, color, size: 13, font: 'ui', alpha: al });
  }
  function claimLine(g, s) {
    if (!s.claim) return;
    const w = g.size.w;
    g.label(s.claim.text, [w / 2, 26], { screen: true, size: 15, color: 'hud', font: 'ui' });
    if (s.claim.cross) g.marker.cross([w / 2 + s.claim.text.length * 3.7 + 22, 26], { screen: true, size: 22 });
  }

  LA.registerLesson({
    id: 'p2-09-non-square',
    part: 2,
    title: 'Non-square matrices change dimension',
    subtitle: 'A 2×3 flattens space like a shadow; a 3×2 lifts a plane into the room.',
    minutes: 9,
    prereqs: ['p2-08-rank-column-null'],
    tags: ['core'],
    badge: { drawnIn: '3D → 2D and 2D → 3D', realSize: 'layers 784 → 128 → 10 are 128×784 and 10×128 matrices', transfers: 'rows = output size, columns = input size, and the chaining rule hold for every shape; only shapes with m, n ≤ 3 can be drawn' },
    cantShow: ['a 128×784 map as a grid motion (we show 3 → 2 and 2 → 3 and chain the shape tiles)'],
    terms: [
      { term: 'm×n matrix', def: 'A matrix with m rows (the output size) and n columns (the input size): it takes n-vectors to m-vectors.' },
      { term: 'wide matrix', def: 'More columns than rows: it reduces dimension and must crush some inputs to zero.' },
      { term: 'tall matrix', def: 'More rows than columns: it places a small space as a flat subspace inside a bigger one.' },
      { term: 'shape chaining', def: '(m×n)(n×p) is allowed and gives m×p: the inner numbers must match.' },
    ],
    atlas: ['mlp', 'tensors'],
    thumbnail: 1,

    state() {
      const d = makeData();
      return {
        view: '2d', cam: cam2(0, 0.2, 2.4), scene: 'shape1',
        B: LA.clone(B0), L: LA.clone(L0), Bm: LA.clone(PZ), m1: 0, dimLeft: 0, hiK: false, link: 0, nullT: -1,
        liftT: 0, plane: 0, compT: 0, ft: 0, claim: null, panel: 'shape',
        dig: d.dig, h: d.h, heat: d.heat,
        pg: false, chk: null, mode: 'flatten', v3: [1, 0, 0], tx: [-2, 0, 2],
      };
    },

    render(g, s, ctx) {
      const sc = (s.pg && ctx.mode === 'play') ? 'play' : (s.chk ? 'chk' : s.scene);
      if (sc === 'play') { renderPlay(g, s); return; }
      if (sc === 'chk') { renderCheck(g, s); return; }
      if (sc === 'shape1') { renderShape1(g, s); return; }
      if (sc === 'flatten') { renderFlatten(g, s, {}); return; }
      if (sc === 'lift') { renderLift(g, s); return; }
      if (sc === 'wrong') { renderWrong(g, s); return; }
      if (sc === 'chain') { renderChain(g, s); return; }
      if (sc === 'compose') { renderCompose(g, s); return; }
      if (sc === 'big') { renderBig(g, s); return; }
      if (sc === 'ml') { renderML(g, s); return; }
      if (sc === 'recap') { renderRecap(g, s); return; }
    },

    math(s, tex) {
      const num = (x, o) => '<span class="tx-n ' + tex.colorClass(o && o.color) + '">' + (typeof x === 'number' ? LA.fmt(x) : x) + '</span>';
      const T = (w, c) => tex.text(w, { color: c });
      const P = [], p = s.panel;
      const cols3 = ['i', 'j', 'k'];
      if (p === 'shape') { P.push(tex.eq(tex.sym('B'), tex.op(':'), tex.dim(2, 3))); P.push(tex.eq(tex.sym('B'), tex.sym('x', { color: 'v' }), tex.op(':'), T('(2×3)(3×1) = (2×1)'))); }
      else if (p === 'B') { P.push(tex.eq(tex.sym('B'), tex.op('='), tex.mat(B0, { colors: cols3 }), tex.dim(2, 3))); }
      else if (p === 'Bm') { P.push(tex.eq(T('shadow matrix'), tex.op('='), tex.mat(s.Bm, { colors: cols3 }))); }
      else if (p === 'k') { P.push(tex.eq(tex.sym('B'), tex.op('='), tex.mat(B0, { colors: cols3, highlight: (i, j) => j === 2 }))); P.push(tex.eq(tex.khat(), tex.op('↦'), tex.vec([0.5, 0.5], { color: 'k' }))); }
      else if (p === 'null') { P.push(tex.eq(tex.mat(B0, { colors: cols3 }), tex.vec(NULLB, { color: 'null' }), tex.op('='), tex.vec([0, 0]))); P.push(tex.eq(T('rank', 'teal'), num(2, { color: 'teal' }), tex.op('+'), T('nullity', 'null'), num(1, { color: 'null' }), tex.op('='), num(3), T('inputs'))); }
      else if (p === 'L') { P.push(tex.eq(tex.sym('L'), tex.op('='), tex.mat(L0, { colors: 'cols' }), tex.dim(3, 2))); P.push(tex.eq(T('Col(L) = tilted plane z = 0.5x + 0.5y · rank 2', 'teal'))); }
      else if (p === 'wrong') { P.push(tex.eq(T('(3×2)(2×1) = (3×1)'), num('✓', { color: 'ok' }))); P.push(tex.eq(T('(3×2)(3×1)'), num('✕', { color: 'bad' }), T('inner 2 ≠ 3', 'bad'))); }
      else if (p === 'chain') { P.push(tex.eq(T('(3×2)(2×4)(4×1)'), tex.op('→'), T('(3×1)', 'gold'))); }
      else if (p === 'BL') { P.push(tex.eq(tex.sym('B'), tex.sym('L'), tex.op('='), tex.mat(B0, { colors: cols3 }), tex.mat(L0, { colors: 'cols' }), tex.op('='), tex.mat(BL, { colors: 'cols' }), tex.dim(2, 2))); }
      else if (p === 'big') { P.push(tex.eq(tex.sym('h'), tex.op('='), tex.sym('W'), tex.sym('x', { color: 'v' }), tex.op(':'), T('(128×784)(784×1) = (128×1)'))); P.push(tex.eq(T('nullity ≥ 784 − 128 ='), num(656, { color: 'null' }))); }
      else if (p === 'ml') { P.push(tex.eq(tex.sym('ŷ'), tex.op('='), tex.sym('W', { sub: '2' }), tex.sym('W', { sub: '1' }), tex.sym('x', { color: 'v' }), tex.op(':'), T('(10×128)(128×784)(784×1) → (10×1)'))); }
      else if (p === 'play') {
        if (s.mode === 'lift') P.push(tex.eq(tex.sym('L'), tex.op('='), tex.mat(s.L, { colors: 'cols' }), tex.dim(3, 2), T('rank'), num(N.rank(s.L, 1e-6), { color: 'teal' })));
        else if (s.mode === 'both') P.push(tex.eq(tex.sym('B'), tex.sym('L'), tex.op('='), tex.mat(N.matMul(s.B, s.L), { colors: 'cols' }), tex.dim(2, 2)));
        else { P.push(tex.eq(tex.sym('B'), tex.op('='), tex.mat(s.B, { colors: cols3 }), tex.dim(2, 3))); P.push(tex.eq(tex.sym('B'), tex.vec(s.v3, { color: 'v' }), tex.op('='), tex.vec(N.matVec(s.B, s.v3), { color: 'v' }))); }
      }
      else if (p === 'chk0') { P.push(tex.eq(tex.sym('B'), tex.op('='), tex.mat(s.B, { colors: cols3 }), tex.dim(2, 3))); }
      else if (p === 'chk1') { P.push(tex.eq(tex.mat(B0, { colors: cols3 }), tex.vec(s.v3, { color: 'v' }), tex.op('='), tex.vec(N.matVec(B0, s.v3)))); }
      else if (p === 'chk2') { const o = tileOrder(s); P.push(tex.eq(T(o.map(k => '(' + TILES[k].r + '×' + TILES[k].c + ')').join('')))); }
      return tex.block.apply(null, P);
    },

    steps: [
      { title: 'Rows out, columns in', duration: 3.5,
        narration: 'So far every matrix was square. An <b>m×n matrix</b> takes an n-vector in and gives an m-vector out: columns count inputs, rows count outputs. A 2×3 matrix eats a yellow 3D vector and spits out an orange 2D one. Shapes tell you the map’s story before any number does.',
        setup(s) { s.view = '2d'; s.scene = 'shape1'; s.cam = cam2(0, 0.2, 2.4); s.ft = 0; s.claim = null; s.panel = 'shape'; },
        at(s, t) { s.ft = t; } },

      { title: 'A 2×3 flattens the cube like a shadow', duration: 6,
        narration: 'The three columns of B = [[1, 0, 0.5],[0, 1, 0.5]] are where the green ' + ihat + ', red ' + jhat + ' and blue ' + khat + ' land — but each landing spot has only two coordinates. The cube is pressed flat into the plane, like a shadow cast by a lamp. Watch the 3D lattice collapse onto the 2D picture on the right.',
        setup(s) { s.view = '3d'; s.scene = 'flatten'; s.cam = LEFT3(); s.Bm = LA.clone(PZ); s.m1 = 0; s.dimLeft = 0; s.hiK = false; s.link = 0; s.nullT = -1; s.panel = 'Bm'; },
        at(s, t) { s.m1 = seg(t, 0.1, 0.45); const m2 = seg(t, 0.5, 0.85); s.Bm = lerpM(PZ, B0, m2); s.dimLeft = 0.5 * m2; LA.cam.tween(s, { yaw: 0.6 }, { yaw: 0.85 }, seg(t, 0, 1, 'linear')); } },

      { title: 'The blue column is two numbers tall', duration: 3.5,
        narration: 'Look at the blue column: ' + khat + ', which pointed out of the floor, lands at [0.5; 0.5] — a flat spot in the plane. The map had to put it somewhere in 2D. Every column of a 2×3 matrix is a 2D arrow; there are three of them because there are three inputs.',
        setup(s) { s.scene = 'flatten'; s.Bm = LA.clone(B0); s.m1 = 1; s.dimLeft = 0.5; s.hiK = true; s.link = 0; s.panel = 'k'; },
        at(s, t) { s.link = seg(t, 0.15, 0.6); } },

      { title: 'Something must be crushed', duration: 4.5,
        narration: 'Three input directions, two output directions: at least one input line must be sent to zero. For B it is the magenta line along [−1; −1; 2]: −1 + 0 + 1 = 0 in the first row and 0 − 1 + 1 = 0 in the second. Every <b>wide matrix</b> has a null space — its shadow hides a whole direction.',
        setup(s) { s.hiK = false; s.link = 0; s.nullT = 0; s.Bm = LA.clone(PZ); s.panel = 'null'; },
        at(s, t) { s.nullT = seg(t, 0.15, 0.75); s.Bm = lerpM(PZ, B0, s.nullT); LA.cam.tween(s, { yaw: 0.85 }, { yaw: 0.2 }, seg(t, 0, 0.6)); } },

      { title: 'A 3×2 lifts the plane into space', duration: 6,
        narration: 'Now the other way: L = [[1, 0],[0, 1],[0.5, 0.5]] has two columns, each three tall. The green ' + ihat + ' and red ' + jhat + ' land at [1; 0; 0.5] and [0; 1; 0.5], so the flat grid rises into a tilted teal plane — the column space of L. This <b>tall matrix</b> loses nothing; its outputs just do not fill the room.',
        setup(s) { s.view = '3d'; s.scene = 'lift'; s.cam = cam3({ dist: 16.5, yaw: 0.7, pitch: 0.38, target: [0, 0, 0.3] }); s.nullT = -1; s.liftT = 0; s.plane = 0; s.panel = 'L'; },
        at(s, t) { s.liftT = seg(t, 0.1, 0.65); s.plane = seg(t, 0.55, 0.8); LA.cam.tween(s, { yaw: 0.7 }, { yaw: 1.22 }, seg(t, 0, 1, 'linear')); } },

      { title: 'Columns are not outputs', duration: 3.5, misconception: true,
        narration: 'Belief: “the number of columns is the output size.” Try it on L: three outputs, but only two columns. The columns count what goes <i>in</i>; the length of each column — the number of rows — is what comes <i>out</i>. A tile that reads 3×2 is tall for a reason.',
        setup(s) { s.view = '2d'; s.scene = 'wrong'; s.cam = cam2(0, 0, 2.6); s.ft = 0; s.claim = null; s.panel = 'wrong'; },
        at(s, t) { s.ft = t; s.claim = { text: 'belief: “columns = output size”', cross: t > 0.6 }; } },

      { title: 'Chaining shapes', duration: 4.5,
        narration: '<b>Shape chaining</b>: maps chain when the inner numbers match, (3×2)(2×4)(4×1) → (3×1). Watch the inner 2s meet and vanish, then the inner 4s. Read the chain right to left as before: the 4-vector goes in, becomes a 2-vector, then a 3-vector. The outer numbers are all that remain.',
        setup(s) { s.view = '2d'; s.scene = 'chain'; s.cam = cam2(0, -0.2, 2.6); s.claim = null; s.ft = 0; s.panel = 'chain'; },
        at(s, t) { s.ft = t; } },

      { title: 'Flatten after lift: a square map', duration: 4.5,
        narration: 'Lift with L, then flatten with B: the composition BL is (2×3)(3×2) = 2×2 — a plane-to-plane map, [[1.25, 0.25],[0.25, 1.25]]. The plane went up into the room and came back down, slightly stretched. Non-square matrices compose exactly like square ones, as long as the shapes chain.',
        setup(s) { s.view = '2d'; s.scene = 'compose'; s.cam = cam2(0.3, 0.3, 2.8); s.compT = 0; s.panel = 'BL'; },
        at(s, t) { s.compT = seg(t, 0.15, 0.85); } },

      { title: '784 in, 128 out', duration: 4.5,
        narration: 'A real layer W is 128×784: 784 pixel inputs, 128 outputs. Its 784 columns are 128-tall landing spots we cannot draw, shown here as a heatmap patch. The flattening is the cube’s shadow again: 784 directions pressed into 128, with at least 656 crushed. The picture stops at 3. The formula does not.',
        setup(s) { s.view = '2d'; s.scene = 'big'; s.cam = cam2(0, 0, 2.6); s.ft = 0; s.panel = 'big'; },
        at(s, t) { s.ft = t; } },

      { title: 'Layers change width', duration: 5, mlTie: true,
        narration: 'A classifier is a chain of non-square matrices: 784 → 128 → 10. Each arrow is one of this lesson’s pictures — a flatten (fewer outputs) or a lift (more outputs). Autoencoders flatten then lift; the capstone of this part lifts 2 → 3 so that a plane can cut what a line could not.',
        setup(s) { s.view = '2d'; s.scene = 'ml'; s.cam = cam2(0, 0, 2.6); s.ft = 0; s.panel = 'ml'; },
        at(s, t) { s.ft = t; } },

      { title: 'Recap', duration: 1.5, recap: true,
        narration: 'An m×n matrix takes n inputs to m outputs: columns in, rows out. Wide matrices flatten and must crush some directions; tall matrices lift a small space into a flat subspace of a bigger one. Shapes chain inner-to-inner, read right to left. Real layers are chains of such maps.',
        cards: ['<b>Columns in, rows out.</b> An m×n matrix eats n-vectors and returns m-vectors; each column is an m-tall landing spot.', '<b>Wide flattens, tall lifts.</b> A 2×3 presses the cube flat and crushes a null line; a 3×2 lifts the plane into a tilted teal sheet.', '<b>Inner shapes match.</b> (m×n)(n×p) = m×p, read right to left: 784 → 128 → 10 is a chain of such maps.'],
        setup(s) { s.view = '2d'; s.scene = 'recap'; s.panel = 'none'; },
        at(s, t) { } },
    ],

    playground: {
      intro: '<p>Edit B (2×3) and L (3×2). <b>Flatten</b> shows the cube and its 2D shadow under B (drag the yellow input in 3D; the magenta line is B’s null space). <b>Lift</b> shows the plane carried into 3D by L. <b>Compose</b> shows BL acting on the plane.</p>',
      setup(s) { s.pg = true; s.chk = null; s.mode = 'flatten'; s.view = '3d'; s.cam = LEFT3(); s.B = LA.clone(B0); s.L = LA.clone(L0); s.v3 = [1, 0.5, 1]; s.panel = 'play'; },
      controls: [
        { type: 'matrix', bind: 'B', label: 'B (2×3)', colors: ['i', 'j', 'k'], step: 0.1, min: -2, max: 2 },
        { type: 'matrix', bind: 'L', label: 'L (3×2)', colors: 'cols', step: 0.1, min: -2, max: 2 },
        { type: 'matrix', bind: 'v3', label: 'input vector (3D)', colors: 'v', step: 0.1 },
        { type: 'preset', label: 'Picture', items: [
          { label: 'Flatten (B)', apply(s) { s.mode = 'flatten'; s.view = '3d'; s.cam = LEFT3(); } },
          { label: 'Lift (L)', apply(s) { s.mode = 'lift'; s.view = '3d'; s.cam = cam3({ dist: 15.5, yaw: 1.1, pitch: 0.38, target: [0, 0, 0.3] }); } },
          { label: 'Compose (BL)', apply(s) { s.mode = 'both'; s.view = '2d'; s.cam = cam2(0.3, 0.3, 2.8); } },
        ] },
        { type: 'preset', label: 'Presets', items: [
          { label: 'Drop z', apply(s) { s.B = [[1, 0, 0], [0, 1, 0]]; s.mode = 'flatten'; s.view = '3d'; s.cam = LEFT3(); } },
          { label: 'Shadow with tilt', apply(s) { s.B = LA.clone(B0); s.mode = 'flatten'; s.view = '3d'; s.cam = LEFT3(); } },
          { label: 'Lift tilted', apply(s) { s.L = LA.clone(L0); s.mode = 'lift'; s.view = '3d'; s.cam = cam3({ dist: 15.5, yaw: 1.1, pitch: 0.38, target: [0, 0, 0.3] }); } },
          { label: 'Chain demo', apply(s) { s.B = LA.clone(B0); s.L = LA.clone(L0); s.mode = 'both'; s.view = '2d'; s.cam = cam2(0.3, 0.3, 2.8); } },
        ] },
      ],
    },

    checks: [
      { kind: 'entry',
        prompt: 'Type a 2×3 matrix that keeps x and y and drops z. The cube’s shadow on the right redraws live.',
        bind: 'B', answer: [[1, 0, 0], [0, 1, 0]], tol: LA.TOL.entry, label: 'B (2×3)', colors: ['i', 'j', 'k'],
        hint: 'î and ĵ should land where they are; k̂ should land on the origin.',
        setup(s) { s.pg = false; s.chk = 'entry'; s.view = '3d'; s.cam = LEFT3(); s.B = LA.clone(B0); s.panel = 'chk0'; } },
      { kind: 'manipulate',
        prompt: 'Drag the yellow 3D vector into the null space of B = [[1, 0, 0.5],[0, 1, 0.5]] so that its shadow on the right vanishes.',
        setup(s) { s.pg = false; s.chk = 'null'; s.view = '3d'; s.cam = LEFT3(); s.B = LA.clone(B0); s.v3 = [1, 0, 0]; s.panel = 'chk1'; },
        judge(s) { const e = N.norm(N.matVec(B0, s.v3)), n = N.norm(s.v3); return { ok: e < LA.TOL.pos && n > 0.5, progress: Math.max(0, 1 - e / 1.5), hint: n <= 0.5 ? 'The zero vector does not count: keep the arrow longer than 0.5.' : undefined }; },
        hint: 'Along [−1; −1; 2] both outputs cancel. Orbit the view to drag in depth.',
        solution(s) { s.v3 = [-1, -1, 2]; } },
      { kind: 'manipulate',
        prompt: 'Drag the three tiles left and right into an order that makes a valid chain: every neighbouring pair of inner numbers must match.',
        setup(s) { s.pg = false; s.chk = 'tiles'; s.view = '2d'; s.cam = cam2(0, 0, 3.0); s.tx = [1.8, -1.8, 0]; s.panel = 'chk2'; },
        judge(s) { const o = tileOrder(s); const ok = o[0] === 0 && o[1] === 1 && o[2] === 2; const good = (o[0] === 0 ? 1 : 0) + (o[1] === 1 ? 1 : 0) + (o[2] === 2 ? 1 : 0); return { ok, progress: good / 3 }; },
        hint: 'Each inner pair must match; the 1 ends the chain on the right.',
        solution(s) { s.tx = [-1.8, 0, 1.8]; } },
    ],
  });

  function tileOrder(s) { return [0, 1, 2].sort((a, b) => s.tx[a] - s.tx[b]); }

  /* ---------- step 1: shapes ---------- */
  function renderShape1(g, s) {
    const t = s.ft, cell = 0.42;
    g.ghostGrid({ alpha: 0.35 });
    tile(g, 2, 3, 0, 0.3, cell, { label: '2×3', color: 'teal' });
    const uin = seg(t, 0, 0.45), uout = seg(t, 0.45, 0.8);
    strip(g, 3, 3.2 - 2.1 * uin, 0.3, cell, 'v', 'x: 3 numbers in', uin > 0 ? 1 : 0.4);
    if (uout > 0) strip(g, 2, -1.0 - 0.65 * uout, 0.3, cell, 'b', 'Bx: 2 out', uout);
    if (t > 0.5) {
      g.dim([-0.63, -0.12], [0.63, -0.12], { label: '3 columns = 3 in', offset: 44, color: 'v' });
      g.dim([0.63, 0.72], [0.63, -0.12], { label: '2 rows', offset: -14, color: 'b' });
    }
    g.hud('columns count inputs · rows count outputs', 14, 22, { size: 14 });
  }

  /* ---------- steps 2–4: the 2×3 shadow ---------- */
  function renderFlatten(g, s, o) {
    const Bm = s.Bm, alL = 1 - s.dimLeft;
    let tipL = null, tipR = null;
    g.split((g2, r) => {
      g2.cage({ range: 2, walls: false, alpha: 0.5 });
      g2.axes({ length: 2.2, labels: true });
      g2.box(I3(), { fill: 'v', alpha: 0.1 * alL, stroke: 'v', strokeAlpha: 0.8 * alL });
      g2.points(LAT3, { r: 3, color: 'hud', alpha: 0.75 * alL });
      g2.vector([0, 0, 0], [1, 0, 0], { color: 'i', label: 'i', hat: true, width: 3, alpha: Math.max(0.4, alL) });
      g2.vector([0, 0, 0], [0, 1, 0], { color: 'j', label: 'j', hat: true, width: 3, alpha: Math.max(0.4, alL) });
      g2.vector([0, 0, 0], [0, 0, 1], { color: 'k', label: 'k', hat: true, width: s.hiK ? 4.5 : 3 });
      if (s.hiK) { const p = g2.P.toScreen([0, 0, 1]); tipL = [p[0], p[1]]; }
      if (s.nullT >= 0) {
        g2.lineThrough([0, 0, 0], NULLB, { color: 'mag', width: 2.4, len: 2.6 });
        [-3, -2, -1, 1, 2, 3].forEach(k => g2.point(N.scale(NULLB, k * 0.28), { r: 4.5, color: 'mag' }));
      }
      g2.hud('3D input space', r.x + 10, r.y + 18, { size: 13 });
    }, (g2, r) => {
      g2.ghostGrid({ range: 5 });
      g2.axes({ ticks: true, labels: false });
      const a = s.m1;
      if (a > 0) {
        EDGES.forEach(e => g2.line(N.matVec(Bm, CUBE[e[0]]), N.matVec(Bm, CUBE[e[1]]), { color: 'v', width: 1.6, alpha: 0.85 * a }));
        g2.points(LAT3.map(p => N.matVec(Bm, p)), { r: 3, color: 'hud', alpha: 0.75 * a });
        g2.vector(N.col(Bm, 0), { color: 'i', label: 'i', hat: true, width: 3, alpha: a });
        g2.vector(N.col(Bm, 1), { color: 'j', label: 'j', hat: true, width: 3, alpha: a });
        const kk = N.col(Bm, 2);
        if (N.norm(kk) > 0.03) g2.vector(kk, { color: 'k', label: 'k', hat: true, width: s.hiK ? 4.5 : 3, alpha: a });
        else g2.point([0, 0], { r: 5, color: 'k', label: 'k̂ ↦ 0', dx: -40, dy: 14 });
        if (s.hiK) { g2.label('[0.5; 0.5]', kk, { color: 'k', anchor: 'left', dx: 14, dy: 22, size: 14, font: 'ui' }); const p = g2.P.toScreen(kk); tipR = [p[0], p[1]]; }
      }
      if (s.nullT >= 0) {
        [-3, -2, -1, 1, 2, 3].forEach(k => g2.point(N.matVec(Bm, N.scale(NULLB, k * 0.28)), { r: 4.5, color: 'mag' }));
        if (s.nullT >= 1) g2.label('six points, one shadow: the origin', [0, 0], { color: 'mag', anchor: 'left', dx: 12, dy: 26, size: 13, font: 'ui' });
      }
      g2.hud('2D output: B x', r.x + 10, r.y + 18, { size: 13 });
      if (s.nullT >= 0) g2.hud('nullity ≥ 3 − 2 = 1', r.x + 10, r.y + 38, { size: 13.5, color: 'mag' });
    }, { ratio: 0.5, left: { view: '3d', cam: s.cam }, right: { view: '2d', cam: RIGHT2() } });
    if (tipL && tipR && s.link > 0) {
      const n = 26;
      for (let i = 0; i <= n * s.link; i++) { const u = i / n; const p = [tipL[0] + (tipR[0] - tipL[0]) * u, tipL[1] + (tipR[1] - tipL[1]) * u - Math.sin(Math.PI * u) * 60]; g.label('·', p, { screen: true, color: 'k', size: 22, bg: false, font: 'ui' }); }
    }
  }

  /* ---------- step 5: the 3×2 lift ---------- */
  function renderLift(g, s) {
    const u = s.liftT;
    g.split((g2, r) => {
      g2.ghostGrid({ range: 5 }); g2.grid(null, { range: 5, alpha: 0.6 });
      g2.points(LAT2, { r: 3.2, color: 'v', alpha: 0.9 });
      g2.vector([1, 0], { color: 'i', label: 'i', hat: true, width: 3 });
      g2.vector([0, 1], { color: 'j', label: 'j', hat: true, width: 3 });
      g2.hud('2D input plane', r.x + 10, r.y + 18, { size: 13 });
    }, (g2, r) => {
      g2.cage({ range: 2.5, walls: false, alpha: 0.5 });
      g2.axes({ length: 2.6, labels: true });
      if (s.plane > 0) g2.planes([{ point: [0, 0, 0], span: [N.col(L0, 0), N.col(L0, 1)], extent: 2.4, color: 'teal', alpha: 0.24 * s.plane }]);
      const M = lerpM([[1, 0], [0, 1], [0, 0]], L0, u);
      g2.points(LAT2.map(p => N.matVec(M, p)), { r: 3.4, color: 'v' });
      g2.vector([0, 0, 0], N.col(M, 0), { color: 'i', label: 'i', hat: true, width: 3 });
      g2.vector([0, 0, 0], N.col(M, 1), { color: 'j', label: 'j', hat: true, width: 3 });
      g2.hud('3D output: L x', r.x + 10, r.y + 18, { size: 13 });
      if (s.plane > 0.5) g2.hud('teal: column space of L (a plane, rank 2)', r.x + 10, r.y + 38, { size: 12.5, color: 'teal' });
    }, { ratio: 0.4, left: { view: '2d', cam: cam2(0, 0, 3.0) }, right: { view: '3d', cam: s.cam } });
  }

  /* ---------- step 6: misconception ---------- */
  function renderWrong(g, s) {
    const t = s.ft, cell = 0.32;
    // correct: 2-strip in, 3-strip out
    tile(g, 3, 2, 0, 1.1, cell, { label: '3×2', color: 'teal' });
    const ui = seg(t, 0, 0.35), uo = seg(t, 0.35, 0.6);
    strip(g, 2, 2.4 - 1.5 * ui, 1.1, cell, 'v', '2 in');
    if (uo > 0) strip(g, 3, -0.7 - 1.3 * uo, 1.1, cell, 'b', '3 out', uo);
    if (uo >= 1) g.marker.check([0.9, 1.75], { size: 22 });
    // wrong: a 3-strip tries to enter and bounces off
    tile(g, 3, 2, 0, -1.2, cell, { label: '3×2', color: 'muted', labelColor: 'muted' });
    const a = seg(t, 0.2, 0.5), bounce = seg(t, 0.5, 0.7, 'out');
    const x = 2.4 - 1.8 * a + 0.8 * bounce;
    strip(g, 3, x, -1.2, cell, 'muted', '3 in?');
    if (t > 0.6) { g.marker.cross([0.9, -0.55], { size: 24 }); g.label('a 3×2 accepts only 2 inputs', [0, -2.2], { color: 'bad', size: 14, font: 'ui' }); }
    claimLine(g, s);
  }

  /* ---------- step 7: chaining ---------- */
  function renderChain(g, s) {
    const t = s.ft, cell = 0.3;
    const close = seg(t, 0, 0.2);
    const xs = [-1.55 + 0.35 * close, 0, 1.45 - 0.35 * close];
    const fade2 = seg(t, 0.25, 0.5), fade4 = seg(t, 0.5, 0.75), res = seg(t, 0.8, 1);
    const tilesAl = 1 - 0.75 * res;
    TILES.forEach((T, k) => tile(g, T.r, T.c, xs[k], 0, cell, { label: false, alpha: tilesAl, color: ['teal', 'w', 'v'][k] }));
    if (res > 0) tile(g, 3, 1, 0, -1.4, cell, { label: '3×1', color: 'gold', alpha: res });
    // the formula, token by token (screen space, monospace)
    const size = 30, cw = size * 0.6, y = 70;
    const toks = [['(3×', 'text', 1], ['2', 'gold', 1 - fade2], [')(', 'text', 1], ['2', 'gold', 1 - fade2], ['×', 'text', 1], ['4', 'gold', 1 - fade4], [')(', 'text', 1], ['4', 'gold', 1 - fade4], ['×1)', 'text', 1]];
    const total = toks.reduce((n, tk) => n + tk[0].length, 0) * cw;
    let x = g.size.w / 2 - total / 2;
    toks.forEach(tk => { const w = tk[0].length * cw; if (tk[2] > 0.02) g.label(tk[0], [x + w / 2, y], { screen: true, size, font: 'mono', color: tk[1], alpha: tk[2], bg: false }); x += w; });
    if (res > 0) g.label('→ (3×1)', [g.size.w / 2, y + 44], { screen: true, size: 26, font: 'mono', color: 'gold', alpha: res, bg: false });
    g.hud('read right to left: 4 numbers in → 2 → 3 out', 14, g.size.h - 18, { size: 13.5 });
  }

  /* ---------- step 8: BL ---------- */
  function renderCompose(g, s) {
    const M = lerpM(I2(), BL, s.compT);
    g.ghostGrid(); g.grid(M);
    g.vector(N.col(M, 0), { color: 'i', label: 'i', hat: true, width: 3 });
    g.vector(N.col(M, 1), { color: 'j', label: 'j', hat: true, width: 3 });
    g.hud('(2×3)(3×2) = 2×2: up with L, down with B', 14, 22, { size: 14, color: 'gold' });
    g.viewport({ x: 0.6, y: 0.62, w: 0.38, h: 0.35 }, { view: '3d', cam: cam3({ dist: 8.5, yaw: 0.9 + 0.4 * s.compT, pitch: 0.4 }), bg: 'bg' }, (g2, r) => {
      g2.cage({ range: 2, walls: false, alpha: 0.4 });
      g2.planes([{ point: [0, 0, 0], span: [N.col(L0, 0), N.col(L0, 1)], extent: 1.8, color: 'teal', alpha: 0.24 }]);
      g2.points(LAT2.filter(p => Math.abs(p[0]) <= 1 && Math.abs(p[1]) <= 1).map(p => N.matVec(L0, p)), { r: 3, color: 'v' });
      g2.hud('the detour through 3D', r.x + 8, r.y + 16, { size: 12, color: 'hud' });
    });
  }

  /* ---------- step 9: a real layer ---------- */
  function renderBig(g, s) {
    const t = s.ft;
    g.label('W: 128×784 (an 8×24 synthetic sample)', [0, 1.55], { color: 'hud', size: 14, font: 'ui' });
    g.heat2d(s.heat, { x0: -2.0, x1: 2.0, y0: -0.1, y1: 1.23 }, { ramp: x => LA.color.diverge(2 * x - 1), range: [-2.5, 2.5], smooth: false, alpha: 1 });
    // the digit and the strips
    const dg = []; for (let r = 0; r < 28; r++) dg.push(s.dig.slice(r * 28, r * 28 + 28));
    g.image(dg, { x0: 1.2, x1: 1.95, y0: -1.0, y1: -0.25 }, { range: [0, 1] });
    g.label('synthetic digit (not MNIST)', [1.575, -1.0], { anchor: 'below', dy: 6, size: 12, color: 'muted', font: 'ui' });
    const W = g.size.w, H = g.size.h;
    const uin = seg(t, 0.05, 0.45), uout = seg(t, 0.5, 0.85);
    const sw = Math.min(260, W * 0.42);
    g.strip(s.dig, { x: W - sw - 16 + (1 - uin) * (sw + 20), y: H * 0.8, w: sw, h: 16, range: [0, 1], title: uin > 0.9 ? 'x: 784 pixels in (1×784)' : '' });
    if (uout > 0) g.strip(s.h, { x: 16 + (1 - uout) * 60, y: H * 0.9, w: sw * 128 / 784 * 2.2, h: 16, ramp: x => LA.color.diverge(2 * x - 1), title: 'h = W x: 128 out' });
    g.hud('The picture stops at 3. The formula does not.', 14, 22, { size: 14, color: 'gold' });
    g.hud('at least 784 − 128 = 656 input directions are crushed', 14, 42, { size: 13, color: 'mag' });
  }

  /* ---------- step 10: ML tie ---------- */
  function renderML(g, s) {
    const t = s.ft;
    // shape tiles chaining right to left
    const box = (x0, x1, y0, y1, col, label, sub) => { g.polygon([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], { fill: col, alpha: 0.22, stroke: col, width: 1.4 }); g.label(label, [(x0 + x1) / 2, y1], { anchor: 'above', dy: -6, size: 13, color: 'text', font: 'ui' }); if (sub) g.label(sub, [(x0 + x1) / 2, y0], { anchor: 'below', dy: 6, size: 12, color: 'muted', font: 'ui' }); };
    box(1.75, 1.9, 0.2, 2.2, 'v', '784×1', 'x');
    box(0.05, 1.45, 0.95, 1.45, 'w', '128×784', 'W₁');
    box(-1.05, -0.3, 1.1, 1.3, 'w', '10×128', 'W₂');
    const p = seg(t, 0.1, 0.6);
    const fx = 1.82 - 3.0 * p;
    if (p > 0 && p < 1) g.point([fx, 1.2], { r: 7, color: 'gold' });
    const bars = seg(t, 0.55, 0.75);
    if (bars > 0) g.bars(OUT10.map(v => v * bars), { origin: [-2.15, 0.25], width: 0.07, gap: 0.02, height: 1.9, max: 0.74, colors: OUT10.map((v, i) => i === 7 ? 'gold' : 'muted'), labels: ['0', '', '', '', '', '', '', '7', '', '9'] });
    if (bars > 0) g.label('10 scores', [-1.75, 0.25], { anchor: 'below', dy: 18, size: 12, color: 'muted', font: 'ui' });
    g.vector([1.7, 0.05], [-1.2, 0.05], { color: 'hud', width: 1.4, head: 9, label: 'read right to left', labelAt: 0.5, dy: 14 });
    // preview of p2-12: 2 → 3 lift
    g.viewport({ x: 0.04, y: 0.6, w: 0.92, h: 0.36 }, { view: '3d', cam: cam3({ dist: 8, yaw: 0.8 + 0.5 * t, pitch: 0.35 }), bg: 'bg' }, (g2, r) => {
      g2.cage({ range: 2, walls: false, alpha: 0.35 });
      const u = seg(t, 0.3, 0.8);
      const M = lerpM([[1, 0], [0, 1], [0, 0]], L0, u);
      g2.planes([{ point: [0, 0, 0], span: [N.col(M, 0), N.col(M, 1)], extent: 1.7, color: 'teal', alpha: 0.2 }]);
      g2.points(LAT2.filter(q => Math.abs(q[0]) <= 1.5 && Math.abs(q[1]) <= 1.5).map(q => N.matVec(M, q)), { r: 3, color: 'v' });
      g2.hud('p2-12: a neural layer lifts 2 → 3, then a plane can cut', r.x + 10, r.y + 18, { size: 12.5, color: 'gold' });
    });
  }

  /* ---------- checks ---------- */
  function renderCheck(g, s) {
    if (s.chk === 'tiles') {
      const cell = 0.3, o = tileOrder(s);
      TILES.forEach((T, k) => {
        tile(g, T.r, T.c, s.tx[k], 0, cell, { color: ['teal', 'w', 'v'][k] });
        g.handle('t' + k, [s.tx[k], 0], (p, st) => { st.tx[k] = Math.max(-2.2, Math.min(2.2, p[0])); }, { color: 'white' });
      });
      for (let i = 0; i < 2; i++) {
        const a = TILES[o[i]], b = TILES[o[i + 1]], xm = (s.tx[o[i]] + s.tx[o[i + 1]]) / 2;
        if (a.c === b.r) g.marker.check([xm, 1.0], { size: 18 }); else g.marker.cross([xm, 1.0], { size: 18 });
        g.label(a.c + ' vs ' + b.r, [xm, 1.0], { anchor: 'above', dy: -16, size: 12.5, color: a.c === b.r ? 'ok' : 'bad', font: 'ui' });
      }
      if (TILES[o[2]].c !== 1) g.hud('the right end must take a vector: a tile ending in ×1', 14, g.size.h - 18, { size: 13, color: 'muted' });
      g.hud('drag the tiles left and right; inner numbers must match', 14, 22, { size: 13.5 });
      return;
    }
    // flatten picture with live B
    const B = s.chk === 'entry' ? s.B : B0;
    g.split((g2, r) => {
      g2.cage({ range: 2, walls: false, alpha: 0.5 });
      g2.axes({ length: 2.2, labels: true });
      g2.box(I3(), { fill: 'v', alpha: 0.08, stroke: 'v', strokeAlpha: 0.6 });
      if (s.chk === 'null') {
        g2.vector([0, 0, 0], s.v3, { color: 'v', label: 'v', width: 3 });
        // snap onto B's null line when within 0.3 (a 3D drag cannot hit ‖Bv‖ < 0.12 by hand otherwise)
        g2.handle('v3', s.v3, (p, st) => { const u = N.normalize(NULLB), f = N.scale(u, N.dot(p, u)); st.v3 = N.dist(p, f) < 0.3 ? f : p.slice(); }, { color: 'v' });
      } else {
        g2.vector([0, 0, 0], [1, 0, 0], { color: 'i', label: 'i', hat: true, width: 3 });
        g2.vector([0, 0, 0], [0, 1, 0], { color: 'j', label: 'j', hat: true, width: 3 });
        g2.vector([0, 0, 0], [0, 0, 1], { color: 'k', label: 'k', hat: true, width: 3 });
      }
      g2.hud('3D input', r.x + 10, r.y + 18, { size: 13 });
    }, (g2, r) => {
      g2.ghostGrid({ range: 5 }); g2.axes({ ticks: true, labels: false });
      EDGES.forEach(e => g2.line(N.matVec(B, CUBE[e[0]]), N.matVec(B, CUBE[e[1]]), { color: 'v', width: 1.6, alpha: 0.8 }));
      if (s.chk === 'null') {
        const out = N.matVec(B, s.v3);
        if (N.norm(out) > 0.02) g2.vector(out, { color: 'v', label: 'Bv', width: 3 }); else g2.point([0, 0], { r: 6, color: 'mag', label: 'shadow vanished', dx: 10, dy: -12 });
      } else {
        ['i', 'j', 'k'].forEach((c, j) => { const v = N.col(B, j); if (N.norm(v) > 0.03) g2.vector(v, { color: c, label: c, hat: true, width: 3 }); else g2.point([0, 0], { r: 5, color: c, label: c + '̂ ↦ 0', dx: -38, dy: 14 + 14 * j }); });
      }
      g2.hud('2D shadow: B x', r.x + 10, r.y + 18, { size: 13 });
    }, { ratio: 0.5, left: { view: '3d', cam: s.cam }, right: { view: '2d', cam: RIGHT2() } });
  }

  /* ---------- playground ---------- */
  function renderPlay(g, s) {
    if (s.mode === 'both') {
      const M = N.matMul(s.B, s.L);
      g.ghostGrid(); g.grid(M);
      g.vector(N.col(M, 0), { color: 'i', label: 'i', hat: true, width: 3 });
      g.vector(N.col(M, 1), { color: 'j', label: 'j', hat: true, width: 3 });
      g.hud('B L: (2×3)(3×2) = 2×2 · det ' + LA.fmt(N.det(M)), 14, 22, { size: 14, color: 'gold' });
      return;
    }
    if (s.mode === 'lift') {
      const L = s.L;
      g.split((g2, r) => {
        g2.ghostGrid({ range: 5 }); g2.points(LAT2, { r: 3, color: 'v' });
        g2.vector([1, 0], { color: 'i', label: 'i', hat: true, width: 3 }); g2.vector([0, 1], { color: 'j', label: 'j', hat: true, width: 3 });
        g2.hud('2D input', r.x + 10, r.y + 18, { size: 13 });
      }, (g2, r) => {
        g2.cage({ range: 2.5, walls: false, alpha: 0.5 }); g2.axes({ length: 2.6, labels: true });
        const c1 = N.col(L, 0), c2 = N.col(L, 1), rk = N.rank(L, 1e-6);
        if (rk === 2) g2.planes([{ point: [0, 0, 0], span: [c1, c2], extent: 2.4, color: 'teal', alpha: 0.22 }]);
        else if (rk === 1) g2.lineThrough([0, 0, 0], N.norm(c1) > N.norm(c2) ? c1 : c2, { color: 'teal', width: 3, len: 4 });
        g2.points(LAT2.map(p => N.matVec(L, p)), { r: 3.2, color: 'v' });
        g2.vector([0, 0, 0], c1, { color: 'i', label: 'i', hat: true, width: 3 });
        g2.vector([0, 0, 0], c2, { color: 'j', label: 'j', hat: true, width: 3 });
        g2.hud('3D output · rank ' + rk, r.x + 10, r.y + 18, { size: 13, color: 'teal' });
        g2.handle('l1', c1, (p, st) => { for (let k = 0; k < 3; k++) st.L[k][0] = p[k]; }, { color: 'i' });
        g2.handle('l2', c2, (p, st) => { for (let k = 0; k < 3; k++) st.L[k][1] = p[k]; }, { color: 'j' });
      }, { ratio: 0.4, left: { view: '2d', cam: cam2(0, 0, 3.0) }, right: { view: '3d', cam: s.cam } });
      return;
    }
    const B = s.B, ns = N.nullSpace(B, 1e-6);
    g.split((g2, r) => {
      g2.cage({ range: 2, walls: false, alpha: 0.5 }); g2.axes({ length: 2.2, labels: true });
      g2.box(I3(), { fill: 'v', alpha: 0.08, stroke: 'v', strokeAlpha: 0.6 });
      ns.forEach(d => g2.lineThrough([0, 0, 0], d, { color: 'mag', width: 2.2, len: 2.6 }));
      g2.vector([0, 0, 0], s.v3, { color: 'v', label: 'v', width: 3 });
      g2.handle('v3', s.v3, (p, st) => { st.v3 = p.slice(); }, { color: 'v' });
      g2.hud('3D input · magenta: null space of B', r.x + 10, r.y + 18, { size: 13 });
    }, (g2, r) => {
      g2.ghostGrid({ range: 5 }); g2.axes({ ticks: true, labels: false });
      EDGES.forEach(e => g2.line(N.matVec(B, CUBE[e[0]]), N.matVec(B, CUBE[e[1]]), { color: 'v', width: 1.4, alpha: 0.7 }));
      ['i', 'j', 'k'].forEach((c, j) => { const v = N.col(B, j); if (N.norm(v) > 0.03) g2.vector(v, { color: c, label: c, hat: true, width: 2.6 }); });
      const out = N.matVec(B, s.v3);
      g2.vector(out, { color: 'v', label: 'Bv', width: 3 });
      g2.hud('2D shadow: B v', r.x + 10, r.y + 18, { size: 13 });
    }, { ratio: 0.5, left: { view: '3d', cam: s.cam }, right: { view: '2d', cam: RIGHT2() } });
  }

  /* ---------- recap ---------- */
  function renderRecap(g, s) {
    const card = (i, vopts, fn) => g.viewport({ x: 0.02, y: 0.02 + i * 0.327, w: 0.96, h: 0.31 }, vopts, fn);
    card(0, { view: '2d', cam: cam2(0.8, 0.5, 1.2) }, (g2, r) => {
      g2.ghostGrid({ range: 4 });
      EDGES.forEach(e => g2.line(N.matVec(B0, CUBE[e[0]]), N.matVec(B0, CUBE[e[1]]), { color: 'v', width: 1.5, alpha: 0.8 }));
      ['i', 'j', 'k'].forEach((c, j) => g2.vector(N.col(B0, j), { color: c, label: c, hat: true, width: 2.6 }));
      g2.hud('2×3: the cube pressed flat, three 2D columns', r.x + 10, r.y + 16, { size: 13, color: 'hud' });
    });
    card(1, { view: '3d', cam: cam3({ dist: 7.5, yaw: 1.0, pitch: 0.35 }) }, (g2, r) => {
      g2.cage({ range: 2, walls: false, alpha: 0.4 });
      g2.planes([{ point: [0, 0, 0], span: [N.col(L0, 0), N.col(L0, 1)], extent: 1.8, color: 'teal', alpha: 0.24 }]);
      g2.vector([0, 0, 0], N.col(L0, 0), { color: 'i', label: 'i', hat: true, width: 2.6 });
      g2.vector([0, 0, 0], N.col(L0, 1), { color: 'j', label: 'j', hat: true, width: 2.6 });
      g2.hud('3×2: the plane lifted into a tilted sheet', r.x + 10, r.y + 16, { size: 13, color: 'hud' });
    });
    card(2, { view: '2d', cam: cam2(0, 0, 1.0) }, (g2, r) => {
      TILES.forEach((T, k) => tile(g2, T.r, T.c, -1.6 + k * 1.1, 0.1, 0.18, { color: ['teal', 'w', 'v'][k], size: 12 }));
      g2.label('→', [1.75, 0.1], { size: 22, color: 'hud', bg: false });
      tile(g2, 3, 1, 2.3, 0.1, 0.18, { color: 'gold', size: 12 });
      g2.hud('inner numbers match and vanish: (3×2)(2×4)(4×1) → 3×1', r.x + 10, r.y + 16, { size: 13, color: 'hud' });
    });
  }
})();
