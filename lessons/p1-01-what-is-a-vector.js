/* p1-01 — Arrow, point, list of numbers. A vector has three faces; components are how far along each axis;
   position does not matter; beyond three dimensions the list is the picture. */
(function () {
  'use strict';
  const N = LA.num, seg = LA.seg, lerpV = LA.lerpV, fmt = LA.fmt;
  const ihat = '<span class="c-i">î</span>', jhat = '<span class="c-j">ĵ</span>', khat = '<span class="c-k">k̂</span>';
  const ROLES = ['i', 'j', 'k'];
  const STRIP5 = [2, -1, 0.5, 3, 1];
  const GHOSTS = [[3, -2], [3, 1.5], [-1, -3.5], [-1.2, 2.2]];      // translation offsets for the no-position step
  const APT = [6, 3, 4.2];                                           // apartment row [size; rooms; price] from Part 0
  const cam2 = () => ({ center: [0.2, 0], halfH: 4.6 });
  const cam3 = () => Object.assign(LA.cam.defaults3d(), { dist: 10.5, yaw: 0.61, pitch: 0.44, target: [0.6, 0, 0.6] });

  // module-level caches (deterministic, regenerated from seeds; kept out of s so snapshots stay small)
  let W784 = null, E300 = null;
  function w784() { if (!W784) { const px = LA.rng.syntheticDigits(2, 8, { size: 28 })[7].pixels; const r = LA.rng(21); W784 = px.map(x => x - 0.45 + 0.07 * LA.rng.gauss(r)); } return W784; }
  function e300() { if (!E300) { const r = LA.rng(31); E300 = LA.rng.gaussV(r, 300).map(x => 0.5 * x); } return E300; }

  const v2 = v => [v[0], v[1]];
  const v3 = v => [v[0], v[1], v.length > 2 ? v[2] : 0];
  const divRamp = t => LA.color.diverge(t * 2 - 1);

  /** Column bracket drawn with bracket glyphs + colored entries, at a world (or screen) anchor. */
  function bracket(g, pos, vals, roles, o) {
    o = o || {};
    const size = o.size || 15, rowH = size * 1.22, n = vals.length, half = o.half || (size * 1.5);
    const sp = o.screen ? pos : g.P.toScreen(pos);
    const sx = sp[0] + (o.dx || 0), sy = sp[1] + (o.dy || 0) - (n - 1) * rowH / 2;
    const al = o.alpha === undefined ? 1 : o.alpha;
    if (al <= 0.01) return;
    for (let i = 0; i < n; i++) {
      const y = sy + i * rowH;
      const lg = n === 1 ? '[' : i === 0 ? '⎡' : i === n - 1 ? '⎣' : '⎢';
      const rg = n === 1 ? ']' : i === 0 ? '⎤' : i === n - 1 ? '⎦' : '⎥';
      g.label(lg, [sx - half, y], { screen: true, color: o.bracketColor || 'hud', size: size * 1.2, font: 'math', alpha: al, bg: false });
      g.label(typeof vals[i] === 'string' ? vals[i] : fmt(vals[i]), [sx, y], { screen: true, color: roles ? roles[i] : (o.color || 'text'), size, font: 'ui', bold: true, alpha: al });
      g.label(rg, [sx + half, y], { screen: true, color: o.bracketColor || 'hud', size: size * 1.2, font: 'math', alpha: al, bg: false });
    }
  }

  /** Dashed component pieces in 2D (green along x, red along y) with optional dimension labels. */
  function comps2(g, a, b, alpha, labels) {
    if (alpha <= 0.01) return;
    const c = [b[0], a[1]];
    g.line(a, c, { color: 'i', dashed: true, width: 1.7, alpha: 0.9 * alpha });
    g.line(c, b, { color: 'j', dashed: true, width: 1.7, alpha: 0.9 * alpha });
    if (labels && alpha > 0.5) {
      if (Math.abs(b[0] - a[0]) > 0.05) g.dim(a, c, { label: fmt(b[0] - a[0]), color: 'i', offset: b[0] >= a[0] ? 30 : -30 });
      if (Math.abs(b[1] - a[1]) > 0.05) g.dim(c, b, { label: fmt(b[1] - a[1]), color: 'j', offset: b[1] >= a[1] ? 16 : -16 });
    }
  }
  /** Dashed component pieces in 3D (green x, red y, blue z) with value labels (or '?' when hidden). */
  function comps3(g, a, b, alpha, labels, hide) {
    if (alpha <= 0.01) return;
    const p1 = [b[0], a[1], a[2]], p2 = [b[0], b[1], a[2]];
    g.line(a, p1, { color: 'i', dashed: true, width: 1.7, alpha: 0.9 * alpha });
    g.line(p1, p2, { color: 'j', dashed: true, width: 1.7, alpha: 0.9 * alpha });
    g.line(p2, b, { color: 'k', dashed: true, width: 1.7, alpha: 0.9 * alpha });
    g.line(a, p2, { color: 'residual', dashed: [3, 4], width: 1, alpha: 0.5 * alpha });
    if (labels && alpha > 0.5) {
      const mid = (p, q) => [(p[0] + q[0]) / 2, (p[1] + q[1]) / 2, (p[2] + q[2]) / 2];
      g.label(hide ? '?' : fmt(b[0] - a[0]), mid(a, p1), { color: 'i', size: 13, dy: 14 });
      g.label(hide ? '?' : fmt(b[1] - a[1]), mid(p1, p2), { color: 'j', size: 13, dy: 14 });
      g.label(hide ? '?' : fmt(b[2] - a[2]), mid(p2, b), { color: 'k', size: 13, dx: 16 });
    }
  }
  /** Unit copies of the basis arrows laid tip to tail along the component path, drawn up to `units` world units. */
  function copies3(g, b, units) {
    const path = [{ from: [0, 0, 0], to: [b[0], 0, 0], color: 'i' }, { from: [b[0], 0, 0], to: [b[0], b[1], 0], color: 'j' }, { from: [b[0], b[1], 0], to: [b[0], b[1], b[2]], color: 'k' }];
    let left = units;
    for (const seg_ of path) {
      const L = N.norm(N.sub(seg_.to, seg_.from)); if (L < 1e-6) continue;
      const dir = N.scale(N.sub(seg_.to, seg_.from), 1 / L);
      for (let k = 0; k < L - 1e-9 && left > 0; k++) {
        const len = Math.min(1, L - k, left); left -= len;
        const from = N.add(seg_.from, N.scale(dir, k));
        g.vector(from, N.add(from, N.scale(dir, len)), { color: seg_.color, width: 2.4, head: 9, alpha: 0.95 });
      }
    }
  }

  function draw3dRoom(g, len) {
    g.cage({ range: 3, walls: true });
    g.axes({ length: len || 3.3, labels: true, names: ['x', 'y', 'z'], colors: ['i', 'j', 'k'] });
  }

  LA.registerLesson({
    id: 'p1-01-what-is-a-vector',
    part: 1,
    title: 'Arrow, point, list of numbers',
    subtitle: 'Three faces of one object; components; no position; beyond 3 the list is the picture.',
    minutes: 8,
    prereqs: ['p0-02-reading-the-symbols'],
    tags: ['core'],
    badge: { drawnIn: '2D and 3D', realSize: 'a feature vector has 10–1000 entries; a word embedding 300–4096; the weight vector of one neuron 784', transfers: '“one entry per axis” and “arrow = point = list” hold in any n; only the arrow drawing stops at 3' },
    cantShow: ['an arrow with 784 components (we show a strip of numbers instead)'],
    terms: [
      { term: 'vector', def: 'One object with three faces: an arrow from the origin, a point, and a column of numbers.' },
      { term: 'component', def: 'One entry of a vector: how far it goes along one axis.' },
      { term: 'origin', def: 'The point where all axes meet, written 0.' },
      { term: 'basis vector', def: 'One of the axis arrows of length 1: î, ĵ, k̂.' },
    ],
    atlas: ['embeddings', 'mlp'],
    thumbnail: 3,

    state() {
      return {
        view: '2d', cam: cam2(),
        v: [3, 2], face: 'all', fa: 1, fp: 1, fl: 1, grow: 1,
        showComponents: false, compA: 1, compLabels: true,
        showGhosts: false, ghostT: 0, cross: false,
        hideGrid: false, showStrip: false, stripT: 0, inset3: 0, showML: false, mlT: 0,
        showBasis: false, basisT: 0, copies: 0,
        probe: [0, 0, 0], showProbe: false, hideValues: false, lockV: false, showLen: false,
        panel: 'none', recap: false,
      };
    },

    render(g, s, ctx) {
      if (s.recap) { renderRecap(g, s); return; }
      if (s.view === '3d') render3d(g, s, ctx); else render2d(g, s, ctx);
    },

    math(s, tex) {
      const vv = s.view === '3d' ? v3(s.v) : v2(s.v);
      const vsym = tex.sym('v', { color: 'v' });
      // plain colored number: tex.num placed after a tex.mat gets written into the wrong cell by tex.reconcile (engine)
      const cnum = (x, c) => '<span class="' + tex.colorClass(c) + '">' + fmt(x) + '</span>';
      switch (s.panel) {
        case 'v': return tex.block(tex.eq(vsym, tex.op('='), tex.vec(vv, { colors: 'rows' })));
        case 'faces': {
          const f = s.face, on = x => (f === x || f === 'all') ? 'v' : 'muted';
          return tex.block(tex.eq(vsym, tex.op('='), tex.vec(vv, { colors: 'rows' })), tex.eq(tex.text('arrow', { color: on('arrow') }), tex.op(' · '), tex.text('point', { color: on('point') }), tex.op(' · '), tex.text('list', { color: on('list') })));
        }
        case 'ghosts': {
          const parts = [];
          for (let i = 0; i < 5; i++) parts.push(tex.eq(tex.vec(v2(s.v), { colors: 'rows' })));
          parts.push(tex.text('five drawings, one vector', { color: 'muted' }));
          return tex.block.apply(null, parts);
        }
        case 'basis': return tex.block(
          tex.eq(tex.ihat(), tex.op('='), tex.vec([1, 0, 0], { color: 'i' })), tex.eq(tex.jhat(), tex.op('='), tex.vec([0, 1, 0], { color: 'j' })), tex.eq(tex.khat(), tex.op('='), tex.vec([0, 0, 1], { color: 'k' })),
          tex.eq(vsym, tex.op('='), cnum(vv[0], 'i'), tex.ihat(), tex.op('+'), cnum(vv[1], 'j'), tex.jhat(), tex.op('+'), cnum(vv[2], 'k'), tex.khat()));
        case 'strip': return tex.block(tex.eq(vsym, tex.op('='), tex.vec(STRIP5, { color: 'v' }), tex.op('∈'), tex.sym('ℝ', { sup: '5' })));
        case 'ml': return tex.block(tex.eq(tex.sym('x', { color: 'v' }), tex.op('∈'), tex.sym('ℝ', { sup: '3' })), tex.eq(tex.sym('w', { color: 'w' }), tex.op('∈'), tex.sym('ℝ', { sup: '784' })), tex.eq(tex.sym('e', { sub: 'cat', color: 'b' }), tex.op('∈'), tex.sym('ℝ', { sup: '300' })));
        case 'probe': return tex.block(tex.eq(vsym, tex.op('='), tex.vec(['?', '?', '?'], { colors: 'rows' })), tex.eq(tex.text('typed', { color: 'b' }), tex.op('='), tex.vec(v3(s.probe), { color: 'b' })));
        default: return '';
      }
    },

    steps: [
      { title: 'A point, then an arrow', duration: 3,
        narration: 'A yellow point appears at 3 along the green axis and 2 along the red axis. Now draw an arrow from the <b>origin</b> to it. Point and arrow are the same information: where to go, starting from the origin. We will draw whichever face of this <b>vector</b> helps most.',
        setup(s) { s.view = '2d'; s.cam = cam2(); s.v = [3, 2]; s.fa = 1; s.fp = 0; s.fl = 0; s.grow = 0; s.face = 'all'; s.showComponents = false; s.panel = 'none'; },
        at(s, t) { s.fp = seg(t, 0, 0.3); s.grow = seg(t, 0.3, 0.9, 'out'); } },

      { title: 'Components become numbers', duration: 4,
        narration: 'Drop dashed lines to the axes. The green piece is 3, the red piece is 2. These are the <b>components</b> of <span class="c-v">v</span>: how far along the green axis and how far along the red axis. Stack them in a bracket, green on top: that is the list face of the same vector.',
        setup(s) { s.grow = 1; s.fp = 1; s.showComponents = true; s.compA = 0; s.compLabels = false; s.fl = 0; },
        at(s, t) { s.compA = seg(t, 0, 0.3); s.compLabels = t > 0.3; s.fl = seg(t, 0.5, 0.9); s.panel = t > 0.5 ? 'v' : 'none'; } },

      { title: 'Three faces, one object', duration: 4,
        narration: 'Watch the same vector show its three faces: only the arrow, only the point, only the list. Nothing changes but the drawing. When we talk about data we think “point”; when we add or compare we think “arrow”; when we compute we think “list”.',
        setup(s) { s.panel = 'faces'; s.fl = 1; s.compLabels = true; },
        at(s, t) {
          const win = (a, b) => seg(t, a, a + 0.07) - seg(t, b, b + 0.07);
          s.fa = LA.clamp01(seg(t, -0.2, -0.1) - seg(t, 0.25, 0.32) + seg(t, 0.75, 0.82));
          s.fp = LA.clamp01(seg(t, -0.2, -0.1) - seg(t, 0, 0.07) + win(0.25, 0.5) + seg(t, 0.75, 0.82));
          s.fl = LA.clamp01(seg(t, -0.2, -0.1) - seg(t, 0, 0.07) + seg(t, 0.5, 0.57));
          s.compA = s.fa;
          s.face = t < 0.25 ? 'arrow' : t < 0.5 ? 'point' : t < 0.75 ? 'list' : 'all';
        } },

      { title: 'Negative components go the other way', duration: 3,
        narration: 'Move the tip left of the origin and the green component becomes negative: −2 means two steps to the left, against the green axis. Negative is a direction, not a mistake. Here <span class="c-v">v</span> = [−2; 1.5] points up and to the left, and the bracket updates with it.',
        setup(s) { s.panel = 'v'; s.face = 'all'; s.fa = 1; s.fp = 1; s.fl = 1; s.compA = 1; s.compLabels = true; s._from = LA.clone(s.v); },
        at(s, t) { s.v = lerpV(s._from, [-2, 1.5], seg(t, 0.1, 0.8)); } },

      { title: 'A vector has no position', duration: 4, misconception: true,
        narration: 'Belief to test: “an arrow drawn somewhere else is a different vector.” Copy the arrow and slide the copies around the plane. Same length, same direction — same components — so they are the same vector. We draw vectors from the origin by convention, nothing more.',
        setup(s) { s.v = [-2, 1.5]; s.showGhosts = true; s.ghostT = 0; s.cross = false; s.panel = 'ghosts'; },
        at(s, t) { s.ghostT = seg(t, 0, 0.6); s.cross = t > 0.8; } },

      { title: 'A third component', duration: 5,
        narration: 'In 3D a vector has a third component, along a third, blue axis.<span class="c-v">v</span> = [2; 1; 1.5] goes 2 along the green axis, 1 along the red, 1.5 up the blue. The bracket grows one row. Orbit and watch the three dashed pieces stay attached to the axes.',
        setup(s) { s.view = '3d'; s.cam = cam3(); s.v = [2, 1, 1.5]; s.showGhosts = false; s.cross = false; s.grow = 0; s.compA = 0; s.compLabels = false; s.fl = 0; s.panel = 'v'; },
        at(s, t) { s.grow = seg(t, 0, 0.4, 'out'); s.compA = seg(t, 0.35, 0.6); s.compLabels = t > 0.5; s.fl = seg(t, 0.6, 0.8); LA.cam.tween(s, { yaw: 0.61 }, { yaw: 1.13 }, seg(t, 0, 1, 'linear')); } },

      { title: 'The axis arrows are vectors too', duration: 3,
        narration: 'The axes have their own unit arrows: ' + ihat + ' = [1; 0; 0], ' + jhat + ' = [0; 1; 0], ' + khat + ' = [0; 0; 1], each of length 1. They are called <b>basis vectors</b>. Every component of <span class="c-v">v</span> says how many copies of one basis vector to walk: two green, one red, one and a half blue.',
        setup(s) { s.showBasis = true; s.basisT = 0; s.copies = 0; s.panel = 'basis'; s.fl = 0; },
        at(s, t) { s.basisT = seg(t, 0, 0.4, 'back'); s.copies = seg(t, 0.5, 1, 'linear'); s.compA = 1 - seg(t, 0.5, 0.6); } },

      { title: 'Beyond three: a strip', duration: 4,
        narration: 'Five components — [2; −1; 0.5; 3; 1] — is a point in ℝ⁵, written v ∈ ℝ⁵ (∈ reads “is in”). No arrow can be drawn, so we draw the list as a strip of colored cells. Every rule you learn will be stated for the strip, and the strip can be as long as you like. The picture stops at 3. The formula does not.',
        setup(s) { s.view = '2d'; s.cam = cam2(); s.hideGrid = true; s.showStrip = true; s.stripT = 0; s.inset3 = 0; s.showBasis = false; s.copies = 0; s.fa = 0; s.fp = 0; s.fl = 0; s.showComponents = false; s.panel = 'strip'; },
        at(s, t) { s.stripT = seg(t, 0, 0.5, 'out'); s.inset3 = seg(t, 0.4, 0.8); } },

      { title: 'Everything in ML is one of these', duration: 5, mlTie: true,
        narration: 'Three vectors from machine learning. An apartment row from Part 0: [size; rooms; price]. A weight vector: 784 numbers, one per pixel, that a neuron will compare with an image. A word embedding: 300 numbers that encode “cat”. All three are arrows we may not be able to draw and lists we always can.',
        setup(s) { s.showStrip = false; s.showML = true; s.mlT = 0; s.panel = 'ml'; },
        at(s, t) { s.mlT = t; } },

      { title: 'Recap', duration: 2,
        recap: true,
        narration: 'A vector is one object with three faces: arrow from the origin, point, column of numbers. Its components say how far along each axis, negative meaning the other way. Position does not matter — only length and direction. In more than three dimensions, the list is the picture.',
        cards: ['<b>Arrow = point = list.</b> Three faces of one object; the components are how far along each axis.', '<b>No position.</b> Only length and direction count; copies drawn elsewhere have the same bracket.', '<b>Beyond 3, the list is the picture.</b> A strip of cells, as long as you like; the formula does not care.'],
        setup(s) { s.showML = false; s.recap = true; s.panel = 'none'; },
        at(s, t) { } },
    ],

    playground: {
      intro: '<p>Drag the tip of <span class="c-v">v</span> (or type its components), switch faces, and go to 3D to add a blue component. In 3D, drag empty space to orbit.</p>',
      setup(s) { s.view = '2d'; s.cam = cam2(); s.v = [3, 2]; s.face = 'all'; s.fa = 1; s.fp = 1; s.fl = 1; s.grow = 1; s.showComponents = true; s.compA = 1; s.compLabels = true; s.panel = 'v'; s.showLen = true; },
      controls: [
        { type: 'preset', label: 'View', items: [
          { label: '2D', apply(s) { s.view = '2d'; s.cam = cam2(); s.v = v2(s.v); } },
          { label: '3D', apply(s) { s.view = '3d'; s.cam = cam3(); s.v = s.v.length > 2 ? s.v : [s.v[0], s.v[1], 1.5]; } },
        ] },
        { type: 'preset', label: 'Face', items: [
          { label: 'arrow', apply(s) { s.face = 'arrow'; s.fa = 1; s.fp = 0; s.fl = 0; } },
          { label: 'point', apply(s) { s.face = 'point'; s.fa = 0; s.fp = 1; s.fl = 0; } },
          { label: 'list', apply(s) { s.face = 'list'; s.fa = 0; s.fp = 0; s.fl = 1; } },
          { label: 'all', apply(s) { s.face = 'all'; s.fa = 1; s.fp = 1; s.fl = 1; } },
        ] },
        { type: 'toggle', bind: 'showComponents', label: 'Component lines' },
        { type: 'matrix', bind: 'v', label: 'v (type, or drag the tip)', colors: 'rows', step: 0.1, min: -4, max: 4 },
        { type: 'preset', label: 'Examples', items: [
          { label: '[3; 2]', apply(s) { s.view = '2d'; s.cam = cam2(); s.v = [3, 2]; } },
          { label: '[−2; 1.5]', apply(s) { s.view = '2d'; s.cam = cam2(); s.v = [-2, 1.5]; } },
          { label: '[2; 1; 1.5] (3D)', apply(s) { s.view = '3d'; s.cam = cam3(); s.v = [2, 1, 1.5]; } },
        ] },
      ],
    },

    checks: [
      { kind: 'manipulate',
        prompt: 'Drag the tip of the yellow arrow to the vector [−2; 3]. Read the bracket as you go.',
        setup(s) { s.view = '2d'; s.cam = cam2(); s.v = [3, 2]; s.fa = 1; s.fp = 1; s.fl = 1; s.grow = 1; s.showComponents = true; s.compA = 1; s.compLabels = true; s.panel = 'v'; },
        judge(s) { return LA.judge.near(v2(s.v), [-2, 3], LA.TOL.pos); },
        hint: 'Green component −2 means two steps left of the origin; red 3 means three steps up.',
        solution(s) { s.v = [-2, 3]; } },
      { kind: 'manipulate',
        prompt: 'Make a vector with red component 0 and length 4.',
        setup(s) { s.view = '2d'; s.cam = cam2(); s.v = [1, 2]; s.fa = 1; s.fp = 1; s.fl = 1; s.grow = 1; s.showComponents = true; s.compA = 1; s.compLabels = true; s.panel = 'v'; s.showLen = true; },
        judge(s) { const v = v2(s.v); return LA.judge.all(LA.judge.scalar(v[1], 0, LA.TOL.pos), LA.judge.scalar(N.norm(v), 4, LA.TOL.pos)); },
        hint: 'Keep the tip on the green axis (red component 0) and read the length readout: 4 to the right or 4 to the left both work.',
        solution(s) { s.v = [4, 0]; } },
      { kind: 'entry',
        prompt: 'Type the components of the yellow arrow. The orange arrow shows what you typed.',
        bind: 'probe', answer: [1, -2, 2], tol: LA.TOL.entry, label: 'v', colors: 'rows',
        hint: 'Count the green steps, then the red (negative goes against ' + jhat + '), then the blue.',
        setup(s) { s.view = '3d'; s.cam = cam3(); s.v = [1, -2, 2]; s.fa = 1; s.fp = 1; s.fl = 0; s.grow = 1; s.showComponents = true; s.compA = 1; s.compLabels = true; s.hideValues = true; s.showProbe = true; s.lockV = true; s.probe = [0, 0, 0]; s.panel = 'probe'; } },
    ],
  });

  /* ---------------- 2D scene ---------------- */
  function render2d(g, s, ctx) {
    const v = v2(s.v);
    if (!s.hideGrid) {
      g.grid(null, { range: 10 });
      g.axes({ labels: true });
      g.line([-10, 0], [10, 0], { color: 'i', alpha: 0.5, width: 2 });
      g.line([0, -10], [0, 10], { color: 'j', alpha: 0.5, width: 2 });
    }
    if (s.showStrip) { renderStrip(g, s); return; }
    if (s.showML) { renderML(g, s); return; }
    // ghost copies (no-position step)
    if (s.showGhosts) {
      GHOSTS.forEach(o => {
        const p0 = N.scale(o, s.ghostT), p1 = N.add(p0, v);
        comps2(g, p0, p1, 0.6, false);
        g.vector(p0, p1, { color: 'v', alpha: 0.5, width: 2.4, label: 'v', labelOffset: 12 });
        if (s.ghostT > 0.95) bracket(g, p1, v, ROLES, { dx: 44, dy: 6, size: 12, alpha: 0.85 });
      });
    }
    // the vector under study
    const tip = N.scale(v, s.grow);
    if (s.showComponents) comps2(g, [0, 0], tip, s.compA, s.compLabels);
    if (s.fa > 0.01 && s.grow > 0.01) g.vector([0, 0], tip, { color: 'v', width: 3.2, alpha: s.fa });
    if (s.fp > 0.01) g.point(tip, { r: 6, color: 'v', alpha: s.fp });
    const la = Math.max(s.fa, s.fp);
    if (la > 0.01 && s.grow > 0.3) g.label('v', tip, { color: 'v', dx: tip[0] >= 0 ? 14 : -14, dy: -14, font: 'math', alpha: la });
    if (s.fl > 0.01) {
      // bracket beside the tip, flipped to whichever side has room on screen
      const sx = g.P.toScreen(tip)[0];
      const right = sx + 96 < g.size.w - 6, left = sx - 96 > 6;
      const side = tip[0] >= 0 ? (right ? 1 : -1) : (left ? -1 : 1);
      bracket(g, tip, v, ROLES, { dx: 60 * side, dy: side > 0 && tip[1] >= 0 ? -8 : -30, size: 15, alpha: s.fl });
    }
    if (s.showLen) g.hud('‖v‖ = ' + fmt(N.norm(v)), 14, 20, { size: 14, color: 'v' });
    if (s.cross) {
      const x = g.size.w / 2, y = 34;
      g.label('different vectors?', [x, y], { screen: true, size: 16, color: 'bad', font: 'ui' });
      g.marker.cross([x + 92, y], { screen: true, size: 22 });
      g.label('same components → same vector', [x, y + 24], { screen: true, size: 13, color: 'ok', font: 'ui' });
    }
    if (!s.lockV) g.handle('v', v, (p, st) => { st.v = [p[0], p[1]]; }, { color: 'v' });
  }

  function renderStrip(g, s) {
    const a = s.stripT, dx = (1 - a) * 9;
    const x0 = -2.75 + dx, cw = 1.1, y0 = 2.4, y1 = 3.5;
    g.heat2d([STRIP5], { x0, x1: x0 + 5 * cw, y0, y1 }, { range: [-3, 3], ramp: divRamp, alpha: 0.9, smooth: false });
    STRIP5.forEach((val, i) => { g.label(fmt(val), [x0 + (i + 0.5) * cw, (y0 + y1) / 2], { size: 15, bold: true, color: 'text', font: 'ui', bg: false }); g.line([x0 + i * cw, y0], [x0 + i * cw, y1], { color: 'line', width: 1 }); });
    g.label('v ∈ ℝ⁵ — five components, one cell each', [x0 + 2.5 * cw, y1 + 0.42], { size: 14, color: 'v', font: 'ui' });
    g.label('no 4th or 5th axis exists to draw', [x0 + 2.5 * cw, y0 - 0.4], { size: 12.5, color: 'muted', font: 'ui' });
    if (s.inset3 > 0.01) {
      const S = g.size, r = { x: Math.round(S.w * 0.1), y: Math.round(S.h * 0.4), w: Math.round(S.w * 0.8), h: Math.round(S.h * 0.42) };
      g.viewport(r, { view: '3d', cam: Object.assign(LA.cam.defaults3d(), { dist: 7.5, yaw: 0.7, pitch: 0.4, target: [0.8, -0.3, 0.2] }), bg: 'panel' }, (g2, rr) => {
        draw3dRoom(g2, 2.4);
        const w = [STRIP5[0], STRIP5[1], STRIP5[2]];
        comps3(g2, [0, 0, 0], w, 1, true, false);
        g2.vector([0, 0, 0], w, { color: 'v', width: 3, label: 'v', alpha: s.inset3 });
        g2.label('first 3 of 5: the part we can draw', [rr.x + 10, rr.y + 14], { screen: true, anchor: 'left', size: 12.5, color: 'hud', font: 'ui' });
      });
    }
    g.label('The picture stops at 3. The formula does not.', [0.2, -4.1], { size: 14.5, color: 'hud', font: 'ui', alpha: a });
  }

  function renderML(g, s) {
    const S = g.size, t = s.mlT, W = S.w - 80;
    const p = [seg(t, 0, 0.24, 'out'), seg(t, 0.33, 0.57, 'out'), seg(t, 0.66, 0.9, 'out')];
    const sl = k => 40 + (1 - p[k]) * (S.w + 40);
    // 1) apartment row: three cells, values printed
    const x1 = sl(0), y1 = 64;
    g.strip(APT, { x: x1, y: y1, w: 168, h: 36, range: [0, 7], ramp: LA.color.heat, title: 'x = [size; rooms; price] — one apartment row (Part 0)' });
    APT.forEach((val, i) => g.label(fmt(val), [x1 + (i + 0.5) * 56, y1 + 18], { screen: true, size: 14, bold: true, color: 'text', font: 'ui', bg: false }));
    g.label('3 entries', [x1 + 180, y1 + 18], { screen: true, anchor: 'left', size: 12.5, color: 'muted', font: 'ui' });
    if (p[0] > 0.9) {
      g.viewport({ x: 40, y: 112, w: 190, h: 150 }, { view: '3d', cam: Object.assign(LA.cam.defaults3d(), { dist: 26, yaw: 0.75, pitch: 0.38, target: [3, 1.5, 1.6], fov: 30 }), bg: 'panel' }, (g2, rr) => {
        g2.cage({ range: 6, walls: false });
        g2.axes({ length: 6.5, labels: true, names: ['size', 'rooms', 'price'], colors: ['i', 'j', 'k'] });
        comps3(g2, [0, 0, 0], APT, 1, false, false);
        g2.vector([0, 0, 0], APT, { color: 'v', width: 2.6, label: 'x' });
        g2.label('its arrow (3D)', [rr.x + 8, rr.y + 12], { screen: true, anchor: 'left', size: 12, color: 'hud', font: 'ui' });
      });
    }
    // 2) weight vector of one neuron: 784 entries
    const y2 = 318;
    g.strip(w784(), { x: sl(1), y: y2, w: W, h: 36, range: [-0.6, 0.6], title: 'w — the weight vector of one neuron, 784 entries (one per pixel)' });
    g.label('synthetic weights · no arrow can be drawn', [sl(1), y2 + 50], { screen: true, anchor: 'left', size: 12, color: 'muted', font: 'ui' });
    // 3) word embedding: 300 entries
    const y3 = 424;
    g.strip(e300(), { x: sl(2), y: y3, w: W, h: 36, range: [-1.2, 1.2], title: 'e_cat — the word embedding of “cat”, 300 entries' });
    g.label('synthetic embedding · no arrow can be drawn', [sl(2), y3 + 50], { screen: true, anchor: 'left', size: 12, color: 'muted', font: 'ui' });
    if (t > 0.9) {
      g.label('Lists we can always draw; arrows only up to three entries.', [S.w / 2, S.h - 64], { screen: true, size: 14, color: 'hud', font: 'ui' });
      g.label('Previews p1-07 (data as vectors) and p6-02 (embeddings)', [S.w / 2, S.h - 40], { screen: true, size: 12.5, color: 'gold', font: 'ui' });
    }
  }

  /* ---------------- 3D scene ---------------- */
  function render3d(g, s, ctx) {
    const v = v3(s.v);
    draw3dRoom(g, 2.9);
    const tip = N.scale(v, s.grow);
    if (s.showComponents) comps3(g, [0, 0, 0], tip, s.compA, s.compLabels, s.hideValues);
    if (s.showBasis && s.basisT > 0.01) {
      const b = s.basisT;
      g.vector([0, 0, 0], [b, 0, 0], { color: 'i', label: 'i', hat: true, width: 3 });
      g.vector([0, 0, 0], [0, b, 0], { color: 'j', label: 'j', hat: true, width: 3 });
      g.vector([0, 0, 0], [0, 0, b], { color: 'k', label: 'k', hat: true, width: 3 });
      if (b > 0.9) {
        bracket(g, [1, 0, 0], [1, 0, 0], ROLES, { dx: 0, dy: 46, size: 11.5 });
        bracket(g, [0, 1, 0], [0, 1, 0], ROLES, { dx: 50, dy: 14, size: 11.5 });
        bracket(g, [0, 0, 1], [0, 0, 1], ROLES, { dx: 58, dy: -4, size: 11.5 });
      }
      if (s.copies > 0.01) copies3(g, v, s.copies * (Math.abs(v[0]) + Math.abs(v[1]) + Math.abs(v[2])));
    }
    if (s.fa > 0.01 && s.grow > 0.01) g.vector([0, 0, 0], tip, { color: 'v', width: 3.2, alpha: s.fa });
    if (s.fp > 0.01) g.point(tip, { r: 6, color: 'v', alpha: s.fp });
    if (s.grow > 0.3) g.label('v', tip, { color: 'v', dx: -14, dy: -14, font: 'math', alpha: Math.max(s.fa, s.fp) });
    if (s.fl > 0.01) bracket(g, tip, v, ROLES, { dx: 14, dy: -66, size: 14, alpha: s.fl });
    if (s.showProbe) {
      const p = v3(s.probe);
      if (N.norm(p) > 0.05) { comps3(g, [0, 0, 0], p, 0.8, false, false); g.vector([0, 0, 0], p, { color: 'b', width: 2.6, label: 'typed', dashed: true }); }
    }
    if (!s.lockV) g.handle('v', v, (p, st) => { st.v = [p[0], p[1], p[2] || 0]; }, { color: 'v', plane: 'auto' });
  }

  /* ---------------- recap cards ---------------- */
  function renderRecap(g, s) {
    const S = g.size, m = 10, h = Math.floor((S.h - 4 * m) / 3), w = S.w - 2 * m;
    const row = i => ({ x: m, y: m + i * (h + m), w, h });
    const title = (rr, txt) => g.label(txt, [rr.x + 10, rr.y + 14], { screen: true, anchor: 'left', size: 13, color: 'hud', font: 'ui', bold: true });
    // card 1: arrow + point + bracket
    g.viewport(row(0), { view: '2d', cam: { center: [1.8, 0.9], halfH: 2.2 }, bg: 'panel' }, (g2, rr) => {
      g2.ghostGrid({ range: 8 });
      comps2(g2, [0, 0], [3, 2], 1, true);
      g2.vector([0, 0], [3, 2], { color: 'v', width: 3, label: 'v' });
      g2.point([3, 2], { r: 6, color: 'v' });
      bracket(g2, [3, 2], [3, 2], ROLES, { dx: 64, dy: 0, size: 14 });
      title(rr, '1 · Arrow = point = list');
    });
    // card 2: ghost copies, identical brackets
    g.viewport(row(1), { view: '2d', cam: { center: [0.3, 0.4], halfH: 2.6 }, bg: 'panel' }, (g2, rr) => {
      g2.ghostGrid({ range: 8 });
      const v = [-2, 1.5];
      [[0, 0], [3, -1.2], [1.2, -2.2]].forEach((o, i) => { const p1 = N.add(o, v); comps2(g2, o, p1, 0.7, false); g2.vector(o, p1, { color: 'v', width: 2.6, alpha: i ? 0.55 : 1, label: 'v' }); bracket(g2, p1, v, ROLES, { dx: 44, dy: 4, size: 11.5 }); });
      title(rr, '2 · No position: only length and direction');
    });
    // card 3: a 3D arrow beside a long strip
    const r3 = row(2);
    g.viewport({ x: r3.x, y: r3.y, w: Math.round(r3.w * 0.5), h: r3.h }, { view: '3d', cam: Object.assign(LA.cam.defaults3d(), { dist: 8, yaw: 0.7, pitch: 0.4, target: [0.6, 0.3, 0.4] }), bg: 'panel' }, (g2, rr) => {
      draw3dRoom(g2, 2.5);
      comps3(g2, [0, 0, 0], [2, 1, 1.5], 1, false, false);
      g2.vector([0, 0, 0], [2, 1, 1.5], { color: 'v', width: 3, label: 'v' });
      title(rr, '3 · Beyond 3, the list is the picture');
    });
    const sx = r3.x + Math.round(r3.w * 0.5) + 16, sw = Math.round(r3.w * 0.5) - 32;
    g.strip(STRIP5, { x: sx, y: r3.y + 50, w: sw, h: 26, range: [-3, 3], ramp: divRamp, title: 'v ∈ ℝ⁵' });
    g.strip(w784(), { x: sx, y: r3.y + 110, w: sw, h: 26, range: [-0.6, 0.6], title: 'w ∈ ℝ⁷⁸⁴ (synthetic)' });
    g.label('The picture stops at 3.', [sx, r3.y + 160], { screen: true, anchor: 'left', size: 12.5, color: 'hud', font: 'ui' });
    g.label('The formula does not.', [sx, r3.y + 178], { screen: true, anchor: 'left', size: 12.5, color: 'hud', font: 'ui' });
  }
})();
