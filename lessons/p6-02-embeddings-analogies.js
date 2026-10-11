/* p6-02 — Embeddings, analogies, cosine in 1000 dimensions. Word vectors: the 3D toy from LA.rng.wordVectors()
   (its ten "people" words; authored, not learned — said on screen). */
(function () {
  'use strict';
  const N = LA.num, seg = LA.seg, C = LA.color, lerp = LA.lerp, lerpV = LA.lerpV, clamp01 = LA.clamp01;
  const PX = 0.38;
  const AXIS = [1, 1, 1];
  const ROT = 70 * Math.PI / 180;
  const DS = [3, 30, 300, 1000];
  const ANALOGIES = [
    { base: 'king', minus: 'man', plus: 'woman', want: 'queen', label: 'king − man + woman' },
    { base: 'prince', minus: 'king', plus: 'queen', want: 'princess', label: 'prince − king + queen' },
    { base: 'boy', minus: 'man', plus: 'woman', want: 'girl', label: 'boy − man + woman' },
  ];

  // ---------- data computed once (deterministic) ----------
  let CACHE = null;
  function heavy() {
    if (CACHE) return CACHE;
    const hists = DS.map(d => LA.rng.cosineHistogram(602, d).counts);
    const Dr = [8, 16, 64, 256, 1024], ratio = [], prec = [];
    let top5 = null;
    Dr.forEach(d => {
      const centers = LA.rng.tokenVectors(603, 5, d), noise = LA.rng.tokenVectors(604, 201, d);
      const docs = []; for (let i = 0; i < 200; i++) docs.push(N.normalize(N.add(centers[i % 5], noise[i])));
      const q = N.normalize(N.add(centers[0], noise[200]));
      const sc = docs.map((v, i) => ({ c: N.dot(q, v), topic: i % 5 })).sort((a, b) => b.c - a.c);
      prec.push(sc.slice(0, 5).filter(x => x.topic === 0).length / 5);
      if (d === 64) top5 = sc.slice(0, 5).map(x => [Math.round(x.c * 1000) / 1000, x.topic]);
      const rnd = LA.rng.tokenVectors(605, 201, d), q2 = rnd[200];
      const ds = rnd.slice(0, 200).map(v => N.dist(q2, v));
      ratio.push(Math.min(...ds) / Math.max(...ds));
    });
    const strip = LA.rng.tokenVectors(606, 1, 12)[0].map(v => Math.round(v * 100) / 100);
    CACHE = { hists, ratio, prec, top5, strip, Dr };
    return CACHE;
  }
  function makeWords() { return LA.rng.wordVectors().slice(0, 10).map(o => ({ word: o.word, v: o.v.slice() })); }
  const idx = (s, w) => s.words.findIndex(o => o.word === w);
  const Rm = s => N.rot3(AXIS, s.rotA);
  const V = (s, i) => N.matVec(Rm(s), s.words[i].v);
  const cosv = (a, b) => N.dot(a, b) / ((N.norm(a) * N.norm(b)) || 1);
  const deg = (a, b) => Math.acos(Math.max(-1, Math.min(1, cosv(a, b)))) * 180 / Math.PI;
  function walkTarget(s, an) { const a = ANALOGIES[an]; const g = w => s.words[idx(s, w)].v; return N.add(N.sub(g(a.base), g(a.minus)), g(a.plus)); }
  /** ranking of the words against a query vector (unrotated coordinates; rotation would not change it) */
  function ranking(s, q, exclude, euclid) {
    const out = [];
    s.words.forEach((o, i) => { if (exclude && exclude.indexOf(o.word) >= 0) return; out.push({ i, word: o.word, val: euclid ? N.dist(q, o.v) : cosv(q, o.v) }); });
    out.sort((a, b) => euclid ? a.val - b.val : b.val - a.val);
    return out;
  }
  function stdOf(counts) { const n = counts.length; let S = 0, T = 0; counts.forEach((c, b) => { const x = -1 + (2 * b + 1) / n; S += c * x * x; T += c; }); return Math.sqrt(S / (T || 1)); }

  // ---------- screen helpers ----------
  function rect(g, x, y, w, h, fill) { g.strip([0], { x, y, w, h, range: [0, 1], ramp: () => fill }); }
  function txt(g, s, x, y, o) { g.label(s, [x, y], Object.assign({ screen: true, size: 12.5, bg: false, font: 'ui', anchor: 'left', color: 'hud' }, o || {})); }
  /** horizontal bars from a zero line; items [{label, val, color}] */
  function hbars(g, items, o) {
    const x = o.x, y = o.y, w = o.w, rh = o.rowH || 17, lw = o.labelW || 62;
    const lo = o.min === undefined ? -1 : o.min, hi = o.max === undefined ? 1 : o.max;
    const X = v => x + lw + (v - lo) / (hi - lo) * (w - lw);
    if (o.title) txt(g, o.title, x, y - 12, { color: o.titleColor || 'hud' });
    items.forEach((it, k) => {
      const yy = y + k * rh, x0 = X(Math.max(lo, Math.min(0, hi))), x1 = X(Math.max(lo, Math.min(hi, it.val)));
      rect(g, Math.min(x0, x1), yy + 2, Math.max(1, Math.abs(x1 - x0)), rh - 4, C.alpha(C.role(it.color || 'data'), it.alpha === undefined ? 0.85 : it.alpha));
      txt(g, it.label, x + lw - 6, yy + rh / 2, { anchor: 'right', size: 11, color: it.labelColor || 'text' });
      txt(g, LA.fmt(it.val), Math.max(x0, x1) + 4, yy + rh / 2, { size: 10.5, color: 'muted' });
    });
    rect(g, X(Math.max(lo, Math.min(0, hi))), y, 1, items.length * rh, C.muted);
  }
  function histPanel(g, counts, x, y, w, h, title) {
    g.bars(counts, { screen: true, x, y, w, h, gap: 1, color: 'teal', alpha: 0.85 });
    txt(g, '−1', x, y + h + 10, { anchor: 'center', size: 10, color: 'muted' });
    txt(g, '0', x + w / 2, y + h + 10, { anchor: 'center', size: 10, color: 'muted' });
    txt(g, '1', x + w, y + h + 10, { anchor: 'center', size: 10, color: 'muted' });
    txt(g, title, x, y - 10, { size: 12, color: 'hud' });
  }

  const camW = () => Object.assign(LA.cam.defaults3d(), { target: [0.0, 0.45, 0.15], dist: 8.6, yaw: -0.95, pitch: 0.36, fov: 36 });

  LA.registerLesson({
    id: 'p6-02-embeddings-analogies',
    part: 6,
    title: 'Embeddings, analogies, cosine in 1000 dimensions',
    subtitle: 'Meaning is direction: cosine for similarity, displacement arrows for relations.',
    minutes: 12,
    prereqs: ['p1-06-dot-product', 'p1-08-high-dimensions-measured', 'p1-02-add-scale', 'p4-01-change-of-basis', 'p3-04-orthonormal-gram-schmidt'],
    tags: ['core', 'capstone'],
    badge: { drawnIn: 'a hand-made 3D toy: 10 words with authored (not learned) vectors', realSize: 'word2vec is 300-D; GPT token embeddings are 4096–12288-D over 50k–200k tokens', transfers: 'cosine, difference vectors, rotation invariance and nearest neighbors are the same formulas in any d; the three named axes are a fiction — learned axes have no names, and real analogies hold only approximately' },
    cantShow: ['the 300–12,288 axes of real embeddings; only the computed cosine histograms speak for them', 'how messy real analogies are: learned vectors make the parallelograms only roughly parallel', 'polysemy (one word, several meanings) and context-dependent embeddings', 'how embeddings are learned (these ten vectors were written by hand)'],
    terms: [
      { term: 'embedding', def: 'A vector assigned to a word or token.' },
      { term: 'cosine similarity', def: 'The dot product of the two unit vectors: the cosine of the angle between them.' },
      { term: 'nearest neighbor', def: 'The stored vector with the largest cosine to a query.' },
      { term: 'analogy', def: '“a is to b as c is to d”, read as b − a ≈ d − c.' },
      { term: 'displacement vector', def: 'The difference between two word vectors: the arrow from one tip to the other.' },
      { term: 'rotation invariance', def: 'All cosines and differences survive when every vector is rotated by the same orthogonal matrix.' },
    ],
    atlas: ['embeddings', 'attention'],
    thumbnail: 5,

    state() {
      const H = heavy();
      return {
        view: '3d', cam: camW(),
        words: makeWords(), probe: [0, 0, 0.5], rotA: 0, useEuclid: false, shadow: false, dSel: 3,
        hists: H.hists, rTop5: H.top5, rRatio: H.ratio, rPrec: H.prec, strip12: H.strip,
        scene: 'words', panel: 'table', mathMode: 'table', u: 0,
        hl: {}, arcs: 0, barsFor: null, pulse: -1, king2: 0, disp: 0, analogy: 0, walk: -1, showProbe: false, probeDrag: false,
        shadowU: 0, axisA: 1, girlDrag: false, checkMode: null,
      };
    },

    render(g, s, ctx) {
      if (s.scene === 'words') {
        g.viewport({ x: PX, y: 0, w: 1 - PX, h: 1 }, { view: '3d', cam: s.cam, border: false }, (gg) => drawWords(gg, s, ctx));
        g.strip([0], { x: PX * g.size.w - 1, y: 0, w: 1, h: g.size.h, range: [0, 1], ramp: () => C.line });
      }
      drawPanel(g, s);
    },

    math(s, tex) {
      const w = n => s.words[idx(s, n)].v;
      const nm = n => tex.text(n);
      switch (s.mathMode) {
        case 'table': return tex.block(tex.eq(tex.sym('v', { sub: 'king' }), tex.op('='), tex.vec(w('king'))), tex.eq(tex.sym('v', { sub: 'prince' }), tex.op('='), tex.vec(w('prince'))));
        case 'cos': return tex.block(tex.eq(tex.text('cos θ ='), tex.frac(tex.sym('u') + tex.op('·') + tex.sym('v'), tex.norm(tex.sym('u')) + tex.norm(tex.sym('v')))), tex.eq(tex.text('cos(king, prince) ='), tex.num(cosv(w('king'), w('prince')), { color: 'gold' })));
        case 'nn': { const r = ranking(s, w('king'), ['king']); return tex.block(tex.eq(tex.text('argmax'), tex.sym('', { sub: 'w ≠ king' }), tex.text(' cos(king, w) = ' + r[0].word))); }
        case 'dist': { const k = w('king'); return tex.block(tex.eq(tex.norm(tex.text('KING − king')), tex.op('='), tex.num(N.norm(k), { color: 'v' }), tex.op('>'), tex.norm(tex.text('prince − king')), tex.op('='), tex.num(N.dist(w('prince'), k), { color: 'w' })), tex.eq(tex.text('cos = '), tex.num(1, { color: 'v' }), tex.text(' vs '), tex.num(cosv(k, w('prince')), { color: 'w' }))); }
        case 'disp': return tex.block(tex.eq(tex.text('woman − man ='), tex.vec(N.sub(w('woman'), w('man')), { color: 'w' })));
        case 'walk': case 'walk2': { const a = ANALOGIES[s.analogy], p = walkTarget(s, s.analogy); return tex.block(tex.eq(tex.text(a.label + ' ='), tex.vec(p, { color: 'mag' })), tex.eq(tex.text('cos(probe, ' + a.want + ') ='), tex.num(cosv(p, w(a.want)), { color: 'gold' }))); }
        case 'rot': return tex.block(tex.eq(tex.paren(tex.sym('R') + tex.sym('u')), tex.op('·'), tex.paren(tex.sym('R') + tex.sym('v')), tex.op('='), tex.sym('u'), tex.op('·'), tex.sym('v')), tex.eq(tex.text('rotation angle'), tex.num(LA.fmt.deg(s.rotA))));
        case 'shadow': return tex.block(tex.eq(tex.text('shadow = (gender, royalty) coordinates')), tex.eq(tex.text('variance kept ='), tex.num(LA.fmt.pct(shadowVar(s)), { color: 'gold' })));
        case 'nd': return tex.block(tex.eq(tex.text('std(cos) ≈ 1/√d')));
        case 'ml': return tex.block(tex.eq(tex.text('score'), tex.sym('', { sub: 'i' }), tex.op('='), tex.frac(tex.sym('q') + tex.op('·') + tex.sym('d', { sub: 'i' }), tex.norm(tex.sym('q')) + tex.norm(tex.sym('d', { sub: 'i' })))));
        case 'probe': { const r = ranking(s, s.probe, null, s.useEuclid); return tex.block(tex.eq(tex.text('probe ='), tex.vec(s.probe, { color: 'mag' })), tex.eq(tex.text((s.useEuclid ? 'closest by distance: ' : 'nearest by cosine: ') + r[0].word))); }
        case 'girl': return tex.block(tex.eq(tex.text('girl − boy ='), tex.vec(N.sub(w('girl'), w('boy')), { color: 'w' })), tex.eq(tex.text('queen − king ='), tex.vec(N.sub(w('queen'), w('king')), { color: 'gold' })));
        default: return '';
      }
    },

    steps: [
      { title: 'Words as arrows', duration: 3,
        narration: 'Ten words from a hand-made toy, each given a 3D vector — an <b>embedding</b> — so we can see it: a gender axis, a royalty axis, a generation axis. A real model learns hundreds of numbers per word from text, with no names on its axes; we named ours so you can check the geometry.',
        setup(s) { Object.assign(s, { scene: 'words', panel: 'table', mathMode: 'table', hl: {}, arcs: 0, barsFor: null, pulse: -1, king2: 0, disp: 0, walk: -1, showProbe: false, rotA: 0, shadowU: 0, axisA: 1, u: 0 }); s.cam = camW(); },
        at(s, t) { s.u = t; LA.cam.tween(s, { yaw: -1.2 }, { yaw: -0.95 }, t); } },

      { title: 'Similar meaning, similar direction', duration: 3.5,
        narration: 'King and prince point roughly the same way: the angle between them is 40°, so their <b>cosine similarity</b> is 0.77 (p1-06). King and girl differ in gender and generation and point apart: 138°, cosine −0.74. Similarity is an angle, not a position.',
        setup(s) { Object.assign(s, { panel: 'cos', mathMode: 'cos', hl: { king: 'v', prince: 'w' }, arcs: 0, u: 0 }); },
        at(s, t) { s.u = t; s.arcs = t < 0.5 ? seg(t, 0.1, 0.45) : 1 + seg(t, 0.55, 0.9); s.hl = t < 0.5 ? { king: 'v', prince: 'w' } : { king: 'v', prince: 'w', girl: 'b' }; } },

      { title: 'Nearest neighbors', duration: 3.5,
        narration: 'Ask “who is most like king?” and rank every word by cosine. The bars answer: uncle, then prince, then man — the other male words, because gender is this toy’s strongest axis. The top word is king’s <b>nearest neighbor</b>. At heart, a retrieval system does the same: one dot product per stored vector, sorted.',
        setup(s) { Object.assign(s, { panel: 'bars', mathMode: 'nn', hl: { king: 'v' }, arcs: 0, barsFor: 'king', pulse: -1, u: 0 }); },
        at(s, t) { s.u = t; s.pulse = t < 0.95 ? Math.floor(9 * seg(t, 0.05, 0.95, 'linear')) : 0; } },

      { title: '✕ Closer in distance means more similar', duration: 4, misconception: true,
        narration: '“Closer in distance means more similar.” Take KING = 2·king: the same direction, twice as long. By Euclidean distance it is 1.48 from king — farther than prince at 1.02. By cosine it is identical: 1.0 against 0.77. Length is not meaning (in learned embeddings it often tracks how frequent a word is); cosine ignores it on purpose.',
        setup(s) { Object.assign(s, { panel: 'dist', mathMode: 'dist', hl: { king: 'v', prince: 'w' }, barsFor: null, pulse: -1, king2: 0, u: 0 }); },
        at(s, t) { s.u = t; s.king2 = seg(t, 0.05, 0.4); } },

      { title: 'A direction that means something', duration: 3.5,
        narration: 'The arrow from man to woman is the <b>displacement vector</b> woman − man ≈ (2, 0, 0): a “gender arrow”. Copy it onto king’s tip and it lands 0.09 from queen. Relations are directions, and the same direction works for different pairs.',
        setup(s) { Object.assign(s, { panel: 'disp', mathMode: 'disp', hl: { man: 'data', woman: 'data', king: 'v', queen: 'gold' }, king2: 0, disp: 0, u: 0 }); },
        at(s, t) { s.u = t; s.disp = seg(t, 0.05, 1); } },

      { title: 'king − man + woman', duration: 4.5,
        narration: 'Walk it as arithmetic (p1-02): start at king, step back along man, forward along woman. The magenta probe lands on queen: an <b>analogy</b> — man is to woman as king is to queen — drawn as a parallelogram. Our hand-made toy makes it nearly exact (cosine ≈ 1); in real embeddings the probe only lands near queen, and the search usually has to skip king, man and woman.',
        setup(s) { Object.assign(s, { panel: 'walk', mathMode: 'walk', analogy: 0, disp: 0, hl: { king: 'v', man: 'data', woman: 'data', queen: 'gold' }, walk: 0, barsFor: null, u: 0 }); },
        at(s, t) { s.u = t; s.walk = t; } },

      { title: 'Another relation', duration: 4,
        narration: 'Same trick, different relation: prince − king + queen. The displacement king → prince means “one generation younger”; it carries over to queen, and the probe lands next to princess. Real embeddings learn many relations like this from text, but only roughly: their arrows are nearly, not exactly, parallel.',
        setup(s) { Object.assign(s, { panel: 'walk', mathMode: 'walk2', analogy: 1, hl: { prince: 'v', king: 'data', queen: 'data', princess: 'gold' }, walk: 0, u: 0 }); },
        at(s, t) { s.u = t; s.walk = t; } },

      { title: 'Rotate the whole space: nothing changes', duration: 4.5,
        narration: 'Apply one rotation R to every word (p3-04: orthogonal, lengths and angles kept). Every arrow moves; every cosine, every bar and the analogy stay exactly the same — <b>rotation invariance</b>. Coordinates mean nothing on their own (p4-01); only the relations between vectors do.',
        setup(s) { Object.assign(s, { panel: 'rot', mathMode: 'rot', analogy: 0, hl: { king: 'v', man: 'data', woman: 'data', queen: 'gold' }, walk: 1, rotA: 0, axisA: 1, u: 0 }); },
        at(s, t) { s.u = t; s.rotA = ROT * seg(t, 0.1, 0.85); s.axisA = 1 - seg(t, 0.1, 0.5); } },

      { title: 'A shadow can lie', duration: 4.5,
        narration: 'Look straight down the generation axis: a shadow on the gender–royalty plane. King and prince land on one spot, and so do man and boy. A 2D plot of embeddings is a shadow of ℝ³ — or of ℝ⁴⁰⁹⁶ — and closeness in a shadow can be an accident of the view. Always read the variance kept.',
        setup(s) { Object.assign(s, { panel: 'shadow', mathMode: 'shadow', walk: -1, hl: { king: 'v', prince: 'w', man: 'data', boy: 'data' }, shadowU: 0, u: 0 }); s._rot0 = s.rotA; s._cam0 = { yaw: s.cam.yaw, pitch: s.cam.pitch }; },
        at(s, t) {
          s.u = t; s.rotA = s._rot0 * (1 - seg(t, 0, 0.25)); s.axisA = seg(t, 0.1, 0.3);
          s.shadowU = seg(t, 0.3, 0.55);
          LA.cam.tween(s, s._cam0, { yaw: -Math.PI / 2, pitch: 1.53 }, seg(t, 0.45, 0.9));
        } },

      { title: 'Real embeddings are strips', duration: 4,
        narration: 'A real embedding is hundreds to thousands of numbers (word2vec: 300) — a strip, not an arrow. In high dimensions random directions are almost all perpendicular (p1-08): the cosine histograms of random pairs narrow around 0 as d grows, like 1/√d. The picture stops at 3. The formula does not.',
        setup(s) { Object.assign(s, { scene: 'nd', panel: 'nd', mathMode: 'nd', u: 0 }); },
        at(s, t) { s.u = t; } },

      { title: 'Why ML cares: retrieval in 64 dimensions', duration: 4, mlTie: true,
        narration: 'Retrieval in 64 dimensions: 200 synthetic documents in five planted topics and one query. The top five by cosine all share the query’s topic. As d grows, random vectors become equally far apart (nearest/farthest ratio → 1), yet the planted topics still rank first by cosine. Attention (p6-03) is a soft version of this lookup.',
        setup(s) { Object.assign(s, { scene: 'ml', panel: 'ml', mathMode: 'ml', u: 0 }); },
        at(s, t) { s.u = t; } },

      { title: 'Recap', duration: 2, recap: true,
        narration: 'Meaning is direction: similarity is cosine, relations are displacement arrows, analogies are parallelograms. Rotating the whole space changes nothing, and a 2D plot is only a shadow. In 1000 dimensions random directions are nearly perpendicular — room for many separate meanings.',
        cards: ['<b>Similarity = cosine of the angle.</b> king and prince: 40°, cos 0.77; length does not count.', '<b>Relations = displacement arrows.</b> king − man + woman lands next to queen: a parallelogram.', '<b>Rotation changes nothing; in 1000-D random directions are nearly ⟂.</b> Random cosines pile up at 0.'],
        setup(s) { Object.assign(s, { scene: 'words', panel: 'recap', mathMode: 'walk', analogy: 0, walk: 1, rotA: 0, shadowU: 0, axisA: 1, hl: { king: 'v', man: 'data', woman: 'data', queen: 'gold', prince: 'w' }, arcs: 0.999, u: 1 }); s.cam = camW(); },
        at(s, t) { } },
    ],

    playground: {
      intro: '<p>Drag any word tip or the magenta probe (the drag plane is the floor when you look from above, the screen otherwise). The bars rank all words against the probe.</p>',
      setup(s) { Object.assign(s, { scene: 'words', panel: 'play', mathMode: 'probe', hl: {}, walk: -1, showProbe: true, probeDrag: true, girlDrag: true, rotA: 0, axisA: 1, analogy: 0 }); s.probe = walkTarget(s, 0); s.cam = camW(); },
      controls: [
        { type: 'preset', label: 'Analogy', items: ANALOGIES.map((a, k) => ({ label: a.label, apply(s) { s.analogy = k; s.probe = walkTarget(s, k); } })) },
        { type: 'button', label: 'Rotate the space (70°)', onClick(s) { s.rotA = s.rotA > 0.01 ? 0 : ROT; } },
        { type: 'toggle', bind: 'useEuclid', label: 'Rank by Euclidean distance instead of cosine' },
        { type: 'preset', label: 'Histogram dimension d', items: DS.map((d, k) => ({ label: 'd = ' + d, values: { dSel: k } })) },
        { type: 'toggle', bind: 'shadow', label: '2D shadow (gender–royalty)' },
        { type: 'preset', label: 'Words', items: [
          { label: 'Default', apply(s) { s.words = makeWords(); s.rotA = 0; } },
          { label: 'Scrambled', apply(s) { s.words = makeWords(); s.words[idx(s, 'girl')].v = [0.1, 0.6, -0.5]; } },
          { label: 'Rotated', apply(s) { s.rotA = ROT; } },
        ] },
      ],
    },

    checks: [
      { kind: 'manipulate',
        prompt: 'Drag the magenta probe to king − man + woman.',
        setup(s) { Object.assign(s, { scene: 'words', panel: 'probeCheck', mathMode: 'probe', hl: { king: 'v', man: 'data', woman: 'data' }, walk: -1, showProbe: true, probeDrag: true, girlDrag: false, rotA: 0, probe: [0, 0, 0.5], checkMode: 'probe' }); s.cam = camW(); },
        judge(s) { return LA.judge.near(s.probe, walkTarget(s, 0), LA.TOL.pos); },
        hint: 'Start at king; step back along man; step forward along woman.',
        solution(s) { s.probe = walkTarget(s, 0); } },
      { kind: 'manipulate',
        prompt: 'Drag “girl” so that girl − boy equals queen − king.',
        setup(s) { Object.assign(s, { scene: 'words', panel: 'girl', mathMode: 'girl', hl: { girl: 'w', boy: 'data', queen: 'gold', king: 'gold' }, walk: -1, showProbe: false, probeDrag: false, girlDrag: true, rotA: 0, checkMode: 'girl' }); s.words = makeWords(); s.words[idx(s, 'girl')].v = [0.1, 0.6, -0.5]; s.cam = camW(); },
        judge(s) { const w = n => s.words[idx(s, n)].v; return LA.judge.near(N.sub(w('girl'), w('boy')), N.sub(w('queen'), w('king')), LA.TOL.pos); },
        hint: 'The gender arrow is the same for every pair: girl = boy + (queen − king).',
        solution(s) { const w = n => s.words[idx(s, n)].v; s.words[idx(s, 'girl')].v = N.add(w('boy'), N.sub(w('queen'), w('king'))); } },
      { kind: 'manipulate',
        prompt: 'Make queen the nearest neighbor of the probe by cosine while keeping the probe shorter than 0.5.',
        setup(s) { Object.assign(s, { scene: 'words', panel: 'short', mathMode: 'probe', hl: { queen: 'gold' }, walk: -1, showProbe: true, probeDrag: true, girlDrag: false, rotA: 0, useEuclid: false, probe: [-0.2, -0.6, 0.6], checkMode: 'short' }); s.words = makeWords(); s.cam = camW(); },
        judge(s) { const r = ranking(s, s.probe, null, false); const top = r[0].word === 'queen', short = N.norm(s.probe) < 0.5; return { ok: top && short, progress: (top ? 0.5 : 0.5 * clamp01((r.find(x => x.word === 'queen').val + 1) / 2)) + (short ? 0.5 : 0.5 * clamp01(1 - (N.norm(s.probe) - 0.5))), hint: top && !short ? 'Right direction — now shrink it toward the origin.' : undefined }; },
        hint: 'Only direction matters — shrink the probe toward the origin along queen’s direction.',
        solution(s) { const q = s.words[idx(s, 'queen')].v; s.probe = N.scale(q, 0.4 / N.norm(q)); } },
    ],
  });

  function shadowVar(s) {
    const X = s.words.map(o => o.v), m = N.mean(X);
    const v = [0, 1, 2].map(k => X.reduce((a, x) => a + (x[k] - m[k]) * (x[k] - m[k]), 0));
    return (v[0] + v[1]) / (v[0] + v[1] + v[2]);
  }

  // ---------- 3D words scene ----------
  function drawWords(g, s, ctx) {
    const R = Rm(s);
    const P = i => N.matVec(R, s.words[i].v);
    const rot = v => N.matVec(R, v);
    g.grid(null, { range: 2, alpha: 0.35 });
    if (s.axisA > 0.02) {
      const names = ['gender', 'royalty', 'generation'], dirs = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
      dirs.forEach((d, k) => { if (k === 2 && s.shadowU > 0.9) return; g.line(N.scale(d, -1.5), N.scale(d, 1.7), { color: 'gridAxis', width: 1.2, alpha: 0.7 * s.axisA }); g.label(names[k], N.scale(d, 1.85), { color: 'muted', size: 11.5, font: 'ui', alpha: s.axisA, dy: k === 0 ? 16 : 0 }); });
    }
    const n = s.words.length;
    let ranks = null;
    if (s.barsFor === 'king' && s.pulse >= 0) ranks = ranking(s, s.words[idx(s, 'king')].v, ['king']);
    const pulseIdx = ranks ? ranks[Math.min(8, s.pulse)].i : -1;
    // shadows on the floor (gender–royalty plane)
    const shadowOn = s.shadowU > 0 || s.shadow;
    if (shadowOn) {
      const su = s.shadow ? 1 : s.shadowU;
      const groups = [];
      for (let i = 0; i < n; i++) {
        const p = P(i), f = [p[0], p[1], 0];
        g.line(p, lerpV(p, f, su), { color: 'residual', dashed: true, width: 1.1, alpha: 0.85 });
        if (su > 0.95) { g.point(f, { r: 4.5, color: 'mag' }); const gr = groups.find(q => Math.hypot(q.f[0] - f[0], q.f[1] - f[1]) < 0.15); if (gr) gr.names.push(s.words[i].word); else groups.push({ f, names: [s.words[i].word] }); }
      }
      groups.forEach(q => g.label(q.names.join(' = '), q.f, { anchor: 'below', dy: 8, size: 11, color: q.names.length > 1 ? 'mag' : 'muted', font: 'ui' }));
    }
    for (let i = 0; i < n; i++) {
      const w = s.words[i].word, p = P(i);
      const role = s.hl[w] || null;
      const isPulse = i === pulseIdx;
      const color = isPulse ? 'gold' : (role || 'data');
      const alpha = (Object.keys(s.hl).length && !role && !isPulse) ? 0.35 : 1;
      g.vector(p, { color, width: role || isPulse ? 2.8 : 1.8, head: 9, alpha, label: (s.shadowU > 0.9 && !s.shadow) ? undefined : w, labelColor: color, labelSize: 12, labelFont: 'ui' });
    }
    // angle arcs
    if (s.arcs > 0) {
      const k = P(idx(s, 'king'));
      const a1 = Math.min(1, s.arcs);
      if (a1 > 0) { const pr = P(idx(s, 'prince')); g.arc([0, 0, 0], k, lerpV(k, pr, a1), { r: 0.55, color: 'w', label: a1 > 0.95 ? Math.round(deg(k, pr)) + '°' : undefined }); }
      if (s.arcs > 1) { const gl = P(idx(s, 'girl')); const a2 = s.arcs - 1; g.arc([0, 0, 0], k, slerp3(k, gl, a2), { r: 0.35, color: 'b', label: a2 > 0.95 ? Math.round(deg(k, gl)) + '°' : undefined }); }
    }
    // KING = 2·king and the two distances
    if (s.king2 > 0) {
      const k = P(idx(s, 'king')), K = N.scale(k, 1 + s.king2), pr = P(idx(s, 'prince'));
      g.vector(K, { color: 'v', dashed: true, width: 2, alpha: 0.8, label: s.king2 > 0.9 ? 'KING = 2·king' : undefined, labelSize: 12, labelFont: 'ui' });
      if (s.king2 > 0.95) {
        g.dim(k, K, { label: LA.fmt(N.dist(k, K)), color: 'v', offset: -14 });
        g.dim(k, pr, { label: LA.fmt(N.dist(k, pr)), color: 'w', offset: 14 });
        g.marker.cross(K, { dx: 24, dy: -26, size: 20 });
      }
    }
    // displacement arrow woman − man, then a copy on king's tip
    if (s.disp > 0) {
      const m = P(idx(s, 'man')), wo = P(idx(s, 'woman')), k = P(idx(s, 'king')), q = P(idx(s, 'queen'));
      const d = N.sub(wo, m);
      g.vector(m, lerpV(m, wo, seg(s.disp, 0, 0.35)), { color: 'w', width: 3, label: s.disp > 0.3 ? 'd' : undefined });
      const c = seg(s.disp, 0.4, 1);
      if (c > 0) { const from = lerpV(m, k, c); g.vector(from, N.add(from, d), { color: 'w', width: 2.4, alpha: 0.65, label: c > 0.95 ? 'd' : undefined }); }
      if (c > 0.98) g.line(N.add(k, d), q, { color: 'residual', dashed: true, width: 1.4 });
    }
    // tip-to-tail analogy walk
    if (s.walk >= 0) {
      const a = ANALOGIES[s.analogy];
      const B = P(idx(s, a.base)), M = P(idx(s, a.minus)), Pl = P(idx(s, a.plus));
      const u1 = seg(s.walk, 0, 0.35), u2 = seg(s.walk, 0.35, 0.7);
      const mid = N.sub(B, N.scale(M, u1));
      g.vector(B, mid, { color: 'data', dashed: true, width: 2.2, label: u1 > 0.6 ? '−' + a.minus : undefined, labelSize: 12, labelFont: 'ui', labelAt: 0.5 });
      if (u2 > 0) { const end = N.add(mid, N.scale(Pl, u2)); g.vector(mid, end, { color: 'data', width: 2.2, label: u2 > 0.6 ? '+' + a.plus : undefined, labelSize: 12, labelFont: 'ui', labelAt: 0.5 }); }
      if (s.walk > 0.7) { const pr = rot(walkTarget(s, s.analogy)); g.point(pr, { r: 7, color: 'mag', stroke: 'white', label: 'probe', labelColor: 'mag' }); g.line(pr, P(idx(s, a.want)), { color: 'residual', dashed: true, width: 1.2 }); }
    }
    if (s.showProbe) {
      const pr = rot(s.probe);
      g.vector(pr, { color: 'mag', width: 2.4, head: 9, alpha: 0.85, label: 'probe', labelSize: 12, labelFont: 'ui' });
      g.point(pr, { r: 5, color: 'mag' });
      if (s.probeDrag) g.handle('probe', pr, (p, st) => { st.probe = N.matVec(N.transpose(Rm(st)), p); }, { color: 'mag' });
    }
    if (s.girlDrag) {
      for (let i = 0; i < n; i++) {
        if (s.checkMode === 'girl' && s.words[i].word !== 'girl') continue;
        g.handle('w' + i, P(i), (p, st) => { st.words[i].v = N.matVec(N.transpose(Rm(st)), p); }, { color: s.words[i].word === 'girl' ? 'w' : 'data' });
      }
    }
    if (s.checkMode === 'girl') {
      const w = nm => P(idx(s, nm));
      g.vector(w('king'), w('queen'), { color: 'gold', width: 2.2, alpha: 0.8, label: 'q−k', labelSize: 12, labelFont: 'ui', labelAt: 0.5 });
      g.vector(w('boy'), w('girl'), { color: 'w', width: 2.4, label: 'g−b', labelSize: 12, labelFont: 'ui', labelAt: 0.5 });
    }
    g.hud(s.panel === 'table' ? 'hand-made 3D toy: vectors written by hand, not learned' : 'hand-made 3D toy', g.P.vp.x + 12, g.P.vp.y + g.P.vp.h - 20, { size: 11.5, color: 'muted' });
  }
  function slerp3(a, b, t) { const na = N.norm(a), nb = N.norm(b); return N.scale(LA.geom.slerp(N.scale(a, 1 / na), N.scale(b, 1 / nb), t), lerp(na, nb, t)); }

  // ---------- left panel / full-screen panels ----------
  function drawPanel(g, s) {
    const S = g.size, x0 = 16, y0 = 40, pw = PX * S.w - 2 * x0;
    const w = nm => s.words[idx(s, nm)].v;
    const p = s.panel, u = s.u;
    if (p === 'table') {
      txt(g, 'toy embeddings (hand-made, 3D)', x0, y0, { color: 'hud', size: 12.5 });
      const cols = ['gender', 'royal', 'gen.'];
      const cw = Math.min(46, (pw - 66) / 3);
      cols.forEach((c, k) => txt(g, c, x0 + 66 + (k + 0.5) * cw, y0 + 22, { anchor: 'center', size: 10.5, color: 'muted' }));
      s.words.forEach((o, i) => {
        const yy = y0 + 40 + i * 20;
        txt(g, o.word, x0 + 60, yy + 9, { anchor: 'right', size: 11.5, color: 'data' });
        g.strip(o.v, { x: x0 + 66, y: yy, w: 3 * cw, h: 18, range: [-1.3, 1.3], ramp: t => C.diverge(2 * t - 1) });
        o.v.forEach((v, k) => txt(g, LA.fmt(v), x0 + 66 + (k + 0.5) * cw, yy + 9, { anchor: 'center', size: 10.5, color: 'text' }));
      });
      txt(g, 'one row = one word’s embedding', x0, y0 + 40 + 10 * 20 + 16, { size: 11.5, color: 'muted' });
      return;
    }
    if (p === 'cos') {
      const k = w('king'), pr = w('prince'), gl = w('girl');
      txt(g, 'cos(king, prince) = ' + LA.fmt(cosv(k, pr)), x0, y0 + 10, { color: 'w', size: 13 });
      txt(g, 'angle ' + Math.round(deg(k, pr)) + '°', x0, y0 + 30, { color: 'muted' });
      if (u >= 0.5) { txt(g, 'cos(king, girl) = ' + LA.fmt(cosv(k, gl)), x0, y0 + 64, { color: 'b', size: 13 }); txt(g, 'angle ' + Math.round(deg(k, gl)) + '°', x0, y0 + 84, { color: 'muted' }); }
      txt(g, 'cos θ = u·v / (‖u‖‖v‖)', x0, y0 + 130, { color: 'hud', size: 13, font: 'math' });
      return;
    }
    if (p === 'bars' || p === 'recap') {
      const r = ranking(s, w('king'), ['king']);
      hbars(g, r.map((it, k) => ({ label: it.word, val: it.val, color: k === 0 ? 'gold' : (p === 'bars' && k === s.pulse ? 'v' : 'data') })), { x: x0, y: y0 + 16, w: pw, title: 'cos(king, ·), sorted' });
      return;
    }
    if (p === 'dist') {
      const k = w('king'), pr = w('prince');
      g.marker.cross([x0 + 10, y0 + 8], { screen: true, size: 18 });
      txt(g, '“distance ranks similarity”', x0 + 24, y0 + 8, { color: 'text', size: 12.5 });
      txt(g, 'distance from king:', x0, y0 + 44, { color: 'muted' });
      txt(g, 'KING  ' + LA.fmt(N.norm(k)), x0 + 10, y0 + 64, { color: 'v' });
      txt(g, 'prince  ' + LA.fmt(N.dist(k, pr)), x0 + 10, y0 + 84, { color: 'w' });
      txt(g, '→ prince looks closer', x0 + 10, y0 + 104, { color: 'bad', size: 11.5 });
      txt(g, 'cosine with king:', x0, y0 + 140, { color: 'muted' });
      txt(g, 'KING  1', x0 + 10, y0 + 160, { color: 'v' });
      txt(g, 'prince  ' + LA.fmt(cosv(k, pr)), x0 + 10, y0 + 180, { color: 'w' });
      txt(g, '→ same direction, same meaning', x0 + 10, y0 + 200, { color: 'ok', size: 11.5 });
      return;
    }
    if (p === 'disp') {
      const d = N.sub(w('woman'), w('man')), land = N.add(w('king'), d);
      txt(g, 'd = woman − man', x0, y0 + 10, { color: 'w', size: 13 });
      txt(g, '= (' + d.map(v => LA.fmt(v)).join(', ') + ')', x0, y0 + 30, { color: 'w' });
      if (u > 0.95) { txt(g, 'king + d lands', x0, y0 + 70, { color: 'hud' }); txt(g, LA.fmt(N.dist(land, w('queen'))) + ' from queen', x0, y0 + 90, { color: 'gold' }); }
      return;
    }
    if (p === 'walk' || p === 'rot') {
      const a = ANALOGIES[s.analogy], pr = walkTarget(s, s.analogy);
      txt(g, a.label, x0, y0 + 4, { color: 'mag', size: 13 });
      if ((p === 'walk' && u > 0.7) || p === 'rot') {
        const r = ranking(s, pr, [a.base, a.minus, a.plus]);
        hbars(g, r.map((it, k) => ({ label: it.word, val: it.val, color: k === 0 ? 'gold' : 'data' })), { x: x0, y: y0 + 40, w: pw, title: 'cos(probe, ·) — inputs excluded' });
        if (p === 'rot' && u > 0.6) { const yb = y0 + 40 + r.length * 17 + 18; g.marker.check([x0 + 10, yb], { screen: true, size: 20 }); txt(g, 'bars unchanged by R', x0 + 26, yb, { color: 'ok', size: 12 }); }
      }
      return;
    }
    if (p === 'shadow') {
      txt(g, 'shadow of ℝ³ on 2 directions', x0, y0 + 6, { color: 'mag', size: 13 });
      txt(g, '(gender, royalty)', x0, y0 + 26, { color: 'mag' });
      txt(g, LA.fmt.pct(shadowVar(s)) + ' of the variance kept', x0, y0 + 50, { color: 'gold' });
      txt(g, 'dropped: the generation axis', x0, y0 + 70, { color: 'muted' });
      if (u > 0.55) { txt(g, 'king = prince in the shadow,', x0, y0 + 110, { color: 'hud' }); txt(g, 'yet cos(king, prince) = ' + LA.fmt(cosv(w('king'), w('prince'))), x0, y0 + 130, { color: 'hud' }); }
      return;
    }
    if (p === 'nd') {
      const vals = s.strip12, cw = Math.min(42, (S.w - 60) / 12), sx = 30, sy = 54;
      txt(g, 'embedding ∈ ℝ¹² (a random synthetic strip)', sx, sy - 14, { size: 13 });
      g.strip(vals, { x: sx, y: sy, w: cw * 12, h: 26, range: [-0.8, 0.8], ramp: t => C.diverge(2 * t - 1) });
      vals.forEach((v, k) => txt(g, LA.fmt(v), sx + (k + 0.5) * cw, sy + 13, { anchor: 'center', size: 10, color: 'text' }));
      const hw = (S.w - 90) / 2, hh = Math.min(120, (S.h - 260) / 2 - 40);
      DS.forEach((d, k) => {
        const hx = 30 + (k % 2) * (hw + 30), hy = 150 + Math.floor(k / 2) * (hh + 70);
        histPanel(g, s.hists[k], hx, hy, hw, hh, 'd = ' + d + ' · std ' + LA.fmt(stdOf(s.hists[k])) + ' (1/√d = ' + LA.fmt(1 / Math.sqrt(d)) + ')');
      });
      txt(g, 'cosines of 2000 random pairs of unit vectors (synthetic)', 30, S.h - 46, { size: 11.5, color: 'muted' });
      txt(g, 'The picture stops at 3. The formula does not.', 30, S.h - 22, { size: 13, color: 'gold' });
      return;
    }
    if (p === 'ml') {
      const top = s.rTop5;
      txt(g, 'one query vs 200 synthetic documents in ℝ⁶⁴ (5 planted topics)', 24, 30, { size: 12.5 });
      hbars(g, top.map((t, k) => ({ label: '#' + (k + 1) + ' topic ' + t[1], val: t[0], color: t[1] === 0 ? 'gold' : 'dataB' })), { x: 24, y: 64, w: Math.min(420, S.w - 60), min: 0, max: 1, labelW: 96, rowH: 22, title: 'top 5 by cosine (query is topic 0)' });
      txt(g, 'all vectors have length 1, so ranking by distance would give the same top 5', 24, 64 + 5 * 22 + 14, { size: 11.5, color: 'muted' });
      const cx = 46, cy = 250, cw = Math.min(600, S.w - 220), ch = Math.min(240, S.h - 380);
      txt(g, '1', cx - 8, cy + ch * (1 - 1 / 1.05), { anchor: 'right', size: 10, color: 'muted' });
      txt(g, '0', cx - 8, cy + ch, { anchor: 'right', size: 10, color: 'muted' });
      const xs = [3, 4, 6, 8, 10];
      g.lineChart([
        { points: xs.map((x, k) => [x, s.rRatio[k]]), color: 'teal', label: 'nearest/farthest' },
        { points: xs.map((x, k) => [x, s.rPrec[k]]), color: 'gold', label: 'precision@5' },
      ], { x: cx, y: cy, w: cw, h: ch, xRange: [3, 10], yRange: [0, 1.05], title: 'as d grows (log scale): d = 8, 16, 64, 256, 1024' });
      xs.forEach((x, k) => txt(g, String(s.rRatio.length ? [8, 16, 64, 256, 1024][k] : ''), cx + (x - 3) / 7 * cw, cy + ch + 26, { anchor: 'center', size: 10, color: 'muted' }));
      txt(g, 'teal: random vectors, Euclidean nearest/farthest ratio → 1', 24, cy + ch + 52, { size: 11.5, color: 'teal' });
      txt(g, 'gold: share of the top 5 by cosine in the query’s topic', 24, cy + ch + 70, { size: 11.5, color: 'gold' });
      txt(g, 'see p6-03: attention is a soft version of this lookup', S.w - 20, S.h - 22, { size: 13, color: 'gold', anchor: 'right' });
      return;
    }
    if (p === 'play' || p === 'probeCheck' || p === 'short') {
      const r = ranking(s, s.probe, null, s.useEuclid);
      const mx = s.useEuclid ? Math.max(...r.map(x => x.val)) : 1;
      hbars(g, r.map((it, k) => ({ label: it.word, val: it.val, color: k === 0 ? 'gold' : 'data' })), { x: x0, y: y0 + 16, w: pw, min: s.useEuclid ? 0 : -1, max: mx, title: s.useEuclid ? 'distance to probe (smallest first)' : 'cos(probe, ·), sorted' });
      let yy = y0 + 16 + 10 * 17 + 20;
      txt(g, '‖probe‖ = ' + LA.fmt(N.norm(s.probe)), x0, yy, { color: p === 'short' && N.norm(s.probe) >= 0.5 ? 'bad' : 'mag' });
      if (p === 'play') {
        const k = s.dSel;
        histPanel(g, s.hists[k], x0, yy + 50, pw, 70, 'random cosines, d = ' + DS[k]);
      }
      return;
    }
    if (p === 'girl') {
      const g1 = N.sub(w('girl'), w('boy')), q1 = N.sub(w('queen'), w('king'));
      txt(g, 'girl − boy = (' + g1.map(v => LA.fmt(v)).join(', ') + ')', x0, y0 + 10, { color: 'w' });
      txt(g, 'queen − king = (' + q1.map(v => LA.fmt(v)).join(', ') + ')', x0, y0 + 32, { color: 'gold' });
      txt(g, 'gap ' + LA.fmt(N.dist(g1, q1)), x0, y0 + 56, { color: 'muted' });
      return;
    }
  }
})();
