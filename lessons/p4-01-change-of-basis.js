/* p4-01 — Same vector, different coordinates (change of basis: B, B⁻¹, B⁻¹AB, eigenbasis). */
(function () {
  'use strict';
  const N = LA.num, lerpM = LA.lerpM, lerpV = LA.lerpV, seg = LA.seg, C = LA.color;
  const I2 = () => [[1, 0], [0, 1]];
  const ROT90 = [[0, -1], [1, 0]];
  const SYM = [[2, 1], [1, 2]];
  const SHEAR = [[1, 1], [0, 1]];
  const ALICE = { b1: [2, 1], b2: [-1, 1] };
  const EIG = { b1: [1, 1], b2: [1, -1] };
  const ihat = '<span class="c-i">î</span>', jhat = '<span class="c-j">ĵ</span>';
  const vS = '<span class="c-v">v</span>', b1S = '<span class="c-w">b₁</span>', b2S = '<span class="c-w">b₂</span>';
  const HONEST = 'The picture stops at 3. The formula does not.';

  /* ---- scene helpers: the whole picture is drawn through S = V·M, V = "whose view" (I = ours, B⁻¹ = Alice's),
     M = the motion applied so far. hop ∈ [0,3]: 0 Alice's view → 1 ours (B applied) → 2 moved (A applied) → 3 Alice's view of the result (B⁻¹ applied). */
  const basisM = s => N.fromCols([s.b1, s.b2]);
  const viewT = hop => hop < 1 ? 1 - hop : hop > 2 ? hop - 2 : 0;
  const motionT = hop => LA.clamp(hop - 1, 0, 1);
  const motionM = (s, m) => s.Akind === 'rot' ? N.rot2(s.Aangle * m) : lerpM(I2(), s.A, m);
  function sceneMats(s) {
    const B = basisM(s), Binv = N.inv(B);
    const vt = viewT(s.hop), mt = motionT(s.hop);
    const V = Binv ? N.matPolar(I2(), Binv, vt) : I2();
    const M = motionM(s, mt);
    return { B, Binv, V, M, S: N.matMul(V, M), vt, mt };
  }
  const fmtB = v => '[' + LA.fmt(v[0]) + '; ' + LA.fmt(v[1]) + ']';
  const hopStage = hop => hop < 0.999 ? 1 : hop < 1.001 ? 0 : hop < 2.001 ? 2 : 3;   // which tile is "lit"
  function eigText(A) {
    const e = N.eig2(A);
    if (!e.real) return 'λ complex, |λ| = ' + LA.fmt(Math.hypot(e.re, e.im));
    return 'λ = ' + LA.fmt(e.values[0]) + ', ' + LA.fmt(e.values[1]);
  }
  function makeML() {
    const cloud = LA.rng.gaussianCloud(41, 40, { mean: [0, 0], cov: [[2, 1.2], [1.2, 1]] });
    const pc = N.pca(cloud, 2);
    const ang = Math.atan2(pc.components[0][1], pc.components[0][0]);
    const rand = LA.rng(43);
    const A6 = N.zerosM(6, 6);
    for (let i = 0; i < 6; i++) for (let j = i; j < 6; j++) { const x = LA.rng.gauss(rand); A6[i][j] = x; A6[j][i] = x; }
    const R6 = N.zerosM(6, 6); for (let i = 0; i < 6; i++) for (let j = 0; j < 6; j++) R6[i][j] = LA.rng.gauss(rand);
    const B6 = N.qr(R6).Q;
    const A6new = N.matMul(N.transpose(B6), N.matMul(A6, B6));
    return { cloud: pc.Z.map(z => [z[0] * pc.components[0][0] + z[1] * pc.components[1][0], z[0] * pc.components[0][1] + z[1] * pc.components[1][1]]), pcAngle: ang, A6, A6new };
  }

  LA.registerLesson({
    id: 'p4-01-change-of-basis',
    part: 4,
    title: 'Same vector, different coordinates',
    subtitle: 'B translates Alice’s coordinates into ours, B⁻¹ translates back, and B⁻¹AB is the same motion in her words.',
    minutes: 10,
    prereqs: ['p1-03-linear-combinations-span', 'p2-07-inverse-solving', 'p2-04-composition'],
    tags: ['core'],
    badge: { drawnIn: '2D', realSize: 'a 768-dimensional hidden layer “rotated” into PCA coordinates is B⁻¹AB with a 768×768 B', transfers: 'B converts new → old, B⁻¹ old → new, B⁻¹AB is the same map in new coordinates, in any dimension' },
    cantShow: ['bases of more than three vectors; the two-grid overlay is strictly 2D (3D would need three skewed grids)', 'non-uniqueness of the basis choice beyond two examples'],
    terms: [
      { term: 'basis', def: 'A set of arrows every vector can be written from in exactly one way.' },
      { term: 'coordinates', def: 'The recipe numbers for a vector in a given basis: how many of each basis arrow to add.' },
      { term: 'change-of-basis matrix B', def: 'The matrix whose columns are the new basis vectors written in the old coordinates; it converts new coordinates into old ones.' },
      { term: 'similar matrices', def: 'A and B⁻¹AB: the same transformation seen in two bases, so they share determinant and eigenvalues.' },
      { term: 'eigenbasis', def: 'A basis made of eigenvectors, in which the transformation is a diagonal matrix: pure stretching along each basis arrow.' },
    ],
    atlas: ['pca', 'embeddings'],
    thumbnail: 7,

    state() {
      const ml = makeML();
      return {
        view: '2d', cam: LA.cam.defaults2d(),
        b1: [2, 1], b2: [-1, 1], v: [1, 2],
        showOurGrid: true, showHerGrid: true, herAlpha: 1, showIJ: true, basisGold: 0,
        A: LA.clone(ROT90), Akind: 'rot', Aangle: Math.PI / 2, hop: 1,
        recipe: 0, ring: false, colTag: 0, readouts: true, showDet: false, showAngle: false, showImageGrid: false,
        wrong: 0, wrongAlpha: 1, wrongCross: false, okCheck: false,
        scene: 'main', mlT: 0, cloud: ml.cloud, pcAngle: ml.pcAngle, A6: ml.A6, A6new: ml.A6new,
        panel: 'coords', hideV: false, stretchTags: false,
      };
    },

    render(g, s, ctx) {
      if (s.scene === 'ml') { renderML(g, s); return; }
      if (s.scene === 'recap') { renderRecap(g, s); return; }
      renderMain(g, s, ctx);
    },

    math(s, tex) {
      // live numbers as plain cells: tex.num values are re-synced by position into every .tx-v element (matrix
      // cells included), which corrupts matrices that precede them; plain cells simply rebuild when they change
      const num = (x, o) => '<span class="tx-cell ' + tex.colorClass(o && o.color) + '">' + (typeof x === 'number' ? LA.fmt(x) : x) + '</span>';
      const inv = x => '<span>' + x + '<sup>&#8202;−1</sup></span>';   // one flex item, so the superscript stays attached
      const B = basisM(s), Binv = N.inv(B);
      const bc = s.basisGold > 0.5 ? 'gold' : 'w', bn = s.basisGold > 0.5 ? 'p' : 'b';
      const c = Binv ? N.matVec(Binv, s.v) : [NaN, NaN];
      const bsym = k => tex.sym(bn, { sub: k, color: bc });
      const Bsym = () => tex.sym(s.basisGold > 0.5 ? 'P' : 'B', { color: bc });
      const vsym = () => tex.sym('v', { color: 'v' });
      const P = [];
      switch (s.panel) {
        case 'coords':
          P.push(tex.eq(vsym(), tex.op('='), num(s.v[0], { color: 'v' }), tex.ihat(), tex.op('+'), num(s.v[1], { color: 'v' }), tex.jhat()));
          P.push(tex.eq(vsym(), tex.op('='), num(c[0], { color: bc }), bsym('1'), tex.op('+'), num(c[1], { color: bc }), bsym('2')));
          break;
        case 'recipe':
          P.push(tex.eq(num(c[0], { color: bc }), tex.vec(s.b1, { color: bc }), tex.op('+'), num(c[1], { color: bc }), tex.vec(s.b2, { color: bc }), tex.op('='), tex.vec(s.v, { color: 'v' })));
          break;
        case 'B':
          P.push(tex.eq(Bsym(), tex.op('='), tex.op('['), bsym('1'), bsym('2'), tex.op(']'), tex.op('='), tex.mat(B, { colors: bc, highlight: (i, j) => s.colTag > j })));
          P.push(tex.eq(Bsym(), tex.vec(c, { color: bc }), tex.op('='), num(c[0], { color: bc }), tex.vec(s.b1, { color: bc }), tex.op('+'), num(c[1], { color: bc }), tex.vec(s.b2, { color: bc }), tex.op('='), tex.vec(s.v, { color: 'v' })));
          break;
        case 'Binv':
          P.push(tex.eq(inv(Bsym()), tex.op('='), Binv ? tex.mat(Binv, { colors: bc }) : tex.text('singular', { color: 'bad' })));
          P.push(tex.eq(inv(Bsym()), tex.vec(s.v, { color: 'v' }), tex.op('='), tex.vec(c, { color: bc })));
          break;
        case 'fixed':
          P.push(tex.eq(vsym(), tex.text(' never moves', { color: 'ok' })));
          P.push(tex.eq(tex.text('ours'), tex.op('='), Bsym(), tex.op('·'), tex.text('Alice’s', { color: bc })));
          P.push(tex.eq(tex.vec(s.v, { color: 'v' }), tex.op('='), Bsym(), tex.vec(c, { color: bc })));
          break;
        case 'three': {
          const st = hopStage(s.hop), Aname = s.Akind === 'rot' ? 'R' : 'A';
          const prod = Binv ? N.matMul(Binv, N.matMul(s.A, B)) : null;
          P.push(tex.eq(tex.sym(Aname, { sub: 'Alice' }), tex.op('='), inv(Bsym()), tex.op('·'), tex.sym(Aname), tex.op('·'), Bsym()));
          P.push(tex.eq(tex.op('='), Binv ? tex.mat(Binv, { colors: bc, highlight: () => st === 3 }) : tex.text('?'), tex.op('·'), tex.mat(s.A, { highlight: () => st === 2 }), tex.op('·'), tex.mat(B, { colors: bc, highlight: () => st === 1 })));
          if (prod) P.push(tex.eq(tex.op('='), tex.mat(prod, { colors: bc })));
          break; }
        case 'similar': {
          const prod = Binv ? N.matMul(Binv, N.matMul(s.A, B)) : null;
          P.push(tex.eq(tex.text('det'), tex.paren(inv(Bsym()) + tex.sym('A') + Bsym()), tex.op('='), num(prod ? N.det(prod) : NaN, { color: bc }), tex.op('='), tex.text('det'), tex.sym('A'), tex.op('='), num(N.det(s.A))));
          P.push(tex.eq(tex.text('eig'), tex.paren(inv(Bsym()) + tex.sym('A') + Bsym()), tex.op('='), tex.text('eig'), tex.sym('A'), tex.op(':'), tex.text(eigText(s.A))));
          break; }
        case 'eigen': {
          const D = Binv ? N.matMul(Binv, N.matMul(s.A, B)) : null;
          P.push(tex.eq(tex.sym('A'), tex.op('='), tex.mat(s.A)));
          P.push(tex.eq(Bsym(), tex.op('='), tex.mat(B, { colors: bc })));
          P.push(tex.eq(tex.sym('D'), tex.op('='), inv(Bsym()), tex.sym('A'), Bsym(), tex.op('='), D ? tex.mat(D, { colors: bc }) : tex.text('singular', { color: 'bad' })));
          break; }
        case 'ml':
          P.push(tex.eq(tex.sym('A', { sub: 'new' }), tex.op('='), inv(tex.sym('B')), tex.sym('A'), tex.sym('B')));
          P.push(tex.text('6×6 here; 768×768 for a hidden layer'));
          break;
        case 'play': {
          const prod = Binv ? N.matMul(Binv, N.matMul(s.A, B)) : null;
          P.push(tex.eq(tex.text('us'), tex.vec(N.matVec(motionM(s, motionT(s.hop)), s.v), { color: 'v' }), tex.text('Alice', { color: bc }), tex.vec(Binv ? N.matVec(Binv, N.matVec(motionM(s, motionT(s.hop)), s.v)) : [NaN, NaN], { color: bc })));
          P.push(tex.eq(Bsym(), tex.op('='), tex.mat(B, { colors: bc }), inv(Bsym()), tex.op('='), Binv ? tex.mat(Binv, { colors: bc }) : tex.text('singular', { color: 'bad' })));
          P.push(tex.eq(inv(Bsym()), tex.sym('A'), Bsym(), tex.op('='), prod ? tex.mat(prod, { colors: bc }) : tex.text('?')));
          P.push(tex.eq(tex.text('det'), tex.op(':'), num(N.det(s.A)), tex.text('(ours)'), num(prod ? N.det(prod) : NaN, { color: bc }), tex.text('(Alice)', { color: bc })));
          break; }
        default:
          P.push(tex.eq(inv(tex.sym('B')), tex.sym('A'), tex.sym('B')));
      }
      return tex.block.apply(null, P);
    },

    steps: [
      { title: 'Two people, one arrow', duration: 3.5,
        narration: 'Two people describe the same yellow arrow ' + vS + '. We use the green ' + ihat + ' and the red ' + jhat + ' and say [1; 2]. Alice uses her own purple arrows ' + b1S + ' = [2; 1] and ' + b2S + ' = [−1; 1] — her <b>basis</b> — and says [1; 1]. Same arrow, different numbers, because different rulers.',
        setup(s) { Object.assign(s, { b1: [2, 1], b2: [-1, 1], v: [1, 2], hop: 1, herAlpha: 0, recipe: 0, ring: false, colTag: 0, readouts: true, panel: 'coords', scene: 'main', showIJ: true, basisGold: 0 }); },
        at(s, t) { s.herAlpha = seg(t, 0.1, 0.55); } },

      { title: 'Alice’s recipe', duration: 3,
        narration: 'Alice’s [1; 1] means one step along ' + b1S + ', then one step along ' + b2S + '. Tip to tail: [2; 1] + [−1; 1] = [1; 2], exactly the tip of ' + vS + '. Her numbers are <b>coordinates</b>: a recipe. Her basis arrows are the ingredients; our numbers are the result.',
        setup(s) { s.herAlpha = 1; s.panel = 'recipe'; s.recipe = 0; s.ring = false; },
        at(s, t) { s.recipe = seg(t, 0.05, 0.8, 'linear'); s.ring = t > 0.85; } },

      { title: 'B translates Alice → us', duration: 3.5,
        narration: 'That recipe is a matrix–vector product. Put her basis arrows side by side as columns: the <b>change-of-basis matrix B</b> = [b₁ b₂] = [[2, −1], [1, 1]]. Then <i>our</i> coordinates = B · (<i>her</i> coordinates): B·[1; 1] = [1; 2]. B is a dictionary from Alice’s words into ours.',
        setup(s) { s.panel = 'B'; s.recipe = 1; s.ring = false; s.colTag = 0; },
        at(s, t) { s.colTag = t > 0.6 ? 2 : t > 0.25 ? 1 : 0; s.ring = t > 0.85; } },

      { title: 'B⁻¹ translates us → Alice', duration: 4,
        narration: 'To go the other way, run the dictionary backwards: her coordinates = <b>B⁻¹</b> · our coordinates. Watch what B⁻¹ does to the picture: it straightens Alice’s skewed purple grid into an ordinary square grid, our blue grid skews instead, and ' + vS + ' lands on [1; 1]. Then we ease back — nothing really moved.',
        setup(s) { s.panel = 'Binv'; s.recipe = 0; s.ring = false; s.colTag = 0; s.hop = 1; },
        at(s, t) { s.hop = 1 - seg(t, 0.05, 0.45) + seg(t, 0.7, 1); } },

      { title: 'Misconception: changing basis moves the arrow', duration: 3.5, misconception: true,
        narration: '✕ “Multiplying by B moves ' + vS + ' somewhere new.” No. The arrow never moves; B only changes which <i>numbers</i> we attach to it. Apply B to the arrow itself and you get a different arrow, [0; 3] — the wrong idea, drawn in red and crossed out. The right picture: one arrow, two labels.',
        setup(s) { s.panel = 'fixed'; s.hop = 1; s.wrong = 0; s.wrongAlpha = 1; s.wrongCross = false; s.okCheck = false; },
        at(s, t) { s.wrong = seg(t, 0.08, 0.45); s.wrongCross = t > 0.45; s.wrongAlpha = 1 - 0.75 * seg(t, 0.75, 1); s.okCheck = t > 0.6; } },

      { title: 'A transformation in Alice’s words: three tweens', duration: 6,
        narration: 'We write a quarter-turn as <b>R</b> = [[0, −1], [1, 0]]. How does Alice write the same motion? Take her coordinates to ours (B), rotate (R), translate back (B⁻¹). Three moves, one matrix: <b>B⁻¹RB</b> = [[0.33, −0.67], [1.67, −0.33]]. Strange numbers, same motion — watch ' + vS + ' take the three hops.',
        setup(s) { s.panel = 'three'; s.wrong = 0; s.wrongCross = false; s.okCheck = false; s.A = LA.clone(ROT90); s.Akind = 'rot'; s.Aangle = Math.PI / 2; s.hop = 1; s.showIJ = false; },
        at(s, t) { s.hop = 1 - seg(t, 0, 0.12) + seg(t, 0.17, 0.4) + seg(t, 0.44, 0.68) + seg(t, 0.72, 0.96); } },

      { title: 'Same motion, different eyes: what stays the same', duration: 3.5,
        narration: 'Different numbers, same motion — so anything that is <i>about the motion</i> must agree. The determinant (area factor) is 1 in both descriptions, and the eigenvalues (next lesson) are the same. Matrices related by B⁻¹AB are called <b>similar matrices</b>: one transformation written in two languages.',
        setup(s) { s.panel = 'similar'; s.hop = 1; s.showDet = true; s.showIJ = true; },
        at(s, t) { s.hop = 1 + seg(t, 0.1, 0.8); } },

      { title: 'The best eyes: an eigenbasis makes A diagonal', duration: 4.5,
        narration: 'Now let A = [[2, 1], [1, 2]]. Choose Alice’s arrows to be the gold directions A merely stretches: <span class="c-gold">p₁</span> = [1; 1] (×3) and <span class="c-gold">p₂</span> = [1; −1] (×1). In that <b>eigenbasis</b> the motion is pure stretching: <b>P⁻¹AP</b> = [[3, 0], [0, 1]], a diagonal matrix. No shear, no turning — just two numbers.',
        setup(s) { s.panel = 'eigen'; s.showDet = false; s.hop = 1; s.A = LA.clone(SYM); s.Akind = 'lin'; s.v = [1, 0.5]; s.showIJ = false; s.hideV = true; s.stretchTags = false; s._b1 = LA.clone(s.b1); s._b2 = LA.clone(s.b2); s.cam = LA.cam.defaults2d(); },
        at(s, t) {
          const u = seg(t, 0, 0.3); s.b1 = lerpV(s._b1, EIG.b1, u); s.b2 = lerpV(s._b2, EIG.b2, u); s.basisGold = u; s.hop = 1 + seg(t, 0.4, 0.9);
          s.stretchTags = t > 0.9;
          LA.cam.tween(s, { center: [0, 0], halfH: 3.6 }, { center: [0.9, 0.7], halfH: 4.5 }, seg(t, 0.3, 0.5));
        } },

      { title: 'Why ML cares: coordinates are a choice', duration: 4, mlTie: true,
        narration: 'PCA is exactly this: rotate the data into the basis where its spread lies along the axes — the gold frame. An embedding space is only meaningful up to a change of basis: rotate every word vector together and nothing changes. The picture is 2D; the right panel is a 6-dimensional B⁻¹AB. ' + HONEST,
        setup(s) { s.scene = 'ml'; s.panel = 'ml'; s.mlT = 0; s.stretchTags = false; },
        at(s, t) { s.mlT = seg(t, 0.1, 0.75); } },

      { title: 'Recap', duration: 2.5, recap: true,
        narration: 'Columns of B are the new basis written in old coordinates; B converts new → old, B⁻¹ converts old → new. The arrow never moves; only its numbers do. A transformation seen in the new basis is B⁻¹AB; in an eigenbasis it is diagonal.',
        cards: ['<b>Same arrow, two labels.</b> We say [1; 2], Alice says [1; 1]; the arrow itself never moved.', '<b>B⁻¹AB = same motion, new words.</b> Into our coordinates (B), move (A), back into hers (B⁻¹).', '<b>Eigenbasis ⇒ diagonal.</b> P⁻¹AP = diag(3, 1): pure stretching, no shear, no turning.'],
        setup(s) { s.scene = 'recap'; s.panel = 'recap'; },
        at(s, t) { } },
    ],

    playground: {
      intro: '<p>Drag the purple ' + b1S + ', ' + b2S + ' or the yellow ' + vS + '. Slide the hop to watch B⁻¹·A·B: Alice’s view → ours → moved → back to hers. Try a basis and a transformation preset.</p>',
      setup(s) { Object.assign(s, { scene: 'main', panel: 'play', hop: 1, recipe: 1, readouts: true, showIJ: true, herAlpha: 1, basisGold: 0, wrong: 0, wrongCross: false, okCheck: false, showDet: false, hideV: false, stretchTags: false }); },
      controls: [
        { type: 'matrix', bind: 'b1', label: 'b₁', colors: 'w', step: 0.1 },
        { type: 'matrix', bind: 'b2', label: 'b₂', colors: 'w', step: 0.1 },
        { type: 'slider', bind: 'hop', min: 0, max: 3, step: 0.01, label: 'hop: Alice → B → A → B⁻¹', fmt: x => x <= 0.01 ? 'Alice’s view' : x < 1 ? 'B…' : x <= 1.01 ? 'our view' : x < 2 ? 'A…' : x <= 2.01 ? 'moved' : x < 3 ? 'B⁻¹…' : 'Alice’s view' },
        { type: 'toggle', bind: 'showHerGrid', label: 'Alice’s grid' },
        { type: 'preset', label: 'Transformation A', items: [
          { label: 'Rotation 90°', apply(s) { s.A = LA.clone(ROT90); s.Akind = 'rot'; s.Aangle = Math.PI / 2; } },
          { label: 'Shear', apply(s) { s.A = LA.clone(SHEAR); s.Akind = 'lin'; } },
          { label: 'A = [[2,1],[1,2]]', apply(s) { s.A = LA.clone(SYM); s.Akind = 'lin'; } },
        ] },
        { type: 'preset', label: 'Basis', items: [
          { label: 'Alice [2;1], [−1;1]', apply(s) { s.b1 = [2, 1]; s.b2 = [-1, 1]; s.basisGold = 0; } },
          { label: 'Eigenbasis of [[2,1],[1,2]]', apply(s) { s.b1 = [1, 1]; s.b2 = [1, -1]; s.basisGold = 1; s.A = LA.clone(SYM); s.Akind = 'lin'; } },
          { label: 'Orthonormal (rotated 30°)', apply(s) { const c = Math.cos(Math.PI / 6), sn = Math.sin(Math.PI / 6); s.b1 = [c, sn]; s.b2 = [-sn, c]; s.basisGold = 0; } },
        ] },
      ],
    },

    checks: [
      { kind: 'manipulate',
        prompt: 'Drag the yellow ' + vS + ' so that Alice reads [2; −1] (her basis is ' + b1S + ' = [2; 1], ' + b2S + ' = [−1; 1]).',
        setup(s) { Object.assign(s, { hideV: false, stretchTags: false, scene: 'main', panel: 'coords', b1: [2, 1], b2: [-1, 1], v: [1, 2], hop: 1, recipe: 1, readouts: true, herAlpha: 1, showIJ: true, basisGold: 0 }); s.cam = Object.assign(LA.cam.defaults2d(), { center: [1.4, 0.4], halfH: 4.4 }); },
        judge(s) { const Binv = N.inv(basisM(s)); if (!Binv) return { ok: false, progress: 0 }; return LA.judge.near(N.matVec(Binv, s.v), [2, -1], LA.TOL.pos); },
        hint: 'Two steps along ' + b1S + ', then one step <i>backwards</i> along ' + b2S + ': 2·[2; 1] − 1·[−1; 1] = [5; 1].',
        solution(s) { s.v = [5, 1]; } },
      { kind: 'manipulate',
        prompt: 'Drag ' + b2S + ' so that Alice’s grid is orthogonal: ' + b1S + ' and ' + b2S + ' at a right angle.',
        setup(s) { Object.assign(s, { hideV: false, stretchTags: false, scene: 'main', panel: 'coords', b1: [2, 1], b2: [-1, 1], v: [1, 2], hop: 1, recipe: 0, readouts: false, herAlpha: 1, showIJ: false, basisGold: 0, showAngle: true }); },
        judge(s) { const d = N.dot(s.b1, s.b2), n = N.norm(s.b1) * N.norm(s.b2); if (n < 0.1) return { ok: false, progress: 0, hint: 'Keep both arrows longer than zero.' }; const c = Math.abs(d) / n; return { ok: c < 0.05, progress: Math.max(0, 1 - c / 0.6) }; },
        hint: 'Perpendicular to [2; 1] is along [−1; 2] (or [1; −2]).',
        solution(s) { s.b2 = [-1, 2]; } },
      { kind: 'manipulate',
        prompt: 'Drag ' + b1S + ' and ' + b2S + ' onto the two directions that A = [[2, 1], [1, 2]] only stretches, so that P⁻¹AP becomes diagonal (the gold image grid lies on top of the purple grid).',
        setup(s) { Object.assign(s, { hideV: false, stretchTags: false, scene: 'main', panel: 'eigen', b1: [2, 1], b2: [-1, 1], v: [1, 0.5], hop: 1, recipe: 0, readouts: false, herAlpha: 1, showIJ: false, basisGold: 0, showAngle: false, showImageGrid: true, A: LA.clone(SYM), Akind: 'lin' }); },
        judge(s) { const B = basisM(s), Binv = N.inv(B); if (!Binv || Math.abs(N.det(B)) < 0.1) return { ok: false, progress: 0, hint: 'The two arrows must not lie on one line.' }; const D = N.matMul(Binv, N.matMul(s.A, B)); const off = Math.max(Math.abs(D[0][1]), Math.abs(D[1][0])); return { ok: off < 0.12, progress: Math.max(0, 1 - off / 1.5) }; },
        hint: 'The columns must be the directions A only stretches: [1; 1] and [1; −1] (any lengths).',
        solution(s) { s.b1 = [1, 1]; s.b2 = [1, -1]; } },
    ],
  });

  /* ---------------- main scene ---------------- */
  function renderMain(g, s, ctx) {
    const { B, Binv, V, M, S, mt } = sceneMats(s);
    const gold = s.basisGold;
    const herColor = gold > 0 ? C.mix(C.w, C.gold, gold) : 'w';
    const bn = gold > 0.5 ? 'p' : 'b';
    g.ghostGrid();
    if (s.showOurGrid) g.grid(S, { alpha: 0.95 });
    if (s.showHerGrid && s.herAlpha > 0.01 && Binv) g.grid(N.matMul(S, B), { color: herColor, alpha: 0.4 * s.herAlpha, width: 1.1 });
    if (s.showImageGrid) g.grid(N.matMul(S, N.matMul(s.A, B)), { color: 'gold', alpha: 0.45, width: 1.1, dash: [5, 4] });
    const sv = N.matVec(S, s.v), sb1 = N.matVec(S, s.b1), sb2 = N.matVec(S, s.b2);
    // the recipe: c1 copies of b1, then c2 copies of b2, tip to tail
    if (s.recipe > 0 && Binv) {
      const c = N.matVec(Binv, s.v);
      const u1 = seg(s.recipe, 0, 0.5, 'linear'), u2 = seg(s.recipe, 0.5, 1, 'linear');
      const p1 = N.scale(sb1, c[0]);
      g.lineThrough([0, 0], sb1, { color: herColor, alpha: 0.5, width: 1.6 });
      g.lineThrough(p1, sb2, { color: herColor, alpha: 0.5, width: 1.6 });
      if (u1 > 0.02) g.vector([0, 0], N.scale(p1, u1), { color: herColor, width: 4.2, alpha: 0.85, head: 13, label: LA.fmt(c[0]) + '·' + bn + '₁', labelAt: 0.5, labelOffset: -16, labelFont: 'ui', labelSize: 12.5 });
      if (u2 > 0.02) g.vector(p1, N.add(p1, N.scale(N.scale(sb2, c[1]), u2)), { color: herColor, width: 4.2, alpha: 0.85, head: 13, label: LA.fmt(c[1]) + '·' + bn + '₂', labelAt: 0.5, labelOffset: -16, labelFont: 'ui', labelSize: 12.5 });
      if (s.ring) g.marker.ring(sv, { r: 13 });
    }
    if (mt > 0.01 && !s.hideV) g.vector(N.matVec(V, s.v), { color: 'v', dashed: true, alpha: 0.4, width: 2, head: 10 });
    if (s.showIJ) {
      g.vector(N.col(S, 0), { color: 'i', label: 'i', hat: true, width: 2.2, head: 10 });
      g.vector(N.col(S, 1), { color: 'j', label: 'j', hat: true, width: 2.2, head: 10 });
    }
    // once the motion has been applied, the basis arrows are carried along: label them R·b₁ / A·p₁ and keep a dashed ghost of the originals
    const mPre = mt > 0.5 ? (s.Akind === 'rot' ? 'R·' : 'A·') : '';
    if (mt > 0.01) {
      g.vector(N.matVec(V, s.b1), { color: herColor, dashed: true, alpha: 0.45, width: 2, head: 9 });
      g.vector(N.matVec(V, s.b2), { color: herColor, dashed: true, alpha: 0.45, width: 2, head: 9 });
    }
    g.vector(sb1, { color: herColor, label: mPre + bn + '₁', width: 3, labelFont: 'math' });
    g.vector(sb2, { color: herColor, label: mPre + bn + '₂', width: 3, labelFont: 'math' });
    if (s.panel === 'three') {
      const cap = s.hop < 0.02 ? 'Alice’s view (her grid drawn square)' : s.hop < 1 ? 'hop 1 · B: Alice’s view → our view' : s.hop < 1.02 ? 'our view' : s.hop < 2 ? 'hop 2 · R: rotate a quarter-turn' : s.hop < 2.98 ? 'hop 3 · B⁻¹: back to Alice’s view' : 'done: the same quarter-turn, in Alice’s view';
      g.hud(cap, 14, 22, { color: 'hud', size: 13, bold: true });
    }
    // column tags sit beside the middle of each shaft (below b₁, left of b₂) so they never cover the tip labels
    if (s.colTag > 0) g.label('column 1', N.scale(sb1, 0.5), { color: herColor, size: 12, dx: 8, dy: 18, font: 'ui' });
    if (s.colTag > 1) g.label('column 2', N.scale(sb2, 0.5), { color: herColor, size: 12, dx: -30, dy: 6, font: 'ui' });
    if (s.showAngle) g.arc([0, 0], sb1, sb2, { r: 0.55, color: 'hud', right: true, label: LA.fmt.deg(N.angle(s.b1, s.b2)) });
    if (!s.hideV) g.vector(sv, { color: 'v', label: 'v', width: 3.2 });
    if (s.wrong > 0) {
      const tip = lerpV(s.v, N.matVec(B, s.v), s.wrong);
      g.vector(tip, { color: 'bad', dashed: true, alpha: s.wrongAlpha, width: 2.4, label: 'Bv', labelFont: 'math' });
      if (s.wrongCross) g.marker.cross(tip, { dx: 46, dy: 6, size: 22 });
    }
    if (s.okCheck) g.marker.check(sv, { dx: -26, dy: -14 });
    if (s.readouts && !s.hideV) {
      const ours = N.matVec(M, s.v);
      g.label('us: ' + fmtB(ours), sv, { color: 'white', anchor: 'left', dx: 16, dy: -18, size: 13, font: 'ui' });
      g.label('Alice: ' + (Binv ? fmtB(N.matVec(Binv, ours)) : '—'), sv, { color: herColor, anchor: 'left', dx: 16, dy: 2, size: 13, font: 'ui' });
    }
    if (s.showDet && Binv) {
      const prod = N.matMul(Binv, N.matMul(s.A, B));
      g.hud('us:     det = ' + LA.fmt(N.det(s.A)) + '   ' + eigText(s.A), 14, 22, { color: 'white' });
      g.hud('Alice: det = ' + LA.fmt(N.det(prod)) + '   ' + eigText(prod), 14, 42, { color: herColor });
    }
    const Sinv = N.inv(S);
    if (s.stretchTags) {
      g.label('×3 along p₁', N.scale(sb1, 0.62), { color: 'gold', size: 12.5, dx: -44, dy: -4, font: 'ui' });
      g.label('×1 along p₂', N.scale(sb2, 0.6), { color: 'gold', size: 12.5, dx: 46, dy: -4, font: 'ui' });
    }
    if (Sinv) {
      if (!s.hideV) g.handle('v', sv, (p, st) => { st.v = N.matVec(Sinv, p); }, { color: 'v' });
      g.handle('b1', sb1, (p, st) => { st.b1 = N.matVec(Sinv, p); }, { color: 'w' });
      g.handle('b2', sb2, (p, st) => { st.b2 = N.matVec(Sinv, p); }, { color: 'w' });
    }
  }

  /* ---------------- ML tie-in: PCA rotation (left) and a 6×6 B⁻¹AB (right) ---------------- */
  function renderML(g, s) {
    const u = s.mlT;
    g.viewport({ x: 0, y: 0, w: 1, h: 0.56 }, { view: '2d', cam: { center: [0, -0.25], halfH: 2.75 }, border: true }, (g2, r) => {
      g2.ghostGrid();
      const Rm = N.rot2(-s.pcAngle * u);
      const pts = s.cloud.map(p => N.matVec(Rm, p));
      g2.points(pts, { r: 3.5, color: 'data', alpha: 0.9 });
      const a1 = N.matVec(Rm, [Math.cos(s.pcAngle), Math.sin(s.pcAngle)]), a2 = [-a1[1], a1[0]];
      g2.lineThrough([0, 0], a1, { color: 'gold', width: 2, alpha: 0.9, len: 5 });
      g2.lineThrough([0, 0], a2, { color: 'gold', width: 1.4, alpha: 0.7, len: 2.5 });
      g2.vector(N.scale(a1, 2.6), { color: 'gold', label: 'PC₁', width: 2.4, head: 10, labelFont: 'ui' });
      g2.vector(N.scale(a2, 1.5), { color: 'gold', label: 'PC₂', width: 2.4, head: 10, labelFont: 'ui' });
      g2.hud('40 synthetic points (seed 41) rotated into the PCA basis', r.x + 12, r.y + 18, { size: 12, color: 'muted' });
      g2.hud('rotation: ' + LA.fmt.deg(-s.pcAngle * u), r.x + 12, r.y + 36, { size: 12, color: 'gold' });
    });
    g.viewport({ x: 0, y: 0.56, w: 1, h: 0.44 }, { view: '2d', cam: { center: [0, 0], halfH: 1.9 }, border: true, bg: 'panel' }, (g2, r) => {
      const m = 2.2;
      const ramp = t => C.diverge(2 * t - 1);
      g2.image(s.A6, { x0: -2.15, x1: -0.55, y0: -0.95, y1: 0.65 }, { ramp, range: [-m, m], cells: true });
      g2.label('A — a 6×6 map', [-1.35, 0.65], { anchor: 'above', dy: -6, size: 12.5, color: 'hud', font: 'ui' });
      g2.vector([-0.4, -0.15], [0.4, -0.15], { color: 'gold', width: 1.8, head: 9, alpha: 0.6 + 0.4 * seg(u, 0.4, 1) });
      g2.label('B⁻¹ · A · B', [0, 0.12], { size: 13, color: 'gold', font: 'ui', bold: true });
      g2.image(s.A6new, { x0: 0.55, x1: 2.15, y0: -0.95, y1: 0.65 }, { ramp, range: [-m, m], cells: true, alpha: 0.15 + 0.85 * seg(u, 0.4, 1) });
      g2.label('same map, new coordinates', [1.35, 0.65], { anchor: 'above', dy: -6, size: 12.5, color: 'gold', font: 'ui' });
      g2.hud('6 dims — ' + HONEST, r.x + 12, r.y + r.h - 16, { size: 12.5, color: 'hud', bold: true });
    });
  }

  /* ---------------- recap: three mini-cards stacked ---------------- */
  function renderRecap(g, s) {
    const B = N.fromCols([ALICE.b1, ALICE.b2]);
    g.viewport({ x: 0, y: 0, w: 1, h: 1 / 3 }, { view: '2d', cam: { center: [0.6, 0.9], halfH: 2.2 }, border: true }, (g2, r) => {
      g2.ghostGrid(); g2.grid(B, { color: 'w', alpha: 0.4, width: 1 });
      g2.vector([2, 1], { color: 'w', label: 'b₁', width: 2.4, head: 10 }); g2.vector([-1, 1], { color: 'w', label: 'b₂', width: 2.4, head: 10 });
      g2.vector([1, 2], { color: 'v', label: 'v', width: 3 });
      g2.label('us: [1; 2]', [1, 2], { color: 'white', anchor: 'left', dx: 16, dy: -18, size: 13, font: 'ui' });
      g2.label('Alice: [1; 1]', [1, 2], { color: 'w', anchor: 'left', dx: 16, dy: 2, size: 13, font: 'ui' });
      g2.hud('Same arrow, two labels', 12, 20, { size: 13, color: 'hud', bold: true });
    });
    g.viewport({ x: 0, y: 1 / 3, w: 1, h: 1 / 3 }, { view: '2d', cam: { center: [0, 0], halfH: 2.2 }, border: true }, (g2, r) => {
      const y = r.y + r.h * 0.5, x = r.x + r.w / 2;
      const tiles = [['B⁻¹', 'back into hers'], ['A', 'move'], ['B', 'into our coords']];
      tiles.forEach((t, i) => {
        const cx = x + (i - 1) * Math.min(150, r.w * 0.26);
        g2.label(t[0], [cx, y - 8], { screen: true, size: 30, color: i === 1 ? 'text' : 'w', font: 'math', bold: true });
        g2.label(t[1], [cx, y + 26], { screen: true, size: 12, color: 'muted', font: 'ui' });
        if (i < 2) g2.label('·', [cx + Math.min(150, r.w * 0.26) / 2, y - 8], { screen: true, size: 30, color: 'muted', font: 'math' });
      });
      g2.hud('B⁻¹AB = same motion, new words (read right to left)', 12, r.y + 20, { size: 13, color: 'hud', bold: true });
    });
    g.viewport({ x: 0, y: 2 / 3, w: 1, h: 1 / 3 }, { view: '2d', cam: { center: [1.2, 0.85], halfH: 2.6 }, border: true }, (g2, r) => {
      const P = N.fromCols([EIG.b1, EIG.b2]);
      g2.grid(P, { color: 'ghost', alpha: 1, width: 1, range: 6 });
      g2.grid(N.matMul(SYM, P), { color: 'gold', alpha: 0.6, width: 1.2, range: 6 });
      g2.vector([3, 3], { color: 'gold', label: '3·p₁', width: 2.6, head: 10, labelFont: 'ui' });
      g2.vector([1, -1], { color: 'gold', label: '1·p₂', width: 2.6, head: 10, labelFont: 'ui' });
      g2.hud('Eigenbasis ⇒ diagonal: D = P⁻¹AP = diag(3, 1)', 12, r.y + 20, { size: 13, color: 'hud', bold: true });
    });
  }
})();
