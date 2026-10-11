/* p5-01 — The gradient is a vector. Two partial slopes assemble into one arrow, perpendicular to the contour. */
(function () {
  'use strict';
  const N = LA.num, lerpV = LA.lerpV, seg = LA.seg, fmt = LA.fmt;
  const ihat = '<span class="c-i">î</span>', jhat = '<span class="c-j">ĵ</span>';
  const PI = Math.PI;
  const SC = 0.4;                                   // world units drawn per unit of slope
  const CB = { x0: -3.6, x1: 3.6, y0: -3.6, y1: 3.6 };   // contour sampling box
  const B3 = { x0: -3, x1: 3, y0: -3, y1: 3 };
  const TOPDOWN = { yaw: -PI / 2, pitch: 1.55 };     // 3D camera that matches the 2D map (x right, y up)

  // ---- the functions of the lesson (closed-form gradients) ----
  const hillA = p => Math.exp(-((p[0] - 1) * (p[0] - 1) + (p[1] - 0.8) * (p[1] - 0.8)));
  const hillB = p => Math.exp(-((p[0] + 1.2) * (p[0] + 1.2) + (p[1] + 0.6) * (p[1] + 0.6)));
  const FNS = {
    bowl: { f: p => p[0] * p[0] + 4 * p[1] * p[1], grad: p => [2 * p[0], 8 * p[1]], levels: [0.5, 1, 2, 3, 4, 6, 8, 12], zs: 0.2, sb: { x0: -3, x1: 3, y0: -2, y1: 2 }, cup: 12, ay: 0.5, yr: [0, 10], name: 'x<sup>2</sup> + 4y<sup>2</sup>', px: '2x', py: '8y' },
    round: { f: p => p[0] * p[0] + p[1] * p[1], grad: p => [2 * p[0], 2 * p[1]], levels: [0.5, 1, 2, 3, 4, 6, 8, 12], zs: 0.25, sb: B3, cup: 9, ay: 1, yr: [0, 10], name: 'x<sup>2</sup> + y<sup>2</sup>', px: '2x', py: '2y' },
    saddle: { f: p => p[0] * p[0] - p[1] * p[1], grad: p => [2 * p[0], -2 * p[1]], levels: [-8, -4, -2, -1, -0.5, 0.5, 1, 2, 4, 8], zs: 0.25, sb: B3, yr: [-10, 10], name: 'x<sup>2</sup> − y<sup>2</sup>', px: '2x', py: '−2y' },
    hills: { f: p => 2 * hillA(p) + 1.5 * hillB(p), grad: p => { const a = hillA(p), b = hillB(p); return [-4 * a * (p[0] - 1) - 3 * b * (p[0] + 1.2), -4 * a * (p[1] - 0.8) - 3 * b * (p[1] + 0.6)]; }, levels: [0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 1.95], zs: 1.2, sb: B3, yr: [0, 2.2], name: 'two hills', px: '', py: '' },
  };
  const FN = s => FNS[s.fn] || FNS.bowl;
  const cam2 = () => Object.assign(LA.cam.defaults2d(), { halfH: 4 });
  const cam3 = o => Object.assign(LA.cam.defaults3d(), { target: [0, 0, 0.7], dist: 13.4, yaw: -1.9, pitch: 0.61, fov: 36 }, o || {});
  const unit = deg => [Math.cos(deg * PI / 180), Math.sin(deg * PI / 180)];
  const clampP = q => [LA.clamp(q[0], -3, 3), LA.clamp(q[1], -3, 3)];
  // Engine gap: tex.reconcile mis-assigns tex.num values when a panel also holds a tex.mat/vec (cells are not in its
  // value list order). Numbers emitted as plain colored spans are part of the skeleton, so the panel simply rebuilds.
  const numH = (tex, x, o) => { o = o || {}; return '<span class="tx-n ' + tex.colorClass(o.color) + '">' + (typeof x === 'number' ? fmt(x, o.digits) : x) + '</span>'; };
  /** Height surface: the quadratic bowls are drawn as an elliptical cup cut at a level (no clipped walls). */
  function drawSurface(g, F, a) {
    const st = { res: 24, mode: 'both', color: 'grid', alpha: 0.28 * a, wireColor: 'gridAxis', wireAlpha: 0.5 * a };
    if (F.cup) {
      const rc = Math.sqrt(F.cup);
      g.paramSurface((u, v) => { const x = rc * u * Math.cos(v), y = F.ay * rc * u * Math.sin(v); return [x, y, F.zs * F.f([x, y])]; }, { u0: 0, u1: 1, v0: 0, v1: 2 * PI }, st);
    } else g.surface(F.f, F.sb, Object.assign({ zScale: F.zs }, st));
  }
  /** Signed bar in screen px (2D view): zero line in the middle, up = positive (green), down = negative (violet). */
  function signedBar(g, x, y, w, h, val, max) {
    const W = (sx, sy) => g.P.toWorld(sx, sy), mid = y + h / 2, bh = Math.min(1, Math.abs(val) / max) * h / 2;
    g.polygon([W(x - 8, y - 4), W(x + w + 8, y - 4), W(x + w + 8, y + h + 4), W(x - 8, y + h + 4)], { fill: 'panel', alpha: 0.8, stroke: 'line', width: 1 });
    if (bh > 0.5) { const y0 = val >= 0 ? mid - bh : mid, y1 = val >= 0 ? mid : mid + bh; g.polygon([W(x, y0), W(x + w, y0), W(x + w, y1), W(x, y1)], { fill: val >= 0 ? 'ok' : 'neg', alpha: 0.9 }); }
    g.line(W(x - 8, mid), W(x + w + 8, mid), { color: 'hud', width: 1.2 });
    g.label(fmt(val), [x + w / 2, val >= 0 ? mid - bh - 9 : mid + bh + 9], { screen: true, size: 11.5, color: 'hud', font: 'ui', bg: false });
  }
  /** Dashed 3D segment drawn as short solid pieces (canvas line dashes on 3D segments blanked some headless frames). */
  function dash3(g, a, b, o) { const n = 12; for (let i = 0; i < n; i += 2) g.line(lerpV(a, b, i / n), lerpV(a, b, (i + 1) / n), o); }
  function makeStrip() { const rand = LA.rng(501), v = []; for (let i = 0; i < 12; i++) v.push(LA.clamp(LA.rng.gauss(rand) * 0.5, -1, 1)); return v; }

  LA.registerLesson({
    id: 'p5-01-gradient',
    part: 5,
    title: 'The gradient is a vector',
    subtitle: 'Two slopes assemble into an arrow perpendicular to the contour — and it does not point at the minimum.',
    minutes: 10,
    prereqs: ['p1-06-dot-product', 'p4-06-symmetric-quadratic-forms', 'p0-02-reading-the-symbols'],
    tags: ['core'],
    badge: { drawnIn: '2D contour map + 3D surface over two inputs', realSize: '∇L of a small CNN has ~10⁶ entries; of GPT-class models 10⁹–10¹²', transfers: 'one partial slope per axis, ∇f ⟂ level sets, steepest ascent along ∇f, f(p+d) ≈ f(p) + ∇f·d — exact in any n' },
    cantShow: ['Level sets in n inputs are (n−1)-dimensional sheets, not curves.', 'A height surface exists only for two inputs; for n inputs there is no “hill” to draw.', 'Kinks (ReLU) where the gradient is undefined; every f drawn here is smooth.'],
    terms: [
      { term: 'function of two variables', def: 'A rule that turns a point (x, y) into one number f(x, y).' },
      { term: 'contour', def: 'All points with the same value of f (a level set).' },
      { term: 'partial derivative', def: 'The slope of f when only one input moves, written ∂f/∂x.' },
      { term: 'gradient', def: 'The vector ∇f = [∂f/∂x; ∂f/∂y] of all partial slopes.' },
      { term: 'directional derivative', def: 'The slope of f along a unit direction u, equal to ∇f·u.' },
      { term: 'steepest ascent', def: 'The direction ∇f; steepest descent is −∇f.' },
      { term: 'tangent plane', def: 'The flat sheet that touches the surface at a point; its tilt is the gradient.' },
    ],
    atlas: ['optimization', 'backprop'],
    thumbnail: 7,

    state() {
      return {
        view: '2d', cam: cam2(),
        fn: 'bowl', p: [1, 0.5], probe: 0, d: [0.35, 0.2], guess: [0, 0], strip: makeStrip(),
        showField: false, showTangent: false, showPerp: false, showProbe: false, showBar: false, showComponents: false, showGuess: false,
        surfAlpha: 1, showValue: true, goldContour: false, levelLabels: false, axisNames: ['x', 'y'],
        pxGrow: 0, pyGrow: 0, tipToTail: 0, gradGrow: 0, slice: null, fieldFade: 0,
        negGrow: 0, negCross: false, negStep: 0, dGrow: 0, mlStep: 0, stripFade: 0, showStrip: false, wPoint: false,
        recap: false, panel: 'f',
      };
    },

    render(g, s, ctx) {
      if (s.recap) { renderRecap(g, s, ctx); return; }
      const F = FN(s);
      if (s.view === '3d') render3d(g, s, ctx, F); else render2d(g, s, ctx, F);
    },

    math(s, tex) {
      const F = FN(s), gr = F.grad(s.p), fv = F.f(s.p);
      const G = tex.sym('∇f', { color: 'v' }), U = tex.sym('u', { color: 'w' }), D = tex.sym('d', { color: 'w' });
      const H = str => tex.text(str, { html: true });
      const vecIJ = v => tex.vec(v, { colors: i => (i === 0 ? 'i' : 'j') });
      const u = unit(s.probe), slope = N.dot(gr, u);
      tex = Object.assign({}, tex, { num: (x, o) => numH(tex, x, o) });
      switch (s.panel) {
        case 'f': return tex.block(tex.eq(tex.sym('f'), tex.op('('), tex.sym('x', { color: 'i' }), tex.op(','), tex.sym('y', { color: 'j' }), tex.op(')'), tex.op('='), H(F.name)), tex.eq(tex.sym('f'), tex.op('('), tex.num(s.p[0]), tex.op(','), tex.num(s.p[1]), tex.op(')'), tex.op('='), tex.num(fv, { color: 'v' })));
        case 'contour': return tex.block(tex.eq(tex.sym('f'), tex.op('='), tex.num(fv, { color: 'gold' }), tex.text('on the gold curve')));
        case 'px': return tex.block(tex.eq(H('∂f/∂x'), tex.op('='), H(F.px), tex.op('='), tex.num(gr[0], { color: 'i' })));
        case 'py': return tex.block(tex.eq(H('∂f/∂y'), tex.op('='), H(F.py), tex.op('='), tex.num(gr[1], { color: 'j' })));
        case 'grad': return tex.block(tex.eq(G, tex.op('='), tex.vec(['∂f/∂x', '∂f/∂y']), tex.op('='), vecIJ(gr)));
        case 'perp': { const tg = N.normalize([-gr[1], gr[0]]); return tex.block(tex.eq(G, tex.op('·'), tex.text('tangent'), tex.op('='), vecIJ(gr), tex.op('·'), tex.vec(tg, { color: 'muted' }), tex.op('='), tex.num(N.dot(gr, tg)))); }
        case 'dir': return tex.block(tex.eq(tex.text('slope along'), U, tex.op('='), G, tex.op('·'), U, tex.op('='), tex.norm(G), tex.text('cos θ'), tex.op('='), tex.num(slope, { color: slope >= 0 ? 'ok' : 'neg' })));
        case 'field': return tex.block(tex.eq(tex.norm(G), tex.op('='), tex.sqrt(H('(2x)<sup>2</sup> + (8y)<sup>2</sup>')), tex.op('='), tex.num(N.norm(gr), { color: 'v' }), tex.text('at'), tex.text(fmt.vec(s.p))));
        case 'mis': { const toMin = N.scale(s.p, -1); const ang = N.angle(N.scale(gr, -1), toMin) * 180 / PI; return tex.block(tex.eq(tex.op('−'), G, tex.op('='), tex.vec(N.scale(gr, -1), { color: 'v' })), tex.eq(tex.text('to the minimum'), tex.op('='), tex.vec(toMin, { color: 'muted' })), tex.eq(tex.text('angle between'), tex.op('='), tex.num(ang, { color: 'bad', digits: 0 }), tex.op('°'))); }
        case 'tangent': { const q = N.add(s.p, s.d), dd = N.dot(gr, s.d); return tex.lines(tex.eq(H('f(p + <span class="c-w">d</span>)'), tex.op('≈'), H('f(p)'), tex.op('+'), G, tex.op('·'), D), tex.eq(tex.num(F.f(q), { color: 'b' }), tex.op('≈'), tex.num(fv), tex.op('+'), tex.num(dd, { color: 'w' }), tex.op('='), tex.num(fv + dd, { color: 'teal' }))); }
        case 'ml': return tex.block(tex.eq(tex.sym('w', { color: 'b' }), tex.op('←'), tex.sym('w', { color: 'b' }), tex.op('−'), tex.sym('η'), tex.sym('∇L', { color: 'v' })), tex.eq(tex.sym('η'), tex.op('='), tex.num(0.1), tex.op(','), tex.sym('∇L', { color: 'v' }), tex.op('='), vecIJ(gr)));
        case 'guess': return tex.block(tex.eq(tex.sym('f'), tex.op('='), H(F.name), tex.op(','), tex.text('p'), tex.op('='), tex.text(fmt.vec(s.p))), tex.eq(tex.text('your'), G, tex.op('='), tex.vec(s.guess, { color: 'w' })));
        case 'none': return '';
        default: return tex.block(tex.eq(tex.sym('f'), tex.op('='), H(F.name), tex.op(','), tex.sym('f'), tex.paren(tex.num(s.p[0]) + tex.op(',') + tex.num(s.p[1])), tex.op('='), tex.num(fv, { color: 'v' })), tex.eq(G, tex.op('='), vecIJ(gr)), s.showProbe ? tex.eq(tex.text('slope along'), U, tex.op('='), tex.num(slope, { color: slope >= 0 ? 'ok' : 'neg' })) : '');
      }
    },

    steps: [
      { title: 'Height is a function', duration: 3,
        narration: 'A <b>function of two variables</b> <b>f</b> takes a point (<span class="c-i">x</span>, <span class="c-j">y</span>) and returns one number. Draw that number as height and you get a landscape. Here <b>f(x, y) = x² + 4y²</b>: a bowl, steeper along the red ' + jhat + ' than along the green ' + ihat + '. The yellow point (1, 0.5) sits at height 2.',
        setup(s) { s.view = '3d'; s.cam = cam3(); s.fn = 'bowl'; s.p = [1, 0.5]; s.surfAlpha = 1; s.showValue = true; s.goldContour = false; s.levelLabels = false; s.panel = 'f'; s.gradGrow = 0; s.pxGrow = 0; s.pyGrow = 0; s.slice = null; },
        at(s, t) { LA.cam.tween(s, { yaw: -1.9 }, { yaw: -1.2 }, seg(t, 0, 1, 'linear')); } },

      { title: 'The map', duration: 3,
        narration: 'Look straight down and the landscape becomes a map. Each gray curve joins points of equal height: a <b>contour</b>. Close curves mean steep ground. The yellow point sits on the contour f = 2, drawn gold.',
        setup(s) { s._yaw0 = s.cam.yaw; s.cam.center = [0, 0]; s.cam.halfH = 4; s.panel = 'contour'; },
        at(s, t) {
          const u = seg(t, 0, 0.6);
          s.surfAlpha = 1 - seg(t, 0, 0.55);
          if (t < 0.6) { s.view = '3d'; LA.cam.tween(s, { yaw: s._yaw0, pitch: 0.61 }, TOPDOWN, u); }
          else { s.view = '2d'; s.cam.center = [0, 0]; s.cam.halfH = 4; }
          s.goldContour = t > 0.3; s.levelLabels = t >= 0.6; s.showValue = t < 0.6;
        } },

      { title: 'Slope along î', duration: 3,
        narration: 'Freeze <span class="c-j">y</span> and walk along the green ' + ihat + ': how fast does the height rise? That is the <b>partial derivative</b> ∂f/∂x. The inset shows the slice f(x, 0.5) with its tangent. At our point the slope is 2: one step right climbs 2.',
        setup(s) { s.view = '2d'; s.cam = cam2(); s.surfAlpha = 0; s.goldContour = true; s.levelLabels = true; s.panel = 'px'; s.slice = 'x'; s.pxGrow = 0; },
        at(s, t) { s.pxGrow = seg(t, 0.1, 0.6, 'out'); } },

      { title: 'Slope along ĵ', duration: 3,
        narration: 'Now freeze <span class="c-i">x</span> and walk along the red ' + jhat + '. The slice is a narrower parabola, so the slope is larger: ∂f/∂y = 8y = 4. Two directions, two slopes, drawn as the green and red arrows from the point.',
        setup(s) { s.pxGrow = 1; s.panel = 'py'; s.slice = 'y'; s.pyGrow = 0; },
        at(s, t) { s.pyGrow = seg(t, 0.1, 0.6, 'out'); } },

      { title: 'Two slopes make one arrow', duration: 3,
        narration: 'Put the two slopes into one bracket. That vector is the <b>gradient</b> <span class="c-v">∇f</span> = [2; 4]. It lives on the map, not on the hill: it is an arrow in the input plane, built tip to tail from the green and red components.',
        setup(s) { s.pyGrow = 1; s.panel = 'grad'; s.slice = null; s.tipToTail = 0; s.gradGrow = 0; },
        at(s, t) { s.tipToTail = seg(t, 0, 0.45); s.gradGrow = seg(t, 0.5, 0.95, 'out'); } },

      { title: 'Perpendicular to the contour', duration: 4,
        narration: 'The gradient is always at right angles to the contour through its point. Watch the point slide along the gold curve: the yellow arrow turns with it, never leaning along the curve. Along a contour the height does not change, so the climb must be straight across.',
        setup(s) { s.tipToTail = 0; s.pxGrow = 0; s.pyGrow = 0; s.gradGrow = 1; s.showPerp = true; s.panel = 'perp'; },
        at(s, t) { const th = PI / 4 + 2 * PI * seg(t, 0.02, 0.98); const r = Math.SQRT2; s.p = [r * Math.cos(th), r / 2 * Math.sin(th)]; } },

      { title: 'Steepest climb', duration: 4,
        narration: 'Choose any unit direction, the purple <span class="c-w">u</span>. The slope along it is the <b>directional derivative</b> <span class="c-v">∇f</span>·<span class="c-w">u</span>, the shadow from p1-06. It is largest when u lines up with the gradient, zero across the contour, most negative straight downhill.',
        setup(s) { s.p = [1, 0.5]; s.showPerp = false; s.showProbe = true; s.showBar = true; s.probe = 0; s.panel = 'dir'; },
        at(s, t) { s.probe = 360 * seg(t, 0, 1, 'linear'); } },

      { title: 'The whole field', duration: 3,
        narration: 'Draw the gradient at every point: a field of arrows, each perpendicular to its contour, longer where the contours crowd together. The map and the field carry the same information; the full yellow arrow is the one at our point.',
        setup(s) { s.showProbe = false; s.showBar = false; s.probe = 0; s.panel = 'field'; s.fieldFade = 0; },
        at(s, t) { s.fieldFade = seg(t, 0, 0.6); } },

      { title: '✕ It does not point at the bottom', duration: 4, misconception: true,
        narration: 'A common belief: “−<span class="c-v">∇f</span> points at the minimum.” Not here. At (2, 0.5) the downhill arrow aims 31° away from the center. The gradient only knows the slope under your feet; the bowl’s shape decides where the bottom is. Next lesson measures that shape.',
        setup(s) { s.fieldFade = 0; s.showField = false; s.gradGrow = 0; s.panel = 'mis'; s.levelLabels = false; s.negGrow = 0; s.negCross = false; s.negStep = 0; s.goldContour = true; },
        at(s, t) { s.p = lerpV([1, 0.5], [2, 0.5], seg(t, 0, 0.3)); s.negGrow = seg(t, 0.3, 0.55, 'out'); s.negStep = seg(t, 0.6, 0.85, 'out'); s.negCross = t > 0.78; } },

      { title: 'Tangent plane and the first-order rule', duration: 4,
        narration: 'Back in 3D: at any point the surface is almost flat — the <b>tangent plane</b>, drawn teal. Its tilt is the gradient. For a small displacement, the purple <span class="c-w">d</span>, the height changes by about <span class="c-v">∇f</span>·<span class="c-w">d</span>. This one line is what every training step uses.',
        setup(s) { s.p = [1, 0.5]; s.negGrow = 0; s.negCross = false; s.negStep = 0; s.view = '3d'; s.cam = cam3(Object.assign({ target: [0, 0, 0.8] }, TOPDOWN)); s.surfAlpha = 1; s.showValue = true; s.gradGrow = 1; s.showTangent = true; s.dGrow = 0; s.panel = 'tangent'; s.levelLabels = false; },
        at(s, t) { LA.cam.tween(s, Object.assign({ target: [0, 0, 0.8], dist: 13.4 }, TOPDOWN), { yaw: -1.07, pitch: 0.5, target: [0.8, 0.4, 0.4], dist: 7.6 }, seg(t, 0, 0.6)); s.surfAlpha = 0.25 + 0.45 * seg(t, 0, 0.4); s.dGrow = seg(t, 0.55, 0.9, 'out'); } },

      { title: 'Why ML cares: the loss landscape', duration: 4, mlTie: true,
        narration: 'Replace (x, y) by two weights (w₁, w₂) and f by a loss <b>L</b>. Training walks downhill: <b>w ← w − η<span class="c-v">∇L</span></b>, the teaser from p1-02. Real models have millions of weights, so ∇L is a long strip of slopes. The picture stops at 3. The formula does not.',
        setup(s) { s.view = '2d'; s.cam = cam2(); s.showTangent = false; s.dGrow = 0; s.axisNames = ['w₁', 'w₂']; s.panel = 'ml'; s.showStrip = true; s.wPoint = true; s.gradGrow = 1; s.goldContour = false; s.levelLabels = false; s.mlStep = 0; s.stripFade = 0; },
        at(s, t) { s.mlStep = seg(t, 0.15, 0.55, 'out'); s.stripFade = seg(t, 0.5, 0.9); } },

      { title: 'Recap', duration: 2, recap: true,
        narration: 'Three things to keep. The gradient collects one slope per axis into one arrow. It is perpendicular to the contour and points uphill fastest. It does not point at the minimum — only the bowl’s shape knows that.',
        cards: ['<b>One slope per axis, one arrow.</b> ∇f = [∂f/∂x; ∂f/∂y]: the green and red slopes become the yellow gradient.', '<b>Perpendicular to contours, uphill.</b> Along a contour nothing changes, so the climb is straight across; ∇f·u is the slope along u.', '<b>Not a pointer to the minimum.</b> −∇f is the fastest local drop, 31° off the center here; the bowl’s shape decides where the bottom is.'],
        setup(s) { s.recap = true; s.panel = 'none'; s.showStrip = false; s.wPoint = false; s.axisNames = ['x', 'y']; },
        at(s, t) { } },
    ],

    playground: {
      intro: '<p>Drag the yellow point <span class="c-v">p</span> or the tip of the purple probe <span class="c-w">u</span>; switch the function, show the arrow field or the 3D surface.</p>',
      setup(s) { s.view = '2d'; s.cam = cam2(); s.fn = 'bowl'; s.p = [1, 0.5]; s.probe = 30; s.showProbe = true; s.showBar = true; s.showComponents = true; s.goldContour = true; s.levelLabels = true; s.gradGrow = 1; s.pxGrow = 1; s.pyGrow = 1; s.panel = 'play'; s.showValue = true; s.surfAlpha = 1; },
      controls: [
        { type: 'preset', label: 'Function', items: [
          { label: 'Bowl', apply(s) { s.fn = 'bowl'; } },
          { label: 'Round bowl', apply(s) { s.fn = 'round'; } },
          { label: 'Saddle', apply(s) { s.fn = 'saddle'; } },
          { label: 'Two hills', apply(s) { s.fn = 'hills'; } },
        ] },
        { type: 'slider', bind: 'probe', min: 0, max: 360, step: 1, label: 'Probe direction θ', color: 'w', fmt: x => fmt(x, 0) + '°' },
        { type: 'toggle', bind: 'showComponents', label: 'Show the two partial slopes' },
        { type: 'toggle', bind: 'showField', label: 'Arrow field' },
        { type: 'toggle', bind: 'showTangent', label: 'Tangent (plane in 3D, line in 2D)' },
        { type: 'preset', label: 'View', items: [
          { label: '2D map', apply(s) { s.view = '2d'; s.cam = cam2(); } },
          { label: '3D surface', apply(s) { s.view = '3d'; s.cam = cam3({ yaw: -1.25 }); } },
        ] },
      ],
    },

    checks: [
      { kind: 'manipulate',
        prompt: 'Drag the yellow point to where the gradient has no ' + jhat + ' component but still has length at least 2.',
        setup(s) { s.fn = 'bowl'; s.view = '2d'; s.cam = Object.assign(cam2(), { center: [0.2, 1.05], halfH: 4.7 }); s.p = [0.6, 1.2]; s.snapAxis = true; s.gradGrow = 1; s.pxGrow = 1; s.pyGrow = 1; s.showComponents = true; s.goldContour = true; s.levelLabels = true; s.panel = 'play'; s.showProbe = false; s.showBar = false; },
        judge(s) {
          const gr = FN(s).grad(s.p);
          const flat = Math.abs(gr[1]) <= 0.5, long = Math.abs(gr[0]) >= 2 - LA.TOL.scalar;
          return { ok: flat && long, progress: Math.max(0, 1 - Math.abs(s.p[1]) / 1.2) * (long ? 1 : 0.8), hint: flat && !long ? 'The red component is gone — now slide further out along ' + ihat + ' until the slope reaches 2.' : undefined };
        },
        hint: 'Along which axis is the slope zero? Look where the red component vanishes.',
        solution(s) { s.p = [1.5, 0]; } },
      { kind: 'manipulate',
        prompt: 'Rotate the purple probe <span class="c-w">u</span> (drag its tip) to the direction of steepest descent at the point.',
        setup(s) { s.fn = 'bowl'; s.view = '2d'; s.cam = cam2(); s.p = [1, 0.5]; s.probe = 0; s.showProbe = true; s.showBar = true; s.gradGrow = 1; s.goldContour = true; s.panel = 'play'; },
        judge(s) { const gr = FN(s).grad(s.p); const r = LA.judge.angleDeg(unit(s.probe), N.scale(gr, -1), 0, LA.TOL.angleDeg); return { ok: r.ok, progress: Math.max(0, 1 - r.dist / 180) }; },
        hint: 'Downhill is exactly opposite the yellow arrow; the bar should read −‖∇f‖.',
        solution(s) { const gr = FN(s).grad(s.p); s.probe = (Math.atan2(-gr[1], -gr[0]) * 180 / PI + 360) % 360; } },
      { kind: 'entry',
        prompt: 'Type the gradient of f = x² + 4y² at p = (−1, 0.5). Your guess is drawn as a dashed purple arrow; a right answer crosses the dashed tangent of the gold contour at a right angle.',
        bind: 'guess', answer: [-2, 4], tol: LA.TOL.entry, label: '∇f', colors: 'w',
        hint: '∂f/∂x = 2x and ∂f/∂y = 8y, evaluated at x = −1, y = 0.5.',
        setup(s) { s.fn = 'bowl'; s.view = '2d'; s.cam = cam2(); s.p = [-1, 0.5]; s.guess = [0, 0]; s.showGuess = true; s.gradGrow = 0; s.pxGrow = 0; s.pyGrow = 0; s.showComponents = false; s.showProbe = false; s.showBar = false; s.goldContour = true; s.levelLabels = true; s.panel = 'guess'; } },
    ],
  });

  /* ---------------- 2D map ---------------- */
  function render2d(g, s, ctx, F) {
    const p = s.p, gr = F.grad(p), gn = N.norm(gr), W = ctx.size.w, H = ctx.size.h;
    g.ghostGrid();
    g.axes({ names: s.axisNames });
    g.contours(F.f, F.levels, CB, { res: 56, color: 'tick', width: 1.2, alpha: 0.9 });
    if (s.levelLabels && (s.fn === 'bowl' || s.fn === 'round')) {
      const c = Math.cos(0.9), sn = Math.sin(0.9);
      F.levels.forEach(lv => { const r = Math.sqrt(lv); g.label(fmt(lv), [r * c, (s.fn === 'bowl' ? r / 2 : r) * sn], { size: 10.5, color: 'muted', font: 'ui' }); });
    }
    if (s.goldContour) g.contours(F.f, [F.f(p)], CB, { res: 56, color: 'gold', width: 2.2, alpha: 1 });
    // arrow field (pre-scaled by √‖∇f‖ so long gradients read as longer arrows without swamping the map)
    const ff = s.showField ? 1 : s.fieldFade;
    if (ff > 0.01) g.field(q => { const v = F.grad(q), n = N.norm(v); return n < 1e-6 ? null : N.scale(v, 0.11 * Math.sqrt(n) / n); }, { bounds: { x0: -3, x1: 3, y0: -3, y1: 3 }, step: 0.5, scale: 1, maxLen: 100, color: 'v', alpha: 0.42 * ff });
    // partial slopes as green / red arrows (red slides tip-to-tail in step 5)
    const px = s.showComponents ? 1 : s.pxGrow, py = s.showComponents ? 1 : s.pyGrow;
    const gx = [p[0] + SC * gr[0], p[1]], shift = [SC * gr[0] * s.tipToTail, 0];
    if (px > 0.01 && Math.abs(gr[0]) > 1e-6) g.vector(p, [p[0] + SC * gr[0] * px, p[1]], { color: 'i', width: 3, label: '∂f/∂x = ' + fmt(gr[0]), labelAt: 1, labelSize: 12.5 });
    if (py > 0.01 && Math.abs(gr[1]) > 1e-6) { const a = N.add(p, shift); g.vector(a, [a[0], a[1] + SC * gr[1] * py], { color: 'j', width: 3, label: '∂f/∂y = ' + fmt(gr[1]), labelAt: s.tipToTail > 0.5 ? 0.3 : 1, dx: s.tipToTail > 0.5 ? 50 : undefined, dy: s.tipToTail > 0.5 ? 0 : undefined, labelSize: 12.5 }); }
    // the gradient
    if (s.gradGrow > 0.01 && gn > 1e-6) g.vector(p, N.add(p, N.scale(gr, SC * s.gradGrow)), { color: 'v', width: 3.2, label: s.wPoint ? '∇L' : '∇f', labelAt: 1 });
    if (s.showGuess) {
      // check 3 shows only the contour's tangent as a direction cue — never the true gradient (that would be the answer)
      if (gn > 1e-6) { const tg = N.normalize([-gr[1], gr[0]]); g.line(N.add(p, N.scale(tg, -1.1)), N.add(p, N.scale(tg, 1.1)), { color: 'residual', dashed: true, width: 1.4 }); g.label('tangent', N.add(p, N.scale(tg, 1.25)), { size: 11.5, color: 'muted', font: 'ui' }); }
      const q = N.add(p, N.scale(s.guess, SC)); if (N.norm(s.guess) > 1e-6) g.vector(p, q, { color: 'w', width: 2.6, label: 'your ∇f', dashed: true });
    }
    // tangent line + right angle
    if ((s.showPerp || (s.showTangent && s.view === '2d')) && gn > 1e-6) {
      const tg = N.normalize([-gr[1], gr[0]]);
      g.line(N.add(p, N.scale(tg, -1)), N.add(p, N.scale(tg, 1)), { color: 'residual', dashed: true, width: 1.4 });
      g.arc(p, tg, gr, { r: 0.32, color: 'hud', right: true });
      g.label('90°', N.add(p, N.add(N.scale(tg, 0.5), N.scale(N.normalize(gr), 0.45))), { size: 11.5, color: 'hud' });
    }
    // misconception: −∇f vs the direction to the minimum
    if (s.negGrow > 0.01 && gn > 1e-6) {
      const neg = N.scale(gr, -SC * s.negGrow), tip = N.add(p, neg);
      g.vector(p, [0, 0], { color: 'residual', dashed: true, width: 1.6, label: 'to the minimum', labelAt: 0.55, labelSize: 12 });
      g.vector(p, tip, { color: 'v', width: 3.2, label: '−∇f', labelAt: 1 });
      g.point([0, 0], { r: 5, color: 'gold', label: 'min', dx: -22, dy: 12 });
      const ang = N.angle(neg, N.scale(p, -1)) * 180 / PI;
      if (s.negGrow > 0.95) g.arc(p, neg, N.scale(p, -1), { r: 0.8, color: 'bad', label: fmt(ang, 0) + '°' });
      if (s.negStep > 0.01) { const st = N.add(p, N.scale(N.normalize(gr), -0.3 * s.negStep)); g.vector(p, st, { color: 'b', width: 4, head: 9, label: 'step', labelAt: 1, labelSize: 12 }); }
      g.label('“−∇f points at the minimum”', [-2.7, 2.5], { color: 'muted', size: 13, anchor: 'left' });
      if (s.negCross) g.marker.cross([-2.7, 2.5], { size: 22, dx: -18 });
    }
    // the ML step
    if (s.wPoint) {
      const eta = 0.1, w1 = N.sub(p, N.scale(gr, eta));
      g.point(p, { r: 6, color: 'b', label: 'w', dx: 14, dy: 6 });
      if (s.mlStep > 0.01) { const q = lerpV(p, w1, s.mlStep); g.vector(p, q, { color: 'w', width: 3, label: '−η∇L (true size)', labelAt: 1, labelSize: 12, dx: -52, dy: 4 }); if (s.mlStep > 0.95) g.point(w1, { r: 5, color: 'b' }); }
    }
    // probe direction u, its shadow on the gradient line, and the slope bar
    if (s.showProbe) {
      const u = unit(s.probe), tip = N.add(p, u);
      if (gn > 1e-6) { const gh = N.normalize(gr); const foot = N.add(p, N.scale(gh, N.dot(u, gh))); g.lineThrough(p, gh, { len: 1.4, color: 'v', alpha: 0.3, width: 1 }); g.dropLine(tip, foot, { color: 'residual', footColor: 'mag' }); }
      g.vector(p, tip, { color: 'w', width: 3, label: 'u', labelAt: 1 });
      g.handle('u', tip, (q, st) => { st.probe = (Math.atan2(q[1] - st.p[1], q[0] - st.p[0]) * 180 / PI + 360) % 360; }, { color: 'w' });
      if (s.showBar) {
        const val = N.dot(gr, u), bx = W - 150, by = H - 170, bh = 120;
        if (s.view === '2d') signedBar(g, bx, by, 30, bh, val, Math.max(gn, 1e-6));
        g.label('slope along u = ∇f·u', [bx - 8, by - 18], { screen: true, anchor: 'left', size: 12, color: 'muted', font: 'ui' });
        g.label('+‖∇f‖ = ' + fmt(gn), [bx + 42, by + 4], { screen: true, anchor: 'left', size: 11, color: 'ok', font: 'ui' });
        g.label('0', [bx + 42, by + bh / 2], { screen: true, anchor: 'left', size: 11, color: 'hud', font: 'ui' });
        g.label('−‖∇f‖ = ' + fmt(-gn), [bx + 42, by + bh - 4], { screen: true, anchor: 'left', size: 11, color: 'neg', font: 'ui' });
        g.label(Math.abs(val) < 0.05 ? 'flat (along the contour)' : val >= 0 ? 'uphill' : 'downhill', [bx - 8, by + bh + 18], { screen: true, anchor: 'left', size: 11.5, color: Math.abs(val) < 0.05 ? 'hud' : val >= 0 ? 'ok' : 'neg', font: 'ui' });
      }
    }
    if (s.slice) sliceInset(g, s, F, s.slice, ctx.size);
    if (s.showStrip) stripView(g, s);
    const pt = s.wPoint ? null : p;
    if (pt) g.point(p, { r: 6, color: 'v', label: s.showGuess || ctx.mode !== 'learn' ? 'p' : undefined, dx: -13, dy: 12 });
    g.handle('p', p, (q, st) => { st.p = clampP(q); if (st.snapAxis && Math.abs(st.p[1]) < 0.08) st.p[1] = 0; }, { color: 'v' });
  }

  /** Inset chart of the 1-D slices through p (bottom-right). */
  function sliceInset(g, s, F, which, size) {
    const w = Math.max(175, Math.round(size.w * 0.3)), h = 112, x = size.w - w - 12, y = size.h - h - 28;
    const yr = F.yr, xr = [-3, 3], p = s.p, gr = F.grad(p), f0 = F.f(p);
    const X = v => x + (v - xr[0]) / (xr[1] - xr[0]) * w, Y = v => y + h - (v - yr[0]) / (yr[1] - yr[0]) * h;
    const mk = (fnOf, color) => { const pts = []; for (let i = 0; i <= 72; i++) { const v = xr[0] + (xr[1] - xr[0]) * i / 72; const fv = fnOf(v); if (fv <= yr[1] && fv >= yr[0]) pts.push([v, fv]); } return { points: pts, color, width: 1.6 }; };
    const series = [];
    if (which === 'x' || which === 'both') series.push(mk(v => F.f([v, p[1]]), 'i'));
    if (which === 'y' || which === 'both') series.push(mk(v => F.f([p[0], v]), 'j'));
    const tangents = [];
    const tang = (v0, m, color) => {
      let a = v0 - 0.9, b = v0 + 0.9;                // keep the tangent inside the chart's value range
      if (Math.abs(m) > 1e-9) { const lo = v0 + (yr[0] - f0) / m, hi = v0 + (yr[1] - f0) / m; a = Math.max(a, Math.min(lo, hi)); b = Math.min(b, Math.max(lo, hi)); }
      tangents.push({ points: [[a, f0 + (a - v0) * m], [b, f0 + (b - v0) * m]], color, width: 2.6 }); };
    if (which === 'x') tang(p[0], gr[0], 'i');
    if (which === 'y') tang(p[1], gr[1], 'j');
    const title = which === 'x' ? 'slice f(x, ' + fmt(p[1]) + ')' : which === 'y' ? 'slice f(' + fmt(p[0]) + ', y)' : 'slices through p';
    g.lineChart(series.concat(tangents), { x, y, w, h, xRange: xr, yRange: yr, title, xLabel: which === 'y' ? 'y' : which === 'x' ? 'x' : 'x or y' });
    const v0 = which === 'y' ? p[1] : p[0];
    if (f0 >= yr[0] && f0 <= yr[1]) g.label('●', [X(v0), Y(f0)], { screen: true, color: 'v', bg: false, size: 13 });
    if (which === 'x' || which === 'y') { const m = which === 'x' ? gr[0] : gr[1]; g.label('slope ' + fmt(m), [X(v0) + (m >= 0 ? -8 : 8), Y(f0) + 16], { screen: true, anchor: m >= 0 ? 'right' : 'left', size: 11.5, color: which, font: 'ui' }); }
  }

  /** The n-D honesty strip: ∇L ∈ ℝ¹² as a 1×12 heat row with printed values. */
  function stripView(g, s) {
    const a = s.stripFade; if (a < 0.01) return;
    const bx0 = -3, bx1 = 3, by0 = -2.75, by1 = -2.3, n = s.strip.length, cw = (bx1 - bx0) / n;
    g.heat2d([s.strip], { x0: bx0, x1: bx1, y0: by0, y1: by1 }, { range: [-1, 1], ramp: t => LA.color.diverge(2 * t - 1), smooth: false, alpha: 0.95 * a });
    for (let i = 0; i <= n; i++) g.line([bx0 + i * cw, by0], [bx0 + i * cw, by1], { color: 'line', width: 1, alpha: a });
    g.line([bx0, by0], [bx1, by0], { color: 'line', alpha: a }); g.line([bx0, by1], [bx1, by1], { color: 'line', alpha: a });
    for (let i = 0; i < n; i++) g.label(fmt(s.strip[i]), [bx0 + (i + 0.5) * cw, (by0 + by1) / 2], { size: 10.5, color: 'text', bg: false, font: 'ui', alpha: a });
    g.label('∇L ∈ ℝ¹² (synthetic, seeded)', [bx0, by1], { anchor: 'above', dy: -4, dx: 70, size: 12, color: 'v', font: 'ui', alpha: a });
    g.label('The picture stops at 3. The formula does not.', [0, by0 - 0.45], { size: 13, color: 'hud', font: 'ui', alpha: a });
  }

  /* ---------------- 3D surface ---------------- */
  function render3d(g, s, ctx, F) {
    const p = s.p, gr = F.grad(p), fv = F.f(p), zs = F.zs, P3 = [p[0], p[1], 0];
    g.cage({ range: 3, walls: false, alpha: 0.6 });
    g.contours(F.f, F.levels, CB, { res: 48, color: 'tick', width: 1.1, alpha: 0.85, z: 0 });
    if (s.goldContour) g.contours(F.f, [fv], CB, { res: 48, color: 'gold', width: 2, z: 0 });
    if (s.surfAlpha > 0.01) drawSurface(g, F, s.surfAlpha);
    g.axes({ length: 3.4, names: ['x', 'y', 'f'] });
    const top = [p[0], p[1], zs * fv];
    if (s.showValue && !(s.showTangent && s.dGrow > 0.3)) {
      dash3(g, P3, top, { color: 'v', width: 1.4 });
      g.point(top, { r: 4, color: 'v' });
      g.label('f = ' + fmt(fv), top, { anchor: 'above', dy: -6, color: 'v' });
    }
    if (s.gradGrow > 0.01 && N.norm(gr) > 1e-6) g.vector(P3, [p[0] + SC * gr[0] * s.gradGrow, p[1] + SC * gr[1] * s.gradGrow, 0], { color: 'v', width: 3, label: '∇f', labelAt: 1 });
    if (s.showTangent) {
      g.plane({ point: top, normal: [-zs * gr[0], -zs * gr[1], 1], extent: 0.7, color: 'teal', alpha: 0.3 });
      if (s.dGrow > 0.01) {
        const d = N.scale(s.d, s.dGrow), q = N.add(p, d), dd = N.dot(gr, d);
        g.vector(P3, [q[0], q[1], 0], { color: 'w', width: 3, label: 'd', labelAt: 1 });
        g.vector(top, [q[0], q[1], zs * (fv + dd)], { color: 'w', width: 3 });
        dash3(g, [q[0], q[1], 0], [q[0], q[1], zs * (fv + dd)], { color: 'residual', width: 1.2 });
        const qt = [q[0], q[1], zs * F.f(q)], qp = [q[0], q[1], zs * (fv + dd)];
        g.point(qt, { r: 4.5, color: 'b' }); g.label('true f(p + d)', qt, { anchor: 'left', dx: 10, dy: -12, size: 12, color: 'b', font: 'ui' });
        g.point(qp, { r: 4, color: 'teal' }); g.label('f(p) + ∇f·d (on the plane)', qp, { anchor: 'left', dx: 10, dy: 12, size: 12, color: 'teal', font: 'ui' });
        g.hud('f(p + d) = ' + fmt(F.f(q), 3) + '   ·   f(p) + ∇f·d = ' + fmt(fv + dd, 3), 14, 22, { size: 13 });
      }
    }
    g.point(P3, { r: 6, color: 'v', label: 'p', dx: -13, dy: 12 });
    g.handle('p', P3, (q, st) => { st.p = clampP(q); }, { color: 'v', plane: 'xy' });
  }

  /* ---------------- recap: three mini-cards ---------------- */
  function renderRecap(g, s, ctx) {
    const F = FNS.bowl;
    const card = (i, title, cam, fn) => {
      g.viewport({ x: 0.04, y: 0.02 + i * 0.33, w: 0.92, h: 0.31 }, { view: '2d', cam, bg: 'panel' }, (gg, r) => { fn(gg, r); gg.label(title, [r.x + 12, r.y + 16], { screen: true, anchor: 'left', size: 13.5, color: 'text', font: 'ui', bold: true, bg: false }); });
    };
    card(0, '1 · One slope per axis, one arrow', { center: [0.9, 0.9], halfH: 1.35 }, gg => {
      gg.ghostGrid({ range: 4 });
      gg.vector([0, 0], [0.8, 0], { color: 'i', width: 3, label: '∂f/∂x = 2', labelAt: 0.5, dy: 18, labelSize: 12 });
      gg.vector([0.8, 0], [0.8, 1.6], { color: 'j', width: 3, label: '∂f/∂y = 4', labelAt: 0.55, dx: 48, labelSize: 12 });
      gg.vector([0, 0], [0.8, 1.6], { color: 'v', width: 3.4, label: '∇f = [2; 4]', labelAt: 1, dx: 46, dy: -4, labelSize: 12.5 });
      gg.point([0, 0], { r: 5, color: 'v', label: 'p', dx: -12, dy: 12 });
    });
    card(1, '2 · Perpendicular to contours, uphill', { center: [0.3, 0.1], halfH: 1.25 }, gg => {
      gg.contours(F.f, [0.5, 2, 3.5, 5], CB, { res: 56, color: 'tick', width: 1.2 });
      gg.contours(F.f, [2], CB, { res: 56, color: 'gold', width: 2.2 });   // f(1, 0.5) = 2: the contour through p
      const p = [1, 0.5], gr = F.grad(p), tg = N.normalize([-gr[1], gr[0]]);
      gg.line(N.add(p, N.scale(tg, -0.8)), N.add(p, N.scale(tg, 0.8)), { color: 'residual', dashed: true, width: 1.4 });
      gg.vector(p, N.add(p, N.scale(gr, 0.15)), { color: 'v', width: 3.2, label: '∇f', labelAt: 1 });
      gg.arc(p, tg, gr, { r: 0.25, color: 'hud', right: true });
      gg.point(p, { r: 5, color: 'v' });
    });
    card(2, '3 · Not a pointer to the minimum', { center: [0.9, -0.2], halfH: 1.45 }, gg => {
      gg.contours(F.f, [0.5, 1.5, 3, 5, 8], CB, { res: 56, color: 'tick', width: 1.2 });
      const p = [2, 0.5], gr = F.grad(p), neg = N.scale(gr, -0.3);
      gg.vector(p, [0, 0], { color: 'residual', dashed: true, width: 1.6, label: 'to the minimum', labelAt: 0.72, labelSize: 11.5, dx: 0, dy: -16 });
      gg.vector(p, N.add(p, neg), { color: 'v', width: 3.2, label: '−∇f', labelAt: 1 });
      gg.arc(p, neg, N.scale(p, -1), { r: 0.7, color: 'bad', label: '31°' });
      gg.point([0, 0], { r: 5, color: 'gold', label: 'min', dx: -14, dy: 12 });
      gg.point(p, { r: 5, color: 'v' });
      gg.label('“points at the minimum”', [1.2, -1.05], { color: 'muted', size: 12.5 });
      gg.marker.cross([1.2, -1.05], { size: 20, dx: -96 });
    });
  }
})();
