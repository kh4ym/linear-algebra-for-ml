/* p1-03 — Linear combinations, span, subspace. Every vector is a recipe a·v + b·w; all recipes paint a plane or
   collapse to a line; a span is a flat region through the origin (a subspace) with a dimension; in 3D a third
   vector counts only if it leaves the plane; word arithmetic is a recipe; strips combine cell by cell. */
(function () {
  'use strict';
  const N = LA.num, seg = LA.seg, lerpV = LA.lerpV, fmt = LA.fmt;
  const V0 = [2, 1], W0 = [-1, 2];
  const V3 = [2, 0, 1], W3 = [0, 2, 1], U_OUT = [0, 0, 2], U_IN = [1, 1, 1];
  const NHAT = N.normalize(N.cross(V3, W3));                       // normal of the teal plane span(v, w)
  const ROW = ['i', 'j'], ROW3 = ['i', 'j', 'k'];
  const vS = '<span class="c-v">v</span>', wS = '<span class="c-w">w</span>', uS = '<span class="c-b">u</span>';
  const ihat = '<span class="c-i">î</span>', jhat = '<span class="c-j">ĵ</span>';
  const cam2 = () => Object.assign(LA.cam.defaults2d(), { center: [0, 0.6], halfH: 4.8 });
  const cam3 = () => Object.assign(LA.cam.defaults3d(), { dist: 12.5, yaw: -2.4, pitch: 0.44, target: [0.3, 0.3, 0.5] });
  const camWord = () => Object.assign(LA.cam.defaults3d(), { dist: 7, yaw: -1.2, pitch: 0.35, target: [1.1, 1.0, 0.8] });
  const ANG_V = Math.atan2(V0[1], V0[0]), ANG_W = Math.atan2(W0[1], W0[0]), LEN_W = N.norm(W0);
  // word vectors (hand-set 3D toys, not real embeddings)
  const KING = [2, 1, 1.5], MAN = [1.5, 0.3, 0.2], WOMAN = [1.2, 1.4, 0.3], QUEEN = [1.78, 2.05, 1.66];
  // strips for the n-D step
  const SV = [1, -0.5, 2, 0, 1.5, -1, 0.5, 1], SW = [0.5, 1, -1, 2, 0, 0.5, -1.5, 1], SA = 2, SB = -1;
  const divRamp = t => LA.color.diverge(t * 2 - 1);

  const RASTER = (() => { const r = []; for (let i = 0; i < 9; i++) for (let j = 0; j < 9; j++) r.push([-2 + 0.5 * i, -2 + 0.5 * j]); return r; })();
  const RASTER3 = (() => { const r = []; for (let i = 0; i < 7; i++) for (let j = 0; j < 7; j++) r.push([-1 + i / 3, -1 + j / 3]); return r; })();
  const combo = (v, w, a, b) => v.map((x, k) => a * x + b * w[k]);
  const det3 = (a, b, c) => N.dot(a, N.cross(b, c));
  function dimSpan2(v, w) { const nv = N.norm(v), nw = N.norm(w); if (nv < 0.05 && nw < 0.05) return 0; if (nv < 0.05 || nw < 0.05) return 1; return Math.abs(v[0] * w[1] - v[1] * w[0]) / (nv * nw) > 0.035 ? 2 : 1; }
  function dimSpan3(v, w, u) {
    const vs = [v, w, u].filter(x => N.norm(x) > 0.05); if (!vs.length) return 0;
    if (vs.length === 3 && Math.abs(det3(v, w, u)) / (N.norm(v) * N.norm(w) * N.norm(u)) > 0.02) return 3;
    for (let i = 0; i < vs.length; i++) for (let j = i + 1; j < vs.length; j++) if (N.norm(N.cross(vs[i], vs[j])) / (N.norm(vs[i]) * N.norm(vs[j])) > 0.035) return 2;
    return 1;
  }

  /** Column bracket at a screen point (or beside a world point). */
  function bracket(g, pos, vals, roles, o) {
    o = o || {};
    const size = o.size || 14, rowH = size * 1.25, n = vals.length, half = o.half || size * 1.45;
    const sp = o.screen ? pos : g.P.toScreen(pos);
    const sx = sp[0] + (o.dx || 0), sy = sp[1] + (o.dy || 0) - (n - 1) * rowH / 2;
    const al = o.alpha === undefined ? 1 : o.alpha; if (al <= 0.01) return;
    for (let i = 0; i < n; i++) {
      const y = sy + i * rowH;
      const lg = n === 1 ? '[' : i === 0 ? '⎡' : i === n - 1 ? '⎣' : '⎢';
      const rg = n === 1 ? ']' : i === 0 ? '⎤' : i === n - 1 ? '⎦' : '⎥';
      g.label(lg, [sx - half, y], { screen: true, color: 'hud', size: size * 1.2, font: 'math', alpha: al, bg: false });
      g.label(typeof vals[i] === 'string' ? vals[i] : fmt(vals[i]), [sx, y], { screen: true, color: roles ? roles[i] : (o.color || 'text'), size, font: 'ui', bold: true, alpha: al });
      g.label(rg, [sx + half, y], { screen: true, color: 'hud', size: size * 1.2, font: 'math', alpha: al, bg: false });
    }
  }

  /** The recipe a·v + b·w drawn tip to tail (dashed copies) with a gold tip. */
  function drawRecipe(g, v, w, a, b, o) {
    o = o || {};
    const av = N.scale(v, a), tip = N.add(av, N.scale(w, b));
    if (Math.abs(a) > 0.02) g.vector(v.map(() => 0), av, { color: 'v', width: 2.2, dashed: true, alpha: 0.9, label: o.labels === false ? '' : 'a·v', labelAt: Math.abs(a) > 1.1 ? 0.85 : 0.5, labelOffset: -16 });
    if (Math.abs(b) > 0.02) g.vector(av, tip, { color: 'w', width: 2.2, dashed: true, alpha: 0.9, label: o.labels === false ? '' : 'b·w', labelAt: 0.5, labelOffset: 16 });
    g.point(tip, { r: 6.5, color: 'gold', stroke: 'bg' });
    if (o.tipLabel !== false) g.label(o.tipLabel || 'a·v + b·w', tip, { color: 'gold', dx: o.ldx === undefined ? 16 : o.ldx, dy: o.ldy === undefined ? -16 : o.ldy, size: 13.5, font: 'ui', anchor: o.anchor || 'left' });
    return tip;
  }

  function fillPlane2(g, alpha) { const b = g.P.bounds(); g.polygon([[b.x0, b.y0], [b.x1, b.y0], [b.x1, b.y1], [b.x0, b.y1]], { fill: 'teal', alpha: 0.16 * alpha }); }

  LA.registerLesson({
    id: 'p1-03-linear-combinations-span',
    part: 1,
    title: 'Linear combinations, span, subspace',
    subtitle: 'Everything two vectors can reach: a recipe, a plane, a line, a subspace with a dimension.',
    minutes: 10,
    prereqs: ['p1-02-add-scale'],
    tags: ['core'],
    badge: { drawnIn: '2D and 3D', realSize: 'a linear model’s predictions live in the span of its feature columns — a d-dimensional flat region inside ℝⁿ with n in the thousands', transfers: '“span = all recipes” and “a span is flat and passes through the origin” are true in any dimension; only the drawing of the flat region stops at planes in 3D' },
    cantShow: ['a 5-dimensional flat region inside ℝ⁷⁸⁴ (we say “flat, through the origin, 5 directions”)'],
    terms: [
      { term: 'linear combination', def: 'A recipe a·v + b·w: scale each vector, then add.' },
      { term: 'span', def: 'The set of every vector you can reach by recipes from the given vectors.' },
      { term: 'collinear', def: 'Lying on the same line through the origin.' },
      { term: 'subspace', def: 'A flat region through the origin — a line, a plane, or all of space; it is always a span.' },
      { term: 'dimension', def: 'For a subspace: the fewest arrows whose recipes sweep it out (1 for a line, 2 for a plane).' },
    ],
    atlas: ['embeddings', 'linear-regression'],
    thumbnail: 2,

    state() {
      return {
        view: '2d', cam: cam2(),
        v: V0.slice(), w: W0.slice(), a: 1, b: 1,
        ijT: 0, showVW: false, showRecipe: false, _touchAB: false,
        trailN: 0, trailAll: false, fill: 0, collapse: 0, wAngle: ANG_W,
        sweep: 0, showTarget: false, cross: false,
        dimsT: 0, hideGrid: false,
        v3: V3.slice(), w3: W3.slice(), u3: U_OUT.slice(), showU: 0, boxA: 0, dots3: 0, fill3: 0, uInPlane: 0,
        wordT: 0, stripT: 0,
        target: null, panel: 'none', recap: false,
        dragV: false, dragW: false, dragA: false, dragB: false, dragU: false, showDim: false,
      };
    },

    render(g, s, ctx) {
      if (s.recap) { renderRecap(g); return; }
      if (s.view === '3d') { render3d(g, s, ctx); return; }
      if (s.dimsT > 0.01) { renderDims(g, s); return; }
      if (s.stripT > 0.01) { renderStrips(g, s); return; }
      g.grid(null, { range: 12 });
      g.axes({ labels: true });
      const v = s.v, w = s.w;

      // the î/ĵ recipe of step 1
      if (s.ijT > 0.01) {
        const t = s.ijT;
        for (let k = 0; k < 2; k++) { const p = seg(t, k * 0.2, k * 0.2 + 0.2); if (p > 0) g.vector([k, 0], [k + p, 0], { color: 'i', width: 3, head: 10, label: p > 0.9 ? 'i' : '', hat: true, labelOffset: 16 }); }
        for (let k = 0; k < 3; k++) { const p = seg(t, 0.4 + k * 0.133, 0.533 + k * 0.133); if (p > 0) g.vector([2, k], [2, k + p], { color: 'j', width: 3, head: 10, label: p > 0.9 ? 'j' : '', hat: true, labelOffset: -16 }); }
        const yv = seg(t, 0.8, 1, 'out');
        if (yv > 0.01) { g.vector([0, 0], [2 * yv, 3 * yv], { color: 'v', width: 3.4 }); if (yv > 0.9) bracket(g, [2, 3], [2, 3], ROW, { dx: -66, dy: -8 }); }
      }

      // teal span: whole plane, collapsing to a band, then a line
      const coll = dimSpan2(v, w) < 2;
      if (s.fill > 0.01) {
        const d = N.normalize(N.norm(v) > 0.05 ? v : (N.norm(w) > 0.05 ? w : [1, 0])), p = [-d[1], d[0]];
        if (s.collapse > 0.01) {
          // the collapse animation: the teal plane thins onto the line through v
          const h = 14 * Math.pow(1 - s.collapse, 2);
          if (h > 0.03) g.polygon([N.add(N.scale(d, -20), N.scale(p, -h)), N.add(N.scale(d, 20), N.scale(p, -h)), N.add(N.scale(d, 20), N.scale(p, h)), N.add(N.scale(d, -20), N.scale(p, h))], { fill: 'teal', alpha: 0.16 * s.fill });
          if (s.collapse > 0.6) g.lineThrough([0, 0], d, { color: 'teal', width: 5, alpha: 0.75 * s.fill, len: 20 });
        } else if (coll) g.lineThrough([0, 0], d, { color: 'teal', width: 5, alpha: 0.75 * s.fill, len: 20 });
        else fillPlane2(g, s.fill);
      }

      // recipe dots (trail)
      if (s.trailN > 0 || s.trailAll) {
        const n = s.trailAll ? RASTER.length : Math.min(RASTER.length, Math.floor(s.trailN));
        const pts = []; for (let k = 0; k < n; k++) pts.push(combo(v, w, RASTER[k][0], RASTER[k][1]));
        g.points(pts, { r: 3.4, color: 'gold', alpha: 0.85 });
        if (!s.trailAll && n > 0 && n < RASTER.length) drawRecipe(g, v, w, RASTER[n - 1][0], RASTER[n - 1][1], { labels: false, tipLabel: false });
      }

      // the misconception: tip races along the line, target stays out of reach
      if (s.showTarget && s.target) {
        const T = s.target;
        g.point(T, { r: 7, color: 'b', stroke: 'bg' });
        g.label('target [' + fmt(T[0]) + '; ' + fmt(T[1]) + ']', T, { color: 'b', dx: 12, dy: -14, size: 13, font: 'ui', anchor: 'left' });
      }
      if (s.sweep > 0.01) {
        const k = 1.9 * Math.sin(2 * Math.PI * 1.6 * s.sweep);      // a − 2b
        const a = 0.6 * Math.sin(2 * Math.PI * 2.3 * s.sweep) + 0.3, b = (a - k) / 2;
        const tip = drawRecipe(g, v, w, a, b, { labels: false, tipLabel: false });
        if (s.target) g.line(tip, s.target, { color: 'residual', dashed: true, width: 1.6 });
        if (s.target) { const foot = N.proj(s.target, v); g.line(foot, s.target, { color: 'residual', width: 1, alpha: 0.6, dashed: [2, 3] }); g.label('closest any recipe gets: ' + fmt(N.norm(N.sub(s.target, foot))) + ' away', s.target, { color: 'muted', dy: -40, size: 12.5, font: 'ui' }); }
      }
      if (s.cross && s.target) { g.marker.cross(s.target, { dx: -22, dy: 0, size: 22 }); g.label('unreachable', s.target, { color: 'bad', dx: -40, dy: 20, size: 14, font: 'ui', anchor: 'right' }); }

      // v, w and the live recipe
      if (s.showVW) {
        g.vector([0, 0], v, { color: 'v', width: 3.2, label: 'v', labelAt: 0.6, labelOffset: 14 });
        g.vector([0, 0], w, { color: 'w', width: 3.2, label: 'w', labelAt: 0.6, labelOffset: -14 });
      }
      if (s.showRecipe) {
        const tipS = g.P.toScreen(combo(v, w, s.a, s.b)), right = tipS[0] < g.size.w - 150;
        const tip = drawRecipe(g, v, w, s.a, s.b, { ldx: right ? 16 : -16, ldy: -16, anchor: right ? 'left' : 'right' });
        bracket(g, tip, tip, ROW, { dx: right ? 40 : -40, dy: 26, size: 13 });
      }
      if (s.target && !s.showTarget) {
        g.point(s.target, { r: 7, color: 'b', stroke: 'bg' });
        g.label('target', s.target, { color: 'b', dx: -12, dy: -14, size: 13, font: 'ui', anchor: 'right' });
      }

      // captions
      const cap = (txt, col) => g.hud(txt, 14, 22, { size: 14, color: col || 'teal' });
      if (s.panel === 'span' && s.fill > 0.5) cap('span = whole plane');
      if (s.panel === 'collinear') cap(s.collapse > 0.6 ? 'collinear → span is a line' : 'different directions → span is still the whole plane');
      if (s.showDim) cap('dimension of span: ' + dimSpan2(v, w), 'teal');

      // handles
      if (s.dragV) g.handle('v', v, (p, st) => { st.v = [p[0], p[1]]; }, { color: 'v' });
      if (s.dragW) g.handle('w', w, (p, st) => { st.w = [p[0], p[1]]; }, { color: 'w' });
      if (s.dragA) g.handle('a', N.scale(v, s.a), (p, st) => { st.a = LA.clamp(N.dot(p, st.v) / (N.dot(st.v, st.v) || 1), -3, 3); st._touchAB = true; }, { color: 'v' });
      if (s.dragB) { const av = N.scale(v, s.a); g.handle('b', N.add(av, N.scale(w, s.b)), (p, st) => { const o = N.scale(st.v, st.a); st.b = LA.clamp(N.dot(N.sub(p, o), st.w) / (N.dot(st.w, st.w) || 1), -3, 3); st._touchAB = true; }, { color: 'w' }); }
    },

    math(s, tex) {
      // plain numbers (tex.num after a tex.mat is written into the wrong cell by tex.reconcile — engine)
      const cn = (x, c) => '<span class="' + tex.colorClass(c || 'text') + '">' + fmt(x) + '</span>';
      const V = tex.sym('v', { color: 'v' }), W = tex.sym('w', { color: 'w' }), U = tex.sym('u', { color: 'b' });
      const rows = { colors: 'rows' }, rows3 = { colors: 'rows' };
      const recipe = tex.eq(tex.sym('a'), tex.op('·'), V, tex.op('+'), tex.sym('b'), tex.op('·'), W);
      switch (s.panel) {
        case 'ij': return tex.block(tex.eq(tex.vec([2, 3], rows), tex.op('='), cn(2, 'i'), tex.op('·'), tex.ihat(), tex.op('+'), cn(3, 'j'), tex.op('·'), tex.jhat()));
        case 'recipe': {
          const tip = combo(s.v, s.w, s.a, s.b);
          return tex.block(tex.eq(recipe, tex.op('='), tex.vec(['2a − b', 'a + 2b'], rows), tex.op('='), tex.vec(tip, rows)),
            tex.eq(tex.sym('a'), tex.op('='), cn(s.a), tex.op(',  '), tex.sym('b'), tex.op('='), cn(s.b)));
        }
        case 'span': return tex.block(tex.eq(tex.text('span', { color: 'teal' }), tex.paren(tex.eq(V, tex.op(','), W)), tex.op('='), tex.op('{'), recipe, tex.op('}')));
        case 'collinear': return tex.block(tex.eq(tex.text('if '), W, tex.text(' lies on the line of '), V, tex.op(':'), tex.text('span', { color: 'teal' }), tex.paren(tex.eq(V, tex.op(','), W)), tex.op('='), tex.text('line through v', { color: 'teal' })),
          tex.eq(tex.text('angle between v and w:'), cn(Math.round(N.angle(s.v, s.w) * 180 / Math.PI)), tex.op('°')));
        case 'unreach': return tex.block(tex.eq(tex.sym('a'), tex.op('·'), V, tex.op('+'), tex.sym('b'), tex.op('·'), tex.paren(tex.eq(tex.op('−'), cn(2), V)), tex.op('='), tex.paren(tex.eq(tex.sym('a'), tex.op('−'), cn(2), tex.sym('b'))), tex.op('·'), V),
          tex.text('always a multiple of v', { color: 'muted' }));
        case 'dims': return tex.block(tex.eq(tex.text('dim'), tex.paren(tex.text('line')), tex.op('='), cn(1)), tex.eq(tex.text('dim'), tex.paren(tex.text('plane')), tex.op('='), cn(2)), tex.eq(tex.text('dim'), tex.paren(tex.op('{0}')), tex.op('='), cn(0)));
        case 'span3': return tex.block(tex.eq(V, tex.op('='), tex.vec(s.v3, rows3)), tex.eq(W, tex.op('='), tex.vec(s.w3, rows3)), tex.eq(tex.text('span', { color: 'teal' }), tex.paren(tex.eq(V, tex.op(','), W)), tex.op('='), tex.text('plane through 0', { color: 'teal' })));
        case 'u3': {
          const d = dimSpan3(s.v3, s.w3, s.u3);
          const parts = [tex.eq(U, tex.op('='), tex.vec(s.u3, rows3)), tex.eq(tex.text('dimension of span(v, w, u):'), cn(d, 'teal'))];
          if (d === 2) parts.push(tex.eq(tex.vec([1, 1, 1], rows3), tex.op('='), cn(0.5), tex.op('·'), V, tex.op('+'), cn(0.5), tex.op('·'), W));
          return tex.block.apply(null, parts);
        }
        case 'word': return tex.block(tex.eq(cn(1), tex.op('·'), tex.text('king', { color: 'v' }), tex.op('−'), cn(1), tex.op('·'), tex.text('man', { color: 'w' }), tex.op('+'), cn(1), tex.op('·'), tex.text('woman', { color: 'b' }), tex.op('≈'), tex.text('queen')),
          tex.eq(tex.vec(N.add(N.sub(KING, MAN), WOMAN), rows3), tex.op('≈'), tex.vec(QUEEN, rows3)));
        case 'strip': return tex.block(tex.eq(tex.paren(recipe), tex.sym('', { sub: 'i' }), tex.op('='), tex.sym('a'), tex.op('·'), tex.sym('v', { color: 'v', sub: 'i' }), tex.op('+'), tex.sym('b'), tex.op('·'), tex.sym('w', { color: 'w', sub: 'i' })),
          tex.eq(tex.sym('a'), tex.op('='), cn(SA), tex.op(',  '), tex.sym('b'), tex.op('='), cn(SB)));
        case 'play': {
          if (s.view === '3d') return tex.block(tex.eq(U, tex.op('='), tex.vec(s.u3, rows3)), tex.eq(recipe, tex.op('='), tex.vec(combo(s.v3, s.w3, s.a, s.b), rows3)), tex.eq(tex.text('dimension of span:'), cn(dimSpan3(s.v3, s.w3, s.u3), 'teal')));
          return tex.block(tex.eq(recipe, tex.op('='), tex.vec(combo(s.v, s.w, s.a, s.b), rows)), tex.eq(tex.text('dimension of span:'), cn(dimSpan2(s.v, s.w), 'teal')));
        }
        case 'check1': return tex.block(tex.eq(V, tex.op('='), tex.vec(s.v, rows)), tex.eq(W, tex.op('='), tex.vec(s.w, rows)), tex.eq(tex.text('dimension of span:'), cn(dimSpan2(s.v, s.w), 'teal')));
        case 'check2': return tex.block(tex.eq(recipe, tex.op('='), tex.vec(combo(s.v, s.w, s.a, s.b), rows)), tex.eq(tex.sym('a'), tex.op('='), cn(s.a), tex.op(',  '), tex.sym('b'), tex.op('='), cn(s.b)));
        case 'check3': return tex.block(tex.eq(U, tex.op('='), tex.vec(s.u3, rows3)), tex.eq(tex.text('height of u’s tip above the teal sheet ='), cn(Math.abs(N.dot(s.u3, NHAT)))));
        default: return '';
      }
    },

    steps: [
      { title: 'Every vector is a recipe of î and ĵ', duration: 3,
        narration: '[2; 3] means “two copies of the green ' + ihat + ', then three copies of the red ' + jhat + '”. Written out, that is 2î + 3ĵ: scale each basis vector, then add. A scale-and-add of vectors is called a <b>linear combination</b>. Every vector in the plane is one.',
        setup(s) { s.view = '2d'; s.cam = cam2(); s.ijT = 0; s.showVW = false; s.showRecipe = false; s.panel = 'ij'; },
        at(s, t) { s.ijT = t; } },

      { title: 'New ingredients', duration: 3, interactive: true,
        narration: 'Replace the basis arrows by any two: the yellow ' + vS + ' = [2; 1] and the purple ' + wS + ' = [−1; 2]. The recipe a·' + vS + ' + b·' + wS + ' still lands somewhere. With a = 1, b = 1 it lands on [1; 3]. Drag the two dots on the dashed arrows and watch the gold tip wander.',
        setup(s) { s.ijT = 0; s.v = V0.slice(); s.w = W0.slice(); s.showVW = true; s.showRecipe = true; s.a = 1; s.b = 1; s._touchAB = false; s.dragA = true; s.dragB = true; s.panel = 'recipe'; },
        at(s, t) { if (s._touchAB) return; const u = seg(t, 0.35, 0.95); s.a = LA.lerp(1, 1.5, u); s.b = LA.lerp(1, -0.5, u); } },

      { title: 'All recipes paint the plane', duration: 5,
        narration: 'Let a and b sweep through every value. The tips leave a trail of dots, and the dots fill the whole plane: every point is reachable by some recipe. The set of everything reachable is the <b>span</b> of ' + vS + ' and ' + wS + '. Here, the span is the entire plane.',
        setup(s) { s.showRecipe = false; s.dragA = false; s.dragB = false; s.trailN = 0; s.fill = 0; s.panel = 'span'; },
        at(s, t) { s.trailN = RASTER.length * seg(t, 0, 0.7, 'linear'); s.fill = seg(t, 0.7, 0.95); } },

      { title: 'When two vectors collapse to a line', duration: 5,
        narration: 'Rotate ' + wS + ' toward the line of ' + vS + '. As long as they point in different directions the span is the plane. At the instant they line up — <b>collinear</b> — every recipe stays on one line. The teal plane thins to a teal line; one direction is lost.',
        setup(s) { s.trailN = 0; s.trailAll = true; s.fill = 1; s.collapse = 0; s.wAngle = ANG_W; s.panel = 'collinear'; },
        at(s, t) {
          s.wAngle = LA.lerp(ANG_W, ANG_V + Math.PI, seg(t, 0, 0.8));
          s.w = t >= 0.8 ? N.scale(s.v, -1) : [LEN_W * Math.cos(s.wAngle), LEN_W * Math.sin(s.wAngle)];
          s.collapse = seg(t, 0.8, 0.97);
        } },

      { title: 'Two vectors do not always reach everything', duration: 4, misconception: true,
        narration: 'Belief: “any two vectors reach any point in the plane.” With ' + wS + ' on ' + vS + '’s line, try to reach the orange target [0; 1]. Every recipe slides along the teal line and never touches it. Two vectors give two directions only if they are not collinear.',
        setup(s) { s.trailAll = false; s.collapse = 1; s.fill = 1; s.w = N.scale(s.v, -2); s.target = [0, 1]; s.showTarget = true; s.sweep = 0; s.cross = false; s.panel = 'unreach'; },
        at(s, t) { s.sweep = Math.max(0.001, seg(t, 0, 0.75, 'linear')); s.cross = t >= 0.8; } },

      { title: 'A span is a subspace', duration: 4,
        narration: 'A span is always a flat thing that passes through the origin: a line, a plane, or all of space. Such a flat thing is called a <b>subspace</b>. Its <b>dimension</b> is the fewest arrows it takes to sweep it out: 1 for a line, 2 for a plane. The origin alone has dimension 0.',
        setup(s) { s.showTarget = false; s.target = null; s.sweep = 0; s.cross = false; s.fill = 0; s.collapse = 0; s.showVW = false; s.dimsT = 0; s.panel = 'dims'; },
        at(s, t) { s.dimsT = Math.max(0.001, t); } },

      { title: 'In 3D, two vectors span a plane', duration: 5,
        narration: 'In 3D, ' + vS + ' = [2; 0; 1] and ' + wS + ' = [0; 2; 1] sweep out a tilted plane: a two-dimensional subspace floating in a three-dimensional room. Orbit around it. Every recipe of v and w stays on the teal sheet; nothing lifts off it.',
        setup(s) { s.dimsT = 0; s.view = '3d'; s.cam = cam3(); s.v3 = V3.slice(); s.w3 = W3.slice(); s.dots3 = 0; s.fill3 = 0; s.showU = 0; s.panel = 'span3'; },
        at(s, t) { s.dots3 = seg(t, 0, 0.6, 'linear'); s.fill3 = seg(t, 0.6, 0.9); LA.cam.tween(s, { yaw: -2.4 }, { yaw: -1.7 }, seg(t, 0, 1, 'linear')); } },

      { title: 'A third vector: in the plane or out of it', duration: 5,
        narration: 'Add an orange third vector ' + uS + '. If it sticks out of the teal plane, three directions are available and the span is all of space — dimension 3. If it lies inside the plane, like [1; 1; 1], it adds nothing: still a plane. The third arrow must leave the sheet to count.',
        setup(s) { s.dots3 = 1; s.fill3 = 1; s.u3 = U_OUT.slice(); s.showU = 0; s.boxA = 0; s.uInPlane = 0; s.panel = 'u3'; },
        at(s, t) {
          s.showU = seg(t, 0, 0.15); s.boxA = seg(t, 0.1, 0.3) - seg(t, 0.5, 0.65);
          s.uInPlane = seg(t, 0.5, 0.9); s.u3 = lerpV(U_OUT, U_IN, s.uInPlane);
        } },

      { title: 'Word arithmetic is a linear combination', duration: 5, mlTie: true,
        narration: 'Word embeddings are vectors, and recipes of them mean something: king − man + woman is a linear combination with scalars 1, −1, 1, and it lands near the vector for queen. In 300 dimensions three arrows span only a flat region, and the answer is reachable inside it. Part 6 does this for real.',
        setup(s) { s.showU = 0; s.boxA = 0; s.dots3 = 0; s.fill3 = 0; s.cam = camWord(); s.wordT = 0; s.panel = 'word'; },
        at(s, t) { s.wordT = t; } },

      { title: 'The same recipe in any dimension', duration: 3,
        narration: 'For strips, a·' + vS + ' + b·' + wS + ' is computed cell by cell: scale each strip, add matching cells. The span of two 784-strips is still a flat two-dimensional sheet through the origin — inside a space we cannot draw. The picture stops at 3. The formula does not.',
        setup(s) { s.wordT = 0; s.view = '2d'; s.cam = cam2(); s.stripT = 0; s.panel = 'strip'; },
        at(s, t) { s.stripT = Math.max(0.001, t); } },

      { title: 'Recap', duration: 3, recap: true,
        narration: 'A linear combination scales and adds. The span is everything reachable by recipes; it is always flat and passes through the origin — a subspace — of dimension 1, 2 or 3 depending on how many directions the ingredients truly provide. Collinear vectors lose a direction. Word analogies are recipes.',
        cards: ['<b>Recipe: a·v + b·w.</b> Scale each ingredient, then add tip to tail; the gold tip is where the recipe lands.', '<b>Span = all recipes = a subspace through 0.</b> Collinear ingredients lose a direction: the plane thins to a line.', '<b>Dimension counts the directions that really add something.</b> A third arrow adds one only if it leaves the plane.'],
        setup(s) { s.stripT = 0; s.recap = true; s.panel = 'none'; },
        at(s, t) { } },
    ],

    playground: {
      intro: '<p>Drag the tips of <span class="c-v">v</span> and <span class="c-w">w</span>, set the recipe with the sliders, and watch the dimension of the span. Try the collinear preset, then the 3D presets (drag the orange <span class="c-b">u</span>; it snaps onto the sheet when close).</p>',
      setup(s) { s.view = '2d'; s.cam = cam2(); s.v = V0.slice(); s.w = W0.slice(); s.a = 1; s.b = 1; s.showVW = true; s.showRecipe = true; s.trailAll = true; s.fill = 1; s.dragV = true; s.dragW = true; s.dragU = true; s.showDim = true; s.showU = 1; s.dots3 = 1; s.fill3 = 1; s.u3 = U_OUT.slice(); s.panel = 'play'; },
      controls: [
        { type: 'matrix', bind: 'v', label: 'v (drag tip)', colors: 'rows', step: 0.1, min: -4, max: 4 },
        { type: 'matrix', bind: 'w', label: 'w (drag tip)', colors: 'rows', step: 0.1, min: -4, max: 4 },
        { type: 'slider', bind: 'a', min: -3, max: 3, step: 0.05, label: 'a', color: 'v', fmt: x => fmt(x) },
        { type: 'slider', bind: 'b', min: -3, max: 3, step: 0.05, label: 'b', color: 'w', fmt: x => fmt(x) },
        { type: 'toggle', bind: 'fill', label: 'Show span' },
        { type: 'preset', label: 'Presets', items: [
          { label: 'Independent', apply(s) { s.view = '2d'; s.cam = cam2(); s.v = V0.slice(); s.w = W0.slice(); } },
          { label: 'Collinear', apply(s) { s.view = '2d'; s.cam = cam2(); s.v = V0.slice(); s.w = [-3, -1.5]; } },
          { label: '3D: u out of the plane', apply(s) { s.view = '3d'; s.cam = cam3(); s.u3 = U_OUT.slice(); } },
          { label: '3D: u in the plane', apply(s) { s.view = '3d'; s.cam = cam3(); s.u3 = U_IN.slice(); } },
        ] },
      ],
    },

    checks: [
      { kind: 'manipulate',
        prompt: 'Drag the purple ' + wS + ' so that the span of ' + vS + ' and ' + wS + ' collapses to a line.',
        setup(s) { s.view = '2d'; s.cam = cam2(); s.v = V0.slice(); s.w = W0.slice(); s.showVW = true; s.fill = 1; s.trailAll = true; s.dragW = true; s.showDim = true; s.panel = 'check1'; },
        judge(s) {
          const nw = N.norm(s.w); if (nw < 0.3) return { ok: false, progress: 0, hint: 'Keep ' + wS + ' longer than a stub: line it up, don’t shrink it.' };
          const a1 = LA.judge.angleDeg(s.w, s.v, 0, LA.TOL.angleDeg), a2 = LA.judge.angleDeg(s.w, s.v, 180, LA.TOL.angleDeg);
          return { ok: a1.ok || a2.ok, progress: Math.max(a1.progress, a2.progress) };
        },
        hint: 'Point ' + wS + ' along the same line as ' + vS + ', either way.',
        solution(s) { s.w = [-2, -1]; } },
      { kind: 'manipulate',
        prompt: 'Drag the two dots (one slides a along ' + vS + ', the other slides b along ' + wS + ') so that the recipe a·v + b·w hits the orange target.',
        setup(s) { s.view = '2d'; s.cam = cam2(); s.v = V0.slice(); s.w = W0.slice(); s.a = 0; s.b = 0; s.showVW = true; s.showRecipe = true; s.dragA = true; s.dragB = true; s.target = [1, 4]; s.panel = 'check2'; },
        judge(s) { return LA.judge.near(combo(s.v, s.w, s.a, s.b), s.target, LA.TOL.pos); },
        hint: 'Raise b to climb along ' + wS + ', then adjust a to slide along ' + vS + '.',
        solution(s) { s.a = 1.2; s.b = 1.4; } },
      { kind: 'manipulate',
        prompt: 'Drag the tip of the orange ' + uS + ' (moving the mouse up or down raises or lowers it) until it lies in the teal plane, so the three vectors no longer span all of space. The sheet is magnetic: once the tip comes within 0.15 of it, u snaps on.',
        setup(s) { s.view = '3d'; s.cam = cam3(); s.v3 = V3.slice(); s.w3 = W3.slice(); s.u3 = U_OUT.slice(); s.showU = 1; s.fill3 = 1; s.dots3 = 0; s.dragU = true; s.panel = 'check3'; },
        judge(s) { const d = Math.abs(det3(s.v3, s.w3, s.u3)); return { ok: d < LA.TOL.det, progress: Math.max(0, 1 - d / 8) }; },
        hint: 'Lower ' + uS + '’s tip until it touches the teal sheet (it snaps on when close).',
        solution(s) { s.u3 = U_IN.slice(); } },
    ],
  });

  /* ---------------- step 6: three subspaces side by side ---------------- */
  function renderDims(g, s) {
    const S = g.size, t = s.dimsT, m = 10;
    const cw = Math.floor((S.w - 4 * m) / 3), ch = Math.floor(S.h * 0.5);
    const cam = { center: [0.35, 0.3], halfH: 2.4 };
    const sv = N.scale(V0, 0.55), sw = N.scale(W0, 0.55);
    const items = [
      { title: 'a line', dim: 1, draw(g2) { g2.lineThrough([0, 0], sv, { color: 'teal', width: 5, alpha: 0.75, len: 12 }); g2.vector([0, 0], sv, { color: 'v', width: 3, label: 'v', labelOffset: -14 }); } },
      { title: 'a plane', dim: 2, draw(g2) { const b = g2.P.bounds(); g2.polygon([[b.x0, b.y0], [b.x1, b.y0], [b.x1, b.y1], [b.x0, b.y1]], { fill: 'teal', alpha: 0.2 }); g2.vector([0, 0], sv, { color: 'v', width: 3, label: 'v' }); g2.vector([0, 0], sw, { color: 'w', width: 3, label: 'w' }); } },
      { title: 'the origin alone', dim: 0, draw(g2) { g2.point([0, 0], { r: 7, color: 'teal' }); g2.label('{0}', [0, 0], { color: 'teal', dx: 18, dy: -16, size: 14, font: 'math' }); } },
    ];
    items.forEach((it, k) => {
      const a = seg(t, k * 0.15, k * 0.15 + 0.2);
      if (a <= 0.01) return;
      const r = { x: m + k * (cw + m), y: m + 30 * (1 - a), w: cw, h: ch };
      g.viewport(r, { view: '2d', cam, bg: 'panel' }, (g2, rr) => {
        g2.ghostGrid({ range: 6 });
        it.draw(g2);
        g2.label(it.title, [rr.x + rr.w / 2, rr.y + 16], { screen: true, size: 13.5, color: 'hud', font: 'ui', bold: true });
        g2.label('dimension ' + it.dim, [rr.x + rr.w / 2, rr.y + rr.h - 16], { screen: true, size: 14, color: 'teal', font: 'ui', bold: true });
      });
    });
    // a tilted line that misses the origin is NOT a subspace
    const b = seg(t, 0.55, 0.75);
    if (b > 0.01) {
      const r = { x: m, y: m * 2 + ch, w: S.w - 2 * m, h: S.h - ch - 3 * m };
      g.viewport(r, { view: '2d', cam: { center: [0.4, 0.6], halfH: 1.9 }, bg: 'panel' }, (g2, rr) => {
        g2.ghostGrid({ range: 8 });
        g2.lineThrough([0, 1.2], [1, 0.5], { color: 'residual', width: 4, alpha: 0.8 * b, len: 12 });
        g2.point([0, 0], { r: 6, color: 'text', label: 'origin', dx: 10, dy: 14 });
        if (t > 0.8) {
          g2.marker.cross([-1.6, 0.4], { size: 22, dy: -22 });
          g2.label('not a subspace: misses the origin', [rr.x + 12, rr.y + rr.h - 16], { screen: true, color: 'bad', size: 14, font: 'ui', anchor: 'left' });
        }
        g2.label('flat, but shifted', [rr.x + 12, rr.y + 16], { screen: true, size: 13, color: 'hud', font: 'ui', anchor: 'left', bold: true });
      });
    }
  }

  /* ---------------- step 10: strips combine cell by cell ---------------- */
  function renderStrips(g, s) {
    const t = s.stripT, x0 = -2.3, cw = 0.74, n = SV.length, hh = 0.32;
    const rowsDef = [
      { label: 'v', vals: SV, y: 3.7, color: 'v', at: 0 },
      { label: 'w', vals: SW, y: 2.8, color: 'w', at: 0 },
      { label: '2·v', vals: SV.map(x => SA * x), y: 1.5, color: 'v', at: 0.15 },
      { label: '−1·w', vals: SW.map(x => SB * x), y: 0.6, color: 'w', at: 0.35 },
      { label: '2·v − w', vals: SV.map((x, i) => SA * x + SB * SW[i]), y: -0.9, color: 'gold', at: 0.6 },
    ];
    rowsDef.forEach(r => {
      const a = seg(t, r.at, r.at + 0.2); if (a <= 0.01) return;
      const xs = x0 + (1 - a) * 6;
      g.heat2d([r.vals], { x0: xs, x1: xs + n * cw, y0: r.y - hh, y1: r.y + hh }, { range: [-5, 5], ramp: divRamp, alpha: 0.95, smooth: false });
      for (let i = 0; i < n; i++) {
        g.line([xs + i * cw, r.y - hh], [xs + i * cw, r.y + hh], { color: 'line', width: 1 });
        g.label(fmt(r.vals[i]), [xs + (i + 0.5) * cw, r.y], { size: 13, bold: true, color: 'text', font: 'ui', bg: false });
      }
      g.label(r.label, [xs - 0.2, r.y], { color: r.color, size: 15, font: 'math', anchor: 'right' });
    });
    if (t > 0.3) g.label('scale each strip …', [x0 + n * cw / 2, 2.15], { size: 12.5, color: 'muted', font: 'ui' });
    if (t > 0.55) { g.label('… then add matching cells', [x0 + n * cw / 2, -0.05], { size: 12.5, color: 'muted', font: 'ui' }); g.line([x0, 0.2], [x0 + n * cw, 0.2], { color: 'hud', width: 1.2 }); }
    if (t > 0.8) {
      g.label('8 cells here; 784 for an image — same rule', [x0 + n * cw / 2, -1.75], { size: 13, color: 'hud', font: 'ui' });
      g.label('The picture stops at 3. The formula does not.', [0, -3.2], { size: 15, color: 'hud', font: 'ui' });
    }
  }

  /* ---------------- 3D scene ---------------- */
  function render3d(g, s, ctx) {
    g.cage({ range: 3, walls: true });
    g.axes({ length: 3.2, labels: true, names: ['x', 'y', 'z'], colors: ['i', 'j', 'k'] });
    if (s.wordT > 0.01) { renderWords(g, s); return; }
    const v = s.v3, w = s.w3, u = s.u3;
    if (s.boxA > 0.01) g.box([[1, 0, 0], [0, 1, 0], [0, 0, 1]], { origin: [-3, -3, -3], size: 6, fill: 'teal', alpha: 0.05 * s.boxA, stroke: 'teal', strokeAlpha: 0.6 * s.boxA });
    if (s.fill3 > 0.01) g.planes([{ span: [v, w], extent: 3.2, color: 'teal', alpha: 0.26 * s.fill3 }]);
    if (s.dots3 > 0.01) {
      const n = Math.floor(RASTER3.length * s.dots3), pts = [];
      for (let k = 0; k < n; k++) pts.push(combo(v, w, RASTER3[k][0], RASTER3[k][1]));
      g.points(pts, { r: 3, color: 'gold', alpha: 0.85 });
    }
    g.vector([0, 0, 0], v, { color: 'v', width: 3.2, label: 'v' });
    g.vector([0, 0, 0], w, { color: 'w', width: 3.2, label: 'w' });
    if (s.showU > 0.01) {
      const uu = N.scale(u, s.showU);
      g.vector([0, 0, 0], uu, { color: 'b', width: 3.4, label: 'u' });
      // where u stands relative to the sheet: dashed drop to the plane
      const h = N.dot(u, NHAT);
      if (Math.abs(h) > 0.05 && s.showU > 0.9) g.line(u, N.sub(u, N.scale(NHAT, h)), { color: 'residual', dashed: true, width: 1.4 });
    }
    // drag u in the upright plane that faces the camera, so moving the mouse up/down raises or lowers its tip
    const f = LA.cam.forward(s.cam), fh = Math.hypot(f[0], f[1]) > 0.05 ? [f[0], f[1], 0] : [1, 0, 0];
    if (s.dragU) g.handle('u', u, (p, st) => {
      let q = [p[0], p[1], p[2] || 0]; const h = N.dot(q, NHAT);
      if (Math.abs(h) < 0.15) q = N.sub(q, N.scale(NHAT, h));      // the teal sheet is magnetic when close
      st.u3 = q.map(x => LA.clamp(x, -3, 3));
    }, { color: 'b', plane: { point: u, normal: fh } });
    if (s.panel === 'span3' && s.fill3 > 0.5) g.hud('span(v, w) = a plane through the origin (dimension 2)', 14, 22, { size: 14, color: 'teal' });
    if (s.panel === 'u3' || s.panel === 'play' || s.panel === 'check3') {
      const d = dimSpan3(v, w, u);
      if (s.showU > 0.5) g.hud(d === 3 ? 'span = all of space, dimension 3' : d === 2 ? 'still a plane, dimension 2' : 'a line, dimension 1', 14, 22, { size: 14.5, color: 'teal' });
    }
  }

  function renderWords(g, s) {
    const t = s.wordT;
    const p1 = seg(t, 0, 0.2), p2 = seg(t, 0.25, 0.45), p3 = seg(t, 0.5, 0.7), p4 = seg(t, 0.75, 0.9);
    const a = KING, b = N.sub(KING, MAN), c = N.add(b, WOMAN);
    // the four word arrows from the origin
    g.vector([0, 0, 0], KING, { color: 'v', width: 3, label: 'king' });
    g.vector([0, 0, 0], MAN, { color: 'w', width: 2.4, alpha: 0.75, label: 'man' });
    g.vector([0, 0, 0], WOMAN, { color: 'b', width: 2.4, alpha: 0.75, label: 'woman' });
    g.vector([0, 0, 0], QUEEN, { color: 'hud', width: 2.6, label: 'queen', dx: 16, dy: -12 });
    // recipe: king, then −man, then woman, tip to tail
    if (p1 > 0.01) g.line([0, 0, 0], N.scale(a, p1), { color: 'v', width: 4, alpha: 0.35 });
    if (p2 > 0.01) g.vector(a, lerpV(a, b, p2), { color: 'w', width: 2.6, dashed: true, label: p2 > 0.8 ? '−man' : '', labelAt: 0.5, dx: 0, dy: 18 });
    if (p3 > 0.01) g.vector(b, lerpV(b, c, p3), { color: 'b', width: 2.6, dashed: true, label: p3 > 0.8 ? '+woman' : '', labelAt: 0.5, dx: 0, dy: -18 });
    if (p3 > 0.99) {
      g.point(c, { r: 6.5, color: 'gold', stroke: 'bg' });
      g.label('result', c, { color: 'gold', dx: -16, dy: 22, size: 13, font: 'ui', anchor: 'right' });
    }
    if (p4 > 0.01) {
      const pulse = 14 + 6 * Math.abs(Math.sin(t * Math.PI * 4));
      g.marker.ring(QUEEN, { r: pulse, color: 'gold' });
      g.label('lands ' + fmt(N.norm(N.sub(c, QUEEN))) + ' from queen', QUEEN, { color: 'gold', dx: -24, dy: -22, size: 12.5, font: 'ui', anchor: 'right' });
    }
    g.hud('hand-set 3D toy vectors (real embeddings have 300 entries)', 14, 22, { size: 12.5, color: 'muted' });
    g.hud('Part 6: embeddings', 14, 42, { size: 13, color: 'gold' });
  }

  /* ---------------- recap cards ---------------- */
  function renderRecap(g) {
    const S = g.size, m = 10, h = Math.floor((S.h - 4 * m) / 3), w = S.w - 2 * m;
    const row = i => ({ x: m, y: m + i * (h + m), w, h });
    const title = (rr, txt) => g.label(txt, [rr.x + 10, rr.y + 14], { screen: true, anchor: 'left', size: 13, color: 'hud', font: 'ui', bold: true });
    g.viewport(row(0), { view: '2d', cam: { center: [0.5, 1.9], halfH: 2.9 }, bg: 'panel' }, (g2, rr) => {
      g2.ghostGrid({ range: 8 });
      g2.vector([0, 0], V0, { color: 'v', width: 3, label: 'v', labelAt: 0.6, labelOffset: 12 });
      g2.vector([0, 0], W0, { color: 'w', width: 3, label: 'w', labelAt: 0.6, labelOffset: -12 });
      drawRecipe(g2, V0, W0, 1.2, 1.4, { labels: true, tipLabel: '1.2·v + 1.4·w', ldx: 14, ldy: 4 });
      title(rr, '1 · Recipe: a·v + b·w');
    });
    const r2 = row(1), half = Math.floor((r2.w - m) / 2);
    g.viewport({ x: r2.x, y: r2.y, w: half, h: r2.h }, { view: '2d', cam: { center: [0, 0.3], halfH: 2.2 }, bg: 'panel' }, (g2, rr) => {
      g2.ghostGrid({ range: 8 }); fillPlane2(g2, 1.2);
      g2.vector([0, 0], V0, { color: 'v', width: 3, label: 'v' }); g2.vector([0, 0], W0, { color: 'w', width: 3, label: 'w' });
      title(rr, '2 · Span = all recipes; subspace through 0');
      g2.label('plane (dim 2)', [rr.x + rr.w / 2, rr.y + rr.h - 14], { screen: true, size: 12.5, color: 'teal', font: 'ui' });
    });
    g.viewport({ x: r2.x + half + m, y: r2.y, w: r2.w - half - m, h: r2.h }, { view: '2d', cam: { center: [0, 0.3], halfH: 2.2 }, bg: 'panel' }, (g2, rr) => {
      g2.ghostGrid({ range: 8 });
      g2.lineThrough([0, 0], V0, { color: 'teal', width: 5, alpha: 0.75, len: 12 });
      g2.vector([0, 0], V0, { color: 'v', width: 3, label: 'v' }); g2.vector([0, 0], [-2, -1], { color: 'w', width: 3, label: 'w' });
      g2.label('collinear → line (dim 1)', [rr.x + rr.w / 2, rr.y + rr.h - 14], { screen: true, size: 12.5, color: 'teal', font: 'ui' });
    });
    g.viewport(row(2), { view: '3d', cam: Object.assign(LA.cam.defaults3d(), { dist: 9, yaw: -2.1, pitch: 0.4, target: [0.3, 0.3, 0.6] }), bg: 'panel' }, (g2, rr) => {
      g2.cage({ range: 3, walls: false });
      g2.planes([{ span: [V3, W3], extent: 2.8, color: 'teal', alpha: 0.26 }]);
      g2.vector([0, 0, 0], V3, { color: 'v', width: 3, label: 'v' });
      g2.vector([0, 0, 0], W3, { color: 'w', width: 3, label: 'w' });
      g2.vector([0, 0, 0], U_OUT, { color: 'b', width: 3, label: 'u' });
      title(rr, '3 · Dimension counts real directions: u leaves the plane → 3');
    });
  }
})();
