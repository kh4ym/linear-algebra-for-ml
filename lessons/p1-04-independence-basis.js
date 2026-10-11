/* p1-04 — Independence, basis, dimension: three arrows frame a box; the box is flat when one arrow is a recipe of the others. */
(function () {
  'use strict';
  const N = LA.num, seg = LA.seg, lerpV = LA.lerpV;
  const V0 = [2, 1, 0], W0 = [0, 2, 1], U0 = [1, 0, 2], P0 = [1, 5, 0], F0 = [3, 3, 3], TARGET = [0, 0, 2.5];
  const CAM3 = { target: [1.0, 2.0, 0.9], dist: 13, yaw: 0.6, pitch: 0.36 };
  const CAMZ = { target: [1.4, 1.6, 1.4], dist: 13.5 };    // the fourth-arrow step: f = v + w + u is the box's long diagonal
  /* Numbers in the math panel. The engine's in-place panel update only tracks spans whose class starts with "tx-v",
     yet matrix cells also carry that class, so mixing tex.num with tex.vec writes numbers into the wrong cells.
     A plain tabular cell sidesteps that: changing it rebuilds the panel, which is cheap. */
  const nm = (x, color, digits) => '<span class="tx-cell ' + LA.tex.colorClass(color || '') + '">' + LA.fmt(x, digits) + '</span>';
  const vS = '<span class="c-v">v</span>', wS = '<span class="c-w">w</span>', uS = '<span class="c-b">u</span>';
  const ihat = '<span class="c-i">î</span>', jhat = '<span class="c-j">ĵ</span>', khat = '<span class="c-k">k̂</span>';
  const CLOUD_A = [1, 0.39, 0], CLOUD_B = [0, 0, 1];

  /** projection of u onto the plane spanned by v and w */
  function planeProj(u, v, w) {
    const n = N.cross(v, w); const nn = N.dot(n, n); if (nn < 1e-12) return u.slice();
    return N.sub(u, N.scale(n, N.dot(u, n) / nn));
  }
  const UP = planeProj(U0, V0, W0);                 // [0.57, 0.86, 0.29]
  const HEIGHT = N.norm(N.sub(U0, UP));             // 1.96
  const colsOf = s => N.fromCols(s.std ? [[1, 0, 0], [0, 1, 0], s.w] : [s.v, s.w, s.u]);
  const volOf = s => Math.abs(N.det(colsOf(s)));

  function makeCloud() {
    const rand = LA.rng(4); const pts = []; const n = N.normalize(N.cross(CLOUD_A, CLOUD_B));
    for (let i = 0; i < 40; i++) {
      const a = LA.rng.gauss(rand) * 1.1, b = LA.rng.gauss(rand) * 0.9, e = LA.rng.gauss(rand) * 0.05;
      pts.push(N.add(N.add(N.scale(CLOUD_A, a), N.scale(CLOUD_B, b)), N.scale(n, e)));
    }
    return pts;
  }
  function makeHeat() {
    const rand = LA.rng(11); const X = [];
    for (let i = 0; i < 6; i++) { const row = []; for (let j = 0; j < 10; j++) row.push(LA.rng.int(rand, -3, 3)); X.push(row); }
    X[2] = X[0].map((x, j) => x + X[1][j]);           // row 3 = row 1 + row 2
    return X;
  }
  const bracket = v => '[' + v.map(x => LA.fmt(x)).join('; ') + ']';

  LA.registerLesson({
    id: 'p1-04-independence-basis',
    part: 1,
    title: 'Independence, basis, dimension',
    subtitle: 'Three arrows frame a box; when one is a recipe of the others, the box goes flat.',
    minutes: 9,
    prereqs: ['p1-03-linear-combinations-span'],
    tags: ['core'],
    badge: { drawnIn: '3D (three vectors)', realSize: 'a design matrix with 50 features whose columns are dependent (duplicated, or one-hot plus intercept) breaks ordinary regression', transfers: 'independence = “no vector is a recipe of the others” in any dimension; it is tested by a computation (rank, Part 2), not by eye' },
    cantShow: ['independence of 50 columns (we show three arrows and a volume; the rank computation is the general test)'],
    terms: [
      { term: 'independent', def: 'No vector in the set is a linear combination of the others; the spanned region has as many dimensions as there are vectors.' },
      { term: 'dependent', def: 'At least one vector is a recipe of the others; the spanned region loses a dimension.' },
      { term: 'basis', def: 'An independent set that spans the whole space: a complete coordinate system with no redundancy.' },
      { term: 'dimension', def: 'The number of vectors in any basis of a space; 3 for our room.' },
      { term: 'coordinates', def: 'The recipe scalars that build a vector from the basis vectors.' },
      { term: 'standard basis', def: 'The axis arrows î, ĵ, k̂.' },
    ],
    atlas: ['linear-regression', 'pca'],
    thumbnail: 2,

    state() {
      return {
        view: '3d', cam: Object.assign(LA.cam.defaults3d(), CAM3),
        v: V0.slice(), w: W0.slice(), u: U0.slice(),
        grow: 1, box: 1, showBox: true, plane: 0, drop: 0,
        target: false, sweep: 0, cross: false,
        showP: false, recipe: 0, pulse: 0, coords: [1, 2, -1], recipeColor: null, brackets: false,
        fourth: 0, fourthRecipe: 0, fourthLabel: false,
        showLattice: false, latticeT: 0, ghostLattice: false,
        cloud: false, cloudPlane: 0, cloudPts: makeCloud(),
        X: makeHeat(), heatT: 0,
        std: false, drag: [], panel: 'vectors', dimHud: false, uDragged: false,
      };
    },

    render(g, s, ctx) {
      if (s.view === '2d') { renderHeat(g, s); return; }
      if (s.cloud) { renderCloud(g, s); return; }
      const M = colsOf(s), V = Math.abs(N.det(M)), dep = V < 0.05;
      g.cage({ range: 4, walls: true });
      if (s.ghostLattice) g.lattice(null, { range: 2, color: 'ghost', axisColor: 'ghostAxis', alpha: 0.7, width: 0.8 });
      if (s.showLattice && s.latticeT > 0) g.lattice(M, { range: 2, alpha: 0.6 * s.latticeT, color: 'grid', axisColor: 'gridAxis', width: 0.9 });
      g.axes({ length: 3.2, labels: true });
      // the teal plane of v and w (or the floor when the standard basis is in play)
      if (s.plane > 0) g.planes([{ point: [0, 0, 0], span: s.std ? [[1, 0, 0], [0, 1, 0]] : [s.v, s.w], extent: 3.4, color: 'teal', alpha: 0.2 * s.plane, label: s.std ? 'span(î, ĵ)' : 'span(v, w)' }]);
      // the box framed by the three vectors
      if (s.showBox && s.box > 0) g.box(M, { fill: dep ? 'mag' : 'v', alpha: (dep ? 0.1 : 0.13) * s.box, stroke: dep ? 'mag' : 'v', faces: V > 0.02, strokeAlpha: 0.9 * s.box });
      // arrows
      const gr = s.grow;
      const a = (vec, color, label, hat) => { if (N.norm(vec) > 1e-6) g.vector([0, 0, 0], N.scale(vec, gr), { color, label, hat, width: 3 }); };
      if (s.std) { a([1, 0, 0], 'i', 'i', true); a([0, 1, 0], 'j', 'j', true); a(s.w, 'w', 'w'); }
      else { a(s.v, 'v', 'v'); a(s.w, 'w', 'w'); a(s.u, 'b', 'u'); }
      // drop from u's tip to the plane
      if (s.drop > 0 && !s.std) {
        const foot = planeProj(s.u, s.v, s.w), end = lerpV(s.u, foot, s.drop);
        g.line(s.u, end, { color: 'residual', dashed: true, width: 1.3 });
        if (s.drop >= 1) { g.point(foot, { r: 3.5, color: 'mag' }); g.dim(s.u, foot, { label: 'height ' + LA.fmt(N.norm(N.sub(s.u, foot))), offset: 18, color: 'hud' }); }
      }
      // volume readout, pinned above the box's top face
      if (s.showBox && s.box > 0 && !(s.fourth > 0)) {
        const corner = N.add(N.add(N.col(M, 0), N.col(M, 1)), N.col(M, 2));
        const txt = dep ? 'volume 0 → dependent' : 'volume ' + LA.fmt(V);
        g.label(txt, corner, { color: dep ? 'mag' : 'v', anchor: 'above', dy: -6, size: 13.5, font: 'ui', bold: dep });
      }
      // the unreachable target (misconception step)
      if (s.target) {
        const a1 = 1.2 * Math.sin(s.sweep * Math.PI * 2.0), b1 = 1.1 * Math.cos(s.sweep * Math.PI * 1.3);
        const tip = N.add(N.scale(s.v, a1), N.scale(s.w, b1));
        g.point(tip, { r: 5, color: 'gold', label: 'a·v + b·w', dx: 12, dy: 10, size: 12 });
        g.line(tip, TARGET, { color: 'residual', dashed: true, width: 1.2 });
        g.point(TARGET, { r: 5.5, color: 'muted', label: 'target', dx: 12, dy: -10 });
        g.label('"three arrows span 3D"', [18, 30], { screen: true, anchor: 'left', color: 'muted', size: 14, font: 'ui' });
        if (s.cross) { g.marker.cross([186, 30], { screen: true, size: 22 }); g.label('unreachable', TARGET, { color: 'bad', dx: 14, dy: 12, size: 12.5, font: 'ui' }); }
      }
      // the gold point and its recipe in the basis {v, w, u}
      if (s.showP) {
        if (s.recipe > 0) drawRecipe(g, s, s.coords, s.recipe, s.recipeColor);
        g.point(P0, { r: 5.5 + 4 * s.pulse, color: 'gold', label: 'p', dx: 12, dy: -10 });
        if (s.brackets) {
          g.label('p = ' + bracket(P0) + ' in {î, ĵ, k̂}', P0, { color: 'text', dx: -14, dy: 14, anchor: 'right', size: 13, font: 'ui' });
          g.label('p = ' + bracket(s.coords) + ' in {v, w, u}', P0, { color: 'gold', dx: -14, dy: 32, anchor: 'right', size: 13, font: 'ui' });
        }
      }
      // a fourth arrow is always a recipe of the other three
      if (s.fourth > 0) {
        g.vector([0, 0, 0], N.scale(F0, s.fourth), { color: 'white', label: 'f = [3; 3; 3]', width: 3.4, labelFont: 'ui', labelSize: 13 });
        if (s.fourthRecipe > 0) drawRecipe(g, s, [1, 1, 1], s.fourthRecipe, null);
        if (s.fourthLabel) g.hud('f = 1·v + 1·w + 1·u: the fourth arrow is a recipe of the other three', 14, g.size.h - 40, { color: 'hud' });
      }
      if (s.dimHud) g.hud('dim = 3: every basis of this room has exactly three vectors', 14, g.size.h - 18, { color: 'hud' });
      // live readouts in Play mode
      if (ctx.mode === 'play') {
        const r = N.rank(M);
        g.hud('volume of the box = ' + LA.fmt(V) + '   ·   dimension of span = ' + r, 14, 22, { color: 'hud' });
        const c = r === 3 ? N.solve(M, P0) : null;
        g.hud('coordinates of p in {v, w, u}: ' + (c ? bracket(c) : 'none (basis is dependent)'), 14, 42, { color: c ? 'gold' : 'mag' });
      }
      // handles: drag in the upright plane facing the camera, so moving the mouse up/down raises or lowers a tip
      // ('auto' picks the floor plane at this camera pitch, which could never lift a tip off the floor)
      const fw = LA.cam.forward(s.cam), fh = Math.hypot(fw[0], fw[1]) > 0.05 ? [fw[0], fw[1], 0] : [1, 0, 0];
      const H = (id, vec, color) => g.handle(id, vec, (p, st) => {
        let q = p.slice();
        if (id === 'u') {
          st.uDragged = true;
          const foot = planeProj(q, st.v, st.w);
          if (N.dist(q, foot) < 0.2) q = foot;          // the teal sheet is magnetic when u's tip is close
        }
        st[id] = q;
      }, { color, plane: { point: vec, normal: fh } });
      if (s.drag.indexOf('v') >= 0) H('v', s.v, 'v');
      if (s.drag.indexOf('w') >= 0) H('w', s.w, 'w');
      if (s.drag.indexOf('u') >= 0) H('u', s.u, 'b');
    },

    math(s, tex) {
      const sv = tex.sym('v', { color: 'v' }), sw = tex.sym('w', { color: 'w' }), su = tex.sym('u', { color: 'b' });
      const V = volOf(s), vc = V < 0.05 ? 'mag' : 'v';
      switch (s.panel) {
        case 'vectors':
          return tex.lines(
            tex.eq(sv, tex.op('='), tex.vec(s.v, { color: 'v' }), tex.op('  '), sw, tex.op('='), tex.vec(s.w, { color: 'w' }), tex.op('  '), su, tex.op('='), tex.vec(s.u, { color: 'b' })),
            tex.eq(tex.text('volume'), tex.op('='), nm(V)));
        case 'indep':
          return tex.block(tex.eq(su, tex.op('≠'), tex.sym('a'), tex.op('·'), sv, tex.op('+'), tex.sym('b'), tex.op('·'), sw, tex.text(' for any a, b')));
        case 'volume': {
          const inside = s.std ? tex.ihat() + tex.op(',') + tex.jhat() + tex.op(',') + sw : sv + tex.op(',') + sw + tex.op(',') + su;
          return tex.block(tex.eq(tex.text('volume of the box framed by'), inside, tex.text(' is '), nm(V, vc)));
        }
        case 'span2':
          return tex.block(tex.eq(tex.text('span'), tex.paren(sv + tex.op(', ') + sw + tex.op(', ') + su), tex.op('='), tex.text('plane', { color: 'teal' }), tex.text('  (dimension 2)')));
        case 'coords':
          return tex.lines(
            tex.eq(tex.sym('p', { color: 'gold' }), tex.op('='), tex.vec(P0, { color: 'gold' }), tex.op('='), nm(s.coords[0], 'v'), tex.op('·'), sv, tex.op('+'), nm(s.coords[1], 'w'), tex.op('·'), sw, tex.op(s.coords[2] < 0 ? '−' : '+'), nm(Math.abs(s.coords[2]), 'b'), tex.op('·'), su),
            tex.eq(tex.text('coordinates of p in {v, w, u}'), tex.op('='), tex.vec(s.coords, { colors: ['gold'] })));
        case 'fourth':
          return tex.block(tex.eq(tex.vec(F0, { colors: ['muted'] }), tex.op('='), tex.text('1·'), sv, tex.op('+'), tex.text('1·'), sw, tex.op('+'), tex.text('1·'), su));
        case 'lattice':
          return tex.lines(
            tex.eq(tex.text('standard basis: '), tex.sym('p', { color: 'gold' }), tex.op('='), tex.vec(P0, { colors: 'rows' })),
            tex.eq(tex.text('basis {v, w, u}: '), tex.sym('p', { color: 'gold' }), tex.op('='), tex.vec(s.coords, { colors: ['gold'] })));
        case 'ml':
          return tex.block(tex.eq(tex.text('inches'), tex.op('='), nm(0.39), tex.op('·'), tex.text('centimeters'), tex.op('  ⇒  '), tex.text('columns dependent', { color: 'mag' })));
        case 'rank':
          return tex.block(tex.eq(tex.text('rank'), tex.paren(tex.sym('X')), tex.op('<'), tex.text('number of rows'), tex.op('  ⇔  '), tex.text('dependent', { color: 'mag' })));
        case 'play': {
          const M = colsOf(s), r = N.rank(M), c = r === 3 ? N.solve(M, P0) : null;
          return tex.lines(
            tex.eq(tex.text('volume'), tex.op('='), nm(V, vc), tex.op('   '), tex.text('dim span'), tex.op('='), nm(r)),
            c ? tex.eq(tex.sym('p', { color: 'gold' }), tex.op('='), tex.vec(P0, { color: 'gold' }), tex.op('='), tex.vec(c, { colors: ['gold'] }), tex.text(' in {v, w, u}')) : tex.eq(tex.text('no coordinates: the set is not a basis', { color: 'mag' })));
        }
        default: return '';
      }
    },

    steps: [
      { title: 'Three vectors, one box', duration: 4,
        narration: 'Three vectors in 3D: the yellow ' + vS + ' = [2; 1; 0], the purple ' + wS + ' = [0; 2; 1], the orange ' + uS + ' = [1; 0; 2]. Together they frame a slanted box. The box has volume 9: it is a real three-dimensional thing, and that is the picture of independence.',
        setup(s) { s.view = '3d'; s.cam = Object.assign(LA.cam.defaults3d(), CAM3); s.std = false; s.v = V0.slice(); s.w = W0.slice(); s.u = U0.slice(); s.grow = 0; s.box = 0; s.showBox = true; s.plane = 0; s.drop = 0; s.panel = 'vectors'; s.drag = []; },
        at(s, t) { s.grow = seg(t, 0.05, 0.5, 'back'); s.box = seg(t, 0.5, 0.9); LA.cam.tween(s, { yaw: 0.52 }, { yaw: 0.68 }, seg(t, 0, 1, 'linear')); } },

      { title: 'Independent means no one is a recipe of the others', duration: 3,
        narration: 'These three are <b>independent</b>: none of them can be built as a recipe of the other two. Any recipe of ' + vS + ' and ' + wS + ' stays on their teal plane, and ' + uS + ' pokes out of it, 1.96 above the sheet. Three independent arrows give three directions.',
        setup(s) { s.grow = 1; s.box = 1; s.panel = 'indep'; },
        at(s, t) { s.plane = seg(t, 0.05, 0.45); s.drop = seg(t, 0.5, 0.95); } },

      { title: 'Drag u into the plane', duration: 5, interactive: true,
        narration: 'Drag the orange ' + uS + ' down toward the teal plane. The box gets thinner and its volume counter falls. The moment u lies in the plane, the volume is 0: the box is flat. Now u <i>is</i> a recipe of v and w, and the set is <b>dependent</b>.',
        setup(s) { s.plane = 1; s.drop = 0; s.panel = 'volume'; s.drag = ['u']; s.uDragged = false; s._uFrom = LA.clone(s.u); s._uTo = UP.slice(); },
        at(s, t) { if (!s.uDragged) s.u = lerpV(s._uFrom, s._uTo, seg(t, 0.05, 0.8)); } },

      { title: 'Three vectors in 3D do not always span 3D', duration: 4, misconception: true,
        narration: 'Belief: "three vectors in 3D always fill 3D." With ' + uS + ' flat in the plane, every recipe of the three stays on the teal sheet. The gray target floating above it can never be reached. Count directions, not arrows.',
        setup(s) { s.u = UP.slice(); s.drag = []; s.target = true; s.sweep = 0; s.cross = false; s.panel = 'span2'; },
        at(s, t) { s.sweep = seg(t, 0, 0.7, 'linear'); s.cross = t > 0.8; } },

      { title: 'A basis gives coordinates', duration: 5,
        narration: 'Put ' + uS + ' back. Three independent vectors that span the room form a <b>basis</b>: a complete coordinate system. The gold point [1; 5; 0] is reached by 1 copy of v, 2 of w, and −1 of u. Those scalars, [1; 2; −1], are its <b>coordinates</b> in the basis {v, w, u}.',
        setup(s) { s.target = false; s.cross = false; s.plane = 0; s.showP = false; s.recipe = 0; s.pulse = 0; s.coords = [1, 2, -1]; s.recipeColor = null; s.brackets = false; s.panel = 'coords'; s._uFrom = LA.clone(s.u); },
        at(s, t) { s.u = lerpV(s._uFrom, U0, seg(t, 0, 0.25)); s.showP = t > 0.25; s.recipe = seg(t, 0.3, 0.8); s.pulse = seg.pulse(t, 0.8, 1); s.brackets = t > 0.85; } },

      { title: 'Dimension: every basis has the same size', duration: 4,
        narration: 'Any basis of this room has exactly three vectors; that number is the <b>dimension</b>. Add a white fourth arrow and it is always a recipe of the other three: f = [3; 3; 3], the box’s far corner, is one copy each of ' + vS + ', ' + wS + ' and ' + uS + '. Four arrows in 3D are never independent.',
        setup(s) { s.u = U0.slice(); s.showP = false; s.recipe = 0; s.pulse = 0; s.fourth = 0; s.fourthRecipe = 0; s.fourthLabel = false; s.panel = 'fourth'; s.dimHud = false; s._camFrom = { target: s.cam.target.slice(), dist: s.cam.dist }; },
        at(s, t) { const z = seg(t, 0, 0.35); s.box = 1 - 0.75 * z; LA.cam.tween(s, s._camFrom, CAMZ, z); s.fourth = seg(t, 0.1, 0.4, 'back'); s.fourthRecipe = seg(t, 0.4, 0.85); s.fourthLabel = t > 0.85; s.dimHud = t > 0.5; } },

      { title: 'The standard basis is one basis among many', duration: 4,
        narration: ihat + ', ' + jhat + ', ' + khat + ' are the <b>standard basis</b>; its lattice is the familiar cube grid. The basis {v, w, u} has its own lattice: a skewed grid of the same room, where every point still has three coordinates. Part 4 changes between such coordinate systems on purpose.',
        setup(s) { s.fourth = 0; s.fourthRecipe = 0; s.fourthLabel = false; s.dimHud = false; s.box = 0.35; s.ghostLattice = true; s.showLattice = true; s.latticeT = 0; s.showP = true; s.recipe = 1; s.brackets = true; s.panel = 'lattice'; s._camFrom = { target: s.cam.target.slice(), dist: s.cam.dist }; },
        at(s, t) { LA.cam.tween(s, s._camFrom, { target: CAM3.target, dist: CAM3.dist }, seg(t, 0, 0.35)); s.latticeT = seg(t, 0.3, 0.8); } },

      { title: 'Redundant features are dependent columns', duration: 5, mlTie: true,
        narration: 'A dataset with height in centimeters and height in inches has two columns that are multiples of each other: dependent. The data cloud then lies flat in a plane inside its 3D feature room; one direction carries no information. Part 3 shows why this breaks regression and Part 4 how to find such flat clouds.',
        setup(s) { s.ghostLattice = false; s.showLattice = false; s.showP = false; s.recipe = 0; s.brackets = false; s.box = 1; s.cloud = true; s.cloudPlane = 0; s.panel = 'ml'; },
        at(s, t) { s.cloudPlane = seg(t, 0.3, 0.7); LA.cam.tween(s, { yaw: 0.68 }, { yaw: 0.95 }, seg(t, 0, 1, 'linear')); } },

      { title: 'In high dimensions, independence is computed', duration: 3,
        narration: 'You cannot look at fifty 10 000-component strips and see whether one is a recipe of the others. Independence is decided by a calculation, the rank in Part 2, that works on strips of any length. Here row 3 equals row 1 plus row 2. The picture stops at 3. The formula does not.',
        setup(s) { s.cloud = false; s.view = '2d'; s.cam = Object.assign(LA.cam.defaults2d(), { center: [0, -0.2], halfH: 3.4 }); s.heatT = 0; s.panel = 'rank'; },
        at(s, t) { s.heatT = t; } },

      { title: 'Recap', duration: 2, recap: true,
        narration: 'Independent vectors give new directions: the box they frame has volume. Dependent sets collapse the box flat, because one vector is a recipe of the others. A basis is an independent spanning set; its size is the dimension, the same for every basis; coordinates are recipe scalars. Redundant features are dependent columns.',
        cards: ['<b>Independent: the box has volume.</b> No arrow is a recipe of the others; three arrows, three directions.', '<b>Dependent: the box is flat.</b> One arrow is a recipe of the others and a direction is lost; the volume reads 0.', '<b>Basis → coordinates; dimension = basis size.</b> The same gold point reads [1; 5; 0] on the cube grid and [1; 2; −1] on the skewed one.'],
        setup(s) { s.view = '3d'; s.cam = Object.assign(LA.cam.defaults3d(), CAM3, { yaw: 0.68 }); s.v = V0.slice(); s.w = W0.slice(); s.u = U0.slice(); s.box = 1; s.plane = 0; s.drop = 0; s.showP = true; s.recipe = 1; s.brackets = true; s.pulse = 0; s.showLattice = true; s.latticeT = 0.6; s.ghostLattice = false; s.panel = 'vectors'; },
        at(s, t) { } },
    ],

    playground: {
      intro: '<p>Drag the tips of ' + vS + ', ' + wS + ' and ' + uS + ' (or edit their numbers). Watch the volume, the dimension of the span, and the coordinates of the gold point in the basis {v, w, u}. Left-drag on empty space orbits.</p>',
      setup(s) { s.view = '3d'; s.cam = Object.assign(LA.cam.defaults3d(), CAM3); s.std = false; s.v = V0.slice(); s.w = W0.slice(); s.u = U0.slice(); s.grow = 1; s.box = 1; s.showBox = true; s.plane = 0; s.drop = 0; s.showP = true; s.recipe = 1; s.brackets = false; s.coords = [1, 2, -1]; s.drag = ['v', 'w', 'u']; s.panel = 'play'; s.showLattice = false; s.latticeT = 1; },
      controls: [
        { type: 'matrix', bind: 'v', label: 'v (drag tip)', colors: 'v', step: 0.1 },
        { type: 'matrix', bind: 'w', label: 'w', colors: 'w', step: 0.1 },
        { type: 'matrix', bind: 'u', label: 'u', colors: 'b', step: 0.1 },
        { type: 'toggle', bind: 'showBox', label: 'Box and volume' },
        { type: 'toggle', bind: 'showLattice', label: 'Lattice of {v, w, u}' },
        { type: 'preset', label: 'Presets', items: [
          { label: 'Independent (volume 9)', values: { v: V0, w: W0, u: U0 } },
          { label: 'u in the plane', values: { v: V0, w: W0, u: UP } },
          { label: 'All collinear', values: { v: [2, 1, 0], w: [-1, -0.5, 0], u: [1, 0.5, 0] } },
          { label: 'Standard basis', values: { v: [1, 0, 0], w: [0, 1, 0], u: [0, 0, 1] } },
        ] },
      ],
    },

    checks: [
      { kind: 'manipulate',
        prompt: 'Drag the tip of the orange ' + uS + ' into the teal plane of ' + vS + ' and ' + wS + ' so the three vectors become dependent. The plane is magnetic: u snaps onto it when its tip comes within 0.2.',
        setup(s) { s.view = '3d'; s.cam = Object.assign(LA.cam.defaults3d(), CAM3); s.std = false; s.v = V0.slice(); s.w = W0.slice(); s.u = U0.slice(); s.grow = 1; s.box = 1; s.showBox = true; s.plane = 1; s.drop = 0; s.showP = false; s.recipe = 0; s.drag = ['u']; s.panel = 'volume'; },
        judge(s) { const d = volOf(s); return { ok: d <= LA.TOL.det, progress: Math.max(0, 1 - d / 9) }; },
        hint: 'Watch the volume counter and move ' + uS + ' toward the teal sheet until it snaps on and the volume reads 0. Any point of the sheet works.',
        solution(s) { s.u = UP.slice(); } },
      { kind: 'manipulate',
        prompt: 'Drag the tip of the purple ' + wS + ' (moving the mouse up or down lifts or lowers it) so that {' + ihat + ', ' + jhat + ', ' + wS + '} is a basis of the room.',
        setup(s) { s.view = '3d'; s.cam = Object.assign(LA.cam.defaults3d(), CAM3, { target: [0.6, 0.7, 0.6], dist: 9.5 }); s.std = true; s.w = [1, 1, 0]; s.grow = 1; s.box = 1; s.showBox = true; s.plane = 1; s.drop = 0; s.showP = false; s.recipe = 0; s.drag = ['w']; s.panel = 'volume'; },
        judge(s) { const z = Math.abs(s.w[2]); return { ok: z > 0.5, progress: Math.min(1, z / 0.5), hint: z > 0.25 ? 'Almost: lift it a little higher above the floor.' : undefined }; },
        hint: wS + ' must leave the floor: give it a blue component, up or down.',
        solution(s) { s.w = [1, 1, 1.5]; } },
      { kind: 'entry',
        prompt: 'Type the coordinates of the gold point p = [1; 5; 0] in the basis {' + vS + ', ' + wS + ', ' + uS + '}: the recipe is drawn live in orange.',
        bind: 'coords', answer: [1, 2, -1], tol: LA.TOL.entry, label: 'coordinates', colors: 'b',
        setup(s) { s.view = '3d'; s.cam = Object.assign(LA.cam.defaults3d(), CAM3); s.std = false; s.v = V0.slice(); s.w = W0.slice(); s.u = U0.slice(); s.grow = 1; s.box = 0.5; s.showBox = true; s.plane = 0; s.drop = 0; s.showP = true; s.recipe = 1; s.recipeColor = 'b'; s.coords = [0, 0, 0]; s.brackets = false; s.drag = []; s.panel = 'coords'; },
        hint: 'Count copies of v, then of w, then of u; one of them is negative.' },
    ],
  });

  /** Lay the recipe c[0]·v + c[1]·w + c[2]·u tip to tail; prog ∈ [0,1] grows the three pieces in turn. */
  function drawRecipe(g, s, c, prog, color, coefText) {
    const parts = [[s.v, 'v', 'v'], [s.w, 'w', 'w'], [s.u, 'b', 'u']];
    let o = [0, 0, 0];
    for (let k = 0; k < 3; k++) {
      const f = LA.clamp01(prog * 3 - k); if (f <= 0) break;
      const d = N.scale(parts[k][0], c[k] * f), e = N.add(o, d);
      if (N.norm(d) > 1e-6) g.vector(o, e, { color: color || parts[k][1], width: 2.2, head: 9, dashed: true, alpha: 0.95, label: (coefText || LA.fmt(c[k])) + '·' + parts[k][2], labelAt: 0.5, labelSize: 12 });
      o = e;
    }
    return o;
  }

  /** ML tie-in: a flat data cloud in a 3D feature room (cm, in, a third feature). */
  function renderCloud(g, s) {
    g.cage({ range: 4, walls: true });
    g.axes({ length: 3.2, labels: true, names: ['cm', 'in', 'f₃'] });
    if (s.cloudPlane > 0) g.planes([{ point: [0, 0, 0], span: [CLOUD_A, CLOUD_B], extent: 3.4, color: 'teal', alpha: 0.22 * s.cloudPlane, label: 'rank 2 cloud' }]);
    g.points(s.cloudPts, { r: 4, color: 'data', alpha: 0.95 });
    // on the floor every row satisfies in = 0.39 · cm: the cloud's footprint is this one line
    g.line(N.scale(CLOUD_A, -3), N.scale(CLOUD_A, 3), { color: 'mag', dashed: true, width: 1.4 });
    g.label('in = 0.39 · cm', N.scale(CLOUD_A, 2.6), { color: 'mag', dy: 16, size: 12.5, font: 'ui' });
    g.hud('synthetic data: 40 rows, 3 features', 14, 22, { color: 'muted' });
    g.hud('rank 2 cloud · Part 2: rank · Part 4: PCA', 14, g.size.h - 18, { color: 'hud' });
  }

  /** n-D honesty step: a 6×10 heatmap whose third row is the sum of the first two. */
  function renderHeat(g, s) {
    const X = s.X, rows = X.length, cols = X[0].length;
    const x0 = -2.5, x1 = 2.5, y0 = -1.5, y1 = 1.5, cw = (x1 - x0) / cols, ch = (y1 - y0) / rows;
    const b = { x0, x1, y0, y1 };
    g.heat2d(X, b, { range: [-6, 6], ramp: t => LA.color.diverge(t * 2 - 1), alpha: 0.95, smooth: false });
    for (let i = 0; i <= rows; i++) g.line([x0, y1 - i * ch], [x1, y1 - i * ch], { color: 'line', width: 1 });
    for (let j = 0; j <= cols; j++) g.line([x0 + j * cw, y0], [x0 + j * cw, y1], { color: 'line', width: 1 });
    for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) g.label(LA.fmt(X[i][j]), [x0 + (j + 0.5) * cw, y1 - (i + 0.5) * ch], { color: 'text', size: 12, font: 'mono', bg: false });
    for (let i = 0; i < rows; i++) g.label('row ' + (i + 1), [x0, y1 - (i + 0.5) * ch], { color: i === 2 ? 'mag' : 'muted', anchor: 'right', dx: -10, size: 12.5, font: 'ui' });
    g.label('X: 6 rows × 10 columns (synthetic)', [0, y1], { color: 'muted', anchor: 'above', dy: -10, size: 13, font: 'ui' });
    const t = s.heatT;
    const outline = (i, color, alpha) => g.polygon([[x0, y1 - i * ch], [x1, y1 - i * ch], [x1, y1 - (i + 1) * ch], [x0, y1 - (i + 1) * ch]], { fill: null, stroke: color, width: 2.4, alpha });
    if (t > 0.1) outline(0, 'v', seg(t, 0.1, 0.3));
    if (t > 0.3) outline(1, 'w', seg(t, 0.3, 0.5));
    if (t > 0.5) outline(2, 'mag', 0.55 + 0.45 * Math.sin(seg(t, 0.5, 1, 'linear') * Math.PI * 3));
    if (t > 0.55) g.label('row 3 = row 1 + row 2  →  dependent', [0, y0], { color: 'mag', anchor: 'below', dy: 10, size: 14, font: 'ui', bold: true });
    if (t > 0.75) g.label('rank(X) = 5 < 6: one row carries no new direction', [0, y0], { color: 'hud', anchor: 'below', dy: 34, size: 12.5, font: 'ui' });
    g.label('The picture stops at 3. The formula does not.', [0, y0], { color: 'muted', anchor: 'below', dy: 62, size: 13, font: 'ui' });
  }
})();
