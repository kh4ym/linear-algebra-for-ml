/* p2-01 — Linear transformations are matrices (REFERENCE LESSON: the model every other lesson follows). */
(function () {
  'use strict';
  const N = LA.num, lerpM = LA.lerpM, lerpV = LA.lerpV, seg = LA.seg, C = LA.color;
  const I2 = () => [[1, 0], [0, 1]];
  const A = [[2, 1], [0.5, 1.5]];                         // the lesson's main example
  const I3 = () => [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  const A3 = [[1, 0.4, 0], [0.3, 1, 0.2], [0, 0.5, 1.3]];
  const CAM0 = { center: [0, 0], halfH: 3.6 }, CAMW = { center: [1.6, 1], halfH: 3.5 };   // default / wide enough for Av = (5, 2.5)
  const ihat = '<span class="c-i">î</span>', jhat = '<span class="c-j">ĵ</span>', khat = '<span class="c-k">k̂</span>';

  // a smooth non-linear warp used in the misconception step
  function warp(p, a) { return [p[0] + a * 0.35 * Math.sin(1.3 * p[1]), p[1] + a * 0.35 * Math.sin(1.1 * p[0])]; }

  function makePoints() {
    const rng = LA.rng(7); const pts = [], labels = [];
    for (let i = 0; i < 36; i++) { const c = i % 2; const cx = c ? 1.2 : -1.0, cy = c ? 0.6 : -0.5; pts.push([cx + LA.rng.gauss(rng) * 0.45, cy + LA.rng.gauss(rng) * 0.45]); labels.push(c); }
    return { pts, labels };
  }

  LA.registerLesson({
    id: 'p2-01-linear-transformations',
    part: 2,
    title: 'Linear transformations are matrices',
    subtitle: 'The grid bends; the columns of the matrix are where î and ĵ land.',
    minutes: 11,
    prereqs: ['p1-01-what-is-a-vector', 'p1-02-add-scale', 'p1-03-linear-combinations-span'],
    tags: ['core'],
    badge: { drawnIn: '2D, with one 3D step', realSize: 'the weight matrix of a dense layer, e.g. 128×784 (784 inputs, 128 outputs)', transfers: 'everything: columns are landing spots in any number of dimensions' },
    cantShow: ['more than three landing spots at once (a 128×784 matrix has 784 of them)', 'what the data “looks like” in 128 dimensions; only the rule transfers'],
    terms: [
      { term: 'linear transformation', def: 'A way of moving every point of space so that grid lines stay straight and evenly spaced and the origin stays put.' },
      { term: 'matrix', def: 'A rectangle of numbers. For a transformation, its columns record where the basis vectors land.' },
      { term: 'basis vector', def: 'One of the reference arrows (î, ĵ, k̂) that every other vector is built from.' },
      { term: 'column', def: 'A vertical slice of a matrix; the j-th column is where the j-th basis vector lands.' },
    ],
    atlas: ['mlp', 'tensors'],
    thumbnail: 4,

    state() {
      const d = makePoints();
      return {
        view: '2d', cam: LA.cam.defaults2d(),
        M: I2(), M3: I3(),
        showGhost: true, showI: true, showJ: true, showV: false, recipe: false, components: null,
        v: [2, 1], warp: 0, shift: [0, 0], cross: null, showPoints: false, pts: d.pts, labels: d.labels,
        highlightCol: null, grow: 1, showMat: true, matName: 'A',
      };
    },

    render(g, s, ctx) {
      if (s.view === '3d') { render3d(g, s); return; }
      const M = s.M;
      if (s.showGhost) g.ghostGrid();
      if (s.warp > 0) {
        // non-linear warp: grid lines become curves (misconception step)
        for (let i = -8; i <= 8; i++) {
          const a = [], b = [];
          for (let k = -8; k <= 8; k += 0.25) { a.push(warp([i, k], s.warp)); b.push(warp([k, i], s.warp)); }
          g.polyline(a, { color: i === 0 ? 'gridAxis' : 'grid', width: i === 0 ? 1.6 : 1 }); g.polyline(b, { color: i === 0 ? 'gridAxis' : 'grid', width: i === 0 ? 1.6 : 1 });
        }
      } else {
        const T = s.shift[0] || s.shift[1] ? null : M;
        if (T) g.grid(M);
        else { // translated grid (misconception step): shift every line
          for (let i = -10; i <= 10; i++) { g.line([i + s.shift[0], -10 + s.shift[1]], [i + s.shift[0], 10 + s.shift[1]], { color: i === 0 ? 'gridAxis' : 'grid', width: i === 0 ? 1.6 : 1 }); g.line([-10 + s.shift[0], i + s.shift[1]], [10 + s.shift[0], i + s.shift[1]], { color: i === 0 ? 'gridAxis' : 'grid', width: i === 0 ? 1.6 : 1 }); }
          g.line([0, 0], s.shift, { color: 'bad', dashed: true, width: 1.5 });
          g.circle([0, 0], 0.12, { stroke: 'muted', width: 1.5 });
          g.point(s.shift, { r: 5, color: 'bad', label: 'origin moved' });
        }
      }
      const ci = N.col(M, 0), cj = N.col(M, 1);
      if (s.showPoints) {
        const tp = s.pts.map(p => N.matVec(M, p));
        g.points(tp, { r: 4, colors: (i) => s.labels[i] ? C.dataB : C.data });
        if (ctx.mode === 'learn') g.hud('Part 2 capstone: a neural layer', 16, 22, { color: 'muted', anchor: 'left', size: 13 });
      }
      if (s.showV) {
        const vv = N.matVec(M, s.v);
        if (s.recipe) {
          // the recipe: v = 2 copies of (landed î) + 1 copy of (landed ĵ), laid tip to tail
          let o = [0, 0];
          for (let k = 0; k < Math.round(s.v[0]); k++) { g.vector(o, N.add(o, ci), { color: 'i', alpha: 0.55, width: 2, head: 9 }); o = N.add(o, ci); }
          for (let k = 0; k < Math.round(s.v[1]); k++) { g.vector(o, N.add(o, cj), { color: 'j', alpha: 0.55, width: 2, head: 9 }); o = N.add(o, cj); }
        }
        g.vector(vv, { color: 'v', label: 'v', width: 3 });
      }
      const gi = s.grow, gj = s.grow;
      const bad = s.warp > 0 || s.shift[0] || s.shift[1];
      if (s.warp > 0.3) g.label('lines curve', [2.6, 2.2], { color: 'bad', anchor: 'left', dx: 22 });
      if (bad) { if (s.cross) g.marker.cross(s.cross, { size: 26, dx: s.warp > 0 ? 0 : -28, dy: s.warp > 0 ? 0 : 22 }); return; }
      if (s.showI) g.vector([ci[0] * gi, ci[1] * gi], { color: 'i', label: 'i', hat: true, width: 3, components: s.components === 'i' ? 'labels' : false, alpha: s.highlightCol === 1 ? 0.35 : 1 });
      if (s.showJ) g.vector([cj[0] * gj, cj[1] * gj], { color: 'j', label: 'j', hat: true, width: 3, components: s.components === 'j' ? 'labels' : false, alpha: s.highlightCol === 0 ? 0.35 : 1 });
      if (s.cross) g.marker.cross(s.cross, { size: 26 });
      // handles (active in Play / Check modes and in interactive steps)
      g.handle('i', ci, (p, st) => { st.M[0][0] = p[0]; st.M[1][0] = p[1]; }, { color: 'i' });
      g.handle('j', cj, (p, st) => { st.M[0][1] = p[0]; st.M[1][1] = p[1]; }, { color: 'j' });
      if (s.showV && ctx.mode === 'play') g.handle('v', N.matVec(M, s.v), (p, st) => { const inv = N.inv(st.M); if (inv) st.v = N.matVec(inv, p); }, { color: 'v' });
    },

    math(s, tex) {
      const parts = [];
      if (s.view === '3d') { parts.push(tex.eq(tex.sym('A'), tex.op('='), tex.mat(s.M3, { colors: 'cols' }))); return tex.block.apply(null, parts); }
      if (s.warp > 0 || s.shift[0] || s.shift[1]) {
        parts.push(tex.text(s.warp > 0 ? 'lines curve: not linear, no matrix can do this' : 'origin moved: not linear, no matrix can do this', { color: 'bad' }));
        return tex.block.apply(null, parts);
      }
      if (!s.showMat) {
        parts.push(tex.eq(tex.ihat(), tex.op('→'), tex.vec(N.col(s.M, 0), { color: 'i' })));
        parts.push(tex.eq(tex.jhat(), tex.op('→'), tex.vec(N.col(s.M, 1), { color: 'j' })));
        return tex.block.apply(null, parts);
      }
      const hl = s.highlightCol === null || s.highlightCol === undefined ? undefined : (i, j) => j === s.highlightCol;
      parts.push(tex.eq(tex.sym(s.matName || 'A'), tex.op('='), tex.mat(s.M, { colors: 'cols', highlight: hl })));
      if (s.showV) {
        const vv = N.matVec(s.M, s.v);
        parts.push(tex.eq(tex.sym('v'), tex.op('='), tex.vec(s.v, { color: 'v' })));
        parts.push(tex.eq(tex.sym('A'), tex.sym('v'), tex.op('='), tex.num(s.v[0], { color: 'v' }), tex.vec(N.col(s.M, 0), { color: 'i' }), tex.op('+'), tex.num(s.v[1], { color: 'v' }), tex.vec(N.col(s.M, 1), { color: 'j' }), tex.op('='), tex.vec(vv, { color: 'v' })));
      }
      return tex.block.apply(null, parts);
    },

    steps: [
      { title: 'The grid and its two reference arrows', duration: 2.2,
        narration: 'This is the plane as a grid. Two reference arrows define it: the green ' + ihat + ' points one unit right, the red ' + jhat + ' one unit up. Every other point is some number of ' + ihat + ' steps plus some number of ' + jhat + ' steps. Keep an eye on these two arrows.',
        setup(s) { s.M = I2(); s.showGhost = false; s.showV = false; s.recipe = false; s.components = null; s.cross = null; s.warp = 0; s.shift = [0, 0]; s.showPoints = false; s.highlightCol = null; s.grow = 0; s.showMat = false; s.matName = 'A'; },
        at(s, t) { s.grow = seg(t, 0.1, 0.9, 'back'); } },

      { title: 'A linear transformation bends the whole grid', duration: 3.5,
        narration: 'Watch the grid move. Lines stay straight. Parallel lines stay parallel and evenly spaced. The origin does not move. That is what <b>linear</b> means, and every such motion is completely determined by where ' + ihat + ' and ' + jhat + ' end up. The faint grid is the original, kept for reference.',
        setup(s) { s.grow = 1; s.showGhost = true; s._from = LA.clone(s.M); s._to = LA.clone(A); },
        at(s, t) { s.M = lerpM(s._from, s._to, seg(t, 0.05, 0.95)); } },

      { title: 'Where did î land?', duration: 3,
        narration: 'Follow the green arrow. It started at (1, 0) and now ends at (2, 0.5): two steps right, half a step up. Those two numbers are written down as the <b>first column</b> of the matrix on the right. A matrix is a notebook of landing spots.',
        setup(s) { s.M = LA.clone(A); s.components = null; s.highlightCol = null; s.showMat = true; },
        at(s, t) { s.components = t > 0.15 ? 'i' : null; s.highlightCol = t > 0.55 ? 0 : null; } },

      { title: 'Where did ĵ land?', duration: 3,
        narration: 'Now the red arrow. It started at (0, 1) and ends at (1, 1.5). Those numbers are the <b>second column</b>. Read any 2×2 matrix this way: first column is the new ' + ihat + ', second column is the new ' + jhat + '. The colors in the matrix match the arrows on purpose.',
        setup(s) { s.components = null; s.highlightCol = null; },
        at(s, t) { s.components = t > 0.15 ? 'j' : null; s.highlightCol = t > 0.55 ? 1 : null; } },

      { title: 'Replay it with the matrix in view', duration: 3.5,
        narration: 'Here is the same motion once more, this time watching the numbers. As the arrows glide to their landing spots, the columns count along with them. Nothing else is stored anywhere: four numbers, two landing spots, one transformation of the entire plane.',
        setup(s) { s.components = null; s.highlightCol = null; s._from = I2(); s._to = LA.clone(A); },
        at(s, t) { s.M = lerpM(s._from, s._to, seg(t, 0.05, 0.95)); } },

      { title: 'Any vector rides along with its recipe', duration: 4,
        narration: 'Take the yellow vector <span class="c-v">v</span> = 2' + ihat + ' + 1' + jhat + ': two green steps, one red step. After the transformation the recipe is unchanged, only the ingredients moved: two copies of the new ' + ihat + ' plus one copy of the new ' + jhat + '. That sum is <span class="c-v">Av</span>, and it is how a matrix acts on every vector.',
        setup(s) { s.showV = true; s.recipe = true; s._from = I2(); s._to = LA.clone(A); },
        at(s, t) { s.M = lerpM(s._from, s._to, seg(t, 0.35, 0.95)); LA.cam.tween(s, CAM0, CAMW, seg(t, 0, 0.35)); } },

      { title: 'Not every bending is linear', duration: 4, misconception: true,
        narration: 'A common belief: any smooth deformation of the grid is a matrix. No. Here the grid lines curve, so no two landing spots can describe it. And here the whole grid slides: the origin moves, which no matrix can do either. Linear means straight lines, even spacing, fixed origin.',
        setup(s) { s.showV = false; s.recipe = false; s.M = I2(); s.warp = 0; s.shift = [0, 0]; s.cross = null; s.cam = Object.assign(s.cam, CAM0); },
        at(s, t) {
          const u = seg(t, 0, 0.4), w = seg(t, 0.55, 0.95);
          s.warp = t < 0.5 ? u : 1 - seg(t, 0.5, 0.55);
          s.shift = [1.5 * w, 0.8 * w];
          s.cross = t > 0.3 && t < 0.5 ? [2.6, 2.2] : (t > 0.85 ? [1.5, 0.8] : null);
        } },

      { title: 'Drag the landing spots yourself', duration: 1.5, interactive: true,
        narration: 'Grab the tip of ' + ihat + ' or ' + jhat + ' and drag. The grid follows instantly, and the matrix columns follow with it. Try putting both arrows on the same line, then try swapping them. Everything the grid can do linearly, you can do by moving two dots.',
        setup(s) { s.warp = 0; s.shift = [0, 0]; s.cross = null; s.M = LA.clone(A); s.showGhost = true; },
        at(s, t) { } },

      { title: 'The same idea in three dimensions', duration: 4.5,
        narration: 'In 3D there is a third reference arrow, the blue ' + khat + ', and a 3×3 matrix with three columns: the landing spots of ' + ihat + ', ' + jhat + ' and ' + khat + '. The cube of grid lines deforms into a skewed box. Drag to orbit; nothing about the rule changed.',
        setup(s) { s.view = '3d'; s.cam = Object.assign(LA.cam.defaults3d(), { dist: 8.5, yaw: 0.7, pitch: 0.42 }); s.M3 = I3(); s._to3 = LA.clone(A3); },
        at(s, t) { s.M3 = lerpM(I3(), s._to3, seg(t, 0.25, 0.95)); LA.cam.tween(s, { yaw: 0.7 }, { yaw: 1.05 }, seg(t, 0, 1, 'linear')); } },

      { title: 'Why machine learning cares', duration: 4, mlTie: true,
        narration: 'A dense layer in a neural network computes <b>Wx</b>: a weight matrix times the input. That is exactly this picture: the dots are data points of two classes, and the layer moves them all by one linear rule. A real W might be 128×784: 784 landing spots, one per input pixel, each with 128 coordinates. The Part 2 capstone builds such a layer.',
        setup(s) { s.view = '2d'; s.cam = LA.cam.defaults2d(); s.showPoints = true; s.showV = false; s.M = I2(); s.matName = 'W'; s._to = [[1.6, 0.6], [-0.4, 1.2]]; },
        at(s, t) { s.M = lerpM(I2(), s._to, seg(t, 0.2, 0.9)); } },

      { title: 'Recap', duration: 1.5, recap: true,
        narration: 'A matrix is two (or three) landing spots: its columns say where ' + ihat + ', ' + jhat + ' (and ' + khat + ') go. Lines stay straight, spacing stays even, the origin stays put. Every vector keeps its recipe, so <span class="c-v">Av</span> is the same combination of the moved arrows.',
        cards: ['<b>Columns are landing spots.</b> Column j is where the j-th basis arrow ends up.', '<b>Linear</b> = straight lines, even spacing, fixed origin. Curving or sliding the grid is not a matrix.', '<b>Recipes survive.</b> v = 2î + ĵ becomes 2(new î) + (new ĵ) = Av.'],
        setup(s) { s.showPoints = false; s.showV = true; s.recipe = true; s.M = LA.clone(A); s.matName = 'A'; s.cam = Object.assign(LA.cam.defaults2d(), CAMW); },
        at(s, t) { } },
    ],

    playground: {
      intro: '<p>Drag the tips of <span class="c-i">î</span> and <span class="c-j">ĵ</span> (or the yellow <span class="c-v">v</span>), edit the matrix, or try a preset. In 3D, drag to orbit.</p>',
      setup(s) { s.showV = true; s.recipe = true; s.M = LA.clone(A); s.showGhost = true; s.cam = Object.assign(LA.cam.defaults2d(), CAMW); },
      controls: [
        { type: 'matrix', bind: 'M', label: 'A', colors: 'cols', step: 0.1 },
        { type: 'matrix', bind: 'v', label: 'v', colors: 'v', step: 0.1 },
        { type: 'toggle', bind: 'showGhost', label: 'Show the original grid' },
        { type: 'toggle', bind: 'recipe', label: 'Show the recipe for v' },
        { type: 'preset', label: 'Presets', items: [
          { label: 'Identity', values: { M: [[1, 0], [0, 1]] } },
          { label: 'Lesson example', values: { M: A } },
          { label: 'Quarter turn', values: { M: [[0, -1], [1, 0]] } },
          { label: 'Shear', values: { M: [[1, 1], [0, 1]] } },
          { label: 'Squash to a line', values: { M: [[1, 2], [0.5, 1]] } },
        ] },
        { type: 'preset', label: 'View', items: [
          { label: '2D', apply(s) { s.view = '2d'; s.cam = Object.assign(LA.cam.defaults2d(), CAMW); } },
          { label: '3D', apply(s) { s.view = '3d'; s.cam = Object.assign(LA.cam.defaults3d(), { dist: 10.5 }); s.M3 = [[1, 0.4, 0], [0.3, 1, 0.2], [0, 0.5, 1.3]]; } },
        ] },
      ],
    },

    checks: [
      { kind: 'manipulate',
        prompt: 'Drag the landing spots so the matrix turns the plane a quarter turn counter-clockwise (' + ihat + ' should point straight up).',
        setup(s) { s.M = I2(); s.showV = false; s.recipe = false; s.showGhost = true; },
        judge(s) { return LA.judge.matrix(s.M, [[0, -1], [1, 0]], LA.TOL.pos); },
        hint: 'After a quarter turn, ' + ihat + ' lands at (0, 1) and ' + jhat + ' lands at (−1, 0).',
        solution(s) { s.M = [[0, -1], [1, 0]]; } },
      { kind: 'entry',
        prompt: 'Type the matrix that doubles every x-coordinate and leaves every y-coordinate alone.',
        bind: 'M', answer: [[2, 0], [0, 1]], tol: LA.TOL.entry,
        setup(s) { s.M = I2(); s.showV = true; s.recipe = false; } },
      { kind: 'manipulate',
        prompt: 'Squash the plane onto a single line: drag the two landing spots so they point along the same line through the origin.',
        setup(s) { s.M = I2(); s.showV = false; s.recipe = false; },
        judge(s) { const d = N.det(s.M), n0 = N.norm(N.col(s.M, 0)), n1 = N.norm(N.col(s.M, 1)); const ok = Math.abs(d) <= LA.TOL.det && n0 > 0.3 && n1 > 0.3; return { ok, progress: Math.max(0, 1 - Math.abs(d) / 1.5), hint: n0 < 0.3 || n1 < 0.3 ? 'Keep both arrows longer than zero; just line them up.' : undefined }; },
        hint: 'Both arrows should lie on one line, e.g. ' + jhat + ' pointing the same way as ' + ihat + ' (or exactly opposite).',
        solution(s) { s.M = [[1, 2], [0.5, 1]]; } },
    ],
  });

  function render3d(g, s) {
    g.cage({ range: 3, walls: true });
    g.lattice(s.M3, { range: 2, alpha: 0.7 });
    g.box(s.M3, { fill: 'v', alpha: 0.12, stroke: 'v' });
    const c = j => N.col(s.M3, j);
    g.vector([0, 0, 0], c(0), { color: 'i', label: 'i', hat: true, width: 3 });
    g.vector([0, 0, 0], c(1), { color: 'j', label: 'j', hat: true, width: 3 });
    g.vector([0, 0, 0], c(2), { color: 'k', label: 'k', hat: true, width: 3 });
    g.axes({ length: 2.7, labels: true });
  }
})();
