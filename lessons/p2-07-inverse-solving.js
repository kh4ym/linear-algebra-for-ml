/* p2-07 — Inverse: running the map backwards; solving Ax = b. */
(function () {
  'use strict';
  const N = LA.num, lerpM = LA.lerpM, lerpV = LA.lerpV, seg = LA.seg;
  const I2 = () => [[1, 0], [0, 1]];
  const A = [[2, 1], [1, 1]], AINV = [[1, -1], [-1, 2]], WRONG = [[0.5, 1], [1, 1]], SING = [[1, 2], [2, 4]];
  const B0 = [3, 2], X0 = [1, 1];
  // six inputs that A = SING sends to the same output FT = (0.5, 1): x + 2y = 0.5
  const FAN = [[0.5, 0], [-1.5, 1], [2.5, -1], [-2.5, 1.5], [1.5, -0.5], [-0.5, 0.5]], FT = [0.5, 1];
  // the ML step: three synthetic data points and their normal equations
  const DX = [0, 1, 2], DY = [0.6, 1.9, 2.4];
  const XTX = [[3, 3], [3, 5]], XTY = [DY[0] + DY[1] + DY[2], DY[1] + 2 * DY[2]];
  const WFIT = N.solve(XTX, XTY);
  // nearly parallel lines (2° apart) through (0, 2); the second right-hand side is nudged
  const TH = 2 * Math.PI / 180, NUDGE = 0.06;
  const ihat = '<span class="c-i">î</span>', jhat = '<span class="c-j">ĵ</span>';
  const bS = '<span class="c-b">b</span>', xS = '<span class="c-v">x</span>';
  const cam2 = (cx, cy, hh) => Object.assign(LA.cam.defaults2d(), { center: [cx, cy], halfH: hh });

  function curvePt(a, b, u, bump) { // point on a bowed path from a to b
    const p = lerpV(a, b, u), d = N.sub(b, a), n = N.normalize([-d[1], d[0]]);
    return N.add(p, N.scale(n, bump * Math.sin(Math.PI * u)));
  }
  function playG(s) { // grid matrix in the playground: forward under M, backward under M⁻¹
    const inv = Math.abs(N.det(s.M)) > 0.02 ? N.inv(s.M) : null;
    if (s.morph >= 0) return { G: lerpM(I2(), s.M, s.morph), inv };
    return { G: inv ? lerpM(I2(), inv, -s.morph) : I2(), inv };
  }

  LA.registerLesson({
    id: 'p2-07-inverse-solving',
    part: 2,
    title: 'Inverse: running the map backwards',
    subtitle: 'Play the movie in reverse to undo a matrix and to solve Ax = b.',
    minutes: 9,
    prereqs: ['p2-06-determinant'],
    tags: ['core'],
    badge: { drawnIn: '2D', realSize: 'the normal equations (XᵀX)w = Xᵀy are solved with XᵀX of size d×d, d up to thousands', transfers: 'inverse = reverse map and “det = 0 ⇒ no inverse” hold for every square matrix; the 2×2 formula is special to 2×2, the idea is not' },
    cantShow: ['a d×d inverse as a grid motion (we show 2D; the reverse-movie meaning is the same)'],
    terms: [
      { term: 'inverse', def: 'The matrix A⁻¹ of the reverse transformation: doing A and then A⁻¹ moves nothing.' },
      { term: 'identity', def: 'The matrix I that moves nothing: 1s on the diagonal, 0s elsewhere.' },
      { term: 'Solving', def: 'Solving Ax = b means finding the input x that A sends to b; when A⁻¹ exists, x = A⁻¹b.' },
      { term: 'invertible', def: 'Having an inverse; for a square matrix this is the same as det ≠ 0.' },
      { term: 'singular', def: 'det = 0: the map flattens the plane, so no inverse exists.' },
    ],
    atlas: ['linear-regression', 'optimization'],
    thumbnail: 2,

    state() {
      return {
        view: '2d', cam: cam2(1.2, 1.0, 3.4), scene: 'grid',
        G: I2(), showGhost: true, cols: true, rider: null, b: null, trav: null, path: false, x: null,
        recipe: 0, ringB: false, okAt: null, claim: null, wrongDim: false, colLine: null, fan: 0, solLine: 0, xMark: null,
        hud: [], ft: 0, mlT: 0, panel: 'A',
        pg: false, M: LA.clone(A), bb: B0.slice(), morph: 1, showInv: false, chk: null, Minv: I2(), xIn: [1, 1],
      };
    },

    render(g, s, ctx) {
      if (s.scene === 'formula') { renderFormula(g, s); return; }
      if (s.scene === 'ml') { renderML(g, s); return; }
      if (s.scene === 'recap') { renderRecap(g); return; }
      if (s.pg && ctx.mode === 'play') { renderPlay(g, s); return; }
      if (s.chk) { renderCheck(g, s); return; }
      const G = s.G;
      if (s.showGhost) g.ghostGrid();
      g.grid(G, { alpha: s.fan > 0 || s.solLine > 0 ? 0.6 : 1 });
      if (s.colLine) { g.lineThrough([0, 0], s.colLine, { color: 'teal', width: 2.4, alpha: 0.9 }); g.label('every output lands here', N.scale(N.normalize(s.colLine), 3.3), { color: 'teal', anchor: 'left', dx: 12, size: 13, font: 'ui' }); }
      if (s.solLine > 0) {
        g.lineThrough([0.5, 0], [2, -1], { color: 'mag', width: 2.2, dashed: true, alpha: s.solLine });
        g.label('solutions x = [0.5; 0] + t·[2; −1]', [-0.9, 0.7], { color: 'mag', size: 13, font: 'ui', alpha: s.solLine, anchor: 'above', dy: -12 });
      }
      const ci = N.col(G, 0), cj = N.col(G, 1);
      if (s.cols) {
        g.vector(ci, { color: 'i', label: 'i', hat: true, width: 3 });
        g.vector(cj, { color: 'j', label: 'j', hat: true, width: 3 });
      }
      if (s.recipe > 0) {
        const c1 = N.col(A, 0), c2 = N.col(A, 1);
        const u1 = seg(s.recipe, 0, 0.5), u2 = seg(s.recipe, 0.5, 1);
        if (u1 > 0) g.vector([0, 0], N.scale(c1, u1), { color: 'i', label: u1 > 0.9 ? '1·col₁' : null, width: 3.4, labelAt: 0.55, dy: 14, dx: 6 });
        if (u2 > 0) g.vector(c1, N.add(c1, N.scale(c2, u2)), { color: 'j', label: u2 > 0.9 ? '1·col₂' : null, width: 3.4, labelAt: 0.5, dx: 30, dy: 12 });
      }
      if (s.rider) g.vector(N.matVec(G, s.rider), { color: 'v', label: 'v', width: 3 });
      if (s.b) {
        g.vector(s.b, { color: 'b', label: 'b', width: 3, alpha: s.trav ? 0.55 : 1 });
        if (s.ringB) g.marker.ring(s.b, { color: 'b', r: 22 });
      }
      if (s.trav) {
        const p = N.matVec(lerpM(I2(), A, s.trav.m), s.trav.x);
        if (s.path) {
          const pts = [];
          for (let k = 0; k <= 30; k++) { const m = 1 - (1 - s.trav.m) * k / 30; pts.push(N.matVec(lerpM(I2(), A, m), s.trav.x)); }
          g.polyline(pts, { color: 'b', width: 1.6, dashed: true, alpha: 0.8 });
        }
        g.point(p, { r: 6.5, color: 'b', stroke: 'white' });
      }
      if (s.x) { g.vector(s.x, { color: 'v', label: 'x', width: 3.2 }); }
      if (s.fan > 0) {
        const T = FT;
        FAN.forEach(p => { g.line(p, T, { color: 'w', dashed: true, width: 1.2, alpha: 0.55 * s.fan }); g.point(p, { r: 5, color: 'w', alpha: s.fan }); });
        g.point(T, { r: 6, color: 'teal', stroke: 'white' });
        g.label('?', T, { color: 'bad', size: 26, dx: -18, dy: -6, bold: true, alpha: s.fan });
        g.hud('purple: six inputs (of infinitely many) that all land on the teal dot', 14, 44, { color: 'w', size: 13.5 });
      }
      if (s.xMark) {
        if (s.xMark.ok) { g.marker.check(s.b, { dx: 32, dy: -12 }); g.label('on the line: solvable', s.b, { color: 'ok', anchor: 'left', dx: 46, dy: -12, size: 13, font: 'ui' }); }
        else { g.marker.cross(s.b, { dx: 32, dy: -12, size: 22 }); g.label('off the line: no solution', s.b, { color: 'bad', anchor: 'left', dx: 48, dy: -12, size: 13, font: 'ui' }); }
      }
      if (s.wrongDim) {
        const p = N.matVec(G, [1, 0]);
        g.point([1, 0], { r: 6, color: 'white', alpha: 0.9, label: 'should return to (1, 0)', labelColor: 'hud', dx: 12, dy: 18 });
        g.point(p, { r: 6, color: 'bad' });
        g.dim([1, 0], p, { label: 'missed by ' + LA.fmt(N.dist(p, [1, 0])), offset: 18, color: 'bad' });
      }
      if (s.okAt) g.marker.check(s.okAt, { dx: 18, dy: -18, size: 22 });
      if (s.claim) {
        const w = g.size.w;
        g.label(s.claim.text, [w / 2, 26], { screen: true, size: 15, color: 'hud', font: 'ui' });
        if (s.claim.cross) g.marker.cross([w / 2 + s.claim.text.length * 3.7 + 22, 26], { screen: true, size: 22 });
      }
      s.hud.forEach((h, i) => g.hud(h.text, 14, 22 + 20 * i, { color: h.color || 'hud', size: 14 }));
    },

    math(s, tex) {
      const num = (x, o) => '<span class="tx-n ' + tex.colorClass(o && o.color) + '">' + (typeof x === 'number' ? LA.fmt(x) : x) + '</span>';
      const P = [], p = s.panel;
      const Aeq = (M, name) => tex.eq(tex.sym(name || 'A'), tex.op('='), tex.mat(M, { colors: 'cols' }));
      const Ainv = () => tex.sym('A⁻¹');
      if (p === 'A') P.push(Aeq(A));
      else if (p === 'inv') { P.push(tex.eq(Ainv(), tex.op('='), tex.mat(AINV, { colors: 'cols' }))); P.push(tex.eq(Ainv(), tex.sym('A'), tex.op('='), tex.mat(I2()), tex.op('='), tex.sym('I'))); }
      else if (p === 'solve') { P.push(tex.eq(tex.sym('x', { color: 'v' }), tex.op('='), Ainv(), tex.sym('b', { color: 'b' }), tex.op('='), tex.mat(AINV, { colors: 'cols' }), tex.vec(B0, { color: 'b' }), tex.op('='), tex.vec(X0, { color: 'v' }))); }
      else if (p === 'fwd') { P.push(tex.eq(tex.sym('A'), tex.sym('x', { color: 'v' }), tex.op('='), num(1, { color: 'v' }), tex.op('·'), tex.vec(N.col(A, 0), { color: 'i' }), tex.op('+'), num(1, { color: 'v' }), tex.op('·'), tex.vec(N.col(A, 1), { color: 'j' }), tex.op('='), tex.vec(B0, { color: 'b' }), num('✓', { color: 'ok' }))); }
      else if (p === 'wrong') { P.push(tex.eq(tex.mat([['1/2', '1/1'], ['1/1', '1/1']]), tex.sym('A'), tex.op('='), tex.mat(N.matMul(WRONG, A)), tex.op('≠'), tex.sym('I'))); P.push(tex.text('entry-by-entry reciprocals do not undo the motion', { color: 'bad' })); }
      else if (p === 'formula') { P.push(tex.eq(Ainv(), tex.op('='), tex.frac('1', 'ad − bc'), tex.mat([['d', '−b'], ['−c', 'a']]), tex.op('='), tex.frac('1', '2·1 − 1·1'), tex.mat([[1, -1], [-1, 2]]), tex.op('='), tex.mat(AINV, { colors: 'cols' }))); }
      else if (p === 'sing') { P.push(Aeq(SING)); P.push(tex.eq(tex.text('det'), tex.sym('A'), tex.op('='), num('1·4 − 2·2'), tex.op('='), num(0, { color: 'bad' }), tex.op('⇒'), tex.text('A⁻¹ does not exist', { color: 'bad' }))); }
      else if (p === 'cases') { P.push(Aeq(SING)); P.push(tex.eq(tex.sym('b', { color: 'b' }), tex.op('∉'), tex.text('line:'), tex.text('no x', { color: 'bad' }))); P.push(tex.eq(tex.sym('b', { color: 'b' }), tex.op('∈'), tex.text('line:'), tex.sym('x', { color: 'v' }), tex.op('='), tex.vec([0.5, 0]), tex.op('+'), tex.sym('t'), tex.op('·'), tex.vec([2, -1], { color: 'null' }))); }
      else if (p === 'both') { P.push(tex.eq(Ainv(), tex.sym('A'), tex.op('='), tex.sym('A'), Ainv(), tex.op('='), tex.sym('I'), tex.op('='), tex.mat(I2()))); P.push(tex.eq(tex.inv(tex.paren(tex.sym('A') + tex.sym('B'))), tex.op('='), tex.inv(tex.sym('B')), Ainv())); }
      else if (p === 'ml') {
        P.push(tex.eq(tex.paren(tex.text('XᵀX')), tex.sym('w', { color: 'w' }), tex.op('='), tex.text('Xᵀy'), tex.op(':'), tex.mat(XTX), tex.sym('w', { color: 'w' }), tex.op('='), tex.vec(XTY), tex.op('→'), tex.sym('w', { color: 'w' }), tex.op('='), tex.vec(WFIT, { color: 'gold' })));
        P.push(tex.eq(tex.norm(tex.text('Δx')), tex.op('/'), tex.norm(tex.text('Δb')), tex.op('≈'), num(1 / Math.sin(TH), { color: 'bad' }), tex.text('(nearly parallel rows)')));
      }
      else if (p === 'play') {
        const d = N.det(s.M), inv = Math.abs(d) > 0.02 ? N.inv(s.M) : null;
        P.push(tex.eq(tex.sym('A'), tex.op('='), tex.mat(s.M, { colors: 'cols' }), tex.text('det'), tex.op('='), num(d, { color: inv ? 'gold' : 'bad' })));
        if (inv) { P.push(tex.eq(Ainv(), tex.op('='), tex.mat(inv, { colors: 'cols' }))); P.push(tex.eq(tex.sym('x', { color: 'v' }), tex.op('='), Ainv(), tex.sym('b', { color: 'b' }), tex.op('='), tex.vec(N.matVec(inv, s.bb), { color: 'v' }))); }
        else P.push(tex.eq(Ainv(), tex.op(':'), tex.text('none (det = 0)', { color: 'bad' })));
      }
      else if (p === 'chk1') { P.push(tex.eq(tex.text('typed'), tex.op('·'), tex.sym('A'), tex.op('='), tex.mat(s.Minv, { colors: 'cols' }), tex.mat(A, { colors: 'cols' }), tex.op('='), tex.mat(N.matMul(s.Minv, A)))); }
      else if (p === 'chk2') { const ax = N.matVec(A, s.xIn); P.push(tex.eq(tex.sym('A'), tex.sym('x', { color: 'v' }), tex.op('='), tex.mat(A, { colors: 'cols' }), tex.vec(s.xIn, { color: 'v' }), tex.op('='), tex.vec(ax), tex.text('target'), tex.vec(s.bb, { color: 'b' }))); }
      else if (p === 'chk3') { P.push(tex.eq(tex.sym('A'), tex.op('='), tex.mat(s.M, { colors: 'cols' }), tex.text('det'), tex.op('='), num(N.det(s.M), { color: Math.abs(N.det(s.M)) < 2 * LA.TOL.det ? 'bad' : 'gold' }))); }
      return tex.block.apply(null, P);
    },

    steps: [
      { title: 'Play the movie backwards', duration: 4,
        narration: 'Here is A = [[2, 1],[1, 1]] moving the grid, with the yellow <span class="c-v">v</span> riding along. Can the motion be undone? Play the movie backwards: every point returns to where it started, and v comes home. If a backwards movie exists, the transformation can be reversed.',
        setup(s) { s.scene = 'grid'; s.cam = cam2(1.2, 1.0, 3.4); s.G = I2(); s.cols = true; s.rider = X0.slice(); s.b = null; s.trav = null; s.x = null; s.recipe = 0; s.ringB = false; s.okAt = null; s.claim = null; s.wrongDim = false; s.colLine = null; s.fan = 0; s.solLine = 0; s.xMark = null; s.hud = []; s.panel = 'A'; },
        at(s, t) { const m = t < 0.5 ? seg(t, 0, 0.45) : 1 - seg(t, 0.55, 1); s.G = lerpM(I2(), A, m); s.hud = [{ text: t < 0.5 ? '▶ forward: A' : '◀ backward', color: t < 0.5 ? 'hud' : 'gold' }]; } },

      { title: 'The reverse movie has a matrix', duration: 4,
        narration: 'The reverse movie is itself linear — lines stay straight, the origin stays — so it has a matrix: the <b>inverse</b>, A<sup>−1</sup> = [[1, −1],[−1, 2]]. Do A, then A<sup>−1</sup>, and every vector is back where it began: A<sup>−1</sup>A = I, the <b>identity</b>, the matrix that moves nothing.',
        setup(s) { s.G = I2(); s.rider = X0.slice(); s.panel = 'inv'; },
        at(s, t) {
          const u1 = seg(t, 0, 0.4), u2 = seg(t, 0.5, 0.9);
          s.G = N.matMul(lerpM(I2(), AINV, u2), lerpM(I2(), A, u1));
          s.okAt = u2 >= 1 ? [0, 0] : null;
          s.hud = t < 0.45 ? [{ text: 'first A' }] : [{ text: 'then A⁻¹ = [[1, −1],[−1, 2]]', color: 'gold' }].concat(u2 >= 1 ? [{ text: 'back on the ghost grid: A⁻¹A = I', color: 'ok' }] : []);
        } },

      { title: 'Solving Ax = b: carry b home', duration: 4,
        narration: '<b>Solving</b> A' + xS + ' = ' + bS + ' asks: which input x does A send to the orange b = [3; 2]? Run the movie backwards with b riding on it: it is carried to the yellow x = [1; 1]. In symbols, x = A<sup>−1</sup>b. The inverse answers “where did this come from?”',
        setup(s) { s.rider = null; s.okAt = null; s.G = LA.clone(A); s.b = B0.slice(); s.trav = { x: X0.slice(), m: 1 }; s.path = true; s.x = null; s.panel = 'solve'; s.hud = []; },
        at(s, t) {
          const m = 1 - seg(t, 0.1, 0.8);
          s.G = lerpM(I2(), A, m); s.trav = { x: X0.slice(), m };
          s.x = t > 0.8 ? X0.slice() : null;
          s.hud = t > 0.8 ? [{ text: 'b came from x = [1; 1]', color: 'v' }] : [{ text: 'playing backwards…', color: 'b' }];
        } },

      { title: 'Check it forward', duration: 3,
        narration: 'Confirm with the recipe: A' + xS + ' = 1·[2; 1] + 1·[1; 1] = [3; 2] = ' + bS + '. The columns of A, one copy each, laid tip to tail, land exactly on the orange b. Solving a system is finding the recipe of columns that reaches b — the span question from Part 1, answered.',
        setup(s) { s.trav = null; s.path = false; s.x = null; s.G = LA.clone(A); s.cols = false; s.b = B0.slice(); s.recipe = 0; s.ringB = false; s.panel = 'fwd'; s.hud = []; },
        at(s, t) { s.recipe = seg(t, 0.05, 0.7, 'linear'); s.ringB = t > 0.72; s.hud = t > 0.72 ? [{ text: 'A x = b ✓', color: 'ok' }] : []; } },

      { title: 'A⁻¹ is not 1/A entrywise', duration: 4, misconception: true,
        narration: 'Belief: “A<sup>−1</sup> means one over each entry.” Try [[1/2, 1/1],[1/1, 1/1]] after A: the grid does not return to the ghost — the point that started at (1, 0) lands far away. The inverse undoes the whole motion at once; it is not an entry-by-entry recipe.',
        setup(s) { s.cam = cam2(1.4, 1.3, 3.4); s.recipe = 0; s.ringB = false; s.b = null; s.cols = true; s.G = LA.clone(A); s.panel = 'wrong'; s.hud = []; },
        at(s, t) {
          s.G = N.matMul(lerpM(I2(), WRONG, seg(t, 0.05, 0.6)), A);
          s.wrongDim = t > 0.62;
          s.claim = { text: 'belief: “A⁻¹ = 1 / each entry”', cross: t > 0.75 };
        } },

      { title: 'The 2×2 formula', duration: 5,
        narration: 'For a 2×2 there is a short formula: swap the diagonal, negate the off-diagonal, divide by the determinant. For A = [[2, 1],[1, 1]], det = 1, so A<sup>−1</sup> = [[1, −1],[−1, 2]] — the matrix from before. A matrix is <b>invertible</b> exactly when that division is allowed: det ≠ 0.',
        setup(s) { s.scene = 'formula'; s.cam = cam2(0, 0.3, 2.6); s.claim = null; s.wrongDim = false; s.ft = 0; s.panel = 'formula'; s.hud = []; },
        at(s, t) { s.ft = t; } },

      { title: 'A flattening cannot be undone', duration: 5,
        narration: 'Now A = [[1, 2],[2, 4]], with det = 0. The plane is flattened onto the teal line. Try to play it backwards: a point on the line came from a whole line of inputs — the purple candidates — and no map can choose among them. No reverse movie, no inverse; the formula would divide by zero. A is <b>singular</b>.',
        setup(s) { s.scene = 'grid'; s.cam = cam2(0.6, 0.9, 3.5); s.G = I2(); s.cols = true; s.colLine = null; s.fan = 0; s.b = null; s.panel = 'sing'; s.hud = []; },
        at(s, t) {
          s.G = lerpM(I2(), SING, seg(t, 0, 0.4));
          s.colLine = t > 0.35 ? [1, 2] : null;
          s.fan = seg(t, 0.5, 0.85);
          s.hud = t > 0.6 ? [{ text: 'A⁻¹ = (1 / det A)·[…] = (1 / 0)·[…]   ÷ 0', color: 'bad' }] : [];
        } },

      { title: 'Then Ax = b has none or infinitely many', duration: 4,
        narration: 'With the singular A, where the orange ' + bS + ' sits decides everything. Off the teal line: no input reaches it, no solution. On the line: a whole line of inputs reaches it — infinitely many solutions, drawn as the dashed magenta line. The next lesson names these: column space and null space.',
        setup(s) { s.G = LA.clone(SING); s.colLine = [1, 2]; s.fan = 0; s.cols = true; s.b = [0.5, 2.5]; s.solLine = 0; s.xMark = null; s.panel = 'cases'; s.hud = []; },
        at(s, t) {
          s.b = lerpV([0.5, 2.5], FT, seg(t, 0.45, 0.65));
          s.xMark = t < 0.42 ? { ok: false } : (t > 0.66 ? { ok: true } : null);
          s.solLine = seg(t, 0.68, 0.9);
        } },

      { title: 'The identity, and inverses on both sides', duration: 4,
        narration: 'The identity I = [[1, 0],[0, 1]] leaves every vector alone: AI = IA = A. An inverse works from either side: A<sup>−1</sup>A = AA<sup>−1</sup> = I — undo-then-do is as harmless as do-then-undo. And (AB)<sup>−1</sup> = B<sup>−1</sup>A<sup>−1</sup>: to undo two steps, undo the last one first.',
        setup(s) { s.cam = cam2(1.2, 1.0, 3.4); s.colLine = null; s.solLine = 0; s.xMark = null; s.b = null; s.G = I2(); s.rider = X0.slice(); s.panel = 'both'; },
        at(s, t) {
          let G = I2(), txt = 'I: nothing moves', col = 'hud', ok = null;
          if (t >= 0.2 && t < 0.6) { const u1 = seg(t, 0.2, 0.37), u2 = seg(t, 0.4, 0.57); G = N.matMul(lerpM(I2(), AINV, u2), lerpM(I2(), A, u1)); txt = 'A, then A⁻¹'; col = 'gold'; if (u2 >= 1) ok = [0, 0]; }
          else if (t >= 0.6) { const v1 = seg(t, 0.6, 0.77), v2 = seg(t, 0.8, 0.97); G = N.matMul(lerpM(I2(), A, v2), lerpM(I2(), AINV, v1)); txt = 'A⁻¹, then A'; col = 'gold'; if (v2 >= 1) ok = [0, 0]; }
          s.G = G; s.okAt = ok; s.hud = [{ text: txt, color: col }].concat(ok ? [{ text: 'back on the ghost grid', color: 'ok' }] : []);
        } },

      { title: 'Solve, don’t invert', duration: 5, mlTie: true,
        narration: 'Linear regression’s weights satisfy (XᵀX)w = Xᵀy. Code solves this system directly rather than forming the inverse — cheaper and more accurate. And when two equations are nearly parallel lines, a 2% nudge of b throws x far away: an almost-singular system amplifies noise. Part 4 measures that with the condition number.',
        setup(s) { s.scene = 'ml'; s.rider = null; s.okAt = null; s.hud = []; s.mlT = 0; s.panel = 'ml'; },
        at(s, t) { s.mlT = t; } },

      { title: 'Recap', duration: 1.5, recap: true,
        narration: 'The inverse is the movie played backwards: A⁻¹A = I. It solves Ax = b by carrying b home to x = A⁻¹b. For 2×2: swap the diagonal, negate the off-diagonal, divide by det. det = 0 means flattened, so no inverse and either no solution or infinitely many. In practice, solve rather than invert.',
        cards: ['<b>A⁻¹ = the reverse movie.</b> Do A, then A⁻¹, and the grid lands back on the ghost: A⁻¹A = AA⁻¹ = I.', '<b>x = A⁻¹b: carry b home.</b> For 2×2, swap the diagonal, negate the off-diagonal, divide by det. In code, solve instead of inverting.', '<b>det = 0: no inverse.</b> The plane is flattened, so Ax = b has no solution (b off the line) or infinitely many (b on it).'],
        setup(s) { s.scene = 'recap'; s.panel = 'none'; },
        at(s, t) { } },
    ],

    playground: {
      intro: '<p>Drag the tips of <span class="c-i">î</span> and <span class="c-j">ĵ</span> and the orange <span class="c-b">b</span>. The yellow <span class="c-v">x</span> = A⁻¹b updates live. Slide the movie backward to see the inverse map; make det = 0 and watch it disappear.</p>',
      setup(s) { s.scene = 'grid'; s.pg = true; s.chk = null; s.cam = cam2(1.2, 1.0, 3.6); s.M = LA.clone(A); s.bb = B0.slice(); s.morph = 1; s.showInv = false; s.panel = 'play'; s.hud = []; },
      controls: [
        { type: 'matrix', bind: 'M', label: 'A', colors: 'cols', step: 0.1 },
        { type: 'matrix', bind: 'bb', label: 'b (orange)', colors: 'b', step: 0.1 },
        { type: 'slider', bind: 'morph', min: -1, max: 1, step: 0.01, label: 'Play forward / backward', color: 'gold', fmt: x => (x >= 0 ? 'A × ' : 'A⁻¹ × ') + LA.fmt(Math.abs(x)) },
        { type: 'toggle', bind: 'showInv', label: 'Show the inverse map (gold grid)' },
        { type: 'preset', label: 'Presets', items: [
          { label: 'Invertible', values: { M: A, bb: B0, morph: 1 } },
          { label: 'Singular', values: { M: SING, bb: [1, 2], morph: 1 } },
          { label: 'Rotation (inverse = reverse turn)', apply(s) { s.M = N.rot2(0.6); s.bb = [2, 1]; s.morph = 1; } },
        ] },
      ],
    },

    checks: [
      { kind: 'entry',
        prompt: 'Type A⁻¹ for A = [[2, 1],[1, 1]]. The grid shows A followed by your matrix; it lands back on the ghost (✓) when you are right.',
        bind: 'Minv', answer: [[1, -1], [-1, 2]], tol: LA.TOL.entry, label: 'A⁻¹', colors: 'cols',
        hint: 'Swap the diagonal, negate the off-diagonal; det is 1, so there is nothing to divide.',
        setup(s) { s.scene = 'grid'; s.pg = false; s.chk = 'inv'; s.cam = cam2(1.2, 1.0, 3.6); s.Minv = I2(); s.panel = 'chk1'; } },
      { kind: 'manipulate',
        prompt: 'Drag the yellow ' + xS + ' so that A' + xS + ' (gray) lands on the orange ' + bS + '. Here A = [[2, 1],[1, 1]].',
        setup(s) { s.scene = 'grid'; s.pg = false; s.chk = 'solve'; s.cam = cam2(0.8, 1.2, 3.4); s.bb = [0, 1]; s.xIn = [1, 1]; s.panel = 'chk2'; },
        judge(s) { const e = N.dist(N.matVec(A, s.xIn), s.bb); return { ok: e < 2 * LA.TOL.pos, progress: Math.max(0, 1 - e / 4) }; },
        hint: 'Move x against the grid’s motion: where would b come from if you played the movie backwards?',
        solution(s) { s.xIn = [-1, 2]; } },
      { kind: 'manipulate',
        prompt: 'Drag ' + jhat + ' so that A has no inverse.',
        setup(s) { s.scene = 'grid'; s.pg = false; s.chk = 'sing'; s.cam = cam2(1.0, 0.8, 3.0); s.M = LA.clone(A); s.panel = 'chk3'; },
        judge(s) { const d = Math.abs(N.det(s.M)), n = N.norm(N.col(s.M, 1)); return { ok: d < 2 * LA.TOL.det && n > 0.3, progress: Math.max(0, 1 - d), hint: n <= 0.3 ? 'Keep ' + jhat + ' longer than zero; lay it on the line through ' + ihat + '’s landing spot.' : undefined }; },
        hint: 'Put ' + jhat + '’s landing spot on the line through ' + ihat + '’s landing spot.',
        solution(s) { s.M = [[2, 1], [1, 0.5]]; } },
    ],
  });

  /* ---------- check scenes ---------- */
  function renderCheck(g, s) {
    g.ghostGrid();
    if (s.chk === 'inv') {
      const G = N.matMul(s.Minv, A);
      g.grid(G);
      g.vector(N.col(G, 0), { color: 'i', label: 'i', hat: true, width: 3 });
      g.vector(N.col(G, 1), { color: 'j', label: 'j', hat: true, width: 3 });
      const ok = LA.judge.matrix(s.Minv, AINV, LA.TOL.entry);
      if (ok === true || (ok && ok.ok)) { g.marker.check([0, 0], { dx: 18, dy: -18, size: 22 }); g.hud('back on the ghost grid: your matrix undoes A', 14, 22, { color: 'ok', size: 14 }); }
      else g.hud('the grid shows (your matrix) · A — make it land on the ghost', 14, 22, { size: 14 });
      return;
    }
    if (s.chk === 'solve') {
      g.grid(A, { alpha: 0.35 });
      const ax = N.matVec(A, s.xIn);
      g.vector(s.bb, { color: 'b', label: 'b', width: 3 });
      g.vector(ax, { color: 'muted', label: 'Ax', width: 2.6, dashed: true });
      g.vector(s.xIn, { color: 'v', label: 'x', width: 3.2 });
      g.handle('x', s.xIn, (p, st) => { st.xIn = p.slice(); }, { color: 'v' });
      g.hud('yellow x is the input · gray Ax is where A sends it', 14, 22, { size: 13.5 });
      return;
    }
    // 'sing'
    const M = s.M, d = N.det(M);
    g.grid(M);
    g.parallelogram(N.col(M, 0), N.col(M, 1), { orient: true, alpha: 0.22 });
    g.vector(N.col(M, 0), { color: 'i', label: 'i', hat: true, width: 3 });
    g.vector(N.col(M, 1), { color: 'j', label: 'j', hat: true, width: 3 });
    g.hud('det A = ' + LA.fmt(d) + (Math.abs(d) < 2 * LA.TOL.det ? '  ≈ 0 — singular: no inverse' : ''), 14, 22, { size: 14, color: Math.abs(d) < 2 * LA.TOL.det ? 'bad' : 'gold' });
    g.handle('j', N.col(M, 1), (p, st) => { st.M[0][1] = p[0]; st.M[1][1] = p[1]; }, { color: 'j' });
  }

  /* ---------- playground ---------- */
  function renderPlay(g, s) {
    const { G, inv } = playG(s);
    if (s.showGhost) g.ghostGrid();
    g.grid(G);
    if (s.showInv && inv) g.grid(inv, { color: 'gold', alpha: 0.28, axis: false });
    const M = s.M, b = s.bb;
    g.vector(N.col(G, 0), { color: 'i', label: 'i', hat: true, width: 3 });
    g.vector(N.col(G, 1), { color: 'j', label: 'j', hat: true, width: 3 });
    g.vector(b, { color: 'b', label: 'b', width: 3 });
    if (inv) {
      g.vector(N.matVec(inv, b), { color: 'v', label: 'x', width: 3 });
    } else {
      const cs = N.colSpace(M);
      if (cs.length === 1) {
        g.lineThrough([0, 0], cs[0], { color: 'teal', width: 2.2, alpha: 0.85 });
        const x0 = N.matVec(N.pinv(M), b), resid = N.dist(N.matVec(M, x0), b);
        const ns = N.nullSpace(M);
        if (resid < LA.TOL.pos && ns.length) { g.lineThrough(x0, ns[0], { color: 'mag', dashed: true, width: 2 }); g.hud('b is on the teal line: infinitely many solutions (dashed magenta)', 14, 22, { color: 'mag', size: 13.5 }); }
        else g.hud('b is off the teal line: no solution', 14, 22, { color: 'bad', size: 13.5 });
      } else g.hud('A = 0 sends everything to the origin', 14, 22, { color: 'bad', size: 13.5 });
      if (s.morph < 0) g.hud('det = 0: there is no backwards movie', 14, 42, { color: 'bad', size: 13.5 });
    }
    g.handle('i', N.col(M, 0), (p, st) => { st.M[0][0] = p[0]; st.M[1][0] = p[1]; }, { color: 'i' });
    g.handle('j', N.col(M, 1), (p, st) => { st.M[0][1] = p[0]; st.M[1][1] = p[1]; }, { color: 'j' });
    g.handle('b', b, (p, st) => { st.bb = p.slice(); }, { color: 'b' });
  }

  /* ---------- the 2×2 formula, animated on the canvas ---------- */
  function renderFormula(g, s) {
    g.ghostGrid({ alpha: 0.5 });
    const t = s.ft;
    const u1 = seg(t, 0.05, 0.4), u2 = seg(t, 0.42, 0.6), u3 = seg(t, 0.62, 0.9);
    const ox = 0.5; // matrix centre x
    const P = { a: [ox - 0.55, 0.55], b: [ox + 0.55, 0.55], c: [ox - 0.55, -0.45], d: [ox + 0.55, -0.45] };
    const br = (x0, dir) => g.polyline([[x0 + 0.22 * dir, 1.15], [x0, 1.15], [x0, -1.05], [x0 + 0.22 * dir, -1.05]], { color: 'muted', width: 2 });
    br(ox - 1.15, 1); br(ox + 1.15, -1);
    const pa = curvePt(P.a, P.d, u1, 0.45), pd = curvePt(P.d, P.a, u1, 0.45);
    const flip = u2 > 0.5, flash = u2 > 0 && u2 < 1;
    const big = { size: 34, font: 'math', bg: false };
    g.label('2', pa, Object.assign({ color: u1 > 0 && u1 < 1 ? 'gold' : (u1 >= 1 ? 'j' : 'i') }, big));
    g.label('1', pd, Object.assign({ color: u1 > 0 && u1 < 1 ? 'gold' : (u1 >= 1 ? 'i' : 'j') }, big));
    g.label(flip ? '−1' : '1', P.b, Object.assign({ color: flash ? 'gold' : 'j' }, big));
    g.label(flip ? '−1' : '1', P.c, Object.assign({ color: flash ? 'gold' : 'i' }, big));
    // the left-hand side and the 1/det factor
    g.label(t > 0.9 ? 'A⁻¹ =' : (u1 > 0 ? '' : 'A ='), [ox - 2.2, 0.05], { size: 30, font: 'math', color: 'text', bg: false });
    if (u3 > 0) {
      g.label('1', [ox - 1.6, 0.35], { size: 26, font: 'math', color: 'gold', alpha: u3, bg: false });
      g.line([ox - 1.85, 0.05], [ox - 1.35, 0.05], { color: 'gold', width: 2, alpha: u3 });
      g.label('det', [ox - 1.6, -0.27], { size: 20, font: 'math', color: 'gold', alpha: u3, bg: false });
      g.label('det A = 2·1 − 1·1 = 1', [ox, -1.55], { size: 16, font: 'ui', color: 'gold', alpha: u3 });
    }
    // the three-part recipe across the top, each phrase lit in turn
    const steps = [['1. swap the diagonal', u1 > 0], ['2. negate the off-diagonal', u2 > 0], ['3. divide by det', u3 > 0]];
    steps.forEach((st, i) => g.hud(st[0], 14 + i * 200, 24, { size: 15, color: st[1] ? 'gold' : 'muted' }));
    if (t > 0.9) g.label('= [[1, −1],[−1, 2]]: the matrix from step 2', [ox, 1.65], { size: 15, font: 'ui', color: 'ok' });
  }

  /* ---------- ML tie: normal equations, and an ill-conditioned pair of lines ---------- */
  function renderML(g, s) {
    const t = s.mlT;
    g.viewport({ x: 0.02, y: 0.02, w: 0.96, h: 0.47 }, { view: '2d', cam: cam2(1.0, 1.5, 1.55) }, (g2, r) => {
      g2.ghostGrid({ range: 6 });
      g2.axes({ ticks: true, labels: false });
      const pts = DX.map((x, i) => [x, DY[i]]);
      const lineA = seg(t, 0.05, 0.3);
      if (lineA > 0) g2.line([-1, WFIT[0] - WFIT[1]], [3.2, WFIT[0] + 3.2 * WFIT[1]], { color: 'gold', width: 2.6, alpha: lineA });
      g2.points(pts, { r: 5.5, color: 'data' });
      g2.hud('fit a line to 3 synthetic points', r.x + 10, r.y + 20, { size: 13 });
      g2.hud('(XᵀX) w = Xᵀy', r.x + 10, r.y + 42, { size: 15, color: 'gold' });
      g2.hud('[[3, 3],[3, 5]] w = [' + LA.fmt(XTY[0]) + ', ' + LA.fmt(XTY[1]) + ']', r.x + 10, r.y + 62, { size: 13 });
      g2.hud('solve → w = [' + LA.fmt(WFIT[0]) + ', ' + LA.fmt(WFIT[1]) + ']  (no inverse formed)', r.x + 10, r.y + 82, { size: 13, color: 'gold' });
      g2.hud('Part 3: least squares', r.x + 10, r.y + r.h - 14, { size: 12.5, color: 'muted' });
    });
    g.viewport({ x: 0.02, y: 0.51, w: 0.96, h: 0.47 }, { view: '2d', cam: cam2(-0.8, 2.0, 0.9) }, (g2, r) => {
      g2.ghostGrid({ range: 6 });
      const u = seg(t, 0.4, 0.75);
      const d = NUDGE * u;
      const n2 = [-Math.sin(TH), Math.cos(TH)];
      const p0 = [0, 2];
      const x = [-d / Math.sin(TH), 2];
      // row 1: y = 2 (fixed); row 2: n2·p = n2·p0 + d (nudged)
      g2.lineThrough(p0, [1, 0], { color: 'teal', width: 2.2 });
      g2.lineThrough(N.add(p0, N.scale(n2, d)), [Math.cos(TH), Math.sin(TH)], { color: 'b', width: 2.2 });
      g2.label('row 1', [0.65, 2], { color: 'teal', anchor: 'right', dy: 14, size: 12.5, font: 'ui' });
      g2.label('row 2 (b₂ nudged)', [0.65, 2 + Math.tan(TH) * 0.65 + d], { color: 'b', anchor: 'right', dy: -14, size: 12.5, font: 'ui' });
      if (u > 0) { g2.point(p0, { r: 5, color: 'white', alpha: 0.5 }); g2.line(p0, x, { color: 'v', dashed: true, width: 1.2, alpha: 0.7 }); }
      g2.point(x, { r: 6.5, color: 'v', label: 'x', dy: -14 });
      if (u > 0.95) g2.dim(p0, x, { label: 'Δx = ' + LA.fmt(N.dist(p0, x)), offset: -30, color: 'v' });
      g2.hud('two equations, 2° apart', r.x + 10, r.y + 20, { size: 13 });
      g2.hud('nudge b by ' + LA.fmt(d) + ' (' + Math.round(100 * d / Math.hypot(2, 2 * Math.cos(TH))) + '% of ‖b‖)', r.x + 10, r.y + 42, { size: 13, color: 'b' });
      if (u > 0.95) g2.hud('‖Δx‖ / ‖Δb‖ ≈ ' + LA.fmt(1 / Math.sin(TH)) + ': noise amplified', r.x + 10, r.y + 62, { size: 13.5, color: 'bad' });
      g2.hud('Part 4: conditioning', r.x + 10, r.y + r.h - 14, { size: 12.5, color: 'muted' });
    });
  }

  /* ---------- recap ---------- */
  function renderRecap(g) {
    const card = (i, vopts, fn) => g.viewport({ x: 0.02, y: 0.02 + i * 0.327, w: 0.96, h: 0.31 }, vopts, fn);
    card(0, { view: '2d', cam: cam2(1.2, 0.9, 1.9) }, (g2, r) => {
      g2.ghostGrid({ range: 6 }); g2.grid(lerpM(I2(), A, 0.35), { range: 6, alpha: 0.45 });
      g2.vector(X0, { color: 'v', label: 'v', width: 2.6 });
      g2.vector(N.matVec(A, X0), { color: 'v', alpha: 0.35, width: 2, dashed: true });
      g2.line(N.matVec(A, X0), X0, { color: 'gold', dashed: true, width: 1.4 });
      g2.marker.check([0, 0], { dx: 14, dy: -12 });
      g2.hud('A⁻¹A = I: the grid returns to the ghost', r.x + 10, r.y + 16, { size: 13, color: 'hud' });
    });
    card(1, { view: '2d', cam: cam2(1.8, 1.2, 1.9) }, (g2, r) => {
      g2.ghostGrid({ range: 6 });
      const pts = []; for (let k = 0; k <= 20; k++) pts.push(N.matVec(lerpM(I2(), A, 1 - k / 20), X0));
      g2.polyline(pts, { color: 'b', dashed: true, width: 1.5 });
      g2.vector(B0, { color: 'b', label: 'b', width: 2.6 });
      g2.vector(X0, { color: 'v', label: 'x', width: 2.6 });
      g2.hud('x = A⁻¹b: carry b home', r.x + 10, r.y + 16, { size: 13, color: 'hud' });
    });
    card(2, { view: '2d', cam: cam2(0.8, 0.9, 1.9) }, (g2, r) => {
      g2.ghostGrid({ range: 6 }); g2.grid(SING, { range: 6, alpha: 0.5 });
      g2.lineThrough([0, 0], [1, 2], { color: 'teal', width: 2.2 });
      g2.lineThrough([0.5, 0], [2, -1], { color: 'mag', width: 1.8, dashed: true });
      FAN.slice(0, 4).forEach(p => g2.point(p, { r: 3.5, color: 'w' }));
      g2.point(FT, { r: 5, color: 'teal' });
      g2.label('÷ 0', [2.6, 1.6], { color: 'bad', size: 18, font: 'math' });
      g2.hud('det = 0: no inverse; none or infinitely many solutions', r.x + 10, r.y + 16, { size: 13, color: 'hud' });
    });
  }
})();
