/* p1-05 — Length, distance, unit vectors, L1/L2/L∞: Pythagoras on the components, and three rulers for one arrow. */
(function () {
  'use strict';
  const N = LA.num, seg = LA.seg, lerpV = LA.lerpV;
  /* PROBE: the storyboard's [0.4, 0.6] has the same 3 neighbours under L2 and L1 for knn30, so nothing changes on
     screen. [0.3, 0.85] gives exactly the storyboard's story: L2 → 2 blue / 1 rose, L1 → 1 blue / 2 rose. */
  const V = [3, 4], V3 = [2, 3, 6], P = [1, 1], Q = [4, 5], PROBE = [0.3, 0.85];
  const vS = '<span class="c-v">v</span>';
  const CAM_PYTH = { center: [1.3, 2.0], halfH: 6.6 };
  const CAM_MAIN = { center: [2.0, 2.0], halfH: 4.4 };
  const CAM_UNIT = { center: [1.4, 1.8], halfH: 3.2 };
  const CAM_BALL = { center: [0, 0.15], halfH: 2.1 };
  const CAM_STRIP = { center: [0, -0.3], halfH: 3.6 };
  const CAM_KNN = { center: [0.3, 0.5], halfH: 3.3 };
  const CAM_PLAY = { center: [2.4, 0.6], halfH: 5.8 };
  const CAM3 = { target: [2.0, 4.2, 2.4], dist: 25, yaw: 0.55, pitch: 0.3 };
  /* Numbers for the math panel. The engine updates spans whose class starts with "tx-v" in place, but matrix cells
     also carry that class, so tex.num next to tex.vec writes into the wrong cell; a plain tabular cell avoids it. */
  const nm = (x, color, digits) => '<span class="tx-cell ' + LA.tex.colorClass(color || '') + '">' + LA.fmt(x, digits) + '</span>';
  const sub = (html, k) => html + '<sub>' + k + '</sub>';

  function makeKnn() {
    const rand = LA.rng(5); const pts = [], labels = [];
    for (let i = 0; i < 15; i++) { pts.push([1.5 + LA.rng.gauss(rand) * 0.7, 1.5 + LA.rng.gauss(rand) * 0.7]); labels.push(0); }
    for (let i = 0; i < 15; i++) { pts.push([-1 + LA.rng.gauss(rand) * 0.7, -0.5 + LA.rng.gauss(rand) * 0.7]); labels.push(1); }
    return { pts, labels };
  }
  function makeStrip() { const rand = LA.rng(8); const v = []; for (let i = 0; i < 8; i++) v.push(LA.rng.int(rand, -3, 3)); v[2] = 2; return v; }
  const normOf = (v, type) => type === 'L1' ? N.norm1(v) : type === 'Linf' ? N.normInf(v) : N.norm(v);
  const NORM_NAME = { L1: 'L1', L2: 'L2', Linf: 'L∞' }, NORM_SUB = { L1: '1', L2: '2', Linf: '∞' };
  /** radius of the unit ball in direction th; m ∈ [0,3]: 0 circle → 1 diamond → 2 square → 3 circle */
  function ballRadius(th, m) {
    const c = Math.abs(Math.cos(th)), s = Math.abs(Math.sin(th));
    const r2 = 1, r1 = 1 / (c + s), rI = 1 / Math.max(c, s);
    m = ((m % 3) + 3) % 3;
    if (m <= 1) return LA.lerp(r2, r1, m);
    if (m <= 2) return LA.lerp(r1, rI, m - 1);
    return LA.lerp(rI, r2, m - 2);
  }
  function ballPoly(m, scale, center) {
    const pts = []; const c = center || [0, 0], k = scale || 1;
    for (let i = 0; i < 64; i++) { const th = i / 64 * Math.PI * 2, r = ballRadius(th, m) * k; pts.push([c[0] + r * Math.cos(th), c[1] + r * Math.sin(th)]); }
    return pts;
  }
  const typeToM = { L2: 0, L1: 1, Linf: 2 };
  /** k nearest neighbours of p under a norm: [{i, d}] sorted */
  function neighbours(pts, p, k, type) {
    const d = pts.map((q, i) => ({ i, d: normOf(N.sub(q, p), type) })).sort((a, b) => a.d - b.d);
    return d.slice(0, k);
  }
  const square = (a, b, c, d) => [a, b, c, d];

  LA.registerLesson({
    id: 'p1-05-norms-distance',
    part: 1,
    title: 'Length, distance, unit vectors, L1/L2/L∞',
    subtitle: 'Square, add, root. Then three rulers: diamond, circle, square.',
    minutes: 9,
    prereqs: ['p1-02-add-scale'],
    tags: ['core'],
    badge: { drawnIn: '2D and 3D', realSize: 'a loss is a norm of a 10 000-component error vector; k-NN measures distances between 784-pixel images', transfers: '‖v‖ = √(Σvᵢ²) is the same sum with more terms; the unit “ball” of each norm keeps its corners (L1) or flat sides (L∞) in every dimension' },
    cantShow: ['the L1 ball in ℝ⁷⁸⁴ (it still has corners on the axes; we show the 2D diamond and say so)'],
    terms: [
      { term: 'length', def: 'How long the arrow is (its norm, ‖v‖): the square root of the sum of squared components.' },
      { term: 'distance', def: 'The length of the difference of two vectors.' },
      { term: 'unit vector', def: 'A vector of length 1.' },
      { term: 'normalize', def: 'Divide a vector by its length to get a unit vector in the same direction.' },
      { term: 'L1 norm', def: 'The sum of absolute components, the taxicab distance.' },
      { term: 'L∞ norm', def: 'The largest absolute component.' },
      { term: 'unit ball', def: 'All vectors of norm at most 1 under a chosen norm.' },
    ],
    atlas: ['knn', 'regularization', 'embeddings'],
    thumbnail: 7,

    state() {
      const knn = makeKnn();
      return {
        view: '2d', cam: Object.assign(LA.cam.defaults2d(), CAM_PYTH),
        v: V.slice(), v3: V3.slice(), p: P.slice(), q: Q.slice(),
        showV: true, comps: 'labels', sqGrow: 0, squares: 0, showDim: false,
        bar: 0, cross: false, overshoot: false,
        showPQ: false, shift: 0,
        unitTween: 0, showCircle: false, ghostV: false,
        taxi: 0, gridBright: 0, maxPulse: 0, showTable: false,
        showBall: false, ballM: 0, normType: 'L2', unitArrow: false,
        strip: makeStrip(), stripT: 0,
        knnPts: knn.pts, knnLabels: knn.labels, showKnn: false, knnT: 0, k: 3, probe: PROBE.slice(),
        panel: 'pyth', drag: false,
      };
    },

    render(g, s, ctx) {
      if (s.view === '3d') { render3d(g, s); return; }
      if (s.showKnn) { renderKnn(g, s, ctx); return; }
      if (s.stripT > 0) { renderStrip(g, s); return; }
      g.ghostGrid();
      if (s.gridBright > 0) g.grid(null, { range: 10, alpha: 0.5 * s.gridBright, width: 1 });
      g.axes({ labels: true });
      const v = s.v, nv = N.norm(v);
      // Pythagoras squares on the two legs, sliding into the square on the hypotenuse
      if (s.sqGrow > 0) {
        const a = v[0], b = v[1], gw = s.sqGrow, sl = s.squares;
        const bigC = [(a - b) / 2, (a + b) / 2];            // centre of the square on the hypotenuse (outer side)
        const shiftTo = (pts, from) => pts.map(p => [p[0] + (bigC[0] - from[0]) * sl, p[1] + (bigC[1] - from[1]) * sl]);
        const fade = 1 - 0.9 * sl;
        const gC = [a / 2, -a / 2], rC = [a + b / 2, b / 2];
        const gS = square([0, 0], [a, 0], [a, -a * gw], [0, -a * gw]), rS = square([a, 0], [a, b], [a + b * gw, b], [a + b * gw, 0]);
        if (sl < 0.97) {   // the leg squares slide into the hypotenuse square and dissolve there (outlines only before sliding)
          g.polygon(shiftTo(gS, gC), { fill: 'i', stroke: sl > 0 ? null : 'i', alpha: 0.28 * fade });
          g.polygon(shiftTo(rS, rC), { fill: 'j', stroke: sl > 0 ? null : 'j', alpha: 0.28 * fade });
        }
        if (gw >= 1) { g.label('area ' + LA.fmt(a * a), shiftTo([gC], gC)[0], { color: 'i', size: 14, alpha: fade }); g.label('area ' + LA.fmt(b * b), shiftTo([rC], rC)[0], { color: 'j', size: 14, alpha: fade }); }
        if (sl > 0) {
          const bS = square([0, 0], [a, b], [a - b, b + a], [-b, a]);
          g.polygon(bS, { fill: 'v', stroke: 'v', alpha: 0.3 * sl });
          if (sl > 0.6) g.label('area ' + LA.fmt(a * a + b * b) + ' = ' + LA.fmt(a * a) + ' + ' + LA.fmt(b * b), bigC, { color: 'v', size: 14 });
        }
      }
      if (s.showPQ) {
        const d = [-2 * s.shift, 1 * s.shift], p = N.add(s.p, d), q = N.add(s.q, d);
        if (s.shift > 0.02) { g.point(s.p, { r: 4, color: 'data', alpha: 0.3 }); g.point(s.q, { r: 4, color: 'dataB', alpha: 0.3 }); }
        g.vector(p, q, { color: 'v', width: 3, label: 'q − p', labelAt: 0.35, labelOffset: 26, labelFont: 'ui', labelSize: 13 });
        g.point(p, { r: 6, color: 'data', label: 'p', dx: -12, dy: 10 });
        g.point(q, { r: 6, color: 'dataB', label: 'q', dx: 12, dy: -8 });
        g.dim(p, q, { label: '‖q − p‖ = ' + LA.fmt(N.dist(p, q)), offset: -22, color: 'hud' });
      }
      // the main arrow (with its ghost when it shrinks to the unit circle)
      if (s.showV) {
        const vv = N.scale(v, 1 - (1 - 1 / Math.max(nv, 1e-9)) * s.unitTween);
        if (s.ghostV) { g.lineThrough([0, 0], v, { color: 'residual', dashed: true, width: 1, alpha: 0.6, len: 8 }); g.vector(v, { color: 'v', width: 3, alpha: 0.35, label: 'v', labelColor: 'muted' }); }
        if (s.showCircle) g.circle([0, 0], 1, { stroke: 'hud', width: 1.2, dashed: true, strokeAlpha: 0.7 });
        if (s.showBall) {
          const m = s.ballM;
          g.polygon(ballPoly(m), { fill: 'teal', stroke: 'teal', alpha: 0.18, width: 2 });
          const name = m <= 0.5 || m >= 2.5 ? 'L2: ‖·‖₂ = 1 is the circle' : m <= 1.5 ? 'L1: ‖·‖₁ = 1 is the diamond' : 'L∞: ‖·‖∞ = 1 is the square';
          g.label(name, [0, -1.35], { color: 'teal', anchor: 'below', size: 13.5, font: 'ui' });
        }
        if (s.unitArrow) {
          // the arrow re-measured under the current ruler: scaled so its norm reads 1
          const th = Math.atan2(v[1], v[0]), r = ballRadius(th, s.ballM);
          const tip = [r * Math.cos(th), r * Math.sin(th)];
          g.vector(tip, { color: 'v', width: 3, label: 'v', hat: true });
          g.label('[' + LA.fmt(tip[0]) + '; ' + LA.fmt(tip[1]) + ']', tip, { color: 'v', dx: 14, dy: 16, anchor: 'left', size: 12.5, font: 'ui' });
        } else {
          g.vector(vv, { color: 'v', width: 3, label: s.unitTween > 0.5 ? 'v' : 'v', hat: s.unitTween > 0.5, components: s.comps });
        }
        if (s.unitTween >= 1) g.label('v̂ = v / ‖v‖ = [' + LA.fmt(vv[0]) + '; ' + LA.fmt(vv[1]) + ']', vv, { color: 'v', dx: 22, dy: 26, anchor: 'left', size: 13, font: 'ui' });
        if (s.showDim) g.dim([0, 0], vv, { label: '‖v‖ = ' + LA.fmt(N.norm(vv)), offset: -20, color: 'hud' });
        // the gray bar of length 3 + 4 (misconception)
        if (s.bar > 0) {
          const L = N.norm1(v), u = N.normalize(v), end = N.scale(u, L * s.bar);
          g.line([0, 0], end, { color: 'residual', width: 9, alpha: 0.55 });
          g.label('bar of length ' + LA.fmt(L), end, { color: 'hud', dx: 14, dy: 0, anchor: 'left', size: 12.5, font: 'ui' });
          if (s.overshoot) g.dim(v, end, { label: 'overshoot ' + LA.fmt(L - nv), offset: 22, color: 'bad' });
          g.label('"length = ' + LA.fmt(v[0]) + ' + ' + LA.fmt(v[1]) + ' = ' + LA.fmt(L) + '"', [18, 30], { screen: true, anchor: 'left', color: 'muted', size: 14, font: 'ui' });
          if (s.cross) g.marker.cross([186, 30], { screen: true, size: 22 });
        }
        // taxicab staircase
        if (s.taxi > 0) {
          const steps = [];
          const sx = Math.sign(v[0]) || 1, sy = Math.sign(v[1]) || 1, ax = Math.abs(v[0]), ay = Math.abs(v[1]);
          for (let i = 0; i < ax; i++) steps.push([[sx * i, 0], [sx * Math.min(i + 1, ax), 0], 'i']);
          for (let j = 0; j < ay; j++) steps.push([[v[0], sy * j], [v[0], sy * Math.min(j + 1, ay)], 'j']);
          const n = steps.length, done = s.taxi * n;
          steps.forEach((st, i) => {
            const f = LA.clamp01(done - i); if (f <= 0) return;
            g.line(st[0], lerpV(st[0], st[1], f), { color: st[2], width: 6, alpha: 0.9 });
            g.point(st[0], { r: 3.5, color: st[2] });
            if (f >= 1) g.label(String(i + 1), lerpV(st[0], st[1], 0.5), { color: st[2], size: 12, dx: st[2] === 'i' ? 0 : -14, dy: st[2] === 'i' ? 16 : 0, font: 'ui' });
          });
          if (s.taxi >= 1) g.label('‖v‖₁ = ' + LA.fmt(ax) + ' + ' + LA.fmt(ay) + ' = ' + LA.fmt(ax + ay) + ' blocks', [v[0], v[1]], { color: 'hud', dx: 16, dy: 34, anchor: 'left', size: 13, font: 'ui' });
        }
        if (s.maxPulse > 0) {
          const big = Math.abs(v[1]) >= Math.abs(v[0]) ? 1 : 0, col = big ? 'j' : 'i';
          const a0 = big ? [v[0], 0] : [0, 0], a1 = big ? [v[0], v[1]] : [v[0], 0];
          g.line(a0, a1, { color: col, width: 3 + 5 * s.maxPulse, alpha: 0.5 + 0.5 * s.maxPulse });
          g.dim(a0, a1, { label: '‖v‖∞ = ' + LA.fmt(N.normInf(v)), offset: 30, color: col });
        }
        if (s.showTable) {
          const rows = [['‖v‖₁ = ' + LA.fmt(N.norm1(v)), 'taxicab', 'teal'], ['‖v‖₂ = ' + LA.fmt(nv), 'straight', 'gold'], ['‖v‖∞ = ' + LA.fmt(N.normInf(v)), 'biggest step', 'mag']];
          rows.forEach((r, i) => { g.label(r[0], [18, 30 + i * 22], { screen: true, anchor: 'left', color: r[2], size: 14, font: 'ui' }); g.label(r[1], [120, 30 + i * 22], { screen: true, anchor: 'left', color: 'muted', size: 12.5, font: 'ui' }); });
        }
        if (ctx.mode === 'play' || ctx.mode === 'check') {
          g.hud('‖v‖₁ = ' + LA.fmt(N.norm1(v)) + '   ‖v‖₂ = ' + LA.fmt(nv) + '   ‖v‖∞ = ' + LA.fmt(N.normInf(v)), 14, 22, { color: 'hud' });
          if (ctx.mode === 'play' && s.normType === 'L2' && nv > 0.3) {
            g.polygon(square([0, 0], [v[0], 0], [v[0], -v[0]], [0, -v[0]]), { fill: 'i', stroke: 'i', alpha: 0.12 });
            g.polygon(square([v[0], 0], [v[0], v[1]], [v[0] + v[1], v[1]], [v[0] + v[1], 0]), { fill: 'j', stroke: 'j', alpha: 0.12 });
          }
        }
        if (s.drag) g.handle('v', s.unitArrow ? v : vv, (p, st) => { st.v = p.slice(); }, { color: 'v' });
      }
    },

    math(s, tex) {
      const sv = tex.sym('v', { color: 'v' }), nv = tex.norm(sv), v = s.v;
      switch (s.panel) {
        case 'pyth': return tex.block(tex.eq(nv, tex.op('='), tex.sqrt(nm(v[0], 'i') + '²' + tex.op('+') + nm(v[1], 'j') + '²'), tex.op('='), tex.sqrt(nm(v[0] * v[0] + v[1] * v[1])), tex.op('='), nm(N.norm(v), 'v')));
        case 'pyth3': { const u = s.v3; return tex.block(tex.eq(nv, tex.op('='), tex.sqrt(nm(u[0], 'i') + '²' + tex.op('+') + nm(u[1], 'j') + '²' + tex.op('+') + nm(u[2], 'k') + '²'), tex.op('='), tex.sqrt(nm(N.dot(u, u))), tex.op('='), nm(N.norm(u), 'v'))); }
        case 'mis': return tex.block(tex.eq(nm(v[0], 'i'), tex.op('+'), nm(v[1], 'j'), tex.op('='), nm(v[0] + v[1]), tex.op('≠'), nv, tex.op('='), nm(N.norm(v), 'v')));
        case 'dist': { const d = N.sub(s.q, s.p); return tex.block(tex.eq(tex.text('dist'), tex.paren(tex.sym('p', { color: 'data' }) + tex.op(', ') + tex.sym('q', { color: 'dataB' })), tex.op('='), tex.norm(tex.sym('q', { color: 'dataB' }) + tex.op('−') + tex.sym('p', { color: 'data' })), tex.op('='), tex.norm(tex.vec(d, { color: 'v' })), tex.op('='), nm(N.norm(d), 'v'))); }
        case 'unit': return tex.block(tex.eq(tex.sym('v', { hat: true, color: 'v' }), tex.op('='), tex.frac(sv, nv), tex.op('='), tex.vec([v[0] / 5, v[1] / 5], { color: 'v' }), tex.text(' = [3/5; 4/5]'), tex.op(','), tex.norm(tex.sym('v', { hat: true, color: 'v' })), tex.op('='), nm(1)));
        case 'l1': return tex.lines(tex.eq(sub(nv, '1'), tex.op('='), tex.abs(nm(v[0], 'i')), tex.op('+'), tex.abs(nm(v[1], 'j')), tex.op('='), nm(N.norm1(v), 'teal')), tex.eq(sub(nv, '2'), tex.op('='), nm(N.norm(v), 'gold')));
        case 'linf': return tex.block(tex.eq(sub(nv, '∞'), tex.op('='), tex.text('max'), tex.paren(tex.abs(nm(v[0], 'i')) + tex.op(', ') + tex.abs(nm(v[1], 'j'))), tex.op('='), nm(N.normInf(v), 'mag')));
        case 'balls': return tex.lines(tex.eq(sub(tex.norm('·'), '2'), tex.op('= 1:'), tex.text('circle', { color: 'teal' })), tex.eq(sub(tex.norm('·'), '1'), tex.op('= 1:'), tex.text('diamond', { color: 'teal' })), tex.eq(sub(tex.norm('·'), '∞'), tex.op('= 1:'), tex.text('square', { color: 'teal' })));
        case 'strip': { const u = s.strip; return tex.lines(tex.eq(sub(nv, '2'), tex.op('='), tex.sqrt(tex.sum(tex.sym('v', { sub: 'i', color: 'v' }) + '²', 'i')), tex.op('='), nm(N.norm(u), 'gold')), tex.eq(sub(nv, '1'), tex.op('='), tex.sum(tex.abs(tex.sym('v', { sub: 'i', color: 'v' })), 'i'), tex.op('='), nm(N.norm1(u), 'teal')), tex.eq(sub(nv, '∞'), tex.op('='), tex.text('max'), tex.sub ? '' : '', tex.abs(tex.sym('v', { sub: 'i', color: 'v' })), tex.op('='), nm(N.normInf(u), 'mag'))); }
        case 'knn': {
          const type = s.knnT < 0.7 ? 'L2' : 'L1', nb = neighbours(s.knnPts, s.probe, s.k, type);
          const blue = nb.filter(n => s.knnLabels[n.i] === 0).length, rose = nb.length - blue;
          return tex.lines(tex.eq(tex.sym('k'), tex.op('='), nm(s.k), tex.op('  ·  '), tex.text('ruler: '), sub(tex.norm('·'), NORM_SUB[type])), tex.eq(tex.text('vote: '), nm(blue, 'data'), tex.text(' blue', { color: 'data' }), tex.op(' / '), nm(rose, 'dataB'), tex.text(' rose', { color: 'dataB' }), tex.op(' → '), tex.text(blue > rose ? 'blue' : rose > blue ? 'rose' : 'tie', { color: blue > rose ? 'data' : rose > blue ? 'dataB' : 'muted' })));
        }
        case 'play': return tex.lines(tex.eq(sv, tex.op('='), tex.vec(v, { colors: 'rows' })), tex.eq(sub(nv, '1'), tex.op('='), nm(N.norm1(v), 'teal'), tex.op('   '), sub(nv, '2'), tex.op('='), nm(N.norm(v), 'gold'), tex.op('   '), sub(nv, '∞'), tex.op('='), nm(N.normInf(v), 'mag')));
        default: return '';
      }
    },

    steps: [
      { title: 'Length from components', duration: 5,
        narration: 'The yellow ' + vS + ' = [3; 4]. How long is it? Build a square on the green component (area 9) and one on the red component (area 16). Slide them together: they fill a square of area 25 whose side is the arrow itself. The <b>length</b> is √25 = 5. That is Pythagoras.',
        setup(s) { s.view = '2d'; s.cam = Object.assign(LA.cam.defaults2d(), CAM_PYTH); s.v = V.slice(); s.showV = true; s.comps = 'labels'; s.sqGrow = 0; s.squares = 0; s.showDim = false; s.panel = 'pyth'; s.drag = false; },
        at(s, t) { s.sqGrow = seg(t, 0.05, 0.35); s.squares = seg(t, 0.45, 0.85, 'inOut'); s.showDim = t > 0.88; } },

      { title: 'Three components, same rule', duration: 4,
        narration: 'In 3D add the blue square too: [2; 3; 6] has squares 4, 9 and 36, total 49, so the length is 7. The rule is the same: square every component, add, take the root. For a strip of 784 pixels it is the same sum with 784 terms.',
        setup(s) { s.view = '3d'; s.cam = Object.assign(LA.cam.defaults3d(), CAM3); s.v3 = V3.slice(); s.sq3 = 0; s.showDim3 = false; s.panel = 'pyth3'; },
        at(s, t) { s.sq3 = seg(t, 0.1, 0.55); s.showDim3 = t > 0.6; LA.cam.tween(s, { yaw: 0.5 }, { yaw: 0.68 }, seg(t, 0, 1, 'linear')); } },

      { title: 'Length is not the sum of components', duration: 3.5, misconception: true,
        narration: 'Belief: "the length is 3 + 4 = 7." Lay a gray bar of length 7 along the arrow: it overshoots by two units. Walking 3 east and 4 north is a 7-unit walk, but the straight arrow is only 5 long. The sum of components measures a different kind of distance, which we meet in a moment.',
        setup(s) { s.view = '2d'; s.cam = Object.assign(LA.cam.defaults2d(), CAM_MAIN); s.sqGrow = 0; s.squares = 0; s.showDim = true; s.bar = 0; s.cross = false; s.overshoot = false; s.panel = 'mis'; },
        at(s, t) { s.bar = seg(t, 0.05, 0.6); s.overshoot = t > 0.62; s.cross = t > 0.8; } },

      { title: 'Distance is the length of a difference', duration: 4,
        narration: 'Two points, the blue p = [1; 1] and the rose q = [4; 5]. The arrow from p to q is q − p = [3; 4], length 5. The <b>distance</b> between two points is the length of their difference. Slide both points together anywhere: the distance does not change, because the difference does not.',
        setup(s) { s.bar = 0; s.cross = false; s.overshoot = false; s.showV = false; s.showDim = false; s.showPQ = true; s.shift = 0; s.panel = 'dist'; },
        at(s, t) { s.shift = Math.sin(seg(t, 0.35, 1, 'linear') * Math.PI); } },

      { title: 'Unit vectors', duration: 3.5,
        narration: 'Divide ' + vS + ' by its own length and you get a <b>unit vector</b>: same direction, length exactly 1, with its tip on the unit circle. To <b>normalize</b> a vector means exactly this: v̂ = v/‖v‖. It keeps only the direction, the information that cosine similarity cares about.',
        setup(s) { s.showPQ = false; s.showV = true; s.comps = null; s.ghostV = true; s.showCircle = false; s.unitTween = 0; s.panel = 'unit'; s._cam = LA.clone(s.cam); },
        at(s, t) { s.showCircle = t > 0.1; s.unitTween = seg(t, 0.25, 0.8, 'inOut'); LA.cam.tween(s, s._cam, CAM_UNIT, seg(t, 0, 0.6)); } },

      { title: 'The taxicab norm', duration: 3.5,
        narration: 'The 7 from before is a legitimate measure: walk along the grid like a taxi, 3 blocks east and 4 north. That is the <b>L1 norm</b>, the sum of absolute components (the bars |·| drop any minus sign): ‖v‖₁ = |3| + |4| = 7. The subscript 1 tells the norms apart; the Pythagoras length is ‖v‖₂.',
        setup(s) { s.unitTween = 0; s.ghostV = false; s.showCircle = false; s.comps = 'labels'; s.cam = Object.assign(LA.cam.defaults2d(), CAM_MAIN); s.taxi = 0; s.gridBright = 0; s.panel = 'l1'; },
        at(s, t) { s.gridBright = seg(t, 0, 0.2); s.taxi = seg(t, 0.1, 0.75, 'linear'); } },

      { title: 'The max norm', duration: 2.5,
        narration: 'One more: the <b>L∞ norm</b> is simply the largest absolute component, ‖v‖∞ = max(|3|, |4|) = 4. It asks "what is the biggest single step?" Three norms, three numbers for the same arrow: 7, 5 and 4.',
        setup(s) { s.taxi = 0; s.gridBright = 0; s.maxPulse = 0; s.showTable = false; s.panel = 'linf'; },
        at(s, t) { s.maxPulse = Math.sin(seg(t, 0.05, 0.75, 'linear') * Math.PI * 2) * 0.5 + 0.5; s.showTable = t > 0.5; } },

      { title: 'Unit balls: diamond, circle, square', duration: 6,
        narration: 'Draw every vector of norm 1. Under L2 that is the circle. Under L1 it is a diamond with corners on the axes. Under L∞ it is a square. Same room, three rulers: the <b>unit ball</b> of each. In 784 dimensions the diamond still has its corners on the axes, which is why L1 makes weights exactly zero in Part 5.',
        setup(s) { s.maxPulse = 0; s.showTable = false; s.comps = null; s.cam = Object.assign(LA.cam.defaults2d(), CAM_BALL); s.showBall = true; s.ballM = 0; s.unitArrow = true; s.panel = 'balls'; },
        at(s, t) { s.ballM = seg(t, 0.1, 0.35) + seg(t, 0.4, 0.65) + seg(t, 0.7, 0.95); } },

      { title: 'Same formula in any dimension', duration: 3.5,
        narration: 'For a strip, ‖v‖₂ squares every cell, adds, and takes the root; ‖v‖₁ adds absolute values; ‖v‖∞ takes the largest. One strip in, three numbers out, by the same three rules. The picture stops at 3. The formula does not.',
        setup(s) { s.showBall = false; s.unitArrow = false; s.cam = Object.assign(LA.cam.defaults2d(), CAM_STRIP); s.stripT = 0.001; s.panel = 'strip'; },
        at(s, t) { s.stripT = Math.max(0.001, t); } },

      { title: 'Which ruler decides your neighbors', duration: 6, mlTie: true,
        narration: 'A nearest-neighbor classifier labels a new point by the labels of the k closest points. "Closest" depends on the ruler. Here the orange probe’s three nearest neighbors under L2 are two blue and one rose; under L1 the set changes and the vote flips to rose. Choosing a norm is choosing what "near" means.',
        setup(s) { s.stripT = 0; s.cam = Object.assign(LA.cam.defaults2d(), CAM_KNN); s.showKnn = true; s.knnT = 0; s.k = 3; s.probe = PROBE.slice(); s.panel = 'knn'; },
        at(s, t) { s.knnT = t; } },

      { title: 'Recap', duration: 2, recap: true,
        narration: 'Length is Pythagoras on the components: square, add, root. Distance is the length of a difference. Normalizing keeps direction and sets length to 1. L1 adds absolute values (diamond), L2 is the straight length (circle), L∞ is the biggest component (square). Which ruler you use decides who counts as near.',
        cards: ['<b>‖v‖₂ = √Σvᵢ²: square, add, root.</b> The two squares on the legs fill the square on the arrow.', '<b>Diamond, circle, square.</b> L1, L2 and L∞ each draw their own unit ball; the diamond keeps its corners in any dimension.', '<b>Distance = ‖q − p‖; the ruler decides the neighbors.</b> A probe can change its nearest neighbors by changing the norm.'],
        setup(s) { s.showKnn = false; s.cam = Object.assign(LA.cam.defaults2d(), CAM_BALL); s.showBall = true; s.ballM = 0; s.unitArrow = true; s.recapBalls = true; s.panel = 'balls'; },
        at(s, t) { } },
    ],

    playground: {
      intro: '<p>Drag the tip of ' + vS + ' and read its three norms. Pick a ruler to see its unit ball; the Pythagoras squares show for L2. The k-NN demo lets you drag the orange probe and change k.</p>',
      setup(s) { s.view = '2d'; s.cam = Object.assign(LA.cam.defaults2d(), CAM_PLAY); s.v = V.slice(); s.showV = true; s.comps = 'labels'; s.showBall = true; s.ballM = 0; s.normType = 'L2'; s.drag = true; s.panel = 'play'; s.showKnn = false; s.k = 3; },
      controls: [
        { type: 'matrix', bind: 'v', label: 'v (drag tip)', colors: 'rows', step: 0.1, min: -6, max: 6 },
        { type: 'preset', label: 'Norm (ruler)', items: [
          { label: 'L1', apply(s) { s.normType = 'L1'; s.ballM = 1; } },
          { label: 'L2', apply(s) { s.normType = 'L2'; s.ballM = 0; } },
          { label: 'L∞', apply(s) { s.normType = 'Linf'; s.ballM = 2; } },
        ] },
        { type: 'toggle', bind: 'showBall', label: 'Unit ball of the chosen norm' },
        { type: 'button', label: 'Normalize v', onClick(s) { const n = N.norm(s.v); if (n > 1e-9) s.v = N.scale(s.v, 1 / n); } },
        { type: 'slider', bind: 'k', min: 1, max: 7, step: 1, label: 'k neighbors (k-NN demo)', color: 'b', fmt: x => String(Math.round(x)) },
        { type: 'preset', label: 'Presets', items: [
          { label: '[3; 4]', apply(s) { s.showKnn = false; s.view = '2d'; s.cam = Object.assign(LA.cam.defaults2d(), CAM_PLAY); s.v = V.slice(); s.panel = 'play'; } },
          { label: '[2; 3; 6] (3D)', apply(s) { s.showKnn = false; s.view = '3d'; s.cam = Object.assign(LA.cam.defaults3d(), CAM3); s.v3 = V3.slice(); s.sq3 = 1; s.showDim3 = true; s.panel = 'pyth3'; } },
          { label: 'k-NN demo', apply(s) { s.view = '2d'; s.showKnn = true; s.knnT = s.normType === 'L1' ? 1 : 0; s.cam = Object.assign(LA.cam.defaults2d(), CAM_KNN); s.probe = PROBE.slice(); s.panel = 'knn'; } },
        ] },
      ],
    },

    checks: [
      { kind: 'manipulate',
        prompt: 'Drag ' + vS + ' so that ‖v‖₂ = 5 and v points into the lower-left quadrant.',
        setup(s) { s.view = '2d'; s.cam = Object.assign(LA.cam.defaults2d(), { center: [0, 0], halfH: 6 }); s.v = V.slice(); s.showV = true; s.comps = 'labels'; s.showBall = false; s.unitArrow = false; s.normType = 'L2'; s.drag = true; s.panel = 'play'; s.showKnn = false; s.stripT = 0; },
        judge(s) { const n = N.norm(s.v), dn = Math.abs(n - 5), quad = s.v[0] < 0 && s.v[1] < 0; return { ok: dn < LA.TOL.pos && quad, progress: Math.max(0, 1 - dn / 3) * (quad ? 1 : 0.4), hint: quad && dn < 0.4 ? 'Nearly: the length must read exactly 5.' : (!quad && dn < LA.TOL.pos ? 'Right length; now point it down and to the left.' : undefined) }; },
        hint: '[−3; −4] works, as does any mirror of [3; 4] into that corner.',
        solution(s) { s.v = [-3, -4]; } },
      { kind: 'manipulate',
        prompt: 'Drag ' + vS + ' onto the L1 unit diamond (make ‖v‖₁ = 1).',
        setup(s) { s.view = '2d'; s.cam = Object.assign(LA.cam.defaults2d(), { center: [0.3, 0.2], halfH: 2.2 }); s.v = [1.5, 1]; s.showV = true; s.comps = 'labels'; s.showBall = true; s.ballM = 1; s.normType = 'L1'; s.unitArrow = false; s.drag = true; s.panel = 'play'; s.showKnn = false; s.stripT = 0; },
        judge(s) { const d = Math.abs(N.norm1(s.v) - 1); return { ok: d < LA.TOL.unitNorm, progress: Math.max(0, 1 - d / 1.5) }; },
        hint: 'The green and red components must add to exactly 1 in absolute value: the tip sits on the teal diamond.',
        solution(s) { s.v = [0.6, 0.4]; } },
      { kind: 'manipulate',
        prompt: 'Drag ' + vS + ' so that ‖v‖₁ = 2 while ‖v‖∞ = 1. The three norms are read out in the top-left corner.',
        setup(s) { s.view = '2d'; s.cam = Object.assign(LA.cam.defaults2d(), { center: [0.3, 0.2], halfH: 2.4 }); s.v = [1.5, 0.2]; s.showV = true; s.comps = 'labels'; s.showBall = false; s.normType = 'L1'; s.unitArrow = false; s.drag = true; s.panel = 'play'; s.showKnn = false; s.stripT = 0; },
        judge(s) { const d1 = Math.abs(N.norm1(s.v) - 2), dI = Math.abs(N.normInf(s.v) - 1); return { ok: d1 < LA.TOL.unitNorm && dI < LA.TOL.unitNorm, progress: Math.max(0, 1 - (d1 + dI) / 2) }; },
        hint: 'Both components must have size exactly 1: one of the four corners [±1; ±1].',
        solution(s) { s.v = [1, 1]; } },
    ],
  });

  /** 3D Pythagoras: three squares flat along the three component pieces. */
  function render3d(g, s) {
    const u = s.v3, f = s.sq3 || 0;
    g.cage({ range: 6, walls: true });
    g.axes({ length: 6.5, labels: true });
    const sq = (pts, color, label, c) => { g.polygon(pts, { fill: color, stroke: color, alpha: 0.26 * f, subdiv: 2 }); if (f >= 1) g.label('area ' + label, c, { color, size: 13.5, font: 'ui' }); };
    const a = u[0], b = u[1], c = u[2];
    if (f > 0) {
      sq([[0, 0, 0], [a, 0, 0], [a, -a * f, 0], [0, -a * f, 0]], 'i', LA.fmt(a * a), [a / 2, -a / 2, 0]);
      sq([[a, 0, 0], [a, b, 0], [a + b * f, b, 0], [a + b * f, 0, 0]], 'j', LA.fmt(b * b), [a + b / 2, b / 2, 0]);
      sq([[a, b, 0], [a, b, c], [a, b + c * f, c], [a, b + c * f, 0]], 'k', LA.fmt(c * c), [a, b + c / 2, c / 2]);
    }
    g.vector(u, { color: 'v', width: 3, label: 'v', components: true });
    g.label(LA.fmt(a), [a / 2, 0, 0], { color: 'i', dy: 14, size: 12.5 });
    g.label(LA.fmt(b), [a, b / 2, 0], { color: 'j', dx: 14, size: 12.5 });
    g.label(LA.fmt(c), [a, b, c / 2], { color: 'k', dx: -14, size: 12.5 });
    if (s.showDim3) g.dim([0, 0, 0], u, { label: '‖v‖ = ' + LA.fmt(N.norm(u)), offset: -22, color: 'hud' });
  }

  /** n-D honesty step: one 8-cell strip, its squares, and three bars. */
  function renderStrip(g, s) {
    const v = s.strip, n = v.length, t = s.stripT, cw = 0.6, x0 = -cw * n / 2;
    const cells = (vals, y, range, ramp, title, tc) => {
      g.heat2d([vals], { x0, x1: x0 + cw * n, y0: y - 0.3, y1: y + 0.3 }, { range, ramp, alpha: 0.9, smooth: false });
      for (let i = 0; i <= n; i++) g.line([x0 + i * cw, y - 0.3], [x0 + i * cw, y + 0.3], { color: 'line', width: 1 });
      g.line([x0, y - 0.3], [x0 + n * cw, y - 0.3], { color: 'line' }); g.line([x0, y + 0.3], [x0 + n * cw, y + 0.3], { color: 'line' });
      for (let i = 0; i < n; i++) g.label(LA.fmt(vals[i]), [x0 + (i + 0.5) * cw, y], { color: 'text', size: 13, font: 'mono', bg: false });
      g.label(title, [x0, y], { color: tc || 'muted', anchor: 'right', dx: -10, size: 13, font: 'ui' });
    };
    cells(v, 2.2, [-3, 3], u => LA.color.diverge(u * 2 - 1), 'v (8 cells)', 'v');
    const k = Math.floor(seg(t, 0.05, 0.5, 'linear') * n + 1e-9);
    if (k > 0) {
      const sqv = v.map((x, i) => i < k ? x * x : 0);
      cells(sqv, 1.2, [0, 9], u => LA.color.heat(u * 0.9), 'vᵢ² ');
      const sum = sqv.reduce((a, b) => a + b, 0);
      g.label('running sum of squares = ' + LA.fmt(sum) + (k >= n ? '   →   √' + LA.fmt(sum) + ' = ' + LA.fmt(Math.sqrt(sum)) : ''), [x0 + cw * n / 2, 0.55], { color: 'gold', size: 13.5, font: 'ui' });
    }
    if (t > 0.6) {
      const vals = [N.norm1(v), N.norm(v), N.normInf(v)];
      const f = seg(t, 0.6, 0.85);
      g.bars(vals.map(x => x * f), { origin: [-1.3, -2.6], width: 0.7, gap: 0.35, height: 1.9, max: Math.max(...vals), colors: ['teal', 'gold', 'mag'], labels: ['‖v‖₁', '‖v‖₂', '‖v‖∞'], values: f >= 1 });
      g.label('one strip in, three numbers out', [0.25, -0.1], { color: 'muted', size: 12.5, font: 'ui' });
    }
    g.label('The picture stops at 3. The formula does not.', [0, -3.2], { color: 'muted', anchor: 'below', dy: 6, size: 13, font: 'ui' });
  }

  /** ML tie-in: k nearest neighbours of a probe under L2, then under L1. */
  function renderKnn(g, s, ctx) {
    g.ghostGrid();
    g.axes({ labels: true });
    const t = s.knnT, type = t < 0.7 ? 'L2' : 'L1';
    const pts = s.knnPts, lab = s.knnLabels, p = s.probe, k = Math.max(1, Math.round(s.k));
    g.points(pts, { r: 5, colors: i => lab[i] ? 'dataB' : 'data' });
    const nb = neighbours(pts, p, k, type);
    const nb2 = neighbours(pts, p, k, 'L2'), nb1 = neighbours(pts, p, k, 'L1');
    const grow = ctx.mode === 'learn' ? seg(t, 0.05, 0.4) : 1;
    nb.forEach(n => g.line(p, lerpV(p, pts[n.i], grow), { color: 'residual', width: 1.4, dashed: true }));
    if (grow >= 1) nb.forEach(n => g.marker.ring(pts[n.i], { r: 10, color: lab[n.i] ? 'dataB' : 'data', width: 1.6 }));
    // the ball through the k-th neighbour: circle under L2, diamond under L1 (morphing between them)
    const r2 = nb2[nb2.length - 1].d, r1 = nb1[nb1.length - 1].d;
    const m = ctx.mode === 'learn' ? seg(t, 0.5, 0.9) : (type === 'L1' ? 1 : 0);
    const rr = LA.lerp(r2, r1, m);
    if (grow >= 1) g.polygon(ballPoly(m, rr, p), { fill: 'b', stroke: 'b', alpha: 0.1, width: 1.6 });
    g.point(p, { r: 7, color: 'b', label: 'probe', dx: 14, dy: -10 });
    const blue = nb.filter(n => lab[n.i] === 0).length, rose = nb.length - blue;
    g.hud('ruler: ' + NORM_NAME[type] + '   ·   k = ' + k, 14, 22, { color: 'hud' });
    if (grow >= 1) g.hud('vote: ' + blue + ' blue / ' + rose + ' rose  →  ' + (blue > rose ? 'blue' : rose > blue ? 'rose' : 'tie'), 14, 42, { color: blue > rose ? 'data' : rose > blue ? 'dataB' : 'muted' });
    g.hud('synthetic data: 15 blue, 15 rose', 14, g.size.h - 40, { color: 'muted' });
    g.hud('Part 4: Mahalanobis distance · Part 6: embeddings', 14, g.size.h - 18, { color: 'hud' });
    if (ctx.mode === 'play') g.handle('probe', p, (q, st) => { st.probe = q.slice(); }, { color: 'b' });
  }
})();
