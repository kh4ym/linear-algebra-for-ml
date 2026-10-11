/* p1-02 — Adding and scaling vectors. Tip to tail and componentwise addition are one rule; scaling stretches,
   shrinks and flips; v − w is the arrow between tips; the mean is a vector; a gradient step is scale-and-subtract. */
(function () {
  'use strict';
  const N = LA.num, seg = LA.seg, lerpV = LA.lerpV, fmt = LA.fmt;
  const V0 = [3, 1], W0 = [1, 2];
  const ROW = ['i', 'j'];
  const cam2 = () => Object.assign(LA.cam.defaults2d(), { center: [1.8, 1.0], halfH: 3.9 });
  const camWide = () => Object.assign(LA.cam.defaults2d(), { center: [1.4, 0.3], halfH: 5.8 });
  const vS = '<span class="c-v">v</span>', wS = '<span class="c-w">w</span>';

  // the ML step: L(w) = ½ (w − m)ᵀ A (w − m) with A = diag(0.4, 1.2); ∇L at W_ML is [0.8, 1.6]
  const W_ML = [2.5, 1.5], M_ML = [0.5, 1 / 6], A_ML = [0.4, 1.2], ETA = 0.5;
  const gradAt = w => [A_ML[0] * (w[0] - M_ML[0]), A_ML[1] * (w[1] - M_ML[1])];
  const stepFrom = w => N.sub(w, N.scale(gradAt(w), ETA));
  const PATH = (() => { const p = [W_ML]; for (let k = 0; k < 4; k++) p.push(stepFrom(p[p.length - 1])); return p; })();

  /** Column bracket at a screen point (or beside a world point). Returns the screen centers of its entries. */
  function bracket(g, pos, vals, roles, o) {
    o = o || {};
    const size = o.size || 15, rowH = size * 1.25, n = vals.length, half = o.half || size * 1.45;
    const sp = o.screen ? pos : g.P.toScreen(pos);
    const sx = sp[0] + (o.dx || 0), sy = sp[1] + (o.dy || 0) - (n - 1) * rowH / 2;
    const al = o.alpha === undefined ? 1 : o.alpha, out = [];
    for (let i = 0; i < n; i++) {
      const y = sy + i * rowH; out.push([sx, y]);
      if (al <= 0.01) continue;
      const lg = n === 1 ? '[' : i === 0 ? '⎡' : i === n - 1 ? '⎣' : '⎢';
      const rg = n === 1 ? ']' : i === 0 ? '⎤' : i === n - 1 ? '⎦' : '⎥';
      g.label(lg, [sx - half, y], { screen: true, color: 'hud', size: size * 1.2, font: 'math', alpha: al, bg: false });
      const ea = o.entryAlpha ? o.entryAlpha[i] : 1;
      if (ea > 0.01) g.label(typeof vals[i] === 'string' ? vals[i] : fmt(vals[i]), [sx, y], { screen: true, color: roles ? roles[i] : (o.color || 'text'), size, font: 'ui', bold: true, alpha: al * ea });
      g.label(rg, [sx + half, y], { screen: true, color: 'hud', size: size * 1.2, font: 'math', alpha: al, bg: false });
    }
    return out;
  }

  /** The sum arrow: a yellow-to-purple shaft (v then w) with a purple head and a "v + w" label. */
  function sumArrow(g, a, b, o) {
    o = o || {};
    const L = N.norm(N.sub(b, a)); if (L < 1e-3) return;
    const al = o.alpha === undefined ? 1 : o.alpha, n = 10, C = LA.color;
    const stop = Math.max(0, 1 - 0.32 / L);
    for (let k = 0; k < n; k++) g.line(lerpV(a, b, stop * k / n), lerpV(a, b, stop * (k + 1) / n), { color: C.mix(C.v, C.w, (k + 0.5) / n), width: 3.4, alpha: al });
    g.vector(lerpV(a, b, Math.max(0, stop - 0.05)), b, { color: 'w', width: 3.4, alpha: al });
    if (o.label) g.label(o.label, b, { color: 'text', dx: o.ldx === undefined ? -14 : o.ldx, dy: o.ldy === undefined ? -16 : o.ldy, font: 'math', size: 15, alpha: al });
  }

  function drawPlane(g) {
    g.grid(null, { range: 10 });
    g.axes({ labels: true });
  }

  LA.registerLesson({
    id: 'p1-02-add-scale',
    part: 1,
    title: 'Adding and scaling vectors',
    subtitle: 'Tip to tail or entry by entry; stretch, shrink and flip; a gradient step is exactly this.',
    minutes: 8,
    prereqs: ['p1-01-what-is-a-vector'],
    tags: ['core'],
    badge: { drawnIn: '2D', realSize: 'a gradient step updates 10⁶–10⁹ weights at once with one vector subtraction', transfers: 'tip-to-tail and componentwise addition are the same rule; in ℝⁿ only the componentwise form is computed, and it is identical' },
    cantShow: ['addition of 784-component vectors as arrows (strips add cell by cell)'],
    terms: [
      { term: 'vector addition', def: 'Place the second arrow’s tail at the first arrow’s tip; the sum runs from the origin to the new tip, and componentwise it adds matching entries.' },
      { term: 'parallelogram rule', def: 'v + w is the far corner of the parallelogram with sides v and w.' },
      { term: 'scalar', def: 'A plain number used to stretch or shrink a vector.' },
      { term: 'scaling', def: 'Multiplying every component of a vector by the same scalar.' },
      { term: 'subtraction', def: 'v − w is the arrow from the tip of w to the tip of v; componentwise it subtracts matching entries.' },
      { term: 'mean', def: 'The mean of several vectors: add them and divide by how many there are; the result is a vector.' },
    ],
    atlas: ['optimization'],
    thumbnail: 3,

    state() {
      return {
        view: '2d', cam: cam2(),
        v: V0.slice(), w: W0.slice(),
        vA: 1, wA: 1, grow: 1, slide: 0, ghostW: 0, brk: 0,
        sumT: 0, sumScale: 1, sumComp: 0,
        route2: 0, para: 0,
        compT: 0, lenBar: 0, cross: false,
        c: 1, showScaled: false, _touchC: false,
        diffT: 0, midT: 0, mlT: 0, showML: false,
        showSum: false, showParallelogram: false, showDiff: false, readouts: false,
        dragV: false, dragW: false, dragC: false, dragGray: false,
        gray: null, target: null,
        panel: 'none', recap: false,
      };
    },

    render(g, s, ctx) {
      if (s.recap) { renderRecap(g); return; }
      drawPlane(g);
      if (s.showML) { renderML(g, s); return; }
      const v = s.v, w = s.w, sum = N.add(v, w);
      const t = ctx.t;

      // --- parallelogram (order does not matter / playground) ---
      if (s.para > 0.01 || s.showParallelogram) {
        const a = s.showParallelogram ? 1 : s.para;
        g.parallelogram(v, w, { fill: 'v', alpha: 0.13 * a, stroke: null });
      }
      // --- scaling: dashed line through v, ghost v, c·v ---
      if (s.showScaled) renderScaled(g, s);

      // --- ghost of w at the origin while it slides ---
      if (s.ghostW > 0.01) g.vector([0, 0], w, { color: 'w', width: 2.2, alpha: 0.32 * s.ghostW, label: 'w', labelOffset: -14 });
      // --- second route: w first, then v ---
      if (s.route2 > 0.01) {
        const o = N.scale(w, s.route2);
        g.vector(o, N.add(o, v), { color: 'v', width: 2.4, alpha: 0.6, label: 'v', labelAt: 0.5, labelOffset: 14 });
      }
      // --- v and w ---
      if (s.vA > 0.01 && s.grow > 0.01) g.vector([0, 0], N.scale(v, s.grow), { color: 'v', width: 3.2, alpha: s.vA, label: s.grow > 0.5 ? 'v' : '', labelAt: 0.55, labelOffset: 14 });
      if (s.wA > 0.01 && s.grow > 0.01) {
        const o = N.scale(v, s.slide);
        g.vector(o, N.add(o, N.scale(w, s.grow)), { color: 'w', width: 3.2, alpha: s.wA, label: s.grow > 0.5 ? 'w' : '', labelAt: 0.55, labelOffset: -14 });
      }
      // --- tip brackets (step 1) ---
      if (s.brk > 0.01) {
        bracket(g, v, v, ROW, { dx: 50, dy: 4, alpha: s.brk });
        bracket(g, w, w, ROW, { dx: -54, dy: -6, alpha: s.brk });
      }
      // --- the sum ---
      if (s.sumT > 0.01 || s.showSum) {
        const k = (s.showSum ? 1 : s.sumT) * s.sumScale;
        if (s.sumComp > 0.01) {
          g.line([0, 0], [sum[0], 0], { color: 'i', dashed: true, width: 1.7, alpha: 0.9 * s.sumComp });
          g.line([sum[0], 0], sum, { color: 'j', dashed: true, width: 1.7, alpha: 0.9 * s.sumComp });
          if (s.sumComp > 0.5) { g.dim([0, 0], [sum[0], 0], { label: fmt(sum[0]), color: 'i', offset: -26 }); g.dim([sum[0], 0], sum, { label: fmt(sum[1]), color: 'j', offset: -18 }); }
        }
        const lab = s.sumScale < 0.9 ? '(v + w)/2' : 'v + w';
        sumArrow(g, [0, 0], N.scale(sum, k), { label: k > 0.4 ? lab : '', ldx: s.sumScale < 0.9 ? -46 : -16, ldy: s.sumScale < 0.9 ? -10 : -16 });
        if (s.panel === 'sum' || s.showSum) bracket(g, N.scale(sum, k), N.scale(sum, k), ROW, { dx: 48, dy: -6, size: 14, alpha: Math.min(1, k) });
      }
      // --- componentwise (step 3) ---
      if (s.compT > 0.01) renderComp(g, s);
      // --- lengths do not add (step 5) ---
      if (s.lenBar > 0.01) renderLengths(g, s);
      // --- subtraction (step 7) ---
      if (s.diffT > 0.01) renderDiff(g, s);
      if (s.showDiff) {
        const d = N.sub(v, w);
        g.vector(w, v, { color: 'b', width: 2.8, label: 'v − w', labelAt: 0.5, labelOffset: 16 });
        g.vector([0, 0], d, { color: 'b', width: 2, alpha: 0.4, dashed: true });
        bracket(g, d, d, ROW, { dx: 46, dy: 6, size: 13, alpha: 0.85 });
      }
      // --- midpoint (step 8) ---
      if (s.midT > 0.01) {
        g.line(v, w, { color: 'residual', dashed: true, width: 1.6, alpha: s.midT });
        const m = N.scale(sum, 0.5);
        g.point(m, { r: 6, color: 'gold', alpha: s.midT });
        if (s.midT > 0.95) {
          const pulse = 15 + 7 * Math.abs(Math.sin(t * Math.PI * 3));
          g.marker.ring(m, { r: pulse, color: 'gold' });
          g.label('midpoint of the tips', m, { color: 'gold', dx: 72, dy: 18, size: 13, font: 'ui' });
          bracket(g, m, m, ROW, { dx: 40, dy: -26, size: 13 });
        }
      }
      // --- checks: target + gray arrow ---
      if (s.target) {
        g.vector([0, 0], s.target, { color: 'residual', width: 3, alpha: 0.75, label: 'target', labelAt: 0.6, labelOffset: 16 });
      }
      if (s.gray) {
        g.vector([0, 0], s.gray, { color: 'residual', width: 3, label: 'g', labelAt: 0.6, labelOffset: -14 });
        bracket(g, s.gray, s.gray, ROW, { dx: -52, dy: 0, size: 13 });
      }
      if (s.readouts) {
        g.hud('‖v‖ = ' + fmt(N.norm(v)) + '    ‖w‖ = ' + fmt(N.norm(w)) + '    ‖v + w‖ = ' + fmt(N.norm(sum)), 14, 20, { size: 13.5 });
        g.hud('‖v‖ + ‖w‖ = ' + fmt(N.norm(v) + N.norm(w)), 14, 40, { size: 12.5, color: 'muted' });
      }
      // --- handles ---
      if (s.dragV) g.handle('v', v, (p, st) => { st.v = [p[0], p[1]]; }, { color: 'v' });
      if (s.dragW) { const o = N.scale(v, s.slide); g.handle('w', N.add(o, w), (p, st) => { st.w = [p[0] - o[0], p[1] - o[1]]; }, { color: 'w' }); }
      if (s.dragC) g.handle('c', N.scale(v, s.c), (p, st) => { const vv = st.v, d = N.dot(vv, vv) || 1; st.c = LA.clamp(N.dot(p, vv) / d, -3, 3); st._touchC = true; }, { color: 'v' });
      if (s.dragGray && s.gray) g.handle('g', s.gray, (p, st) => { st.gray = [p[0], p[1]]; }, { color: 'white' });
    },

    math(s, tex) {
      const v = s.v, w = s.w, sum = N.add(v, w);
      const V = tex.sym('v', { color: 'v' }), W = tex.sym('w', { color: 'w' });
      const rows = { colors: 'rows' };
      // numbers as plain operators: tex.num after a tex.mat is written into the wrong cell by tex.reconcile (engine)
      const num = x => '<span>' + fmt(x) + '</span>';
      switch (s.panel) {
        case 'vw': return tex.block(tex.eq(V, tex.op('='), tex.vec(v, rows)), tex.eq(W, tex.op('='), tex.vec(w, rows)));
        case 'sum': return tex.block(tex.eq(V, tex.op('+'), W, tex.op('='), tex.vec(sum, rows)));
        case 'comp': return tex.block(tex.eq(tex.vec(v, rows), tex.op('+'), tex.vec(w, rows), tex.op('='), tex.vec([fmt(v[0]) + '+' + fmt(w[0]), fmt(v[1]) + '+' + fmt(w[1])], rows), tex.op('='), tex.vec(sum, rows)));
        case 'comm': return tex.block(tex.eq(V, tex.op('+'), W, tex.op('='), W, tex.op('+'), V, tex.op('='), tex.vec(sum, rows)));
        case 'len': return tex.block(
          tex.eq(tex.norm(V), tex.op('+'), tex.norm(W), tex.op('='), num(N.norm(v)), tex.op('+'), num(N.norm(w)), tex.op('='), num(N.norm(v) + N.norm(w))),
          tex.eq(tex.op('≠'), tex.norm(tex.eq(V, tex.op('+'), W)), tex.op('='), num(N.norm(sum))));
        case 'scale': return tex.block(
          tex.eq(tex.sym('c'), tex.op('·'), V, tex.op('='), tex.vec(['3c', 'c'], rows)),
          tex.eq(num(s.c), tex.op('·'), tex.vec(v, rows), tex.op('='), tex.vec(N.scale(v, s.c), rows)));
        case 'diff': return tex.block(tex.eq(V, tex.op('−'), W, tex.op('='), tex.vec([fmt(v[0]) + '−' + fmt(w[0]), fmt(v[1]) + '−' + fmt(w[1])], rows), tex.op('='), tex.vec(N.sub(v, w), rows)));
        case 'mid': return tex.block(tex.eq(tex.paren(tex.eq(V, tex.op('+'), W)), tex.op('/'), num(2), tex.op('='), tex.vec(N.scale(sum, 0.5), rows)));
        case 'ml': {
          const u = seg(s.mlT, 0.5, 0.8), nw = lerpV(PATH[0], PATH[1], u);
          return tex.block(
            tex.eq(W, tex.op('←'), W, tex.op('−'), tex.sym('η'), tex.sym('∇L'), tex.text('(η = 0.5)', { color: 'muted' })),
            tex.eq(tex.vec(PATH[0], { color: 'w' }), tex.op('−'), num(0.5), tex.op('·'), tex.vec(gradAt(PATH[0])), tex.op('='), tex.vec(nw, { color: 'w' })));
        }
        case 'play': return tex.block(
          tex.eq(V, tex.op('+'), W, tex.op('='), tex.vec(sum, rows)),
          tex.eq(num(s.c), tex.op('·'), V, tex.op('='), tex.vec(N.scale(v, s.c), rows)),
          tex.eq(V, tex.op('−'), W, tex.op('='), tex.vec(N.sub(v, w), rows)));
        case 'check1': return tex.block(tex.eq(V, tex.op('+'), W, tex.op('='), tex.vec(sum, rows)), tex.eq(tex.text('goal', { color: 'muted' }), tex.vec([1, 4], rows)));
        case 'check2': return tex.block(tex.eq(tex.sym('c'), tex.op('='), num(s.c)), tex.eq(tex.sym('c'), tex.op('·'), V, tex.op('='), tex.vec(N.scale(v, s.c), rows)));
        case 'check3': return tex.block(tex.eq(V, tex.op('='), tex.vec(v, rows)), tex.eq(W, tex.op('='), tex.vec(w, rows)), tex.eq(tex.text('gray', { color: 'muted' }), tex.op('='), tex.vec(s.gray || [0, 0], rows)));
        default: return '';
      }
    },

    steps: [
      { title: 'Two vectors', duration: 2.5,
        narration: 'Two vectors from the origin: the yellow ' + vS + ' = [3; 1] and the purple ' + wS + ' = [1; 2]. What should it mean to add them? Whatever we choose, it must make sense both for arrows and for lists.',
        setup(s) { s.view = '2d'; s.cam = cam2(); s.v = V0.slice(); s.w = W0.slice(); s.grow = 0; s.vA = 1; s.wA = 1; s.slide = 0; s.brk = 0; s.panel = 'vw'; },
        at(s, t) { s.grow = seg(t, 0, 0.6, 'out'); s.brk = seg(t, 0.6, 0.9); } },

      { title: 'Tip to tail', duration: 4,
        narration: 'Slide ' + wS + ', without turning it, until its tail sits on the tip of ' + vS + '. Walk v, then walk w. Where you end up is the sum ' + vS + ' + ' + wS + ': the arrow from the origin to that final tip, [4; 3]. This is <b>vector addition</b>.',
        setup(s) { s.grow = 1; s.brk = 0; s.ghostW = 1; s.slide = 0; s.sumT = 0; s.panel = 'vw'; },
        at(s, t) { s.slide = seg(t, 0, 0.6); s.sumT = seg(t, 0.6, 1, 'out'); s.panel = t > 0.6 ? 'sum' : 'vw'; } },

      { title: 'Componentwise', duration: 3,
        narration: 'The list face says the same thing. Add the green components: 3 + 1 = 4. Add the red components: 1 + 2 = 3. Matching entries add, nothing crosses rows. For strips of 784 numbers this is the only rule you need.',
        setup(s) { s.slide = 1; s.sumT = 1; s.ghostW = 0.6; s.compT = 0; s.sumComp = 0; s.panel = 'comp'; },
        at(s, t) { s.compT = t; s.sumComp = seg(t, 0.75, 0.95); } },

      { title: 'Order does not matter', duration: 3,
        narration: 'Walk ' + wS + ' first and then ' + vS + ': you land on the same corner. The two routes form a parallelogram with the sum as its diagonal — the <b>parallelogram rule</b>. ' + vS + ' + ' + wS + ' = ' + wS + ' + ' + vS + '.',
        setup(s) { s.compT = 0; s.sumComp = 0; s.ghostW = 1; s.route2 = 0; s.para = 0; s.panel = 'comm'; },
        at(s, t) { s.route2 = seg(t, 0, 0.6); s.para = seg(t, 0.6, 0.9); } },

      { title: 'Lengths do not add', duration: 4, misconception: true,
        narration: 'Belief: “adding vectors adds their lengths.” ‖' + vS + '‖ ≈ 3.16 and ‖' + wS + '‖ ≈ 2.24, so the sum should be 5.40 long — but ‖v + w‖ = 5 exactly. Lengths add only when the arrows point the same way. Direction matters.',
        setup(s) { s.route2 = 0; s.para = 0; s.ghostW = 0; s.slide = 1; s.sumT = 1; s.lenBar = 0; s.cross = false; s.panel = 'len'; },
        at(s, t) { s.lenBar = seg(t, 0.15, 0.6); s.cross = t >= 0.8; } },

      { title: 'Scaling: stretch, shrink, flip', duration: 5, interactive: true,
        narration: 'Multiply ' + vS + ' by a <b>scalar</b> c. <b>Scaling</b> multiplies every component, so the arrow stays on its line. c = 2 doubles it; c = 0.5 halves it; c = 0 collapses it to the origin; c = −1 flips it to point the opposite way. Drag the yellow tip along its line.',
        setup(s) { s.lenBar = 0; s.cross = false; s.sumT = 0; s.wA = 0; s.vA = 0; s.slide = 0; s.showScaled = true; s.c = 1; s._touchC = false; s.dragC = true; s.panel = 'scale'; },
        at(s, t) {
          LA.cam.tween(s, { center: cam2().center, halfH: cam2().halfH }, { center: camWide().center, halfH: camWide().halfH }, seg(t, 0, 0.15));
          if (s._touchC) return;
          const k = [[0, 0.2, 1, 2], [0.25, 0.45, 2, 0.5], [0.5, 0.68, 0.5, 0], [0.73, 0.93, 0, -1]];
          let c = 1;
          for (const [a, b, c0, c1] of k) { if (t >= a) c = LA.lerp(c0, c1, seg(t, a, b)); }
          s.c = c;
        } },

      { title: 'Subtraction is an arrow between tips', duration: 4,
        narration: vS + ' − ' + wS + ' means v + (−w): flip w and add. This <b>subtraction</b> gives [2; −1], the arrow from the tip of w to the tip of v. “From w to v” — it points at the vector written first. Draw it from the origin when you need its components.',
        setup(s) { s.cam = cam2(); s.showScaled = false; s.dragC = false; s.c = 1; s.vA = 1; s.wA = 1; s.slide = 0; s.sumT = 0; s.diffT = 0; s.panel = 'diff'; },
        at(s, t) { s.diffT = t; } },

      { title: 'The midpoint is a vector', duration: 3,
        narration: 'Add ' + vS + ' and ' + wS + ' and scale by ½: the result [2; 1.5] lands exactly halfway between the two tips. The <b>mean</b> of vectors is a vector, found by adding and dividing. Averaging a thousand data points works the same way, entry by entry.',
        setup(s) { s.diffT = 0; s.slide = 0; s.sumT = 1; s.sumScale = 1; s.midT = 0; s.panel = 'mid'; },
        at(s, t) { s.sumScale = 1 - 0.5 * seg(t, 0.1, 0.6); s.midT = seg(t, 0.5, 0.8); } },

      { title: 'A learning step is “walk along an arrow”', duration: 5, mlTie: true,
        narration: 'Training a model updates its weight vector ' + wS + ' by subtracting a small multiple of another vector, the gradient ∇L, which points where the error L grows fastest: w ← w − η∇L. That is scaling (by the step size η) and subtraction (walk against the gradient). Every optimizer in machine learning is this picture, repeated, in millions of dimensions.',
        setup(s) { s.midT = 0; s.sumT = 0; s.sumScale = 1; s.showML = true; s.mlT = 0; s.panel = 'ml'; },
        at(s, t) { s.mlT = t; } },

      { title: 'Recap', duration: 3, recap: true,
        narration: 'Adding: tip to tail, or add matching components; the order does not matter. Scaling: multiply every component; negative scalars flip. Subtracting: v − w points from w’s tip to v’s tip. Lengths do not add unless directions agree. A training step is one scale-and-subtract.',
        cards: ['<b>Add tip to tail, or entry by entry.</b> v + w is the diagonal of the parallelogram; order does not matter.', '<b>Scale: stretch, shrink, flip through 0.</b> c·v stays on the line through v; c = −1 gives −v.', '<b>w ← w − η∇L is a scaled subtraction.</b> Every training step walks a little against the gradient arrow.'],
        setup(s) { s.showML = false; s.recap = true; s.panel = 'none'; },
        at(s, t) { } },
    ],

    playground: {
      intro: '<p>Drag the tips of <span class="c-v">v</span> and <span class="c-w">w</span>, or type them. The sum, c·v and the length readouts update live. Try a preset where the lengths do add.</p>',
      setup(s) { s.cam = camWide(); s.v = V0.slice(); s.w = W0.slice(); s.slide = 0; s.showSum = true; s.ghostW = 0; s.showScaled = true; s.c = 1; s.readouts = true; s.dragV = true; s.dragW = true; s.panel = 'play'; },
      controls: [
        { type: 'matrix', bind: 'v', label: 'v (drag tip)', colors: 'rows', step: 0.1, min: -4, max: 4 },
        { type: 'matrix', bind: 'w', label: 'w (drag tip)', colors: 'rows', step: 0.1, min: -4, max: 4 },
        { type: 'slider', bind: 'c', min: -3, max: 3, step: 0.05, label: 'scalar c', color: 'v', fmt: x => fmt(x) },
        { type: 'toggle', bind: 'showParallelogram', label: 'Parallelogram' },
        { type: 'toggle', bind: 'showDiff', label: 'Show v − w' },
        { type: 'preset', label: 'Presets', items: [
          { label: 'Same direction (lengths add)', values: { v: [2, 1], w: [2, 1] } },
          { label: 'Perpendicular', values: { v: [3, 0], w: [0, 2] } },
          { label: 'Opposite', values: { v: [3, 1], w: [-1.5, -0.5] } },
        ] },
      ],
    },

    checks: [
      { kind: 'manipulate',
        prompt: 'Drag the tip of the purple ' + wS + ' so that ' + vS + ' + ' + wS + ' = [1; 4]. The yellow ' + vS + ' stays fixed.',
        setup(s) { s.v = V0.slice(); s.w = W0.slice(); s.slide = 1; s.showSum = true; s.ghostW = 0; s.dragW = true; s.panel = 'check1'; },
        judge(s) { return LA.judge.near(N.add(s.v, s.w), [1, 4], LA.TOL.pos); },
        hint: 'The sum’s green component must drop from 4 to 1: move ' + wS + ' 3 to the left (and 1 up).',
        solution(s) { s.w = [-2, 3]; } },
      { kind: 'manipulate',
        prompt: 'Drag the yellow tip of c·' + vS + ' along its dashed line until it lands on the tip of the gray target arrow. The readout c = … in the corner shows the scalar you have chosen.',
        setup(s) { s.cam = camWide(); s.v = V0.slice(); s.wA = 0; s.vA = 0; s.showScaled = true; s.c = 1; s.dragC = true; s.target = [-4.5, -1.5]; s.panel = 'check2'; },
        judge(s) { return LA.judge.scalar(s.c, -1.5, LA.TOL.scalar); },
        hint: 'The target points the opposite way and is half again as long: c is negative.',
        solution(s) { s.c = -1.5; } },
      { kind: 'manipulate',
        prompt: 'Drag the tip of the gray arrow g so that it equals ' + vS + ' − ' + wS + '.',
        setup(s) { s.v = V0.slice(); s.w = W0.slice(); s.slide = 0; s.gray = [0.5, 0.5]; s.dragGray = true; s.panel = 'check3'; },
        judge(s) { return LA.judge.near(s.gray, N.sub(s.v, s.w), LA.TOL.pos); },
        hint: 'Subtract matching components: 3 − 1 and 1 − 2.',
        solution(s) { s.gray = [2, -1]; } },
    ],
  });

  /* ---------------- pieces of the 2D scene ---------------- */
  function renderScaled(g, s) {
    const v = s.v, cv = N.scale(v, s.c);
    const near = (x, y) => Math.abs(x - y) < 0.03;
    g.lineThrough([0, 0], v, { color: 'v', dashed: true, width: 1.2, alpha: 0.35, len: 20 });
    g.hud('c = ' + fmt(s.c), 14, s.readouts ? 64 : 22, { size: 15, color: 'v' });
    if (s.vA > 0.01) { if (near(s.c, 1)) return; }        // playground: c·v = v is already drawn
    else g.vector([0, 0], v, { color: 'v', width: 2.2, alpha: 0.3, label: 'v', labelAt: 0.5, labelOffset: -14 });
    const lab = near(s.c, -1) ? '−v' : near(s.c, 0) ? '' : 'c·v';
    if (Math.abs(s.c) > 0.03) g.vector([0, 0], cv, { color: 'v', width: 3.4, alpha: s.vA > 0.01 ? 0.7 : 1, dashed: s.vA > 0.01, label: lab, labelOffset: 16 });
    else { g.point([0, 0], { r: 6, color: 'v' }); g.label('0·v = 0', [0, 0], { color: 'v', dx: -40, dy: -18, size: 14, font: 'math' }); }
    if (s.vA > 0.01) return;
    const side = cv[0] >= 0 ? 1 : -1;
    bracket(g, cv, cv, ROW, { dx: -40 * side, dy: side > 0 ? -40 : 40, size: 14 });
  }

  function renderComp(g, s) {
    const v = s.v, w = s.w, sum = N.add(v, w), t = s.compT;
    const S = g.size;
    // the two tip brackets slide into one equation in the empty upper-left corner
    const u = seg(t, 0, 0.3, 'inOut');
    const ex = 70, ey = 74;
    const pv0 = N.add(g.P.toScreen(v), [50, 4]), pw0 = N.add(g.P.toScreen(N.add(v, w)), [50, -6]);
    const pv = lerpV(pv0, [ex, ey], u), pw = lerpV(pw0, [ex + 92, ey], u);
    const hiTop = seg(t, 0.3, 0.42), hiBot = seg(t, 0.55, 0.67);
    const cv = bracket(g, pv, v, ROW, { screen: true });
    const cw = bracket(g, pw, w, ROW, { screen: true });
    if (u > 0.95) {
      g.label('+', [ex + 46, ey], { screen: true, size: 20, color: 'hud', font: 'math' });
      g.label('=', [ex + 138, ey], { screen: true, size: 20, color: 'hud', font: 'math' });
      bracket(g, [ex + 184, ey], sum, ROW, { screen: true, entryAlpha: [hiTop, hiBot] });
      const ring = (p, c, a) => { if (a > 0.05) g.marker.ring(g.P.toWorld(p[0], p[1]), { r: 15, color: c }); };
      const topA = seg.pulse(t, 0.28, 0.5), botA = seg.pulse(t, 0.53, 0.75);
      ring(cv[0], 'i', topA); ring(cw[0], 'i', topA); ring(cv[1], 'j', botA); ring(cw[1], 'j', botA);
      if (hiTop > 0.5) g.label('3 + 1 = 4', [ex + 250, ey - 9], { screen: true, size: 13, color: 'i', font: 'ui', anchor: 'left' });
      if (hiBot > 0.5) g.label('1 + 2 = 3', [ex + 250, ey + 10], { screen: true, size: 13, color: 'j', font: 'ui', anchor: 'left' });
    }
    if (S && t > 0.75) g.label('matching entries add — nothing crosses rows', [ex - 30, ey + 46], { screen: true, size: 12.5, color: 'muted', font: 'ui', anchor: 'left' });
  }

  function renderLengths(g, s) {
    const v = s.v, w = s.w, sum = N.add(v, w);
    const nv = N.norm(v), nw = N.norm(w), ns = N.norm(sum), d = N.normalize(sum);
    const perp = [-d[1], d[0]], off = N.scale(perp, 0.38);
    const L = (nv + nw) * s.lenBar;
    const at = x => N.add(off, N.scale(d, x));
    // gray bar = ‖v‖ + ‖w‖ laid along the sum's direction (beside it), overshooting the tip
    g.line(at(0), at(Math.min(L, nv)), { color: 'residual', width: 7, alpha: 0.75 });
    if (L > nv) g.line(at(nv), at(Math.min(L, ns)), { color: 'residual', width: 7, alpha: 0.55 });
    if (L > ns) g.line(at(ns), at(L), { color: 'bad', width: 7, alpha: 0.8 });
    if (L > nv) g.line(N.add(at(nv), N.scale(perp, -0.12)), N.add(at(nv), N.scale(perp, 0.12)), { color: 'text', width: 1.5 });
    if (s.lenBar > 0.98) {
      g.dim(at(0), at(nv), { label: '3.16', color: 'v', offset: -14 });
      g.dim(at(nv), at(nv + nw), { label: '2.24', color: 'w', offset: -14 });
      g.dim([0, 0], sum, { label: '5.00', color: 'text', offset: 14 });
      g.line(sum, at(ns), { color: 'text', dashed: [3, 3], width: 1.2 });
      g.label('overshoots by 0.40', at(nv + nw), { color: 'bad', dx: -8, dy: -16, size: 12.5, font: 'ui', anchor: 'right' });
    }
    if (s.cross) {
      const x = 196, y = 34;
      g.label('“lengths add”: 3.16 + 2.24 = 5.40', [x, y], { screen: true, size: 15, color: 'bad', font: 'ui' });
      g.marker.cross([x + 150, y], { screen: true, size: 22 });
      g.label('‖v + w‖ = 5 — direction matters', [x, y + 24], { screen: true, size: 13, color: 'ok', font: 'ui' });
    }
  }

  function renderDiff(g, s) {
    const v = s.v, w = s.w, d = N.sub(v, w), t = s.diffT;
    const nwA = seg(t, 0, 0.15), sl = seg(t, 0.18, 0.42), gr = seg(t, 0.42, 0.6, 'out'), mv = seg(t, 0.62, 0.85);
    // −w appears flipped at the origin, then slides to v's tip
    const o = N.scale(v, sl);
    if (nwA > 0.01) g.vector(o, N.sub(o, w), { color: 'w', width: 2.4, alpha: 0.55 * nwA, label: '−w', labelAt: 0.5, labelOffset: 14 });
    // the result from the origin
    if (gr > 0.01) {
      g.vector([0, 0], N.scale(d, gr), { color: 'b', width: 3, alpha: mv > 0 ? 1 - 0.55 * mv : 1, label: gr > 0.6 ? 'v − w' : '', labelAt: 0.45, labelOffset: -18 });
      if (gr > 0.9) bracket(g, d, d, ROW, { dx: 10, dy: 34, size: 13 });
    }
    // a copy rides to w's tip and ends at v's tip
    if (mv > 0.01) {
      const st = N.scale(w, mv);
      g.vector(st, N.add(st, d), { color: 'b', width: 3.2, label: 'v − w', labelAt: 0.5, labelOffset: 16 });
    }
    if (t > 0.86) {
      const dir = N.normalize(N.sub(w, v)), pp = N.scale([-dir[1], dir[0]], -0.6);
      const a = N.add(v, pp), b = N.add(w, pp);
      g.vector(a, b, { color: 'residual', width: 2.2, alpha: 0.8, label: 'not v − w', labelAt: 0.5, labelOffset: -18 });
      g.marker.cross(lerpV(a, b, 0.5), { dx: 30, dy: -40, size: 18 });
      g.label('v − w runs from the tip of w to the tip of v', [g.size.w / 2, 34], { screen: true, color: 'b', size: 13.5, font: 'ui' });
    }
  }

  function renderML(g, s) {
    const t = s.mlT, m = M_ML;
    // faint level curves of the loss (ellipses) and the minimum
    [0.4, 0.9, 1.6, 2.4].forEach(r => g.ellipse([[r / Math.sqrt(A_ML[0]), 0], [0, r / Math.sqrt(A_ML[1])]], m, { stroke: 'grid', width: 1.2, strokeAlpha: 0.9 }));
    g.point(m, { r: 7, color: 'gold' });
    g.label('minimum', m, { color: 'gold', dy: 22, size: 13, font: 'ui' });
    const w0 = PATH[0], w1 = PATH[1], gr = gradAt(w0), step = N.scale(gr, -ETA);
    const a1 = seg(t, 0, 0.2), a2 = seg(t, 0.2, 0.5), a3 = seg(t, 0.5, 0.8), a4 = seg(t, 0.8, 1);
    g.vector([0, 0], w0, { color: 'w', width: 3, alpha: 1 - 0.6 * a3, label: 'w', labelAt: 0.55, labelOffset: -14 });
    if (a1 > 0.01) g.vector(w0, N.add(w0, N.scale(gr, a1)), { color: 'residual', width: 2.2, dashed: true, label: '∇L', labelOffset: 14 });
    if (a2 > 0.01) g.vector(w0, N.add(w0, N.scale(step, a2)), { color: 'b', width: 3, label: a2 > 0.6 ? '−η∇L' : '', labelOffset: 16 });
    if (a3 > 0.01) g.vector([0, 0], lerpV(w0, w1, 1), { color: 'w', width: 3.2, alpha: a3, label: 'new w', labelAt: 0.6, labelOffset: 16 });
    if (a4 > 0.01) {
      for (let k = 1; k < PATH.length - 1; k++) {
        const p = seg(a4, (k - 1) / 3, k / 3);
        if (p <= 0) break;
        g.line(PATH[k], lerpV(PATH[k], PATH[k + 1], p), { color: 'w', width: 2, alpha: 0.6 });
        g.point(lerpV(PATH[k], PATH[k + 1], p), { r: 3.5, color: 'w', alpha: 0.7 });
      }
      g.line(PATH[PATH.length - 1], m, { color: 'w', dashed: [3, 4], width: 1.2, alpha: 0.4 * a4 });
    }
    g.point(w0, { r: 3.5, color: 'w', alpha: 0.8 });
    g.hud('w ← w − η∇L   (η = 0.5)', 14, 22, { size: 15, color: 'w' });
    g.hud('Part 5: gradient descent', 14, 44, { size: 13, color: 'gold' });
  }

  /* ---------------- recap cards ---------------- */
  function renderRecap(g) {
    const S = g.size, m = 10, h = Math.floor((S.h - 4 * m) / 3), w = S.w - 2 * m;
    const row = i => ({ x: m, y: m + i * (h + m), w, h });
    const title = (rr, txt) => g.label(txt, [rr.x + 10, rr.y + 14], { screen: true, anchor: 'left', size: 13, color: 'hud', font: 'ui', bold: true });
    g.viewport(row(0), { view: '2d', cam: { center: [2.2, 1.5], halfH: 2.1 }, bg: 'panel' }, (g2, rr) => {
      g2.ghostGrid({ range: 8 });
      g2.parallelogram(V0, W0, { fill: 'v', alpha: 0.12, stroke: null });
      g2.vector([0, 0], W0, { color: 'w', width: 2, alpha: 0.35 });
      g2.vector([0, 0], V0, { color: 'v', width: 3, label: 'v', labelAt: 0.55, labelOffset: 12 });
      g2.vector(V0, N.add(V0, W0), { color: 'w', width: 3, label: 'w', labelAt: 0.55, labelOffset: -12 });
      sumArrow(g2, [0, 0], N.add(V0, W0), { label: 'v + w' });
      bracket(g2, [4, 3], [4, 3], ROW, { dx: 46, dy: 6, size: 12 });
      title(rr, '1 · Add tip to tail, or entry by entry');
    });
    g.viewport(row(1), { view: '2d', cam: { center: [0.6, 0.2], halfH: 2.1 }, bg: 'panel' }, (g2, rr) => {
      g2.ghostGrid({ range: 10 });
      g2.lineThrough([0, 0], V0, { color: 'v', dashed: true, width: 1.2, alpha: 0.4, len: 20 });
      g2.vector([0, 0], N.scale(V0, 1.6), { color: 'v', width: 2.2, alpha: 0.45, label: '1.6v', labelOffset: 14 });
      g2.vector([0, 0], V0, { color: 'v', width: 3, label: 'v', labelOffset: -14 });
      g2.vector([0, 0], N.scale(V0, -1), { color: 'v', width: 3, label: '−v', labelOffset: 14 });
      g2.point([0, 0], { r: 4, color: 'v' });
      title(rr, '2 · Scale: stretch, shrink, flip through 0');
    });
    g.viewport(row(2), { view: '2d', cam: { center: [1.6, 0.9], halfH: 1.75 }, bg: 'panel' }, (g2, rr) => {
      g2.ghostGrid({ range: 8 });
      [0.4, 0.9, 1.6].forEach(r => g2.ellipse([[r / Math.sqrt(A_ML[0]), 0], [0, r / Math.sqrt(A_ML[1])]], M_ML, { stroke: 'grid', width: 1 }));
      g2.point(M_ML, { r: 5, color: 'gold' });
      const w0 = PATH[0], gr = gradAt(w0);
      g2.vector(w0, N.add(w0, gr), { color: 'residual', width: 2, dashed: true, label: '∇L', labelOffset: 12 });
      g2.vector(w0, PATH[1], { color: 'b', width: 2.6, label: '−η∇L', labelOffset: 14 });
      g2.vector([0, 0], w0, { color: 'w', width: 2.4, alpha: 0.4, label: 'w', labelAt: 0.5, labelOffset: -12 });
      g2.vector([0, 0], PATH[1], { color: 'w', width: 3, label: 'new w', labelAt: 0.6, labelOffset: 14 });
      title(rr, '3 · w ← w − η∇L is a scaled subtraction');
    });
  }
})();
