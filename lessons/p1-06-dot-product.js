/* p1-06 — Dot product: one number, three faces (sum of products, shadow × length, ‖v‖‖w‖cos θ), and cosine similarity. */
(function () {
  'use strict';
  const N = LA.num, seg = LA.seg, lerpV = LA.lerpV;
  const V0 = [3, 1], W0 = [1, 2], V3 = [2, 0, 1], W3 = [1, 2, 1];
  const TEMPL = [2, 1], INPUTS = [[2, 1.2], [1, -1], [-1.5, 0.5], [0.5, 2]];
  const CV = [2, -1, 3], CW = [1, 4, -2];                      // entry check: v·w = −8
  const CAM_A = { center: [1.5, 1.15], halfH: 2.5 };           // the shadow picture (x −0.6..3.6)
  const CAM_ROT = { center: [0.6, 0.55], halfH: 3.6 };         // room for w to turn a full circle under the score strip
  const CAM_WIDE = { center: [2.9, 1.5], halfH: 4.1 };         // v doubled
  const CAM_P = { center: [0.7, 0.6], halfH: 3.9 };            // playground / checks
  const CAM_STRIP = { center: [0, -0.15], halfH: 3.3 };
  const CAM_ENTRY = { center: [0, 0.2], halfH: 2.7 };
  const CAM3 = { target: [1.0, 0.7, 0.5], dist: 7.2, yaw: -0.2, pitch: 0.7 };
  /* Numbers in the math panel. The engine updates spans whose class starts with "tx-v" in place, but matrix cells
     also carry that class, so tex.num next to tex.vec writes numbers into the wrong cells. A plain tabular cell
     avoids that: a changed number rebuilds the (small) panel. */
  const nm = (x, color, digits) => '<span class="tx-cell ' + LA.tex.colorClass(color || '') + '">' + LA.fmt(Math.abs(x) < 0.005 ? 0 : x, digits) + '</span>';
  const vS = '<span class="c-v">v</span>', wS = '<span class="c-w">w</span>', xS = '<span class="c-v">x</span>';
  const cam2 = c => Object.assign(LA.cam.defaults2d(), LA.clone(c));
  const projOnto = (a, b) => { const bb = N.dot(b, b); return bb < 1e-12 ? b.map(() => 0) : N.scale(b, N.dot(a, b) / bb); };
  const cosOf = (a, b) => { const d = N.norm(a) * N.norm(b); return d < 1e-9 ? 0 : N.dot(a, b) / d; };
  const degOf = (a, b) => Math.acos(LA.clamp(cosOf(a, b), -1, 1)) * 180 / Math.PI;
  const SUBS = ['₁', '₂', '₃', '₄'];
  /** full turn of w with short holds at the two right angles (w starts 45° from v, so they come at +45° and +225°) */
  function turnAngle(u) {
    const q = Math.PI / 4, K = [[0, 0], [0.14, q], [0.24, q], [0.6, 5 * q], [0.7, 5 * q], [1, 8 * q]];
    for (let i = 1; i < K.length; i++) if (u <= K[i][0]) return LA.lerp(K[i - 1][1], K[i][1], LA.ease.inOut ? LA.ease.inOut((u - K[i - 1][0]) / (K[i][0] - K[i - 1][0])) : (u - K[i - 1][0]) / (K[i][0] - K[i - 1][0]));
    return 8 * q;
  }

  function makeStrips() {
    const rand = LA.rng(6); const a = [], b = [];
    for (let i = 0; i < 12; i++) a.push(LA.rng.int(rand, -3, 3));
    for (let i = 0; i < 12; i++) b.push(LA.rng.int(rand, -3, 3));
    return { a, b };
  }
  /** the vectors as drawn: v scaled by scaleV, both shrunk toward unit length by unitT (or the Unit toggle) */
  function eff(s) {
    let v = N.scale(s.v, s.scaleV), w = s.w.slice();
    const u = s.unit ? 1 : s.unitT;
    if (u > 0) {
      const nv = N.norm(v), nw = N.norm(w);
      if (nv > 1e-9) v = N.scale(v, LA.lerp(1, 1 / nv, u));
      if (nw > 1e-9) w = N.scale(w, LA.lerp(1, 1 / nw, u));
    }
    return { v, w, u };
  }
  /** small right-angle mark at P between directions d1 and d2 (2D or 3D) */
  function rightMark(g, P, d1, d2, size, color) {
    const a = N.scale(N.normalize(d1), size), b = N.scale(N.normalize(d2), size);
    g.polyline([N.add(P, a), N.add(N.add(P, a), b), N.add(P, b)], { color: color || 'hud', width: 1.3 });
  }

  function base(s) {
    Object.assign(s, {
      mode: 'main', view: '2d', grow: 1, arc: false, arcLabel: false, showList: false, listT: 0,
      shadowT: 0, segT: 0, dims: false, swap: 0, swapOn: false, showBar: false, showCos: false, barPulse: 0, barNote: '',
      mis: 0, cross: false, unitT: 0, unit: false, ghostV: false, ghostW: false, unitCircle: false, cosLabel: false,
      scaleV: 1, projT: 0, projResid: false, note: '', rot: false, wNorm: false, drag: [],
      d3T: 0, seg3: 0, dims3: false, stripT: 0, mlT: 0,
    });
  }

  LA.registerLesson({
    id: 'p1-06-dot-product',
    part: 1,
    title: 'Dot product: shadow, angle, sum of products, cosine similarity',
    subtitle: 'One number for two arrows: multiply-and-add, shadow times length, ‖v‖‖w‖cos θ.',
    minutes: 11,
    prereqs: ['p1-05-norms-distance', 'p0-03-angles-and-circles'],
    tags: ['core'],
    badge: { drawnIn: '2D and 3D', realSize: 'a neuron takes the dot product of a 784-strip with a 784-strip; attention scores are dot products of 64–128 entry queries and keys; embedding search compares 768-entry vectors', transfers: 'Σvᵢwᵢ is the same sum with more terms; “shadow times length” and the cosine formula remain exactly true in ℝⁿ' },
    cantShow: ['the angle between two 768-component vectors (it is defined by the formula and computed, never seen)'],
    terms: [
      { term: 'dot product', def: 'One number from two vectors: multiply matching components and add; it equals the shadow of one vector on the other times that other’s length.' },
      { term: 'shadow', def: 'How far along one vector’s line the tip of the other falls when dropped perpendicularly (the projection length).' },
      { term: 'perpendicular', def: 'At a right angle (also called orthogonal); the dot product is zero.' },
      { term: 'cosine similarity', def: 'The dot product divided by both lengths: the cosine of the angle, from −1 to 1.' },
      { term: 'projection', def: 'The shadow drawn as a vector along the other vector’s line.' },
    ],
    atlas: ['mlp', 'attention', 'embeddings'],
    thumbnail: 2,

    state() {
      const st = makeStrips();
      const s = { view: '2d', cam: cam2(CAM_A), v: V0.slice(), w: W0.slice(), v3: V3.slice(), w3: W3.slice(), sA: st.a, sB: st.b, n: 10, guess: 0, panel: 'vectors' };
      base(s);
      return s;
    },

    render(g, s, ctx) {
      if (s.mode === '3d') { render3d(g, s); return; }
      if (s.mode === 'strip') { renderStrip(g, s, ctx); return; }
      if (s.mode === 'ml') { renderMl(g, s); return; }
      if (s.mode === 'entry') { renderEntry(g, s); return; }
      renderMain(g, s, ctx);
    },

    math(s, tex) {
      const sv = tex.sym('v', { color: 'v' }), sw = tex.sym('w', { color: 'w' }), dot = sv + tex.op('·') + sw;
      const nv = tex.norm(sv), nwn = tex.norm(sw), E = eff(s), d = N.dot(E.v, E.w);
      switch (s.panel) {
        case 'vectors': return tex.block(tex.eq(sv, tex.op('='), tex.vec(s.v, { color: 'v' }), tex.op('   '), sw, tex.op('='), tex.vec(s.w, { color: 'w' })));
        case 'sum': return tex.block(tex.eq(dot, tex.op('='), nm(3, 'v'), tex.op('·'), nm(1, 'w'), tex.op('+'), nm(1, 'v'), tex.op('·'), nm(2, 'w'), tex.op('='), nm(3, 'i'), tex.op('+'), nm(2, 'j'), tex.op('='), nm(5, 'gold')));
        case 'shadow': return tex.block(tex.eq(dot, tex.op('='), tex.text('(shadow of w on v)', { color: 'mag' }), tex.op('×'), nv, tex.op('='), nm(d / N.norm(E.v), 'mag'), tex.op('×'), nm(N.norm(E.v), 'v'), tex.op('='), nm(d, 'gold')));
        case 'swap': return tex.block(tex.eq(sw + tex.op('·') + sv, tex.op('='), nm(d / N.norm(E.w), 'mag'), tex.op('×'), nm(N.norm(E.w), 'w'), tex.op('='), nm(d, 'gold'), tex.op('='), dot));
        case 'sign': {
          const perp = Math.abs(d) < 0.05;
          return tex.block(tex.eq(dot, tex.op('='), nm(d, d < -0.05 ? 'neg' : 'gold')), perp ? tex.eq(tex.op('   '), dot, tex.op('='), nm(0), tex.op('⇔'), tex.text('perpendicular', { color: 'gold' })) : '');
        }
        case 'mis': return tex.block(tex.eq(dot, tex.op('='), nm(5, 'gold'), tex.op('∈'), tex.text('ℝ'), tex.text('  (a number, not a vector)', { color: 'muted' })));
        case 'cos': return tex.block(tex.eq(dot, tex.op('='), nv, nwn, tex.text('cos θ'), tex.op('='), nm(N.norm(s.v), 'v'), tex.op('×'), nm(N.norm(s.w), 'w'), tex.op('×'), nm(cosOf(s.v, s.w), 'mag'), tex.op('='), nm(N.dot(s.v, s.w), 'gold')));
        case 'cossim': return tex.block(tex.eq(tex.text('cos θ'), tex.op('='), tex.frac(dot, nv + nwn), tex.op('='), tex.frac(nm(d, 'gold'), nm(N.norm(E.v), 'v') + tex.op('×') + nm(N.norm(E.w), 'w')), tex.op('='), nm(cosOf(E.v, E.w), 'teal')));
        case 'proj': return tex.block(tex.eq(tex.sym('proj', { sub: 'v' }), tex.paren(sw), tex.op('='), tex.paren(tex.frac(dot, sv + tex.op('·') + sv)), tex.op('·'), sv, tex.op('='), tex.paren(tex.frac(nm(5), nm(10))), tex.op('·'), tex.vec(V0, { color: 'v' }), tex.op('='), tex.vec(projOnto(W0, V0), { color: 'mag' })));
        case '3d': return tex.block(tex.eq(dot, tex.op('='), nm(2, 'v'), tex.op('·'), nm(1, 'w'), tex.op('+'), nm(0, 'v'), tex.op('·'), nm(2, 'w'), tex.op('+'), nm(1, 'v'), tex.op('·'), nm(1, 'w'), tex.op('='), nm(3, 'gold'), tex.op('='), nm(N.dot(V3, W3) / N.norm(V3), 'mag'), tex.op('×'), nm(N.norm(V3), 'v')));
        case 'strip': {
          const n = Math.round(s.n), a = s.sA.slice(0, n), b = s.sB.slice(0, n), D = N.dot(a, b);
          return tex.lines(
            tex.eq(dot, tex.op('='), tex.sum(tex.sym('v', { sub: 'i', color: 'v' }) + tex.sym('w', { sub: 'i', color: 'w' }), 'i'), tex.op('='), nm(D, 'gold')),
            tex.eq(tex.text('cos θ'), tex.op('='), tex.frac(dot, nv + nwn), tex.op('='), nm(cosOf(a, b), 'teal')));
        }
        case 'ml': return tex.lines(
          tex.eq(tex.text('score'), tex.op('='), sw + tex.op('·') + tex.sym('x', { color: 'v' }), tex.op(':'), INPUTS.map(x => nm(N.dot(TEMPL, x), 'gold')).join(tex.op(','))),
          tex.eq(tex.op('['), [1, 2, 3, 4].map(i => tex.sym('q') + tex.op('·') + tex.sym('k', { sub: String(i) })).join(tex.op(',')), tex.op(']'), tex.text('  one row of attention scores', { color: 'muted' })));
        case 'entry': return tex.block(tex.eq(dot, tex.op('='), nm(CV[0], 'v'), tex.op('·'), nm(CW[0], 'w'), tex.op('+'), tex.paren(nm(CV[1], 'v')), tex.op('·'), nm(CW[1], 'w'), tex.op('+'), nm(CV[2], 'v'), tex.op('·'), tex.paren(nm(CW[2], 'w')), tex.op('='), tex.text('?', { color: 'b' })));
        case 'play': {
          if (s.mode === 'strip') { const n = Math.round(s.n), a = s.sA.slice(0, n), b = s.sB.slice(0, n); return tex.lines(tex.eq(tex.text('strips, n = ' + n + ':  '), dot, tex.op('='), tex.sum(tex.sym('v', { sub: 'i', color: 'v' }) + tex.sym('w', { sub: 'i', color: 'w' }), 'i'), tex.op('='), nm(N.dot(a, b), 'gold')), tex.eq(tex.text('cos θ'), tex.op('='), nm(cosOf(a, b), 'teal'))); }
          return tex.lines(
            tex.eq(dot, tex.op('='), nm(E.v[0], 'v'), tex.op('·'), nm(E.w[0], 'w'), tex.op('+'), nm(E.v[1], 'v'), tex.op('·'), nm(E.w[1], 'w'), tex.op('='), nm(d, d < -0.05 ? 'neg' : 'gold')),
            tex.eq(tex.text('cos θ'), tex.op('='), tex.frac(dot, nv + nwn), tex.op('='), nm(cosOf(E.v, E.w), 'teal'), tex.op('   '), tex.text('θ'), tex.op('='), nm(Math.round(degOf(E.v, E.w))) + tex.text('°')));
        }
        default: return '';
      }
    },

    steps: [
      { title: 'One number for two arrows', duration: 2.5,
        narration: 'Two vectors: the yellow ' + vS + ' = [3; 1] and the purple ' + wS + ' = [1; 2]. The <b>dot product</b> is a single number that answers one question: how much do they point the same way? We will get that number in three different ways and watch them agree.',
        setup(s) { base(s); s.cam = cam2(CAM_A); s.v = V0.slice(); s.w = W0.slice(); s.grow = 0; s.panel = 'vectors'; },
        at(s, t) { s.grow = seg(t, 0.05, 0.6, 'back'); s.arc = t > 0.6; } },

      { title: 'Multiply matching components, add', duration: 4,
        narration: 'The list face first. Multiply the green components, 3·1 = 3. Multiply the red ones, 1·2 = 2. Add: 3 + 2 = 5. Written out, ' + vS + '·' + wS + ' = 3·1 + 1·2 = 5. Nothing crosses rows; it is the simplest possible recipe for a number.',
        setup(s) { base(s); s.cam = cam2(CAM_A); s.v = V0.slice(); s.w = W0.slice(); s.arc = true; s.showList = true; s.listT = 0; s.panel = 'sum'; },
        at(s, t) { s.listT = t; } },

      { title: 'The shadow', duration: 5,
        narration: 'Now the picture. From ' + wS + '’s tip, drop straight down at a right angle onto the line of ' + vS + '; the magenta segment it lands on is w’s <b>shadow</b>. Its length is 1.58, and v’s length is 3.16. Multiply: 1.58 × 3.16 = 5. Same number. The dot product is shadow length times length.',
        setup(s) { base(s); s.cam = cam2(CAM_A); s.v = V0.slice(); s.w = W0.slice(); s.panel = 'shadow'; },
        at(s, t) { s.shadowT = seg(t, 0.05, 0.4); s.segT = seg(t, 0.4, 0.7); s.dims = t > 0.7; s.note = t > 0.8 ? 'shadow × length: 1.58 × 3.16 = 5' : ''; } },

      { title: 'Either one can cast the shadow', duration: 3.5,
        narration: 'Swap roles: let ' + vS + ' cast its shadow on ' + wS + '. The shadow is 2.24 long and ‖w‖ = 2.24; the product is again 5. It does not matter who casts the shadow: v·w = w·v.',
        setup(s) { base(s); s.cam = cam2(CAM_A); s.v = V0.slice(); s.w = W0.slice(); s.shadowT = 1; s.segT = 1; s.dims = true; s.panel = 'swap'; },
        at(s, t) { s.swap = seg(t, 0.1, 0.7, 'inOut'); s.note = t > 0.78 ? 'shadow × length: 2.24 × 2.24 = 5' : ''; } },

      { title: 'The sign tells the direction', duration: 6,
        narration: 'Rotate ' + wS + ' a full turn and watch the number. Same general direction: positive. At a right angle the shadow collapses to a dot and the dot product is 0: the vectors are <b>perpendicular</b>. Past that, the shadow falls behind the origin and the number goes negative. Opposite directions give the most negative value.',
        setup(s) { base(s); s.cam = cam2(CAM_A); s.v = V0.slice(); s.w = W0.slice(); s.shadowT = 1; s.segT = 1; s.showBar = true; s.rot = true; s.panel = 'sign'; },
        at(s, t) { s.w = N.matVec(N.rot2(turnAngle(seg(t, 0.05, 0.94, 'linear'))), W0); LA.cam.tween(s, CAM_A, CAM_ROT, seg(t, 0, 0.12)); } },

      { title: 'The dot product is not an arrow', duration: 3.5, misconception: true,
        narration: 'Belief: "multiplying two vectors gives another vector." Try to draw v·w as an arrow: which direction would it point? There is none. The dot product is a number (a bar, a length, a score), never an arrow. Keep the bar in view whenever you compute it.',
        setup(s) { base(s); s.cam = cam2(CAM_ROT); s.v = V0.slice(); s.w = W0.slice(); s.showBar = true; s.panel = 'mis'; },
        at(s, t) { s.mis = seg(t, 0.02, 0.6, 'inOut'); s.cross = t > 0.7; s.barPulse = t > 0.65 ? 0.5 - 0.5 * Math.cos(seg(t, 0.65, 1, 'linear') * Math.PI * 4) : 0; s.barNote = t > 0.65 ? 'a number' : ''; } },

      { title: 'The angle: v·w = ‖v‖‖w‖cos θ', duration: 5.5,
        narration: 'The two views meet through the unit circle. Shrink both arrows to unit length: w’s shadow on v is now exactly cos θ, the green coordinate from Part 0. Scale the lengths back in and the shadow scales with them. Written out: ' + vS + '·' + wS + ' = ‖v‖‖w‖cos θ. Here θ = 45°, so 3.16 × 2.24 × 0.71 = 5.',
        setup(s) { base(s); s.cam = cam2(CAM_ROT); s.v = V0.slice(); s.w = W0.slice(); s.arc = true; s.showBar = true; s.ghostV = true; s.ghostW = true; s.panel = 'cos'; },
        at(s, t) {
          LA.cam.tween(s, CAM_ROT, CAM_A, seg(t, 0, 0.2));
          s.unitT = seg(t, 0.03, 0.35) * (1 - seg(t, 0.72, 0.95)); s.unitCircle = t > 0.12;
          s.shadowT = seg(t, 0.38, 0.5); s.segT = seg(t, 0.45, 0.62); s.arcLabel = t > 0.4; s.cosLabel = t > 0.5 && t < 0.72;
          s.note = t > 0.8 ? 'v·w = ‖v‖‖w‖cos θ = 3.16 × 2.24 × 0.71 = 5' : '';
        } },

      { title: 'Cosine similarity ignores length', duration: 4,
        narration: 'Divide the lengths out: v·w / (‖v‖‖w‖) = cos θ. This is <b>cosine similarity</b>, a number between −1 and 1 that depends only on directions. Double ' + vS + ': the dot product doubles to 10, but the cosine stays 0.71. When you compare meanings, not sizes, this is the number.',
        setup(s) { base(s); s.cam = cam2(CAM_A); s.v = V0.slice(); s.w = W0.slice(); s.arc = true; s.arcLabel = true; s.showBar = true; s.showCos = true; s.ghostV = true; s.panel = 'cossim'; },
        at(s, t) { s.scaleV = 1 + seg(t, 0.1, 0.65, 'inOut'); LA.cam.tween(s, CAM_A, CAM_WIDE, seg(t, 0, 0.4)); } },

      { title: 'The shadow as a vector: projection', duration: 4.5,
        narration: 'The shadow itself can be a vector along ' + vS + ': the <b>projection</b> of w onto v, proj<sub>v</sub>(w) = (v·w / v·v)·v = 0.5·v = [1.5; 0.5]. If v is a unit vector this is just (v·w)·v. Part 3 builds least squares out of this one arrow.',
        setup(s) { base(s); s.cam = cam2(CAM_WIDE); s.v = V0.slice(); s.w = W0.slice(); s.scaleV = 2; s.showBar = true; s.panel = 'proj'; },
        at(s, t) { s.scaleV = 2 - seg(t, 0, 0.25); LA.cam.tween(s, CAM_WIDE, CAM_A, seg(t, 0, 0.25)); s.shadowT = seg(t, 0.25, 0.4); s.segT = seg(t, 0.3, 0.42); s.projT = seg(t, 0.42, 0.62); s.projResid = t > 0.65; } },

      { title: 'In 3D the shadow falls the same way', duration: 5,
        narration: 'In 3D, ' + vS + ' = [2; 0; 1] and ' + wS + ' = [1; 2; 1]. Sum of products: 2 + 0 + 1 = 3. The perpendicular drop from w’s tip onto v’s line gives a shadow 1.34 long, times ‖v‖ = 2.24, is 3. Orbit: the right angle stays a right angle from every side.',
        setup(s) { base(s); s.mode = '3d'; s.view = '3d'; s.cam = Object.assign(LA.cam.defaults3d(), CAM3); s.panel = '3d'; },
        at(s, t) { s.d3T = seg(t, 0.08, 0.35); s.seg3 = seg(t, 0.3, 0.5); s.dims3 = t > 0.5; LA.cam.tween(s, { yaw: -0.2 }, { yaw: -0.2 - 0.87 }, seg(t, 0, 1, 'linear')); } },

      { title: 'Ten components: strips', duration: 4,
        narration: 'For longer vectors you cannot see the angle, but the formula is identical: multiply matching cells, add. Two 10-cell strips give a product strip and a running sum; divide by the two lengths and you have a cosine similarity, −1 to 1, with the same meaning. The picture stops at 3. The formula does not.',
        setup(s) { base(s); s.mode = 'strip'; s.view = '2d'; s.cam = cam2(CAM_STRIP); s.n = 10; s.stripT = 0; s.panel = 'strip'; },
        at(s, t) { s.stripT = t; } },

      { title: 'Scores: a neuron and an attention row', duration: 6, mlTie: true,
        narration: 'A neuron holds a fixed weight vector ' + wS + ' and scores every input ' + xS + ' by the dot product w·x: a template match, big when x lines up with w. Attention does the same with one query against every key, producing a row of scores. Both are this lesson’s shadow, computed many times.',
        setup(s) { base(s); s.mode = 'ml'; s.view = '2d'; s.cam = cam2(CAM_A); s.mlT = 0; s.panel = 'ml'; },
        at(s, t) { s.mlT = t; } },

      { title: 'Recap', duration: 3, recap: true,
        narration: 'One number, three faces: sum of products; shadow length times length; ‖v‖‖w‖cos θ. Positive means roughly the same way, zero means perpendicular, negative means opposite. Divide the lengths out to get cosine similarity, which compares directions only. Every neuron and every attention score is a dot product.',
        cards: ['<b>Σvᵢwᵢ = shadow × length = ‖v‖‖w‖cos θ.</b> Multiply-and-add, the perpendicular drop, and the angle all give the same 5.', '<b>Sign: same way / perpendicular / opposite.</b> The bar is positive, zero at a right angle, negative past it.', '<b>Cosine similarity = directions only.</b> Doubling v doubles v·w to 10; cos θ stays 0.71.'],
        setup(s) { base(s); s.cam = cam2(CAM_A); s.v = V0.slice(); s.w = W0.slice(); s.shadowT = 1; s.segT = 1; s.arc = true; s.showBar = true; s.showCos = true; s.panel = 'recap'; },
        at(s, t) { } },
    ],

    playground: {
      intro: '<p>Drag the tips of ' + vS + ' and ' + wS + ' (or edit their numbers). The strip on top is v·w as a number; the magenta segment is the shadow. Try a right angle, then point them apart. The strips preset shows the same formula on long lists.</p>',
      setup(s) { base(s); s.cam = cam2(CAM_P); s.v = V0.slice(); s.w = W0.slice(); s.shadowT = 1; s.segT = 1; s.dims = true; s.arc = true; s.arcLabel = true; s.showBar = true; s.showCos = true; s.drag = ['v', 'w']; s.stripT = 1; s.n = 10; s.panel = 'play'; },
      controls: [
        { type: 'matrix', bind: 'v', label: 'v (drag tip)', colors: 'v', step: 0.1, min: -5, max: 5 },
        { type: 'matrix', bind: 'w', label: 'w (drag tip)', colors: 'w', step: 0.1, min: -5, max: 5 },
        { type: 'toggle', bind: 'swapOn', label: 'v casts the shadow (off: w casts)' },
        { type: 'toggle', bind: 'unit', label: 'Unit vectors (then v·w = cos θ)' },
        { type: 'slider', bind: 'n', min: 2, max: 12, step: 1, label: 'Strip mode n', color: 'gold', fmt: x => String(Math.round(x)) },
        { type: 'preset', label: 'Presets', items: [
          { label: 'Perpendicular', apply(s) { s.mode = 'main'; s.cam = cam2(CAM_P); s.v = [3, 1]; s.w = [-1, 3]; } },
          { label: 'Same direction', apply(s) { s.mode = 'main'; s.cam = cam2(CAM_P); s.v = [3, 1]; s.w = [1.5, 0.5]; } },
          { label: 'Opposite', apply(s) { s.mode = 'main'; s.cam = cam2(CAM_P); s.v = [3, 1]; s.w = [-2, -0.67]; } },
          { label: 'Strips n = 10', apply(s) { s.mode = 'strip'; s.cam = cam2(CAM_STRIP); s.n = 10; s.stripT = 1; } },
          { label: 'Arrows', apply(s) { s.mode = 'main'; s.cam = cam2(CAM_P); } },
        ] },
      ],
    },

    checks: [
      { kind: 'manipulate',
        prompt: 'Drag ' + wS + ' so that v·w = 0.',
        setup(s) { base(s); s.cam = cam2(CAM_P); s.v = V0.slice(); s.w = W0.slice(); s.shadowT = 1; s.segT = 1; s.showBar = true; s.drag = ['w']; s.panel = 'play'; },
        judge(s) {
          const nw = N.norm(s.w), dd = Math.abs(degOf(s.v, s.w) - 90);
          if (nw < 0.3) return { ok: false, progress: 0, hint: 'Keep ' + wS + ' longer than zero; turn it instead.' };
          return { ok: dd < LA.TOL.angleDeg, progress: Math.max(0, 1 - dd / 45) };
        },
        hint: 'Make the shadow collapse to a point: aim ' + wS + ' at a right angle to ' + vS + '.',
        solution(s) { s.w = [-1, 3]; } },
      { kind: 'manipulate',
        prompt: 'Drag the tip of ' + wS + ' so that its magenta shadow on ' + vS + ' is 2 long while ‖w‖ = 3 (both lengths are labeled on the picture).',
        setup(s) { base(s); s.cam = cam2(CAM_P); s.v = V0.slice(); s.w = W0.slice(); s.shadowT = 1; s.segT = 1; s.dims = true; s.wNorm = true; s.drag = ['w']; s.panel = 'play'; },
        judge(s) {
          const sh = N.dot(s.v, s.w) / N.norm(s.v), L = N.norm(s.w), e1 = Math.abs(sh - 2), e2 = Math.abs(L - 3);
          return { ok: e1 < LA.TOL.pos && e2 < LA.TOL.pos, progress: Math.max(0, 1 - (e1 + e2) / 3), hint: e1 < LA.TOL.pos && e2 >= LA.TOL.pos ? 'Shadow is right; now stretch or shrink w along that angle to length 3.' : undefined };
        },
        hint: 'cos θ must be 2/3: about 48° from ' + vS + ', at length 3.',
        solution(s) { const u = N.normalize(V0); s.w = N.scale(N.matVec(N.rot2(Math.acos(2 / 3)), u), 3); } },
      { kind: 'entry',
        prompt: 'Type v·w for ' + vS + ' = [2; −1; 3] and ' + wS + ' = [1; 4; −2]. Your value draws as an orange bar beside the hidden true one.',
        bind: 'guess', answer: -8, tol: LA.TOL.entry, label: 'v·w', colors: 'b',
        setup(s) { base(s); s.mode = 'entry'; s.view = '2d'; s.cam = cam2(CAM_ENTRY); s.guess = 0; s.panel = 'entry'; },
        hint: '2·1 + (−1)·4 + 3·(−2).' },
    ],
  });

  /* ---------------- the 2D picture ---------------- */
  /* The scene is drawn inside a full-canvas viewport so that the top strips (their own viewports) paint over it:
     a viewport flushes when it is called, while the main buffer is flushed only at the end of the frame. */
  function renderMain(g, s, ctx) {
    const E = eff(s), v = E.v, w = E.w, u = E.u;
    g.viewport({ x: 0, y: 0, w: 1, h: 1 }, { view: '2d', cam: s.cam, border: false }, (g2) => drawScene(g2, s, E));
    if (s.showList) listPanel(g, s);
    if (s.showBar || ctx.mode === 'play' || ctx.mode === 'check') {
      const d = N.dot(v, w), rows = [{ label: u > 0.5 ? 'v̂·ŵ' : 'v·w', value: d, max: 11, ticks: [-10, -5, 0, 5, 10], color: 'gold', pulse: s.barPulse, note: s.barNote }];
      if (s.showCos || ctx.mode === 'play') rows.push({ label: 'cos θ', value: cosOf(v, w), max: 1.1, ticks: [-1, 0, 1], color: 'teal' });
      scorePanel(g, rows);
    }
    for (const id of s.drag) g.handle(id, s[id], (p, st) => { st[id] = p.slice(); }, { color: id });
  }

  function drawScene(g, s, E) {
    g.ghostGrid();
    g.axes({ labels: true });
    const v = E.v, w = E.w, u = E.u, gr = s.grow, O = [0, 0];
    if (s.ghostV) g.vector(s.v, { color: 'v', width: 2.6, alpha: 0.25 });
    if (s.ghostW && u > 0.02) g.vector(s.w, { color: 'w', width: 2.6, alpha: 0.25 });
    if (s.unitCircle) g.circle(O, 1, { stroke: 'hud', dashed: true, width: 1.2, strokeAlpha: 0.6 });
    // the shadow: w dropped onto v's line (swap → v dropped onto w's line)
    const sw = s.swapOn ? 1 : s.swap, hasShadow = s.shadowT > 0 || s.segT > 0 || s.projT > 0;
    const f0 = projOnto(w, v), f1 = projOnto(v, w), tip = lerpV(w, v, sw), foot = lerpV(f0, f1, sw), B = sw < 0.5 ? v : w;
    if (hasShadow) {
      if (N.norm(B) > 1e-6) g.lineThrough(O, B, { color: 'residual', dashed: [3, 5], width: 1, alpha: 0.55, len: 16 });
      if (s.shadowT > 0) g.line(tip, lerpV(tip, foot, s.shadowT), { color: 'residual', dashed: true, width: 1.5 });
    }
    // the angle between them
    if (s.arc && N.norm(v) > 0.2 && N.norm(w) > 0.2) {
      const deg = Math.round(degOf(v, w));
      g.arc(O, v, w, { r: u > 0.5 ? 0.32 : 0.55, color: 'hud', label: s.arcLabel ? 'θ = ' + deg + '°' : 'θ', size: 12.5 });
    }
    // the two arrows
    g.vector(N.scale(w, gr), { color: 'w', label: 'w', width: 3, hat: u > 0.98 });
    g.vector(N.scale(v, gr), { color: 'v', label: 'v', width: 3, hat: u > 0.98 });
    // the shadow segment / projection arrow lies on top of the receiving arrow
    if (hasShadow) {
      if (s.segT > 0 && s.projT < 1) g.line(O, lerpV(O, foot, s.segT), { color: 'mag', width: 5, alpha: 0.9 * (1 - s.projT) });
      if (s.projT > 0) g.vector(O, f0, { color: 'mag', width: 3.6, alpha: s.projT, label: 'projᵥ(w)', labelAt: 0.5, labelOffset: 24, labelFont: 'ui', labelSize: 13 });
      if (s.projResid) g.label('w − proj', lerpV(w, f0, 0.5), { color: 'hud', dx: 44, dy: -6, size: 12.5, font: 'ui' });
      if (s.shadowT >= 1) {
        if (N.norm(foot) > 0.12 && N.dist(tip, foot) > 0.12) rightMark(g, foot, N.scale(foot, -1), N.sub(tip, foot), 0.17);
        else if (N.norm(foot) <= 0.03 && N.norm(B) > 0.1 && N.norm(tip) > 0.1) {      // perpendicular: the shadow is a dot
          rightMark(g, O, B, tip, 0.3, 'gold');
          g.point(O, { r: 5, color: 'mag' });
          g.label('perpendicular: v·w = 0', O, { color: 'gold', dx: 16, dy: 22, anchor: 'left', size: 13, font: 'ui', bold: true });
        }
      }
      if (s.cosLabel) g.label('cos θ = ' + LA.fmt(N.dot(v, w) / Math.max(N.norm(v), 1e-9)), foot, { color: 'mag', dx: 8, dy: 24, anchor: 'left', size: 13, font: 'ui' });
      if (s.dims && s.projT === 0) {
        if (sw < 0.05 && N.norm(v) > 0.2) {
          g.dim(O, f0, { label: LA.fmt(N.dot(v, w) / N.norm(v)), color: 'mag', offset: 16 });
          g.dim(O, v, { label: '‖v‖ = ' + LA.fmt(N.norm(v)), color: 'v', offset: 40 });
        } else if (sw > 0.95 && N.norm(w) > 0.2) {
          g.dim(O, f1, { label: LA.fmt(N.dot(v, w) / N.norm(w)), color: 'mag', offset: 16 });
          g.dim(O, w, { label: '‖w‖ = ' + LA.fmt(N.norm(w)), color: 'w', offset: -22 });
        }
      }
    }
    if (s.wNorm && N.norm(w) > 0.2) g.dim(O, w, { label: '‖w‖ = ' + LA.fmt(N.norm(w)), color: 'w', offset: -22 });
    // the misconception: an arrow for v·w, which has no direction to point in
    if (s.mis > 0) {
      const a = 1.95 + Math.PI * s.mis + 0.3 * Math.sin(s.mis * Math.PI * 3), tipM = [1.9 * Math.cos(a), 1.9 * Math.sin(a)];
      g.vector(tipM, { color: 'muted', width: 2.6, label: 'v·w ?', labelFont: 'ui', labelSize: 13 });
      g.label('claim: "v·w is a vector"', [16, g.size.h - 46], { screen: true, color: 'muted', anchor: 'left', size: 14, font: 'ui' });
      if (s.cross) { g.marker.cross([206, g.size.h - 46], { screen: true, size: 22 }); g.label('no direction to point in', tipM, { color: 'bad', dy: 26, size: 12.5, font: 'ui' }); }
    }
    if (s.note) g.hud(s.note, 14, g.size.h - 18, { color: 'hud', size: 14 });
  }

  /** the score strip across the top of the canvas: a number line with one bar per row */
  function scorePanel(g, rows) {
    const two = rows.length > 1;
    g.viewport({ x: 0.02, y: 0.015, w: 0.96, h: two ? 0.17 : 0.1 }, { view: '2d', cam: { center: [0, 0], halfH: 1 }, bg: 'panel' }, (g2) => {
      const b = g2.P.bounds(), W = b.x1 - b.x0, a0 = b.x0 + 0.17 * W, a1 = b.x1 - 0.17 * W, xm = (a0 + a1) / 2, half = (a1 - a0) / 2;
      const px = g2.P.scale || 40;
      rows.forEach((r, k) => {
        const y = two ? (k === 0 ? 0.45 : -0.45) : 0.1, th = (two ? 0.17 : 0.24) * (1 + 0.4 * (r.pulse || 0));
        g2.line([a0, y], [a1, y], { color: 'line', width: 1.2 });
        r.ticks.forEach(tv => { const x = xm + tv / r.max * half; g2.line([x, y - 0.1], [x, y + 0.1], { color: 'tick', width: 1 }); g2.label(String(tv).replace('-', '−'), [x, y], { anchor: 'below', dy: th * px + 1, color: 'tick', size: 10.5, font: 'ui', bg: false }); });
        const val = Math.abs(r.value) < 0.005 ? 0 : r.value, x = xm + LA.clamp(val / r.max, -1.06, 1.06) * half, col = val < 0 ? 'neg' : r.color;
        if (Math.abs(x - xm) > 1e-4) g2.polygon([[xm, y - th], [x, y - th], [x, y + th], [xm, y + th]], { fill: col, alpha: 0.85, stroke: null });
        else g2.line([xm, y - th * 1.4], [xm, y + th * 1.4], { color: 'gold', width: 2.5 });
        g2.label(r.label, [b.x0 + 0.025 * W, y], { anchor: 'left', color: 'text', size: 14, font: 'ui', bg: false });
        g2.label('= ' + LA.fmt(val), [b.x1 - 0.025 * W, y], { anchor: 'right', color: col, size: 14.5, font: 'ui', bold: true, bg: false });
        if (r.note) g2.label(r.note, [x, y], { anchor: val < 0 ? 'right' : 'left', dx: val < 0 ? -10 : 10, color: 'hud', size: 13, font: 'ui', bold: true, bg: false });
      });
    });
  }

  /** the list face: two brackets, matching products sliding out, then summing */
  function listPanel(g, s) {
    const t = s.listT, top = seg(t, 0.05, 0.38), bot = seg(t, 0.4, 0.68), sum = seg(t, 0.72, 0.95);
    g.viewport({ x: 0.02, y: 0.015, w: 0.96, h: 0.21 }, { view: '2d', cam: { center: [0, 0], halfH: 1.2 }, bg: 'panel' }, (g2) => {
      const b = g2.P.bounds(), xv = b.x0 + 1.0, xw = xv + 1.25;
      const bracket = (x, color) => {
        g2.polyline([[x - 0.18, 0.72], [x - 0.3, 0.72], [x - 0.3, -0.72], [x - 0.18, -0.72]], { color: 'hud', width: 1.4 });
        g2.polyline([[x + 0.18, 0.72], [x + 0.3, 0.72], [x + 0.3, -0.72], [x + 0.18, -0.72]], { color: 'hud', width: 1.4 });
      };
      if (top > 0 && top < 1 || (top >= 1 && bot === 0)) g2.polygon([[xv - 0.34, 0.12], [xw + 0.34, 0.12], [xw + 0.34, 0.68], [xv - 0.34, 0.68]], { fill: 'i', alpha: 0.18, stroke: null });
      if (bot > 0 && sum === 0) g2.polygon([[xv - 0.34, -0.68], [xw + 0.34, -0.68], [xw + 0.34, -0.12], [xv - 0.34, -0.12]], { fill: 'j', alpha: 0.18, stroke: null });
      bracket(xv); bracket(xw);
      g2.label('3', [xv, 0.4], { color: 'v', size: 17, font: 'math', bg: false }); g2.label('1', [xv, -0.4], { color: 'v', size: 17, font: 'math', bg: false });
      g2.label('1', [xw, 0.4], { color: 'w', size: 17, font: 'math', bg: false }); g2.label('2', [xw, -0.4], { color: 'w', size: 17, font: 'math', bg: false });
      g2.label('v', [xv - 0.62, 0], { color: 'v', size: 17, font: 'math', bg: false }); g2.label('·', [(xv + xw) / 2, 0], { color: 'hud', size: 20, bg: false }); g2.label('w', [xw + 0.62, 0], { color: 'w', size: 17, font: 'math', bg: false });
      const xs = xw + 1.3, xe = xw + 3.3, xf = xw + 4.4;
      if (top > 0) g2.label('3·1 = 3', [LA.lerp(xs, xe, top), LA.lerp(0.4, 0.05, sum)], { color: 'i', size: 15, font: 'ui', bg: false, alpha: 1 - sum, anchor: 'left' });
      if (bot > 0) g2.label('1·2 = 2', [LA.lerp(xs, xe, bot), LA.lerp(-0.4, -0.05, sum)], { color: 'j', size: 15, font: 'ui', bg: false, alpha: 1 - sum, anchor: 'left' });
      if (sum > 0) g2.label('3 + 2 = 5', [xf, 0], { color: 'gold', size: 18, font: 'ui', bold: true, bg: false, alpha: sum, anchor: 'left' });
    });
  }

  /* ---------------- 3D ---------------- */
  function render3d(g, s) {
    g.cage({ range: 3, walls: true });
    g.axes({ length: 2.6, labels: true });
    const v = s.v3, w = s.w3, O = [0, 0, 0], foot = projOnto(w, v);
    g.lineThrough(O, v, { color: 'residual', dashed: [3, 5], width: 1, alpha: 0.6, len: 7 });
    if (s.d3T > 0) g.line(w, lerpV(w, foot, s.d3T), { color: 'residual', dashed: true, width: 1.5 });
    if (s.seg3 > 0) g.line(O, lerpV(O, foot, s.seg3), { color: 'mag', width: 5, alpha: 0.9 });
    if (s.d3T >= 1) rightMark(g, foot, N.scale(foot, -1), N.sub(w, foot), 0.26);
    g.vector(O, w, { color: 'w', label: 'w', width: 3 });
    g.vector(O, v, { color: 'v', label: 'v', width: 3 });
    if (s.dims3) {
      g.dim(O, foot, { label: 'shadow ' + LA.fmt(N.dot(v, w) / N.norm(v)), color: 'mag', offset: 18 });
      g.dim(O, v, { label: '‖v‖ = ' + LA.fmt(N.norm(v)), color: 'v', offset: 44 });
      g.hud('v·w = 2·1 + 0·2 + 1·1 = 3 = 1.34 × 2.24', 14, g.size.h - 18, { color: 'hud', size: 14 });
    }
  }

  /* ---------------- n-D: strips ---------------- */
  function renderStrip(g, s, ctx) {
    const n = Math.max(2, Math.min(12, Math.round(s.n))), a = s.sA.slice(0, n), b = s.sB.slice(0, n);
    const cw = Math.min(0.5, 5.0 / n), x0 = -cw * n / 2, t = ctx.mode === 'learn' ? s.stripT : 1;
    const cells = (vals, y, range, ramp, title, tc, show) => {
      g.heat2d([vals], { x0, x1: x0 + cw * n, y0: y - 0.28, y1: y + 0.28 }, { range, ramp, alpha: 0.9, smooth: false });
      for (let i = 0; i <= n; i++) g.line([x0 + i * cw, y - 0.28], [x0 + i * cw, y + 0.28], { color: 'line', width: 1 });
      g.line([x0, y - 0.28], [x0 + n * cw, y - 0.28], { color: 'line' }); g.line([x0, y + 0.28], [x0 + n * cw, y + 0.28], { color: 'line' });
      for (let i = 0; i < n; i++) if (!show || show(i)) g.label(LA.fmt(vals[i]), [x0 + (i + 0.5) * cw, y], { color: 'text', size: 12.5, font: 'mono', bg: false });
      g.label(title, [x0, y + 0.28], { color: tc || 'muted', anchor: 'left', dy: -11, size: 13, font: 'ui', bg: false });
    };
    const div = u => LA.color.diverge(u * 2 - 1);
    cells(a, 2.0, [-3, 3], div, 'v (' + n + ' cells)', 'v');
    cells(b, 0.95, [-3, 3], div, 'w (' + n + ' cells)', 'w');
    const k = Math.floor(seg(t, 0.05, 0.7, 'linear') * n + 1e-9);
    const prods = a.map((x, i) => i < k ? x * b[i] : 0);
    cells(prods, -0.1, [-9, 9], div, 'vᵢ · wᵢ', 'gold', i => i < k);
    if (k > 0 && k < n) g.polygon([[x0 + (k - 1) * cw, -0.38], [x0 + k * cw, -0.38], [x0 + k * cw, 2.28], [x0 + (k - 1) * cw, 2.28]], { fill: null, stroke: 'gold', width: 2 });
    const S = prods.reduce((p, q) => p + q, 0);
    if (k > 0) g.label('running sum = ' + LA.fmt(S), [0, -0.75], { color: 'gold', size: 14, font: 'ui', bold: k >= n, bg: false });
    if (k >= n && t > 0.75) {
      g.label('‖v‖ = ' + LA.fmt(N.norm(a)) + '    ‖w‖ = ' + LA.fmt(N.norm(b)), [0, -1.3], { color: 'hud', size: 13, font: 'ui', bg: false });
      g.label('cos θ = ' + LA.fmt(S) + ' / (' + LA.fmt(N.norm(a)) + ' × ' + LA.fmt(N.norm(b)) + ') = ' + LA.fmt(cosOf(a, b)), [0, -1.75], { color: 'teal', size: 14, font: 'ui', bold: true, bg: false });
    }
    g.label('The picture stops at 3. The formula does not.', [0, -2.55], { color: 'muted', size: 13, font: 'ui', bg: false });
    g.label('synthetic strips: integers −3..3', [0, -2.95], { color: 'muted', size: 11.5, font: 'ui', bg: false });
  }

  /* ---------------- ML tie-in: a template and four inputs; attention's row of scores ---------------- */
  function renderMl(g, s) {
    const t = s.mlT, scores = INPUTS.map(x => N.dot(TEMPL, x)), slot = i => [0.04 + 0.13 * i, 0.04 + 0.13 * (i + 1)];
    g.viewport({ x: 0, y: 0, w: 1, h: 0.6 }, { view: '2d', cam: { center: [0.25, 0.5], halfH: 1.85 }, border: false }, (g2) => {
      g2.ghostGrid(); g2.axes({ labels: false });
      g2.lineThrough([0, 0], TEMPL, { color: 'residual', dashed: [3, 5], width: 1, alpha: 0.55, len: 12 });
      INPUTS.forEach((x, i) => {
        const sl = slot(i), d = seg(t, sl[0], sl[0] + 0.07), e = seg(t, sl[0] + 0.05, sl[1]), foot = projOnto(x, TEMPL), cur = t >= sl[0] && t < sl[1] + 0.02;
        g2.vector(x, { color: 'v', width: 2.6, label: 'x' + SUBS[i], labelFont: 'math' });
        if (d > 0) g2.line(x, lerpV(x, foot, d), { color: 'residual', dashed: true, width: 1.3, alpha: cur ? 1 : 0.45 });
        if (e > 0) { if (cur) g2.line([0, 0], lerpV([0, 0], foot, e), { color: 'mag', width: 4.5, alpha: 0.9 }); g2.point(foot, { r: 4, color: 'mag' }); }
      });
      g2.vector(TEMPL, { color: 'w', width: 3.4, label: 'w (template)', labelFont: 'ui', labelSize: 13 });
    });
    g.viewport({ x: 0, y: 0.6, w: 1, h: 0.4 }, { view: '2d', cam: { center: [0, 0], halfH: 1.5 }, bg: 'panel', border: true }, (g2) => {
      const b = g2.P.bounds(), bw = 0.45, gap = 0.22, ox = b.x0 + 0.5, base = -0.1, k = 0.17;
      const fade = 1 - 0.4 * seg(t, 0.6, 0.8);
      g2.line([ox - 0.15, base], [ox + 4 * (bw + gap), base], { color: 'line', width: 1.2 });
      scores.forEach((sc, i) => {
        const sl = slot(i), f = seg(t, sl[0] + 0.05, sl[1]), x0 = ox + i * (bw + gap), h = sc * k * f, col = sc < 0 ? 'neg' : 'gold';
        if (Math.abs(h) > 1e-4) g2.polygon([[x0, base], [x0 + bw, base], [x0 + bw, base + h], [x0, base + h]], { fill: col, alpha: 0.85 * fade, stroke: null });
        g2.label('x' + SUBS[i], [x0 + bw / 2, base], { anchor: sc < 0 ? 'above' : 'below', dy: sc < 0 ? -4 : 4, color: 'muted', size: 12, font: 'math', bg: false });
        if (f >= 1) g2.label(LA.fmt(sc), [x0 + bw / 2, base + h], { anchor: sc < 0 ? 'below' : 'above', dy: sc < 0 ? 3 : -3, color: 'hud', size: 12, font: 'ui', bg: false });
      });
      g2.label('neuron: score = w·x', [ox - 0.15, 1.25], { anchor: 'left', color: 'hud', size: 13, font: 'ui', bg: false });
      const r = seg(t, 0.62, 0.85);
      if (r > 0) {
        const hx0 = ox + 4 * (bw + gap) + 0.55, cwid = 0.62, hb = { x0: hx0, x1: hx0 + 4 * cwid, y0: -0.05, y1: 0.45 };
        g2.heat2d([scores], hb, { range: [-6, 6], ramp: u => LA.color.diverge(u * 2 - 1), alpha: 0.95 * r, smooth: false });
        for (let i = 0; i <= 4; i++) g2.line([hx0 + i * cwid, hb.y0], [hx0 + i * cwid, hb.y1], { color: 'line', width: 1 });
        g2.line([hb.x0, hb.y0], [hb.x1, hb.y0], { color: 'line' }); g2.line([hb.x0, hb.y1], [hb.x1, hb.y1], { color: 'line' });
        scores.forEach((sc, i) => {
          g2.label(LA.fmt(sc), [hx0 + (i + 0.5) * cwid, 0.2], { color: 'text', size: 12.5, font: 'mono', bg: false, alpha: r });
          g2.label('q·k' + SUBS[i], [hx0 + (i + 0.5) * cwid, hb.y0], { anchor: 'below', dy: 5, color: 'muted', size: 11.5, font: 'ui', bg: false, alpha: r });
        });
        g2.label('attention: query q = w, keys kᵢ = xᵢ', [hx0, 1.25], { anchor: 'left', color: 'hud', size: 13, font: 'ui', bg: false, alpha: r });
        g2.label('→ one row of scores', [hx0, 0.9], { anchor: 'left', color: 'hud', size: 13, font: 'ui', bg: false, alpha: r });
      }
      g2.label('next: p1-09 · Part 6: attention', [b.x1 - 0.15, b.y0 + 0.22], { anchor: 'right', color: 'muted', size: 12, font: 'ui', bg: false });
    });
  }

  /* ---------------- entry check: three-cell strips, the typed bar beside the hidden true bar ---------------- */
  function renderEntry(g, s) {
    const n = 3, cw = 0.8, x0 = -cw * n / 2, truth = N.dot(CV, CW), guess = Number(s.guess) || 0, solved = Math.abs(guess - truth) < LA.TOL.entry;
    const div = u => LA.color.diverge(u * 2 - 1);
    const cells = (vals, y, range, title, tc, hide) => {
      g.heat2d([hide ? [0, 0, 0] : vals], { x0, x1: x0 + cw * n, y0: y - 0.25, y1: y + 0.25 }, { range, ramp: div, alpha: 0.9, smooth: false });
      for (let i = 0; i <= n; i++) g.line([x0 + i * cw, y - 0.25], [x0 + i * cw, y + 0.25], { color: 'line', width: 1 });
      g.line([x0, y - 0.25], [x0 + n * cw, y - 0.25], { color: 'line' }); g.line([x0, y + 0.25], [x0 + n * cw, y + 0.25], { color: 'line' });
      for (let i = 0; i < n; i++) g.label(hide ? '?' : LA.fmt(vals[i]), [x0 + (i + 0.5) * cw, y], { color: hide ? 'muted' : 'text', size: 14, font: 'mono', bg: false });
      g.label(title, [x0, y], { anchor: 'right', dx: -10, color: tc, size: 13.5, font: 'ui', bg: false });
    };
    cells(CV, 2.0, [-3, 3], 'v', 'v');
    cells(CW, 1.3, [-4, 4], 'w', 'w');
    cells(CV.map((x, i) => x * CW[i]), 0.6, [-8, 8], 'vᵢ · wᵢ', 'gold', !solved);
    const baseY = -0.55, k = 0.17, bw = 0.6;
    g.line([-1.6, baseY], [1.6, baseY], { color: 'line', width: 1.2 });
    const gh = LA.clamp(guess, -12, 12) * k;
    if (Math.abs(gh) > 1e-4) g.polygon([[-1.0, baseY], [-1.0 + bw, baseY], [-1.0 + bw, baseY + gh], [-1.0, baseY + gh]], { fill: 'b', alpha: 0.85, stroke: null });
    g.label('your v·w = ' + LA.fmt(guess), [-0.7, baseY], { anchor: guess < 0 ? 'above' : 'below', dy: guess < 0 ? -6 : 6, color: 'b', size: 13, font: 'ui', bg: false });
    const th = truth * k, tb = [[0.4, baseY], [0.4 + bw, baseY], [0.4 + bw, baseY + th], [0.4, baseY + th]];
    if (solved) { g.polygon(tb, { fill: 'gold', alpha: 0.85, stroke: null }); g.label('true v·w = ' + LA.fmt(truth), [0.7, baseY], { anchor: 'above', dy: -6, color: 'gold', size: 13, font: 'ui', bg: false }); }
    else {   // a neutral box that gives away neither sign nor size
      g.polygon([[0.4, baseY - 0.5], [0.4 + bw, baseY - 0.5], [0.4 + bw, baseY + 0.5], [0.4, baseY + 0.5]], { fill: null, stroke: 'muted', width: 1.2, dash: [4, 4] });
      g.label('?', [0.7, baseY], { color: 'muted', size: 18, font: 'ui', bg: false });
      g.label('true v·w: hidden', [0.7, baseY + 0.5], { anchor: 'above', dy: -6, color: 'muted', size: 13, font: 'ui', bg: false });
    }
  }
})();
