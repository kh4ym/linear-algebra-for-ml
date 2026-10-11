/* p0-02 — How to read the symbols. (Part 0 primer) */
(function () {
  'use strict';
  const N = LA.num, seg = LA.seg, lerp = LA.lerp, lerpV = LA.lerpV;
  const A0 = [[1, 4, 7], [2, 5, 8]];
  const ACHECK = [[3, -2, 6], [5, 1, 7]];
  const STRIP12 = [2, -1, 3, 4, 1, -2, 5, 0, 3, -1, 2, 4];
  const COLS = ['i', 'j', 'k'];
  const SUB = ['₀', '₁', '₂', '₃', '₄', '₅', '₆', '₇', '₈', '₉'];
  const sub = n => String(n).split('').map(d => SUB[+d]).join('');
  const vhat = '<span class="c-v">v</span>';
  // one synthetic "7", 28×28, flattened row by row (784 numbers)
  const DIGIT7 = LA.rng.syntheticDigits(2, 8, { size: 28 })[7].pixels;
  const grayRamp = t => LA.color.mix('#0b0e14', '#e8ebf3', t);
  const divRamp = t => LA.color.diverge(2 * t - 1);
  // In panels that contain a matrix, numbers after it are drawn as text spans: the panel's in-place updater pairs live
  // numbers with matrix cells by position, so a tex.num placed after a tex.mat can be written into the wrong cell.
  const numT = (tex, x, o) => tex.text(typeof x === 'number' ? LA.fmt(x) : String(x), o);
  const camPoint = () => ({ center: [2, 1.6], halfH: 3.6 });
  const camMid = () => ({ center: [0, 0], halfH: 3.4 });

  /* ---------------- small drawing helpers ---------------- */
  function bracket(g, x0, y0, x1, y1, o) {
    o = o || {};
    const lip = Math.min(0.16, (x1 - x0) * 0.25), st = { color: o.color || 'hud', width: o.width || 2, alpha: o.alpha === undefined ? 1 : o.alpha };
    g.polyline([[x0 + lip, y1], [x0, y1], [x0, y0], [x0 + lip, y0]], st);
    g.polyline([[x1 - lip, y1], [x1, y1], [x1, y0], [x1 - lip, y0]], st);
  }
  const cellC = (o, i, j, c) => [o[0] + (j + 0.5) * c, o[1] - (i + 0.5) * c];
  function rect(g, cx, cy, w, h, o) {
    g.polygon([[cx - w / 2, cy - h / 2], [cx + w / 2, cy - h / 2], [cx + w / 2, cy + h / 2], [cx - w / 2, cy + h / 2]], o);
  }
  /** A matrix as a grid of tinted cells inside a bracket. o: top-left corner; c: cell size. */
  function drawMatrix(g, M, o, c, opt) {
    opt = opt || {};
    const n = M.length, m = M[0].length, al = opt.alpha === undefined ? 1 : opt.alpha;
    for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) {
      const p = cellC(o, i, j, c), role = opt.tint === false ? 'muted' : COLS[(opt.rowsTint ? i : j) % 3];
      rect(g, p[0], p[1], c * 0.94, c * 0.94, { fill: role, alpha: 0.13 * al, stroke: role, width: 1 });
      g.label(LA.fmt(M[i][j]), p, { color: opt.tint === false ? 'text' : role, size: opt.size || 20, font: 'math', bg: false, alpha: al });
    }
    bracket(g, o[0] - 0.1, o[1] - n * c - 0.1, o[0] + m * c + 0.1, o[1] + 0.1, { alpha: al });
  }
  function rowColLabels(g, o, n, m, c) {
    for (let i = 0; i < n; i++) g.label('row ' + (i + 1), [o[0] - 0.2, o[1] - (i + 0.5) * c], { anchor: 'right', color: 'muted', size: 12, font: 'ui', bg: false });
    for (let j = 0; j < m; j++) g.label('col ' + (j + 1), [o[0] + (j + 0.5) * c, o[1] + 0.25], { anchor: 'above', color: 'muted', size: 12, font: 'ui', bg: false });
  }
  function axesFrame(g) { g.grid(null, { range: 8, alpha: 0.45 }); g.axes(); }

  /* ---------------- scenes ---------------- */
  function scenePoint(g, s) {
    const u = s.u, v = [3, 2], step = s.scene;
    axesFrame(g);
    const hiI = step === 'sub' && u < 0.5, hiJ = step === 'sub' && u >= 0.5;
    g.line([0, 0], [3, 0], { color: 'i', dashed: true, width: hiI ? 3.2 : 1.6 });
    g.line([3, 0], v, { color: 'j', dashed: true, width: hiJ ? 3.2 : 1.6 });
    // the canvas bracket v = [3; 2]
    const B = { top: [1.25, 4.3], bot: [1.25, 3.7] };
    bracket(g, 0.9, 3.35, 1.6, 4.65);
    g.label('v =', [0.72, 4.0], { anchor: 'right', color: 'v', size: 20, font: 'math', bg: false });
    if (step === 'bracket') {
      const f = seg(u, 0.3, 0.8);
      g.label('3', lerpV([1.5, -0.35], B.top, f), { color: 'i', size: 20, font: 'math', bg: f < 0.98 });
      g.label('2', lerpV([3.35, 1.0], B.bot, f), { color: 'j', size: 20, font: 'math', bg: f < 0.98 });
      const a = seg(u, 0.82, 1);
      if (a > 0) g.vector([0, 0], v, { color: 'v', label: 'v', width: 3, alpha: a });
    } else {
      g.label('3', B.top, { color: 'i', size: 20, font: 'math', bg: false });
      g.label('2', B.bot, { color: 'j', size: 20, font: 'math', bg: false });
      g.vector([0, 0], v, { color: 'v', label: 'v', width: 3, alpha: step === 'sub' ? 0.5 : 1 });
    }
    g.point(v, { r: 6, color: 'v' });
    if (step === 'sub') {
      const pulse = 0.27 + 0.03 * Math.sin(u * 30);
      g.circle(u < 0.5 ? B.top : B.bot, pulse, { stroke: 'gold', width: 2.4 });
      g.label('v' + sub(1) + ' = 3  (first entry)', [1.9, 4.3], { anchor: 'left', color: 'i', size: 15, font: 'ui', alpha: u >= 0.04 ? 1 : 0.25 });
      if (u >= 0.5) g.label('v' + sub(2) + ' = 2  (second entry)', [1.9, 3.7], { anchor: 'left', color: 'j', size: 15, font: 'ui' });
    }
    if (step === 'norm') {
      const e = lerpV([0, 0], v, seg(u, 0, 0.6));
      g.dim([0, 0], e, { offset: 18, color: 'v', label: u >= 0.7 ? '‖v‖ ≈ 3.61' : undefined, size: 15 });
    }
  }

  function sceneAddress(g, s) {
    const u = s.u, c = 1, o = [-1.5, 1];
    drawMatrix(g, s.A, o, c);
    rowColLabels(g, o, 2, 3, c);
    g.label('A =', [o[0] - 1.0, o[1] - c], { anchor: 'right', color: 'text', size: 20, font: 'math', bg: false });
    const r = seg(u, 0.1, 0.4), k = seg(u, 0.4, 0.7);
    if (u > 0.08) { const y = lerp(o[1] - 0.5, o[1] - 1.5, r); rect(g, o[0] + 1.5, y, 3.1, 0.9, { fill: 'gold', alpha: 0.12, stroke: 'gold', width: 1.5 }); }
    if (u > 0.4) { const x = lerp(o[0] + 0.5, o[0] + 2.5, k); rect(g, x, o[1] - 1, 0.9, 2.1, { fill: 'gold', alpha: 0.12, stroke: 'gold', width: 1.5 }); }
    g.label('1. down to row 2', [o[0] - 0.2, -2.0], { anchor: 'left', color: u > 0.1 ? 'gold' : 'muted', size: 14, font: 'ui' });
    g.label('2. across to column 3', [o[0] - 0.2, -2.5], { anchor: 'left', color: u > 0.4 ? 'gold' : 'muted', size: 14, font: 'ui' });
    if (u > 0.7) {
      const p = cellC(o, 1, 2, c);
      g.circle(p, 0.42 + 0.05 * Math.sin((u - 0.7) * 40), { stroke: 'gold', width: 2.6 });
      g.label('a' + sub(23) + ' = 8', [p[0] + 0.65, p[1]], { anchor: 'left', color: 'gold', size: 17, font: 'ui', bold: true });
    }
  }

  function sceneShape(g, s) {
    const u = s.u, c = 0.7;
    const oA = [-2.2, 1.0], oB = [1.0, 1.35];
    const B = [[1, 4], [7, 2], [5, 8]];
    drawMatrix(g, A0, oA, c, { size: 17 });
    g.label('A', [oA[0] + 1.05, oA[1] + 0.45], { color: 'text', size: 18, font: 'math', bg: false });
    g.dim([oA[0] - 0.1, oA[1] - 2 * c], [oA[0] - 0.1, oA[1]], { offset: -24, label: '2 rows', color: 'hud' });
    g.dim([oA[0], oA[1] - 2 * c - 0.1], [oA[0] + 3 * c, oA[1] - 2 * c - 0.1], { offset: 14, label: '3 columns', color: 'hud' });
    const b = seg(u, 0.3, 0.6);
    if (b > 0) {
      drawMatrix(g, B, oB, c, { size: 17, alpha: b });
      g.label('B', [oB[0] + 0.7, oB[1] + 0.45], { color: 'text', size: 18, font: 'math', bg: false, alpha: b });
      if (b > 0.9) {
        g.dim([oB[0] + 2 * c + 0.1, oB[1] - 3 * c], [oB[0] + 2 * c + 0.1, oB[1]], { offset: 14, label: '3 rows', color: 'hud' });
        g.dim([oB[0], oB[1] - 3 * c - 0.1], [oB[0] + 2 * c, oB[1] - 3 * c - 0.1], { offset: 14, label: '2 columns', color: 'hud' });
      }
    }
    // the 2×3 outline tries to sit on top of the 3×2 and does not fit
    const m = seg(u, 0.6, 0.8);
    if (m > 0) {
      const o = lerpV(oA, [oB[0] - 0.35, oB[1]], m);
      g.polygon([[o[0], o[1]], [o[0] + 3 * c, o[1]], [o[0] + 3 * c, o[1] - 2 * c], [o[0], o[1] - 2 * c]], { stroke: 'muted', width: 2, alpha: 0.9, dash: [5, 4] });
    }
    g.label('2×3 = 3×2 ?', [0, 2.55], { color: 'text', size: 18, font: 'ui' });
    if (u >= 0.8) g.marker.cross([0, 2.55], { dx: 78, size: 22 });
    if (u >= 0.8) g.label('different shapes: rows first, always', [0, -2.75], { color: 'bad', size: 14, font: 'ui' });
  }

  function sceneSum(g, s) {
    const u = s.u, vals = [2, -1, 3, 4], cw = 0.95, y0 = 1.3, x0 = -2.6;
    const box = { x: 2.15, y: -0.9, w: 1.3, h: 2.6 };
    rect(g, box.x, box.y, box.w, box.h, { fill: 'panel2', alpha: 0.6, stroke: 'muted', width: 1.2 });
    g.label('pile', [box.x, box.y - box.h / 2 - 0.25], { color: 'muted', size: 13, font: 'ui' });
    let total = 0, landed = 0;
    vals.forEach((v, k) => {
      const f = seg(u, 0.1 + 0.2 * k, 0.28 + 0.2 * k);
      if (f >= 1) { total += v; landed++; }
      const start = [x0 + (k + 0.5) * cw, y0], end = [box.x, box.y - box.h / 2 + 0.35 + 0.6 * k];
      const p = lerpV(start, end, f), h = lerp(cw * 0.94, 0.5, f);
      if (f < 1) rect(g, start[0], start[1], cw * 0.94, cw * 0.94, { stroke: 'muted', width: 1, alpha: 0.5, dash: [3, 3] });
      rect(g, p[0], p[1], cw * 0.94, h, { fill: LA.color.diverge(v / 4), alpha: 0.95, stroke: 'line' });
      g.label(LA.fmt(v), p, { color: 'text', size: 17, font: 'math', bg: false });
      g.label('s' + sub(k + 1), [start[0], y0 - 0.75], { color: 'muted', size: 14, font: 'math', bg: false });
    });
    g.label('i = 1 … 4', [x0 + 2 * cw, y0 + 0.8], { color: 'muted', size: 13, font: 'ui', bg: false });
    g.label(landed === 4 ? 'Σ = ' + LA.fmt(total) : 'running total: ' + LA.fmt(total), [box.x, box.y + box.h / 2 + 0.35], { color: landed === 4 ? 'gold' : 'text', size: landed === 4 ? 19 : 14, font: 'ui', bold: landed === 4 });
  }

  function sceneR(g, s) {
    const u = s.u;
    const ph = u < 0.3 ? 0 : u < 0.6 ? 1 : u < 0.8 ? 2 : 3;
    const fadeIn = [seg(u, 0, 0.06), seg(u, 0.3, 0.36), seg(u, 0.6, 0.66), seg(u, 0.8, 0.86)][ph];
    const al = fadeIn;
    if (ph === 0) {
      g.line([-1.2, 1.2], [4.4, 1.2], { color: 'gridAxis', width: 2, alpha: al });
      for (let x = -1; x <= 4; x++) { g.line([x, 1.05], [x, 1.35], { color: 'gridAxis', alpha: al }); g.label(String(x).replace('-', '−'), [x, 1.2], { anchor: 'below', dy: 12, color: 'tick', size: 12, font: 'ui', bg: false, alpha: al }); }
      g.point([2.5, 1.2], { r: 6, color: 'v', alpha: al });
      g.label('2.5', [2.5, 1.2], { dy: -18, color: 'v', size: 14, font: 'ui', alpha: al });
      g.label('ℝ · the number line', [1.6, 3.4], { color: 'gold', size: 22, font: 'ui', bold: true, alpha: al });
    } else if (ph === 1) {
      g.grid(null, { range: 6, alpha: 0.45 * al });
      g.line([-1.4, 0], [4.6, 0], { color: 'gridAxis', width: 1.6, alpha: al }); g.line([0, -1.4], [0, 4.0], { color: 'gridAxis', width: 1.6, alpha: al });
      g.line([0, 0], [3, 0], { color: 'i', dashed: true, alpha: al }); g.line([3, 0], [3, 2], { color: 'j', dashed: true, alpha: al });
      g.point([3, 2], { r: 6, color: 'v', alpha: al, label: 'v = [3; 2]', size: 14 });
      g.label('ℝ² · the plane: all pairs', [1.6, 4.4], { color: 'gold', size: 22, font: 'ui', bold: true, alpha: al });
    } else if (ph === 2) {
      const vals = [2, 1, 1.5];
      vals.forEach((v, k) => { rect(g, 0.6 + k * 1.0, 1.4, 0.94, 0.94, { fill: COLS[k], alpha: 0.18 * al, stroke: COLS[k] }); g.label(LA.fmt(v), [0.6 + k * 1.0, 1.4], { color: COLS[k], size: 19, font: 'math', bg: false, alpha: al }); });
      g.label('ℝ³ — three numbers, a point in a room', [1.6, 3.4], { color: 'gold', size: 18, font: 'ui', bold: true, alpha: al });
      g.label('x, y, z: our 3D room is the last space we can draw', [1.6, -0.3], { color: 'muted', size: 13, font: 'ui', alpha: al });
    } else {
      const b = { x0: -1.2, x1: 4.4, y0: 1.5, y1: 2.1 };
      g.heat2d([DIGIT7], b, { range: [0, 1], ramp: grayRamp, smooth: false, alpha: al });
      g.polyline([[b.x0, b.y0], [b.x1, b.y0], [b.x1, b.y1], [b.x0, b.y1]], { color: 'line', closed: true, width: 1, alpha: al });
      g.label('784 thin cells: one synthetic “7”, read row by row', [1.6, 2.45], { color: 'muted', size: 12.5, font: 'ui', alpha: al });
      const w = (b.x1 - b.x0) / 12;
      for (let k = 0; k < 12; k++) g.label(LA.fmt.fixed(DIGIT7[k], 2), [b.x0 + (k + 0.5) * w, 0.9], { color: 'hud', size: 10.5, font: 'mono', bg: false, alpha: al });
      g.label('first 12 of 784 values (the blank top edge of the image)', [1.6, 0.45], { color: 'muted', size: 12, font: 'ui', alpha: al });
      g.label('ℝ⁷⁸⁴ · one point we cannot draw', [1.6, 3.6], { color: 'gold', size: 21, font: 'ui', bold: true, alpha: al });
      g.label('The picture stops at 3. The formula does not.', [1.6, -0.6], { color: 'text', size: 14, font: 'ui', alpha: al });
    }
  }

  function sceneFunction(g, s, ctx) {
    const u = s.u, y = 2.3;
    const pul = seg.pulse(u, 0.38, 0.55);
    rect(g, 0, y, 1.3 + 0.2 * pul, 1.0 + 0.15 * pul, { fill: 'w', alpha: 0.18 + 0.25 * pul, stroke: 'w', width: 2 });
    g.label('f', [0, y], { color: 'w', size: 28, font: 'math', bg: false });
    g.label('doubles the distance from the origin', [0, y - 0.85], { color: 'muted', size: 12.5, font: 'ui' });
    g.vector([-2.75, y], [-0.8, y], { color: 'hud', width: 1.4, head: 9 });
    g.vector([0.8, y], [2.75, y], { color: 'hud', width: 1.4, head: 9 });
    g.label('input', [-2.3, y + 0.42], { color: 'muted', size: 12, font: 'ui', bg: false });
    g.label('output', [2.3, y + 0.42], { color: 'muted', size: 12, font: 'ui', bg: false });
    const a = seg(u, 0, 0.4), b = seg(u, 0.5, 0.9);
    if (u < 0.45) { const p = lerpV([-2.6, y], [-0.2, y], a); g.point(p, { r: 7, color: 'v' }); g.label('v', p, { dy: -18, color: 'v', size: 16 }); }
    if (u >= 0.5) { const p = lerpV([0.2, y], [2.6, y], b); g.point(p, { r: 7, color: 'v' }); g.label('f(v)', p, { dy: -18, color: 'v', size: 15 }); }
    // the small plane underneath
    g.viewport({ x: 0.08, y: 0.5, w: 0.84, h: 0.46 }, { view: '2d', cam: { center: [3.2, 2.1], halfH: 2.9 }, bg: 'panel' }, (gg) => {
      gg.grid(null, { range: 8, alpha: 0.4 }); gg.axes();
      gg.point([3, 2], { r: 6, color: 'v', alpha: 0.4, label: 'v', size: 14, dx: 16, dy: 12 });
      gg.circle([3, 2], 0.16, { stroke: 'v', dashed: true });
      if (u >= 0.5) {
        const p = lerpV([3, 2], [6, 4], b);
        gg.vector([3, 2], p, { color: 'v', dashed: true, width: 1.6, head: 9 });
        gg.point(p, { r: 6.5, color: 'v', label: b >= 1 ? 'f(v) = [6; 4]' : undefined, size: 14, dx: -50, dy: -14 });
      }
    });
  }

  function sceneTranspose(g, s) {
    const f = LA.clamp01(+s.flip), rot = s.rotV, c = 0.85, o = [-2.75, 1.7];
    g.line(o, [o[0] + 2.75, o[1] - 2.75], { color: 'muted', dashed: true, width: 1.2 });
    g.label('diagonal', [o[0] + 2.75, o[1] - 2.75], { anchor: 'left', dx: 4, color: 'muted', size: 11.5, font: 'ui', bg: false });
    for (let i = 0; i < 2; i++) for (let j = 0; j < 3; j++) {
      // straight slide to the mirrored cell, nudged along the diagonal so swapping pairs pass side by side
      const side = Math.sign(j - i) * 0.32 * c * Math.sin(Math.PI * f);
      const p = N.add(lerpV(cellC(o, i, j, c), cellC(o, j, i, c), f), [side, -side]), role = COLS[j];
      rect(g, p[0], p[1], c * 0.94, c * 0.94, { fill: role, alpha: 0.14, stroke: role, width: 1 });
      g.label(LA.fmt(s.A[i][j]), p, { color: role, size: 19, font: 'math', bg: false });
    }
    const w = lerp(3, 2, f) * c, h = lerp(2, 3, f) * c;
    bracket(g, o[0] - 0.1, o[1] - h - 0.1, o[0] + w + 0.1, o[1] + 0.1);
    g.label(f < 0.5 ? 'A  (2×3)' : 'Aᵀ  (3×2)', [o[0] + w / 2, o[1] + 0.45], { color: 'text', size: 16, font: 'ui', bg: false });
    if (f > 0.02 && f < 0.98) g.label('row 2, col 3 → row 3, col 2', [o[0] + 1.2, o[1] - 3.25], { color: 'muted', size: 12, font: 'ui' });
    // the yellow vector lies down
    const ctr = [1.75, 0.4], ang = rot * Math.PI / 2, R = q => [ctr[0] + q[0] * Math.cos(ang) - q[1] * Math.sin(ang), ctr[1] + q[0] * Math.sin(ang) + q[1] * Math.cos(ang)];
    const vw = lerp(0.6, 1.25, rot), vh = lerp(1.25, 0.6, rot);
    bracket(g, ctr[0] - vw / 2, ctr[1] - vh / 2, ctr[0] + vw / 2, ctr[1] + vh / 2, { color: 'v' });
    g.label('3', R([0, 0.3]), { color: 'i', size: 19, font: 'math', bg: false });
    g.label('2', R([0, -0.3]), { color: 'j', size: 19, font: 'math', bg: false });
    g.label(rot < 0.5 ? 'v (standing)' : 'vᵀ (lying down)', [ctr[0], ctr[1] + 1.05], { color: 'v', size: 14, font: 'ui', bg: false });
  }

  function sceneInverse(g, s, ctx) {
    const k = Math.pow(2, s.movie), M = [[k, 0], [0, 1]], v = [3, 2], p = N.matVec(M, v);
    g.ghostGrid({ range: 10 });
    g.grid(M, { range: 10, alpha: 0.8 });
    g.circle(v, 0.17, { stroke: 'v', dashed: true });
    g.label('home', v, { dy: 20, color: 'muted', size: 11.5, font: 'ui' });
    g.point(p, { r: 7, color: 'v', label: 'v', size: 15 });
    if (s.scene === 'inverse') {
      const u = s.u;
      if (u < 0.5) g.hud('f : stretch sideways by 2', 16, 30, { size: 18, color: 'w', bold: true });
      else if (u >= 0.55) g.hud('f⁻¹ : the same movie backwards', 16, 30, { size: 18, color: 'w', bold: true });
      if (u >= 0.96) g.marker.check(v, { dx: 26, dy: -16 });
    }
  }

  function drawGreekPanel(g, k, active, r) {
    const al = active ? 1 : 0.3;
    if (k === 0) {
      g.vector([-1, -0.8], [1.2, -0.8], { color: 'hud', label: 'a', width: 2.2, alpha: al });
      g.vector([-1, -0.8], [0.4, 0.75], { color: 'hud', label: 'b', width: 2.2, alpha: al });
      if (al > 0.5) g.arc([-1, -0.8], [1, 0], [1.4, 1.55], { r: 0.75, color: 'gold', label: 'θ', size: 18 });
    } else if (k === 1) {
      const L = active ? lerp(1, 2.5, seg(r, 0.15, 0.8)) : 2.5;
      g.line([-1.3, -0.6], [-0.3, -0.6], { color: 'muted', dashed: true, width: 3, alpha: al });
      g.vector([-1.3, -0.6], [-1.3 + L, -0.6], { color: 'v', label: 'v', width: 2.6, alpha: al, dy: -16 });
      g.label('λ = ' + LA.fmt.fixed(L, 1), [0, 0.6], { color: 'gold', size: 17, font: 'ui', alpha: al });
    } else if (k === 2) {
      g.points(GREEK_CLOUD, { r: 3, color: 'data', alpha: 0.85 * al });
      g.circle([0, -0.1], 0.9, { stroke: 'gold', dashed: true, width: 1.6, strokeAlpha: al });
      g.line([0, -0.1], [0.9, -0.1], { color: 'gold', width: 1.6, alpha: al });
      g.label('σ', [0.45, -0.1], { dy: -12, color: 'gold', size: 17, alpha: al });
    } else {
      g.line([-1.4, -1.0], [1.4, 0.9], { color: 'muted', dashed: true, alpha: al });
      const st = active ? seg(r, 0.15, 0.7) : 1;
      const a = [-0.9, -0.66], b = lerpV(a, [-0.15, -0.15], st);
      g.point(a, { r: 5, color: 'b', alpha: al });
      if (st > 0.05) g.vector(a, b, { color: 'gold', width: 2.6, head: 9, alpha: al, label: 'η', labelAt: 0.5, labelOffset: 16 });
      g.label('a short step', [0.5, -0.9], { color: 'muted', size: 12, font: 'ui', alpha: al });
    }
  }
  const GREEK_CLOUD = LA.rng.gaussianCloud(5, 40, { mean: [0, -0.1], cov: [[0.3, 0], [0, 0.3]] });
  const GREEK = [['θ', 'theta: an angle'], ['λ', 'lambda: a stretch factor'], ['σ', 'sigma: a size or spread'], ['η', 'eta: a step size']];

  function sceneGreek(g, s) {
    const u = s.u, act = Math.min(3, Math.floor(u * 4)), r = u * 4 - act;
    for (let k = 0; k < 4; k++) {
      const x = k % 2 ? 0.51 : 0.03, y = k < 2 ? 0.03 : 0.51;
      g.viewport({ x, y, w: 0.46, h: 0.46 }, { view: '2d', cam: { center: [0, -0.1], halfH: 1.75 }, bg: k === act ? 'panel2' : 'panel' }, (gg, rc) => {
        drawGreekPanel(gg, k, k === act, r);
        gg.hud(GREEK[k][0], rc.x + 12, rc.y + 22, { size: 24, color: k === act ? 'gold' : 'muted', font: 'math' });
        gg.hud(GREEK[k][1], rc.x + 36, rc.y + 22, { size: 13, color: k === act ? 'text' : 'muted' });
      });
    }
  }

  function sceneFormula(g, s) {
    const u = s.u, Y = 2.75;
    const X = { yh: -2.4, eq: -1.75, W: -0.95, x: 0.15, plus: 0.85, b: 1.55 };
    g.label('y', [X.yh, Y], { color: 'gold', size: 34, font: 'math', hat: true, bg: false });
    g.label('=', [X.eq, Y], { color: 'text', size: 30, font: 'math', bg: false });
    g.label('W', [X.W, Y], { color: 'text', size: 34, font: 'math', bg: false });
    g.label('x', [X.x, Y], { color: 'v', size: 34, font: 'math', bg: false });
    g.label('+', [X.plus, Y], { color: 'text', size: 30, font: 'math', bg: false });
    g.label('b', [X.b, Y], { color: 'b', size: 34, font: 'math', bg: false });
    // shape tiles
    rect(g, X.yh, 1.55, 0.2, 0.6, { fill: 'gold', alpha: 0.35, stroke: 'gold' });
    rect(g, X.W, 1.55, 1.3, 0.6, { fill: 'muted', alpha: 0.12, stroke: 'hud', width: 1.4 });
    g.label('10 rows · 784 columns', [X.W, 1.55], { color: 'muted', size: 10.5, font: 'ui', bg: false });
    g.heat2d(DIGIT7.map(x => [x]), { x0: X.x - 0.1, x1: X.x + 0.1, y0: 0.45, y1: 2.25 }, { range: [0, 1], ramp: grayRamp, smooth: false });
    g.polyline([[X.x - 0.1, 0.45], [X.x + 0.1, 0.45], [X.x + 0.1, 2.25], [X.x - 0.1, 2.25]], { color: 'v', closed: true, width: 1.2 });
    rect(g, X.b, 1.55, 0.2, 0.6, { fill: 'b', alpha: 0.35, stroke: 'b' });
    g.label('10×1', [X.yh, 0.95], { color: 'gold', size: 13, font: 'ui', bg: false });
    g.label('10×784', [X.W, 0.95], { color: 'hud', size: 13, font: 'ui', bg: false });
    g.label('784×1', [X.x + 0.45, 0.2], { color: 'v', size: 13, font: 'ui', bg: false });
    g.label('10×1', [X.b, 0.95], { color: 'b', size: 13, font: 'ui', bg: false });
    g.label('synthetic “7”', [X.x + 0.2, 2.25], { anchor: 'left', dx: 6, color: 'muted', size: 11, font: 'ui', bg: false });
    // the inner 784s slide together and snap
    const m = seg(u, 0.5, 0.8), yS = -0.35;
    const L = lerp(-1.75, -0.8, m), R = lerp(1.05, -0.06, m);   // end with ")(" side by side, not crossed
    g.label('(10×', [L - 0.44, yS], { color: 'hud', size: 18, font: 'ui', bg: false });
    g.label('784', [L, yS], { color: m >= 1 ? 'ok' : 'gold', size: 18, font: 'ui', bold: true, bg: false });
    g.label(')', [L + 0.3, yS], { color: 'hud', size: 18, font: 'ui', bg: false });
    g.label('(', [R - 0.3, yS], { color: 'hud', size: 18, font: 'ui', bg: false });
    g.label('784', [R, yS], { color: m >= 1 ? 'ok' : 'gold', size: 18, font: 'ui', bold: true, bg: false });
    g.label('×1)', [R + 0.4, yS], { color: 'hud', size: 18, font: 'ui', bg: false });
    if (m >= 1) g.label('→  10×1', [1.05, yS], { anchor: 'left', color: 'gold', size: 18, font: 'ui', bold: true, bg: false });
    if (m >= 1) g.label('inner numbers match', [-0.43, yS - 0.45], { color: 'ok', size: 12, font: 'ui', bg: false });
    const o = seg(u, 0.85, 1);
    if (o > 0) {
      const vals = [0.05, 0.02, 0.08, 0.04, 0.03, 0.06, 0.02, 0.92 * o, 0.05, 0.03];
      g.bars(vals, { origin: [-1.75, -3.05], width: 0.26, gap: 0.09, height: 1.5, max: 1, labels: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], colors: vals.map((x, i) => i === 7 ? 'gold' : 'muted') });
      g.label('ŷ: 10 scores, the biggest at 7', [0, -1.35], { color: 'gold', size: 12.5, font: 'ui' });
    }
  }

  function sceneRecap(g, s) {
    const cards = [
      { title: 'Addresses: vᵢ, aᵢⱼ — row first', cam: { center: [0, -0.1], halfH: 1.25 }, draw(gg) {
        drawMatrix(gg, A0, [-1.05, 0.6], 0.7, { size: 16 });
        gg.circle(cellC([-1.05, 0.6], 1, 2, 0.7), 0.3, { stroke: 'gold', width: 2.4 });
        gg.label('a' + sub(23) + ' = 8', [1.5, -0.45], { color: 'gold', size: 15, font: 'ui' });
      } },
      { title: 'Σ adds the strip; ℝⁿ is lists of n numbers', cam: { center: [0, -0.1], halfH: 1.25 }, draw(gg) {
        [2, -1, 3, 4].forEach((v, k) => { const p = k < 2 ? [-2.4 + k * 0.7, 0] : [1.6, -0.8 + 0.42 * (k - 2)]; rect(gg, p[0], p[1], 0.64, k < 2 ? 0.64 : 0.36, { fill: LA.color.diverge(v / 4), alpha: 0.95, stroke: 'line' }); gg.label(LA.fmt(v), p, { color: 'text', size: 13, font: 'math', bg: false }); });
        gg.vector([-1.2, 0], [0.9, -0.3], { color: 'hud', width: 1.6, head: 8 });
        gg.label('Σ = 8', [1.6, 0.35], { color: 'gold', size: 16, font: 'ui', bold: true });
      } },
      { title: 'ᵀ flips, ⁻¹ reverses, ‖v‖ measures', cam: { center: [1.2, 0.6], halfH: 1.25 }, draw(gg) {
        gg.ghostGrid({ range: 6 }); gg.grid([[1.6, 0], [0, 1]], { range: 6, alpha: 0.7 });
        gg.vector([0, 0], [1.5, 1], { color: 'v', label: 'v', width: 2.6 });
        gg.dim([0, 0], [1.5, 1], { offset: 12, color: 'v', label: '‖v‖', size: 12 });
        gg.label('f  ⇄  f⁻¹', [2.6, 1.55], { color: 'w', size: 15, font: 'ui', bold: true });
      } },
    ];
    cards.forEach((c, k) => {
      g.viewport({ x: 0.04, y: 0.03 + k * 0.325, w: 0.92, h: 0.295 }, { view: '2d', cam: c.cam, bg: 'panel' }, (gg, r) => {
        c.draw(gg);
        gg.hud(c.title, r.x + 12, r.y + 18, { size: 13, color: 'text', bold: true });
      });
    });
  }

  /* ---------------- checks scenes ---------------- */
  const CO = [-1.5, 1];   // check grid: top-left, 1-unit cells
  function sceneCheckGrid(g, s, ctx) {
    drawMatrix(g, s.A, CO, 1, { size: 22 });
    rowColLabels(g, CO, 2, 3, 1);
    g.label('A =', [CO[0] - 1.0, CO[1] - 1], { anchor: 'right', color: 'text', size: 20, font: 'math', bg: false });
    if (s.scene === 'check1') {
      for (let i = 0; i < 2; i++) for (let j = 0; j < 3; j++) if (Math.abs(s.A[i][j] - s.guess[0]) < 0.05) g.circle(cellC(CO, i, j, 1), 0.44, { stroke: 'gold', width: 2.8 });
      g.label('typed: ' + LA.fmt(s.guess[0]), [0, -1.8], { color: 'b', size: 15, font: 'ui' });
    } else {
      g.circle(s.ring, 0.44, { stroke: 'gold', width: 3 });
      g.handle('ring', s.ring, (q, st) => { st.ring = [LA.clamp(q[0], -3, 3), LA.clamp(q[1], -2.5, 2.5)]; }, { color: 'gold', r: 16 });
    }
  }
  const TILE = [{ t: '2×3', r: 2, c: 3 }, { t: '3×1', r: 3, c: 1 }, { t: '= 2×1', r: 2, c: 1 }];
  const SLOTS = [[-2, 0.8], [0, 0.8], [2, 0.8]];
  const slotOf = p => { for (let k = 0; k < 3; k++) if (N.dist(p, SLOTS[k]) < 0.3) return k; return -1; };
  function sceneTiles(g, s, ctx) {
    g.label('write the product here:', [-2.7, 1.7], { anchor: 'left', color: 'muted', size: 13, font: 'ui' });
    SLOTS.forEach((p, k) => { rect(g, p[0], p[1], 1.6, 0.8, { stroke: 'muted', width: 1.2, dash: [5, 4], alpha: 0.7 }); g.label('slot ' + (k + 1), [p[0], p[1] - 0.6], { color: 'muted', size: 11, font: 'ui', bg: false }); });
    g.line([-3, -1.25], [3, -1.25], { color: 'line', width: 2 });
    g.label('shelf (drag a tile)', [-2.7, -2.6], { anchor: 'left', color: 'muted', size: 12, font: 'ui' });
    const inSlot = [-1, -1, -1];
    s.tiles.forEach((p, k) => { const sl = slotOf(p); if (sl >= 0) inSlot[sl] = k; });
    s.tiles.forEach((p, k) => {
      const hot = ctx.hot === 'tile' + k;
      rect(g, p[0], p[1], 1.5, 0.7, { fill: hot ? 'panel' : 'panel2', alpha: 0.95, stroke: hot ? 'gold' : 'hud', width: hot ? 2.2 : 1.4 });
      g.label(TILE[k].t, p, { color: 'text', size: 19, font: 'ui', bold: true, bg: false });
      // the grip sits on the tile's left end so it does not cover the label
      g.handle('tile' + k, [p[0] - 0.55, p[1]], (q, st) => {
        let np = [LA.clamp(q[0] + 0.55, -3, 3), LA.clamp(q[1], -2.6, 2.4)];
        const sl = SLOTS.findIndex(sp => N.dist(np, sp) < 0.45);   // snap into a slot when close
        if (sl >= 0) np = SLOTS[sl].slice();
        st.tiles[k] = np;
      }, { color: 'white', r: 18 });
    });
    if (inSlot[0] >= 0 && inSlot[1] >= 0) {
      const L = TILE[inSlot[0]], R = TILE[inSlot[1]];
      const ok = inSlot[1] !== 2 && inSlot[0] !== 2 && L.c === R.r;
      if (ok) g.label(L.c + ' = ' + R.r + ' · inner numbers touch', [-1, 1.65], { color: 'ok', size: 13, font: 'ui' });
      else { g.marker.cross([-1, 0.8], { size: 20 }); g.label('inner numbers do not match', [-1, 1.65], { color: 'bad', size: 13, font: 'ui' }); }
    }
  }

  /* ---------------- playground scene ---------------- */
  function scenePlay(g, s, ctx) {
    const f = +s.flip > 0.5 ? 1 : 0, c = 0.8, o = [-3.0, 3.9];
    const M = f ? N.transpose(s.A) : s.A;
    drawMatrix(g, M, o, c, { size: 17, rowsTint: !!f });
    const ci = s.cell ? s.cell[0] : -1, cj = s.cell ? s.cell[1] : -1;
    if (s.cell) {
      const p = f ? cellC(o, cj, ci, c) : cellC(o, ci, cj, c);
      g.circle(p, 0.36, { stroke: 'gold', width: 2.6 });
      g.label('a' + sub(String(ci + 1) + String(cj + 1)) + ' = ' + LA.fmt(s.A[ci][cj]), [0.6, 3.4], { anchor: 'left', color: 'gold', size: 16, font: 'ui', bold: true });
    }
    g.label(f ? 'Aᵀ (3×2)' : 'A (2×3)', [0.6, 2.8], { anchor: 'left', color: 'text', size: 14, font: 'ui' });
    // strip and its sum
    const n = Math.round(s.n), vals = s.stripAll.slice(0, n), w = Math.min(0.62, 6.4 / n), y = 0.85;
    vals.forEach((v, k) => {
      const x = -3.1 + (k + 0.5) * w;
      rect(g, x, y, w * 0.94, 0.62, { fill: LA.color.diverge(v / 5), alpha: 0.95, stroke: 'line' });
      g.label(LA.fmt(v), [x, y], { color: 'text', size: 13, font: 'math', bg: false });
    });
    g.label('Σ = ' + LA.fmt(N.sum(vals)) + '   (n = ' + n + ', a point in ℝ' + (n < 10 ? SUPS[n] : n === 10 ? '¹⁰' : n === 11 ? '¹¹' : '¹²') + ')', [-3.1, y - 0.6], { anchor: 'left', color: 'gold', size: 14, font: 'ui' });
    // f / f⁻¹ movie
    const k = Math.pow(2, s.movie);
    g.viewport({ x: 0.05, y: 0.6, w: 0.9, h: 0.37 }, { view: '2d', cam: { center: [1.6, 0.9], halfH: 1.9 }, bg: 'panel' }, (gg, r) => {
      gg.ghostGrid({ range: 8 }); gg.grid([[k, 0], [0, 1]], { range: 8, alpha: 0.8 });
      gg.circle([1.5, 1], 0.12, { stroke: 'v', dashed: true });
      gg.point([1.5 * k, 1], { r: 6, color: 'v', label: 'v', size: 14 });
      gg.hud(s.movie >= 0 ? 'f played ' + LA.fmt(s.movie) + ' of the way: stretch ×' + LA.fmt(k) : 'f⁻¹ played ' + LA.fmt(-s.movie) + ' of the way: stretch ×' + LA.fmt(k), r.x + 10, r.y + 18, { size: 13, color: 'w' });
    });
  }
  const SUPS = ['⁰', '¹', '²', '³', '⁴', '⁵', '⁶', '⁷', '⁸', '⁹'];

  const SCENES = { bracket: scenePoint, sub: scenePoint, norm: scenePoint, address: sceneAddress, shape: sceneShape, sum: sceneSum, R: sceneR,
    func: sceneFunction, transpose: sceneTranspose, inverse: sceneInverse, greek: sceneGreek, formula: sceneFormula, recap: sceneRecap,
    check1: sceneCheckGrid, check2: sceneCheckGrid, tiles: sceneTiles, play: scenePlay };

  LA.registerLesson({
    id: 'p0-02-reading-the-symbols',
    part: 0,
    title: 'How to read the symbols',
    subtitle: 'Brackets, subscripts, Σ, ℝⁿ, f(x), ᵀ, ⁻¹, ‖v‖ — each one is the name of a picture.',
    minutes: 10,
    prereqs: ['p0-01-data-is-geometry'],
    tags: ['primer'],
    badge: { drawnIn: '2D', realSize: 'a digit classifier is written ŷ = Wx + b with W 10×784, x 784×1, b 10×1', transfers: 'the symbols mean exactly the same thing at any size; only the number of cells grows' },
    cantShow: ['a 784-cell strip at readable size (we draw 784 thin cells and print the first 12 values)'],
    terms: [
      { term: 'vector', def: 'A list of numbers written in a tall bracket; also an arrow or a point.' },
      { term: 'entry', def: 'One number inside a bracket.' },
      { term: 'subscript', def: 'The small number written below a letter that says which entry is meant.' },
      { term: 'matrix', def: 'A rectangular grid of numbers in a bracket.' },
      { term: 'shape', def: 'The size of a matrix written rows × columns, rows first.' },
      { term: 'sum', def: 'The result of adding up every entry in a strip; written with the capital Greek letter Σ.' },
      { term: 'real numbers', def: 'All the ordinary numbers on a number line, written ℝ.' },
      { term: 'ℝⁿ', def: 'All lists of n real numbers: the space our points live in.' },
      { term: 'function', def: 'A rule that turns an input into exactly one output.' },
      { term: 'transpose', def: 'Flipping a matrix across its diagonal so rows become columns.' },
      { term: 'inverse', def: 'The rule that undoes another rule.' },
      { term: 'norm', def: 'The length of a vector.' },
    ],
    atlas: ['mlp', 'tensors'],
    thumbnail: 2,

    state() {
      return {
        view: '2d', cam: camPoint(), scene: 'bracket', u: 0, panel: 'v',
        v: [3, 2], A: LA.clone(A0), cell: [1, 2], stripAll: STRIP12.slice(), n: 4, seed: 0,
        flip: 0, rotV: 0, movie: 0, ring: [-1, -0.5], tiles: [[-2, -1.9], [0, -1.9], [2, -1.9]], guess: [0],
      };
    },

    render(g, s, ctx) { (SCENES[s.scene] || scenePoint)(g, s, ctx); },

    math(s, tex) {
      const T = (t, o) => tex.text(t, o), u = s.u;
      const v = () => tex.vec([3, 2], { colors: 'rows' });
      switch (s.panel) {
        case 'v': return tex.block(tex.eq(tex.sym('v', { color: 'v' }), tex.op('='), v()));
        case 'sub': {
          const parts = [tex.eq(tex.sym('v', { color: 'v', sub: '1' }), tex.op('='), tex.num(3, { color: 'i' }))];
          if (u >= 0.5) parts.push(tex.eq(tex.sym('v', { color: 'v', sub: '2' }), tex.op('='), tex.num(2, { color: 'j' })));
          return tex.block.apply(null, parts);
        }
        case 'address': return tex.block(tex.eq(tex.sym('A'), tex.op('='), tex.mat(s.A, { colors: 'cols', highlight: u > 0.7 ? [1, 2] : null })), u > 0.7 ? tex.eq(tex.sym('a', { sub: '23' }), tex.op('='), numT(tex, 8, { color: 'gold' })) : '');
        case 'shape': return tex.block(tex.eq(T('A is'), tex.dim(2, 3)), tex.eq(tex.op('·')), tex.eq(T('B is'), tex.dim(3, 2)), tex.eq(T('(rows × columns)', { color: 'muted' })));
        case 'sum': {
          const vals = [2, -1, 3, 4]; let n = 0; vals.forEach((x, k) => { if (seg(u, 0.1 + 0.2 * k, 0.28 + 0.2 * k) >= 1) n++; });
          const parts = [tex.sym('Σ', { sub: 'i' }), tex.sym('s', { sub: 'i' }), tex.op('=')];
          vals.slice(0, n).forEach((x, k) => { if (k) parts.push(tex.op('+')); parts.push(x < 0 ? tex.paren(tex.num(x)) : tex.num(x)); });
          if (n === 4) parts.push(tex.op('='), tex.num(8, { color: 'gold' })); else parts.push(tex.op('…'));
          return tex.block(tex.eq.apply(null, parts));
        }
        case 'R': {
          const ph = u < 0.3 ? 0 : u < 0.6 ? 1 : u < 0.8 ? 2 : 3, names = ['ℝ', 'ℝ²', 'ℝ³', 'ℝⁿ'], parts = [];
          names.forEach((nm, k) => { if (k) parts.push(tex.op('·')); parts.push(tex.num(nm, { color: k === ph ? 'gold' : 'muted' })); });
          parts.push(T(': a list of n numbers'));
          return tex.block(tex.eq.apply(null, parts));
        }
        case 'func': return tex.lines(tex.eq(tex.sym('f'), tex.paren(tex.sym('v', { color: 'v' })), tex.op('='), numT(tex, 2), tex.sym('v', { color: 'v' })), u >= 0.5 ? tex.eq(tex.sym('f'), tex.paren(v()), tex.op('='), tex.vec([6, 4], { colors: 'rows' })) : '');
        case 'transpose': return tex.block(tex.eq(tex.T(tex.sym('A')), tex.op('='), tex.mat(N.transpose(s.A), { colors: 'rows' })), tex.eq(tex.T(tex.sym('v', { color: 'v' })), tex.op('='), tex.rowvec([3, 2], { colors: ['i', 'j'] })));
        case 'inverse': {
          const num = tex.eq(T('2⁻¹'), tex.op('·'), T('2'), tex.op('='), T('1'), T('(½ undoes ×2)', { color: 'muted' }));
          return u < 0.5 ? tex.block(num) : tex.block(num, tex.eq(tex.sym('f', { sup: '−1' }), tex.paren(tex.sym('f') + tex.paren(tex.sym('v', { color: 'v' }))), tex.op('='), tex.sym('v', { color: 'v' })));
        }
        case 'norm': return tex.block(tex.eq(tex.norm(tex.sym('v', { color: 'v' })), tex.op('≈'), tex.num(u >= 0.7 ? '3.61' : '?', { color: 'v' })));
        case 'greek': {
          const act = Math.min(3, Math.floor(u * 4)), parts = [];
          GREEK.forEach((gk, k) => { if (k) parts.push(tex.op('·')); parts.push(tex.num(gk[0] + ' ' + gk[1].split(': ')[1], { color: k === act ? 'gold' : 'muted' })); });
          return tex.block(tex.eq.apply(null, parts));
        }
        case 'formula': return tex.lines(
          tex.eq(tex.sym('y', { hat: true, color: 'gold' }), tex.op('='), tex.sym('W'), tex.sym('x', { color: 'v' }), tex.op('+'), tex.sym('b', { color: 'b' })),
          tex.eq(tex.dim(10, 1), tex.op('  '), tex.dim(10, 784), tex.dim(784, 1), tex.op('  '), tex.dim(10, 1)));
        case 'play': {
          const parts = [tex.eq(tex.sym('A'), tex.op('='), tex.mat(s.A, { colors: 'cols', highlight: s.cell || null }))];
          if (s.cell) parts.push(tex.eq(tex.sym('a', { sub: String(s.cell[0] + 1) + String(s.cell[1] + 1) }), tex.op('='), numT(tex, s.A[s.cell[0]][s.cell[1]], { color: 'gold' })));
          const vals = s.stripAll.slice(0, Math.round(s.n));
          parts.push(tex.eq(tex.sym('Σ', { sub: 'i' }), tex.sym('s', { sub: 'i' }), tex.op('='), numT(tex, N.sum(vals), { color: 'gold' })));
          return tex.block.apply(null, parts);
        }
        case 'check1': return tex.block(tex.eq(tex.sym('a', { sub: '23' }), tex.op('='), tex.num(s.guess[0], { color: 'b' }), T('?')));
        case 'check2': {
          const j = Math.round(s.ring[0] + 1), i = Math.round(0.5 - s.ring[1]);
          const on = i >= 0 && i < 2 && j >= 0 && j < 3 && N.dist(s.ring, [-1 + j, 0.5 - i]) < 0.35;
          return tex.block(tex.eq(T('ring on'), on ? tex.sym('a', { sub: String(i + 1) + String(j + 1) }) : T('—')));
        }
        case 'tiles': return tex.block(tex.eq(T('(rows × columns)(rows × columns): the inner numbers must match')));
        default: return '';
      }
    },

    steps: [
      { title: 'A bracket holds a point', duration: 3,
        narration: 'Here is the point from the last lesson, 3 along the green axis and 2 along the red. Write the two numbers in a tall bracket, green on top, red below. That bracket is a <b>vector</b>: a list of numbers that is also a point and also an arrow. We name it ' + vhat + '.',
        setup(s) { s.view = '2d'; s.cam = camPoint(); s.scene = 'bracket'; s.panel = 'v'; },
        at(s, t) { s.u = t; } },

      { title: 'Subscripts say which entry', duration: 3,
        narration: 'The small number under a letter is a <b>subscript</b>. It picks an <b>entry</b>: ' + vhat + '<sub>1</sub> is the first entry, 3, the green one; ' + vhat + '<sub>2</sub> is the second, 2, the red one. Subscripts are addresses, not powers and not multiplication.',
        setup(s) { s.scene = 'sub'; s.panel = 'sub'; },
        at(s, t) { s.u = t; } },

      { title: 'Two subscripts: row, then column', duration: 4,
        narration: 'A grid of numbers in a bracket is a <b>matrix</b>, written with a capital letter, <i>A</i>. Each entry needs two addresses: <i>a</i><sub>23</sub> means row 2, column 3 — the row number always comes first. Read it like a street grid: go down to the row, then across to the column.',
        setup(s) { s.cam = camMid(); s.scene = 'address'; s.panel = 'address'; s.A = LA.clone(A0); },
        at(s, t) { s.u = t; } },

      { title: 'Shape: rows × columns', duration: 4, misconception: true,
        narration: 'The <b>shape</b> of a matrix is rows × columns, rows first: this <i>A</i> is 2×3, two rows and three columns. Is a 3×2 matrix the same thing? No. Flip the roles and the bracket is tall instead of wide; the entries live in different cells. Shape is read in a fixed order.',
        setup(s) { s.cam = { center: [0.1, 0], halfH: 3.75 }; s.scene = 'shape'; s.panel = 'shape'; },
        at(s, t) { s.u = t; } },

      { title: 'Σ means “add the strip”', duration: 4,
        narration: 'A row of four numbers: 2, −1, 3, 4. The capital Greek letter Σ (sigma) means “add them all up”: the <b>sum</b>. Written out, Σ<sub><i>i</i></sub> <i>s</i><sub><i>i</i></sub> says: let <i>i</i> run over every address, and add the entry at each one. Watch the cells slide onto a pile and the total grow.',
        setup(s) { s.cam = { center: [0.3, 0.3], halfH: 3.2 }; s.scene = 'sum'; s.panel = 'sum'; },
        at(s, t) { s.u = t; } },

      { title: 'ℝ, ℝ², ℝⁿ', duration: 6,
        narration: '<b>ℝ</b> is the set of all <b>real numbers</b>, the number line. ℝ² is all pairs of numbers — the whole plane, where ' + vhat + ' lives. ℝ³ is all triples, our 3D room. <b>ℝⁿ</b> is all lists of n numbers; ℝ⁷⁸⁴ holds one synthetic digit image. The picture stops at 3. The formula does not.',
        setup(s) { s.cam = { center: [1.6, 1.4], halfH: 3.5 }; s.scene = 'R'; s.panel = 'R'; },
        at(s, t) { s.u = t; } },

      { title: 'f(x): a rule with an input and an output', duration: 4,
        narration: '<b>f(x)</b> reads “f of x”: the rule <i>f</i> applied to the input <i>x</i>. It is not multiplication. A rule like this is a <b>function</b>. Here <i>f</i> doubles the distance from the origin. Feed it the point ' + vhat + ' and out comes <i>f</i>(' + vhat + '), a new point. Later, rules that move points are called transformations.',
        setup(s) { s.cam = { center: [0, 0], halfH: 3.6 }; s.scene = 'func'; s.panel = 'func'; },
        at(s, t) { s.u = t; } },

      { title: 'ᵀ flips rows and columns', duration: 4,
        narration: 'The small ᵀ means <b>transpose</b>: flip the matrix across its diagonal, so each row becomes a column. The 2×3 matrix <i>A</i> becomes the 3×2 matrix <i>A</i><sup>ᵀ</sup>, and the entry at row 2, column 3 moves to row 3, column 2. A standing vector transposed is a lying-down vector.',
        setup(s) { s.cam = camMid(); s.scene = 'transpose'; s.panel = 'transpose'; s.A = LA.clone(A0); s.flip = 0; s.rotV = 0; },
        at(s, t) { s.u = t; s.flip = seg(t, 0.05, 0.72); s.rotV = seg(t, 0.8, 1); } },

      { title: '⁻¹ plays the movie backwards', duration: 4,
        narration: 'The small ⁻¹ means <b>inverse</b>: the rule that undoes another rule. On numbers, 2<sup>−1</sup> = ½ undoes multiplying by 2. On pictures, if <i>f</i> stretches the grid, <i>f</i><sup>−1</sup> is the same movie played backwards, and <i>f</i><sup>−1</sup>(<i>f</i>(' + vhat + ')) brings ' + vhat + ' home.',
        setup(s) { s.cam = { center: [3.1, 1.5], halfH: 3.8 }; s.scene = 'inverse'; s.panel = 'inverse'; s.flip = 0; s.rotV = 0; s.movie = 0; },
        at(s, t) { s.u = t; s.movie = seg(t, 0, 0.45) - seg(t, 0.55, 1); } },

      { title: '‖v‖ is the length', duration: 3,
        narration: 'Double bars around a vector, ‖' + vhat + '‖, mean its <b>norm</b>: its length — how long the arrow is, measured with a tape from the origin to the tip. For [3; 2] it is about 3.61. How to compute it from the entries is the subject of a later lesson; for now, learn to read it.',
        setup(s) { s.cam = camPoint(); s.scene = 'norm'; s.panel = 'norm'; s.movie = 0; },
        at(s, t) { s.u = t; } },

      { title: 'Greek letters are just names', duration: 6,
        narration: 'Greek letters are names for numbers, chosen by habit. θ (theta) is an angle. λ (lambda) is a stretch factor. σ (sigma, lowercase) is a size or spread. η (eta) is a step size. You already met capital Σ, which is an instruction, not a name. Each shows next to its picture.',
        setup(s) { s.scene = 'greek'; s.panel = 'greek'; },
        at(s, t) { s.u = t; } },

      { title: 'One line of ML, symbol by symbol', duration: 5, mlTie: true,
        narration: 'Here is the formula of a digit classifier: ŷ = <i>W</i><span class="c-v">x</span> + <span class="c-b">b</span>. You can now read every mark. <span class="c-v">x</span> is a vector in ℝ<sup>784</sup>; <i>W</i> is a 10×784 matrix; <span class="c-b">b</span> is a vector of 10; ŷ (read “y-hat”) is the output vector. The shapes chain: (10×784)(784×1) gives 10×1. Part 2 teaches what the multiplication does.',
        setup(s) { s.cam = { center: [0, 0], halfH: 3.6 }; s.scene = 'formula'; s.panel = 'formula'; },
        at(s, t) { s.u = t; } },

      { title: 'Recap', duration: 3, recap: true,
        narration: 'You can now read the course’s alphabet. Brackets hold vectors and matrices; subscripts are addresses, row before column; shape is rows × columns. Σ adds a strip, ℝⁿ is “lists of n numbers”, f(x) is a rule applied, ᵀ flips, ⁻¹ reverses, ‖v‖ measures. When a symbol appears later, its picture will appear first.',
        cards: ['<b>Addresses:</b> v<sub>i</sub>, a<sub>ij</sub> — row first; shape is rows × columns.', '<b>Σ adds the strip;</b> ℝⁿ is “lists of n numbers”; f(x) is a rule applied to x.', '<b>ᵀ flips, ⁻¹ reverses, ‖v‖ measures</b> the length.'],
        setup(s) { s.scene = 'recap'; s.panel = 'none'; s.cam = camMid(); },
        at(s, t) { s.u = t; } },
    ],

    playground: {
      intro: '<p>Edit the matrix, ring an entry, grow the strip and watch Σ, flip with ᵀ, and scrub the <i>f</i> / <i>f</i><sup>−1</sup> movie.</p>',
      setup(s) { s.scene = 'play'; s.panel = 'play'; s.cam = { center: [0, 0], halfH: 4.4 }; s.u = 1; s.flip = false; s.movie = 0; s.cell = [1, 2]; },
      controls: [
        { type: 'matrix', bind: 'A', label: 'Matrix A', colors: 'cols', step: 1, digits: 0, min: -9, max: 9 },
        { type: 'preset', label: 'Highlight entry', items: [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]].map(c => ({ label: 'a' + sub(String(c[0] + 1) + String(c[1] + 1)), apply(s) { s.cell = c.slice(); } })) },
        { type: 'slider', bind: 'n', min: 2, max: 12, step: 1, label: 'Strip length n', color: 'gold', fmt: x => String(Math.round(x)) },
        { type: 'toggle', bind: 'flip', label: 'Transpose (ᵀ)' },
        { type: 'slider', bind: 'movie', min: -1, max: 1, step: 0.01, label: 'Play f (right) / f⁻¹ (left)', color: 'w', fmt: x => LA.fmt.fixed(x, 2) },
        { type: 'preset', label: 'Presets', items: [
          { label: 'Identity-ish', apply(s) { s.A = [[1, 0, 0], [0, 1, 0]]; s.flip = false; } },
          { label: 'Random strip', apply(s) { s.seed = (s.seed || 0) + 1; const r = LA.rng(100 + s.seed); s.stripAll = STRIP12.map(() => LA.rng.int(r, -4, 5)); s.n = 8; } },
          { label: 'Tall 3×2', apply(s) { s.flip = true; } },
          { label: 'Lesson A', apply(s) { s.A = LA.clone(A0); s.flip = false; s.stripAll = STRIP12.slice(); s.n = 4; s.movie = 0; } },
        ] },
      ],
    },

    checks: [
      { kind: 'entry',
        prompt: 'Type the entry <i>a</i><sub>23</sub> of the matrix on screen. A gold ring marks every cell holding the number you typed.',
        bind: 'guess', answer: [7], tol: LA.TOL.entry, label: 'a₂₃', colors: ['b'],
        hint: 'Row first: the second row. Then across to the third column.',
        setup(s) { s.scene = 'check1'; s.panel = 'check1'; s.cam = camMid(); s.A = LA.clone(ACHECK); s.guess = [0]; } },
      { kind: 'manipulate',
        prompt: 'Drag the gold ring onto the entry <i>a</i><sub>13</sub>.',
        setup(s) { s.scene = 'check2'; s.panel = 'check2'; s.cam = camMid(); s.A = LA.clone(ACHECK); s.ring = [-1, -0.5]; },
        judge(s) { return LA.judge.near(s.ring, [1, 0.5], LA.TOL.pos); },
        hint: 'Row 1 is the top row; column 3 is the rightmost.',
        solution(s) { s.ring = [1, 0.5]; } },
      { kind: 'manipulate',
        prompt: 'Drag the three shape tiles into the slots so the product (2×3)(3×1) is written in a valid order.',
        setup(s) { s.scene = 'tiles'; s.panel = 'tiles'; s.cam = { center: [0, 0], halfH: 3.9 }; s.tiles = [[2, -1.9], [-2, -1.9], [0, -1.9]]; },
        judge(s) {
          let right = 0; s.tiles.forEach((p, k) => { if (N.dist(p, SLOTS[k]) < 0.3) right++; });
          return { ok: right === 3, progress: right / 3 * 0.95 };
        },
        hint: 'The inner numbers must touch: columns of the left tile = rows of the right.',
        solution(s) { s.tiles = SLOTS.map(p => p.slice()); } },
    ],
  });
})();
