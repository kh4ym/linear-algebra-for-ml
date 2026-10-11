/* p0-03 — Angles, the unit circle, cos and sin. (Part 0 primer) */
(function () {
  'use strict';
  const N = LA.num, seg = LA.seg, lerp = LA.lerp;
  const RAD = Math.PI / 180;
  const P = d => [Math.cos(d * RAD), Math.sin(d * RAD)];
  const f2 = x => LA.fmt(Math.abs(x) < 0.005 ? 0 : +x.toFixed(2));
  const deg = d => LA.fmt.fixed(d, 0) + '°';
  const LANDMARKS = [0, 30, 45, 60, 90, 180, 270];
  const HOPS = [0, 90, 180, 270, 45, 60, 30];
  const ihat = '<span class="c-i">î</span>', jhat = '<span class="c-j">ĵ</span>';
  const camCircle = () => ({ center: [0, 0], halfH: 1.75 });
  const camChart = () => ({ center: [0, -0.78], halfH: 2.05 });
  const camTable = () => ({ center: [0.62, 0], halfH: 2.2 });
  const T = (tex, t, o) => tex.text(t, o);

  /* ---------------- helpers ---------------- */
  function arcPts(r, a0, a1, c) { c = c || [0, 0]; const n = Math.max(6, Math.ceil(Math.abs(a1 - a0) / 4)), pts = []; for (let k = 0; k <= n; k++) { const a = lerp(a0, a1, k / n) * RAD; pts.push([c[0] + r * Math.cos(a), c[1] + r * Math.sin(a)]); } return pts; }
  function arc(g, r, a0, a1, o) { if (Math.abs(a1 - a0) < 0.3) return; g.polyline(arcPts(r, a0, a1, o.c), { color: o.color || 'gold', width: o.width || 1.8, alpha: o.alpha === undefined ? 1 : o.alpha, dashed: o.dashed }); }
  function frame(g, o) {
    o = o || {};
    g.grid(null, { range: 3, step: 0.5, alpha: 0.3 });
    const R = o.R || 1.45;
    g.line([-R, 0], [R, 0], { color: 'gridAxis', width: 1.4 }); g.line([0, -R], [0, R], { color: 'gridAxis', width: 1.4 });
    [[1, 0, '1', 'below', 0, 8], [-1, 0, '−1', 'below', 0, 8], [0, 1, '1', 'right', -8, 0], [0, -1, '−1', 'right', -8, 0]].forEach(q => {
      g.line([q[0] - (q[1] ? 0.03 : 0), q[1] - (q[0] ? 0.03 : 0)], [q[0] + (q[1] ? 0.03 : 0), q[1] + (q[0] ? 0.03 : 0)], { color: 'gridAxis', width: 1.4 });
      g.label(q[2], [q[0], q[1]], { anchor: q[3], dx: q[4] + (q[3] === 'below' ? 9 : 0), dy: q[5], color: 'tick', size: 11.5, font: 'ui', bg: false });
    });
  }
  function bracketAt(g, c, top, bot, o) {
    o = o || {};
    const w = o.w || 0.42, h = o.h || 0.25, x0 = c[0] - w / 2, x1 = c[0] + w / 2, y0 = c[1] - h, y1 = c[1] + h, lip = 0.05;
    g.polyline([[x0 + lip, y1], [x0, y1], [x0, y0], [x0 + lip, y0]], { color: o.color || 'hud', width: 1.6 });
    g.polyline([[x1 - lip, y1], [x1, y1], [x1, y0], [x1 - lip, y0]], { color: o.color || 'hud', width: 1.6 });
    g.label(top, [c[0], c[1] + h * 0.47], { color: 'i', size: o.size || 14, font: 'ui', bg: false });
    g.label(bot, [c[0], c[1] - h * 0.47], { color: 'j', size: o.size || 14, font: 'ui', bg: false });
  }
  /** cos/sin drops for the unit point at angle th. prog 0..1: green half, then red half. */
  function drops(g, th, prog, o) {
    o = o || {};
    const p = P(th), c = p[0], s = p[1], a = seg(prog, 0, 0.5), b = seg(prog, 0.5, 1);
    if (a > 0) {
      g.line(p, [c, 0], { color: 'i', dashed: [4, 4], width: 1.3, alpha: a });
      g.line([0, 0], [c * a, 0], { color: 'i', width: 4 });
      if (a >= 1 && o.labels !== false && Math.abs(c) > 0.02) g.dim([0, 0], [c, 0], { offset: (c >= 0 ? 1 : -1) * 20, color: 'i', label: (o.short ? '' : 'cos θ = ') + f2(c), size: 13 });
    }
    if (b > 0) {
      g.line([c, 0], [c, s * b], { color: 'j', width: 4 });
      if (b >= 1 && o.labels !== false && o.sinLabel !== false && Math.abs(s) > 0.02) {
        const sgn = c >= 0 ? 1 : -1;
        g.dim([c, 0], [c, s], { offset: sgn * (s >= 0 ? 1 : -1) * 14, color: 'j' });
        g.label((o.short ? '' : 'sin θ = ') + f2(s), [c, s / 2], { anchor: sgn > 0 ? 'left' : 'right', dx: sgn * 24, color: 'j', size: 13, font: 'ui' });
      }
    }
  }
  function thetaArc(g, th, o) {
    o = o || {};
    arc(g, o.r || 0.32, 0, th, { color: 'gold', width: 2 });
    if (o.label !== false && th > 3) { const m = P(th / 2), r = (o.r || 0.32) + 0.17; g.label(o.text || ('θ = ' + deg(th)), [m[0] * r, m[1] * r], { color: 'gold', size: 13, font: 'ui', anchor: th / 2 > 100 ? 'right' : 'left', dx: th / 2 > 100 ? -2 : 2 }); }
  }
  function unitPoint(g, th, o) {
    o = o || {};
    const p = P(th);
    g.line([0, 0], p, { color: 'v', width: 1.6, alpha: o.alpha === undefined ? 0.9 : o.alpha });
    g.point(p, { r: 6.5, color: 'v', alpha: o.alpha === undefined ? 1 : o.alpha });
  }
  function pointHandle(g, id, th, key) {
    g.handle(id, P(th), (q, st) => { let d = Math.atan2(q[1], q[0]) / RAD; if (d < 0) d += 360; st[key] = Math.round(d * 10) / 10; }, { color: 'v', r: 16 });
  }

  /* ---------------- scenes ---------------- */
  function sceneTurn(g, s) {
    const u = s.u, th = 40 * seg(u, 0, 0.5), gh = 40 + 320 * seg(u, 0.5, 1, 'inOut');
    frame(g);
    g.vector([0, 0], [1, 0], { color: 'i', label: 'i', hat: true, width: 3 });
    if (u > 0.5) {
      arc(g, 0.62, 40, gh, { color: 'muted', width: 1.2, dashed: true });
      const q = P(gh);
      g.line([0, 0], q, { color: 'muted', width: 2, dashed: true });
      if (gh >= 90) { g.polyline([[0.13, 0], [0.13, 0.13], [0, 0.13]], { color: 'hud', width: 1.3 }); g.label('90° · a quarter turn', [0.05, 1.12], { anchor: 'left', color: 'hud', size: 12.5, font: 'ui' }); }
      if (gh >= 180) g.label('180° · a half turn', [-1.08, 0.12], { anchor: 'left', color: 'hud', size: 12.5, font: 'ui' });
      if (gh >= 359) g.label('360° · a full turn: back to the start', [0.05, -0.75], { anchor: 'left', color: 'hud', size: 12.5, font: 'ui' });
    }
    g.vector([0, 0], P(th), { color: 'v', label: 'v', width: 3 });
    thetaArc(g, th, { r: 0.36 });
  }

  function sceneRadian(g, s) {
    const u = s.u;
    const a = u < 0.3 ? 40 : u < 0.5 ? lerp(40, 90, seg(u, 0.3, 0.45)) : u < 0.65 ? lerp(90, 180, seg(u, 0.5, 0.62)) : lerp(180, 360, seg(u, 0.72, 0.92));
    frame(g);
    g.circle([0, 0], 1, { stroke: 'hud', width: 1.4, strokeAlpha: seg(u, 0, 0.15) });
    g.vector([0, 0], [1, 0], { color: 'i', label: 'i', hat: true, width: 2.6, alpha: 0.7 });
    arc(g, 1, 0, a, { color: 'gold', width: 5 });
    const r = a * RAD, m = P(a / 2);
    const names = { 90: 'π/2 ≈ 1.57', 180: 'π ≈ 3.14', 360: '2π ≈ 6.28' };
    const key = Math.round(a);
    const txt = names[key] && Math.abs(a - key) < 0.2 ? 'arc length ' + names[key] : 'arc length ' + LA.fmt.fixed(r, 2);
    if (a > 300) g.label(txt + ' (the whole circumference)', [0, 1.3], { color: 'gold', size: 14, font: 'ui', bold: true });
    else g.label(txt, [m[0] * 1.15, m[1] * 1.15], { color: 'gold', size: 14, font: 'ui', bold: true, anchor: m[0] < -0.3 ? 'right' : m[0] > 0.3 ? 'left' : 'center' });
    g.label('radius 1', [0.5, -0.1], { color: 'muted', size: 11.5, font: 'ui', bg: false });
    g.vector([0, 0], P(a), { color: 'v', label: 'v', width: 2.4 });
  }

  function sceneWalk(g, s, ctx) {
    frame(g);
    g.circle([0, 0], 1, { stroke: 'hud', width: 1.6 });
    g.vector([0, 0], [1, 0], { color: 'i', label: 'i', hat: true, width: 2.6, alpha: 0.75 });
    thetaArc(g, s.theta);
    unitPoint(g, s.theta);
    g.label('length 1', P(s.theta).map(x => x * 0.55), { color: 'v', size: 11.5, font: 'ui', dx: -10, dy: -10 });
    pointHandle(g, 'pt', s.theta, 'theta');
  }

  function sceneDrops(g, s) {
    const th = s.theta, p = P(th);
    frame(g);
    g.circle([0, 0], 1, { stroke: 'hud', width: 1.6 });
    thetaArc(g, th, { r: 0.26, text: 'θ' });
    drops(g, th, s.drops);
    unitPoint(g, th);
    if (s.drops >= 1) bracketAt(g, [Math.min(p[0] + 0.1, 1.15), p[1] + 0.5], f2(p[0]), f2(p[1]), { color: 'v', w: 0.5 });
  }

  function hopAngle(u) {
    const n = HOPS.length, x = Math.min(n - 1e-6, u * n), k = Math.floor(x), f = seg(x - k, 0, 0.45);
    const from = k === 0 ? 0 : HOPS[k - 1];
    let to = HOPS[k]; if (from === 270 && to === 45) to = 405;   // keep walking counter-clockwise
    return { k, th: lerp(from, to, f) % 360, landed: f >= 1 };
  }
  function sceneLandmarks(g, s, ctx) {
    const H = hopAngle(s.u), th = H.th;
    frame(g);
    g.circle([0, 0], 1, { stroke: 'hud', width: 1.6 });
    for (let k = 0; k < H.k + (H.landed ? 1 : 0); k++) {
      const a = HOPS[k], q = P(a);
      g.point(q, { r: 4, color: 'gold' });
      g.label(a + '°', [q[0] * 1.17, q[1] * 1.17], { color: 'gold', size: 11.5, font: 'ui', bg: false });
    }
    drops(g, th, 1, { labels: false });
    unitPoint(g, th);
    // the table, top right
    const W = ctx.size.w, x0 = W - 178, y0 = 64, rh = 23;
    g.hud('θ', x0, y0, { color: 'gold', size: 13, bold: true });
    g.hud('cos θ', x0 + 54, y0, { color: 'i', size: 13, bold: true });
    g.hud('sin θ', x0 + 116, y0, { color: 'j', size: 13, bold: true });
    for (let k = 0; k < HOPS.length; k++) {
      if (k > H.k || (k === H.k && !H.landed)) continue;
      const a = HOPS[k], q = P(a), y = y0 + (k + 1) * rh, cur = k === H.k;
      g.hud(a + '°', x0, y, { color: cur ? 'gold' : 'hud', size: 13, font: 'mono' });
      g.hud(f2(q[0]), x0 + 54, y, { color: 'i', size: 13, font: 'mono' });
      g.hud(f2(q[1]), x0 + 116, y, { color: 'j', size: 13, font: 'mono' });
    }
  }

  function chartBox(ctx) { const W = ctx.size.w, H = ctx.size.h; return { x: 54, y: Math.round(H * 0.64), w: W - 110, h: Math.round(H * 0.27) }; }
  function sceneBelief(g, s, ctx) {
    const u = s.u, th = s.theta, swept = Math.min(180, s.sweepMax);
    frame(g);
    g.circle([0, 0], 1, { stroke: 'hud', width: 1.6 });
    thetaArc(g, th, { r: 0.26, text: 'θ = ' + deg(th) });
    drops(g, th, 1, { sinLabel: false });
    unitPoint(g, th);
    // live graph strip under the circle
    const B = chartBox(ctx), pts = [], spts = [], bel = [];
    for (let a = 0; a <= swept + 1e-9; a += 3) { pts.push([a, Math.cos(a * RAD)]); spts.push([a, Math.sin(a * RAD)]); bel.push([a, a / 180]); }
    const series = [{ points: bel, color: 'muted', dash: [6, 4], width: 2 }, { points: spts, color: 'j', dash: [2, 4], width: 1.4 }, { points: pts, color: 'i', width: 2.6 }];
    if (pts.length < 2) series.forEach(sr => sr.points = [[0, sr.points.length ? sr.points[0][1] : 0], [0.1, sr.points.length ? sr.points[0][1] : 0]]);
    g.lineChart(series, { x: B.x, y: B.y, w: B.w, h: B.h, xRange: [0, 180], yRange: [-1.1, 1.1], hline: 0, hlineColor: 'muted', xLabel: 'θ from 0° to 180°' });
    const X = a => B.x + a / 180 * B.w, Y = v => B.y + B.h - (v + 1.1) / 2.2 * B.h;
    g.hud('cos θ', B.x + 6, B.y + 12, { color: 'i', size: 12, bold: true });
    g.hud('sin θ (faint)', B.x + 54, B.y + 12, { color: 'j', size: 12 });
    g.hud('belief: “cos grows”', B.x + 150, B.y + 12, { color: 'muted', size: 12 });
    if (u >= 0.85) {
      g.marker.cross([X(120), Y(120 / 180)], { screen: true, size: 20 });
      g.hud('belief says 0.67', X(120) + 14, Y(120 / 180) - 12, { color: 'bad', size: 12 });
      g.point(g.P.toWorld(X(120), Y(-0.5)), { r: 5, color: 'gold' });
      g.hud('truth: cos 120° = −0.5', X(120) + 10, Y(-0.5) + 14, { color: 'gold', size: 12, bold: true });
    }
    // the current angle on the curve
    g.point(g.P.toWorld(X(th), Y(Math.cos(th * RAD))), { r: 4.5, color: 'v' });
  }

  function sceneRot(g, s) {
    const th = s.theta, rI = s.rotI, rJ = s.rotJ, p = P(th);
    frame(g);
    g.circle([0, 0], 1, { stroke: 'hud', width: 1.6 });
    // ghosts of the originals
    g.vector([0, 0], [1, 0], { color: 'i', width: 2, alpha: 0.3, dashed: true });
    if (s.showJ) g.vector([0, 0], [0, 1], { color: 'j', width: 2, alpha: 0.3, dashed: true });
    const aI = th * rI;
    arc(g, 0.3, 0, aI, { color: 'gold', width: 2 });
    if (s.scene === 'rotI') { unitPoint(g, th, { alpha: 0.55 }); if (rI >= 1) g.marker.ring(p, { r: 13 + 3 * Math.sin(s.u * 40), color: 'v' }); }
    if (s.showJ) {
      const aJ = 90 + th * rJ, q = P(aJ);
      arc(g, 0.3, 90, aJ, { color: 'gold', width: 2 });
      if (rJ >= 1) {
        g.line(q, [q[0], 0], { color: 'j', dashed: [4, 4], width: 1.2 });
        g.line([0, 0], [q[0], 0], { color: 'i', width: 4 });
        g.line([q[0], 0], q, { color: 'j', width: 4 });
        g.dim([0, 0], [q[0], 0], { offset: -20, color: 'i', label: '−sin θ = ' + f2(q[0]), size: 12.5 });
        g.dim([q[0], 0], q, { offset: -14, color: 'j' });
        g.label('cos θ = ' + f2(q[1]), [q[0], q[1] / 2], { anchor: 'right', dx: -24, color: 'j', size: 12.5, font: 'ui' });
        g.arc([0, 0], P(aI), q, { r: 0.2, right: true, color: 'hud' });
      }
      g.vector([0, 0], q, { color: 'j', label: 'j', hat: true, width: 3 });
      if (rJ >= 1) bracketAt(g, [Math.max(q[0] - 0.1, -1.15), q[1] + 0.5], '−' + f2(p[1]), f2(p[0]), { color: 'j', w: 0.55 });
    }
    g.vector([0, 0], P(aI), { color: 'i', label: 'i', hat: true, width: 3 });
    if (rI >= 1) bracketAt(g, [Math.min(p[0] + 0.1, 1.15), p[1] + 0.5], f2(p[0]), f2(p[1]), { color: 'i', w: 0.5 });
    if (s.scene === 'rotI' && rI >= 1) g.label('î → [cos θ; sin θ]', [0.15, -0.55], { anchor: 'left', color: 'i', size: 15, font: 'ui', bold: true });
  }

  function cosBar(g, c, y) {
    y = y === undefined ? -1.42 : y;
    g.line([-1, y], [1, y], { color: 'gridAxis', width: 1.2 });
    [-1, 0, 1].forEach(x => { g.line([x, y - 0.05], [x, y + 0.05], { color: 'gridAxis', width: 1.2 }); g.label(x === -1 ? '−1' : String(x), [x, y], { anchor: 'below', dy: 10, color: 'tick', size: 11, font: 'ui', bg: false }); });
    g.polygon([[0, y - 0.07], [c, y - 0.07], [c, y + 0.07], [0, y + 0.07]], { fill: 'gold', alpha: 0.85, stroke: 'gold' });
    g.label('cos = ' + f2(c), [c, y], { anchor: c >= 0 ? 'left' : 'right', dx: c >= 0 ? 8 : -8, dy: -16, color: 'gold', size: 13, font: 'ui', bold: true });
  }
  function scenePair(g, s, label) {
    const a = s.pair.a, b = s.pair.b, d = Math.abs(b - a), c = Math.cos(d * RAD);
    frame(g, { R: 1.3 });
    g.circle([0, 0], 1, { stroke: 'hud', width: 1.2, strokeAlpha: 0.6 });
    arc(g, 0.38, Math.min(a, b), Math.max(a, b), { color: 'gold', width: 2 });
    const m = P((a + b) / 2);
    g.label('cos ' + deg(d) + ' = ' + f2(c), [m[0] * 0.62, m[1] * 0.62], { color: 'gold', size: 13, font: 'ui', bold: true });
    g.vector([0, 0], P(a), { color: 'v', label: label ? label[0] : 'a', width: 3, labelFont: 'ui' });
    g.vector([0, 0], P(b), { color: 'w', label: label ? label[1] : 'b', width: 3, labelFont: 'ui' });
    cosBar(g, c);
  }
  function sceneAlign(g, s) {
    scenePair(g, s, ['cat', s.pair.b > 150 ? 'invoice' : 'kitten']);
    g.label('toy 2D arrows; real embeddings have 768 entries', [0, 1.52], { color: 'muted', size: 11.5, font: 'ui' });
  }

  function sceneRecap(g, s) {
    const th = 40, p = P(th);
    const cards = [
      { title: 'A unit arrow at θ is [cos θ; sin θ]', draw(gg) {
        gg.circle([0, 0], 1, { stroke: 'hud', width: 1.4 }); gg.line([-1.2, 0], [1.2, 0], { color: 'gridAxis' }); gg.line([0, -1.2], [0, 1.2], { color: 'gridAxis' });
        drops(gg, th, 1, { labels: false }); unitPoint(gg, th);
        bracketAt(gg, [2.4, 0], 'cos θ', 'sin θ', { color: 'v', w: 1.0, size: 13, h: 0.42 });
      } },
      { title: 'Rotated î and ĵ: [cos θ; sin θ], [−sin θ; cos θ]', draw(gg) {
        gg.circle([0, 0], 1, { stroke: 'hud', width: 1.2, strokeAlpha: 0.5 });
        gg.vector([0, 0], [1, 0], { color: 'i', alpha: 0.3, dashed: true, width: 2 }); gg.vector([0, 0], [0, 1], { color: 'j', alpha: 0.3, dashed: true, width: 2 });
        gg.vector([0, 0], p, { color: 'i', label: 'i', hat: true, width: 2.6 }); gg.vector([0, 0], P(90 + th), { color: 'j', label: 'j', hat: true, width: 2.6 });
        bracketAt(gg, [2.1, 0], 'cos θ', 'sin θ', { color: 'i', w: 0.95, size: 12.5, h: 0.42 }); bracketAt(gg, [3.3, 0], '−sin θ', 'cos θ', { color: 'j', w: 1.05, size: 12.5, h: 0.42 });
      } },
      { title: 'cos = alignment: 1, 0, −1', draw(gg) {
        gg.vector([0, 0], P(20), { color: 'v', label: 'a', width: 2.6 }); gg.vector([0, 0], P(65), { color: 'w', label: 'b', width: 2.6 });
        arc(gg, 0.4, 20, 65, { color: 'gold' });
        const c = Math.cos(45 * RAD);
        gg.line([1.6, -0.4], [3.6, -0.4], { color: 'gridAxis' }); gg.polygon([[2.6, -0.48], [2.6 + c, -0.48], [2.6 + c, -0.32], [2.6, -0.32]], { fill: 'gold', alpha: 0.85, stroke: 'gold' });
        [['−1', 1.6], ['0', 2.6], ['1', 3.6]].forEach(q => gg.label(q[0], [q[1], -0.4], { anchor: 'below', dy: 10, color: 'tick', size: 11, font: 'ui', bg: false }));
        gg.label('cos 45° = 0.71', [2.6, 0.2], { color: 'gold', size: 13, font: 'ui', bold: true });
      } },
    ];
    cards.forEach((c, k) => {
      g.viewport({ x: 0.04, y: 0.03 + k * 0.325, w: 0.92, h: 0.295 }, { view: '2d', cam: { center: [1.35, 0], halfH: 1.3 }, bg: 'panel' }, (gg, r) => {
        c.draw(gg);
        gg.hud(c.title, r.x + 12, r.y + 18, { size: 13, color: 'text', bold: true });
      });
    });
  }

  /* ---------------- playground & checks ---------------- */
  function scenePlay(g, s, ctx) {
    if (s.pairOn) {
      s.pair = { a: s.theta, b: s.pairB };
      scenePair(g, s);
      pointHandle(g, 'pt', s.theta, 'theta');
      pointHandle(g, 'pb', s.pairB, 'pairB');
      return;
    }
    const th = s.theta, p = P(th);
    frame(g);
    g.circle([0, 0], 1, { stroke: 'hud', width: 1.6 });
    thetaArc(g, th, { r: 0.26 });
    if (s.showDrops) drops(g, th, 1);
    if (s.rotBoth) {
      g.vector([0, 0], [1, 0], { color: 'i', width: 2, alpha: 0.3, dashed: true });
      g.vector([0, 0], [0, 1], { color: 'j', width: 2, alpha: 0.3, dashed: true });
      g.vector([0, 0], P(90 + th), { color: 'j', label: 'j', hat: true, width: 3 });
      g.vector([0, 0], p, { color: 'i', label: 'i', hat: true, width: 3 });
    }
    unitPoint(g, th, { alpha: s.rotBoth ? 0.6 : 1 });
    if (!s.showDrops) bracketAt(g, [p[0] * 1.45, p[1] * 1.45 + 0.05], f2(p[0]), f2(p[1]), { color: 'v', w: 0.5 });
    if (s.showGraph) {
      const B = chartBox(ctx), pts = [], spts = [];
      for (let a = 0; a <= 360; a += 5) { pts.push([a, Math.cos(a * RAD)]); spts.push([a, Math.sin(a * RAD)]); }
      g.lineChart([{ points: pts, color: 'i', width: 2.2 }, { points: spts, color: 'j', width: 2.2 }], { x: B.x, y: B.y, w: B.w, h: B.h, xRange: [0, 360], yRange: [-1.1, 1.1], hline: 0, hlineColor: 'muted', xLabel: 'θ from 0° to 360°' });
      const X = a => B.x + a / 360 * B.w, Y = v => B.y + B.h - (v + 1.1) / 2.2 * B.h;
      g.point(g.P.toWorld(X(th), Y(p[0])), { r: 5, color: 'i' });
      g.point(g.P.toWorld(X(th), Y(p[1])), { r: 5, color: 'j' });
      g.hud('cos θ', B.x + 6, B.y + 12, { color: 'i', size: 12, bold: true }); g.hud('sin θ', B.x + 54, B.y + 12, { color: 'j', size: 12, bold: true });
    }
    pointHandle(g, 'pt', th, 'theta');
  }

  function sceneCheck(g, s, ctx) {
    const th = s.theta, p = P(th);
    frame(g);
    g.circle([0, 0], 1, { stroke: 'hud', width: 1.6 });
    if (s.scene === 'check2') {
      g.vector([0, 0], [1, 0], { color: 'i', width: 2, alpha: 0.3, dashed: true });
      arc(g, 0.3, 0, th, { color: 'gold', width: 2 });
      g.vector([0, 0], p, { color: 'i', label: 'i', hat: true, width: 3 });
      g.handle('tip', p, (q, st) => { let d = Math.atan2(q[1], q[0]) / RAD; if (d < 0) d += 360; st.theta = Math.round(d * 10) / 10; }, { color: 'i', r: 16 });
      return;
    }
    thetaArc(g, th, { r: 0.26 });
    if (s.scene === 'check1') { drops(g, th, 1); unitPoint(g, th); pointHandle(g, 'pt', th, 'theta'); return; }
    // check 3: the red height without its value; the typed value as an orange tick on the y-axis
    drops(g, th, 1, { sinLabel: false });
    unitPoint(g, th);
    g.label('sin 30° = ?', [p[0] + 0.12, p[1] / 2], { anchor: 'left', color: 'j', size: 14, font: 'ui' });
    const y = LA.clamp(s.guess[0], -1.4, 1.4);
    g.line([-0.12, y], [0.12, y], { color: 'b', width: 3 });
    g.label('typed ' + f2(s.guess[0]), [-0.15, y], { anchor: 'right', color: 'b', size: 13, font: 'ui' });
    g.line([0, p[1]], [p[0], p[1]], { color: 'muted', dashed: [3, 4], width: 1 });
  }

  const SCENES = { turn: sceneTurn, radian: sceneRadian, walk: sceneWalk, drops: sceneDrops, landmarks: sceneLandmarks, belief: sceneBelief,
    rotI: sceneRot, rotJ: sceneRot, align: sceneAlign, recap: sceneRecap, play: scenePlay, check1: sceneCheck, check2: sceneCheck, check3: sceneCheck };

  LA.registerLesson({
    id: 'p0-03-angles-and-circles',
    part: 0,
    title: 'Angles, the unit circle, cos and sin',
    subtitle: 'cos θ and sin θ are just the coordinates of a point on a circle of radius 1.',
    minutes: 8,
    prereqs: ['p0-02-reading-the-symbols'],
    tags: ['primer'],
    badge: { drawnIn: '2D', realSize: 'cosine similarity compares 768-dimensional embeddings; a 2D rotation block appears inside every rotary position encoding', transfers: 'the angle between two arrows is defined the same way in any dimension (via the dot product, Part 1); the unit-circle picture is all the trigonometry this course needs' },
    cantShow: ['an angle between two vectors in ℝ⁷⁶⁸ (it exists; it is computed, not seen)'],
    terms: [
      { term: 'angle', def: 'The amount of turn between two arrows from the same point, measured in degrees.' },
      { term: 'degrees', def: 'Units of turn: one degree is 1/360 of a full turn.' },
      { term: 'radians', def: 'Units of turn measured by arc length: one radian is the angle whose arc on the unit circle has length 1, about 57°.' },
      { term: 'unit circle', def: 'The circle of radius 1 centered at the origin.' },
      { term: 'cos θ', def: 'The cosine: the x-coordinate of the unit-circle point at angle θ.' },
      { term: 'sin θ', def: 'The sine: the y-coordinate of the unit-circle point at angle θ.' },
      { term: 'rotation', def: 'Turning every arrow by the same angle about the origin.' },
    ],
    atlas: ['embeddings', 'attention'],
    thumbnail: 3,

    state() {
      return {
        view: '2d', cam: camCircle(), scene: 'turn', u: 0, panel: 'theta',
        theta: 40, drops: 0, rotI: 0, rotJ: 0, showJ: false, sweepMax: 0,
        pair: { a: 20, b: 45 }, pairOn: false, pairB: 130, showDrops: true, showGraph: false, rotBoth: false, landmark: -1, guess: [0],
      };
    },

    render(g, s, ctx) { (SCENES[s.scene] || sceneTurn)(g, s, ctx); },

    math(s, tex) {
      const th = s.theta, p = P(th);
      const vec2 = (a, b) => tex.vec([a, b], { colors: 'rows' });
      switch (s.panel) {
        case 'theta': return tex.block(tex.eq(tex.sym('θ'), tex.op('='), tex.num(deg(s.scene === 'turn' ? 40 * seg(s.u, 0, 0.5) : th), { color: 'gold' })));
        case 'radian': return tex.block(tex.eq(tex.sym('θ'), tex.op('='), T(tex, '40°'), tex.op('='), tex.num('0.70 rad', { color: 'gold' })), tex.eq(T(tex, '360°'), tex.op('='), T(tex, '2π ≈ 6.28 rad')));
        case 'walk': return tex.block(tex.eq(tex.sym('θ'), tex.op('='), tex.num(deg(th), { color: 'gold' }), tex.op('='), T(tex, LA.fmt.fixed(th * RAD, 2) + ' rad')));
        case 'drops': return tex.block(tex.eq(vec2('cos θ', 'sin θ'), tex.op('='), vec2(f2(p[0]), f2(p[1]))));
        case 'landmarks': {
          const H = hopAngle(s.u);
          const rows = LANDMARKS.map(a => [a + '°', f2(P(a)[0]), f2(P(a)[1])]);
          const cur = LANDMARKS.indexOf(HOPS[H.k]);
          return tex.block(tex.eq(T(tex, 'θ → (cos θ, sin θ)')), tex.eq(tex.mat(rows, { colors: (i, j) => ['gold', 'i', 'j'][j], highlight: (i, j) => i === cur })));
        }
        case 'belief': return tex.block(tex.eq(T(tex, 'cos ' + deg(th)), tex.op('='), tex.num(f2(p[0]), { color: 'i' })), tex.eq(tex.op('·')), tex.eq(T(tex, 'sin ' + deg(th)), tex.op('='), tex.num(f2(p[1]), { color: 'j' })));
        case 'rotI': return tex.block(tex.eq(tex.ihat(), tex.op('↦'), vec2('cos θ', 'sin θ'), tex.op('='), vec2(f2(p[0]), f2(p[1]))));
        case 'rotJ': return tex.block(tex.eq(tex.jhat(), tex.op('↦'), vec2('−sin θ', 'cos θ'), tex.op('='), vec2(f2(-p[1]), f2(p[0]))));
        case 'align': { const d = Math.abs(s.pair.b - s.pair.a); return tex.block(tex.eq(T(tex, 'cos(angle between) = cos ' + deg(d)), tex.op('='), tex.num(f2(Math.cos(d * RAD)), { color: 'gold' }))); }
        case 'play': {
          if (s.pairOn) { const d = Math.abs(s.pairB - th); return tex.block(tex.eq(T(tex, 'angle between = ' + deg(d)), tex.op('·'), tex.num('cos = ' + f2(Math.cos(d * RAD)), { color: 'gold' }))); }
          const parts = [tex.eq(tex.sym('θ'), tex.op('='), tex.num(deg(th), { color: 'gold' }), tex.op('='), T(tex, LA.fmt.fixed(th * RAD, 2) + ' rad')), tex.eq(vec2('cos θ', 'sin θ'), tex.op('='), vec2(f2(p[0]), f2(p[1])))];
          if (s.rotBoth) parts.push(tex.eq(T(tex, 'columns:'), tex.ihat(), tex.op('↦'), vec2(f2(p[0]), f2(p[1])), tex.jhat(), tex.op('↦'), vec2(f2(-p[1]), f2(p[0]))));
          return tex.block.apply(null, parts);
        }
        case 'check1': return tex.block(tex.eq(T(tex, 'cos θ'), tex.op('='), tex.num(f2(p[0]), { color: 'i' })), tex.eq(T(tex, 'sin θ'), tex.op('='), tex.num(f2(p[1]), { color: 'j' })));
        case 'check2': return tex.block(tex.eq(tex.sym('θ'), tex.op('='), tex.num(deg(th), { color: 'gold' })));
        case 'check3': return tex.block(tex.eq(T(tex, 'sin 30°'), tex.op('='), tex.num(f2(s.guess[0]) + ' ?', { color: 'b' })));
        default: return '';
      }
    },

    steps: [
      { title: 'An angle is an amount of turn', duration: 4,
        narration: 'Two arrows from the origin: the green ' + ihat + ' (“i-hat”), one unit along the x-axis, and the yellow <span class="c-v">v</span>. The <b>angle</b> between them is how far you turn one to reach the other, measured in <b>degrees</b>: a quarter turn is 90°, a square corner; a half turn is 180°; a full turn brings you back, 360°. We draw the angle as a small arc and call it θ.',
        setup(s) { s.view = '2d'; s.cam = camCircle(); s.scene = 'turn'; s.panel = 'theta'; s.theta = 40; },
        at(s, t) { s.u = t; } },

      { title: 'Radians: the arc’s own length', duration: 4,
        narration: 'Code measures angles in <b>radians</b>: the length of the arc on a circle of radius 1. A full turn is the circumference, 2π ≈ 6.28; a half turn is π ≈ 3.14; a quarter turn is π/2 ≈ 1.57. We will say degrees and show radians beside them, because every formula uses radians.',
        setup(s) { s.cam = camCircle(); s.scene = 'radian'; s.panel = 'radian'; },
        at(s, t) { s.u = t; } },

      { title: 'A point walks the unit circle', duration: 4, interactive: true,
        narration: 'The <b>unit circle</b> is every arrow of length exactly 1. Put a yellow point on it at angle θ from the green ' + ihat + '. As θ grows the point walks counter-clockwise around the circle. Drag it; the angle readout follows. Everything in trigonometry is about where this point is.',
        setup(s) { s.cam = camCircle(); s.scene = 'walk'; s.panel = 'walk'; s.theta = 0; },
        at(s, t) { s.u = t; s.theta = Math.round(40 * seg(t, 0, 0.6) * 10) / 10; } },

      { title: 'cos θ is the x-coordinate, sin θ is the y-coordinate', duration: 4,
        narration: 'Drop the point onto the axes. Its green shadow on the x-axis is <b>cos θ</b>; its red height is <b>sin θ</b>. That is the whole definition: cos is “how far along ' + ihat + '”, sin is “how far along ' + jhat + '” (“j-hat”, one unit up), for a unit arrow at angle θ. At 40°, cos ≈ 0.77 and sin ≈ 0.64.',
        setup(s) { s.cam = camCircle(); s.scene = 'drops'; s.panel = 'drops'; s.theta = 40; s.drops = 0; },
        at(s, t) { s.u = t; s.drops = t; } },

      { title: 'Landmarks you will see again', duration: 6,
        narration: 'Watch the point hop to the angles that appear everywhere. 0°: [1; 0]. 90°: [0; 1]. 180°: [−1; 0]. 270°: [0; −1]. 45°: both coordinates equal, about 0.71. 60°: [0.5; 0.87]; 30°: [0.87; 0.5]. The shadows swap roles as the point climbs.',
        setup(s) { s.cam = camTable(); s.scene = 'landmarks'; s.panel = 'landmarks'; s.drops = 1; },
        at(s, t) { s.u = t; s.theta = hopAngle(t).th; } },

      { title: 'cos does not grow with the angle', duration: 5, misconception: true,
        narration: 'A tempting belief: “a bigger angle gives a bigger cosine.” Watch the green shadow as the point passes 90°: it shrinks to zero, then turns negative. At 120°, cos θ = −0.5. Cosine measures how much the arrow still points along ' + ihat + ', and past a quarter turn it points the other way.',
        setup(s) { s.cam = camChart(); s.scene = 'belief'; s.panel = 'belief'; s.theta = 0; s.sweepMax = 0; },
        at(s, t) { s.u = t; s.sweepMax = 180 * seg(t, 0, 0.7, 'linear'); s.theta = t < 0.7 ? s.sweepMax : lerp(180, 120, seg(t, 0.72, 0.82)); } },

      { title: 'Rotating î lands on that point', duration: 4,
        narration: 'Now rotate the green ' + ihat + ' itself by θ. Its tip lands exactly on the unit-circle point: the rotated ' + ihat + ' is [cos θ; sin θ]. Nothing to memorize — the two numbers are simply where the green arrow went. Part 2 writes this as the first column of a <b>rotation</b> matrix.',
        setup(s) { s.cam = camCircle(); s.scene = 'rotI'; s.panel = 'rotI'; s.theta = 40; s.rotI = 0; s.rotJ = 0; s.showJ = false; s.sweepMax = 0; },
        at(s, t) { s.u = t; s.rotI = seg(t, 0.1, 0.8); } },

      { title: 'ĵ is a quarter turn ahead', duration: 4,
        narration: 'The red ' + jhat + ' starts a quarter turn ahead of ' + ihat + ', at 90°. Rotate it by the same θ and it lands at 90° + θ: coordinates [−sin θ; cos θ]. Its shadow along the x-axis is the negative of ' + ihat + '’s height. Both rotated arrows stay perpendicular and stay length 1.',
        setup(s) { s.cam = camCircle(); s.scene = 'rotJ'; s.panel = 'rotJ'; s.theta = 40; s.showJ = true; s.rotI = 0; s.rotJ = 0; },
        at(s, t) { s.u = t; s.rotI = s.rotJ = seg(t, 0.1, 0.8); } },

      { title: 'cos θ measures alignment', duration: 6, mlTie: true,
        narration: 'Take two unit arrows. The cosine of the angle between them says how aligned they are: 1 when they point the same way, 0 when perpendicular, −1 when opposite. Machine learning uses exactly this number to compare meanings: the arrows for “cat” and “kitten” have cosine near 0.9. Part 1 shows how to compute it without ever drawing the angle.',
        setup(s) { s.cam = { center: [0, -0.1], halfH: 1.85 }; s.scene = 'align'; s.panel = 'align'; s.showJ = false; s.pair = { a: 20, b: 45 }; },
        at(s, t) { s.u = t; s.pair = { a: 20, b: t < 0.55 ? lerp(45, 110, seg(t, 0.25, 0.5)) : lerp(110, 200, seg(t, 0.6, 0.85)) }; } },

      { title: 'Recap', duration: 3, recap: true,
        narration: 'The unit circle is all of trigonometry you need. A unit arrow at angle θ has coordinates [cos θ; sin θ]. Rotating ' + ihat + ' by θ lands there; rotating ' + jhat + ' lands at [−sin θ; cos θ]. Cosine runs from 1 (same way) through 0 (perpendicular) to −1 (opposite) and is how ML compares directions.',
        cards: ['<b>A unit arrow at θ is [cos θ; sin θ]:</b> cos is the green shadow, sin the red height.', '<b>Rotated î and ĵ</b> land at [cos θ; sin θ] and [−sin θ; cos θ] — still perpendicular, still length 1.', '<b>cos = alignment:</b> 1 same way, 0 perpendicular, −1 opposite.'],
        setup(s) { s.scene = 'recap'; s.panel = 'none'; s.cam = camCircle(); },
        at(s, t) { s.u = t; } },
    ],

    playground: {
      intro: '<p>Drag the yellow point around the circle or slide θ. Turn on the graph to see cos and sin as curves, rotate ' + ihat + ' and ' + jhat + ', or compare two arrows.</p>',
      setup(s) { s.scene = 'play'; s.panel = 'play'; s.cam = camChart(); s.theta = 40; s.showDrops = true; s.showGraph = true; s.rotBoth = false; s.pairOn = false; s.pairB = 130; },
      controls: [
        { type: 'slider', bind: 'theta', min: 0, max: 360, step: 1, label: 'θ (degrees)', color: 'gold', fmt: x => deg(x) },
        { type: 'toggle', bind: 'showDrops', label: 'Show cos/sin shadows' },
        { type: 'toggle', bind: 'showGraph', label: 'Graph cos and sin' },
        { type: 'toggle', bind: 'rotBoth', label: 'Rotate î and ĵ' },
        { type: 'preset', label: 'Landmark', items: LANDMARKS.map(a => ({ label: a + '°', apply(s) { s.theta = a; s.pairOn = false; } })).concat([
          { label: 'Perpendicular pair (cos = 0)', apply(s) { s.pairOn = true; s.theta = 30; s.pairB = 120; } },
          { label: 'Single point', apply(s) { s.pairOn = false; } },
        ]) },
      ],
    },

    checks: [
      { kind: 'manipulate',
        prompt: 'Drag the point on the circle so that cos θ = −0.5 with sin θ positive.',
        setup(s) { s.scene = 'check1'; s.panel = 'check1'; s.cam = camCircle(); s.theta = 40; },
        judge(s) { const p = P(s.theta), r = LA.judge.scalar(p[0], -0.5, LA.TOL.scalar); return { ok: r.ok && p[1] > 0, progress: p[1] > 0 ? r.progress : 0.2, hint: p[1] <= 0 ? 'Keep sin θ positive: stay in the upper half of the circle.' : undefined }; },
        hint: 'Past 90° the green shadow goes negative; stop when it reads −0.5.',
        solution(s) { s.theta = 120; } },
      { kind: 'manipulate',
        prompt: 'Rotate ' + ihat + ' (drag its tip around the circle) until it points straight up.',
        setup(s) { s.scene = 'check2'; s.panel = 'check2'; s.cam = camCircle(); s.theta = 0; },
        judge(s) { const d = Math.abs(s.theta - 90); return { ok: d <= LA.TOL.angleDeg, progress: Math.max(0, 1 - d / 90) }; },
        hint: 'Straight up means cos θ = 0 and sin θ = 1.',
        solution(s) { s.theta = 90; } },
      { kind: 'entry',
        prompt: 'Type sin θ for θ = 30°. Your number appears as an orange tick on the y-axis.',
        bind: 'guess', answer: [0.5], tol: LA.TOL.entry, label: 'sin 30°', colors: ['b'], step: 0.05,
        hint: 'It is one of the landmark pairs; 30° is the mirror of 60°.',
        setup(s) { s.scene = 'check3'; s.panel = 'check3'; s.cam = camCircle(); s.theta = 30; s.guess = [0]; } },
    ],
  });
})();
