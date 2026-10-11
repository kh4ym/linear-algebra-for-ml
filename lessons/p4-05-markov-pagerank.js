/* p4-05 — Markov chains converge to an eigenvector; PageRank (capstone). */
(function () {
  'use strict';
  const N = LA.num, seg = LA.seg, lerpV = LA.lerpV;
  const WEATHER = [[0.8, 0.1, 0.2], [0.1, 0.7, 0.3], [0.1, 0.2, 0.5]];
  const STICKY = [[0.9, 0.05, 0.1], [0.05, 0.9, 0.1], [0.05, 0.05, 0.8]];
  const SLOW = [[0.97, 0.02, 0.01], [0.02, 0.96, 0.03], [0.01, 0.02, 0.96]];
  const CYCLE = [[0, 0, 1], [0, 1, 0], [1, 0, 0]];
  const MULTI = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0.2, 0.3, 0.5]];
  const MULTI_COL = ['i', 'j', 'k', 'b'];
  const STATES = ['sunny', 'cloudy', 'rainy'];
  const SCOL = ['i', 'j', 'k'];
  const WEB = [[0, 1], [0, 2], [1, 2], [2, 0], [3, 2], [3, 4], [4, 3], [4, 2]];      // 1→2, 1→3, 2→3, 3→1, 4→3, 4→5, 5→4, 5→3 (0-based)
  const WEB_DEAD = [[0, 1], [0, 2], [1, 2], [2, 0], [3, 2], [3, 4], [1, 4], [0, 4]]; // page 5 has no out-links
  const HONEST = 'The picture stops at 3. The formula does not.';
  const goldS = x => '<span class="c-gold">' + x + '</span>';

  /* ---------- numerics ---------- */
  function colNorm(P) {
    const n = P.length, out = N.zerosM(n, n);
    for (let j = 0; j < n; j++) {
      let s = 0; for (let i = 0; i < n; i++) s += Math.max(0, P[i][j]);
      for (let i = 0; i < n; i++) out[i][j] = s > 1e-9 ? Math.max(0, P[i][j]) / s : 1 / n;
    }
    return out;
  }
  function trail(P, p0, k) { const pts = [p0.slice()]; let p = p0.slice(); for (let i = 0; i < k; i++) { p = N.matVec(P, p); pts.push(p); } return pts; }
  /** point at continuous time k along a trail (linear between days) */
  function trailAt(pts, k) { const a = Math.min(Math.floor(k), pts.length - 1), b = Math.min(a + 1, pts.length - 1); return lerpV(pts[a], pts[b], k - a); }
  /** eigenvalues of a 3×3 column-stochastic matrix: 1 and the roots of λ² − (tr−1)λ + det/1 */
  function eig3(P) {
    const tr = P[0][0] + P[1][1] + P[2][2] - 1, det = N.det(P);
    const disc = tr * tr - 4 * det;
    if (disc >= -1e-12) { const q = Math.sqrt(Math.max(0, disc)); return { real: true, values: [1, (tr + q) / 2, (tr - q) / 2] }; }
    return { real: false, values: [1, Math.sqrt(Math.max(0, det)), Math.sqrt(Math.max(0, det))], re: tr / 2, im: Math.sqrt(-disc) / 2 };
  }
  function projSimplex(p) { const q = p.map(x => Math.max(0, x)); const s = q[0] + q[1] + q[2]; return s > 1e-9 ? q.map(x => x / s) : [1 / 3, 1 / 3, 1 / 3]; }
  function webP(edges) {
    const n = 5, out = [[], [], [], [], []];
    edges.forEach(e => { if (e[0] !== e[1] && out[e[0]].indexOf(e[1]) < 0) out[e[0]].push(e[1]); });
    const P = N.zerosM(n, n);
    for (let j = 0; j < n; j++) { if (!out[j].length) { for (let i = 0; i < n; i++) P[i][j] = 1 / n; } else out[j].forEach(i => { P[i][j] = 1 / out[j].length; }); }
    return { P, out };
  }
  function webM(P, d) { return P.map(r => r.map(x => d * x + (1 - d) / 5)); }
  function rankHistory(edges, d, iters) {
    const M = webM(webP(edges).P, d); let r = [0.2, 0.2, 0.2, 0.2, 0.2]; const h = [r];
    for (let i = 0; i < iters; i++) { r = N.matVec(M, r); h.push(r); }
    return h;
  }
  /** seeded random surfer: 400 hops with damping d */
  function surferWalk(edges, d, hops) {
    const out = webP(edges).out, rand = LA.rng(55); const walk = [0]; let pos = 0;
    for (let i = 0; i < hops; i++) {
      if (rand() > d || !out[pos].length) pos = LA.rng.int(rand, 0, 4); else pos = LA.rng.pick(rand, out[pos]);
      walk.push(pos);
    }
    return walk;
  }
  function rankAt(edges, d, t) { const h = rankHistory(edges, d, 12); const a = Math.min(Math.floor(t), 12), b = Math.min(a + 1, 12); return lerpV(h[a], h[b], t - a); }

  /* ---------- drawing: weather graph (2D) ---------- */
  const GPOS = [[0, 1.97], [-1.0, 0.34], [1.0, 0.34]];   // spread out so the inner edge labels do not collide
  const GC = [0, 0.88];
  function bez(A, B, off, n) {
    const d = [B[0] - A[0], B[1] - A[1]], L = Math.hypot(d[0], d[1]) || 1, nrm = [d[1] / L, -d[0] / L];
    const C = [(A[0] + B[0]) / 2 + nrm[0] * off, (A[1] + B[1]) / 2 + nrm[1] * off];
    const pts = []; for (let k = 0; k <= n; k++) { const t = k / n, u = 1 - t; pts.push([u * u * A[0] + 2 * u * t * C[0] + t * t * B[0], u * u * A[1] + 2 * u * t * C[1] + t * t * B[1]]); }
    return pts;
  }
  function curvedArrow(g, A, B, off, r0, r1, opts) {
    const pts = bez(A, B, off, 24).filter(p => Math.hypot(p[0] - A[0], p[1] - A[1]) > r0 && Math.hypot(p[0] - B[0], p[1] - B[1]) > r1);
    if (pts.length < 3) return null;
    g.polyline(pts.slice(0, -1), { color: opts.color, width: opts.width, alpha: opts.alpha });
    g.vector(pts[pts.length - 3], pts[pts.length - 1], { color: opts.headColor || opts.color, width: opts.headColor ? opts.width + 1 : opts.width, head: opts.head || 9, alpha: opts.headColor ? 1 : opts.alpha });
    return pts[Math.floor(pts.length / 2)];
  }
  function drawGraph(g, P, opts) {
    opts = opts || {};
    const R = 0.24, focus = opts.focus;
    for (let j = 0; j < 3; j++) for (let i = 0; i < 3; i++) {
      const p = P[i][j]; if (p < 0.005) continue;
      const al = focus === undefined || focus === null ? 0.85 : (j === focus ? 1 : 0.25);
      const w = 1 + 3 * p;
      if (i === j) {
        const o = N.normalize(N.sub(GPOS[j], GC)), c = N.add(GPOS[j], N.scale(o, R + 0.2));
        const pts = []; for (let k = 0; k <= 40; k++) { const a = Math.atan2(o[1], o[0]) + Math.PI + 0.75 + (2 * Math.PI - 1.5) * k / 40; pts.push([c[0] + 0.2 * Math.cos(a), c[1] + 0.2 * Math.sin(a)]); }
        g.polyline(pts, { color: 'hud', width: w, alpha: al });
        g.vector(pts[37], pts[40], { color: 'hud', width: w, head: 8, alpha: al });
        g.label(LA.fmt(p), N.add(c, N.scale(o, 0.36)), { color: 'hud', size: 12, alpha: al });
      } else {
        const mid = curvedArrow(g, GPOS[j], GPOS[i], 0.22, R + 0.02, R + 0.04, { color: 'hud', width: w, alpha: al });
        if (mid) { const d = N.normalize(N.sub(GPOS[i], GPOS[j])); g.label(LA.fmt(p), [mid[0] + d[1] * 0.17, mid[1] - d[0] * 0.17], { color: 'hud', size: 12, alpha: al }); }
      }
    }
    for (let k = 0; k < 3; k++) {
      g.circle(GPOS[k], R, { fill: SCOL[k], alpha: 0.35, stroke: SCOL[k], width: 2 });
      g.label(STATES[k], GPOS[k], { color: SCOL[k], size: 12, bg: false, font: 'ui', bold: true });
    }
    if (opts.pulse !== undefined && opts.pulse !== null) g.marker.ring(GPOS[opts.pulse], { r: 24, color: 'gold' });
  }

  /* ---------- drawing: the simplex (3D) ---------- */
  function trailsOf(s) { return s.multi ? MULTI : (s.pi0 ? [s.pi0] : []); }
  function drawSimplex(g, s, o) {
    o = o || {};
    const P = colNorm(s.P), pi = N.stationary(P);
    const E = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
    E.forEach((e, k) => { g.line([0, 0, 0], N.scale(e, 1.25), { color: SCOL[k], width: 1.2, alpha: 0.45 }); });
    g.polygon(E, { fill: 'teal', alpha: 0.16, stroke: 'teal', width: 1.4 });
    if (!o.mini) E.forEach((e, k) => g.label(STATES[k], N.scale(e, 1.13), { color: SCOL[k], size: 12.5, font: 'ui' }));
    if (s.imageTri > 0) {
      const cols = [0, 1, 2].map(j => N.col(P, j));
      g.polygon(cols, { fill: 'w', alpha: 0.18 * s.imageTri, stroke: 'w', width: 1.4 });
    }
    if (s.showCols > 0 && !o.mini) for (let j = 0; j < 3; j++) g.point(N.col(P, j), { r: 5, color: 'w', alpha: s.showCols, label: 'c' + '₁₂₃'[j], labelColor: 'w', size: 12 });
    // trails
    const starts = o.starts || trailsOf(s);
    const K = Math.max(0, s.k);
    starts.forEach((p0, idx) => {
      const pts = trail(P, p0, Math.ceil(K) + 1);
      const cur = trailAt(pts, K);
      const shown = pts.slice(0, Math.floor(K) + 1).concat([cur]);
      const col = starts.length > 1 ? MULTI_COL[idx % 4] : 'hud';
      if (shown.length > 1) g.polyline(shown, { color: col, width: 1.6, alpha: 0.8 });
      for (let i = 0; i <= Math.floor(K) && i < pts.length; i++) g.point(pts[i], { r: 2.6, color: col, alpha: 0.8 });
      g.point(cur, { r: o.mini ? 4 : 6, color: starts.length > 1 ? col : 'white', stroke: 'bg' });
    });
    if (s.showPi) g.point(pi, { r: o.mini ? 5 : 7, color: 'gold', label: o.mini ? undefined : 'π', labelColor: 'gold' });
    if (s.recipe > 0) {
      // the recipe π₁c₁ + π₂c₂ + π₃c₃ as a weighted mix of the purple dots, built tip to tail inside the plane
      const c = [0, 1, 2].map(j => N.col(P, j));
      const B = N.scale(N.add(N.scale(c[0], pi[0]), N.scale(c[1], pi[1])), 1 / (pi[0] + pi[1]));
      const Cp = N.add(N.scale(B, 1 - pi[2]), N.scale(c[2], pi[2]));
      const u1 = LA.clamp01(s.recipe * 2), u2 = LA.clamp01(s.recipe * 2 - 1);
      g.point(c[0], { r: 6, color: 'i', stroke: 'white' });
      if (u1 > 0) g.vector(c[0], lerpV(c[0], B, u1), { color: 'j', width: 2.6, head: 10, label: o.mini ? undefined : 'mix in c₂, then c₃', labelAt: 0.35 });
      if (u2 > 0) g.vector(B, lerpV(B, Cp, u2), { color: 'k', width: 2.6, head: 10 });
      if (!o.mini) { [0, 1, 2].forEach(j => g.line(c[j], pi, { color: 'w', width: 1, dashed: true, alpha: 0.5 })); }
      if (s.recipe >= 1) g.marker.ring(pi, { r: 15, color: 'gold' });
    }
    if (s.showImageArrow && s.pi0) {
      const img = N.matVec(P, s.pi0);
      g.vector(s.pi0, img, { color: 'v', width: 2, head: 8 });
      g.point(img, { r: 4, color: 'v', label: o.mini ? undefined : 'Pπ₀', labelColor: 'v', size: 12 });
    }
    if (o.handle) g.handle('pi0', s.pi0 || pi, (p, st) => { st.pi0 = projSimplex(p); }, { plane: { point: [1 / 3, 1 / 3, 1 / 3], normal: [1, 1, 1] }, color: 'white' });
  }
  const simplexCam = () => Object.assign(LA.cam.defaults3d(), { target: [0.36, 0.36, 0.32], dist: 3.3, yaw: 0.95, pitch: 0.5, fov: 36 });

  function eigLine(g, vals, y, opts) {
    opts = opts || {};
    const X = l => l * 1.3;
    g.line([X(-1), y], [X(1), y], { color: 'gridAxis', width: 1.2 });
    [-1, 0, 1].forEach(l => { g.line([X(l), y - 0.06], [X(l), y + 0.06], { color: 'gridAxis', width: 1 }); g.label(String(l === -1 ? '−1' : l), [X(l), y], { anchor: 'below', dy: 8, size: 11, color: 'muted', font: 'ui' }); });
    const seen = {};
    vals.forEach((l, i) => {
      const key = l.toFixed(2); seen[key] = (seen[key] || 0) + 1;
      const col = i === 0 ? 'gold' : (Math.abs(Math.abs(l) - 1) < 1e-6 ? 'bad' : 'hud');
      g.point([X(l), y], { r: 5, color: col });
      g.label('λ' + '₁₂₃'[i] + ' = ' + LA.fmt(l), [X(l), y], { anchor: 'above', dy: -8 - 14 * i, size: 11.5, color: col, font: 'ui' });
    });
    if (opts.title) g.label(opts.title, [X(-1), y], { anchor: 'left', dy: 34, size: 12, color: 'muted', font: 'ui' });
  }

  /* ---------- drawing: the web (2D) ---------- */
  const WC = [-1.5, 1.85], WR = 1.0;
  const wpos = k => [WC[0] + WR * Math.cos(Math.PI / 2 - k * 2 * Math.PI / 5), WC[1] + WR * Math.sin(Math.PI / 2 - k * 2 * Math.PI / 5)];
  function drawWebGraph(g, s, opts) {
    opts = opts || {};
    const edges = s.edges, rank = opts.rank;
    const rad = k => rank ? 0.14 + 0.62 * rank[k] : 0.22;
    const pos = opts.pos || wpos;
    const seen = {};
    edges.forEach((e, idx) => {
      if (e[0] === e[1]) return;
      const key = e[0] + '-' + e[1]; if (seen[key]) return; seen[key] = 1;
      const back = edges.some(f => f[0] === e[1] && f[1] === e[0]);
      curvedArrow(g, pos(e[0]), pos(e[1]), back ? 0.18 : 0.06, rad(e[0]) + 0.03, rad(e[1]) + 0.05, { color: 'hud', width: 1.5, alpha: 0.8, head: opts.mini ? 6 : (opts.headColor ? 12 : 9), headColor: opts.headColor });
    });
    let best = -1; if (rank) best = rank.indexOf(Math.max.apply(null, rank));
    for (let k = 0; k < 5; k++) {
      g.circle(pos(k), rad(k), { fill: k === best ? 'gold' : 'data', alpha: 0.3, stroke: k === best ? 'gold' : 'data', width: k === best ? 3 : 1.6 });
      if (!opts.mini) g.label(String(k + 1), pos(k), { color: 'text', size: 13, bg: false, font: 'ui', bold: true });
    }
  }
  /** arrowhead handles: drag an arrowhead onto another page to re-point that link */
  function linkHandles(g, s, rank) {
    s.edges.forEach((e, idx) => {
      const A = wpos(e[0]), B = wpos(e[1]);
      const back = s.edges.some(f => f[0] === e[1] && f[1] === e[0]);
      const r1 = (rank ? 0.14 + 0.62 * rank[e[1]] : 0.22) + 0.05;   // same cut as curvedArrow: the handle sits on the arrowhead tip
      const pts = bez(A, B, back ? 0.18 : 0.06, 24).filter(p => Math.hypot(p[0] - B[0], p[1] - B[1]) > r1);
      if (!pts.length) return;
      const h = pts[pts.length - 1];
      g.handle('link' + idx, h, (p, st) => {
        let bk = -1, bd = 0.55;
        for (let k = 0; k < 5; k++) { const d = Math.hypot(p[0] - wpos(k)[0], p[1] - wpos(k)[1]); if (d < bd) { bd = d; bk = k; } }
        if (bk >= 0 && bk !== st.edges[idx][0]) st.edges[idx] = [st.edges[idx][0], bk];
      }, { r: 10, color: 'v' });
    });
  }
  function surferState(s) {
    const walk = surferWalk(s.edges, s.damping, 400);
    const H = LA.clamp(s.hops, 0, 400), a = Math.floor(H), b = Math.min(a + 1, 400);
    const counts = [0, 0, 0, 0, 0]; for (let i = 0; i <= a; i++) counts[walk[i]]++;
    const tot = a + 1;
    return { pos: lerpV(wpos(walk[a]), wpos(walk[b]), H - a), counts, freq: counts.map(c => c / tot), at: walk[a] };
  }
  function renderWeb(g, s, ctx) {
    const W = ctx.size.w, H = ctx.size.h;
    const P = webP(s.edges).P;
    const rank = s.rankT >= 0 ? rankAt(s.edges, s.damping, s.rankT) : null;
    const editable = ctx.mode !== 'learn' || s.linkHandles;
    if (s.inset < 0.5) {
      drawWebGraph(g, s, { rank, headColor: editable ? 'v' : undefined });
      if (editable) g.label('drag a yellow arrowhead onto another page to re-point that link', [WC[0], WC[1] - WR - (s.hops > 0 ? 0.8 : 0.45)], { color: 'v', size: 12, font: 'ui' });
      g.label('5 web pages, links = moves', [WC[0], WC[1] + WR + 0.55], { color: 'muted', size: 12.5, font: 'ui' });
    }
    // transition matrix heatmap, filled column by column
    const b = { x0: 0.5, x1: 3.0, y0: 0.65, y1: 3.15 };
    g.image(P, b, { ramp: LA.color.heat, range: [0, 1], cells: true });
    const cw = 0.5;
    for (let j = 0; j < 5; j++) {
      g.label(String(j + 1), [b.x0 + cw * (j + 0.5), b.y1], { anchor: 'above', dy: -4, size: 11.5, color: 'muted', font: 'ui' });
      g.label(String(j + 1), [b.x0, b.y1 - cw * (j + 0.5)], { anchor: 'right', dx: -6, size: 11.5, color: 'muted', font: 'ui' });
      if (j >= s.fillCols) g.polygon([[b.x0 + cw * j, b.y0], [b.x0 + cw * (j + 1), b.y0], [b.x0 + cw * (j + 1), b.y1], [b.x0 + cw * j, b.y1]], { fill: 'panel', alpha: 1 });
      else for (let i = 0; i < 5; i++) if (P[i][j] > 0.001) g.label(P[i][j] === 0.5 ? '½' : P[i][j] === 1 ? '1' : LA.fmt(P[i][j]), [b.x0 + cw * (j + 0.5), b.y1 - cw * (i + 0.5)], { size: 11, color: 'bg', bg: false, font: 'ui', bold: true });
    }
    g.label('P: column j = from page j, row i = to page i', [(b.x0 + b.x1) / 2 - 0.15, b.y0 - 0.28], { color: 'muted', size: 11.5, font: 'ui' });
    // surfer
    const sv = surferState(s);
    if (s.hops > 0 && s.hops < 399 && s.inset < 0.5) g.point(sv.pos, { r: 7, color: 'v', stroke: 'white' });
    if (s.hops > 0 && s.inset <= 0) g.label('random surfer: ' + Math.floor(s.hops) + ' hops', [WC[0], WC[1] - WR - 0.45], { color: 'v', size: 12.5, font: 'ui' });
    // bars
    const t10 = s.inset;
    const bx = W * 0.08, by = H * 0.62, bw = W * 0.84, bh = H * 0.21;
    const gap = 18, bwi = (bw - gap * 4) / 5;
    if (s.hops > 0) g.bars(sv.freq, { screen: true, x: bx + (rank ? bwi * 0.45 : 0), y: by, w: bw, h: bh, gap, max: 0.45, color: 'hud', alpha: rank ? 0.3 : 0.75, labels: rank ? undefined : ['page 1', 'page 2', 'page 3', 'page 4', 'page 5'] });
    if (rank) {
      const best = rank.indexOf(Math.max.apply(null, rank));
      g.bars(rank, { screen: true, x: bx, y: by, w: bw, h: bh, gap, max: 0.45, colors: rank.map((r, k) => k === best ? 'gold' : 'data'), labels: ['page 1', 'page 2', 'page 3', 'page 4', 'page 5'], values: true, title: s.hops > 0 ? 'PageRank r (bars) vs surfer visits (dim)' : 'PageRank r' });
    } else if (s.hops > 0) g.hud('visit frequency', bx, by - 10, { color: 'muted', size: 12 });
    if (s.inset > 0) {
      g.hud(HONEST, bx, by - 36, { color: 'text', size: 14.5, alpha: t10 });
      if (rank) { g.strip(rank, { x: bx, y: by + bh + 34, w: bw, h: 20, range: [0, 0.45], ramp: LA.color.heat }); g.hud('r as a 5-cell strip', bx, by + bh + 66, { color: 'muted', size: 11.5 }); }
      const sc = 0.55 + 0.45 * t10;
      g.viewport({ x: 0.02, y: 0.02, w: 0.46 * sc, h: 0.42 * sc }, { view: '3d', cam: Object.assign(simplexCam(), { dist: 3.3 / sc }), bg: 'bg' }, (g2, r) => {
        drawSimplex(g2, Object.assign({}, s, { P: WEATHER, pi0: null, k: 12, multi: true, showPi: true, imageTri: 0, recipe: 0, showCols: 0, showImageArrow: false }), { mini: true });
        g2.hud('3 states: drawable', r.x + 8, r.y + 14, { color: 'teal', size: 12.5 });
      });
    }
    if (editable) linkHandles(g, s, rank);
  }

  /* ---------- two-state chain (check 2) ---------- */
  function renderTwo(g, s) {
    const P2 = [[0.9, 0.5], [0.1, 0.5]];
    const A = [-2.1, -0.8], B = [2.1, -0.8];
    g.line(A, B, { color: 'teal', width: 4, alpha: 0.7 });
    g.point(A, { r: 5, color: 'i', label: 'state 1 for sure', labelColor: 'i', dx: 0, dy: 20 });
    g.point(B, { r: 5, color: 'j', label: 'state 2 for sure', labelColor: 'j', dx: 0, dy: 20 });
    const p = s.pi2, sum = p[0] + p[1];
    const q = Math.abs(sum) > 1e-6 ? [p[0] / sum, p[1] / sum] : [0.5, 0.5];
    const pos = v => lerpV(A, B, LA.clamp01(v[1]));
    const img = N.matVec(P2, q);
    g.vector(pos(q), pos(img), { color: 'v', width: 2, head: 8 });
    g.point(pos(img), { r: 4, color: 'v', label: 'Pπ', labelColor: 'v' });
    g.point(pos(q), { r: 7, color: 'white', label: 'π', labelColor: 'white', dy: -16 });
    if (Math.abs(sum - 1) > 0.02) g.label('entries should sum to 1', [0, -1.7], { color: 'bad', size: 12.5, font: 'ui' });
    // two-node graph
    const n1 = [-1.2, 1.5], n2 = [1.2, 1.5];
    curvedArrow(g, n1, n2, 0.3, 0.32, 0.34, { color: 'hud', width: 1.4 }); g.label('0.1', [0, 1.05], { color: 'hud', size: 12 });
    curvedArrow(g, n2, n1, 0.3, 0.32, 0.34, { color: 'hud', width: 3 }); g.label('0.5', [0, 1.95], { color: 'hud', size: 12 });
    g.label('the teal segment = all two-state distributions', [0, -2.3], { color: 'teal', size: 12, font: 'ui' });
    g.circle(n1, 0.3, { fill: 'i', alpha: 0.35, stroke: 'i', width: 2 }); g.label('1', n1, { color: 'i', bg: false, bold: true });
    g.circle(n2, 0.3, { fill: 'j', alpha: 0.35, stroke: 'j', width: 2 }); g.label('2', n2, { color: 'j', bg: false, bold: true });
    g.label('stay 0.9', [n1[0] - 0.75, n1[1] + 0.42], { color: 'hud', size: 12 }); g.label('stay 0.5', [n2[0] + 0.75, n2[1] + 0.42], { color: 'hud', size: 12 });
  }

  /* ---------- ML tie-in and recap ---------- */
  const TOKENS = ['the', 'cat', 'sat', 'on', 'a', 'mat'];
  function tokenMatrix() {
    const rand = LA.rng(57); const R = [];
    for (let i = 0; i < 6; i++) { const z = []; for (let j = 0; j < 6; j++) z.push(LA.rng.gauss(rand) * 1.4 + (j === (i + 1) % 6 ? 2.2 : 0)); R.push(N.softmax(z)); }
    return R;
  }
  function renderML(g, s, ctx) {
    const W = ctx.size.w, H = ctx.size.h, u = s.mlT;
    const boxes = ROWS;
    const cam = { center: [0, 0], halfH: 1.9 };
    g.viewport(boxes[0], { view: '2d', cam }, (g2) => {
      const rank = rankHistory(WEB, 0.85, 30)[30];
      drawWebGraph(g2, { edges: WEB }, { rank: rank.map(r => r * LA.clamp01(u * 1.5) + 0.2 * (1 - LA.clamp01(u * 1.5))), mini: true, pos: k => [1.2 * Math.cos(Math.PI / 2 - k * 1.2566), 1.2 * Math.sin(Math.PI / 2 - k * 1.2566) - 0.1] });
    });
    g.viewport(boxes[1], { view: '2d', cam }, (g2) => {
      const c = 0.6, o = -1.5;
      for (let i = 0; i <= 5; i++) { g2.line([o + i * c, o], [o + i * c, o + 5 * c], { color: 'grid', width: 1 }); g2.line([o, o + i * c], [o + 5 * c, o + i * c], { color: 'grid', width: 1 }); }
      g2.polygon([[o + 4 * c, o + 4 * c], [o + 5 * c, o + 4 * c], [o + 5 * c, o + 5 * c], [o + 4 * c, o + 5 * c]], { fill: 'gold', alpha: 0.45 });
      const path = [[0, 0], [1, 0], [1, 1], [2, 1], [2, 2], [3, 2], [3, 3], [4, 3], [4, 4]];
      const T = u * (path.length - 1), a = Math.min(Math.floor(T), path.length - 1), b = Math.min(a + 1, path.length - 1);
      const cell = p => [o + (p[0] + 0.5) * c, o + (p[1] + 0.5) * c];
      g2.polyline(path.slice(0, a + 1).map(cell).concat([lerpV(cell(path[a]), cell(path[b]), T - a)]), { color: 'v', width: 2, alpha: 0.7 });
      const w = lerpV(cell(path[a]), cell(path[b]), T - a);
      [[1, 0, 0.4], [0, 1, 0.4], [-1, 0, 0.1], [0, -1, 0.1]].forEach(d => g2.vector(w, [w[0] + d[0] * d[2] * 0.9, w[1] + d[1] * d[2] * 0.9], { color: 'hud', width: 1.4, head: 6 }));
      g2.point(w, { r: 6, color: 'v' });
    });
    g.viewport(boxes[2], { view: '2d', cam }, (g2) => {
      const R = tokenMatrix();
      g2.image(R, { x0: -1.2, x1: 1.5, y0: -1.4, y1: 1.3 }, { ramp: LA.color.heat, range: [0, 0.8], cells: true });
      TOKENS.forEach((tk, i) => { g2.label(tk, [-1.2, 1.3 - 0.45 * (i + 0.5)], { anchor: 'right', dx: -5, size: 11, color: 'muted', font: 'ui' }); g2.label(tk, [-1.2 + 0.45 * (i + 0.5), 1.3], { anchor: 'above', dy: -3, size: 10, color: 'muted', font: 'ui' }); });
      g2.label('each row: softmax, sums to 1', [0.15, -1.75], { size: 11.5, color: 'hud', font: 'ui' });
    });
    const caps = [['PageRank', 'node size = stationary π'], ['Reinforcement learning', 'Markov decision process'], ['Language model', 'next-token stochastic matrix']];
    rowCaptions(g, W, H, caps);
  }
  const ROWS = [{ x: 0.03, y: 0.03, w: 0.55, h: 0.3 }, { x: 0.03, y: 0.35, w: 0.55, h: 0.3 }, { x: 0.03, y: 0.67, w: 0.55, h: 0.3 }];
  function rowCaptions(g, W, H, caps) {
    ROWS.forEach((b, i) => { const y = (b.y + b.h / 2) * H; g.hud(caps[i][0], (b.x + b.w + 0.03) * W, y - 10, { color: 'text', size: 14, bold: true }); g.hud(caps[i][1], (b.x + b.w + 0.03) * W, y + 12, { color: 'muted', size: 12.5 }); });
  }
  function renderRecap(g, s, ctx) {
    const W = ctx.size.w, H = ctx.size.h;
    const boxes = ROWS;
    const base = Object.assign({}, s, { P: WEATHER, imageTri: 0, showCols: 0, showImageArrow: false });
    g.viewport(boxes[0], { view: '3d', cam: simplexCam() }, (g2) => drawSimplex(g2, Object.assign({}, base, { multi: true, k: 12, showPi: true, recipe: 0 }), { mini: true }));
    g.viewport(boxes[1], { view: '3d', cam: Object.assign(simplexCam(), { yaw: Math.PI / 4, pitch: 0.6155, dist: 2.1, target: [0.39, 0.35, 0.26] }) }, (g2) => drawSimplex(g2, Object.assign({}, base, { multi: false, pi0: null, k: 0, showPi: true, recipe: 1, imageTri: 1 }), { mini: true }));
    g.viewport(boxes[2], { view: '2d', cam: { center: [1.6, 1.1], halfH: 1.8 } }, (g2) => {
      const r = rankHistory(WEB, 0.85, 30)[30];
      g2.bars(r, { origin: [0, 0], width: 0.5, gap: 0.2, height: 2.4, max: 0.4, colors: r.map((x, k) => k === 2 ? 'gold' : 'data'), labels: ['1', '2', '3', '4', '5'] });
    });
    rowCaptions(g, W, H, [['Step = Pπ', 'every start meets gold π'], ['Fixed point', 'Pπ = π, λ = 1'], ['PageRank', '5 pages → 5 bars']]);
  }

  /* ---------- the simplex scene (split: graph left, triangle right) ---------- */
  function renderSimplexScene(g, s, ctx) {
    const P = colNorm(s.P), pi = N.stationary(P);
    // bottom: the triangle in 3D (uses the lesson camera, so the learner can orbit)
    g.viewport({ x: 0, y: 0.37, w: 1, h: 0.63 }, { view: '3d', cam: s.cam, border: false }, (g2) => {
      drawSimplex(g2, s, { handle: ctx.mode !== 'learn' });
      if (s.crossA > 0) {
        g2.label('different endings?', [0.9, 0.05, 0.62], { color: 'bad', size: 13, font: 'ui', alpha: s.crossA });
        if (s.crossA > 0.5) g2.marker.cross([0.9, 0.05, 0.62], { dy: -24 });
      }
      if (s.okMark) g2.marker.check(pi, { dx: 24, dy: -22 });
    });
    // top left: the state graph
    g.viewport({ x: 0, y: 0, w: 0.5, h: 0.37 }, { view: '2d', cam: { center: [0, 1.15], halfH: 1.78 } }, (g2) => {
      drawGraph(g2, P, { focus: s.focus, pulse: s.pulse });
    });
    // top right: eigenvalues, bars or the convergence chart
    g.viewport({ x: 0.5, y: 0, w: 0.5, h: 0.37 }, { view: '2d', cam: { center: [0, 0], halfH: 1.6 } }, (g2, r) => {
      const starts = trailsOf(s);
      if (s.showEig) eigLine(g2, eig3(P).values, 0.75, { title: s.showBars && !s.chart ? null : 'eigenvalues of P' });
      if (s.chart > 0 && starts.length) {
        const pts = trail(P, starts[0], 20), K = Math.floor(12 * s.chart);
        const l2 = Math.abs(eig3(P).values[1]), d0 = N.norm(N.sub(pts[0], pi));
        const ser = [], ref = [];
        for (let k = 0; k <= K; k++) { ser.push([k, Math.log10(Math.max(1e-6, N.norm(N.sub(pts[k], pi))))]); ref.push([k, Math.log10(Math.max(1e-6, d0 * Math.pow(l2, k)))]); }
        g2.lineChart([{ points: ref, color: 'muted', dash: [5, 4], label: LA.fmt(l2) + 'ᵏ' }, { points: ser, color: 'v' }], { x: r.x + 20, y: r.y + r.h * 0.58, w: r.w - 70, h: r.h * 0.28, xRange: [0, 12], yRange: [-3, 0.3], xLabel: 'day k', title: 'log₁₀ |π_k − π| (yellow) vs ' + LA.fmt(l2) + 'ᵏ (dashed)' });
      } else if (starts.length && s.showBars) {
        const pts = trail(P, starts[0], Math.ceil(s.k) + 1), cur = trailAt(pts, s.k);
        const y0 = s.showEig ? r.y + r.h * 0.5 : r.y + r.h * 0.2, hh = s.showEig ? r.h * 0.3 : r.h * 0.58;
        g2.bars(cur, { screen: true, x: r.x + r.w * 0.14, y: y0, w: r.w * 0.72, h: hh, max: 1, colors: SCOL, labels: STATES, values: true, title: 'π on day ' + Math.round(s.k) });
      } else if (s.panel === 'check1') {
        g2.hud('yellow arrow: today π₀ → tomorrow Pπ₀', r.x + 12, r.y + r.h * 0.3, { size: 12.5, color: 'v' });
        g2.hud('drag the white point until the arrow', r.x + 12, r.y + r.h * 0.48, { size: 12.5, color: 'hud' });
        g2.hud('shrinks to nothing', r.x + 12, r.y + r.h * 0.62, { size: 12.5, color: 'hud' });
      } else if (s.panel === 'cycle') {
        g2.hud('day ' + Math.floor(s.k) + ': the point hops sunny ↔ rainy', r.x + 12, r.y + r.h * 0.62, { size: 12.5, color: 'text' });
        g2.hud('a second |λ| = 1: it never forgets, no limit', r.x + 12, r.y + r.h * 0.78, { size: 12.5, color: 'bad' });
      } else if (s.recipe > 0) {
        g2.hud('Pπ = π₁c₁ + π₂c₂ + π₃c₃', r.x + 12, r.y + r.h * 0.22, { size: 14, color: 'text' });
        g2.hud('weights π = ' + pi.map(x => LA.fmt(x)).join(', '), r.x + 12, r.y + r.h * 0.4, { size: 12.5, color: 'gold' });
        g2.hud('a mix of the purple dots c₁ c₂ c₃', r.x + 12, r.y + r.h * 0.56, { size: 12.5, color: 'w' });
        g2.hud(s.recipe >= 1 ? 'lands back on π  ✓' : '…', r.x + 12, r.y + r.h * 0.72, { size: 12.5, color: 'gold' });
      } else if (s.multi) {
        g2.hud('four different starts:', r.x + 12, r.y + r.h * 0.2, { size: 12.5, color: 'hud' });
        ['sunny corner [1, 0, 0]', 'cloudy corner [0, 1, 0]', 'rainy corner [0, 0, 1]', 'inside: [0.2, 0.3, 0.5]'].forEach((txt, i) => g2.hud('● ' + txt, r.x + 18, r.y + r.h * (0.36 + 0.15 * i), { size: 12.5, color: MULTI_COL[i] }));
      } else if (!s.showEig) {
        g2.hud('axes: x = sunny, y = cloudy, z = rainy', r.x + 12, r.y + r.h * 0.3, { size: 12.5, color: 'hud' });
        g2.hud('purple dots c₁ c₂ c₃ = columns of P', r.x + 12, r.y + r.h * 0.48, { size: 12.5, color: 'w' });
        g2.hud('teal triangle = all distributions', r.x + 12, r.y + r.h * 0.66, { size: 12.5, color: 'teal' });
      }
    });
  }

  LA.registerLesson({
    id: 'p4-05-markov-pagerank',
    part: 4,
    title: 'Markov chains converge to an eigenvector; PageRank',
    subtitle: 'Step a distribution with a stochastic matrix and it settles on the eigenvector with λ = 1. PageRank is that eigenvector for the web.',
    minutes: 13,
    prereqs: ['p4-04-simplex-softmax', 'p4-03-diagonalization-powers', 'p2-02-matrix-vector'],
    tags: ['core', 'capstone'],
    badge: { drawnIn: '3D triangle (3 states) + bars (5 pages)', realSize: 'PageRank ran on ~10⁹ pages; a language model’s next-token matrix is 50 000 × 50 000', transfers: 'π ← Pπ, Pπ = π and convergence at rate |λ₂| hold for any number of states; only the triangle becomes undrawable' },
    cantShow: ['the 4-dimensional simplex of five pages (we draw bars instead and say so)', 'chains that never settle beyond one two-cycle example', 'why damping guarantees a unique answer (a theorem, stated not proved)'],
    terms: [
      { term: 'Markov chain', def: 'A system that jumps between states with fixed probabilities that depend only on the current state.' },
      { term: 'transition matrix', def: 'The matrix P whose column j holds the probabilities of going from state j to each state.' },
      { term: 'stationary distribution', def: 'A distribution π that one more step leaves unchanged: Pπ = π.' },
      { term: 'PageRank', def: 'The stationary distribution of a random web surfer who follows links, with occasional random jumps (damping).' },
      { term: 'random surfer', def: 'The imaginary walker whose long-run share of visits to each page is its PageRank.' },
    ],
    atlas: ['markov', 'softmax', 'embeddings'],
    thumbnail: 2,

    state() {
      return {
        view: '3d', cam: simplexCam(),
        scene: 'simplex', panel: 'P',
        P: LA.clone(WEATHER), pi0: null, k: 0, multi: false,
        showCols: 1, showPi: false, imageTri: 0, recipe: 0, showImageArrow: false, showBars: false,
        focus: null, pulse: null, showEig: false, chart: 0, crossA: 0, okMark: false,
        edges: LA.clone(WEB), damping: 0.85, rankT: -1, hops: 0, fillCols: 5, inset: 0, linkHandles: false,
        pi2: [0.5, 0.5], mlT: 1,
      };
    },

    render(g, s, ctx) {
      if (s.scene === 'pagerank') return renderWeb(g, s, ctx);
      if (s.scene === 'two') return renderTwo(g, s);
      if (s.scene === 'ml') return renderML(g, s, ctx);
      if (s.scene === 'recap') return renderRecap(g, s, ctx);
      renderSimplexScene(g, s, ctx);
    },

    math(s, tex) {
      const P = colNorm(s.P), pi = N.stationary(P);
      const sym = (n, o) => tex.sym(n, o);
      const tn = (x, o) => tex.text(LA.fmt(x), o);   // scalar as text (see report: tex.num beside tex.mat can be mis-updated)
      const starts = trailsOf(s);
      const cur = starts.length ? trailAt(trail(P, starts[0], Math.ceil(Math.max(0, s.k)) + 1), Math.max(0, s.k)) : pi;
      const ev = eig3(P);
      const evText = ev.real ? ev.values.map(v => LA.fmt(v)).join(', ') : '1, |λ| = ' + LA.fmt(ev.values[1]) + ' (complex pair)';
      switch (s.panel) {
        case 'P': return tex.block(tex.eq(sym('P'), tex.op('='), tex.mat(P, { colors: SCOL })), tex.eq(tex.text('column j = “from state j”, each sums to 1')));
        case 'step1': return tex.block(tex.eq(sym('P'), tex.op('='), tex.mat(P, { colors: SCOL, highlight: (i, j) => j === 0 })), tex.eq(sym('π', { sub: '1' }), tex.op('='), sym('P'), sym('π', { sub: '0' }), tex.op('='), tex.text('col₁'), tex.op('='), tex.vec(N.col(P, 0), { digits: 2 })));
        case 'iter': return tex.block(tex.eq(sym('π', { sub: 'k' }), tex.op('='), sym('P', { sup: 'k' }), sym('π', { sub: '0' }), tex.op('='), tex.vec(cur, { colors: 'rows' })), tex.eq(sym('π', { color: 'gold' }), tex.op('≈'), tex.vec(pi, { color: 'gold' })));
        case 'forget': return tex.block(tex.eq(tex.text('lim'), sym('P', { sup: 'k' }), sym('π', { sub: '0' }), tex.op('='), sym('π', { color: 'gold' }), tex.text(' for every π₀')));
        case 'fixed': return tex.block(tex.eq(sym('P'), sym('π', { color: 'gold' }), tex.op('='), tex.vec(N.matVec(P, pi), { color: 'gold' }), tex.op('='), sym('π', { color: 'gold' })), tex.eq(sym('λ', { sub: '1' }), tex.op('='), tn(1, { color: 'gold' })));
        case 'lambda2': return tex.block(tex.eq(tex.abs(sym('π', { sub: 'k' }) + tex.op('−') + sym('π', { color: 'gold' })), tex.op('∝'), tex.abs(sym('λ', { sub: '2' })) + '<sup>k</sup>'), tex.eq(sym('λ'), tex.op('='), tex.text(evText)));
        case 'cycle': return tex.block(tex.eq(sym('P'), tex.op('='), tex.mat(P, { colors: SCOL })), tex.eq(sym('λ'), tex.op('='), tex.text(evText)), tex.eq(tex.text('no limit', { color: 'bad' })));
        case 'prP': return tex.block(tex.eq(sym('P', { sub: 'ij' }), tex.op('='), tex.text('1/out(j) if j → i, else 0')), tex.eq(sym('P'), tex.op('='), tex.mat(webP(s.edges).P, { digits: 2 })));
        case 'prM': {
          const r = s.rankT >= 0 ? rankAt(s.edges, s.damping, s.rankT) : rankHistory(s.edges, s.damping, 40)[40];
          return tex.block(tex.eq(sym('M'), tex.op('='), tn(s.damping), sym('P'), tex.op('+'), tex.text(LA.fmt(1 - s.damping) + '/5 · 𝟙𝟙ᵀ')), tex.eq(sym('M'), sym('r', { color: 'data' }), tex.op('='), sym('r', { color: 'data' })), tex.eq(sym('r', { color: 'data' }), tex.op('≈'), tex.vec(r, { color: 'data' })));
        }
        case 'prIter': return tex.block(tex.eq(sym('r', { sub: 't+1' }), tex.op('='), sym('M'), sym('r', { sub: 't' })), tex.eq(tex.text('5 entries → bars; 10⁹ entries → same formula')));
        case 'two': return tex.block(tex.eq(sym('P'), tex.op('='), tex.mat([[0.9, 0.5], [0.1, 0.5]], { colors: ['i', 'j'] })), tex.eq(sym('π'), tex.op('='), tex.vec(s.pi2), tex.op('→'), sym('P'), sym('π'), tex.op('='), tex.vec(N.matVec([[0.9, 0.5], [0.1, 0.5]], s.pi2), { color: 'v' })));
        case 'check1': return tex.block(tex.eq(sym('P'), tex.op('='), tex.mat(P, { colors: SCOL })), tex.eq(sym('π', { sub: '0' }), tex.op('='), tex.vec(s.pi0)), tex.eq(tex.text('step |Pπ₀ − π₀| = ' + LA.fmt(N.norm(N.sub(N.matVec(P, s.pi0), s.pi0)), 3), { color: 'v' })));
        case 'play': {
          const parts = [tex.eq(sym('P'), tex.op('='), tex.mat(P, { colors: SCOL }))];
          if (s.pi0) parts.push(tex.eq(sym('P'), sym('π', { sub: '0' }), tex.op('='), tex.vec(N.matVec(P, s.pi0), { color: 'v' }), tex.text(' step size ' + LA.fmt(N.norm(N.sub(N.matVec(P, s.pi0), s.pi0))))));
          parts.push(tex.eq(sym('π', { color: 'gold' }), tex.op('≈'), tex.vec(pi, { color: 'gold' })));
          parts.push(tex.eq(sym('λ'), tex.op('='), tex.text(evText)));
          return tex.block.apply(null, parts);
        }
        default: return tex.block(tex.eq(sym('P'), sym('π', { color: 'gold' }), tex.op('='), sym('π', { color: 'gold' })));
      }
    },

    steps: [
      { title: 'A weather machine', duration: 3.5,
        narration: 'Three kinds of day: sunny, cloudy, rainy. From a sunny day, tomorrow is sunny with 0.8, cloudy 0.1, rainy 0.1. Each state has its own column of chances: the <b>transition matrix</b> P. Columns sum to 1, so each is a point on the teal triangle from last lesson: the purple dots.',
        setup(s) { s.scene = 'simplex'; s.view = '3d'; s.cam = simplexCam(); s.P = LA.clone(WEATHER); s.pi0 = null; s.k = 0; s.multi = false; s.showPi = false; s.imageTri = 0; s.recipe = 0; s.showImageArrow = false; s.showBars = false; s.focus = null; s.pulse = null; s.showEig = false; s.chart = 0; s.crossA = 0; s.okMark = false; s.panel = 'P'; s.showCols = 0; },
        at(s, t) { s.showCols = seg(t, 0.3, 0.8); LA.cam.tween(s, { yaw: 0.8 }, { yaw: 0.95 }, seg(t, 0, 1)); } },

      { title: 'One day: the point jumps to a column', duration: 3.5,
        narration: 'Today is certainly sunny: π₀ = [1, 0, 0], the sunny corner. Tomorrow’s distribution is Pπ₀, the recipe “1 × column 1”, so the white point jumps to the first purple dot, [0.8, 0.1, 0.1]. Matrix times vector, exactly as in Part 2.',
        setup(s) { s.showCols = 1; s.pi0 = [1, 0, 0]; s.k = 0; s.panel = 'step1'; s.focus = 0; s.pulse = 0; s.showBars = true; },
        at(s, t) { s.k = seg(t, 0.3, 1); } },

      { title: 'Twelve days: the trail settles', duration: 5,
        narration: 'Keep going: π₂ = Pπ₁, π₃ = Pπ₂, and so on. The point moves less each day and settles at a gold spot, π ≈ [0.41, 0.36, 0.23]: in the long run 41 % of days are sunny whatever today was like. This is power iteration, and the dominant eigenvalue is 1.',
        setup(s) { s.focus = null; s.pulse = null; s.panel = 'iter'; s.showBars = true; },
        at(s, t) { s.k = 1 + 11 * seg(t, 0.03, 0.9, 'linear'); s.showPi = t > 0.75; } },

      { title: 'Misconception: the start decides the end', duration: 4, misconception: true,
        narration: '✕ “Start on a rainy day and you get a different long-run forecast.” Four trails start from the three corners and from an interior point: all four arrive at the same gold spot. The chain <i>forgets</i> its start. The <b>stationary distribution</b> belongs to P, not to π₀.',
        setup(s) { s.multi = true; s.showBars = false; s.showPi = true; s.panel = 'forget'; s.k = 0; s.crossA = 1; s.okMark = false; },
        at(s, t) { s.k = 12 * seg(t, 0.05, 0.85, 'in'); s.crossA = 1 - seg(t, 0.6, 0.9); s.okMark = t > 0.85; } },

      { title: 'It is an eigenvector: Pπ = π', duration: 3.5,
        narration: 'At the gold spot one more day changes nothing: <b>Pπ = π</b>. That is Av = λv with λ = 1, so π is an eigenvector of P. P maps the teal triangle into the purple one inside it, so something stays fixed. The recipe π₁c₁ + π₂c₂ + π₃c₃ lands back on π.',
        setup(s) { s.multi = false; s.pi0 = null; s.k = 0; s.crossA = 0; s.okMark = false; s.showPi = true; s.panel = 'fixed'; s.imageTri = 0; s.recipe = 0; },
        at(s, t) {
          s.imageTri = seg(t, 0, 0.3); s.recipe = seg(t, 0.3, 0.95, 'linear');
          const c0 = simplexCam();
          LA.cam.tween(s, { yaw: c0.yaw, pitch: c0.pitch, dist: c0.dist, target: c0.target }, { yaw: Math.PI / 4, pitch: 0.6155, dist: 2.3, target: [0.39, 0.35, 0.26] }, seg(t, 0, 0.3));
        } },

      { title: 'How fast it forgets: λ₂', duration: 3.5,
        narration: 'The other eigenvalues of P are 0.67 and 0.33. The distance from π<sub>k</sub> to π shrinks by about 0.67 per day: the second eigenvalue, from p4-03. A chain with λ₂ close to 1 remembers its start for a long time; a chain with λ₂ near 0 forgets overnight.',
        setup(s) { s.cam = simplexCam(); s.imageTri = 0; s.recipe = 0; s.pi0 = [1, 0, 0]; s.k = 12; s.panel = 'lambda2'; s.showEig = true; s.chart = 0; s.showBars = false; },
        at(s, t) { s.chart = Math.max(0.001, seg(t, 0.1, 0.9, 'linear')); } },

      { title: 'Not every chain settles', duration: 3,
        narration: 'Swap sunny and rainy every day with certainty: P is a permutation. The point bounces between two corners forever; the eigenvalues are 1, 1 and −1. A second eigenvalue of size 1 means no forgetting. Real chains with some chance of staying put avoid this; PageRank forces it with damping.',
        setup(s) { s.P = LA.clone(CYCLE); s.chart = 0; s.showEig = true; s.showPi = false; s.pi0 = [1, 0, 0]; s.k = 0; s.panel = 'cycle'; s.showCols = 0; },
        at(s, t) { s.k = 6 * seg(t, 0.05, 0.95, 'linear'); } },

      { title: 'PageRank: pages as states, links as moves', duration: 4,
        narration: 'Now the states are five web pages and a move is clicking a random link. Page 1 links to 2 and 3, so column 1 is [0, ½, ½, 0, 0]. The <b>random surfer</b> hops along links; the share of time on each page is a 5-entry distribution, drawn as bars because the 5-state triangle is 4-dimensional.',
        setup(s) { s.scene = 'pagerank'; s.view = '2d'; s.cam = LA.cam.defaults2d(); s.P = LA.clone(WEATHER); s.showEig = false; s.edges = LA.clone(WEB); s.damping = 0.85; s.rankT = -1; s.hops = 0; s.fillCols = 0; s.inset = 0; s.panel = 'prP'; },
        at(s, t) { s.fillCols = 5 * seg(t, 0, 0.5, 'linear'); s.hops = 8 * seg(t, 0.3, 1, 'linear'); } },

      { title: 'Damping and the ranking', duration: 4.5,
        narration: 'With probability 0.15 the surfer ignores links and jumps to a random page. That blends P with a uniform matrix: M = 0.85·P + 0.15·(1/5)·ones. M has no zero entries, so the chain settles on one answer. Power-iterate: the bars converge to <b>PageRank</b>. Page 3 wins: everyone links to it.',
        setup(s) { s.fillCols = 5; s.rankT = 0; s.hops = 8; s.panel = 'prM'; },
        at(s, t) { s.rankT = 12 * seg(t, 0.1, 0.85, 'linear'); s.hops = LA.lerp(8, 400, seg(t, 0, 0.8)); } },

      { title: 'The picture stops at 3; the bars do not', duration: 3,
        narration: 'Five pages give a distribution with five entries: a point on a 4-dimensional simplex we cannot draw, so we draw bars. Google’s version had billions. The picture stops at 3. The formula does not: r = Mr, found by power iteration, exactly as for three kinds of weather.',
        setup(s) { s.rankT = 12; s.hops = 400; s.inset = 0; s.panel = 'prIter'; },
        at(s, t) { s.inset = seg(t, 0, 0.6); } },

      { title: 'Why ML cares', duration: 3.5, mlTie: true,
        narration: 'PageRank is linear algebra’s most famous product: one stochastic matrix, one eigenvector. The same mathematics runs Markov decision processes in reinforcement learning, random-walk embeddings on graphs, and the view of a language model as a giant stochastic matrix over tokens whose rows are softmax outputs.',
        setup(s) { s.scene = 'ml'; s.view = '2d'; s.cam = LA.cam.defaults2d(); s.mlT = 0; s.panel = 'ml'; },
        at(s, t) { s.mlT = seg(t, 0.05, 0.95, 'linear'); } },

      { title: 'Recap', duration: 2.5, recap: true,
        narration: 'Columns of P are probability vectors; a step is π ← Pπ. Repeating converges to the stationary π with Pπ = π (λ = 1), from any start, at speed |λ₂|. PageRank is this with pages as states and damping to guarantee one answer.',
        cards: ['<b>Step = Pπ.</b> Every start walks the triangle to the same gold π.', '<b>Fixed point.</b> Pπ = π: π is the eigenvector with λ = 1; |λ₂| sets the speed.', '<b>PageRank.</b> Pages are states, links are moves, damping makes the answer unique: r = Mr.'],
        setup(s) { s.scene = 'recap'; s.view = '2d'; s.cam = LA.cam.defaults2d(); s.panel = 'recap'; },
        at(s, t) { } },
    ],

    playground: {
      intro: '<p>Drag the white point on the triangle to choose today’s distribution, edit P (columns are renormalized to sum to 1), or pick a preset. In the web presets, drag a yellow arrowhead onto another page to re-point that link.</p>',
      setup(s) { s.scene = 'simplex'; s.view = '3d'; s.cam = simplexCam(); s.pi0 = [1, 0, 0]; s.k = 8; s.showPi = true; s.showImageArrow = true; s.showBars = true; s.showEig = true; s.showCols = 1; s.panel = 'play'; },
      controls: [
        { type: 'matrix', bind: 'P', label: 'P', colors: ['i', 'j', 'k'], step: 0.05, digits: 2, min: 0, max: 1 },
        { type: 'slider', bind: 'k', min: 0, max: 20, step: 1, label: 'steps k', color: 'v' },
        { type: 'slider', bind: 'damping', min: 0.5, max: 1, step: 0.01, label: 'damping', color: 'gold' },
        { type: 'slider', bind: 'rankT', min: 0, max: 12, step: 1, label: 'PageRank iterations' },
        { type: 'preset', label: 'Presets', items: [
          { label: 'Weather', apply(s) { s.scene = 'simplex'; s.view = '3d'; s.cam = simplexCam(); s.P = LA.clone(WEATHER); s.panel = 'play'; s.showEig = true; } },
          { label: 'Sticky (λ₂ = 0.95)', apply(s) { s.scene = 'simplex'; s.view = '3d'; s.cam = simplexCam(); s.P = LA.clone(SLOW); s.panel = 'play'; s.showEig = true; s.k = 20; } },
          { label: 'Two-cycle', apply(s) { s.scene = 'simplex'; s.view = '3d'; s.cam = simplexCam(); s.P = LA.clone(CYCLE); s.panel = 'play'; s.showEig = true; } },
          { label: 'Web of 5', apply(s) { s.scene = 'pagerank'; s.view = '2d'; s.cam = LA.cam.defaults2d(); s.edges = LA.clone(WEB); s.fillCols = 5; s.rankT = 12; s.hops = 400; s.panel = 'prM'; } },
          { label: 'Web with a dead end', apply(s) { s.scene = 'pagerank'; s.view = '2d'; s.cam = LA.cam.defaults2d(); s.edges = LA.clone(WEB_DEAD); s.fillCols = 5; s.rankT = 12; s.hops = 400; s.panel = 'prM'; } },
        ] },
        { type: 'info', html: 'A page with no out-links (a dead end) is treated as “jump anywhere”: its column is uniform.' },
      ],
    },

    checks: [
      { kind: 'manipulate',
        prompt: 'Drag the white point to the stationary distribution of the “Sticky” matrix: the place where one more step changes nothing.',
        setup(s) { s.scene = 'simplex'; s.view = '3d'; s.cam = simplexCam(); s.P = LA.clone(STICKY); s.pi0 = [0, 0, 1]; s.k = 0; s.showPi = false; s.showImageArrow = true; s.showBars = false; s.showCols = 0; s.panel = 'check1'; },
        judge(s) { const P = colNorm(s.P); const d = N.norm(N.sub(N.matVec(P, s.pi0), s.pi0)); return { ok: d < 0.012, progress: Math.max(0, 1 - d / 0.25) }; },
        hint: 'Watch the yellow arrow from the point to its image Pπ₀ shrink.',
        solution(s) { s.pi0 = N.stationary(colNorm(s.P)); } },
      { kind: 'entry',
        prompt: 'Type the stationary distribution of the two-state chain P = [[0.9, 0.5], [0.1, 0.5]].',
        bind: 'pi2', answer: [0.83, 0.17], tol: LA.TOL.entry, label: 'π', colors: 'v',
        setup(s) { s.scene = 'two'; s.view = '2d'; s.cam = LA.cam.defaults2d(); s.pi2 = [0.5, 0.5]; s.panel = 'two'; } },
      { kind: 'manipulate',
        prompt: 'Make page 2 outrank page 1. Each link ends in a yellow arrowhead: drag one and drop it on another page, and that link now points there. Watch the bars.',
        setup(s) { s.scene = 'pagerank'; s.view = '2d'; s.cam = LA.cam.defaults2d(); s.edges = LA.clone(WEB); s.damping = 0.85; s.rankT = 12; s.hops = 0; s.fillCols = 5; s.panel = 'prM'; },
        judge(s) { const r = rankHistory(s.edges, s.damping, 40)[40]; const d = r[1] - r[0]; return { ok: d > 0.02, progress: LA.clamp01((d + 0.2) / 0.22) }; },
        hint: 'Rank flows along links: point more links at page 2, or take page 1’s incoming link away.',
        solution(s) { s.edges = [[0, 1], [0, 2], [1, 2], [2, 1], [3, 1], [3, 4], [4, 3], [4, 1]]; } },
    ],
  });
})();
