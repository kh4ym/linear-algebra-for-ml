/* p1-08 — High dimensions: what we can't draw, we measure. */
(function () {
  'use strict';
  const N = LA.num, seg = LA.seg, lerp = LA.lerp, lerpV = LA.lerpV, C = LA.color;
  const DS = [2, 3, 5, 10, 20, 50, 100, 200, 300, 500, 1000];   // dimensions measured (index = s.di)
  const I3 = DS.indexOf(3), I300 = DS.indexOf(300), I1000 = DS.indexOf(1000), I2 = 0;
  const HONEST = 'The picture stops at 3. The formula does not.';
  const POOL = 64;                                   // 64 random directions → all 2016 pairs
  const PAIRS = POOL * (POOL - 1) / 2;
  const BINS = 40;
  const TPL = ['horizontal bar', 'vertical bar', 'diagonal'];
  const TPL_COL = ['data', 'dataB', 'dataC'];
  const WORDS = ['cat', 'kitten', 'dog', 'invoice', 'tax'];
  const WABBR = ['cat', 'kit', 'dog', 'inv', 'tax'];
  const SUP = { '-': '⁻', 0: '⁰', 1: '¹', 2: '²', 3: '³', 4: '⁴', 5: '⁵', 6: '⁶', 7: '⁷', 8: '⁸', 9: '⁹' };
  const pow10 = e => '10' + String(e).split('').map(ch => SUP[ch]).join('');
  const f2 = x => LA.fmt.fixed(x, 2);
  const pct = x => Math.round(100 * x) + '%';

  // ---------------- measurements (computed once per seed, memoized) ----------------
  const MEMO = {};
  /** distances from probe 0 to 199 uniform points in [0,1]^d, for every d in DS; cosines of random unit pairs */
  function measure(seed) {
    if (MEMO[seed]) return MEMO[seed];
    const rand = LA.rng(seed);
    const out = { ratio: [], d1: [], d10: [], d199: [], near: [], far: [], sorted: [], pts: {}, std: [], counts: [] };
    DS.forEach((d) => {
      const keep = d <= 3;
      const p0 = new Array(d); for (let k = 0; k < d; k++) p0[k] = rand();
      const pts = keep ? [p0] : null, ds = new Array(199);
      for (let j = 0; j < 199; j++) {
        let sum = 0; const q = keep ? new Array(d) : null;
        for (let k = 0; k < d; k++) { const x = rand(); if (keep) q[k] = x; const e = p0[k] - x; sum += e * e; }
        ds[j] = Math.sqrt(sum); if (keep) pts.push(q);
      }
      let ni = 0, fi = 0; for (let j = 1; j < 199; j++) { if (ds[j] < ds[ni]) ni = j; if (ds[j] > ds[fi]) fi = j; }
      const so = ds.slice().sort((a, b) => a - b);
      out.ratio.push(so[0] / so[198]); out.d1.push(so[0]); out.d10.push(so[9]); out.d199.push(so[198]);
      out.near.push(ni + 1); out.far.push(fi + 1); out.sorted.push(so);
      if (keep) out.pts[d] = pts;
      // cosines between random unit vectors (LA.rng.tokenVectors): all pairs of a pool of 64
      const V = LA.rng.tokenVectors(seed * 7919 + d, POOL, d);
      const counts = new Array(BINS).fill(0); let s1 = 0, s2 = 0;
      for (let i = 0; i < POOL; i++) for (let j = i + 1; j < POOL; j++) {
        const a = V[i], b = V[j]; let c = 0; for (let k = 0; k < d; k++) c += a[k] * b[k];
        s1 += c; s2 += c * c; let bi = Math.floor((c + 1) / 2 * BINS); if (bi < 0) bi = 0; if (bi >= BINS) bi = BINS - 1; counts[bi]++;
      }
      const m = s1 / PAIRS; out.std.push(Math.sqrt(Math.max(0, s2 / PAIRS - m * m))); out.counts.push(counts);
    });
    // 16 random directions in ℝ⁷⁶⁸: |cos| matrix
    const T = LA.rng.tokenVectors(seed * 7919 + 768, 16, 768); const room = []; let mx = 0;
    for (let i = 0; i < 16; i++) { room.push([]); for (let j = 0; j < 16; j++) { const c = i === j ? 1 : Math.abs(N.dot(T[i], T[j])); room[i].push(c); if (i !== j) mx = Math.max(mx, c); } }
    out.room = room; out.roomMax = mx;
    MEMO[seed] = out;
    return out;
  }
  /** the compact statistics kept in the state (counts only, no raw samples) */
  function compact(m) {
    const r = x => Math.round(x * 1e4) / 1e4;
    return { ratio: m.ratio.map(r), d1: m.d1.map(r), d10: m.d10.map(r), d199: m.d199.map(r), near: m.near.slice(0, 2), far: m.far.slice(0, 2), std: m.std.map(r), counts: m.counts.map(c => c.slice()), roomMax: r(m.roomMax) };
  }
  // inscribed-ball fraction: V_d = (2π/d)·V_{d−2}, V₀ = 1, V₁ = 2; fraction = V_d / 2ᵈ (unit-radius ball in a side-2 cube)
  const FRACS = (() => { const V = [1, 2]; for (let d = 2; d <= 20; d++) V.push(2 * Math.PI / d * V[d - 2]); const out = []; for (let d = 1; d <= 20; d++) out.push(V[d] / Math.pow(2, d)); return out; })();

  // synthetic 8×8 images: three templates + noise 0.15 (LA.rng(88)); PCA (top two components by power iteration)
  let IMG = null;
  function images() {
    if (IMG) return IMG;
    const rand = LA.rng(88); const X = [], lab = [];
    for (let i = 0; i < 60; i++) {
      const t = i % 3, px = [];
      for (let r = 0; r < 8; r++) for (let c = 0; c < 8; c++) {
        const v = t === 0 ? (r === 3 || r === 4 ? 1 : 0) : t === 1 ? (c === 3 || c === 4 ? 1 : 0) : (r === c ? 1 : 0);
        px.push(LA.clamp(v + 0.15 * LA.rng.gauss(rand), 0, 1));
      }
      X.push(px); lab.push(t);
    }
    const mu = N.mean(X), Xc = X.map(x => N.sub(x, mu));
    const Cv = N.zerosM(64, 64);
    for (const x of Xc) for (let i = 0; i < 64; i++) { const xi = x[i], Ci = Cv[i]; for (let j = 0; j < 64; j++) Ci[j] += xi * x[j]; }
    for (let i = 0; i < 64; i++) for (let j = 0; j < 64; j++) Cv[i][j] /= 59;
    const p1 = N.powerIter(Cv, 200, mu.map((_, i) => 1 + 0.01 * i));
    const C2 = N.subM(Cv, N.scaleM(N.outer(p1.vector, p1.vector), p1.value));
    const p2 = N.powerIter(C2, 200, mu.map((_, i) => 1 - 0.013 * i));
    const pcs = [p1.vector, p2.vector], kept = (p1.value + p2.value) / N.trace(Cv);
    const Z = Xc.map(x => [N.dot(x, pcs[0]), N.dot(x, pcs[1])]);
    // one mosaic image of the 60 thumbnails (6 rows × 10 columns, 1-pixel gaps at value −1 → black)
    const mosaic = []; for (let r = 0; r < 53; r++) mosaic.push(new Array(89).fill(-1));
    X.forEach((px, i) => { const R = Math.floor(i / 10), Cc = i % 10; for (let r = 0; r < 8; r++) for (let c = 0; c < 8; c++) mosaic[R * 9 + r][Cc * 9 + c] = px[r * 8 + c]; });
    IMG = { X, lab, mu, pcs, kept, Z, mosaic };
    return IMG;
  }
  const grid8 = px => { const g = []; for (let r = 0; r < 8; r++) g.push(px.slice(r * 8, r * 8 + 8)); return g; };

  // five 300-component word vectors with planted structure (seeded): cat/kitten share a direction, invoice/tax another
  let WV = null;
  function words() {
    if (WV) return WV;
    const B = LA.rng.tokenVectors(606, 7, 300), A = B[0], Fn = B[1], e = B.slice(2);
    const mk = (len, a, dir, ee) => N.scale(N.normalize(N.add(N.scale(dir, a), N.scale(ee, Math.sqrt(1 - a * a)))), len);
    const V = [mk(2.0, Math.sqrt(0.9), A, e[0]), mk(0.7, Math.sqrt(0.9), A, e[1]), mk(1.3, 0.32, A, e[2]), mk(1.6, Math.sqrt(0.85), Fn, e[3]), mk(0.6, Math.sqrt(0.85), Fn, e[4])];
    const cos = [], dist = []; let bc = [0, 1], bd = [0, 1];
    for (let i = 0; i < 5; i++) { cos.push([]); dist.push([]); for (let j = 0; j < 5; j++) { cos[i].push(N.dot(V[i], V[j]) / (N.norm(V[i]) * N.norm(V[j]))); dist[i].push(N.dist(V[i], V[j])); } }
    for (let i = 0; i < 5; i++) for (let j = i + 1; j < 5; j++) { if (cos[i][j] > cos[bc[0]][bc[1]]) bc = [i, j]; if (dist[i][j] < dist[bd[0]][bd[1]]) bd = [i, j]; }
    WV = { V, cos, dist, norms: V.map(v => N.norm(v)), bestCos: bc, bestDist: bd };
    return WV;
  }

  // ---------------- layout ----------------
  const CAMC = () => ({ center: [5, 3], halfH: 4 });
  const CAMSQ = () => ({ center: [0.5, 0.5], halfH: 0.74 });
  const CAMCUBE = () => Object.assign(LA.cam.defaults3d(), { target: [0.5, 0.5, 0.45], dist: 4.6, yaw: 0.55, pitch: 0.42 });
  const CAMROOM = () => Object.assign(LA.cam.defaults3d(), { target: [0, 0, -0.55], dist: 7, yaw: 0.65, pitch: 0.38 });
  const FM = { x0: 2.3, x1: 8.1, y0: 1.0, y1: 6.2 };
  const SL = { x0: 2.6, x1: 7.9, y: -0.35 };      // the in-canvas d slider
  const mx = (F, xr, v) => F.x0 + (v - xr[0]) / (xr[1] - xr[0]) * (F.x1 - F.x0);
  const my = (F, yr, v) => F.y0 + (v - yr[0]) / (yr[1] - yr[0]) * (F.y1 - F.y0);
  const LX = [Math.log10(2), 3];

  function frame(g, F, o) {
    o = o || {};
    g.line([F.x0, F.y0], [F.x1, F.y0], { color: 'gridAxis', width: 1.2 });
    g.line([F.x0, F.y0], [F.x0, F.y1], { color: 'gridAxis', width: 1.2 });
    (o.xt || []).forEach(([v, lab]) => { const x = mx(F, o.xr, v); g.line([x, F.y0], [x, F.y0 - 0.07], { color: 'gridAxis', width: 1 }); if (lab !== '') g.label(lab, [x, F.y0], { anchor: 'below', dy: 6, size: 11, color: 'tick', font: 'ui', bg: false }); });
    (o.yt || []).forEach(([v, lab]) => { const y = my(F, o.yr, v); if (o.grid) g.line([F.x0, y], [F.x1, y], { color: 'ghost', width: 1 }); g.line([F.x0 - 0.07, y], [F.x0, y], { color: 'gridAxis', width: 1 }); g.label(lab, [F.x0, y], { anchor: 'right', dx: -6, size: 11, color: 'tick', font: 'ui', bg: false }); });
    if (o.title) g.label(o.title, [F.x0, F.y1], { anchor: 'left', dy: -12, size: 12.5, color: o.titleColor || 'hud', font: 'ui', bg: false });
    if (o.right) g.label(o.right, [F.x1, F.y1], { anchor: 'right', dy: -12, size: 12.5, color: o.rightColor || 'gold', font: 'ui', bg: false });
    if (o.xLabel) g.label(o.xLabel, [(F.x0 + F.x1) / 2, F.y0], { anchor: 'below', dy: 22, size: 11.5, color: 'muted', font: 'ui', bg: false });
  }
  /** vertical bars filling frame F; vals mapped with yr; colorOf(i) → role */
  function barsIn(g, F, vals, yr, colorOf, o) {
    o = o || {}; const n = vals.length, w = (F.x1 - F.x0) / n, gap = o.gap === undefined ? 0.12 : o.gap;
    for (let i = 0; i < n; i++) {
      const h = Math.max(0, Math.min(1, (vals[i] - yr[0]) / (yr[1] - yr[0]))) * (F.y1 - F.y0) * (o.grow === undefined ? 1 : o.grow);
      if (h <= 0) continue;
      const x0 = F.x0 + i * w + w * gap / 2, x1 = F.x0 + (i + 1) * w - w * gap / 2;
      g.polygon([[x0, F.y0], [x1, F.y0], [x1, F.y0 + h], [x0, F.y0 + h]], { fill: colorOf(i), alpha: o.alpha === undefined ? 0.85 : o.alpha });
    }
  }
  function honest(g) { g.hud(HONEST, 14, g.size.h - 18, { size: 12.5, color: 'hud', bold: true }); }

  // ---------------- scenes ----------------
  function renderSquareOrCube(g, s, d) {
    const m = measure(s.seed), P = m.pts[d], di = DS.indexOf(d);
    const is3 = d === 3, n = P.length;
    const show = Math.max(1, Math.round(s.appear * n));
    if (is3) g.box(null, { faces: false, stroke: 'grid', width: 1.2 });
    else g.polyline([[0, 0], [1, 0], [1, 1], [0, 1]], { closed: true, color: 'grid', width: 1.4 });
    const pts = []; for (let i = 1; i < show; i++) pts.push(P[i]);
    g.points(pts, { r: is3 ? 3 : 3.4, color: 'data', alpha: 0.75 });
    const probe = P[0], near = P[m.near[di]], far = P[m.far[di]];
    if (s.links > 0) {
      g.line(probe, near, { color: 'hud', width: 1.4, alpha: s.links });
      g.line(probe, far, { color: 'hud', width: 1.4, alpha: s.links });
      g.marker.ring(near, { r: 9, color: 'ok' }); g.marker.ring(far, { r: 9, color: 'bad' });
      if (s.links >= 1) {
        g.dim(probe, near, { label: 'nearest ' + f2(m.d1[di]), color: 'ok', offset: 16 });
        g.dim(probe, far, { label: 'farthest ' + f2(m.d199[di]), color: 'bad', offset: -16 });
      }
    }
    g.point(probe, { r: 6.5, color: 'b', stroke: 'white', label: 'probe', labelColor: 'b', dx: -28, dy: -14 });
    g.hud((is3 ? 'unit cube' : 'unit square') + ' · 200 uniform random points · d = ' + d, 14, 22, { size: 12.5, color: 'muted' });
    if (s.readout > 0) g.hud('nearest / farthest = ' + f2(m.ratio[di]), 14, 46, { size: 17, color: 'gold', bold: true, alpha: s.readout });
  }

  function dSlider(g, s, ctx) {
    const n = DS.length, xs = k => SL.x0 + (SL.x1 - SL.x0) * k / (n - 1);
    g.line([SL.x0, SL.y], [SL.x1, SL.y], { color: 'gridAxis', width: 2 });
    for (let k = 0; k < n; k++) { g.line([xs(k), SL.y - 0.06], [xs(k), SL.y + 0.06], { color: 'gridAxis', width: 1 }); g.label(String(DS[k]), [xs(k), SL.y], { anchor: 'below', dy: 8, size: 10.5, color: k === s.di ? 'gold' : 'tick', font: 'ui', bg: false }); }
    g.label('dimension d', [SL.x0, SL.y], { anchor: 'right', dx: -10, size: 11.5, color: 'muted', font: 'ui', bg: false });
    g.point([xs(s.di), SL.y], { r: 7, color: 'gold', stroke: 'white' });
    g.handle('dslider', [xs(s.di), SL.y], (p, st) => { st.di = LA.clamp(Math.round((p[0] - SL.x0) / (SL.x1 - SL.x0) * (n - 1)), 0, n - 1); }, { color: 'gold', r: 16 });
  }

  function ratioChart(g, s, F, o) {
    o = o || {};
    const st = s.st, k = o.k === undefined ? DS.length : o.k;
    frame(g, F, { xr: LX, yr: [0, 1], xt: [2, 5, 10, 20, 50, 100, 200, 500, 1000].map(d => [Math.log10(d), [2, 10, 100, 1000].indexOf(d) >= 0 ? String(d) : '']), yt: o.compact ? [[0, '0'], [1, '1']] : [[0, '0'], [0.25, '0.25'], [0.5, '0.5'], [0.75, '0.75'], [1, '1']], grid: !o.compact, title: o.title, xLabel: o.compact ? null : 'dimension d (log scale)' });
    const y1 = my(F, [0, 1], 1);
    g.line([F.x0, y1], [F.x1, y1], { color: 'muted', width: 1.2, dashed: true });
    if (!o.compact) g.label('1 = everything equally far', [F.x1, y1], { anchor: 'right', dy: 11, size: 11, color: 'muted', font: 'ui', bg: false });
    const pts = []; for (let i = 0; i < k; i++) pts.push([mx(F, LX, Math.log10(DS[i])), my(F, [0, 1], st.ratio[i])]);
    if (pts.length > 1) g.polyline(pts, { color: 'gold', width: 2.4 });
    pts.forEach((p, i) => g.point(p, { r: o.compact ? 3.5 : 4.5, color: 'gold' }));
    if (k > 0 && !o.compact) { const i = k - 1; g.label('≈ ' + f2(st.ratio[i]), pts[i], { anchor: i >= DS.length - 3 ? 'right' : 'left', dx: i >= DS.length - 3 ? -4 : 9, dy: 18, size: 12.5, color: 'gold', font: 'ui' }); }
    if (o.hi !== undefined) g.marker.ring(pts[o.hi] || pts[0], { r: 10, color: 'b' });
  }

  function sortedPanel(g, s, F, di, grow, o) {
    o = o || {};
    const so = measure(s.seed).sorted[di], top = so[198];
    frame(g, F, { xr: [0, 199], yr: [0, top], xt: o.noX ? [] : [[0, '1'], [99, '100'], [198, '199']], yt: [[0, '0'], [top, f2(top)]], title: 'd = ' + DS[di] + ': distances to the 199 others, sorted', xLabel: o.noX ? null : null });
    barsIn(g, F, so, [0, top], i => i < 10 ? 'gold' : 'data', { gap: 0, grow, alpha: 0.8 });
    if (grow >= 1) g.label('nearest = ' + f2(so[0]) + ' · 10th nearest = ' + f2(so[9]), [F.x0 + 0.05, my(F, [0, top], so[9])], { anchor: 'left', dy: -11, size: 11.5, color: 'gold', font: 'ui' });
  }

  function cosPanel(g, s, F, di, grow, o) {
    o = o || {};
    const c = s.st.counts[di], mxC = Math.max.apply(null, c);
    const shown = grow >= 1 ? c : c.map(x => Math.round(x * grow));
    frame(g, F, { xr: [-1, 1], yr: [0, mxC], xt: [[-1, '−1'], [-0.5, '−0.5'], [0, '0'], [0.5, '0.5'], [1, '1']], yt: o.compact ? [] : [[0, '0'], [mxC, String(mxC)]], title: o.title === undefined ? 'd = ' + DS[di] + ': cosines of random unit pairs' : o.title, right: 'spread ≈ ' + f2(s.st.std[di]) });
    barsIn(g, F, shown, [0, mxC], () => o.color || 'teal', { gap: 0.1 });
    if (grow < 1 && !o.compact) g.label('pairs counted: ' + Math.round(PAIRS * grow) + ' of ' + PAIRS, [F.x1, F.y1], { anchor: 'right', dy: 10, size: 11, color: 'muted', font: 'ui', bg: false });
  }

  function renderRatio(g, s, ctx) {
    ratioChart(g, s, FM, { k: s.k, title: 'nearest / farthest from the probe, 200 uniform points in a d-dimensional unit cube', hi: s.showSlider ? s.di : undefined });
    if (s.showSlider) { dSlider(g, s, ctx); g.hud('d = ' + DS[s.di] + ' · nearest / farthest = ' + f2(s.st.ratio[s.di]), 14, 22, { size: 14, color: 'gold', bold: true }); }
    else g.hud('no picture for d > 3 — only the formula ‖x − y‖', 14, 22, { size: 12.5, color: 'muted' });
    if (!s.showSlider) honest(g);
  }

  function renderSorted(g, s, ctx) {
    const FT = { x0: 2.3, x1: 8.1, y0: 3.85, y1: 6.25 }, FB = { x0: 2.3, x1: 8.1, y0: 0.35, y1: 2.75 };
    sortedPanel(g, s, FT, I1000, s.growTop, { noX: true });
    sortedPanel(g, s, FB, I2, s.growBot, {});
    if (s.cross) {
      g.label('“a clear nearest neighbor”', [FT.x0 + 0.6, 3.35], { anchor: 'left', size: 13.5, color: 'bad', font: 'ui' });
      g.marker.cross([FT.x0 + 0.3, 3.35], { size: 18 });
      g.label('in 1000-D the staircase is flat', [FT.x1, 3.35], { anchor: 'right', size: 11.5, color: 'muted', font: 'ui', bg: false });
    }
    g.hud('each panel is scaled to its own farthest point', 14, 22, { size: 12.5, color: 'muted' });
    honest(g);
  }

  function renderCos(g, s, ctx) {
    if (s.scene === 'cos') {
      cosPanel(g, s, { x0: 2.3, x1: 8.1, y0: 3.95, y1: 6.25 }, I3, 1, { color: 'w' });
      cosPanel(g, s, { x0: 2.3, x1: 8.1, y0: 0.45, y1: 2.75 }, I300, s.growCos, {});
      g.hud('the spread shrinks like 1/√d: 1/√3 ≈ 0.58, 1/√300 ≈ 0.06', 14, 22, { size: 12.5, color: 'muted' });
      honest(g);
      return;
    }
    // playground / check: the current d
    cosPanel(g, s, FM, s.di, 1, {});
    dSlider(g, s, ctx);
    g.hud('d = ' + DS[s.di] + ' · spread of the cosines ≈ ' + f2(s.st.std[s.di]), 14, 22, { size: 14, color: 'gold', bold: true });
  }

  function renderRoom(g, s, ctx) {
    const u = s.roomT;
    const len = 1.0 * seg(u, 0, 0.4, 'back');
    if (len > 0.01) {
      g.vector([0, 0, 0], [len, 0, 0], { color: 'i', label: 'i', hat: true, width: 3 });
      g.vector([0, 0, 0], [0, len, 0], { color: 'j', label: 'j', hat: true, width: 3 });
      g.vector([0, 0, 0], [0, 0, len], { color: 'k', label: 'k', hat: true, width: 3 });
      if (u > 0.4) { const q = 0.22; [[[q, 0, 0], [q, q, 0], [0, q, 0]], [[0, q, 0], [0, q, q], [0, 0, q]], [[q, 0, 0], [q, 0, q], [0, 0, q]]].forEach(pl => g.polyline(pl, { color: 'muted', width: 1.2 })); }
    }
    g.hud('in 3D: at most three exactly perpendicular directions', 14, 22, { size: 12.5, color: 'muted' });
    const a = seg(u, 0.35, 0.8);
    if (a > 0) {
      const m = measure(s.seed);
      g.viewport({ x: 0.04, y: 0.5, w: 0.92, h: 0.44 }, { view: '2d', cam: { center: [0, 0], halfH: 1.2 }, bg: 'bg', border: true }, (gg, r) => {
        const half = 0.82, B = { x0: -half, x1: half, y0: -half - 0.08, y1: half - 0.08 };
        gg.heat2d(m.room, B, { range: [0, 1], ramp: t => C.mix(C.panel2, C.gold, t), alpha: a, smooth: false });
        gg.polyline([[B.x0, B.y0], [B.x1, B.y0], [B.x1, B.y1], [B.x0, B.y1]], { closed: true, color: 'line', width: 1 });
        gg.hud('|cos| between 16 random directions in ℝ⁷⁶⁸ (diagonal: each with itself = 1)', r.x + 10, r.y + 16, { size: 12, color: 'gold', alpha: a });
        gg.hud('largest off-diagonal |cos| = ' + f2(s.st.roomMax) + ' · 16 of thousands', r.x + 10, r.y + r.h - 14, { size: 12, color: 'hud', alpha: a });
      });
      g.hud(HONEST, 14, g.size.h - 16, { size: 12, color: 'hud', bold: true, alpha: a });
    }
  }

  function renderCorner(g, s, ctx) {
    const u1 = seg(s.cornerT, 0, 0.3), u2 = seg(s.cornerT, 0.25, 0.5), u3 = seg(s.cornerT, 0.4, 0.95, 'linear');
    // square with its inscribed circle (or, in check 3, the typed fraction as a disc)
    const S0 = s.guessMode ? [3.0, 1.2] : [2.3, 4.0], side = s.guessMode ? 4 : 2.4, cc = [S0[0] + side / 2, S0[1] + side / 2];
    g.polygon([[S0[0], S0[1]], [S0[0] + side, S0[1]], [S0[0] + side, S0[1] + side], [S0[0], S0[1] + side]], { fill: 'panel2', alpha: 0.6, stroke: 'gridAxis', width: 1.2 });
    if (s.guessMode) {
      const gv = LA.clamp(s.guess[0], 0, 1), rr = Math.sqrt(gv * side * side / Math.PI);
      g.circle(cc, side / 2, { stroke: 'teal', width: 1.4, dashed: true });
      if (rr > 0.005) g.circle(cc, rr, { fill: 'teal', alpha: 0.45, stroke: 'teal', width: 1.2 });
      g.label('typed fraction: ' + f2(gv) + ' of the square', [S0[0], S0[1]], { anchor: 'left', dy: 16, size: 12, color: 'teal', font: 'ui' });
      g.hud('the teal disc covers exactly the fraction you type; make it fill the dashed circle', 14, 22, { size: 12, color: 'muted' });
      return;
    }
    if (u1 > 0) {
      g.circle(cc, side / 2, { fill: 'teal', alpha: 0.4 * u1, stroke: 'teal', width: 1.4, strokeAlpha: u1 });
      g.label('circle / square = ' + pct(FRACS[1]), [cc[0], S0[1]], { anchor: 'below', dy: 6, size: 12, color: 'teal', font: 'ui', alpha: u1 });
    }
    if (u2 > 0) {
      g.viewport({ x: 0.52, y: 0.03, w: 0.44, h: 0.38 }, { view: '3d', cam: { target: [0.5, 0.5, 0.5], dist: 4.4, yaw: 0.6 + 0.4 * s.cornerT, pitch: 0.4, fov: 36 }, border: true }, (gg, r) => {
        gg.box(null, { faces: false, stroke: 'gridAxis', width: 1, strokeAlpha: u2 });
        gg.sphere([0.5, 0.5, 0.5], 0.5, { color: 'teal', fill: true, fillAlpha: 0.25 * u2 });
        gg.hud('ball / cube = ' + pct(FRACS[2]), r.x + 10, r.y + 16, { size: 12, color: 'teal', alpha: u2 });
      });
    }
    const F = { x0: 2.3, x1: 8.1, y0: 0.25, y1: 3.05 };
    const lv = FRACS.map(f => Math.log10(f) + 8);
    frame(g, F, { xr: [0.5, 20.5], yr: [0, 8], xt: [1, 5, 10, 15, 20].map(d => [d, String(d)]), yt: [[8, '1'], [6, pow10(-2)], [4, pow10(-4)], [2, pow10(-6)], [0, pow10(-8)]], grid: true, title: 'fraction of the hypercube inside its ball, d = 1 … 20 (log scale)', xLabel: null });
    const nShow = Math.round(u3 * 20);
    const hiD = s.scene === 'play' && DS[s.di] <= 20 ? DS[s.di] : 0;
    barsIn(g, F, lv.map((v, i) => i < nShow ? v : 0), [0, 8], i => (hiD && i === hiD - 1) ? 'gold' : 'teal', { gap: 0.25 });
    if (nShow >= 10) g.label('d = 10: ' + LA.fmt.fixed(FRACS[9], 4), [mx(F, [0.5, 20.5], 10), my(F, [0, 8], lv[9])], { anchor: 'above', dy: -6, size: 11, color: 'gold', font: 'ui' });
    if (nShow >= 20) g.label('d = 20: ' + LA.fmt.fixed(FRACS[19] * 1e8, 1) + '·' + pow10(-8), [mx(F, [0.5, 20.5], 20), my(F, [0, 8], lv[19])], { anchor: 'right', dx: 6, dy: -24, size: 11, color: 'gold', font: 'ui' });
  }

  function renderImages(g, s, ctx, compact) {
    const D = images(), u = compact ? 1 : seg(s.imgT, 0.3, 0.8);
    const FS = compact ? { x0: 2.3, x1: 8.1, y0: 0.3, y1: 6.0 } : { x0: 2.3, x1: 8.1, y0: -0.35, y1: 3.15 };
    const zx = D.Z.map(z => z[0]), zy = D.Z.map(z => z[1]);
    const xr = [Math.min.apply(null, zx) - 0.3, Math.max.apply(null, zx) + 0.3], yr = [Math.min.apply(null, zy) - 0.3, Math.max.apply(null, zy) + 0.3];
    const to = i => [mx(FS, xr, D.Z[i][0]), my(FS, yr, D.Z[i][1])];
    const MOS = { x0: 2.25, x1: 8.0, y0: 3.35, y1: 3.35 + 5.75 * 53 / 89 };
    const cellC = i => { const R = Math.floor(i / 10), Cc = i % 10; return [MOS.x0 + (Cc * 9 + 4) / 89 * (MOS.x1 - MOS.x0), MOS.y1 - (R * 9 + 4) / 53 * (MOS.y1 - MOS.y0)]; };
    const gridA = compact ? 0 : 1 - seg(s.imgT, 0.3, 0.45);
    if (gridA > 0) {
      g.image(D.mosaic, MOS, { range: [0, 1], alpha: gridA });
      g.label('60 synthetic 8×8 images: three patterns + noise', [MOS.x0, MOS.y1], { anchor: 'left', dy: -11, size: 12, color: 'muted', font: 'ui', bg: false, alpha: gridA });
    }
    if (u > 0 || compact) {
      if (u >= 0.999 || compact) {
        frame(g, FS, { xr, yr, title: compact ? null : 'principal component shadow of ℝ⁶⁴ · ' + pct(D.kept) + ' of variance kept', titleColor: 'gold' });
        if (!compact) {
          g.label('PC₁', [FS.x1, FS.y0], { anchor: 'right', dy: -10, size: 11.5, color: 'muted', font: 'ui', bg: false });
          g.label('PC₂', [FS.x0, FS.y1], { anchor: 'left', dx: 6, dy: 10, size: 11.5, color: 'muted', font: 'ui', bg: false });
        }
      }
      for (let i = 0; i < 60; i++) {
        const p = compact ? to(i) : lerpV(cellC(i), to(i), seg(u, (i % 10) * 0.03, 0.7 + (i % 10) * 0.03));
        g.point(p, { r: compact ? 3.5 : 4.5, color: TPL_COL[D.lab[i]], alpha: 0.9 });
      }
      if (u >= 0.999 && !compact) TPL.forEach((t, k) => g.label('● ' + t, [FS.x1, FS.y0 + 0.9 - k * 0.28], { anchor: 'right', dx: -4, size: 11.5, color: TPL_COL[k], font: 'ui', bg: false }));
    }
    const rA = compact ? 0 : seg(s.imgT, 0.82, 1);
    if (rA > 0) {
      const i = Math.round(LA.clamp(s.recon, 0, 59)), z = D.Z[i];
      const rec = D.mu.map((m, k) => LA.clamp(m + z[0] * D.pcs[0][k] + z[1] * D.pcs[1][k], 0, 1));
      const A = { x0: 2.6, x1: 4.4, y0: 4.1, y1: 5.9 }, Bx = { x0: 5.8, x1: 7.6, y0: 4.1, y1: 5.9 };
      g.image(grid8(D.X[i]), A, { range: [0, 1], alpha: rA }); g.polyline([[A.x0, A.y0], [A.x1, A.y0], [A.x1, A.y1], [A.x0, A.y1]], { closed: true, color: 'line', width: 1, alpha: rA });
      g.image(grid8(rec), Bx, { range: [0, 1], alpha: rA, smooth: false }); g.polyline([[Bx.x0, Bx.y0], [Bx.x1, Bx.y0], [Bx.x1, Bx.y1], [Bx.x0, Bx.y1]], { closed: true, color: 'gold', width: 1, alpha: rA });
      g.label('image #' + (i + 1) + ': 64 numbers', [(A.x0 + A.x1) / 2, A.y1], { anchor: 'above', dy: -6, size: 12, color: 'hud', font: 'ui', alpha: rA });
      g.label('reconstruction from 2 numbers', [(Bx.x0 + Bx.x1) / 2, Bx.y1], { anchor: 'above', dy: -6, size: 12, color: 'gold', font: 'ui', alpha: rA });
      g.vector([4.6, 5.0], [5.6, 5.0], { color: 'gold', width: 1.6, head: 9, alpha: rA, label: 'z', labelAt: 0.5 });
      g.label('z = (' + f2(z[0]) + ', ' + f2(z[1]) + ')', [5.1, 5.0], { anchor: 'below', dy: 10, size: 11, color: 'gold', font: 'ui', alpha: rA });
      g.marker.ring(to(i), { r: 9, color: 'gold' });
      g.handle('recon', to(i), (p, st) => { let bi = 0, bd = Infinity; for (let k = 0; k < 60; k++) { const q = to(k), dd = Math.hypot(q[0] - p[0], q[1] - p[1]); if (dd < bd) { bd = dd; bi = k; } } st.recon = bi; }, { color: 'gold', r: 12 });
    }
    if (!compact) honest(g);
  }

  function renderWords(g, s, ctx) {
    const W = words(), u1 = seg(s.wordsT, 0, 0.35), u2 = seg(s.wordsT, 0.35, 0.7), u3 = seg(s.wordsT, 0.75, 0.95);
    // strips
    WORDS.forEach((w, i) => {
      const y0 = 6.25 - i * 0.42, B = { x0: 3.05, x1: 8.1, y0, y1: y0 + 0.3 };
      g.heat2d([W.V[i]], B, { range: [-0.25, 0.25], ramp: C.diverge, alpha: 0.95 * u1, smooth: false });
      g.polyline([[B.x0, B.y0], [B.x1, B.y0], [B.x1, B.y1], [B.x0, B.y1]], { closed: true, color: 'line', width: 1, alpha: u1 });
      g.label(w + '  ‖·‖ = ' + f2(W.norms[i]), [B.x0, (B.y0 + B.y1) / 2], { anchor: 'right', dx: -6, size: 11.5, color: 'hud', font: 'ui', bg: false, alpha: u1 });
    });
    g.label('five synthetic 300-component word vectors', [3.05, 6.55], { anchor: 'left', dy: -8, size: 12, color: 'muted', font: 'ui', bg: false, alpha: u1 });
    // heatmaps
    const cw = 0.48, maps = [{ x0: 2.75, M: W.dist, title: 'L2 distance ‖x − y‖', best: W.bestDist, range: [0, 2.6], ramp: t => C.heat(1 - t) }, { x0: 5.7, M: W.cos, title: 'cosine similarity', best: W.bestCos, range: [-0.1, 1], ramp: C.heat }];
    maps.forEach((mp, mi) => {
      const a = u2; if (a <= 0) return;
      const B = { x0: mp.x0, x1: mp.x0 + 5 * cw, y0: 0.75, y1: 0.75 + 5 * cw };
      g.heat2d(mp.M, B, { range: mp.range, ramp: mp.ramp, alpha: 0.9 * a, smooth: false });
      for (let i = 0; i < 5; i++) for (let j = 0; j < 5; j++) g.label(f2(mp.M[i][j]), [B.x0 + (j + 0.5) * cw, B.y1 - (i + 0.5) * cw], { size: 10, color: 'text', font: 'mono', alpha: a });
      for (let j = 0; j < 5; j++) g.label(WABBR[j], [B.x0 + (j + 0.5) * cw, B.y1], { anchor: 'above', dy: -3, size: 10, color: 'muted', font: 'ui', bg: false, alpha: a });
      if (mi === 0) for (let i = 0; i < 5; i++) g.label(WORDS[i], [B.x0, B.y1 - (i + 0.5) * cw], { anchor: 'right', dx: -5, size: 10.5, color: 'muted', font: 'ui', bg: false, alpha: a });
      g.label(mp.title, [B.x0, B.y1], { anchor: 'left', dy: -24, size: 12, color: 'hud', font: 'ui', bg: false, alpha: a });
      if (u3 > 0) {
        const [i, j] = mp.best;
        [[i, j], [j, i]].forEach(([r, c]) => { const x0 = B.x0 + c * cw, y1 = B.y1 - r * cw; g.polyline([[x0, y1 - cw], [x0 + cw, y1 - cw], [x0 + cw, y1], [x0, y1]], { closed: true, color: 'gold', width: 2.5, alpha: u3 }); });
        g.label((mi === 0 ? 'closest: ' : 'most similar: ') + WORDS[i] + ' & ' + WORDS[j], [B.x0, B.y0], { anchor: 'left', dy: 14, size: 12, color: 'gold', font: 'ui', alpha: u3 });
      }
    });
    if (u3 > 0) g.hud('Part 6: embeddings — retrieval by cosine similarity', 14, g.size.h - 42, { size: 12.5, color: 'gold', alpha: u3 });
    honest(g);
  }

  function renderRecap(g, s, ctx) {
    const card = (k, title, fn) => g.viewport({ x: 0, y: 0.02 + k * 0.327, w: 1, h: 0.31 }, { view: '2d', cam: { center: [5, 3.1], halfH: 4.1 } }, (gg, r) => { fn(gg); gg.hud(title, r.x + 10, r.y + 16, { size: 12.5, color: 'gold' }); });
    card(0, 'distances concentrate: nearest / farthest → 1', gg => ratioChart(gg, s, { x0: 2.6, x1: 8.0, y0: 0.6, y1: 5.6 }, { compact: true }));
    card(1, 'random directions are nearly perpendicular: spread ≈ 1/√d', gg => {
      const F = { x0: 2.6, x1: 8.0, y0: 0.6, y1: 5.6 }, c3 = s.st.counts[I3], c300 = s.st.counts[I300], m = Math.max.apply(null, c300);
      frame(gg, F, { xr: [-1, 1], yr: [0, m], xt: [[-1, '−1'], [0, '0'], [1, '1']] });
      barsIn(gg, F, c3, [0, m], () => 'w', { gap: 0.1, alpha: 0.6 });
      barsIn(gg, F, c300, [0, m], () => 'teal', { gap: 0.1, alpha: 0.85 });
      gg.label('d = 300', [mx(F, [-1, 1], 0.12), my(F, [0, m], m * 0.8)], { anchor: 'left', size: 11, color: 'teal', font: 'ui' });
      gg.label('d = 3', [mx(F, [-1, 1], -0.95), my(F, [0, m], Math.max.apply(null, c3))], { anchor: 'left', dy: -12, size: 11, color: 'w', font: 'ui' });
    });
    card(2, 'images are points; label the shadow: ' + pct(images().kept) + ' of variance kept', gg => renderImages(gg, s, ctx, true));
  }

  function renderPlay(g, s, ctx) {
    if (s.chart === 'cos') return renderCos(g, s, ctx);
    if (s.chart === 'corner') return renderCorner(g, s, ctx);
    if (s.chart === 'images') return renderImages(g, s, ctx, false);
    // ratio: curve + sorted distances for the current d
    ratioChart(g, s, { x0: 2.3, x1: 8.1, y0: 3.7, y1: 6.3 }, { compact: true, title: 'nearest / farthest vs d', hi: s.di });
    sortedPanel(g, s, { x0: 2.3, x1: 8.1, y0: 0.55, y1: 2.8 }, s.di, 1, { noX: true });
    dSlider(g, s, ctx);
    g.hud('d = ' + DS[s.di] + ' · nearest / farthest = ' + f2(s.st.ratio[s.di]), 14, 22, { size: 14, color: 'gold', bold: true });
  }

  // ---------------- state helpers ----------------
  function base(s) {
    s.scene = 'square'; s.view = '2d'; s.cam = CAMSQ();
    s.appear = 1; s.links = 1; s.readout = 1; s.k = DS.length; s.showSlider = false;
    s.growTop = 1; s.growBot = 1; s.cross = false; s.growCos = 1; s.roomT = 1; s.cornerT = 1; s.hiD = 0;
    s.imgT = 1; s.wordsT = 1; s.guessMode = false;
  }
  const sceneCam = sc => sc === 'square' ? CAMSQ() : sc === 'cube' ? CAMCUBE() : sc === 'room' ? CAMROOM() : CAMC();
  function goScene(s, sc) { s.scene = sc; s.view = (sc === 'cube' || sc === 'room') ? '3d' : '2d'; s.cam = sceneCam(sc); }

  LA.registerLesson({
    id: 'p1-08-high-dimensions-measured',
    part: 1,
    title: 'High dimensions: what we can’t draw, we measure',
    subtitle: 'Distances concentrate, random directions are nearly perpendicular, volume hides in corners.',
    minutes: 10,
    prereqs: ['p1-05-norms-distance', 'p1-06-dot-product', 'p1-07-data-as-vectors'],
    tags: ['core'],
    badge: { drawnIn: '2D/3D samples plus charts for d up to 1000', realSize: 'embeddings d = 768–4096; images d = 784–150 528', transfers: 'everything here is computed with the strip formulas from p1-05 and p1-06; the charts are measurements, not drawings, and they hold for any d' },
    cantShow: ['a cloud in ℝ¹⁰⁰⁰ (we show its distance and cosine statistics instead)', 'a 64-D image cloud (we show a labeled 2D shadow with its variance kept and a reconstruction)'],
    terms: [
      { term: 'distance concentration', def: 'In high dimensions the nearest and farthest points from any point are almost equally far.' },
      { term: 'near-orthogonality', def: 'Random directions in high dimensions have cosine close to 0: almost perpendicular.' },
      { term: 'hypercube', def: 'The cube with 2ᵈ corners in d dimensions.' },
      { term: 'inscribed sphere', def: 'The largest ball that fits inside a square, cube or hypercube.' },
      { term: 'principal component shadow', def: 'The 2D projection of a cloud onto its two directions of greatest spread, labeled with the fraction of variance kept.' },
    ],
    atlas: ['knn', 'embeddings', 'pca'],
    thumbnail: 4,

    state() {
      const seed = 8;
      const s = { view: '2d', cam: CAMSQ(), seed, st: compact(measure(seed)), di: I3, chart: 'ratio', recon: 0, guess: [0] };
      images(); words();
      base(s);
      return s;
    },

    render(g, s, ctx) {
      switch (s.scene) {
        case 'square': return renderSquareOrCube(g, s, 2);
        case 'cube': return renderSquareOrCube(g, s, 3);
        case 'ratio': return renderRatio(g, s, ctx);
        case 'sorted': return renderSorted(g, s, ctx);
        case 'cos': case 'cosd': return renderCos(g, s, ctx);
        case 'room': return renderRoom(g, s, ctx);
        case 'corner': return renderCorner(g, s, ctx);
        case 'images': return renderImages(g, s, ctx, false);
        case 'words': return renderWords(g, s, ctx);
        case 'recap': return renderRecap(g, s, ctx);
        default: return renderPlay(g, s, ctx);
      }
    },

    math(s, tex) {
      const st = s.st, eq = tex.eq, T = tex.text, n = (x, c) => tex.num(x, { color: c || 'gold' });
      switch (s.scene) {
        case 'square': case 'cube': { const di = s.scene === 'cube' ? I3 : I2; return tex.block(eq(T('min dist / max dist ='), n(st.ratio[di]), T('(d = ' + DS[di] + ')'))); }
        case 'ratio': { const di = s.showSlider ? s.di : Math.max(0, s.k - 1); return tex.block(eq(T('ratio = nearest ‖x₀ − xⱼ‖ / farthest ‖x₀ − xⱼ‖  (x₀ = the probe)')), eq(T('d = ' + DS[di] + ':'), T('ratio ='), n(st.ratio[di]))); }
        case 'sorted': return tex.block(eq(T('d = 1000:'), T('nearest ='), n(st.d1[I1000]), T('· 10th ='), n(st.d10[I1000]), T('· farthest ='), n(st.d199[I1000])), eq(T('d = 2:'), T('nearest ='), n(st.d1[I2], 'text'), T('· 10th ='), n(st.d10[I2], 'text'), T('· farthest ='), n(st.d199[I2], 'text')));
        case 'cos': return tex.block(eq(T('cos θ = u·v for random unit u, v')), eq(T('spread (typical size of a cosine) ≈ 1/√d:'), T('d = 3 →'), n(st.std[I3]), T('· d = 300 →'), n(st.std[I300])));
        case 'room': return tex.block(eq(tex.abs(T('u·v')), T('≤'), n(st.roomMax), T('for all pairs (d = 768, 16 vectors)')));
        case 'corner': return tex.block(eq(T('fraction(d) = volume of the ball / volume of the cube, d = 1, 2, 3, 4, 5:')), eq(n(1, 'teal'), T(','), n(FRACS[1], 'teal'), T(','), n(FRACS[2], 'teal'), T(','), n(FRACS[3], 'teal'), T(','), n(FRACS[4], 'teal'), T(', …,'), T(LA.fmt.fixed(FRACS[9] * 1000, 1) + '·10⁻³ (d = 10)', { color: 'teal' })));
        case 'images': { const D = images(); return tex.block(eq(T('X: 60 rows × 64 pixels · PC₁, PC₂ = the two directions of greatest spread')), eq(T('variance (spread) kept by the shadow ='), n(D.kept))); }
        case 'words': { const W = words(), bd = W.bestDist; return tex.block(eq(T('cos(cat, kitten) ='), n(W.cos[0][1])), eq(tex.norm(T('cat − kitten')), T('='), n(W.dist[0][1], 'text'), T('·'), tex.norm(T(WORDS[bd[0]] + ' − ' + WORDS[bd[1]])), T('='), n(W.dist[bd[0]][bd[1]], 'text'))); }
        case 'recap': return tex.block(eq(T('measured, not drawn: ratio → 1 · spread ≈ 1/√d · shadow labeled with % variance')));
        default:
          if (s.chart === 'cos' || s.scene === 'cosd') return tex.block(eq(T('d = ' + DS[s.di] + ':'), T('spread of cos ='), n(st.std[s.di]), T('· 1/√d ='), n(1 / Math.sqrt(DS[s.di]), 'text')));
          if (s.chart === 'corner' && s.guessMode) return tex.block(eq(T('typed fraction of the square ='), n(s.guess[0], 'teal')));
          if (s.chart === 'corner') return tex.block(eq(T('fraction(2) = π/4 ='), n(FRACS[1], 'teal'), T('· fraction(10) ='), T(LA.fmt.fixed(FRACS[9], 4), { color: 'teal' })));
          if (s.chart === 'images') { const D = images(); return tex.block(eq(T('variance kept ='), n(D.kept))); }
          return tex.block(eq(T('d = ' + DS[s.di] + ':'), T('nearest / farthest ='), n(st.ratio[s.di])));
      }
    },

    steps: [
      { title: 'Nearest and farthest in a square', duration: 4,
        narration: (s) => 'Two hundred random points in a square. Pick one, the orange probe, and measure its distance to every other: the nearest is ' + f2(s.st.d1[I2]) + ' away, the farthest ' + f2(s.st.d199[I2]) + '. Nearest divided by farthest is about ' + f2(s.st.ratio[I2]) + ' — some points are very close, most are far. That is how distance behaves in a picture.',
        setup(s) { base(s); goScene(s, 'square'); s.appear = 0; s.links = 0; s.readout = 0; },
        at(s, t) { s.appear = seg(t, 0, 0.4, 'linear'); s.links = seg(t, 0.45, 0.65); s.readout = seg(t, 0.75, 0.9); } },

      { title: 'The same in a cube', duration: 4,
        narration: (s) => 'In a cube with the same two hundred points the ratio climbs a little, to about ' + f2(s.st.ratio[I3]) + ': with more room, the nearest neighbor is not quite so near. Drag to orbit the cube. We can still see this one. For the next step we cannot — so we compute.',
        setup(s) { base(s); goScene(s, 'cube'); },
        at(s, t) { LA.cam.tween(s, { yaw: 0.55 }, { yaw: 1.25 }, seg(t, 0, 1, 'linear')); } },

      { title: 'Distances concentrate', duration: 5,
        narration: (s) => 'Repeat the measurement in 5, 10, 100, 1000 dimensions, with no picture, only the formula ‖x − y‖. The ratio nearest/farthest rises toward 1: in a thousand dimensions the closest point is only about ' + Math.round(100 * (1 - s.st.ratio[I1000])) + '% closer than the farthest. Everything is roughly equally far from everything. This is <b>distance concentration</b>.',
        setup(s) { base(s); goScene(s, 'ratio'); s.k = 1; },
        at(s, t) { s.k = Math.max(1, Math.min(DS.length, Math.ceil(seg(t, 0, 0.8, 'linear') * DS.length))); } },

      { title: '“Nearest neighbor” loses its meaning', duration: 4, misconception: true,
        narration: 'Belief: “k-nearest-neighbors works the same in 1000 dimensions as in 2.” Here are the probe’s distances to all 199 others at d = 1000, sorted: a nearly flat staircase. The nearest and the tenth-nearest differ by a hair; one noisy feature reorders them. Plain distance stops being informative.',
        setup(s) { base(s); goScene(s, 'sorted'); s.growTop = 0; s.growBot = 0; s.cross = false; },
        at(s, t) { s.growBot = seg(t, 0, 0.35); s.growTop = seg(t, 0.3, 0.7); s.cross = t > 0.75; } },

      { title: 'Random directions are nearly perpendicular', duration: 5,
        narration: (s) => 'Take two random unit vectors and compute their cosine. In 3D the cosines spread across −1 to 1. In 300 dimensions they pile up at 0: almost every pair is nearly perpendicular. This is <b>near-orthogonality</b>; the spread shrinks like 1/√d, from ' + f2(s.st.std[I3]) + ' to ' + f2(s.st.std[I300]) + '. There is far more room for different directions than in any picture.',
        setup(s) { base(s); goScene(s, 'cos'); s.growCos = 0; },
        at(s, t) { s.growCos = seg(t, 0.05, 0.8, 'linear'); } },

      { title: 'Room for many concepts', duration: 4,
        narration: 'Near-orthogonality is why embeddings work. In 768 dimensions you can place thousands of concept directions that barely interfere, each almost perpendicular to the rest, while a dot product still picks out the one you ask for. In 3D you could fit only three: the green î, the red ĵ and the blue k̂.',
        setup(s) { base(s); goScene(s, 'room'); s.roomT = 0; },
        at(s, t) { s.roomT = seg(t, 0, 0.9, 'linear'); LA.cam.tween(s, { yaw: 0.65 }, { yaw: 1.05 }, seg(t, 0, 1, 'linear')); } },

      { title: 'Volume hides in the corners', duration: 5,
        narration: 'The <b>inscribed sphere</b> of a square — its circle — covers 79% of it. In a cube, 52%. In 10 dimensions the ball fills a quarter of a percent of the cube; by 20, almost nothing. Uniform data in a <b>hypercube</b> lives in its corners, far from the center.',
        setup(s) { base(s); goScene(s, 'corner'); s.cornerT = 0; },
        at(s, t) { s.cornerT = seg(t, 0, 1, 'linear'); } },

      { title: 'Images are points: a shadow that clusters', duration: 6,
        narration: () => 'Sixty synthetic 8×8 images — three patterns with noise — are sixty points in ℝ⁶⁴. We cannot draw them, but we can compute their two directions of greatest spread and project: this <b>principal component shadow</b> keeps ' + pct(images().kept) + ' of the variance (the spread), and the three patterns fall into three clusters. The picture stops at 3. The formula does not.',
        setup(s) { base(s); goScene(s, 'images'); s.imgT = 0; s.recon = 0; },
        at(s, t) { s.imgT = seg(t, 0, 1, 'linear'); } },

      { title: 'Why embedding search uses cosine', duration: 5, mlTie: true,
        narration: () => { const W = words(), b = W.bestDist; return 'Five 300-component word vectors of different lengths. By plain L2 distance the closest pair is ' + WORDS[b[0]] + ' and ' + WORDS[b[1]] + ' — two short vectors with nothing in common. By cosine similarity, cat and kitten win. Distance is swayed by length; cosine compares directions, which in high dimensions are plentiful and meaningful. Part 6 builds retrieval on it.'; },
        setup(s) { base(s); goScene(s, 'words'); s.wordsT = 0; },
        at(s, t) { s.wordsT = seg(t, 0, 1, 'linear'); } },

      { title: 'Recap', duration: 3, recap: true,
        narration: 'High dimensions are measured, not drawn. Distances concentrate, so nearest neighbors blur. Random directions are nearly perpendicular, so there is room for many concepts and cosine stays meaningful. Volume hides in corners. Images are points, and their shadows, honestly labeled, still cluster.',
        cards: ['<b>Distances concentrate.</b> Nearest / farthest climbs toward 1 as d grows; plain nearest neighbors blur.', '<b>Random directions are nearly perpendicular.</b> The cosine spread shrinks like 1/√d: room for thousands of concepts.', '<b>Images are points; label the shadow.</b> A 2D principal component shadow, with its % of variance kept, still shows the clusters.'],
        setup(s) { base(s); goScene(s, 'recap'); },
        at(s, t) { } },
    ],

    playground: {
      intro: '<p>Drag the gold dot on the <b>dimension</b> track (or use the slider) and watch the measurements change. Switch charts, resample the random points, or pick an image to reconstruct from two numbers.</p>',
      setup(s) { base(s); s.scene = 'play'; s.view = '2d'; s.cam = CAMC(); s.chart = 'ratio'; s.di = I3; },
      controls: [
        { type: 'slider', bind: 'di', min: 0, max: DS.length - 1, step: 1, label: 'dimension d', color: 'gold', fmt: x => 'd = ' + DS[Math.round(x)] },
        { type: 'select', bind: 'chart', label: 'Chart', options: [{ label: 'distance ratio', value: 'ratio' }, { label: 'cosine histogram', value: 'cos' }, { label: 'corner fraction', value: 'corner' }, { label: 'image shadow', value: 'images' }] },
        { type: 'slider', bind: 'recon', min: 0, max: 59, step: 1, label: 'Image to reconstruct', color: 'gold', fmt: x => '#' + (Math.round(x) + 1) },
        { type: 'button', label: 'Resample (new seed)', onClick(s) { s.seed = s.seed + 1; s.st = compact(measure(s.seed)); } },
        { type: 'preset', label: 'Presets', items: [
          { label: 'd = 3', apply(s) { s.scene = 'play'; s.view = '2d'; s.cam = CAMC(); s.chart = 'cos'; s.di = I3; } },
          { label: 'd = 300', apply(s) { s.scene = 'play'; s.view = '2d'; s.cam = CAMC(); s.chart = 'cos'; s.di = I300; } },
          { label: 'Images', apply(s) { s.scene = 'play'; s.view = '2d'; s.cam = CAMC(); s.chart = 'images'; s.imgT = 1; } },
        ] },
      ],
    },

    checks: [
      { kind: 'manipulate',
        prompt: 'Drag the gold dot along the dimension track until the spread of the cosine histogram (read out at the top) falls below 0.1.',
        setup(s) { base(s); s.scene = 'play'; s.view = '2d'; s.cam = CAMC(); s.chart = 'cos'; s.di = I3; },
        judge(s) { const v = s.st.std[s.di]; return { ok: v < 0.1, progress: LA.clamp01(1 - (v - 0.1) / 0.5) }; },
        hint: 'The spread shrinks like 1/√d; you need d above 100.',
        solution(s) { s.di = DS.indexOf(200); } },
      { kind: 'manipulate',
        prompt: 'Drag the gold dot along the dimension track until the nearest/farthest ratio exceeds 0.8.',
        setup(s) { base(s); s.scene = 'play'; s.view = '2d'; s.cam = CAMC(); s.chart = 'ratio'; s.di = I2; },
        judge(s) { const v = s.st.ratio[s.di]; return { ok: v > 0.8, progress: LA.clamp01(v / 0.8) * 0.95 }; },
        hint: 'Distances concentrate slowly; go to several hundred dimensions.',
        solution(s) { let k = s.st.ratio.findIndex(v => v > 0.8); s.di = k < 0 ? DS.length - 1 : k; } },
      { kind: 'entry',
        prompt: 'Type the fraction of a square covered by its inscribed circle (two decimals).',
        bind: 'guess', answer: [0.79], tol: LA.TOL.entry, step: 0.01, label: 'fraction', colors: 'teal',
        setup(s) { base(s); s.scene = 'play'; s.view = '2d'; s.cam = CAMC(); s.chart = 'corner'; s.guessMode = true; s.guess = [0]; },
        hint: 'π/4.' },
    ],
  });
})();
