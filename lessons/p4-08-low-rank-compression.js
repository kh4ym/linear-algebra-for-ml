/* p4-08 — Low-rank approximation, SVD image compression, recommenders / LoRA (capstone). */
(function () {
  'use strict';
  const N = LA.num, seg = LA.seg, lerp = LA.lerp;
  const n = 32, PR = 17, PC = 20;
  const gold = x => '<span class="c-gold">' + x + '</span>';
  const U6 = [[1.5, 0.5], [1.2, -0.8], [-1.0, 1.0], [-1.3, -0.6], [0.2, 1.4], [0.8, 0.1]];
  const V8 = [[1, 0], [0.8, 0.6], [0, 1], [-1, 0.2], [-0.6, -0.8], [0.5, -1], [0.7, 0.7], [-0.3, 1]];
  const HIDDEN = [[1, 3], [1, 6], [1, 8], [2, 2], [2, 8], [3, 1], [3, 5], [4, 1], [4, 4], [4, 7], [5, 2], [5, 6], [6, 3], [6, 5]].map(p => [p[0] - 1, p[1] - 1]);
  const isHidden = (i, j) => HIDDEN.some(p => p[0] === i && p[1] === j);
  const W0 = [[1, 0.2], [0, 1]], LU = [0.8, 0.6], LV = [0.6, 0.8];
  const grayRamp = t => { const v = Math.round(255 * LA.clamp01(t)); return 'rgb(' + v + ',' + v + ',' + v + ')'; };
  const divRamp = t => LA.color.diverge(2 * t - 1);

  /* ---------- images and their SVDs (deterministic, cached outside the lesson state) ---------- */
  function makeImage(key) {
    if (key === 'letter') { const A = N.zerosM(n, n); for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) { const bar = (j >= 7 && j <= 11) || (j >= 20 && j <= 24); const cross = i >= 14 && i <= 17 && j >= 7 && j <= 24; A[i][j] = (i >= 5 && i <= 26 && (bar || cross)) ? 0.92 : 0.1; } return A; }
    if (key === 'checker') { const A = N.zerosM(n, n); for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) A[i][j] = ((Math.floor(i / 4) + Math.floor(j / 4)) % 2) ? 0.85 : 0.15; return A; }
    if (key === 'gradient') { const A = N.zerosM(n, n); for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) A[i][j] = 0.1 + 0.8 * (i + j) / (2 * n - 2); return A; }
    return LA.rng.smallImage(48, { size: 32 });
  }
  let NOISE = null;
  function noiseField() { if (!NOISE) { const r = LA.rng(83); NOISE = []; for (let i = 0; i < n; i++) { const row = []; for (let j = 0; j < n; j++) row.push(LA.rng.gauss(r)); NOISE.push(row); } } return NOISE; }
  const CACHE = {};
  function data(key, noise) {
    const nz = Math.round((noise || 0) * 100) / 100, id = key + '|' + nz;
    if (CACHE[id]) return CACHE[id];
    const clean = makeImage(key), Z = noiseField();
    const A = nz > 0 ? clean.map((r, i) => r.map((x, j) => x + nz * 0.5 * Z[i][j])) : clean;
    const sv = N.svd(A);
    const d = { clean, A, U: N.cols(sv.U), S: sv.S, V: N.cols(sv.V), norm: N.frobenius(A) };
    CACHE[id] = d; return d;
  }
  const D = s => data(s.imgKey, s.noise);
  function sheet(d, i) { return N.scaleM(N.outer(d.U[i], d.V[i]), d.S[i]); }
  function reconSet(d, on) { const R = N.zerosM(n, n); for (let i = 0; i < n; i++) { if (!on(i)) continue; const s = d.S[i], u = d.U[i], v = d.V[i]; for (let a = 0; a < n; a++) { const ua = s * u[a]; if (ua === 0) continue; const row = R[a]; for (let b = 0; b < n; b++) row[b] += ua * v[b]; } } return R; }
  function errOf(d, on) { let tot = 0, drop = 0; d.S.forEach((x, i) => { tot += x * x; if (!on(i)) drop += x * x; }); return Math.sqrt(drop / tot); }
  const errK = (d, k) => errOf(d, i => i < k);
  function minK(d, thr) { for (let k = 1; k <= n; k++) if (errK(d, k) < thr) return k; return n; }
  const onSet = s => i => i < s.k && s.on[i];
  /** “keep fewer pixels”: keep every f-th row and show each kept row f times */
  function subRows(A, f) { return A.map((r, i) => A[i - (i % f)].slice()); }
  const MIS_F = 6, MIS_K = 3;                     // 6 kept rows · 32 = 192 numbers vs 3 sheets · 65 = 195
  function relDist(X, Y) { return N.frobenius(N.subM(X, Y)) / N.frobenius(Y); }

  /* ---------- drawing helpers ---------- */
  function fitHalfH(g, rect, extent) { const pw = rect.w * g.size.w, ph = rect.h * g.size.h; return Math.max(extent, extent * ph / pw); }
  function panel(g, rect, grid, title, o) {
    o = o || {};
    const hh = fitHalfH(g, rect, 1.25);
    g.viewport(rect, { view: '2d', cam: { center: [0, 0.1], halfH: hh } }, (g2, r) => {
      g2.image(grid, { x0: -1, x1: 1, y0: -1, y1: 1 }, { ramp: o.ramp || grayRamp, range: o.range || [0, 1] });
      if (title) g2.label(title, [0, 1], { anchor: 'above', dy: -5, size: 12.5, color: o.titleColor || 'text', font: 'ui' });
      if (o.sub) g2.label(o.sub, [0, -1], { anchor: 'below', dy: 5, size: 12, color: o.subColor || 'muted', font: 'ui' });
      if (o.extra) o.extra(g2, r);
    });
  }
  /** horizontal progress bar in screen px (drawn through the 2D projector) */
  function hbar(g, x, y, w, h, frac, color) {
    const P = g.P, q = (sx, sy) => P.toWorld(sx, sy);
    g.polygon([q(x, y), q(x + w, y), q(x + w, y + h), q(x, y + h)], { fill: 'panel2', alpha: 1, stroke: 'line' });
    const f = LA.clamp01(frac);
    if (f > 0) g.polygon([q(x, y), q(x + w * f, y), q(x + w * f, y + h), q(x, y + h)], { fill: color, alpha: 0.9 });
  }
  const symRange = M => { let m = 1e-6; M.forEach(r => r.forEach(x => { m = Math.max(m, Math.abs(x)); })); return [-m, m]; };
  /** σ bars (screen space) + optional on/off toggle dots (handles in world coordinates) */
  function sigmaBars(g, s, d, o) {
    o = o || {};
    const W = g.size.w, H = g.size.h;
    const x = W * 0.08, y = H * (o.y || 0.52), w = W * 0.84, h = H * (o.h || 0.2), gap = 2;
    const raw = d.S.slice();
    const floor = raw[0] * 1e-3;
    const vals = s.logScale ? raw.map(v => Math.max(0, Math.log10(Math.max(v, floor) / floor))) : raw;
    const kk = o.k === undefined ? s.k : o.k;
    const cols = raw.map((v, i) => (o.colorAt ? o.colorAt(i) : (i < kk && (!s.on || s.on[i]) ? 'gold' : (o.dropColor || 'muted'))));
    if (o.ghost) {
      const gv = s.logScale ? o.ghost.map(v => Math.max(0, Math.log10(Math.max(v, floor) / floor))) : o.ghost;
      g.bars(gv, { screen: true, x, y, w, h, gap, color: 'hud', alpha: 0.25, max: Math.max.apply(null, vals.concat(gv)) });
    }
    const max = 1.14 * (o.ghost ? Math.max.apply(null, vals.concat(s.logScale ? o.ghost.map(v => Math.max(0, Math.log10(Math.max(v, floor) / floor))) : o.ghost)) : Math.max.apply(null, vals));
    g.bars(vals.map((v, i) => (o.reveal === undefined || i < o.reveal) ? v : 0), { screen: true, x, y, w, h, gap, colors: cols, max, labels: raw.map((v, i) => (i === 0 || (i + 1) % 4 === 0) ? String(i + 1) : ' '), title: o.title || (s.logScale ? 'singular values σ₁ … σ₃₂ (log scale)' : 'singular values σ₁ … σ₃₂') });
    const bw = (w - gap * (n - 1)) / n;
    if (o.valueLabels) for (let i = 0; i < o.valueLabels; i++) { const bh = vals[i] / (max || Math.max.apply(null, vals)) * h; g.hud(LA.fmt(raw[i]), x + i * (bw + gap) + bw / 2, y + h - bh - 9, { anchor: 'center', size: 11, color: 'gold' }); }
    if (o.toggles && !g.is3d) {
      const P = g.P, yOn = y + h + 24, yOff = y + h + 44;
      g.hud('on', x - 26, yOn, { size: 10.5, color: 'muted' }); g.hud('off', x - 26, yOff, { size: 10.5, color: 'muted' });
      for (let i = 0; i < n; i++) {
        const cx = x + i * (bw + gap) + bw / 2, on = s.on[i];
        const wp = P.toWorld(cx, on ? yOn : yOff);
        g.point(wp, { r: 3.5, color: on ? 'gold' : 'bad' });
        g.handle('bar' + i, wp, (p, st) => { const sy = P.toScreen(p)[1]; const nowOn = sy < (yOn + yOff) / 2; if (st.on[i] !== nowOn) { st.on[i] = nowOn; st.lastSheet = i; } }, { r: 7, show: false });
      }
    }
    return { x, y, w, h, bw, gap };
  }

  /* ---------- scenes ---------- */
  function renderIntro(g, s, ctx) {
    const d = D(s), W = g.size.w;
    const big = { x: 0.03, y: 0.04, w: 0.6, h: 0.62 };
    panel(g, big, d.A, 'A: 32 × 32 = 1024 numbers (synthetic image)', { extra: (g2) => {
      const c = 2 / n, x0 = -1 + PC * c, y1 = 1 - PR * c;
      g2.polygon([[x0, y1], [x0 + 4 * c, y1], [x0 + 4 * c, y1 - 4 * c], [x0, y1 - 4 * c]], { stroke: 'gold', width: 2, alpha: 0 });
    } });
    const mag = { x: 0.66, y: 0.12, w: 0.32, h: 0.34 };
    g.viewport(mag, { view: '2d', cam: { center: [0, 0], halfH: fitHalfH(g, mag, 1.3) } }, (g2) => {
      const patch = [0, 1, 2, 3].map(i => [0, 1, 2, 3].map(j => d.A[PR + i][PC + j]));
      g2.image(patch, { x0: -1, x1: 1, y0: -1, y1: 1 }, { ramp: grayRamp, range: [0, 1], cells: true });
      for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) g2.label(LA.fmt(patch[i][j]), [-0.75 + 0.5 * j, 0.75 - 0.5 * i], { size: 11.5, color: patch[i][j] > 0.5 ? 'bg' : 'text', bg: false, bold: true });
      g2.label('magnified 4×4 patch', [0, 1], { anchor: 'above', dy: -5, size: 12, color: 'gold', font: 'ui' });
    });
    g.hud('bright = big number, dark = small', W * 0.66, g.size.h * 0.52, { size: 12.5, color: 'muted' });
    if (s.showBars > 0) sigmaBars(g, s, d, { y: 0.72, h: 0.17, k: s.k, reveal: Math.ceil(32 * s.showBars), valueLabels: s.showBars >= 1 ? 4 : 0 });
  }

  function renderSheet1(g, s, ctx) {
    const d = D(s), W = g.size.w, H = g.size.h;
    const rect = { x: 0.06, y: 0.03, w: 0.88, h: 0.66 };
    const hh = fitHalfH(g, rect, 1.45);
    const S1 = sheet(d, 0), rng = [0, Math.max.apply(null, S1.map(r => Math.max.apply(null, r)))];
    g.viewport(rect, { view: '2d', cam: { center: [-0.1, 0.1], halfH: hh } }, (g2) => {
      const c = 2 / n;
      // u₁ as a vertical strip, v₁ᵀ as a horizontal strip
      const sg = N.sum(d.U[0]) < 0 ? -1 : 1, u1 = d.U[0].map(x => x * sg), v1 = d.V[0].map(x => x * sg);   // u₁v₁ᵀ = (−u₁)(−v₁)ᵀ
      g2.image(u1.map(x => [x]), { x0: -1.32, x1: -1.32 + c, y0: -1, y1: 1 }, { ramp: grayRamp, range: [0, Math.max.apply(null, u1)] });
      g2.image([v1], { x0: -1, x1: 1, y0: 1.08, y1: 1.08 + c }, { ramp: grayRamp, range: [0, Math.max.apply(null, v1)] });
      g2.label('u₁ (32×1)', [-1.32, -1], { anchor: 'below', dy: 6, size: 12, color: 'gold', font: 'ui' });
      g2.label('v₁ᵀ (1×32)', [1, 1.08 + c], { anchor: 'left', dx: 6, size: 12, color: 'gold', font: 'ui' });
      const rows = Math.floor(n * s.fill);
      const shown = S1.map((r, i) => i < rows ? r : r.map(() => rng[0]));
      g2.image(shown, { x0: -1, x1: 1, y0: -1, y1: 1 }, { ramp: grayRamp, range: rng });
      if (rows < n) {
        g2.polygon([[-1, 1 - rows * c], [1, 1 - rows * c], [1, -1], [-1, -1]], { fill: 'panel', alpha: 1 });
        const yr = 1 - (rows + 0.5) * c;
        g2.polygon([[-1, yr + c / 2], [1, yr + c / 2], [1, yr - c / 2], [-1, yr - c / 2]], { stroke: 'gold', width: 1.5, alpha: 0 });
        g2.polygon([[-1.32, yr + c / 2], [-1.32 + c, yr + c / 2], [-1.32 + c, yr - c / 2], [-1.32, yr - c / 2]], { stroke: 'gold', width: 2, alpha: 0 });
      }
      g2.label('sheet 1 = σ₁u₁v₁ᵀ', [0, -1], { anchor: 'below', dy: 6, size: 13, color: 'text', font: 'ui' });
    });
    g.hud('every row is the same row v₁ᵀ, scaled by one entry of u₁', W * 0.06, H * 0.76, { size: 13, color: 'text' });
    g.hud('cost: 32 + 32 + 1 = 65 numbers instead of 1024', W * 0.06, H * 0.81, { size: 13, color: 'gold' });
    g.hud('a blurry, stripy ghost of the picture: its “average structure”', W * 0.06, H * 0.86, { size: 12.5, color: 'muted' });
  }

  function renderStack(g, s, ctx) {
    const d = D(s), W = g.size.w, H = g.size.h;
    const k = Math.max(1, Math.round(s.k)), on = onSet(s);
    const R = reconSet(d, on), e = errOf(d, on);
    const last = s.lastSheet >= 0 ? s.lastSheet : k - 1;
    const Sk = sheet(d, last);
    panel(g, { x: 0.01, y: 0.03, w: 0.32, h: 0.4 }, d.A, s.noise > 0 ? 'noisy original' : 'original A');
    panel(g, { x: 0.34, y: 0.03, w: 0.32, h: 0.4 }, R, s.mode2 === 'toggle' ? 'sheets that are on' : 'A' + subK(k), { titleColor: 'gold', sub: 'error ' + LA.fmt.pct(e, 1), subColor: 'gold' });
    panel(g, { x: 0.67, y: 0.03, w: 0.32, h: 0.4 }, Sk, 'sheet ' + (last + 1) + ' = σ' + subK(last + 1) + 'u' + subK(last + 1) + 'v' + subK(last + 1) + 'ᵀ', { ramp: divRamp, range: symRange(Sk), sub: s.mode2 === 'toggle' ? (s.on[last] ? 'switched on' : 'switched off') : 'the newest sheet', subColor: s.mode2 === 'toggle' && !s.on[last] ? 'bad' : 'muted' });
    const b = sigmaBars(g, s, d, { y: 0.55, h: 0.2, k: s.mode2 === 'toggle' ? n : k, toggles: s.mode2 === 'toggle' || ctx.mode === 'play', colorAt: s.mode2 === 'toggle' ? (i => s.on[i] ? 'gold' : 'muted') : undefined });
    if (s.showErr) {
      const y = H * (ctx.mode === 'check' ? 0.8 : 0.88);
      g.hud('relative error ‖A − A' + subK(k) + '‖/‖A‖ = ' + LA.fmt.pct(e, 1), W * 0.08, y, { size: 13, color: 'text' });
      g.hud('storage k·(32 + 32 + 1) = ' + (k * 65) + ' of 1024 numbers' + (k * 65 >= 1024 ? ' (no saving any more!)' : ''), W * 0.08, y + 22, { size: 13, color: k * 65 >= 1024 ? 'bad' : 'gold' });
    }
    if (ctx.mode === 'check' && s.mode2 !== 'toggle') slider(g, s, { x: 0.04, y: 0.87, w: 0.92, h: 0.11 });
  }
  const subK = k => String(k).split('').map(c => '₀₁₂₃₄₅₆₇₈₉'[+c]).join('');

  function renderBudget(g, s, ctx) {
    const d = D(s), W = g.size.w, H = g.size.h, k = 8;
    panel(g, { x: 0.02, y: 0.03, w: 0.46, h: 0.42 }, d.A, 'original');
    panel(g, { x: 0.52, y: 0.03, w: 0.46, h: 0.42 }, reconSet(d, i => i < k), 'A₈', { titleColor: 'gold' });
    sigmaBars(g, s, d, { y: 0.52, h: 0.18, k, dropColor: 'mag', title: 'kept σ₁…σ₈ (gold) · dropped σ₉…σ₃₂ (magenta)' });
    const e = errK(d, k), y = H * 0.8;
    g.hud('error = √(Σ dropped σ²) / √(Σ all σ²) = ' + LA.fmt.pct(e, 1), W * 0.08, y, { size: 13.5, color: 'mag' });
    hbar(g, W * 0.08, y + 12, W * 0.4, 12, e, 'mag');
    g.hud('storage 8 · 65 = 520 of 1024 numbers', W * 0.08, y + 46, { size: 13.5, color: 'gold' });
    if (s.chip > 0) g.hud('Theorem (Eckart–Young): no rank-8 matrix is closer', W * 0.42, y + 46, { size: 12.5, color: 'ok', alpha: s.chip, bold: true });
  }

  function renderMis(g, s, ctx) {
    const d = data('letter', 0), W = g.size.w, H = g.size.h;
    const sub = subRows(d.A, MIS_F), Ak = reconSet(d, i => i < MIS_K);
    const eS = relDist(sub, d.A), eK = relDist(Ak, d.A);
    panel(g, { x: 0.02, y: 0.06, w: 0.46, h: 0.5 }, sub, 'keep every 6th row: 6 × 32 = 192 numbers', { sub: 'crossbar lost · error ' + LA.fmt.pct(eS, 1), subColor: 'bad' });
    panel(g, { x: 0.52, y: 0.06, w: 0.46, h: 0.5 }, Ak, 'keep 3 sheets: 3 · 65 = 195 numbers', { titleColor: 'gold', sub: 'the whole letter · error ' + LA.fmt.pct(eK, 1), subColor: 'ok' });
    if (s.cross) { g.marker.cross([W * 0.25, H * 0.66], { screen: true, size: 30 }); g.marker.check([W * 0.75, H * 0.66], { screen: true, size: 30 }); }
    g.hud('“to store fewer numbers, keep fewer pixels”', W * 0.04, H * 0.76, { size: 13.5, color: 'bad' });
    g.hud('the SVD keeps every pixel and drops faint sheets instead', W * 0.04, H * 0.82, { size: 13.5, color: 'gold' });
    g.hud('same budget (≈ 190 of 1024 numbers), synthetic block letter H', W * 0.04, H * 0.88, { size: 12.5, color: 'muted' });
  }

  function renderNoise(g, s, ctx) {
    const d = D(s), clean = data(s.imgKey, 0), W = g.size.w, H = g.size.h;
    const K = 3, Ak = reconSet(d, i => i < K);
    panel(g, { x: 0.02, y: 0.03, w: 0.46, h: 0.42 }, d.A, 'noisy letter A + N (synthetic)', { sub: 'off by ' + LA.fmt.pct(relDist(d.A, clean.A), 1) + ' from the clean letter', subColor: 'bad' });
    panel(g, { x: 0.52, y: 0.03, w: 0.46, h: 0.42 }, Ak, 'A₃ of the noisy letter', { titleColor: 'gold', sub: 'off by ' + LA.fmt.pct(relDist(Ak, clean.A), 1) + ' from the clean letter', subColor: 'ok' });
    sigmaBars(g, s, d, { y: 0.55, h: 0.22, k: K, ghost: clean.S, title: 'σ of the noisy letter (bars) vs the clean letter (faint): the tail rises', colorAt: i => i < K ? 'gold' : 'mag' });
    g.hud('the clean letter has only 3 nonzero σ; noise spreads over all 32', W * 0.04, H * 0.85, { size: 12.5, color: 'text' });
    g.hud('structure is low-rank, noise is not: keeping 3 sheets keeps the letter and drops most of the speckle', W * 0.04, H * 0.9, { size: 12.5, color: 'muted' });
  }

  function ratings(U) { return U.map(u => V8.map(v => LA.clamp(3 + N.dot(u, v), 1, 5))); }
  function renderRecsys(g, s, ctx) {
    const W = g.size.w, H = g.size.h, R = ratings(s.U6);
    const top = { x: 0.02, y: 0.02, w: 0.96, h: 0.44 };
    const hot = s.hotCell;
    g.viewport(top, { view: '2d', cam: { center: [-0.1, -0.05], halfH: Math.max(1.72, 2.4 * (top.h * g.size.h) / (top.w * g.size.w)) } }, (g2) => {
      const c = 0.42, x0 = -1.68, y1 = 1.26;
      const cell = (i, j) => [[x0 + j * c, y1 - i * c], [x0 + (j + 1) * c, y1 - i * c], [x0 + (j + 1) * c, y1 - (i + 1) * c], [x0 + j * c, y1 - (i + 1) * c]];
      g2.image(R, { x0, x1: x0 + 8 * c, y0: y1 - 6 * c, y1 }, { ramp: LA.color.heat, range: [1, 5], cells: true });
      for (let i = 0; i < 6; i++) {
        g2.label('user ' + (i + 1), [x0, y1 - (i + 0.5) * c], { anchor: 'right', dx: -6, size: 11, color: 'w', font: 'ui' });
        for (let j = 0; j < 8; j++) {
          const ctr = [x0 + (j + 0.5) * c, y1 - (i + 0.5) * c];
          if (isHidden(i, j)) {
            if (s.fillIn < 1) g2.polygon(cell(i, j), { fill: 'panel', alpha: 1 - s.fillIn });
            if (s.fillIn > 0) { g2.polygon(cell(i, j), { stroke: 'white', width: 1.4, dash: [3, 3], alpha: 0 }); g2.label(LA.fmt.fixed(R[i][j], 1), ctr, { size: 10.5, color: 'text', bg: false, alpha: s.fillIn }); }
            else g2.label('?', ctr, { size: 12, color: 'muted', bg: false });
          } else g2.label(LA.fmt.fixed(R[i][j], 1), ctr, { size: 10.5, color: 'bg', bg: false, bold: true });
        }
      }
      for (let j = 0; j < 8; j++) g2.label('f' + (j + 1), [x0 + (j + 0.5) * c, y1], { anchor: 'above', dy: -3, size: 11, color: 'v', font: 'ui' });
      if (hot) g2.polygon(cell(hot[0], hot[1]), { stroke: 'gold', width: 2.5, alpha: 0 });
      g2.label('ratings R (6 × 8): ? = not rated yet', [x0 + 4 * c, y1 - 6 * c], { anchor: 'below', dy: 6, size: 12, color: 'muted', font: 'ui' });
    });
    const bot = { x: 0.08, y: 0.48, w: 0.84, h: 0.5 };
    g.viewport(bot, { view: '2d', cam: { center: [0, 0.1], halfH: fitHalfH(g, bot, 1.9) } }, (g2) => {
      g2.grid(null, { range: 2, alpha: 0.35 }); g2.axes({ ticks: false });
      g2.label('action ↔ drama', [1.9, 0], { anchor: 'below', dy: 8, size: 11, color: 'muted', font: 'ui' });
      g2.label('light ↔ dark', [0, 1.85], { anchor: 'left', dx: 6, size: 11, color: 'muted', font: 'ui' });
      V8.forEach((v, j) => g2.point(v, { r: 5, color: 'v', label: 'f' + (j + 1), labelColor: 'v', size: 11 }));
      s.U6.forEach((u, i) => g2.point(u, { r: 6, color: 'w', label: 'u' + (i + 1), labelColor: 'w', size: 11.5 }));
      if (hot) {
        const u = s.U6[hot[0]], v = V8[hot[1]];
        g2.vector(u, { color: 'w', width: 2.4, label: '' });
        g2.vector(v, { color: 'v', width: 2.4, label: '' });
        const pr = N.scale(u, N.dot(u, v) / N.dot(u, u));
        g2.dropLine(v, pr, { color: 'residual', footColor: 'mag' });
        g2.line([0, 0], pr, { color: 'mag', width: 3 });
        g2.hud('u' + (hot[0] + 1) + '·f' + (hot[1] + 1) + ' = ' + LA.fmt(N.dot(u, v)) + '  →  rating 3 + ' + LA.fmt(N.dot(u, v)) + ' = ' + LA.fmt(LA.clamp(3 + N.dot(u, v), 1, 5), 1), (bot.x + 0.01) * W, (bot.y + 0.04) * H, { size: 13, color: 'gold' });
      }
      if (ctx.mode !== 'learn' || s.dragUsers) s.U6.forEach((u, i) => g2.handle('user' + i, u, (p, st) => { st.U6[i] = [LA.clamp(Math.round(p[0] * 100) / 100, -2, 2), LA.clamp(Math.round(p[1] * 100) / 100, -2, 2)]; }, { color: 'w' }));
    });
    g.hud('48 cells from 28 numbers: 6 users × 2 + 8 films × 2 (rank 2)', W * 0.08, H * 0.97, { size: 12, color: 'muted' });
  }

  function renderLora(g, s, ctx) {
    const Wp = N.addM(W0, N.scaleM(N.outer(LU, LV), s.alpha));
    g.grid(W0, { color: 'ghost', axisColor: 'ghostAxis', back: true });
    g.grid(Wp, { alpha: 0.75 });
    const perp = [-LV[1], LV[0]];
    g.lineThrough([0, 0], N.matVec(W0, perp), { color: 'teal', width: 1.4, dashed: true, alpha: 0.8 });
    [-2, -1, 1, 2].forEach(t => g.point(N.matVec(Wp, N.scale(perp, t)), { r: 4.5, color: 'teal' }));
    g.label('inputs ⟂ v do not move', N.matVec(W0, N.scale(perp, 2.3)), { color: 'teal', size: 12, font: 'ui' });
    const x = N.scale(LV, 1.6), Wx = N.matVec(W0, x), Wpx = N.matVec(Wp, x);
    g.line(Wx, Wpx, { color: 'mag', width: 2.2, dashed: true });
    g.point(Wpx, { r: 5, color: 'b', label: 'W′x for x along v', labelColor: 'b' });
    g.vector(LU, { color: 'gold', width: 3, label: 'u', labelColor: 'gold' });
    g.vector(LV, { color: 'gold', width: 2.2, dashed: true, label: 'v', labelColor: 'gold' });
    g.hud('α = ' + LA.fmt(s.alpha) + ':  W′ = W + α·u vᵀ  (4 numbers + α steer all of W)', 12, 22, { size: 13.5, color: 'gold' });
    slider(g, s, { x: 0.04, y: 0.87, w: 0.92, h: 0.11 }, 'alpha', 'α', -1, 1);
  }

  function slider(g, s, rect, bind, label, lo, hi) {
    bind = bind || 'k'; label = label || 'k'; lo = lo === undefined ? 1 : lo; hi = hi === undefined ? n : hi;
    g.viewport(rect, { view: '2d', cam: { center: [0, 0], halfH: 0.32 } }, (g2) => {
      const X = f => -2 + 4 * f;
      g2.line([X(0), -0.06], [X(1), -0.06], { color: 'gridAxis', width: 2 });
      [0, 0.5, 1].forEach(f => g2.label(LA.fmt(bind === 'k' ? Math.round(lerp(lo, hi, f)) : lerp(lo, hi, f)), [X(f), -0.1], { anchor: 'below', dy: 4, size: 10.5, color: 'muted', font: 'ui' }));
      const v = s[bind], xx = X((v - lo) / (hi - lo));
      g2.label(label + ' = ' + LA.fmt(v) + '  (drag)', [X(0), 0.17], { anchor: 'left', size: 12, color: 'v', font: 'ui' });
      g2.point([xx, -0.06], { r: 6, color: 'v' });
      g2.handle('sl-' + bind, [xx, -0.06], (q, st) => { let val = lerp(lo, hi, LA.clamp01((q[0] + 2) / 4)); val = bind === 'k' ? Math.round(val) : Math.round(val * 100) / 100; st[bind] = val; }, { color: 'v' });
    });
  }

  const ROWS = [{ x: 0.03, y: 0.03, w: 0.55, h: 0.3 }, { x: 0.03, y: 0.35, w: 0.55, h: 0.3 }, { x: 0.03, y: 0.67, w: 0.55, h: 0.3 }];
  function rowCaptions(g, caps) {
    const W = g.size.w, H = g.size.h;
    ROWS.forEach((b, i) => { const y = (b.y + b.h / 2) * H, x = (b.x + b.w + 0.03) * W; g.hud(caps[i][0], x, y - 18, { color: 'text', size: 14, bold: true }); g.hud(caps[i][1], x, y + 4, { color: 'muted', size: 12.5 }); if (caps[i][2]) g.hud(caps[i][2], x, y + 24, { color: 'gold', size: 12.5 }); });
  }
  function renderML(g, s, ctx) {
    const u = s.mlT;
    g.viewport(ROWS[0], { view: '2d', cam: { center: [0, 0], halfH: 1.1 } }, (g2, r) => {
      const m = 64, vals = []; for (let i = 0; i < m; i++) vals.push(Math.exp(-i / 6) + 0.04);
      const shown = Math.ceil(m * LA.clamp01(u * 1.4));
      g2.bars(vals.map((v, i) => i < shown ? v : 0), { screen: true, x: r.x + 14, y: r.y + 24, w: r.w - 28, h: r.h - 48, gap: 0.5, colors: vals.map((v, i) => i < 8 ? 'gold' : 'muted') });
      g2.hud('spectrum of a trained 4096×4096 layer (sketch)', r.x + 10, r.y + 14, { size: 11.5, color: 'muted' });
    });
    g.viewport(ROWS[1], { view: '2d', cam: { center: [0, 0], halfH: 1.15 } }, (g2) => {
      const box = (x0, y0, w, h, c, l) => { g2.polygon([[x0, y0], [x0 + w, y0], [x0 + w, y0 + h], [x0, y0 + h]], { fill: c, alpha: 0.35, stroke: c }); g2.label(l, [x0 + w / 2, y0 + h / 2], { size: 11.5, color: 'text', bg: false }); };
      box(-1.7, -0.8, 1.6, 1.6, 'grid', 'W 16.8 M');
      g2.label('+', [0.05, 0], { size: 16, color: 'hud' });
      box(0.25, -0.8, 0.16, 1.6, 'gold', 'B'); box(0.5, 0.64, 1.3, 0.16, 'gold', 'A');
      g2.label('B: 4096×8, A: 8×4096 → 65 536 numbers', [0.25, -0.8], { anchor: 'left', dy: 14, size: 11, color: 'gold', font: 'ui' });
    });
    g.viewport(ROWS[2], { view: '2d', cam: { center: [0, 0], halfH: 1.15 } }, (g2) => {
      const box = (x0, y0, w, h, c, l) => { g2.polygon([[x0, y0], [x0 + w, y0], [x0 + w, y0 + h], [x0, y0 + h]], { fill: c, alpha: 0.3, stroke: c }); g2.label(l, [x0 + w / 2, y0 + h / 2], { size: 11.5, color: 'text', bg: false }); };
      box(-1.9, -0.85, 1.5, 1.7, 'teal', 'R (users × items)');
      g2.label('≈', [-0.25, 0], { size: 18, color: 'hud' });
      box(-0.05, -0.85, 0.22, 1.7, 'w', 'U'); box(0.3, 0.6, 1.5, 0.25, 'v', 'Vᵀ');
    });
    rowCaptions(g, [['Decaying spectra', 'trained weights look like', 'the picture’s bars'], ['LoRA', 'rank-8 update of a', '4096×4096 layer'], ['Recommenders', 'a huge sparse table', '= two thin factors']]);
  }
  function renderRecap(g, s, ctx) {
    const d = data('seed48', 0);
    g.viewport(ROWS[0], { view: '2d', cam: { center: [0, 0], halfH: 1.2 } }, (g2) => {
      const sg = N.sum(d.U[0]) < 0 ? -1 : 1, u1 = d.U[0].map(x => x * sg), v1 = d.V[0].map(x => x * sg);
      g2.image(u1.map(x => [x]), { x0: -1.0, x1: -0.92, y0: -0.8, y1: 0.8 }, { ramp: grayRamp, range: [0, Math.max.apply(null, u1)] });
      g2.image([v1], { x0: -0.8, x1: 0.8, y0: 0.88, y1: 0.96 }, { ramp: grayRamp, range: [0, Math.max.apply(null, v1)] });
      const S1 = sheet(d, 0);
      g2.image(S1, { x0: -0.8, x1: 0.8, y0: -0.8, y1: 0.8 }, { ramp: grayRamp, range: [0, Math.max.apply(null, S1.map(r => Math.max.apply(null, r)))] });
    });
    g.viewport(ROWS[1], { view: '2d', cam: { center: [0, 0], halfH: 1.1 } }, (g2, r) => {
      g2.bars(d.S.slice(0, 16), { screen: true, x: r.x + 12, y: r.y + 20, w: r.w * 0.5, h: r.h - 40, gap: 1, colors: d.S.slice(0, 16).map((v, i) => i < 4 ? 'gold' : 'muted') });
      g2.image(reconSet(d, i => i < 4), { x0: 0.75, x1: 2.1, y0: -0.7, y1: 0.65 }, { range: [0, 1] });
    });
    g.viewport(ROWS[2], { view: '2d', cam: { center: [0.75, 0.32], halfH: 0.72 } }, (g2) => {
      const u = U6[0], v = V8[1], pr = N.scale(u, N.dot(u, v) / N.dot(u, u));
      g2.vector(u, { color: 'w', width: 2.4, label: 'user', labelColor: 'w' });
      g2.vector(v, { color: 'v', width: 2.4, label: 'film', labelColor: 'v' });
      g2.dropLine(v, pr, { footColor: 'mag' }); g2.line([0, 0], pr, { color: 'mag', width: 3 });
    });
    rowCaptions(g, [['Sheets', 'column × row = rank 1'], ['Keep the big σ’s', 'best rank-k, small error'], ['Low-rank tastes', 'rating = 3 + user · film']]);
  }

  LA.registerLesson({
    id: 'p4-08-low-rank-compression',
    part: 4,
    title: 'Low-rank approximation, SVD image compression, recommenders/LoRA',
    subtitle: 'A matrix is a stack of rank-1 sheets weighted by σ; keep the biggest few.',
    minutes: 13,
    prereqs: ['p4-07-svd', 'p2-10-transpose-outer-symmetric', 'p2-08-rank-column-null'],
    tags: ['core', 'capstone'],
    badge: { drawnIn: '2D heatmaps (the matrix is the picture, no projection)', realSize: 'a 1000×1000 image has 1000 σ’s; LoRA adapts 4096×4096 weights with rank 8–64; a streaming service’s ratings matrix is 10⁸ × 10⁵', transfers: 'A = Σσᵢuᵢvᵢᵀ and “keep the biggest σ’s” are exact at every size; only the number of sheets grows' },
    cantShow: ['latent spaces with k > 2 (real recommenders use k ≈ 50–200); the 2-D taste plane is exact only for this toy', 'why truncation is optimal (Eckart–Young): stated as a theorem, not proved', 'color images (three stacked matrices): grayscale only'],
    terms: [
      { term: 'rank-1 sheet', def: 'A matrix uvᵀ: one column times one row, so every row is a multiple of the same row.' },
      { term: 'truncated SVD', def: 'Keep the k largest singular values and their u, v: A_k, the best rank-k approximation (Eckart–Young).' },
      { term: 'Frobenius error', def: '‖A − A_k‖ / ‖A‖: the square root of the summed squared pixel errors, relative to the size of A.' },
      { term: 'latent space', def: 'A hidden coordinate plane where users and items are points and a rating is their dot product.' },
      { term: 'LoRA', def: 'Fine-tuning a weight matrix W by adding a low-rank product BA instead of changing all of W.' },
    ],
    atlas: ['svd', 'embeddings', 'regularization', 'pca'],
    thumbnail: 3,

    state() {
      data('seed48', 0);                         // compute the image SVD once
      return {
        view: '2d', cam: LA.cam.defaults2d(),
        scene: 'intro', panel: 'A', imgKey: 'seed48', noise: 0, k: 1, on: new Array(n).fill(true), lastSheet: -1, logScale: false,
        showBars: 0, fill: 1, mode2: 'stack', showErr: true, chip: 0, cross: false,
        U6: LA.clone(U6), hotCell: null, fillIn: 0, dragUsers: false, alpha: 0, mlT: 1, mode: 'image',
      };
    },

    render(g, s, ctx) {
      switch (s.scene) {
        case 'sheet1': return renderSheet1(g, s, ctx);
        case 'stack': return renderStack(g, s, ctx);
        case 'budget': return renderBudget(g, s, ctx);
        case 'mis': return renderMis(g, s, ctx);
        case 'noise': return renderNoise(g, s, ctx);
        case 'recsys': return renderRecsys(g, s, ctx);
        case 'lora': return renderLora(g, s, ctx);
        case 'ml': return renderML(g, s, ctx);
        case 'recap': return renderRecap(g, s, ctx);
        default: return renderIntro(g, s, ctx);
      }
    },

    math(s, tex) {
      const T = (x, o) => tex.text(x, o);
      const d = D(s), k = Math.max(1, Math.round(s.k));
      switch (s.panel) {
        case 'A': return tex.block(tex.eq(tex.sym('A'), tex.op('∈'), T('ℝ³²ˣ³²')));
        case 'sigma': return tex.block(tex.eq(T('σ₁ ≥ σ₂ ≥ … ≥ σ₃₂ ≥ 0', { color: 'gold' })), tex.eq(T('σ₁…σ₄ = ' + d.S.slice(0, 4).map(x => LA.fmt(x)).join(', '))));
        case 'sheet1': return tex.block(tex.eq(tex.sym('σ', { sub: '1' }), tex.sym('u', { sub: '1', color: 'gold' }), tex.sym('v', { sub: '1', sup: 'T', color: 'gold' })), tex.eq(T('(32×1)(1×32) = 32×32')), tex.eq(T('cost 32 + 32 + 1 = 65', { color: 'gold' })));
        case 'stack': return tex.block(tex.eq(tex.sym('A', { sub: 'k' }), tex.op('='), tex.sum(tex.sym('σ', { sub: 'i' }) + tex.sym('u', { sub: 'i', color: 'gold' }) + tex.sym('v', { sub: 'i', sup: 'T', color: 'gold' }), 'i≤k')), tex.eq(T('k = ' + k + ', error ' + LA.fmt.pct(errK(d, k), 1), { color: 'gold' })));
        case 'budget': return tex.block(tex.eq(T('err₈ = √(σ₉² + … + σ₃₂²) / √(σ₁² + … + σ₃₂²) = ' + LA.fmt.pct(errK(d, 8), 1))), tex.eq(T('storage = k(32 + 32 + 1) = 8 · 65 = 520', { color: 'gold' })));
        case 'mis': { const d0 = data('letter', 0); return tex.block(tex.eq(T('err(every 6th row) = ' + LA.fmt.pct(relDist(subRows(d0.A, MIS_F), d0.A), 1), { color: 'bad' })), tex.eq(T('err(A₃) = ' + LA.fmt.pct(errK(d0, MIS_K), 1), { color: 'ok' }))); }
        case 'toggle': return tex.block(tex.eq(tex.sym('A', { sub: 'on' }), tex.op('='), tex.sum(tex.sym('σ', { sub: 'i' }) + tex.sym('u', { sub: 'i', color: 'gold' }) + tex.sym('v', { sub: 'i', sup: 'T', color: 'gold' }), 'i∈on')), tex.eq(T('off: ' + (s.on.map((o, i) => o ? null : i + 1).filter(x => x).join(', ') || 'none'))));
        case 'noise': return tex.block(tex.eq(tex.sym('A'), tex.op('+'), tex.sym('N')), tex.eq(T('σ tail ↑, first σ’s barely move', { color: 'mag' })));
        case 'recsys': return tex.block(tex.eq(tex.sym('R'), tex.op('≈'), T('3 + U Vᵀ')), tex.eq(T('(6×2)(2×8)')), tex.eq(tex.sym('r', { sub: 'ij' }), tex.op('='), T('3 + uᵢ · vⱼ', { color: 'gold' })));
        case 'lora': return tex.block(tex.eq(tex.sym('W', { sup: '′' }), tex.op('='), tex.sym('W'), tex.op('+'), tex.sym('α'), tex.sym('u', { color: 'gold' }), tex.sym('v', { sup: 'T', color: 'gold' })), tex.eq(T('rank(u vᵀ) = 1, α = ' + LA.fmt(s.alpha))), tex.eq(tex.sym('W', { sup: '′' }), tex.op('='), tex.mat(N.addM(W0, N.scaleM(N.outer(LU, LV), s.alpha)), { colors: 'cols' })));
        case 'ml': return tex.block(tex.eq(tex.sym('W'), tex.op('+'), tex.sym('B'), tex.sym('A')), tex.eq(T('rank ≤ r')));
        case 'check': return tex.block(tex.eq(T('k = ' + k + ', error ' + LA.fmt.pct(errK(d, k), 1), { color: 'gold' })));
        case 'check3': { const u = s.U6[2], sc = V8.map(v => N.dot(u, v)); const b = sc.indexOf(Math.max.apply(null, sc)); return tex.block(tex.eq(T('user 3’s favourite: film ' + (b + 1) + ' (score ' + LA.fmt(sc[b]) + ')', { color: 'w' }))); }
        default: return tex.block(tex.eq(tex.sym('A', { sub: 'k' }), tex.op('='), tex.sum(tex.sym('σ', { sub: 'i' }) + tex.sym('u', { sub: 'i', color: 'gold' }) + tex.sym('v', { sub: 'i', sup: 'T', color: 'gold' }), 'i≤k')));
      }
    },

    steps: [
      { title: 'An image is a matrix', duration: 3,
        narration: 'This 32×32 grayscale picture (synthetic) is 1024 numbers in a grid: a matrix <b>A</b>. Bright cells are big numbers, dark cells small; the gold square is magnified on the right. Everything we know about matrices applies to it, including the SVD. Let us take it apart.',
        setup(s) { s.scene = 'intro'; s.view = '2d'; s.cam = LA.cam.defaults2d(); s.imgKey = 'seed48'; s.noise = 0; s.k = 1; s.on = new Array(n).fill(true); s.showBars = 0; s.logScale = false; s.panel = 'A'; },
        at(s, t) { } },

      { title: 'Its singular values', duration: 3.5,
        narration: 'A = UΣVᵀ. The 32 singular values, drawn as bars, fall off a cliff: the first few are large, most are tiny. Whatever is in this picture is mostly carried by a handful of directions. In the playground a log-scale switch shows the tail.',
        setup(s) { s.showBars = 0; s.k = 32; s.panel = 'sigma'; },
        at(s, t) { s.showBars = seg(t, 0.05, 0.9, 'linear'); } },

      { title: 'One sheet = a column times a row', duration: 4,
        narration: 'The first piece, σ₁' + gold('u₁v₁ᵀ') + ', is a 32-cell column times a 32-cell row: an outer product, a <b>rank-1 sheet</b>. Every row of it is the same row, scaled. On its own it is a blurry, stripy ghost of the picture, the “average structure”, and it costs 65 numbers instead of 1024.',
        setup(s) { s.scene = 'sheet1'; s.fill = 0; s.panel = 'sheet1'; },
        at(s, t) { s.fill = seg(t, 0.05, 0.6, 'linear'); } },

      { title: 'Stack sheets: the picture sharpens', duration: 5,
        narration: 'Add sheet 2, then 3, then 4… Each one adds detail the previous ones missed. Large structures appear by k = 3; the disc’s edge needs about 8; fine texture waits for the small σ’s. The reconstruction A_k = Σᵢ≤k σᵢuᵢvᵢᵀ is the <b>truncated SVD</b>.',
        setup(s) { s.scene = 'stack'; s.mode2 = 'stack'; s.k = 1; s.on = new Array(n).fill(true); s.lastSheet = -1; s.showErr = true; s.panel = 'stack'; },
        at(s, t) { s.k = 1 + Math.floor(15 * seg(t, 0.03, 0.95, 'linear')); } },

      { title: 'Error and storage', duration: 3.5,
        narration: 'Two numbers to watch. The relative <b>Frobenius error</b> comes straight from the dropped magenta bars: √(Σᵢ>k σᵢ²)/√(Σσᵢ²). The storage is k·(32+32+1): at k = 8 that is 520 numbers, about half, for a picture that keeps every big shape. No rank-8 matrix does better: the Eckart–Young theorem.',
        setup(s) { s.scene = 'budget'; s.k = 8; s.chip = 0; s.panel = 'budget'; },
        at(s, t) { s.chip = seg(t, 0.5, 0.8); } },

      { title: 'Misconception: compression means dropping pixels', duration: 3.5, misconception: true,
        narration: '✕ “To store fewer numbers, keep fewer pixels.” Try it on a synthetic block letter H with about 190 numbers: keeping every 6th row (192 numbers) misses the crossbar entirely, about 47 % error. Three sheets (195 numbers) rebuild the whole letter, 0 % error. The SVD keeps <i>every</i> pixel and drops faint <i>sheets</i>: what goes is weak structure, not whole locations.',
        setup(s) { s.scene = 'mis'; s.cross = false; s.panel = 'mis'; },
        at(s, t) { s.cross = t > 0.4; } },

      { title: 'Switch sheets off and on: they are independent', duration: 3.5, interactive: true,
        narration: 'Drag a bar’s dot down to switch that sheet off, up to switch it back on. Remove sheet 1 and the average brightness goes with it, leaving only the detail. Remove sheet 5 alone and one checkered pattern near the dark square vanishes. The sheets add independently: each is perpendicular to the others, in its row and its column.',
        setup(s) { s.scene = 'stack'; s.mode2 = 'toggle'; s.k = n; s.on = new Array(n).fill(true); s.lastSheet = 0; s.showErr = false; s.panel = 'toggle'; },
        at(s, t) {
          const on = new Array(n).fill(true);
          if (t > 0.15 && t < 0.5) on[0] = false;
          if (t >= 0.6) on[4] = false;
          s.on = on; s.lastSheet = t < 0.55 ? 0 : 4;
        } },

      { title: 'Noise lives in the small σ’s', duration: 3.5,
        narration: 'Back to the block letter H, built from just 3 sheets: now sprinkle random noise on it. The big singular values barely move; the tail lifts. Keeping k = 3 now <i>removes</i> most of the noise, because random speckle has no large sheets. Truncation denoises for the same reason it compresses: structure is low-rank, noise is not.',
        setup(s) { s.scene = 'noise'; s.imgKey = 'letter'; s.on = new Array(n).fill(true); s.k = 3; s.noise = 0; s.panel = 'noise'; },
        at(s, t) { s.noise = 0.3 * seg(t, 0.05, 0.6); } },

      { title: 'A ratings table is low-rank', duration: 5,
        narration: 'Six people rated eight films, with gaps. Place each person and each film as a point in a 2-D <b>latent space</b> of taste. A rating is 3 + the dot product of the two points: the shadow of one on the other. All 48 cells come from 28 coordinates: rank 2, and the gaps fill in with the same dot products.',
        setup(s) { s.scene = 'recsys'; s.noise = 0; s.imgKey = 'seed48'; s.U6 = LA.clone(U6); s.hotCell = [0, 0]; s.fillIn = 0; s.panel = 'recsys'; },
        at(s, t) {
          const cells = [[0, 0], [1, 5], [2, 2], [0, 2]];
          s.hotCell = cells[Math.min(3, Math.floor(t * 4))];
          s.fillIn = seg(t, 0.5, 0.9);
        } },

      { title: 'LoRA: a rank-1 nudge to a weight matrix', duration: 4, interactive: true,
        narration: 'A weight matrix W is a grid transformation. Fine-tuning changes it, but a rank-1 change α·' + gold('u') + gold('vᵀ') + ' moves the grid along <i>one</i> output direction u, by an amount read off one input direction v. Slide α: inputs perpendicular to v stay put, everything else shifts along u. That is <b>LoRA</b> in miniature.',
        setup(s) { s.scene = 'lora'; s.view = '2d'; s.cam = Object.assign(LA.cam.defaults2d(), { halfH: 2.8 }); s.alpha = -1; s.hotCell = null; s.panel = 'lora'; },
        at(s, t) { s.alpha = -1 + 2 * seg(t, 0.05, 0.9); } },

      { title: 'Why ML cares', duration: 3.5, mlTie: true,
        narration: 'Trained weight matrices have decaying singular values, just like the picture, so a rank-8 update can steer a 4096×4096 layer with 65 536 numbers instead of 16 million: that is LoRA. Recommenders factor a huge sparse table into two thin ones. The same bars, the same sheets, bigger.',
        setup(s) { s.scene = 'ml'; s.cam = LA.cam.defaults2d(); s.mlT = 0; s.panel = 'ml'; },
        at(s, t) { s.mlT = seg(t, 0.05, 0.9, 'linear'); } },

      { title: 'Recap', duration: 2.5, recap: true,
        narration: 'A matrix is a stack of rank-1 sheets weighted by σ. Keep the largest few: the best rank-k approximation, small error, small storage, less noise. Ratings tables and weight updates are low-rank: a few directions explain the whole.',
        cards: ['<b>Sheets.</b> σᵢuᵢvᵢᵀ is a column times a row; A is their sum.', '<b>Keep the big σ’s.</b> A_k is the best rank-k matrix; error = √(dropped σ²) / √(all σ²).', '<b>Low-rank worlds.</b> Ratings = 3 + user·film; LoRA updates W by BA of tiny rank.'],
        setup(s) { s.scene = 'recap'; s.panel = 'recap'; },
        at(s, t) { } },
    ],

    playground: {
      intro: '<p>Choose k, add noise, switch the log scale or the image, drag a bar’s dot down/up to switch a sheet off/on, drag users in the taste plane (recommender mode), or slide α (LoRA mode).</p>',
      setup(s) { s.scene = 'stack'; s.view = '2d'; s.cam = LA.cam.defaults2d(); s.mode2 = 'stack'; s.k = 8; s.on = new Array(n).fill(true); s.lastSheet = -1; s.showErr = true; s.panel = 'stack'; },
      controls: [
        { type: 'slider', bind: 'k', min: 1, max: 32, step: 1, label: 'k (sheets kept)', color: 'gold' },
        { type: 'toggle', bind: 'logScale', label: 'log scale for σ' },
        { type: 'slider', bind: 'noise', min: 0, max: 0.5, step: 0.01, label: 'noise', color: 'mag' },
        { type: 'preset', label: 'Image', items: [
          { label: 'Synthetic (seed 48)', values: { imgKey: 'seed48' } },
          { label: 'Letter H', values: { imgKey: 'letter' } },
          { label: 'Checkerboard', values: { imgKey: 'checker' } },
          { label: 'Gradient', values: { imgKey: 'gradient' } },
        ] },
        { type: 'preset', label: 'Mode', items: [
          { label: 'Image', apply(s) { s.scene = 'stack'; s.view = '2d'; s.cam = LA.cam.defaults2d(); s.panel = 'stack'; } },
          { label: 'Recommender', apply(s) { s.scene = 'recsys'; s.view = '2d'; s.cam = LA.cam.defaults2d(); s.fillIn = 1; s.hotCell = [2, 0]; s.panel = 'recsys'; } },
          { label: 'LoRA', apply(s) { s.scene = 'lora'; s.view = '2d'; s.cam = Object.assign(LA.cam.defaults2d(), { halfH: 2.8 }); s.panel = 'lora'; } },
        ] },
        { type: 'slider', bind: 'alpha', min: -1, max: 1, step: 0.01, label: 'LoRA α', color: 'gold' },
      ],
    },

    checks: [
      { kind: 'manipulate',
        prompt: 'Drag the k slider so the relative error falls below 10 %.',
        setup(s) { s.scene = 'stack'; s.view = '2d'; s.cam = LA.cam.defaults2d(); s.mode2 = 'stack'; s.imgKey = 'seed48'; s.noise = 0; s.k = 1; s.on = new Array(n).fill(true); s.lastSheet = -1; s.showErr = true; s.panel = 'check'; },
        judge(s) { const e = errK(D(s), Math.round(s.k)); return { ok: e < 0.10, progress: LA.clamp01(1 - (e - 0.10) / 0.5) }; },
        hint: 'Watch the error under the middle picture; each sheet you add removes its σ² from it.',
        solution(s) { s.k = minK(D(s), 0.10); } },
      { kind: 'manipulate',
        prompt: 'Find the smallest k with error under 15 %.',
        setup(s) { s.scene = 'stack'; s.view = '2d'; s.cam = LA.cam.defaults2d(); s.mode2 = 'stack'; s.imgKey = 'seed48'; s.noise = 0; s.k = n; s.on = new Array(n).fill(true); s.lastSheet = -1; s.showErr = true; s.panel = 'check'; },
        judge(s) { const d = D(s), k = Math.round(s.k), e = errK(d, k), ep = k > 1 ? errK(d, k - 1) : 1; const target = minK(d, 0.15); return { ok: e < 0.15 && ep >= 0.15, progress: LA.clamp01(1 - Math.abs(k - target) / 16), hint: e >= 0.15 ? 'Too few sheets: the error is above 15 %.' : undefined }; },
        hint: 'Go down until the error just crosses 15 %, then one back.',
        solution(s) { s.k = minK(D(s), 0.15); } },
      { kind: 'manipulate',
        prompt: 'Drag user 3 in the taste plane so their favourite film becomes film 1.',
        setup(s) { s.scene = 'recsys'; s.view = '2d'; s.cam = LA.cam.defaults2d(); s.U6 = LA.clone(U6); s.hotCell = [2, 0]; s.fillIn = 1; s.panel = 'check3'; },
        judge(s) { const u = s.U6[2], sc = V8.map(v => N.dot(u, v)); const best = sc[0], rest = Math.max.apply(null, sc.slice(1)); return { ok: best - rest > 0.1, progress: LA.clamp01((best - rest + 2) / 2.1) }; },
        hint: 'Film 1 sits at [1, 0]; move user 3 toward it.',
        solution(s) { s.U6[2] = [1.4, -0.2]; } },
    ],
  });
})();
