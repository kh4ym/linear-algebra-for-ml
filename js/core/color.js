/* LA.color — single source of truth for every color used on canvas and in HTML.
   applyCssVars() writes the same values to CSS custom properties at boot so notation and pictures never drift. */
window.LA = window.LA || {};
(function () {
  'use strict';

  const base = {
    // surfaces & text
    bg: '#0b0e14', panel: '#11161f', panel2: '#182030', line: '#242d3d', text: '#e8ebf3', muted: '#98a2b8',
    // canvas furniture
    grid: '#3a4f7c', gridAxis: '#93a6cc', ghost: '#2a3140', ghostAxis: '#3b4458', tick: '#6b7690', hud: '#c7cfe0',
    // roles (the pedagogical contract)
    i: '#83c167', j: '#fc6255', k: '#58c4dd',          // basis vectors / matrix columns
    v: '#ffd166', w: '#c77dff', b: '#ff9f43',          // generic vectors
    gold: '#f6c945',                                   // answers: eigen/singular/PC/fit
    mag: '#ff5fa2',                                    // projections, null space
    teal: '#2ec4b6',                                   // subspaces (span, column space)
    neg: '#8e6cff',                                    // negative orientation (det < 0)
    residual: '#8b93a7',                               // dashed residuals
    ok: '#58d68d', bad: '#ff5c5c', white: '#ffffff', accent: '#58c4dd',
    heatLo: '#1b2a4a', heatHi: '#ffd166',              // heatmap ramp ends
    data: '#58c4dd', dataB: '#ff7eb6', dataC: '#b5e36b', // data classes (sky blue / rose / lime)
  };
  base.gridGhost = base.ghost;
  base['null'] = base.mag;          // null space role (same hue as projections, per STYLE.md)

  const highContrast = Object.assign({}, base, {
    i: '#00e676', j: '#ff2d2d', k: '#40c4ff', v: '#ffff00', w: '#ea80fc', b: '#ffab40',
    gold: '#ffe600', mag: '#ff4dbd', 'null': '#ff4dbd', teal: '#00ffd0', neg: '#b388ff', grid: '#4a5a78', text: '#ffffff',
  });

  const color = Object.assign({}, base);
  color.palettes = { base, highContrast };
  color.mode = 'base';

  /** Switch palette ('base' | 'highContrast') and rewrite CSS variables. */
  color.setPalette = function (name) {
    const p = color.palettes[name] || base;
    color.mode = color.palettes[name] ? name : 'base';
    Object.keys(base).forEach(k => { color[k] = p[k]; });
    color.applyCssVars();
  };

  /** Write every token as --c-<name> on :root. */
  color.applyCssVars = function () {
    const root = document.documentElement;
    Object.keys(base).forEach(k => root.style.setProperty('--c-' + k, color[k]));
  };

  /** Hex color + alpha → rgba() string. Accepts #rgb / #rrggbb / rgba() passthrough. */
  color.alpha = function (hex, a) {
    if (!hex) return 'rgba(255,255,255,' + a + ')';
    if (hex[0] !== '#') {
      const m = /^rgba?\(([^)]+)\)$/.exec(hex);
      if (m) { const parts = m[1].split(',').map(s => s.trim()); return 'rgba(' + parts[0] + ',' + parts[1] + ',' + parts[2] + ',' + a + ')'; }
      return hex;
    }
    const rgb = color.rgb(hex);
    return 'rgba(' + rgb[0] + ',' + rgb[1] + ',' + rgb[2] + ',' + a + ')';
  };

  /** '#rrggbb' → [r,g,b]. */
  color.rgb = function (hex) {
    let h = hex.replace('#', '');
    if (h.length === 3) h = h.split('').map(c => c + c).join('');
    const n = parseInt(h, 16);
    return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
  };

  /** Linear mix of two hex colors, t in [0,1] → '#rrggbb'. */
  color.mix = function (a, b, t) {
    const A = color.rgb(a), B = color.rgb(b);
    const c = [0, 1, 2].map(i => Math.round(A[i] + (B[i] - A[i]) * Math.max(0, Math.min(1, t))));
    return '#' + c.map(x => x.toString(16).padStart(2, '0')).join('');
  };

  /** Heatmap ramp: value in [0,1] → color (dark blue → yellow via teal). */
  color.heat = function (t) {
    t = Math.max(0, Math.min(1, t));
    if (t < 0.5) return color.mix(color.heatLo, color.teal, t * 2);
    return color.mix(color.teal, color.heatHi, (t - 0.5) * 2);
  };

  /** Diverging ramp: value in [-1,1] → magenta … dark … teal. */
  color.diverge = function (t) {
    t = Math.max(-1, Math.min(1, t));
    return t < 0 ? color.mix('#1a1d26', color.mag, -t) : color.mix('#1a1d26', color.teal, t);
  };

  /** Color for the j-th basis direction / matrix column (0 → î, 1 → ĵ, 2 → k̂, then cycles through accents). */
  color.basis = function (j) {
    return [color.i, color.j, color.k, color.v, color.w, color.b][j % 6];
  };

  /** Resolve a role name ('i', 'gold', …) or a literal color string. */
  color.role = function (name) {
    return (typeof name === 'string' && color[name] && name !== 'palettes') ? color[name] : name;
  };

  LA.color = color;
})();
