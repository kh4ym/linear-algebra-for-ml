/* LA.fmt — number formatting shared by canvas labels and HTML notation.
   Integers print without decimals, everything else with two; negatives use the real minus sign (U+2212). */
window.LA = window.LA || {};
(function () {
  'use strict';
  const MINUS = '−';

  function fmt(x, digits) {
    if (x === null || x === undefined || Number.isNaN(x)) return '?';
    if (!Number.isFinite(x)) return x > 0 ? '∞' : MINUS + '∞';
    if (digits === undefined) digits = 2;
    let s;
    if (Math.abs(x - Math.round(x)) < 1e-9) s = String(Math.round(x));
    else s = x.toFixed(digits);
    if (s === '-0') s = '0';
    return s[0] === '-' ? MINUS + s.slice(1) : s;
  }
  /** Always `digits` decimals (for tabular columns). */
  fmt.fixed = function (x, digits) {
    if (!Number.isFinite(x)) return fmt(x);
    let s = x.toFixed(digits === undefined ? 2 : digits);
    if (/^-0(\.0+)?$/.test(s)) s = s.slice(1);
    return s[0] === '-' ? MINUS + s.slice(1) : s;
  };
  /** Vector → "(1, 2)" */
  fmt.vec = function (v, digits) { return '(' + v.map(x => fmt(x, digits)).join(', ') + ')'; };
  /** Radians → "45°" */
  fmt.deg = function (rad, digits) { return fmt(rad * 180 / Math.PI, digits === undefined ? 0 : digits) + '°'; };
  /** Fraction → "37%" */
  fmt.pct = function (x, digits) { return fmt(x * 100, digits === undefined ? 0 : digits) + '%'; };
  /** Signed with explicit + for positives (used in sums like 2·3 + 1·4). */
  fmt.signed = function (x, digits) { const s = fmt(Math.abs(x), digits); return (x < 0 ? MINUS : '+') + s; };
  fmt.MINUS = MINUS;
  LA.fmt = fmt;
})();
