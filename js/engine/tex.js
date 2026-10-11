/* LA.tex — HTML math notation with per-entry colored spans that update live.
   Output is plain HTML strings; the player reconciles them into the math panel every frame by comparing the
   structural skeleton (so only changed numbers are rewritten, never the whole DOM). */
window.LA = window.LA || {};
(function () {
  'use strict';
  const esc = s => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  const ROLE_CLASS = { i: 'c-i', j: 'c-j', k: 'c-k', v: 'c-v', w: 'c-w', b: 'c-b', gold: 'c-gold', mag: 'c-mag', teal: 'c-teal', null: 'c-null', neg: 'c-neg', ok: 'c-ok', bad: 'c-bad', muted: 'c-muted', data: 'c-data', dataB: 'c-dataB', dataC: 'c-dataC', text: '' };
  const cls = c => c ? (ROLE_CLASS[c] !== undefined ? ROLE_CLASS[c] : (c.startsWith('c-') ? c : '')) : '';

  const tex = {};
  tex.colorClass = cls;

  /** A live number. opts: color, digits, key */
  tex.num = function (x, opts) {
    opts = opts || {};
    const v = typeof x === 'number' ? LA.fmt(x, opts.digits) : esc(x);
    return '<span class="tx-v ' + cls(opts.color) + '"' + (opts.key ? ' data-k="' + esc(opts.key) + '"' : '') + '>' + v + '</span>';
  };
  /** Italic symbol with optional hat / subscript / superscript. opts: color, hat, sub, sup, bold */
  tex.sym = function (name, opts) {
    opts = opts || {};
    let inner = esc(name);
    if (opts.hat) inner = '<span class="tx-hat">' + inner + '</span>';
    else inner = '<span class="tx-sym">' + inner + '</span>';
    if (opts.sub !== undefined) inner += '<sub>' + esc(opts.sub) + '</sub>';
    if (opts.sup !== undefined) inner += '<sup>' + esc(opts.sup) + '</sup>';
    return '<span class="' + cls(opts.color) + (opts.bold ? ' tx-bold' : '') + '">' + inner + '</span>';
  };
  /** Operator / punctuation (=, +, −, ·, ×, →, (, ) …). */
  tex.op = function (s) { return '<span class="tx-op">' + esc(s) + '</span>'; };
  /** Plain words inside an equation. */
  tex.text = function (s, opts) { opts = opts || {}; const c = cls(opts.color); return '<span class="' + (c ? 'tx-word ' + c : 'tx-label') + '">' + (opts.html ? s : esc(s)) + '</span>'; };
  /** Dimension annotation like 2×3. */
  tex.dim = function (r, c) { return '<span class="tx-dim">' + r + '×' + c + '</span>'; };
  /** Matrix. opts: colors 'cols' | 'rows' | array of role per column | fn(i,j)→role, digits, key, highlight [i,j] or fn(i,j)→bool, dim(bool) */
  tex.mat = function (M, opts) {
    opts = opts || {};
    const rows = M.length, cols = Array.isArray(M[0]) ? M[0].length : 1;
    const colorAt = (i, j) => {
      const c = opts.colors;
      if (!c) return '';
      if (c === 'cols') return cls(LA_basisRole(j));
      if (c === 'rows') return cls(LA_basisRole(i));
      if (typeof c === 'function') return cls(c(i, j));
      if (Array.isArray(c)) return cls(c[j % c.length]);
      return cls(c);
    };
    const hl = (i, j) => opts.highlight ? (typeof opts.highlight === 'function' ? opts.highlight(i, j) : (opts.highlight[0] === i && opts.highlight[1] === j)) : false;
    let h = '<span class="tx-mat"><span class="tx-grid" style="grid-template-columns:repeat(' + cols + ',auto)">';
    for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) {
      const x = Array.isArray(M[i]) ? M[i][j] : M[i];
      const v = typeof x === 'number' ? LA.fmt(x, opts.digits) : (x && x.html ? x.html : esc(x));
      h += '<span class="tx-cell tx-v ' + colorAt(i, j) + (hl(i, j) ? ' tx-glow' : '') + '"' + (opts.key ? ' data-k="' + esc(opts.key) + '.' + i + '.' + j + '"' : '') + '>' + v + '</span>';
    }
    h += '</span></span>';
    if (opts.dim) h += tex.dim(rows, cols);
    return h;
  };
  /** Column vector. opts: color (single role) or colors 'rows', digits, key */
  tex.vec = function (v, opts) {
    opts = opts || {};
    const o = Object.assign({}, opts);
    if (opts.color && !opts.colors) o.colors = opts.color;
    return tex.mat(v.map(x => [x]), o);
  };
  /** Row vector [a b c]. */
  tex.rowvec = function (v, opts) { opts = opts || {}; const o = Object.assign({}, opts); if (opts.color && !opts.colors) o.colors = opts.color; return tex.mat([v.slice()], o); };
  /** Inline equation: joins HTML parts. */
  tex.eq = function () { return '<span class="tx">' + Array.prototype.slice.call(arguments).join('') + '</span>'; };
  /** Several equations / lines, wrapping. */
  tex.block = function () { return '<div class="tx-block">' + Array.prototype.slice.call(arguments).join('') + '</div>'; };
  /** Stack lines vertically. */
  tex.lines = function () { return Array.prototype.slice.call(arguments).map(l => '<div class="tx-block">' + l + '</div>').join(''); };
  tex.frac = function (a, b) { return '<span class="tx-frac"><span>' + a + '</span><span>' + b + '</span></span>'; };
  tex.sum = function (expr, from, to) { return '<span class="tx-sum"><span class="lim">' + (to || '') + '</span><span class="sig">∑</span><span class="lim">' + (from || '') + '</span></span>' + expr; };
  tex.sqrt = function (x) { return '<span class="tx-op">√</span><span style="border-top:1.3px solid currentColor;padding:0 2px">' + x + '</span>'; };
  tex.norm = function (x) { return tex.op('‖') + x + tex.op('‖'); };
  tex.abs = function (x) { return tex.op('|') + x + tex.op('|'); };
  tex.paren = function (x) { return tex.op('(') + x + tex.op(')'); };
  // wrap base + superscript in one inline span so a flex parent cannot split them apart
  tex.T = function (x) { return '<span class="tx-sup">' + x + '<sup>T</sup></span>'; };
  tex.inv = function (x) { return '<span class="tx-sup">' + x + '<sup>−1</sup></span>'; };
  /** Dot product of two vectors shown as a colored sum of products: a₁b₁ + a₂b₂ = result */
  tex.dotExpanded = function (a, b, opts) {
    opts = opts || {};
    const parts = [];
    for (let i = 0; i < a.length; i++) {
      parts.push(tex.num(a[i], { color: opts.colorA || 'v' }) + tex.op('·') + tex.num(b[i], { color: opts.colorB || 'w' }));
    }
    let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i];
    return tex.eq(parts.join(tex.op(' + ')), tex.op(' = '), tex.num(s, { color: opts.colorResult || 'gold' }));
  };
  /** Standard colored basis labels. */
  tex.ihat = () => tex.sym('i', { hat: true, color: 'i' });
  tex.jhat = () => tex.sym('j', { hat: true, color: 'j' });
  tex.khat = () => tex.sym('k', { hat: true, color: 'k' });

  /** Reconcile container with new html: rebuild only when the structure changed; otherwise update values in place. */
  tex.reconcile = function (container, html) {
    html = html || '';
    // value spans = any span whose class list contains tx-v and whose content is plain text (no child tags)
    const skeleton = html.replace(/(<span class="[^"]*\btx-v\b[^"]*"[^>]*>)[^<]*(<\/span>)/g, '$1$2');
    if (container._skeleton !== skeleton) {
      container.innerHTML = html;
      container._skeleton = skeleton;
      container._vals = Array.from(container.querySelectorAll('.tx-v')).filter(e => e.children.length === 0);
      return true;
    }
    const vals = container._vals || [];
    const re = /<span class="[^"]*\btx-v\b[^"]*"[^>]*>([^<]*)<\/span>/g;
    let m, i = 0;
    while ((m = re.exec(html)) !== null && i < vals.length) {
      const el = vals[i++];
      const text = m[1].replace(/&amp;/g, '&').replace(/&lt;/g, '<').replace(/&gt;/g, '>');
      if (el.textContent !== text) el.textContent = text;
    }
    return false;
  };

  function LA_basisRole(j) { return ['i', 'j', 'k', 'v', 'w', 'b'][j % 6]; }
  LA.tex = tex;
})();
