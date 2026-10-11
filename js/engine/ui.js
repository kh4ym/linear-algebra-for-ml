/* LA.ui — declarative playground controls bound to paths in the lesson state.
   build(container, controls, s, onChange) renders them; the returned object's refresh() re-reads s. */
window.LA = window.LA || {};
(function () {
  'use strict';
  const ui = {};
  const el = (tag, cls, html) => { const e = document.createElement(tag); if (cls) e.className = cls; if (html !== undefined) e.innerHTML = html; return e; };
  const roleClass = c => c ? LA.tex.colorClass(c) : '';

  /** Build all controls. Returns { refresh(), destroy() }. */
  ui.build = function (container, controls, s, onChange) {
    container.innerHTML = '';
    const items = [];
    (controls || []).forEach(def => {
      let item = null;
      try {
        switch (def.type) {
          case 'slider': item = ui.slider(def, s, onChange); break;
          case 'matrix': case 'vector': item = ui.matrixInput(def, s, onChange); break;
          case 'toggle': item = ui.toggle(def, s, onChange); break;
          case 'preset': case 'presets': item = ui.presets(def, s, onChange); break;
          case 'button': item = ui.button(def, s, onChange); break;
          case 'select': item = ui.select(def, s, onChange); break;
          case 'info': item = { el: el('div', 'hint-box', def.html || def.label || ''), refresh() { } }; break;
        }
      } catch (e) { item = { el: el('div', 'hint-box c-bad', 'control error: ' + (e && e.message)), refresh() { } }; }
      if (item) { container.appendChild(item.el); items.push(item); }
    });
    return { refresh() { items.forEach(it => { try { it.refresh(); } catch (_) { } }); }, destroy() { items.forEach(it => it.destroy && it.destroy()); container.innerHTML = ''; } };
  };

  /** {type:'slider', bind, min, max, step, label, color, fmt(fn), live(bool)} */
  ui.slider = function (def, s, onChange) {
    const wrap = el('div', 'ctl');
    const lab = el('div', 'ctl-label');
    const name = el('b', roleClass(def.color), def.label || def.bind);
    const val = el('span', 'val');
    lab.appendChild(name); lab.appendChild(val);
    const inp = document.createElement('input'); inp.type = 'range';
    inp.min = def.min === undefined ? -3 : def.min; inp.max = def.max === undefined ? 3 : def.max; inp.step = def.step === undefined ? 0.05 : def.step;
    const fmt = def.fmt || (x => LA.fmt(x));
    const read = () => { const x = LA.getPath(s, def.bind); if (typeof x === 'number') { inp.value = x; val.textContent = fmt(x); } };
    inp.addEventListener('input', () => { LA.setPath(s, def.bind, parseFloat(inp.value)); val.textContent = fmt(parseFloat(inp.value)); onChange('slider', def); });
    wrap.appendChild(lab); wrap.appendChild(inp);
    if (def.marks) { const row = el('div', 'ctl-row'); def.marks.forEach(m => { const b = el('button', 'btn small', m.label); b.addEventListener('click', () => { LA.setPath(s, def.bind, m.value); read(); onChange('slider', def); }); row.appendChild(b); }); wrap.appendChild(row); }
    read();
    return { el: wrap, refresh: read };
  };

  /** {type:'matrix', bind, label, colors:'cols'|'rows'|array, step, min, max, digits} — drag a number sideways to change it, double-click to type. */
  ui.matrixInput = function (def, s, onChange) {
    const wrap = el('div', 'ctl');
    if (def.label) { const lab = el('div', 'ctl-label'); lab.appendChild(el('b', roleClass(def.color), def.label)); wrap.appendChild(lab); }
    const box = el('div', 'matrix-input');
    const grid = el('div', 'mi-grid');
    box.appendChild(grid);
    const help = el('span', 'mi-help', def.help === undefined ? 'drag ↔ / double-click to type' : def.help);
    box.appendChild(help);
    wrap.appendChild(box);
    const cells = [];
    const getM = () => { const M = LA.getPath(s, def.bind); return Array.isArray(M[0]) ? M : M.map(x => [x]); };
    const isVec = () => !Array.isArray(LA.getPath(s, def.bind)[0]);
    const setVal = (i, j, v) => {
      if (def.min !== undefined) v = Math.max(def.min, v); if (def.max !== undefined) v = Math.min(def.max, v);
      const M = LA.getPath(s, def.bind);
      if (isVec()) M[i] = v; else M[i][j] = v;
      onChange('matrix', def);
    };
    const colorOf = (i, j) => {
      const c = def.colors === undefined ? 'cols' : def.colors;
      if (c === 'cols') return roleClass(['i', 'j', 'k', 'v'][j % 4]);
      if (c === 'rows') return roleClass(['i', 'j', 'k', 'v'][i % 4]);
      if (Array.isArray(c)) return roleClass(c[j % c.length]);
      return roleClass(c);
    };
    const build = () => {
      const M = getM(); const rows = M.length, cols = M[0].length;
      grid.style.gridTemplateColumns = 'repeat(' + cols + ', auto)';
      grid.innerHTML = ''; cells.length = 0;
      for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) {
        const c = el('span', 'mi-cell ' + colorOf(i, j), LA.fmt(M[i][j], def.digits));
        c.title = 'drag left/right, double-click to type, arrow keys to nudge'; c.tabIndex = 0;
        let drag = null;
        c.addEventListener('pointerdown', e => { if (c.classList.contains('editing')) return; drag = { x: e.clientX, v: getM()[i][j], moved: false }; c.setPointerCapture(e.pointerId); e.preventDefault(); });
        c.addEventListener('pointermove', e => {
          if (!drag) return; const dx = e.clientX - drag.x; if (Math.abs(dx) > 2) drag.moved = true;
          const step = def.step || 0.05, per = e.shiftKey ? 12 : 4;
          const v = Math.round((drag.v + dx / per * step) / step) * step;
          setVal(i, j, +v.toFixed(6)); c.textContent = LA.fmt(v, def.digits);
        });
        c.addEventListener('pointerup', () => { drag = null; });
        c.addEventListener('dblclick', () => edit(c, i, j));
        c.addEventListener('keydown', e => {
          const step = def.step || 0.05;
          if (e.key === 'ArrowUp' || e.key === 'ArrowRight') { setVal(i, j, +(getM()[i][j] + step).toFixed(6)); refresh(); e.preventDefault(); }
          else if (e.key === 'ArrowDown' || e.key === 'ArrowLeft') { setVal(i, j, +(getM()[i][j] - step).toFixed(6)); refresh(); e.preventDefault(); }
          else if (e.key === 'Enter') edit(c, i, j);
        });
        grid.appendChild(c); cells.push({ c, i, j });
      }
    };
    const edit = (c, i, j) => {
      c.classList.add('editing');
      const inp = document.createElement('input'); inp.type = 'text'; inp.value = String(+getM()[i][j].toFixed(4));
      c.textContent = ''; c.appendChild(inp); inp.focus(); inp.select();
      const done = (commit) => { if (commit) { const v = parseFloat(inp.value.replace('−', '-')); if (Number.isFinite(v)) setVal(i, j, v); } c.classList.remove('editing'); refresh(); };
      inp.addEventListener('blur', () => done(true));
      inp.addEventListener('keydown', e => { if (e.key === 'Enter') { e.preventDefault(); inp.blur(); } if (e.key === 'Escape') { inp.value = ''; done(false); } });
    };
    const refresh = () => {
      const M = getM();
      if (cells.length !== M.length * M[0].length) { build(); return; }
      cells.forEach(({ c, i, j }) => { if (!c.classList.contains('editing')) c.textContent = LA.fmt(M[i][j], def.digits); });
    };
    build();
    return { el: wrap, refresh };
  };

  /** {type:'toggle', bind, label} */
  ui.toggle = function (def, s, onChange) {
    const lab = el('label', 'toggle');
    const inp = document.createElement('input'); inp.type = 'checkbox';
    lab.appendChild(inp); lab.appendChild(el('span', '', def.label || def.bind));
    const read = () => { inp.checked = !!LA.getPath(s, def.bind); };
    inp.addEventListener('change', () => { LA.setPath(s, def.bind, inp.checked); onChange('toggle', def); });
    read();
    return { el: lab, refresh: read };
  };

  /** {type:'preset', label, items:[{label, apply(s) | values:{path:value}}]} */
  ui.presets = function (def, s, onChange) {
    const wrap = el('div', 'ctl');
    if (def.label) { const lab = el('div', 'ctl-label'); lab.appendChild(el('b', '', def.label)); wrap.appendChild(lab); }
    const row = el('div', 'ctl-row');
    (def.items || []).forEach(it => {
      const b = el('button', 'btn small', it.label);
      b.addEventListener('click', () => {
        if (typeof it.apply === 'function') it.apply(s);
        if (it.values) Object.keys(it.values).forEach(p => LA.setPath(s, p, LA.clone(it.values[p])));
        row.querySelectorAll('.btn').forEach(x => x.classList.remove('active')); b.classList.add('active');
        onChange('preset', def, it);
      });
      row.appendChild(b);
    });
    wrap.appendChild(row);
    return { el: wrap, refresh() { } };
  };

  /** {type:'button', label, onClick(s), primary} */
  ui.button = function (def, s, onChange) {
    const b = el('button', 'btn' + (def.primary ? ' primary' : ''), def.label);
    b.addEventListener('click', () => { if (def.onClick) def.onClick(s); onChange('button', def); });
    return { el: b, refresh() { } };
  };

  /** {type:'select', bind, label, options:[{label, value}]} */
  ui.select = function (def, s, onChange) {
    const wrap = el('div', 'ctl');
    const lab = el('div', 'ctl-label'); lab.appendChild(el('b', roleClass(def.color), def.label || def.bind)); wrap.appendChild(lab);
    const sel = document.createElement('select'); sel.className = 'btn';
    (def.options || []).forEach((o, i) => { const op = document.createElement('option'); op.value = String(i); op.textContent = o.label; sel.appendChild(op); });
    const read = () => { const v = LA.getPath(s, def.bind); const idx = (def.options || []).findIndex(o => JSON.stringify(o.value) === JSON.stringify(v)); if (idx >= 0) sel.value = String(idx); };
    sel.addEventListener('change', () => { const o = def.options[parseInt(sel.value, 10)]; LA.setPath(s, def.bind, LA.clone(o.value)); onChange('select', def); });
    wrap.appendChild(sel); read();
    return { el: wrap, refresh: read };
  };

  LA.ui = ui;
})();
