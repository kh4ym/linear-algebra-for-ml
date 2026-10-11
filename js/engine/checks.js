/* LA.checks — "check yourself" items judged by manipulating the picture (kind 'manipulate') or by entering a
   number / matrix that drives the picture live (kind 'entry'). Shared tolerances in LA.TOL. A check passes when
   judge(s).ok holds for 400 ms; near misses show a hint; "Show me" animates the solution (does not count as a pass). */
window.LA = window.LA || {};
(function () {
  'use strict';

  LA.TOL = { angleDeg: 4, pos: 0.12, scalar: 0.05, entry: 0.05, det: 0.08, unitNorm: 0.04 };
  /** Convenience judges for lesson authors. */
  LA.judge = {
    near(a, b, tol) { tol = tol === undefined ? LA.TOL.pos : tol; const d = Math.hypot.apply(null, a.map((x, i) => x - (b[i] || 0))); return { ok: d <= tol, progress: Math.max(0, 1 - d / (tol * 6)), dist: d }; },
    scalar(x, target, tol) { tol = tol === undefined ? LA.TOL.scalar : tol; const d = Math.abs(x - target); return { ok: d <= tol, progress: Math.max(0, 1 - d / (tol * 8)), dist: d }; },
    angleDeg(a, b, targetDeg, tol) {
      tol = tol === undefined ? LA.TOL.angleDeg : tol;
      const na = Math.hypot.apply(null, a) || 1, nb = Math.hypot.apply(null, b) || 1;
      let dot = 0; for (let i = 0; i < a.length; i++) dot += a[i] * (b[i] || 0);
      const ang = Math.acos(Math.max(-1, Math.min(1, dot / (na * nb)))) * 180 / Math.PI;
      const d = Math.abs(ang - targetDeg);
      return { ok: d <= tol, progress: Math.max(0, 1 - d / 60), dist: d, angle: ang };
    },
    matrix(M, target, tol) {
      tol = tol === undefined ? LA.TOL.entry : tol; let worst = 0;
      for (let i = 0; i < target.length; i++) for (let j = 0; j < target[i].length; j++) worst = Math.max(worst, Math.abs((M[i] && M[i][j] !== undefined ? M[i][j] : NaN) - target[i][j]));
      if (!Number.isFinite(worst)) worst = 99;
      return { ok: worst <= tol, progress: Math.max(0, 1 - worst / (tol * 10)), dist: worst };
    },
    /** camera looks along ±dir (3D). tolDeg defaults to 2·angleDeg. */
    lookAlong(cam, dir, tolDeg) {
      tolDeg = tolDeg === undefined ? 2 * LA.TOL.angleDeg : tolDeg;
      const f = LA.cam.forward(cam); const n = Math.hypot(dir[0], dir[1], dir[2]) || 1;
      const dot = Math.abs((f[0] * dir[0] + f[1] * dir[1] + f[2] * dir[2]) / n);
      const ang = Math.acos(Math.max(-1, Math.min(1, dot))) * 180 / Math.PI;
      return { ok: ang <= tolDeg, progress: Math.max(0, 1 - ang / 90), dist: ang };
    },
    /** combine several partial judgements: all must be ok; progress = mean */
    all() { const rs = Array.prototype.slice.call(arguments); return { ok: rs.every(r => r.ok), progress: rs.reduce((a, r) => a + (r.progress || 0), 0) / rs.length }; },
  };

  const checks = { active: null, index: -1, okSince: 0, passed: {}, shown: false, dom: null, player: null };

  /** Enter check mode: list the lesson's checks in `container`. */
  checks.begin = function (player, container) {
    checks.player = player; checks.dom = container; checks.active = null; checks.index = -1; checks.shown = false;
    const L = player.lesson;
    checks.passed = (LA.store && LA.store.lesson(L.id).checks) || {};
    checks.renderList();
    const list = L.checks || [];
    const first = list.findIndex((c, i) => !checks.passed[i]);
    if (list.length) checks.select(first >= 0 ? first : 0);
  };
  checks.stop = function () { checks.active = null; checks.index = -1; if (checks.dom) checks.dom.innerHTML = ''; };

  checks.renderList = function () {
    const c = checks.dom; if (!c) return;
    const L = checks.player.lesson, list = L.checks || [];
    c.innerHTML = '';
    if (!list.length) { c.innerHTML = '<div class="hint-box">This lesson has no checks yet.</div>'; return; }
    const ul = document.createElement('div'); ul.className = 'check-list';
    list.forEach((ck, i) => {
      const it = document.createElement('div'); it.className = 'check-item' + (checks.passed[i] ? ' passed' : '') + (i === checks.index ? ' active' : '');
      it.innerHTML = '<span class="mark">' + (checks.passed[i] ? '✓' : (i + 1)) + '</span><span>' + ck.prompt + '</span>';
      it.addEventListener('click', () => checks.select(i));
      ul.appendChild(it);
    });
    c.appendChild(ul);
    const box = document.createElement('div'); box.className = 'check-box'; box.id = 'check-box'; c.appendChild(box);
  };

  checks.select = function (i) {
    const p = checks.player, L = p.lesson, ck = (L.checks || [])[i]; if (!ck) return;
    checks.index = i; checks.active = ck; checks.okSince = 0; checks.shown = false; checks.justPassed = false;
    LA.assignInPlace(p.s, p.initial); p.s.cam = p.s.cam || {}; p.s.cam.auto = true;
    try { if (ck.setup) ck.setup(p.s); } catch (e) { LA.app && LA.app.reportError(e, L.id + ' → check.setup'); }
    checks.renderList();
    checks.renderBox();
    LA.canvas.invalidate();
  };

  checks.renderBox = function () {
    const box = document.getElementById('check-box'); if (!box) return;
    const p = checks.player, ck = checks.active; if (!ck) { box.innerHTML = ''; return; }
    box.innerHTML = '';
    const prompt = document.createElement('div'); prompt.className = 'check-prompt'; prompt.innerHTML = ck.prompt; box.appendChild(prompt);
    if (ck.kind === 'entry') {
      const holder = document.createElement('div'); box.appendChild(holder);
      const def = { type: 'matrix', bind: ck.bind, label: ck.label || '', colors: ck.colors === undefined ? 'cols' : ck.colors, step: ck.step || 0.05, digits: 2 };
      checks.entryUI = LA.ui.build(holder, [def], p.s, () => { LA.canvas.invalidate(); });
    }
    const status = document.createElement('div'); status.className = 'check-status'; status.innerHTML = '<div class="bar"><i></i></div><span class="label">try it</span>'; box.appendChild(status);
    checks.statusEl = status;
    const hint = document.createElement('div'); hint.className = 'hint-box hidden'; box.appendChild(hint); checks.hintEl = hint;
    const row = document.createElement('div'); row.className = 'ctl-row'; row.style.marginTop = '10px';
    const bReset = btn('Reset', () => checks.select(checks.index));
    const bShow = btn('Show me', () => checks.showMe());
    const bNext = btn('Next check →', () => { const n = (p.lesson.checks || []).length; if (checks.index < n - 1) checks.select(checks.index + 1); });
    row.appendChild(bReset); row.appendChild(bShow); row.appendChild(bNext);
    box.appendChild(row);
    if (checks.passed[checks.index]) { const d = document.createElement('div'); d.className = 'check-pass'; d.textContent = 'Passed ✓'; box.appendChild(d); }
  };
  function btn(label, fn) { const b = document.createElement('button'); b.className = 'btn small'; b.textContent = label; b.addEventListener('click', fn); return b; }

  checks.showMe = function () {
    const p = checks.player, ck = checks.active; if (!ck) return;
    checks.shown = true;
    try { if (ck.solution) ck.solution(p.s); } catch (e) { LA.app && LA.app.reportError(e, p.lesson.id + ' → check.solution'); }
    if (checks.entryUI) checks.entryUI.refresh();
    p._dirtyControls = true;
    LA.canvas.invalidate();
  };

  /** Called each frame while in check mode. Returns true if it needs more frames (glow timing). */
  checks.tick = function (p, dt) {
    const ck = checks.active; if (!ck) return false;
    let r;
    try { r = checks.judgeOf(ck, p.s); } catch (e) { r = { ok: false, progress: 0, hint: 'judge error: ' + e.message }; }
    const st = checks.statusEl; if (!st) return false;
    const bar = st.querySelector('.bar > i'), lab = st.querySelector('.label');
    const prog = Math.max(0, Math.min(1, r.ok ? 1 : (r.progress || 0)));
    bar.style.width = (prog * 100).toFixed(0) + '%';
    st.classList.toggle('ok', !!r.ok);
    if (r.ok) {
      if (!checks.okSince) checks.okSince = performance.now();
      const held = performance.now() - checks.okSince;
      lab.textContent = checks.shown ? 'solution shown' : (held >= 400 ? 'passed ✓' : 'hold…');
      if (held >= 400 && !checks.shown && !checks.passed[checks.index]) {
        checks.passed[checks.index] = true;
        if (LA.store) LA.store.markCheck(p.lesson.id, checks.index);
        checks.renderList(); checks.renderBox();
        if (LA.app && LA.app.onProgress) LA.app.onProgress();
      }
      if (checks.hintEl) checks.hintEl.classList.add('hidden');
      return held < 450;
    }
    checks.okSince = 0;
    lab.textContent = prog > 0.7 ? 'almost' : prog > 0.3 ? 'closer' : 'try it';
    if (checks.hintEl) {
      const h = r.hint || (prog > 0.7 ? (ck.hint || '') : '');
      if (h) { checks.hintEl.innerHTML = h; checks.hintEl.classList.remove('hidden'); } else checks.hintEl.classList.add('hidden');
    }
    return false;
  };

  checks.judgeOf = function (ck, s) {
    if (ck.kind === 'entry') {
      const M = LA.getPath(s, ck.bind);
      const target = ck.answer;
      const tol = ck.tol === undefined ? LA.TOL.entry : ck.tol;
      if (Array.isArray(target[0])) return LA.judge.matrix(M, target, tol);
      if (Array.isArray(target)) return LA.judge.matrix(M.map(x => [x]), target.map(x => [x]), tol);
      return LA.judge.scalar(M, target, tol);
    }
    const r = ck.judge ? ck.judge(s) : { ok: false, progress: 0 };
    if (typeof r === 'boolean') return { ok: r, progress: r ? 1 : 0 };
    return r || { ok: false, progress: 0 };
  };

  LA.checks = checks;
})();
