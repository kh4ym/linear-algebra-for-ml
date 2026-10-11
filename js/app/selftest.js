/* LA.selftest — #/selftest runs every lesson offscreen: validator, every step at t = 0, 0.5, 1 (setup/at/render),
   every check (setup → judge must be false → solution → judge must be true), playground bindings, math panel,
   numeric self-checks, manifest diff. Results are printed as a table and as machine-readable JSON
   (#selftest-json, body[data-selftest="done"], #selftest-summary[data-status]). ?only=<id> filters; ?gallery=1
   renders every step as a thumbnail for one-screenshot review. */
window.LA = window.LA || {};
(function () {
  'use strict';
  const el = (tag, cls, html) => { const e = document.createElement(tag); if (cls) e.className = cls; if (html !== undefined) e.innerHTML = html; return e; };
  const esc = s => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  const selftest = {};

  function errStr(e) { return e && e.stack ? String(e.stack).split('\n').slice(0, 2).join(' | ') : String(e); }

  /** Test one lesson synchronously. Returns a result object. */
  selftest.testLesson = function (L, opts) {
    opts = opts || {};
    const res = { id: L.id, title: L.title, violations: L._violations || [], steps: [], checks: [], playground: null, math: null, ms: 0, pass: true, records: 0 };
    const t0 = performance.now();
    const off = LA.canvas.offscreen(opts.w || 320, opts.h || 200, 1);
    const g = new LA.Gfx();
    let s;
    try { s = LA.clone(L.state()); } catch (e) { res.steps.push({ k: -1, ok: false, error: 'state(): ' + errStr(e) }); res.pass = false; return res; }
    const initial = LA.clone(s);
    const ctx = { s, mode: 'learn', step: 0, t: 1, size: off.size, dpr: 1, hot: null, time: 0, instant: true };
    (L.steps || []).forEach((st, k) => {
      const r = { k, ok: true, error: null, words: 0 };
      try {
        s.cam = s.cam || {}; s.cam.auto = true;
        if (st.setup) st.setup(s);
        for (const t of [0, 0.5, 1]) {
          if (st.at) st.at(s, t);
          g.handlesEnabled = !!st.interactive;
          g.begin(off.ctx, off.size, s, { instant: true });
          L.render(g, s, Object.assign(ctx, { step: k, t }));
          const stats = g.end();
          res.records = Math.max(res.records, stats.records);
          if (L.math) L.math(s, LA.tex);
        }
        let n = st.narration; if (typeof n === 'function') n = n(s);
        r.words = String(n || '').replace(/<[^>]+>/g, ' ').split(/\s+/).filter(Boolean).length;
        if (opts.gallery) { const thumb = LA.canvas.offscreen(300, 190, 1); const g2 = new LA.Gfx(); g2.begin(thumb.ctx, thumb.size, s, { instant: true }); L.render(g2, s, Object.assign({}, ctx, { size: thumb.size })); g2.end(); r.thumb = thumb.el; }
      } catch (e) { r.ok = false; r.error = errStr(e); res.pass = false; }
      res.steps.push(r);
    });
    (L.checks || []).forEach((ck, i) => {
      const r = { i, ok: true, error: null, note: '' };
      try {
        const s2 = LA.clone(initial); s2.cam = s2.cam || {}; s2.cam.auto = true;
        if (ck.setup) ck.setup(s2);
        const before = LA.checks.judgeOf(ck, s2);
        if (before.ok) r.note = 'already satisfied before solving (check too easy?)';
        if (ck.kind === 'entry') LA.setPath(s2, ck.bind, LA.clone(ck.answer)); else if (ck.solution) ck.solution(s2);
        const after = LA.checks.judgeOf(ck, s2);
        if (!after.ok) { r.ok = false; r.error = 'solution does not satisfy judge (progress ' + (after.progress || 0).toFixed(2) + ')'; res.pass = false; }
        g.handlesEnabled = true; g.begin(off.ctx, off.size, s2, { instant: true }); L.render(g, s2, Object.assign({}, ctx, { s: s2, mode: 'check' })); g.end();
      } catch (e) { r.ok = false; r.error = errStr(e); res.pass = false; }
      res.checks.push(r);
    });
    if (L.playground) {
      const r = { ok: true, error: null, controls: (L.playground.controls || []).length };
      try {
        const s3 = LA.clone(initial); s3.cam = s3.cam || {}; s3.cam.auto = true;
        if (L.playground.setup) L.playground.setup(s3);
        (L.playground.controls || []).forEach((c, i) => { if (c.bind !== undefined && LA.getPath(s3, c.bind) === undefined) throw new Error('control ' + i + ' binds to missing path ' + c.bind); });
        g.handlesEnabled = true; g.begin(off.ctx, off.size, s3, { instant: true }); L.render(g, s3, Object.assign({}, ctx, { s: s3, mode: 'play' })); g.end();
        if (L.math) L.math(s3, LA.tex);
      } catch (e) { r.ok = false; r.error = errStr(e); res.pass = false; }
      res.playground = r;
    }
    if (res.violations.length) res.pass = false;
    res.ms = Math.round(performance.now() - t0);
    return res;
  };

  selftest.run = function (query, container) {
    query = query || {};
    const gallery = query.gallery === '1';
    const only = query.only;
    container.innerHTML = '<h1>Self-test</h1><p class="lead">Running every lesson offscreen…</p>';
    const ordered = LA.lessons.ordered();
    // ?only= accepts a full id or a prefix (e.g. "p3-06" or "p3")
    const specs = ordered.filter(e => e.spec && (!only || e.id === only || e.id.startsWith(only + '-') || e.id.startsWith(only))).map(e => e.spec);
    const results = []; let i = 0;
    const numRes = (LA.num && LA.num.selfTest) ? safeNum() : [{ name: 'LA.num.selfTest missing', pass: false }];
    const manifestMissing = ordered.filter(e => !e.spec).map(e => e.id);
    const loadErrors = LA.lessons.loadErrors.slice();
    const unlisted = Object.keys(LA.lessons.byId).filter(id => !ordered.find(e => e.id === id));
    const step = () => {
      const t0 = performance.now();
      while (i < specs.length && performance.now() - t0 < 60) results.push(selftest.testLesson(specs[i++], { gallery }));
      if (i < specs.length) { container.querySelector('.lead').textContent = 'Testing ' + i + ' / ' + specs.length + '…'; setTimeout(step, 0); return; }
      selftest.report(container, results, numRes, { manifestMissing, unlisted, loadErrors, gallery, only });
    };
    setTimeout(step, 0);
  };
  function safeNum() { try { return LA.num.selfTest(); } catch (e) { return [{ name: 'LA.num.selfTest threw', pass: false, detail: errStr(e) }]; } }

  selftest.report = function (container, results, numRes, info) {
    const lessonsPass = results.filter(r => r.pass).length;
    const numPass = numRes.filter(r => r.pass).length;
    const allPass = lessonsPass === results.length && numPass === numRes.length && info.loadErrors.length === 0 && info.unlisted.length === 0;
    container.innerHTML = '';
    container.appendChild(el('h1', '', 'Self-test ' + (allPass ? '<span class="ok">PASS</span>' : '<span class="fail">FAIL</span>')));
    const sum = el('p', 'lead', 'Lessons: ' + lessonsPass + ' / ' + results.length + ' pass · numeric checks: ' + numPass + ' / ' + numRes.length + ' pass · planned (not built): ' + info.manifestMissing.length + (info.only ? ' · filter: ' + esc(info.only) : ''));
    sum.id = 'selftest-summary'; sum.dataset.status = allPass ? 'pass' : 'fail';
    sum.dataset.lessons = lessonsPass + '/' + results.length; sum.dataset.num = numPass + '/' + numRes.length;
    container.appendChild(sum);
    if (info.loadErrors.length) container.appendChild(el('div', 'hint-box fail', '<b>Load errors</b><br>' + info.loadErrors.map(esc).join('<br>')));
    if (info.unlisted.length) container.appendChild(el('div', 'hint-box fail', '<b>Registered but not in manifest</b>: ' + info.unlisted.map(esc).join(', ')));
    const table = el('table', 'plain');
    let rows = '<tr><th>Lesson</th><th>Status</th><th>Steps</th><th>Checks</th><th>Playground</th><th>Spec</th><th>ms</th><th>Details</th></tr>';
    results.forEach(r => {
      const stepsOk = r.steps.filter(s => s.ok).length, checksOk = r.checks.filter(c => c.ok).length;
      const details = [];
      r.steps.filter(s => !s.ok).forEach(s => details.push('step ' + (s.k + 1) + ': ' + esc(s.error)));
      r.checks.filter(c => !c.ok).forEach(c => details.push('check ' + (c.i + 1) + ': ' + esc(c.error)));
      r.checks.filter(c => c.ok && c.note).forEach(c => details.push('<span class="warn">check ' + (c.i + 1) + ': ' + esc(c.note) + '</span>'));
      if (r.playground && !r.playground.ok) details.push('playground: ' + esc(r.playground.error));
      r.violations.forEach(v => details.push('<span class="warn">spec: ' + esc(v) + '</span>'));
      if (r.records > 6000) details.push('<span class="warn">record budget exceeded: ' + r.records + '</span>');
      rows += '<tr><td><a href="' + LA.router.lesson(r.id) + '">' + esc(r.title) + '</a><br><small>' + r.id + '</small></td><td class="' + (r.pass ? 'ok' : 'fail') + '">' + (r.pass ? 'PASS' : 'FAIL') + '</td><td>' + stepsOk + '/' + r.steps.length + '</td><td>' + checksOk + '/' + r.checks.length + '</td><td>' + (r.playground ? (r.playground.ok ? r.playground.controls + ' ctl' : 'FAIL') : '—') + '</td><td>' + (r.violations.length ? '<span class="warn">' + r.violations.length + '</span>' : 'ok') + '</td><td>' + r.ms + '</td><td>' + (details.length ? '<pre class="err">' + details.join('\n') + '</pre>' : '') + '</td></tr>';
    });
    table.innerHTML = rows;
    container.appendChild(table);
    container.appendChild(el('h2', '', 'Numeric library'));
    const nt = el('table', 'plain');
    nt.innerHTML = '<tr><th>Check</th><th>Status</th><th>Detail</th></tr>' + numRes.map(r => '<tr><td>' + esc(r.name) + '</td><td class="' + (r.pass ? 'ok' : 'fail') + '">' + (r.pass ? 'pass' : 'FAIL') + '</td><td>' + esc(r.detail || '') + '</td></tr>').join('');
    container.appendChild(nt);
    if (info.manifestMissing.length) container.appendChild(el('p', '', '<b>Planned, not built:</b> ' + info.manifestMissing.join(', ')));
    if (info.gallery) {
      container.appendChild(el('h2', '', 'Gallery: end state of every step'));
      results.forEach(r => {
        container.appendChild(el('h3', '', esc(r.title) + ' <small>' + r.id + '</small>'));
        const gal = el('div', 'gallery');
        r.steps.forEach(s => { if (!s.thumb) return; const f = el('figure'); f.appendChild(s.thumb); f.appendChild(el('figcaption', '', (s.k + 1) + '. ' + esc((LA.lessons.get(r.id).steps[s.k] || {}).title || '') + (s.ok ? '' : ' — ERROR'))); gal.appendChild(f); });
        container.appendChild(gal);
      });
    }
    const json = { status: allPass ? 'pass' : 'fail', lessons: results.map(r => ({ id: r.id, pass: r.pass, steps: r.steps.map(s => ({ k: s.k, ok: s.ok, error: s.error, words: s.words })), checks: r.checks, playground: r.playground, violations: r.violations, ms: r.ms, records: r.records })), num: numRes, loadErrors: info.loadErrors, unlisted: info.unlisted, planned: info.manifestMissing };
    const pre = el('pre', 'hidden', esc(JSON.stringify(json))); pre.id = 'selftest-json'; container.appendChild(pre);
    document.body.dataset.selftest = 'done';
  };

  LA.selftest = selftest;
})();
