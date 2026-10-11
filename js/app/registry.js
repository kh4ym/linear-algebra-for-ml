/* LA.registerLesson / LA.lessons — lesson registry with a loud validator.
   A lesson file is an IIFE that calls LA.registerLesson({...}). Violations never block registration (so authors
   can see their lesson) but are listed on the lesson page in dev, on #/selftest, and fail tools/check.ps1. */
window.LA = window.LA || {};
(function () {
  'use strict';
  const lessons = { byId: {}, violations: {}, loadErrors: [], missing: [] };

  const REQUIRED = ['id', 'part', 'title', 'state', 'render', 'steps'];

  LA.validateLesson = function (L) {
    const v = [];
    REQUIRED.forEach(k => { if (L[k] === undefined || L[k] === null) v.push('missing required field "' + k + '"'); });
    if (!L.id || !/^p[0-6]-\d{2}-[a-z0-9-]+$/.test(L.id)) v.push('id must look like p2-01-slug');
    if (typeof L.state !== 'function') v.push('state must be a function returning the initial state');
    if (typeof L.render !== 'function') v.push('render(g, s, ctx) must be a function');
    const steps = Array.isArray(L.steps) ? L.steps : [];
    if (steps.length < 6 || steps.length > 14) v.push('steps: expected 6–14, got ' + steps.length);
    if (!L.badge || !L.badge.drawnIn || !L.badge.realSize || !L.badge.transfers) v.push('badge needs drawnIn, realSize and transfers');
    if (!Array.isArray(L.terms) || !L.terms.length) v.push('terms[] must list at least one glossary term');
    if (L.terms) L.terms.forEach((t, i) => { if (!t || !t.term || !t.def) v.push('terms[' + i + '] needs term and def'); });
    if (!Array.isArray(L.cantShow)) v.push('cantShow[] is required (may be empty)');
    const checks = Array.isArray(L.checks) ? L.checks : [];
    if (checks.length < 2 || checks.length > 3) v.push('checks: expected 2–3, got ' + checks.length);
    checks.forEach((c, i) => {
      if (!c.prompt) v.push('checks[' + i + '] needs a prompt');
      if (c.kind === 'manipulate') { if (typeof c.judge !== 'function') v.push('checks[' + i + '] manipulate needs judge(s)'); if (typeof c.solution !== 'function') v.push('checks[' + i + '] manipulate needs solution(s)'); }
      else if (c.kind === 'entry') { if (!c.bind || c.answer === undefined) v.push('checks[' + i + '] entry needs bind and answer'); }
      else v.push('checks[' + i + '] kind must be "manipulate" or "entry"');
    });
    if (L.playground) {
      const n = (L.playground.controls || []).length;
      if (n > 6) v.push('playground: at most 6 controls, got ' + n);
      if (L.playground.setup && typeof L.playground.setup !== 'function') v.push('playground.setup must be a function');
    }
    // step rules
    let mlTie = 0, recap = 0, misconception = 0;
    let s0 = null; try { s0 = L.state(); } catch (e) { v.push('state() threw: ' + e.message); }
    steps.forEach((st, k) => {
      if (!st.title) v.push('step ' + (k + 1) + ' needs a title');
      if (typeof st.at !== 'function' && typeof st.setup !== 'function') v.push('step ' + (k + 1) + ' needs setup(s) and/or at(s,t)');
      let n = st.narration;
      if (typeof n === 'function') { try { n = n(s0 || {}); } catch (e) { v.push('step ' + (k + 1) + ' narration() threw: ' + e.message); n = ''; } }
      if (typeof n !== 'string' || !n.trim()) v.push('step ' + (k + 1) + ' needs narration');
      else {
        const words = n.replace(/<[^>]+>/g, ' ').replace(/&[a-z]+;/g, ' ').split(/\s+/).filter(Boolean).length;
        if (words < 20 || words > 70) v.push('step ' + (k + 1) + ' narration has ' + words + ' words (expected 20–70)');
        if (/style\s*=/.test(n)) v.push('step ' + (k + 1) + ' narration uses inline style');
        if (/#[0-9a-f]{3,6}\b/i.test(n)) v.push('step ' + (k + 1) + ' narration uses a hex color');
        if (/<script/i.test(n)) v.push('step ' + (k + 1) + ' narration contains <script>');
      }
      if (st.mlTie) { mlTie++; if (k < Math.floor(steps.length * 2 / 3) - 1) v.push('mlTie step ' + (k + 1) + ' should be in the last third'); }
      if (st.recap) recap++;
      if (st.misconception) misconception++;
    });
    if (steps.length) {
      if (mlTie !== 1) v.push('exactly one step must have mlTie:true (found ' + mlTie + ')');
      if (!steps[steps.length - 1].recap) v.push('the last step must have recap:true');
      if (recap > 1) v.push('only the last step should be recap');
      if (misconception < 1) v.push('at least one step must have misconception:true');
    }
    if (L.minutes && (L.minutes < 3 || L.minutes > 25)) v.push('minutes looks wrong: ' + L.minutes);
    return v;
  };

  LA.registerLesson = function (spec) {
    if (!spec || !spec.id) { lessons.loadErrors.push('registerLesson called without an id'); return; }
    if (lessons.byId[spec.id]) lessons.loadErrors.push('duplicate lesson id ' + spec.id);
    spec.tags = spec.tags || ['core'];
    spec.steps = spec.steps || [];
    spec.checks = spec.checks || [];
    spec.terms = spec.terms || [];
    spec.cantShow = spec.cantShow || [];
    spec.atlas = spec.atlas || [];
    let vio;
    try { vio = LA.validateLesson(spec); } catch (e) { vio = ['validator threw: ' + e.message]; }
    lessons.violations[spec.id] = vio;
    spec._violations = vio;
    lessons.byId[spec.id] = spec;
    if (LA.app && LA.app.onLessonRegistered) LA.app.onLessonRegistered(spec);
    return spec;
  };

  /** Ordered list of manifest entries merged with registered specs. */
  lessons.ordered = function () {
    const man = (LA.manifest && LA.manifest.lessons) || [];
    return man.map(m => Object.assign({}, m, { spec: lessons.byId[m.id] || null, status: lessons.byId[m.id] ? 'ready' : (m.status || 'planned') }));
  };
  lessons.get = id => lessons.byId[id] || null;
  lessons.parts = function () { return (LA.manifest && LA.manifest.parts) || []; };
  lessons.indexOf = function (id) { return lessons.ordered().findIndex(e => e.id === id); };
  lessons.neighbors = function (id) {
    const o = lessons.ordered(); const i = o.findIndex(e => e.id === id);
    return { prev: i > 0 ? o[i - 1] : null, next: i >= 0 && i < o.length - 1 ? o[i + 1] : null, index: i, total: o.length };
  };
  /** Glossary: every term from every registered lesson. */
  lessons.glossary = function () {
    const out = [];
    lessons.ordered().forEach(e => { if (e.spec) (e.spec.terms || []).forEach(t => out.push({ term: t.term, def: t.def, lesson: e.id, title: e.spec.title })); });
    return out.sort((a, b) => a.term.localeCompare(b.term));
  };

  LA.lessons = lessons;
})();
