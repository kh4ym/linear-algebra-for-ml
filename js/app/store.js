/* LA.store — progress and settings in localStorage (key la.v1), debounced writes, JSON export/import.
   Works on file:// ; tolerates a missing or blocked store. */
window.LA = window.LA || {};
(function () {
  'use strict';
  const KEY = 'la.v1';
  const defaults = () => ({ v: 1, progress: {}, settings: { speed: 1, reducedMotion: false, highContrast: false, autoplay: false, sidebar: true, devBadges: true }, last: null });
  const store = { data: defaults(), _timer: 0, available: true };

  store.load = function () {
    try {
      const raw = localStorage.getItem(KEY);
      if (raw) { const d = JSON.parse(raw); if (d && d.v === 1) { store.data = Object.assign(defaults(), d); store.data.settings = Object.assign(defaults().settings, d.settings || {}); } }
    } catch (e) { store.available = false; }
    try { const mq = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)'); if (mq && mq.matches && store.data.settings.reducedMotion === false && !store.data.settings._rmTouched) store.data.settings.reducedMotion = true; } catch (_) { }
    return store.data;
  };
  store.save = function () {
    clearTimeout(store._timer);
    store._timer = setTimeout(() => { try { localStorage.setItem(KEY, JSON.stringify(store.data)); } catch (e) { store.available = false; } }, 250);
  };
  store.lesson = function (id) {
    const p = store.data.progress;
    if (!p[id]) p[id] = { seen: [], checks: {}, lastStep: 0, done: false, visits: 0 };
    return p[id];
  };
  store.markSeen = function (id, k) {
    const r = store.lesson(id);
    if (!r.seen.includes(k)) r.seen.push(k);
    r.lastStep = k;
    store.data.last = { id, step: k };
    store.recompute(id); store.save();
  };
  store.markCheck = function (id, i) { const r = store.lesson(id); r.checks[i] = true; store.recompute(id); store.save(); };
  store.recompute = function (id) {
    const r = store.lesson(id); const L = LA.lessons && LA.lessons.get(id);
    if (!L) return;
    const steps = (L.steps || []).length, nChecks = (L.checks || []).length;
    const passed = Object.keys(r.checks).filter(k => r.checks[k]).length;
    r.done = r.seen.length >= steps && passed >= Math.min(2, nChecks);
  };
  store.status = function (id) {
    const r = store.data.progress[id]; if (!r) return 'new';
    if (r.done) return 'done';
    return (r.seen.length || Object.keys(r.checks).length) ? 'partial' : 'new';
  };
  store.setting = function (k, v) {
    if (v === undefined) return store.data.settings[k];
    store.data.settings[k] = v; if (k === 'reducedMotion') store.data.settings._rmTouched = true; store.save(); return v;
  };
  store.courseProgress = function () {
    const ids = (LA.lessons ? LA.lessons.ordered() : []).filter(e => e.spec).map(e => e.id);
    if (!ids.length) return 0;
    return ids.filter(id => store.status(id) === 'done').length / ids.length;
  };
  store.exportJSON = function () { return JSON.stringify(store.data, null, 2); };
  store.importJSON = function (json) { const d = JSON.parse(json); if (!d || d.v !== 1) throw new Error('not a la.v1 progress file'); store.data = Object.assign(defaults(), d); store.save(); };
  store.reset = function () { store.data = defaults(); try { localStorage.removeItem(KEY); } catch (_) { } };

  LA.store = store;
})();
