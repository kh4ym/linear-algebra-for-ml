/* boot — inject one classic <script> per manifest lesson, count loads, then start the app.
   Missing files simply mean "coming soon"; nothing else in the app needs editing when a lesson is added. */
(function () {
  'use strict';
  const man = window.LA && LA.manifest;
  const status = document.getElementById('boot-status');
  function start() { try { LA.app.start(); } catch (e) { console.error(e); if (status) status.textContent = 'Failed to start: ' + e.message; } }
  if (!man || !man.lessons) { if (status) status.textContent = 'Manifest missing (lessons/_manifest.js).'; return; }
  const ids = man.lessons.filter(l => l.status !== 'planned').map(l => l.id);
  let pending = ids.length, started = false;
  const done = () => { if (started) return; started = true; start(); };
  if (!pending) { done(); return; }
  ids.forEach(id => {
    const sc = document.createElement('script');
    sc.src = 'lessons/' + id + '.js';
    sc.async = false;
    sc.onload = () => { if (!LA.lessons.byId[id]) LA.lessons.loadErrors.push('lessons/' + id + '.js loaded but did not register (syntax error or wrong id?)'); if (--pending === 0) done(); };
    sc.onerror = () => { LA.lessons.missing.push(id); if (--pending === 0) done(); };
    document.body.appendChild(sc);
  });
  if (status) status.textContent = 'Loading ' + ids.length + ' lessons…';
  setTimeout(done, 8000); // safety: never hang on a stuck script
})();
