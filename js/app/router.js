/* LA.router — hash routing (works on file://).
   #/                         home
   #/lesson/<id>[/<k>][?instant=1&mode=play|check&t=0.5]
   #/review/<part>            part review page
   #/atlas                    ML atlas
   #/glossary                 glossary page
   #/selftest[?only=<id>&gallery=1]  */
window.LA = window.LA || {};
(function () {
  'use strict';
  const router = { _fn: null, current: null };

  router.parse = function (hash) {
    hash = (hash === undefined ? location.hash : hash) || '#/';
    let path = hash.replace(/^#/, ''), query = {};
    const qi = path.indexOf('?');
    if (qi >= 0) { path.slice(qi + 1).split('&').forEach(kv => { const [k, v] = kv.split('='); if (k) query[decodeURIComponent(k)] = v === undefined ? '1' : decodeURIComponent(v); }); path = path.slice(0, qi); }
    const parts = path.split('/').filter(Boolean);
    const r = { route: parts[0] || 'home', query, raw: hash };
    if (r.route === 'lesson') { r.id = parts[1] || null; r.step = parts[2] !== undefined ? parseInt(parts[2], 10) : undefined; }
    else if (r.route === 'review') r.part = parseInt(parts[1], 10);
    return r;
  };
  router.go = function (hash) { if (location.hash === hash) router.dispatch(); else location.hash = hash; };
  router.lesson = function (id, k, extra) { return '#/lesson/' + id + (k !== undefined ? '/' + k : '') + (extra ? '?' + extra : ''); };
  /** Update the step in the hash without triggering navigation side effects. */
  router.replaceStep = function (id, k) {
    const cur = router.parse();
    const q = Object.keys(cur.query).filter(x => x !== 'mode' && x !== 't').map(x => x + '=' + encodeURIComponent(cur.query[x])).join('&');
    const h = '#/lesson/' + id + '/' + k + (q ? '?' + q : '');
    router._silent = true;
    try { history.replaceState(null, '', h); } catch (_) { location.hash = h; }
    router._silent = false;
  };
  router.onChange = function (fn) { router._fn = fn; };
  router.dispatch = function () { router.current = router.parse(); if (router._fn) router._fn(router.current); };
  router.start = function () { window.addEventListener('hashchange', () => { if (!router._silent) router.dispatch(); }); router.dispatch(); };

  LA.router = router;
})();
