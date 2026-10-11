/* LA.app — application shell: sidebar, lesson page (canvas | player bar | panel), glossary drawer, settings,
   error banner, keyboard shortcuts, routing glue. */
window.LA = window.LA || {};
(function () {
  'use strict';
  const $ = (sel, root) => (root || document).querySelector(sel);
  const el = (tag, cls, html) => { const e = document.createElement(tag); if (cls) e.className = cls; if (html !== undefined) e.innerHTML = html; return e; };
  const esc = s => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

  const app = { started: false, current: null, dom: {}, _bannerKeys: new Set() };

  /* ---------------- errors ---------------- */
  app.reportError = function (e, where) {
    console.error('[LA]', where, e);
    const key = where + ':' + (e && e.message);
    if (app._bannerKeys.has(key)) return;
    app._bannerKeys.add(key);
    const b = $('#banner'); if (!b) return;
    b.classList.remove('hidden');
    const stack = (e && e.stack ? String(e.stack).split('\n').slice(0, 3).join('\n') : String(e));
    b.innerHTML = '<button class="btn small" id="banner-close">dismiss</button><b>Error in ' + esc(where) + '</b>: ' + esc(e && e.message ? e.message : e) + '<pre>' + esc(stack) + '</pre>';
    $('#banner-close').addEventListener('click', () => { b.classList.add('hidden'); app._bannerKeys.clear(); });
  };
  window.addEventListener('error', ev => { if (ev && ev.error) app.reportError(ev.error, 'window'); });

  /* ---------------- start ---------------- */
  app.start = function () {
    if (app.started) return; app.started = true;
    LA.store.load();
    const st = LA.store.data.settings;
    LA.color.setPalette(st.highContrast ? 'highContrast' : 'base');
    LA.player.setSpeed(st.speed || 1); LA.player.setAutoplay(!!st.autoplay); LA.player.setReducedMotion(!!st.reducedMotion);
    const q = LA.router.parse().query;
    if (q.instant === '1') LA.canvas.instant = true;
    $('#main').classList.toggle('sidebar-collapsed', st.sidebar === false || window.innerWidth < 820);
    $('#btn-menu').addEventListener('click', () => app.toggleSidebar());
    $('#btn-glossary').addEventListener('click', () => app.toggleDrawer('glossary'));
    $('#btn-settings').addEventListener('click', () => app.toggleDrawer('settings'));
    document.addEventListener('keydown', app.onKey);
    window.addEventListener('resize', () => { if (window.innerWidth < 820 && !$('#main').classList.contains('sidebar-collapsed')) $('#main').classList.add('sidebar-collapsed'); });
    app.renderSidebar();
    app.updateProgress();
    LA.router.onChange(app.route);
    LA.router.start();
    const boot = $('#boot-status'); if (boot) boot.remove();
    if (LA.lessons.loadErrors.length) console.warn('lesson load problems:', LA.lessons.loadErrors);
  };
  app.onLessonRegistered = function () { if (app.started) { app.renderSidebar(); app.updateProgress(); } };
  app.onProgress = function () { app.renderSidebar(); app.updateProgress(); };

  app.toggleSidebar = function (force) {
    const m = $('#main'); const collapsed = force === undefined ? !m.classList.contains('sidebar-collapsed') : !force;
    m.classList.toggle('sidebar-collapsed', collapsed);
    if (window.innerWidth >= 820) LA.store.setting('sidebar', !collapsed);
    LA.canvas.resize();
  };
  app.updateProgress = function () {
    const p = LA.store.courseProgress();
    const tp = $('#top-progress'); if (tp) { tp.innerHTML = '<i style="width:' + (p * 100).toFixed(1) + '%"></i>'; tp.title = Math.round(p * 100) + '% of lessons complete'; }
  };

  /* ---------------- sidebar ---------------- */
  app.renderSidebar = function () {
    const nav = $('#sidebar'); if (!nav) return;
    nav.innerHTML = '';
    const ordered = LA.lessons.ordered();
    const cur = app.current && app.current.id;
    LA.lessons.parts().forEach(part => {
      const block = el('div', 'side-part' + (app._collapsedParts && app._collapsedParts[part.n] ? ' collapsed' : ''));
      const items = ordered.filter(e => e.part === part.n);
      const done = items.filter(e => LA.store.status(e.id) === 'done').length;
      const title = el('div', 'side-part-title', '<b>' + part.n + '. ' + esc(part.title) + '</b><span>' + done + '/' + items.length + '</span>');
      title.addEventListener('click', () => { app._collapsedParts = app._collapsedParts || {}; app._collapsedParts[part.n] = !app._collapsedParts[part.n]; block.classList.toggle('collapsed'); });
      block.appendChild(title);
      const list = el('div', 'side-lessons');
      items.forEach(e => {
        const ready = !!e.spec;
        const a = el('a', 'side-lesson' + (e.id === cur ? ' active' : '') + (ready ? '' : ' planned'));
        a.href = ready ? LA.router.lesson(e.id) : '#';
        const status = LA.store.status(e.id);
        const tag = (e.tags || []).includes('capstone') ? '<span class="tag capstone">★</span>' : (e.tags || []).includes('extension') ? '<span class="tag ext">ext</span>' : (e.tags || []).includes('primer') ? '<span class="tag">primer</span>' : '<span class="tag"></span>';
        a.innerHTML = '<span class="dot ' + status + '"></span><span>' + esc(e.title) + (ready ? '' : ' <small>(coming soon)</small>') + '</span>' + tag;
        if (!ready) a.addEventListener('click', ev => ev.preventDefault());
        list.appendChild(a);
      });
      block.appendChild(list);
      nav.appendChild(block);
    });
    const links = el('div', 'side-links', '<a href="#/">Home</a><a href="#/atlas">ML atlas</a><a href="#/glossary">Glossary</a><a href="#/selftest">Self-test</a>');
    nav.appendChild(links);
  };

  /* ---------------- routing ---------------- */
  app.route = function (r) {
    const content = $('#content');
    if (r.route !== 'lesson' && app.current) { LA.player.unmount(); app.current = null; }
    if (r.route === 'lesson' && r.id) {
      const spec = LA.lessons.get(r.id);
      if (!spec) { content.innerHTML = ''; content.appendChild(LA.pages.missing(r.id)); return; }
      app.openLesson(spec, r);
      return;
    }
    content.innerHTML = '';
    if (r.route === 'review') content.appendChild(LA.pages.review(r.part));
    else if (r.route === 'atlas') content.appendChild(LA.pages.atlas());
    else if (r.route === 'glossary') content.appendChild(LA.pages.glossary());
    else if (r.route === 'selftest') { const pg = el('div', 'page'); content.appendChild(pg); LA.selftest.run(r.query, pg); }
    else content.appendChild(LA.pages.home());
    app.renderSidebar();
    if (window.innerWidth < 820) $('#main').classList.add('sidebar-collapsed');
  };

  /* ---------------- lesson page ---------------- */
  app.openLesson = function (spec, r) {
    const content = $('#content');
    const sameLesson = app.current && app.current.id === spec.id && app.dom.lesson && app.dom.lesson.isConnected;
    if (!sameLesson) {
      content.innerHTML = '';
      // on screens narrower than ~1500 px the sidebar would squeeze the canvas; collapse it for lessons (m reopens)
      if (!app.current && window.innerWidth < 1500 && window.innerWidth >= 820) $('#main').classList.add('sidebar-collapsed');
      app.buildLessonDom(spec);
      content.appendChild(app.dom.lesson);
      LA.canvas.mount(app.dom.stage);
      app.current = spec;
      const mode = r.query.mode === 'play' || r.query.mode === 'check' ? r.query.mode : 'learn';
      const last = LA.store.lesson(spec.id);
      last.visits = (last.visits || 0) + 1;
      const step = r.step !== undefined && !Number.isNaN(r.step) ? r.step : (last.lastStep || 0);
      LA.player.mount(spec, { step, mode, instant: r.query.instant === '1', dom: { math: app.dom.math, controls: app.dom.controls }, t: r.query.t !== undefined ? parseFloat(r.query.t) : undefined });
      if (mode === 'check' && r.query.check !== undefined) LA.checks.select(parseInt(r.query.check, 10) || 0);
      app.renderSidebar();
      app.syncMode();
    } else if (r.step !== undefined && r.step !== LA.player.step) {
      LA.player.goTo(r.step, LA.canvas.instant ? 1 : 0, { play: !LA.canvas.instant });
    }
  };

  app.buildLessonDom = function (spec) {
    const d = app.dom = {};
    d.lesson = el('div', 'lesson');
    d.stage = el('div', 'stage');
    d.hud = el('div', 'stage-hud');
    d.viewChip = el('span', 'chip view', '2D'); d.viewChip.title = 'Current view. In Play mode click to switch 2D/3D when the lesson supports it.';
    d.viewChip.addEventListener('click', () => { if (LA.player.mode === 'play' && LA.player.s) { LA.player.s.view = LA.player.s.view === '3d' ? '2d' : '3d'; LA.player.s.cam.auto = true; LA.canvas.invalidate(); } });
    d.hud.appendChild(d.viewChip);
    d.modeChip = el('span', 'chip', 'Learn'); d.hud.appendChild(d.modeChip);
    d.stage.appendChild(d.hud);
    d.hint = el('div', 'stage-hint', ''); d.stage.appendChild(d.hint);
    d.lesson.appendChild(d.stage);

    // player bar
    d.player = el('div', 'player');
    d.bPrev = el('button', 'pbtn', '⏮'); d.bPrev.title = 'Previous step (←)';
    d.bPlay = el('button', 'pbtn primary', '▶'); d.bPlay.title = 'Play / pause (space)';
    d.bNext = el('button', 'pbtn', '⏭'); d.bNext.title = 'Next step (→)';
    d.bRestart = el('button', 'pbtn', '↻'); d.bRestart.title = 'Replay this step (r)';
    d.scrubWrap = el('div', 'scrub');
    d.stepLabel = el('span', 'kbd', '1 / 1');
    d.scrub = document.createElement('input'); d.scrub.type = 'range'; d.scrub.min = 0; d.scrub.max = 1000; d.scrub.value = 0; d.scrub.title = 'Scrub the animation';
    d.scrubWrap.appendChild(d.stepLabel); d.scrubWrap.appendChild(d.scrub);
    d.dots = el('div', 'steps');
    d.speed = document.createElement('select'); d.speed.className = 'speed'; d.speed.title = 'Playback speed';
    [0.5, 0.75, 1, 1.5, 2].forEach(v => { const o = document.createElement('option'); o.value = v; o.textContent = v + '×'; d.speed.appendChild(o); });
    d.speed.value = String(LA.player.speed);
    d.modes = el('div', 'modes');
    ['learn', 'play', 'check'].forEach(m => { const b = el('button', '', m === 'learn' ? 'Learn' : m === 'play' ? 'Play' : 'Check'); b.dataset.mode = m; b.title = (m === 'learn' ? 'Narrated steps (1)' : m === 'play' ? 'Free playground (2)' : 'Check yourself (3)'); b.addEventListener('click', () => LA.player.setMode(m)); d.modes.appendChild(b); });
    [d.bPrev, d.bPlay, d.bNext, d.bRestart, d.scrubWrap, d.dots, d.speed, d.modes].forEach(x => d.player.appendChild(x));
    d.lesson.appendChild(d.player);

    // panel
    d.panel = el('aside', 'panel');
    const nb = LA.lessons.neighbors(spec.id);
    const partTitle = (LA.lessons.parts().find(p => p.n === spec.part) || {}).title || '';
    const tag = (spec.tags || []).includes('capstone') ? '<span class="tag">★ capstone</span>' : (spec.tags || []).includes('extension') ? '<span class="tag">extension</span>' : (spec.tags || []).includes('primer') ? '<span class="tag">primer</span>' : '';
    const head = el('div', 'panel-head');
    head.innerHTML = '<div class="crumbs"><span>Part ' + spec.part + ' · ' + esc(partTitle) + '</span>' + tag + '<span>· ' + (nb.index + 1) + '/' + nb.total + '</span></div><h1>' + esc(spec.title) + '</h1>' + (spec.subtitle ? '<div class="subtitle">' + esc(spec.subtitle) + '</div>' : '');
    if (spec.badge) {
      const b = spec.badge;
      head.innerHTML += '<div class="badge"><span>Drawn in</span><b>' + esc(b.drawnIn) + '</b><span>Real ML size</span><b>' + esc(b.realSize) + '</b><span>What transfers</span><b>' + esc(b.transfers) + '</b>' + (spec.cantShow && spec.cantShow.length ? '<div class="cant">What the picture can’t show:<ul>' + spec.cantShow.map(x => '<li>' + esc(x) + '</li>').join('') + '</ul></div>' : '') + '</div>';
    }
    if (LA.store.data.settings.devBadges && spec._violations && spec._violations.length) head.innerHTML += '<div class="hint-box c-bad"><b>Lesson spec violations (' + spec._violations.length + ')</b><br>' + spec._violations.map(esc).join('<br>') + '</div>';
    d.panel.appendChild(head);
    d.body = el('div', 'panel-body');
    d.stepHead = el('div', 'step-head');
    d.narration = el('div', 'narration');
    d.math = el('div', 'math');
    d.controls = el('div', 'controls');
    d.body.appendChild(d.stepHead); d.body.appendChild(d.narration); d.body.appendChild(d.math); d.body.appendChild(d.controls);
    d.panel.appendChild(d.body);
    const foot = el('div', 'panel-foot');
    foot.innerHTML = (nb.prev && nb.prev.spec ? '<a href="' + LA.router.lesson(nb.prev.id) + '">← ' + esc(nb.prev.title) + '</a>' : '<span></span>') + (nb.next && nb.next.spec ? '<a href="' + LA.router.lesson(nb.next.id) + '">' + esc(nb.next.title) + ' →</a>' : (nb.next ? '<span class="c-muted">' + esc(nb.next.title) + ' (coming soon)</span>' : '<a href="#/">Course home →</a>'));
    d.panel.appendChild(foot);
    d.lesson.appendChild(d.panel);

    // wiring
    d.bPrev.addEventListener('click', () => LA.player.prev());
    d.bNext.addEventListener('click', () => { if (!LA.player.next()) { const n = LA.lessons.neighbors(spec.id).next; if (n && n.spec) LA.router.go(LA.router.lesson(n.id)); } });
    d.bPlay.addEventListener('click', () => LA.player.toggle());
    d.bRestart.addEventListener('click', () => LA.player.restartStep());
    d.scrub.addEventListener('input', () => LA.player.scrub(parseInt(d.scrub.value, 10) / 1000));
    d.speed.addEventListener('change', () => { const v = parseFloat(d.speed.value); LA.player.setSpeed(v); LA.store.setting('speed', v); });
    app.bindPlayerEvents(spec);
  };

  app.bindPlayerEvents = function (spec) {
    if (app._bound) { Object.keys(app._bound).forEach(k => LA.player.off(k, app._bound[k])); }
    const d = app.dom;
    const onStep = (k, t) => { app.renderStep(spec, k); d.scrub.value = Math.round((t || 0) * 1000); LA.router.replaceStep(spec.id, k); app.syncPlayIcon(); app.renderDots(spec, k); app.updateProgress(); };
    const onT = t => { d.scrub.value = Math.round(t * 1000); app.syncPlayIcon(); };
    const onPlay = () => app.syncPlayIcon();
    const onMode = () => app.syncMode();
    const onBroken = b => { const o = el('div', 'stage-broken', 'This lesson has an error in <b>' + esc(b.where) + '</b>.<br>Other lessons keep working. Details are in the banner below.'); d.stage.appendChild(o); };
    const onChange = kind => { if (kind === 'camera' || kind === 'drag' || kind === 'control') app.syncHud(); };
    app._bound = { step: onStep, t: onT, play: onPlay, mode: onMode, broken: onBroken, change: onChange };
    Object.keys(app._bound).forEach(k => LA.player.on(k, app._bound[k]));
    LA.player.addFrameHook(app.frameHook);
  };
  app.frameHook = function (p) { if (app.dom && app.dom.viewChip && p.s) { const v = p.s.view === '3d' ? '3D' : '2D'; if (app.dom.viewChip.textContent !== v) app.syncHud(); } return false; };

  app.syncPlayIcon = function () { const d = app.dom; if (!d.bPlay) return; d.bPlay.innerHTML = LA.player.playing ? '❚❚' : (LA.player.t >= 1 ? '↻' : '▶'); d.bPlay.title = LA.player.playing ? 'Pause (space)' : LA.player.t >= 1 ? 'Replay step (space)' : 'Play (space)'; };
  app.syncHud = function () {
    const d = app.dom, p = LA.player; if (!d.viewChip || !p.s) return;
    d.viewChip.textContent = p.s.view === '3d' ? '3D' : '2D';
    d.modeChip.textContent = p.mode === 'learn' ? 'Learn' : p.mode === 'play' ? 'Play' : 'Check';
    const st = p.currentStep();
    const interactive = p.mode !== 'learn' || (st && st.interactive);
    d.hint.textContent = (p.s.view === '3d' ? 'drag to orbit · wheel to zoom · shift-drag to pan' : 'drag to pan · wheel to zoom') + (interactive ? ' · drag the handles' : '') + ' · double-click resets the view';
  };
  app.syncMode = function () {
    const d = app.dom, p = LA.player; if (!d.modes) return;
    d.modes.querySelectorAll('button').forEach(b => b.classList.toggle('active', b.dataset.mode === p.mode));
    const learn = p.mode === 'learn';
    [d.bPrev, d.bPlay, d.bNext, d.bRestart, d.scrub].forEach(x => { x.disabled = !learn; });
    d.scrubWrap.style.opacity = learn ? 1 : 0.4; d.dots.style.opacity = learn ? 1 : 0.4;
    if (learn) app.renderStep(app.current, p.step);
    else if (p.mode === 'play') {
      d.stepHead.innerHTML = '<span class="step-num">Playground</span><h2>Explore freely</h2>';
      const pg = app.current.playground || {};
      d.narration.innerHTML = pg.intro ? pg.intro : '<p>Drag the handles on the picture and move the controls. Everything updates live, including the numbers.</p>';
      const row = el('div', 'ctl-row'); const b = el('button', 'btn small', 'Reset playground'); b.addEventListener('click', () => LA.player.resetPlayground()); row.appendChild(b); d.narration.appendChild(row);
    } else {
      d.stepHead.innerHTML = '<span class="step-num">Check yourself</span><h2>Show it on the picture</h2>';
      d.narration.innerHTML = '<p>Each check is answered by moving the picture into the asked configuration (or typing the numbers that do it). Hold it there for a moment to pass. Checks never block you — skip ahead any time.</p>';
    }
    app.syncHud();
    LA.canvas.resize();
  };

  app.renderDots = function (spec, k) {
    const d = app.dom; if (!d.dots) return;
    const seen = LA.store.lesson(spec.id).seen;
    d.dots.innerHTML = '';
    spec.steps.forEach((st, i) => { const dot = el('i', (seen.includes(i) ? 'seen' : '') + (i === k ? ' cur' : '')); dot.title = (i + 1) + '. ' + st.title; dot.addEventListener('click', () => LA.player.goTo(i, 0, { play: true })); d.dots.appendChild(dot); });
    d.stepLabel.textContent = (k + 1) + ' / ' + spec.steps.length;
  };

  app.renderStep = function (spec, k) {
    const d = app.dom; if (!d.stepHead || !spec) return;
    const st = spec.steps[k]; if (!st) return;
    const flags = ['interactive', 'misconception', 'mlTie', 'recap'].filter(f => st[f]).map(f => '<span class="step-flag ' + f + '">' + (f === 'mlTie' ? 'why ML cares' : f) + '</span>').join('');
    d.stepHead.innerHTML = '<span class="step-num">' + (k + 1) + ' / ' + spec.steps.length + '</span><h2>' + esc(st.title) + '</h2>' + flags;
    let html = st.narration; if (typeof html === 'function') { try { html = html(LA.player.s); } catch (e) { html = '<i>narration error</i>'; } }
    if (!/^\s*<p/.test(html || '')) html = '<p>' + html + '</p>';
    if (st.mlTie) html = '<div class="ml-box">' + html + '</div>';
    d.narration.innerHTML = html;
    if (st.recap && Array.isArray(st.cards)) { const rc = el('div', 'recap-cards'); st.cards.forEach(c => rc.appendChild(el('div', 'recap-card', typeof c === 'string' ? c : (c.html || c.text || '')))); d.narration.appendChild(rc); }
    app.linkTerms(d.narration, spec);
    app.renderDots(spec, k);
    app.syncHud();
  };

  /** Underline the first occurrence of each glossary term in the narration. */
  app.linkTerms = function (root, spec) {
    const terms = (spec.terms || []).slice().sort((a, b) => b.term.length - a.term.length);
    if (!terms.length) return;
    const done = new Set();
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
    const nodes = []; let n; while ((n = walker.nextNode())) nodes.push(n);
    for (const node of nodes) {
      if (node.parentNode.closest('.term, .tx, .math, .ml-box .tx')) continue;
      for (const t of terms) {
        if (done.has(t.term)) continue;
        const re = new RegExp('(^|[^\\w])(' + t.term.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') + 's?)(?=[^\\w]|$)', 'i');
        const m = re.exec(node.nodeValue);
        if (!m) continue;
        const idx = m.index + m[1].length;
        const before = node.nodeValue.slice(0, idx), match = node.nodeValue.slice(idx, idx + m[2].length), after = node.nodeValue.slice(idx + m[2].length);
        const span = el('span', 'term', esc(match)); span.dataset.term = t.term; span.title = t.def;
        span.addEventListener('click', () => app.openDrawer('glossary', t.term));
        const parent = node.parentNode;
        parent.insertBefore(document.createTextNode(before), node);
        parent.insertBefore(span, node);
        node.nodeValue = after;
        done.add(t.term);
        break;
      }
    }
  };

  /* ---------------- drawers ---------------- */
  app.toggleDrawer = function (kind) { const dr = $('#drawer'); if (!dr.classList.contains('hidden') && dr.dataset.kind === kind) { dr.classList.add('hidden'); return; } app.openDrawer(kind); };
  app.openDrawer = function (kind, focusTerm) {
    const dr = $('#drawer'); dr.classList.remove('hidden'); dr.dataset.kind = kind; dr.innerHTML = '';
    const close = el('button', 'icon-btn', '✕'); close.addEventListener('click', () => dr.classList.add('hidden'));
    if (kind === 'glossary') {
      const h = el('h2', '', 'Glossary'); h.appendChild(close); dr.appendChild(h);
      const search = document.createElement('input'); search.type = 'search'; search.placeholder = 'Search terms…'; dr.appendChild(search);
      const list = el('div'); dr.appendChild(list);
      const items = LA.lessons.glossary();
      const render = q => {
        list.innerHTML = '';
        const seen = new Set();
        items.filter(it => !q || it.term.toLowerCase().includes(q) || it.def.toLowerCase().includes(q)).forEach(it => {
          const key = it.term.toLowerCase(); if (seen.has(key)) return; seen.add(key);
          const d = el('div', 'gloss-item' + (focusTerm && key === focusTerm.toLowerCase() ? ' hl' : ''), '<b>' + esc(it.term) + '</b>' + esc(it.def) + '<small>from <a href="' + LA.router.lesson(it.lesson) + '">' + esc(it.title) + '</a></small>');
          list.appendChild(d);
          if (focusTerm && key === focusTerm.toLowerCase()) setTimeout(() => d.scrollIntoView({ block: 'center' }), 0);
        });
        if (!list.children.length) list.innerHTML = '<div class="hint-box">No terms yet — lessons add their own as they load.</div>';
      };
      search.addEventListener('input', () => render(search.value.trim().toLowerCase()));
      render(focusTerm ? '' : '');
    } else {
      const h = el('h2', '', 'Settings'); h.appendChild(close); dr.appendChild(h);
      const st = LA.store.data.settings;
      const row = (label, control) => { const r = el('div', 'settings-row'); r.appendChild(el('span', '', label)); r.appendChild(control); dr.appendChild(r); };
      const chk = (key, onChange) => { const c = document.createElement('input'); c.type = 'checkbox'; c.checked = !!st[key]; c.addEventListener('change', () => { LA.store.setting(key, c.checked); onChange && onChange(c.checked); }); return c; };
      row('Autoplay: advance to the next step after reading time', chk('autoplay', v => LA.player.setAutoplay(v)));
      row('Reduced motion (animations 3× shorter, no inertia)', chk('reducedMotion', v => LA.player.setReducedMotion(v)));
      row('High-contrast colors', chk('highContrast', v => { LA.color.setPalette(v ? 'highContrast' : 'base'); LA.canvas.invalidate(); }));
      row('Show lesson spec violations (for authors)', chk('devBadges'));
      const sp = document.createElement('select'); sp.className = 'btn'; [0.5, 0.75, 1, 1.5, 2].forEach(v => { const o = document.createElement('option'); o.value = v; o.textContent = v + '×'; sp.appendChild(o); }); sp.value = String(st.speed || 1);
      sp.addEventListener('change', () => { LA.player.setSpeed(parseFloat(sp.value)); LA.store.setting('speed', parseFloat(sp.value)); if (app.dom.speed) app.dom.speed.value = sp.value; });
      row('Default playback speed', sp);
      const exp = el('button', 'btn small', 'Export progress'); exp.addEventListener('click', () => { const blob = new Blob([LA.store.exportJSON()], { type: 'application/json' }); const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = 'la-progress.json'; a.click(); });
      row('Progress backup (JSON file)', exp);
      const imp = document.createElement('input'); imp.type = 'file'; imp.accept = '.json'; imp.className = 'btn small';
      imp.addEventListener('change', () => { const f = imp.files[0]; if (!f) return; const rd = new FileReader(); rd.onload = () => { try { LA.store.importJSON(rd.result); app.onProgress(); alert('Progress imported.'); } catch (e) { alert('Could not import: ' + e.message); } }; rd.readAsText(f); });
      row('Import progress', imp);
      const rst = el('button', 'btn small', 'Reset all progress'); rst.addEventListener('click', () => { if (confirm('Erase all progress on this browser?')) { LA.store.reset(); app.onProgress(); } });
      row('Start over', rst);
      dr.appendChild(el('div', 'hint-box', 'Keyboard: ← → steps · space play/pause · r replay · 1/2/3 Learn/Play/Check · m contents · g glossary · Esc closes drawers'));
    }
  };

  /* ---------------- keyboard ---------------- */
  app.onKey = function (e) {
    const tag = (e.target && e.target.tagName) || '';
    if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || (e.target && e.target.isContentEditable)) return;
    if (e.key === 'Escape') { $('#drawer').classList.add('hidden'); return; }
    if (e.key === 'm') { app.toggleSidebar(); return; }
    if (e.key === 'g') { app.toggleDrawer('glossary'); return; }
    if (!app.current) return;
    switch (e.key) {
      case 'ArrowRight': case 'PageDown': LA.player.next(); e.preventDefault(); break;
      case 'ArrowLeft': case 'PageUp': LA.player.prev(); e.preventDefault(); break;
      case ' ': LA.player.toggle(); e.preventDefault(); break;
      case 'r': LA.player.restartStep(); break;
      case 'Home': LA.player.goTo(0, 0, { play: true }); break;
      case 'End': LA.player.goTo(LA.player.stepCount() - 1, 1); break;
      case '1': LA.player.setMode('learn'); break;
      case '2': LA.player.setMode('play'); break;
      case '3': LA.player.setMode('check'); break;
    }
  };

  LA.app = app;
})();
