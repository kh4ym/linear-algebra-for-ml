/* LA.player — mounts a lesson on the engine canvas and runs its steps.
   Step model: step.setup(s) puts the state at the step's start configuration, step.at(s, t) writes the animated
   fields as a pure function of t. An instant pre-pass snapshots the end state of every step, so prev/next/jump/
   scrub are deterministic: goTo(k, t) = restore snapshot k−1 → setup → at(t). The state object `s` keeps its
   identity for the whole mount (reset in place). */
window.LA = window.LA || {};
(function () {
  'use strict';

  const player = {
    lesson: null, s: null, initial: null, snaps: [], gfx: null,
    mode: 'learn', step: 0, t: 0, playing: false, speed: 1, autoplay: false, reducedMotion: false,
    broken: null, dom: null, controlsUI: null, _autoTimer: 0, _listeners: {}, _dirtyControls: false, _frameHooks: [],
    activeCheck: null, instant: false, time: 0,
  };

  /** Cosmetic effects hook (no-op for now; lessons may call ctx.fx.fly(worldPos, target) safely). */
  const NOFX = { fly() { }, glow() { } };
  player.NOFX = NOFX;
  player.on = function (ev, fn) { (player._listeners[ev] = player._listeners[ev] || []).push(fn); };
  player.off = function (ev, fn) { const l = player._listeners[ev]; if (l) player._listeners[ev] = l.filter(f => f !== fn); };
  function emit(ev, a, b) { (player._listeners[ev] || []).forEach(fn => { try { fn(a, b); } catch (e) { console.error(e); } }); }

  /** Run a lesson callback with quarantine. */
  function safe(where, fn, fallback) {
    if (player.broken) return fallback;
    try { return fn(); }
    catch (e) {
      player.broken = { where, error: e };
      if (LA.app && LA.app.reportError) LA.app.reportError(e, (player.lesson ? player.lesson.id : '?') + ' → ' + where);
      else console.error(where, e);
      emit('broken', player.broken);
      return fallback;
    }
  }

  /** Mount a lesson. opts: {step, t, mode, instant, dom:{math, controls, hud}} */
  player.mount = function (lesson, opts) {
    opts = opts || {};
    player.unmount();
    player.lesson = lesson; player.broken = null; player.instant = !!opts.instant || !!LA.canvas.instant;
    player.dom = opts.dom || null;
    player.gfx = player.gfx || new LA.Gfx();
    player.s = {};
    const init = safe('state', () => lesson.state(), {});
    player.initial = LA.clone(init);
    LA.assignInPlace(player.s, init);
    player.snaps = [];
    // instant pre-pass: end state of every step
    const tmp = LA.clone(init);
    (lesson.steps || []).forEach((st, k) => {
      safe('prepass:' + k, () => { tmp.cam = tmp.cam || {}; tmp.cam.auto = true; if (st.setup) st.setup(tmp); if (st.at) st.at(tmp, 1); });
      player.snaps.push(LA.clone(tmp));
    });
    LA.canvas.onFrame(player.frame);
    LA.input.attach(LA.canvas.el, player.gfx, {
      state: () => player.s,
      changed: (kind) => { if (kind === 'drag' || kind === 'dragend') player._dirtyControls = true; LA.canvas.invalidate(); emit('change', kind); },
      cameraLocked: () => { const st = player.currentStep(); return !!(st && st.lockCamera); },
      resetCamera: () => player.goTo(player.step, player.t),
      handlesEnabled: () => player.gfx.handlesEnabled,
    });
    player.setMode(opts.mode || 'learn', { silent: true });
    if (player.mode === 'learn') player.goTo(opts.step || 0, opts.t === undefined ? (player.instant ? 1 : 0) : opts.t, { play: !player.instant && opts.play !== false && opts.t === undefined, exactT: opts.t !== undefined });
    emit('mount', lesson);
  };

  player.unmount = function () {
    player.pause();
    clearTimeout(player._autoTimer);
    LA.input.stop();
    if (player.controlsUI) { player.controlsUI.destroy(); player.controlsUI = null; }
    if (LA.checks && LA.checks.stop) LA.checks.stop();
    player.lesson = null; player.s = null; player.snaps = []; player.activeCheck = null;
    LA.canvas.animate(false);
  };

  player.currentStep = function () { return player.lesson && player.lesson.steps ? player.lesson.steps[player.step] : null; };
  player.stepCount = function () { return player.lesson && player.lesson.steps ? player.lesson.steps.length : 0; };

  /** Jump to step k at progress t (deterministic). opts: {play} */
  player.goTo = function (k, t, opts) {
    opts = opts || {};
    const L = player.lesson; if (!L) return;
    const n = player.stepCount(); if (!n) return;
    k = LA.clamp(k | 0, 0, n - 1);
    t = t === undefined ? 0 : LA.clamp01(t);
    if (player.mode !== 'learn') player.setMode('learn', { keepState: true });
    clearTimeout(player._autoTimer);
    const start = k > 0 ? player.snaps[k - 1] : player.initial;
    LA.assignInPlace(player.s, start);
    player.s.cam = player.s.cam || {};
    player.s.cam.auto = true;
    const st = L.steps[k];
    safe('setup:' + k, () => { if (st.setup) st.setup(player.s); });
    if (player.instant && !opts.exactT) t = 1;
    safe('at:' + k, () => { if (st.at) st.at(player.s, t); });
    player.step = k; player.t = t;
    player.playing = !!opts.play && !player.instant && t < 1;
    if (LA.store) LA.store.markSeen(L.id, k);
    player._dirtyControls = true;
    emit('step', k, t);
    LA.canvas.animate(player.playing);
    LA.canvas.invalidate();
  };

  /** Scrub within the current step. */
  player.scrub = function (t) {
    const L = player.lesson; if (!L) return;
    const k = player.step, st = L.steps[k];
    const start = k > 0 ? player.snaps[k - 1] : player.initial;
    const camUser = player.s.cam && player.s.cam.auto === false ? LA.clone(player.s.cam) : null;
    LA.assignInPlace(player.s, start);
    player.s.cam = player.s.cam || {}; player.s.cam.auto = true;
    safe('setup:' + k, () => { if (st.setup) st.setup(player.s); });
    player.t = LA.clamp01(t);
    safe('at:' + k, () => { if (st.at) st.at(player.s, player.t); });
    if (camUser) { player.s.cam = camUser; }
    player.playing = false; clearTimeout(player._autoTimer);
    emit('t', player.t);
    LA.canvas.invalidate();
  };

  player.play = function () {
    if (!player.lesson || player.mode !== 'learn') return;
    if (player.t >= 1) { player.goTo(player.step, 0, { play: true }); return; }
    player.playing = true; LA.canvas.animate(true); emit('play', true);
  };
  player.pause = function () { if (player.playing) { player.playing = false; emit('play', false); } LA.canvas.animate(false); clearTimeout(player._autoTimer); };
  player.toggle = function () { if (player.playing) player.pause(); else player.play(); };
  /** First Next finishes the running animation; second Next advances. */
  player.next = function () {
    if (!player.lesson) return false;
    if (player.mode !== 'learn') { player.setMode('learn'); return true; }
    if (player.playing && player.t < 1) { player.scrub(1); emit('step', player.step, 1); return true; }
    if (player.step < player.stepCount() - 1) { player.goTo(player.step + 1, 0, { play: true }); return true; }
    emit('end'); return false;
  };
  player.prev = function () {
    if (!player.lesson) return false;
    if (player.mode !== 'learn') { player.setMode('learn'); return true; }
    if (player.step > 0) { player.goTo(player.step - 1, 1); return true; }
    player.goTo(0, 1); return false;
  };
  player.restartStep = function () { player.goTo(player.step, 0, { play: true }); };

  /** Switch mode: 'learn' | 'play' | 'check'. */
  player.setMode = function (mode, opts) {
    opts = opts || {};
    const L = player.lesson; if (!L) return;
    const prev = player.mode;
    player.mode = mode;
    player.pause();
    if (player.controlsUI) { player.controlsUI.destroy(); player.controlsUI = null; }
    if (LA.checks && LA.checks.stop) LA.checks.stop();
    if (mode === 'play') {
      LA.assignInPlace(player.s, player.initial);
      player.s.cam = player.s.cam || {}; player.s.cam.auto = true;
      safe('playground.setup', () => { if (L.playground && L.playground.setup) L.playground.setup(player.s); });
      if (player.dom && player.dom.controls) {
        const defs = (L.playground && L.playground.controls) || [];
        player.controlsUI = LA.ui.build(player.dom.controls, defs, player.s, () => { player._dirtyControls = true; LA.canvas.invalidate(); emit('change', 'control'); });
      }
    } else if (mode === 'check') {
      LA.assignInPlace(player.s, player.initial);
      player.s.cam = player.s.cam || {}; player.s.cam.auto = true;
      if (LA.checks) LA.checks.begin(player, player.dom ? player.dom.controls : null);
    } else if (!opts.keepState && prev !== 'learn') {
      // back to learn: restore the current step at its end
      const k = player.step; const start = k > 0 ? player.snaps[k - 1] : player.initial;
      LA.assignInPlace(player.s, start); player.s.cam = player.s.cam || {}; player.s.cam.auto = true;
      const st = L.steps[k];
      safe('setup:' + k, () => { if (st && st.setup) st.setup(player.s); });
      safe('at:' + k, () => { if (st && st.at) st.at(player.s, 1); });
      player.t = 1;
    }
    if (!opts.silent) emit('mode', mode);
    LA.canvas.invalidate();
  };

  /** Reset the playground to the lesson's initial state. */
  player.resetPlayground = function () { if (player.mode === 'play') player.setMode('play'); };

  player.setSpeed = function (v) { player.speed = v; emit('speed', v); };
  player.setAutoplay = function (v) { player.autoplay = !!v; };
  player.setReducedMotion = function (v) { player.reducedMotion = !!v; };

  /** Per-frame render. Returns true while animation continues. */
  player.frame = function (dt, size, ctx) {
    const L = player.lesson; if (!L || !player.s) { ctx.fillStyle = LA.color.bg; ctx.fillRect(0, 0, size.w, size.h); return false; }
    player.time += dt;
    let more = false;
    if (player.playing && player.mode === 'learn') {
      const st = L.steps[player.step];
      const dur = Math.max(0.2, (st.duration || 2.5) * (player.reducedMotion ? 0.35 : 1));
      player.t = Math.min(1, player.t + dt * player.speed / dur);
      safe('at:' + player.step, () => { if (st.at) st.at(player.s, player.t); });
      emit('t', player.t);
      if (player.t >= 1) {
        player.playing = false; emit('play', false);
        if (player.autoplay && player.step < player.stepCount() - 1) {
          const words = narrationWords(st, player.s);
          player._autoTimer = setTimeout(() => { if (player.mode === 'learn' && !player.playing) player.next(); }, Math.max(2200, words * 260) / Math.max(0.5, player.speed));
        }
      } else more = true;
    }
    if (!player.instant && LA.input.tick(dt)) more = true;
    const g = player.gfx;
    const st = L.steps[player.step];
    g.handlesEnabled = player.mode !== 'learn' || !!(st && st.interactive);
    if (player.broken) { drawBroken(ctx, size); return false; }
    g.begin(ctx, size, player.s, { instant: player.instant, time: player.time });
    safe('render', () => L.render(g, player.s, player.ctx()));
    if (player.broken) { drawBroken(ctx, size); return false; }
    g.end();
    // math panel
    if (player.dom && player.dom.math) {
      if (L.math) { const html = safe('math', () => L.math(player.s, LA.tex), ''); LA.tex.reconcile(player.dom.math, html || ''); }
      else if (player.dom.math.innerHTML) LA.tex.reconcile(player.dom.math, '');
    }
    if (player._dirtyControls) { player._dirtyControls = false; if (player.controlsUI) player.controlsUI.refresh(); }
    if (player.mode === 'check' && LA.checks && LA.checks.tick(player, dt)) more = true;
    for (const h of player._frameHooks) { try { if (h(player, dt)) more = true; } catch (_) { } }
    return more;
  };

  player.ctx = function () {
    return { s: player.s, mode: player.mode, step: player.step, t: player.t, size: LA.canvas.size, dpr: LA.canvas.size.dpr, hot: player.gfx.hot, time: player.time, instant: player.instant, fx: NOFX };
  };
  player.addFrameHook = function (fn) { player._frameHooks.push(fn); };
  player.removeFrameHook = function (fn) { player._frameHooks = player._frameHooks.filter(f => f !== fn); };

  function narrationWords(st, s) {
    let n = st.narration; if (typeof n === 'function') { try { n = n(s); } catch (_) { n = ''; } }
    return String(n || '').replace(/<[^>]+>/g, ' ').split(/\s+/).filter(Boolean).length;
  }
  function drawBroken(ctx, size) {
    ctx.fillStyle = LA.color.bg; ctx.fillRect(0, 0, size.w, size.h);
    ctx.fillStyle = LA.color.bad; ctx.font = '15px system-ui, sans-serif'; ctx.textAlign = 'center';
    ctx.fillText('This lesson hit an error (' + (player.broken && player.broken.where) + '). See the banner.', size.w / 2, size.h / 2);
  }

  /** Render a lesson's step k (end state) into an arbitrary context without touching the live player. */
  player.renderStatic = function (lesson, k, ctx, size, opts) {
    opts = opts || {};
    const g = new LA.Gfx();
    const s = LA.clone(lesson.state());
    for (let i = 0; i <= k; i++) { const st = lesson.steps[i]; s.cam = s.cam || {}; s.cam.auto = true; if (st.setup) st.setup(s); if (st.at) st.at(s, i === k ? (opts.t === undefined ? 1 : opts.t) : 1); }
    g.handlesEnabled = false;
    g.noText = !!opts.noText;
    g.begin(ctx, size, s, { instant: true });
    lesson.render(g, s, { s, mode: 'learn', step: k, t: opts.t === undefined ? 1 : opts.t, size, dpr: size.dpr || 1, hot: null, time: 0, instant: true });
    g.end();
    return s;
  };

  LA.player = player;
})();
