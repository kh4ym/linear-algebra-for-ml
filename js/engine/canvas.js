/* LA.canvas — the single engine-owned canvas: sizing (DPR capped at 2), ResizeObserver, and a dirty-flag
   requestAnimationFrame loop that renders zero frames while idle. */
window.LA = window.LA || {};
(function () {
  'use strict';

  const canvas = {
    el: null, ctx: null, wrapper: null,
    size: { w: 300, h: 150, dpr: 1 },
    _dirty: false, _animating: false, _raf: 0, _last: 0, _frame: null, _ro: null, _dprTimer: 0,
    instant: false,          // true in ?instant=1 / selftest mode: no shimmer, no inertia
  };

  /** Create (once) and attach the canvas to a wrapper element; subsequent calls just re-parent it. */
  canvas.mount = function (wrapper) {
    if (!canvas.el) {
      canvas.el = document.createElement('canvas');
      canvas.el.tabIndex = 0;
      canvas.ctx = canvas.el.getContext('2d', { alpha: false });
      canvas._ro = new ResizeObserver(() => canvas.resize());
      document.addEventListener('visibilitychange', () => { if (!document.hidden) canvas.invalidate(); });
    }
    if (canvas.wrapper && canvas.wrapper !== wrapper) canvas._ro.unobserve(canvas.wrapper);
    canvas.wrapper = wrapper;
    if (canvas.el.parentNode !== wrapper) wrapper.appendChild(canvas.el);
    canvas._ro.observe(wrapper);
    canvas.resize();
    return canvas.el;
  };

  /** Read wrapper size and devicePixelRatio (cap 2); resize backing store; request a frame. */
  canvas.resize = function () {
    if (!canvas.wrapper || !canvas.el) return;
    const r = canvas.wrapper.getBoundingClientRect();
    const w = Math.max(50, Math.round(r.width)), h = Math.max(50, Math.round(r.height));
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    if (w !== canvas.size.w || h !== canvas.size.h || dpr !== canvas.size.dpr) {
      canvas.size = { w, h, dpr };
      canvas.el.width = Math.round(w * dpr);
      canvas.el.height = Math.round(h * dpr);
      canvas.el.style.width = w + 'px';
      canvas.el.style.height = h + 'px';
    }
    canvas.invalidate();
  };

  /** Register the per-frame callback: fn(dt, size) → returns true to keep animating. */
  canvas.onFrame = function (fn) { canvas._frame = fn; };

  /** Ask for one redraw. */
  canvas.invalidate = function () {
    canvas._dirty = true;
    if (!canvas._raf) canvas._raf = requestAnimationFrame(canvas._tick);
  };

  /** Keep redrawing every frame while `on` (tweens, inertia). */
  canvas.animate = function (on) {
    canvas._animating = !!on;
    if (on) canvas.invalidate();
  };

  canvas._tick = function (now) {
    canvas._raf = 0;
    let dt = canvas._last ? (now - canvas._last) / 1000 : 1 / 60;
    canvas._last = now;
    if (dt > 0.1) dt = 0.1;                       // after background throttling, never jump
    canvas._dirty = false;
    let more = false;
    if (canvas._frame && canvas.ctx) {
      const ctx = canvas.ctx, s = canvas.size;
      ctx.setTransform(s.dpr, 0, 0, s.dpr, 0, 0);
      try { more = !!canvas._frame(dt, s, ctx); }
      catch (e) { if (LA.app && LA.app.reportError) LA.app.reportError(e, 'render'); else console.error(e); }
    }
    // re-check DPR occasionally (window moved between monitors)
    if (now - canvas._dprTimer > 1000) { canvas._dprTimer = now; const d = Math.min(2, window.devicePixelRatio || 1); if (d !== canvas.size.dpr) canvas.resize(); }
    if (more || canvas._animating || canvas._dirty) canvas._raf = requestAnimationFrame(canvas._tick);
    else canvas._last = 0;
  };

  /** An independent offscreen drawing surface (selftest, gallery thumbnails, TOC cards). */
  canvas.offscreen = function (w, h, dpr) {
    const el = document.createElement('canvas');
    dpr = dpr || 1;
    el.width = Math.round(w * dpr); el.height = Math.round(h * dpr);
    el.style.width = w + 'px'; el.style.height = h + 'px';
    const ctx = el.getContext('2d', { alpha: false });
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { el, ctx, size: { w, h, dpr } };
  };

  LA.canvas = canvas;
})();
