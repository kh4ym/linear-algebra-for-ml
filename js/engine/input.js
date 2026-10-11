/* LA.input — pointer handling on the engine canvas: handle dragging (2D and 3D ray–plane picking), 2D pan/zoom,
   3D orbit/dolly/pan with inertia, pinch, hover. Camera edits write into the lesson state (s.cam) and set
   s.cam.auto = false so a step's camera tween stops fighting the user. */
window.LA = window.LA || {};
(function () {
  'use strict';

  const input = {
    el: null, gfx: null, hooks: null,
    pointers: new Map(), drag: null, vel: { yaw: 0, pitch: 0 }, pinch: null, lastHot: null,
  };

  /** hooks: { state() → s, changed(kind), cameraLocked() → bool, handlesEnabled() → bool } */
  input.attach = function (el, gfx, hooks) {
    if (input.el === el) { input.gfx = gfx; input.hooks = hooks; return; }
    input.el = el; input.gfx = gfx; input.hooks = hooks;
    el.addEventListener('pointerdown', onDown);
    el.addEventListener('pointermove', onMove);
    el.addEventListener('pointerup', onUp);
    el.addEventListener('pointercancel', onUp);
    el.addEventListener('pointerleave', () => { if (input.gfx && input.gfx.hot) { input.gfx.hot = null; LA.canvas.invalidate(); } });
    el.addEventListener('wheel', onWheel, { passive: false });
    el.addEventListener('contextmenu', e => e.preventDefault());
    el.addEventListener('dblclick', onDbl);
  };

  function local(e) { const r = input.el.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; }
  function S() { return input.hooks && input.hooks.state ? input.hooks.state() : null; }

  function hitHandle(sx, sy) {
    const hs = input.gfx ? input.gfx.handles : [];
    let best = null, bd = Infinity;
    for (const h of hs) {
      if (!h.enabled) continue;
      const d = Math.hypot(h.screen[0] - sx, h.screen[1] - sy);
      if (d <= h.r + 4 && d < bd) { bd = d; best = h; }
    }
    return best;
  }

  function onDown(e) {
    const s = S(); if (!s) return;
    input.el.focus({ preventScroll: true });
    const [sx, sy] = local(e);
    input.pointers.set(e.pointerId, [sx, sy]);
    if (input.pointers.size === 2) { // start pinch
      const pts = [...input.pointers.values()];
      input.pinch = { d0: Math.hypot(pts[0][0] - pts[1][0], pts[0][1] - pts[1][1]), halfH: s.cam.halfH, dist: s.cam.dist };
      input.drag = null; return;
    }
    const h = e.button === 0 ? hitHandle(sx, sy) : null;
    if (h) {
      input.drag = { kind: 'handle', id: h.id, P: h.P, is3d: h.is3d, plane: h.plane, startWorld: h.world, start: [sx, sy] };
      input.gfx.dragging = h.id;
      try { input.el.setPointerCapture(e.pointerId); } catch (_) { }
      input.el.classList.add('grabbing');
      e.preventDefault();
      return;
    }
    if (input.hooks.cameraLocked && input.hooks.cameraLocked()) return;
    const pan = e.button === 1 || e.button === 2 || e.shiftKey;
    input.drag = { kind: pan ? 'pan' : 'orbit', last: [sx, sy], moved: false };
    input.vel = { yaw: 0, pitch: 0 };
    try { input.el.setPointerCapture(e.pointerId); } catch (_) { }
    e.preventDefault();
  }

  function onMove(e) {
    const s = S(); if (!s) return;
    const [sx, sy] = local(e);
    if (input.pointers.has(e.pointerId)) input.pointers.set(e.pointerId, [sx, sy]);
    if (input.pinch && input.pointers.size === 2) {
      const pts = [...input.pointers.values()];
      const d = Math.hypot(pts[0][0] - pts[1][0], pts[0][1] - pts[1][1]) || 1;
      const f = input.pinch.d0 / d;
      userCamera(s);
      if (s.view === '3d') s.cam.dist = LA.clamp(input.pinch.dist * f, 2, 60); else s.cam.halfH = LA.clamp(input.pinch.halfH * f, 0.5, 40);
      input.hooks.changed('camera'); return;
    }
    if (!input.drag) {
      const h = hitHandle(sx, sy);
      const id = h ? h.id : null;
      if (id !== input.gfx.hot) { input.gfx.hot = id; input.el.classList.toggle('pointer', !!id); LA.canvas.invalidate(); }
      return;
    }
    const d = input.drag;
    if (d.kind === 'handle') {
      const h = (input.gfx.handles || []).find(x => x.id === d.id);   // latest closure, never the one from pointerdown
      if (!h) return;
      if (!d.moved && Math.hypot(sx - d.start[0], sy - d.start[1]) < 3) return;   // below click threshold
      d.moved = true;
      let w = null;
      if (!d.is3d) w = h.P.toWorld(sx, sy);
      else {
        const pl = resolvePlane(h, s);
        w = h.P.pick(sx, sy, pl.point, pl.normal);
        if (!w) { const f = h.P.forward; w = h.P.pick(sx, sy, h.world, f); }
      }
      if (w) { try { h.onDrag(w, s); } catch (err) { if (LA.app && LA.app.reportError) LA.app.reportError(err, 'drag'); } input.hooks.changed('drag'); }
      e.preventDefault(); return;
    }
    const dx = sx - d.last[0], dy = sy - d.last[1];
    d.last = [sx, sy];
    const now = performance.now(), mdt = Math.max(0.008, (now - (d.lastT || now - 16)) / 1000); d.lastT = now;
    if (Math.abs(dx) + Math.abs(dy) > 0) d.moved = true;
    userCamera(s);
    if (s.view === '3d') {
      if (d.kind === 'pan') {
        const P = input.gfx.P, k = 1 / (P.pxPerUnit || 50);
        s.cam.target = [s.cam.target[0] - (P.right[0] * dx - P.up[0] * dy) * k, s.cam.target[1] - (P.right[1] * dx - P.up[1] * dy) * k, s.cam.target[2] - (P.right[2] * dx - P.up[2] * dy) * k];
      } else {
        const ky = -dx * 0.0075, kp = dy * 0.0075;
        s.cam.yaw += ky; s.cam.pitch = LA.clamp(s.cam.pitch + kp, -LA.cam.PITCH_MAX, LA.cam.PITCH_MAX);
        input.vel = { yaw: LA.clamp(ky / mdt, -5, 5), pitch: LA.clamp(kp / mdt, -5, 5) };
      }
    } else {
      const sc = input.gfx.P.scale || 50;
      s.cam.center = [s.cam.center[0] - dx / sc, s.cam.center[1] + dy / sc];
    }
    input.hooks.changed('camera');
    e.preventDefault();
  }

  function onUp(e) {
    input.pointers.delete(e.pointerId);
    if (input.pointers.size < 2) input.pinch = null;
    if (!input.drag) return;
    const d = input.drag; input.drag = null;
    input.el.classList.remove('grabbing');
    try { input.el.releasePointerCapture(e.pointerId); } catch (_) { }
    if (d.kind === 'handle') {
      input.gfx.dragging = null;
      if (!d.moved) {   // a click: no drag happened
        const h = (input.gfx.handles || []).find(x => x.id === d.id);
        const s = S();
        if (h && h.onClick && s) { try { h.onClick(s); } catch (err) { if (LA.app && LA.app.reportError) LA.app.reportError(err, 'click'); } }
      }
      input.hooks.changed('dragend'); return;
    }
    const stale = d.lastT && performance.now() - d.lastT > 80;   // pointer paused before release: no flick
    if (d.kind === 'orbit' && !stale && !LA.canvas.instant && (Math.abs(input.vel.yaw) > 0.02 || Math.abs(input.vel.pitch) > 0.02)) LA.canvas.animate(true);
    else input.vel = { yaw: 0, pitch: 0 };
  }

  function onWheel(e) {
    const s = S(); if (!s) return;
    if (input.hooks.cameraLocked && input.hooks.cameraLocked()) return;
    e.preventDefault();
    const f = Math.exp(LA.clamp(e.deltaY, -60, 60) * 0.0025);
    userCamera(s);
    if (s.view === '3d') s.cam.dist = LA.clamp(s.cam.dist * f, 2, 60);
    else {
      const [sx, sy] = local(e), P = input.gfx.P;
      const before = P.toWorld(sx, sy);
      s.cam.halfH = LA.clamp(s.cam.halfH * f, 0.4, 40);
      const P2 = LA.cam.projector('2d', s.cam, input.gfx.size);
      const after = P2.toWorld(sx, sy);
      s.cam.center = [s.cam.center[0] + before[0] - after[0], s.cam.center[1] + before[1] - after[1]];
    }
    input.hooks.changed('camera');
  }
  function onDbl(e) { const s = S(); if (!s) return; if (hitHandle(...local(e))) return; if (input.hooks.resetCamera) input.hooks.resetCamera(); }

  function userCamera(s) { if (s.cam) s.cam.auto = false; }

  /** Decide the drag plane for a 3D handle. */
  function resolvePlane(h, s) {
    const P = h.P, w = h.world;
    let plane = h.plane;
    if (plane && typeof plane === 'object') return { point: plane.point || w, normal: plane.normal };
    if (plane === 'auto') plane = Math.abs(P.forward[2]) >= 0.35 ? 'xy' : 'view';
    switch (plane) {
      case 'xy': return { point: w, normal: [0, 0, 1] };
      case 'xz': return { point: w, normal: [0, 1, 0] };
      case 'yz': return { point: w, normal: [1, 0, 0] };
      default: return { point: w, normal: P.forward };
    }
  }

  /** Called each frame by the player: apply orbit inertia. Returns true while still moving. */
  input.tick = function (dt) {
    const s = S(); if (!s || input.drag || s.view !== '3d') return false;
    const v = input.vel;
    if (Math.abs(v.yaw) < 0.02 && Math.abs(v.pitch) < 0.02) { v.yaw = v.pitch = 0; return false; }
    s.cam.yaw += v.yaw * dt; s.cam.pitch = LA.clamp(s.cam.pitch + v.pitch * dt, -LA.cam.PITCH_MAX, LA.cam.PITCH_MAX);
    const k = Math.exp(-dt * 7); v.yaw *= k; v.pitch *= k;
    return true;
  };
  input.stop = function () { input.vel = { yaw: 0, pitch: 0 }; input.drag = null; input.pinch = null; input.pointers.clear(); };

  LA.input = input;
})();
