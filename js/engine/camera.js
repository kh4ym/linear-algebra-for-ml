/* LA.cam — camera defaults and the per-frame projector for 2D (orthographic, pan/zoom) and 3D
   (right-handed, z-up orbit camera; perspective or orthographic). Camera parameters live in the lesson state
   (s.cam) so steps can tween them and snapshots capture them. */
window.LA = window.LA || {};
(function () {
  'use strict';
  const PITCH_MAX = 89 * Math.PI / 180;

  const cam = {
    /** 2D defaults: world center and visible half-height in world units. */
    defaults2d: () => ({ center: [0, 0], halfH: 3.6 }),
    /** 3D defaults: orbit target, distance, yaw/pitch (radians), vertical fov (degrees), ortho flag + halfH. */
    defaults3d: () => ({ target: [0, 0, 0], dist: 9.5, yaw: 0.62, pitch: 0.40, fov: 36, ortho: false, halfH: 4.2 }),
    /** Fill in any missing fields for the given view mode (mutates and returns c). */
    ensure(c, view) {
      c = c || {};
      const d = view === '3d' ? cam.defaults3d() : cam.defaults2d();
      Object.keys(d).forEach(k => { if (c[k] === undefined) c[k] = d[k]; });
      // keep both families present so view switches never hit undefined
      const d2 = view === '3d' ? cam.defaults2d() : cam.defaults3d();
      Object.keys(d2).forEach(k => { if (c[k] === undefined) c[k] = d2[k]; });
      c.pitch = LA.clamp(c.pitch, -PITCH_MAX, PITCH_MAX);
      return c;
    },
    PITCH_MAX,
    /** Unit vector the 3D camera looks along (from eye toward target). */
    forward(c) { const cp = Math.cos(c.pitch), sp = Math.sin(c.pitch); return [-cp * Math.cos(c.yaw), -cp * Math.sin(c.yaw), -sp]; },
    /** Camera that looks straight down the given axis ('x' | 'y' | 'z' or a vector); keeps dist. */
    lookAlong(c, axis) {
      const d = typeof axis === 'string' ? ({ x: [1, 0, 0], y: [0, 1, 0], z: [0, 0, 1] })[axis] : axis;
      const n = Math.hypot(d[0], d[1], d[2]) || 1, v = [d[0] / n, d[1] / n, d[2] / n];
      const out = Object.assign({}, c);
      out.pitch = LA.clamp(Math.asin(v[2]), -PITCH_MAX, PITCH_MAX);
      out.yaw = (Math.abs(v[0]) + Math.abs(v[1]) > 1e-9) ? Math.atan2(v[1], v[0]) : c.yaw;
      return out;
    },
    /** Convenience for steps: write camera fields toward `to` with progress u, unless the user took over. */
    tween(s, from, to, u) {
      if (s.cam && s.cam.auto === false) return s.cam;
      const base = s.cam || {};
      const out = LA.lerpObj(Object.assign({}, base, from), Object.assign({}, base, from, to), u);
      Object.keys(out).forEach(k => { base[k] = out[k]; });
      s.cam = base;
      return base;
    },
  };

  /** Build a projector for this frame. size = {w,h}; viewport = {x,y,w,h} in CSS px (defaults to full size). */
  cam.projector = function (view, c, size, viewport) {
    const vp = viewport || { x: 0, y: 0, w: size.w, h: size.h };
    const cx = vp.x + vp.w / 2, cy = vp.y + vp.h / 2;
    c = cam.ensure(c, view);
    if (view !== '3d') {
      const scale = (vp.h / 2) / c.halfH;
      const ox = c.center[0], oy = c.center[1];
      return {
        mode: '2d', vp, cx, cy, scale, pxPerUnit: scale, near: 0,
        toScreen(p) { return [cx + (p[0] - ox) * scale, cy - ((p[1] || 0) - oy) * scale, 0]; },
        toWorld(sx, sy) { return [(sx - cx) / scale + ox, -(sy - cy) / scale + oy]; },
        /** world-space bounds visible in this viewport */
        bounds() { const hw = (vp.w / 2) / scale, hh = c.halfH; return { x0: ox - hw, x1: ox + hw, y0: oy - hh, y1: oy + hh }; },
        depth() { return 0; },
      };
    }
    const cp = Math.cos(c.pitch), sp = Math.sin(c.pitch), cy_ = Math.cos(c.yaw), sy_ = Math.sin(c.yaw);
    const eye = [c.target[0] + c.dist * cp * cy_, c.target[1] + c.dist * cp * sy_, c.target[2] + c.dist * sp];
    const forward = norm3([c.target[0] - eye[0], c.target[1] - eye[1], c.target[2] - eye[2]]);
    const right = [-sy_, cy_, 0];
    const up = cross3(right, forward);
    const focal = (vp.h / 2) / Math.tan((c.fov * Math.PI / 180) / 2);
    const oscale = (vp.h / 2) / c.halfH;
    const near = 0.15;
    const P = {
      mode: '3d', vp, cx, cy, eye, forward, right, up, focal, near, ortho: !!c.ortho,
      pxPerUnit: c.ortho ? oscale : focal / c.dist,
      /** camera-space coords [xc, yc, zc] (zc > 0 in front of the eye) */
      toCam(p) {
        const dx = p[0] - eye[0], dy = p[1] - eye[1], dz = (p[2] || 0) - eye[2];
        return [dx * right[0] + dy * right[1] + dz * right[2], dx * up[0] + dy * up[1] + dz * up[2], dx * forward[0] + dy * forward[1] + dz * forward[2]];
      },
      /** camera-space → screen */
      camToScreen(q) {
        if (c.ortho) return [cx + q[0] * oscale, cy - q[1] * oscale, q[2]];
        const z = Math.max(q[2], near);
        return [cx + q[0] / z * focal, cy - q[1] / z * focal, q[2]];
      },
      toScreen(p) { return P.camToScreen(P.toCam(p)); },
      depth(p) { const q = P.toCam(p); return q[2]; },
      /** picking ray through a screen point */
      ray(sx, sy) {
        if (c.ortho) {
          const xc = (sx - cx) / oscale, yc = -(sy - cy) / oscale;
          return { origin: [eye[0] + right[0] * xc + up[0] * yc, eye[1] + right[1] * xc + up[1] * yc, eye[2] + right[2] * xc + up[2] * yc], dir: forward.slice() };
        }
        const xc = (sx - cx) / focal, yc = -(sy - cy) / focal;
        const d = norm3([right[0] * xc + up[0] * yc + forward[0], right[1] * xc + up[1] * yc + forward[1], right[2] * xc + up[2] * yc + forward[2]]);
        return { origin: eye.slice(), dir: d };
      },
      /** intersection of the ray through (sx,sy) with the plane through p0 with normal n; null if parallel */
      pick(sx, sy, p0, n) {
        const r = P.ray(sx, sy);
        const den = r.dir[0] * n[0] + r.dir[1] * n[1] + r.dir[2] * n[2];
        if (Math.abs(den) < 1e-6) return null;
        const t = ((p0[0] - r.origin[0]) * n[0] + (p0[1] - r.origin[1]) * n[1] + (p0[2] - r.origin[2]) * n[2]) / den;
        if (t < 0) return null;
        return [r.origin[0] + r.dir[0] * t, r.origin[1] + r.dir[1] * t, r.origin[2] + r.dir[2] * t];
      },
      /** approximate screen px per world unit at a point (for sizing arrowheads/labels) */
      scaleAt(p) { if (c.ortho) return oscale; const z = Math.max(P.depth(p), near); return focal / z; },
    };
    return P;
  };

  function norm3(v) { const l = Math.hypot(v[0], v[1], v[2]) || 1; return [v[0] / l, v[1] / l, v[2] / l]; }
  function cross3(a, b) { return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]; }
  cam.norm3 = norm3; cam.cross3 = cross3;

  LA.cam = cam;
})();
