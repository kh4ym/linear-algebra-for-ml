/* LA.Gfx — immediate-mode drawing API shared by 2D and 3D scenes.
   Every primitive produces screen-space records in a per-frame buffer; records carry a depth (camera-space z in
   3D, insertion order in 2D) and are painted far→near at end(). Text, markers and handles go to an overlay pass. */
window.LA = window.LA || {};
(function () {
  'use strict';
  const C = () => LA.color;
  const role = c => LA.color.role(c);

  class Gfx {
    constructor() {
      this.records = []; this.overlay = []; this.handles = [];
      this.handlesEnabled = false; this.hot = null; this.dragging = null;
      this.instant = false; this.time = 0;
      this.ctx = null; this.size = { w: 300, h: 150, dpr: 1 }; this.P = null; this.view = '2d'; this.cam = null;
      this.stats = { records: 0 };
      this._vpStack = [];
    }

    /** Start a frame. s is the lesson state (reads s.view / s.cam). */
    begin(ctx, size, s, opts) {
      opts = opts || {};
      this.ctx = ctx; this.size = size; this.time = opts.time || 0; this.instant = !!opts.instant;
      this.records.length = 0; this.overlay.length = 0; this.handles.length = 0; this.stats.records = 0;
      this.view = (s && s.view === '3d') ? '3d' : '2d';
      if (s) { s.cam = LA.cam.ensure(s.cam, this.view); this.cam = s.cam; } else { this.cam = LA.cam.ensure({}, this.view); }
      this.P = LA.cam.projector(this.view, this.cam, size);
      ctx.save();
      ctx.fillStyle = opts.bg || C().bg;
      ctx.fillRect(0, 0, size.w, size.h);
      ctx.restore();
      ctx.lineCap = 'round'; ctx.lineJoin = 'round';
      return this;
    }

    get is3d() { return this.P && this.P.mode === '3d'; }

    /** Draw a sub-scene in a rectangle (fractions of the canvas or px). vopts: {view, cam, border}. */
    viewport(rect, vopts, fn) {
      const S = this.size;
      const r = { x: rect.x <= 1 ? rect.x * S.w : rect.x, y: rect.y <= 1 ? rect.y * S.h : rect.y, w: rect.w <= 1 ? rect.w * S.w : rect.w, h: rect.h <= 1 ? rect.h * S.h : rect.h };
      // paint what the main scene has so far, so the inset sits on top of it (call order = paint order)
      this._flush(this.records); this.records.length = 0;
      const saved = { P: this.P, view: this.view, cam: this.cam, records: this.records, overlay: this.overlay };
      this.records = []; this.overlay = [];
      this.view = vopts && vopts.view === '3d' ? '3d' : '2d';
      const c = LA.cam.ensure(vopts && vopts.cam ? vopts.cam : (this.view === '3d' ? LA.cam.defaults3d() : LA.cam.defaults2d()), this.view);
      this.cam = c;
      this.P = LA.cam.projector(this.view, c, S, r);
      const ctx = this.ctx;
      ctx.save(); ctx.beginPath(); ctx.rect(r.x, r.y, r.w, r.h); ctx.clip();
      if (vopts && vopts.bg) { ctx.fillStyle = role(vopts.bg); ctx.fillRect(r.x, r.y, r.w, r.h); }
      try { fn(this, r); } finally {
        this._flush(this.records); this._paintOverlay(this.overlay);
        ctx.restore();
        if (vopts && vopts.border !== false) { ctx.save(); ctx.strokeStyle = C().line; ctx.lineWidth = 1; ctx.strokeRect(r.x + 0.5, r.y + 0.5, r.w - 1, r.h - 1); ctx.restore(); }
        this.P = saved.P; this.view = saved.view; this.cam = saved.cam; this.records = saved.records; this.overlay = saved.overlay;
      }
    }
    /** Two side-by-side viewports. opts: {ratio, left:{view,cam}, right:{view,cam}} */
    split(fnL, fnR, opts) {
      opts = opts || {}; const ratio = opts.ratio || 0.5;
      this.viewport({ x: 0, y: 0, w: ratio, h: 1 }, opts.left || {}, fnL);
      this.viewport({ x: ratio, y: 0, w: 1 - ratio, h: 1 }, opts.right || {}, fnR);
    }

    /* ---------------- projection helpers ---------------- */
    _toCam(p) { return this.is3d ? this.P.toCam(p) : null; }
    /** world → [sx, sy, depth]; null if behind the camera (3D) */
    _proj(p) {
      if (!this.is3d) return this.P.toScreen(p);
      const q = this.P.toCam(p);
      if (q[2] < this.P.near) return null;
      return this.P.camToScreen(q);
    }
    /** screen-space px per world unit near a point */
    _scaleAt(p) { return this.is3d ? this.P.scaleAt(p) : this.P.scale; }
    _push(rec) { this.records.push(rec); this.stats.records++; }

    /* ---------------- core record builders ---------------- */
    /** segment between world points */
    _seg(a, b, st) {
      if (!this.is3d) {
        const A = this.P.toScreen(a), B = this.P.toScreen(b);
        this._push({ k: 's', a: A, b: B, d: 0, c: st.color, w: st.width, dash: st.dash, al: st.alpha });
        return;
      }
      let qa = this.P.toCam(a), qb = this.P.toCam(b);
      const n = this.P.near;
      if (qa[2] < n && qb[2] < n) return;
      if (qa[2] < n || qb[2] < n) {                 // lerp-clip against the near plane
        const t = (n - qa[2]) / (qb[2] - qa[2]);
        const m = [qa[0] + (qb[0] - qa[0]) * t, qa[1] + (qb[1] - qa[1]) * t, n];
        if (qa[2] < n) qa = m; else qb = m;
      }
      const A = this.P.camToScreen(qa), B = this.P.camToScreen(qb);
      const d = (qa[2] + qb[2]) / 2 - (st.bias === undefined ? 0.004 : st.bias);
      this._push({ k: 's', a: A, b: B, d: st.back ? 1e9 : d, c: st.color, w: st.width, dash: st.dash, al: st.alpha });
    }
    /** filled/stroked polygon from world points */
    _poly(pts, st) {
      if (!this.is3d) {
        const sp = pts.map(p => this.P.toScreen(p));
        this._push({ k: 'p', pts: sp, d: 0, f: st.fill, c: st.stroke, w: st.width, al: st.alpha, dash: st.dash });
        return;
      }
      let q = pts.map(p => this.P.toCam(p));
      const n = this.P.near;
      if (q.every(v => v[2] < n)) return;
      if (q.some(v => v[2] < n)) q = clipNear(q, n);
      if (q.length < 3) return;
      let zs = 0; for (const v of q) zs += v[2];
      const d = zs / q.length + (st.bias || 0);
      const sp = q.map(v => this.P.camToScreen(v));
      this._push({ k: 'p', pts: sp, d: st.back ? 1e9 : d, f: st.fill, c: st.stroke, w: st.width, al: st.alpha, dash: st.dash });
    }
    _dot(p, st) {
      const s = this._proj(p); if (!s) return;
      this._push({ k: 'd', p: s, d: s[2] - 0.006, r: st.r, c: st.color, al: st.alpha, sc: st.stroke, sw: st.strokeWidth });
    }

    /* ---------------- public primitives ---------------- */
    /** Grid of lines transformed by M (2×2 or 3×3, or null for identity). opts: range, step, color, alpha, width, axis(bool), back(bool), z */
    grid(M, opts) {
      opts = opts || {};
      const range = opts.range || (this.is3d ? 4 : 10), step = opts.step || 1;
      const color = role(opts.color || 'grid'), axisColor = role(opts.axisColor || opts.color || 'gridAxis');
      const alpha = opts.alpha === undefined ? 1 : opts.alpha, width = opts.width || 1;
      const T = p => applyM(M, p, this.is3d, opts.z || 0);
      const seg = this.is3d ? 1 : (2 * range);            // subdivide in 3D for depth sorting
      const vis = this.is3d ? null : this.P.bounds();
      for (let i = -range; i <= range + 1e-9; i += step) {
        const isAxis = Math.abs(i) < 1e-9;
        const st = { color: isAxis && opts.axis !== false ? axisColor : color, width: isAxis && opts.axis !== false ? width * 1.6 : width, alpha, back: opts.back, dash: opts.dash };
        for (let j = -range; j < range - 1e-9; j += (2 * range) / seg) {
          const j2 = j + (2 * range) / seg;
          const a1 = T([i, j]), b1 = T([i, j2]);            // line x = i
          const a2 = T([j, i]), b2 = T([j2, i]);            // line y = i
          if (!vis || segVisible(a1, b1, vis)) this._seg(a1, b1, st);
          if (!vis || segVisible(a2, b2, vis)) this._seg(a2, b2, st);
        }
      }
    }
    /** The faded, untransformed reference grid. */
    ghostGrid(opts) { this.grid(null, Object.assign({ color: 'ghost', axisColor: 'ghostAxis', alpha: 1, axis: true, back: true }, opts || {})); }
    /** Axes with ticks and numbers (2D) or three labelled arrows (3D). opts: ticks, labels, length, names */
    axes(opts) {
      opts = opts || {};
      if (!this.is3d) {
        const b = this.P.bounds();
        const st = { color: role(opts.color || 'gridAxis'), width: 1.6, alpha: 1 };
        this._seg([b.x0, 0], [b.x1, 0], st); this._seg([0, b.y0], [0, b.y1], st);
        if (opts.ticks !== false) {
          const tl = 4 / this.P.scale;
          for (let i = Math.ceil(b.x0); i <= Math.floor(b.x1); i++) if (i !== 0) { this._seg([i, -tl], [i, tl], st); if (opts.labels !== false && this.P.scale > 22) this.label(String(i), [i, 0], { color: 'tick', size: 11, anchor: 'below', dy: 4, font: 'ui' }); }
          for (let i = Math.ceil(b.y0); i <= Math.floor(b.y1); i++) if (i !== 0) { this._seg([-tl, i], [tl, i], st); if (opts.labels !== false && this.P.scale > 22) this.label(String(i), [0, i], { color: 'tick', size: 11, anchor: 'left', dx: -5, font: 'ui' }); }
        }
        if (opts.names) { this.label(opts.names[0] || 'x', [b.x1, 0], { anchor: 'right', dx: -8, dy: -12, color: 'hud', font: 'math' }); this.label(opts.names[1] || 'y', [0, b.y1], { anchor: 'left', dx: 10, dy: 12, color: 'hud', font: 'math' }); }
        return;
      }
      const L = opts.length || 4, names = opts.names || ['x', 'y', 'z'];
      const cols = opts.colors || ['gridAxis', 'gridAxis', 'gridAxis'];
      const dirs = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
      dirs.forEach((d, i) => {
        this.vector([-L * d[0], -L * d[1], -L * d[2]], [L * d[0], L * d[1], L * d[2]], { color: cols[i], width: 1.4, head: 8, alpha: 0.9, bias: 0.002 });
        if (opts.labels !== false) this.label(names[i], [L * 1.08 * d[0], L * 1.08 * d[1], L * 1.08 * d[2]], { color: 'hud', font: 'math' });
      });
    }
    /** Arrow. vector(to, opts) or vector(from, to, opts). opts: color, width, head, label, labelAt(0..1), dashed, alpha, components, hat, dx, dy */
    vector(a, b, opts) {
      if (!Array.isArray(b)) { opts = b; b = a; a = b.length === 3 ? [0, 0, 0] : [0, 0]; }
      opts = opts || {};
      const color = role(opts.color || 'v'), width = opts.width || 2.6, alpha = opts.alpha === undefined ? 1 : opts.alpha;
      const A = this._proj(a), B = this._proj(b);
      if (opts.components) this._components(a, b, opts);
      if (!A || !B) { this._seg(a, b, { color, width, alpha, dash: opts.dashed ? [6, 5] : null }); return; }
      const dx = B[0] - A[0], dy = B[1] - A[1], len = Math.hypot(dx, dy);
      let head = opts.head === undefined ? 12 : opts.head;
      if (len < 1e-6) { if (opts.label) this.label(opts.label, b, { color, hat: opts.hat, dx: opts.dx, dy: opts.dy }); return; }
      head = Math.min(head, len * 0.6);
      const ux = dx / len, uy = dy / len;
      // shaft stops short of the tip so the head is crisp
      const sb = this.is3d ? b : this.P.toWorld(B[0] - ux * head * 0.75, B[1] - uy * head * 0.75);
      if (this.is3d) {
        // shorten in world space proportionally
        const f = Math.max(0, 1 - (head * 0.75) / len);
        const bw = [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, (a[2] || 0) + ((b[2] || 0) - (a[2] || 0)) * f];
        this._seg(a, bw, { color, width, alpha, dash: opts.dashed ? [6, 5] : null, bias: opts.bias === undefined ? 0.004 : opts.bias });
        const q = this.P.toCam(b);
        if (q[2] >= this.P.near) {
          const tip = this.P.camToScreen(q);
          const hw = head * 0.42;
          const p1 = [tip[0] - ux * head - uy * hw, tip[1] - uy * head + ux * hw], p2 = [tip[0] - ux * head + uy * hw, tip[1] - uy * head - ux * hw];
          this._push({ k: 'p', pts: [tip, p1, p2], d: q[2] - 0.008, f: color, al: alpha });
        }
      } else {
        this._seg(a, sb, { color, width, alpha, dash: opts.dashed ? [6, 5] : null });
        const hw = head * 0.42;
        const p1 = [B[0] - ux * head - uy * hw, B[1] - uy * head + ux * hw], p2 = [B[0] - ux * head + uy * hw, B[1] - uy * head - ux * hw];
        this._push({ k: 'p', pts: [B, p1, p2], d: 0, f: color, al: alpha });
      }
      if (opts.label) {
        const t = opts.labelAt === undefined ? 1 : opts.labelAt;
        const lp = a.map((x, i) => x + ((b[i] || 0) - x) * t);
        // offset label perpendicular-ish to the arrow
        const off = opts.labelOffset === undefined ? 14 : opts.labelOffset;
        const ddx = opts.dx !== undefined ? opts.dx : -uy * off + ux * (t >= 1 ? 8 : 0);
        const ddy = opts.dy !== undefined ? opts.dy : ux * off + uy * (t >= 1 ? 8 : 0);
        this.label(opts.label, lp, { color: opts.labelColor || color, dx: ddx, dy: ddy, hat: opts.hat, size: opts.labelSize, font: opts.labelFont });
      }
    }
    _components(a, b, opts) {
      const n = b.length;
      const cols = [C().i, C().j, C().k];
      const st = (c) => ({ color: c, width: 1.3, dash: [4, 4], alpha: 0.85 });
      if (n === 2) {
        this._seg([b[0], a[1]], b, st(cols[1]));          // vertical: y-component
        this._seg([a[0], a[1]], [b[0], a[1]], st(cols[0])); // horizontal: x-component
        if (opts.components === 'labels') {
          this.label(LA.fmt(b[0] - a[0]), [(a[0] + b[0]) / 2, a[1]], { color: cols[0], anchor: 'below', dy: 6, size: 12 });
          this.label(LA.fmt(b[1] - a[1]), [b[0], (a[1] + b[1]) / 2], { color: cols[1], anchor: 'left', dx: 8, size: 12 });
        }
      } else {
        const p1 = [b[0], a[1], a[2]], p2 = [b[0], b[1], a[2]];
        this._seg(a, p1, st(cols[0])); this._seg(p1, p2, st(cols[1])); this._seg(p2, b, st(cols[2]));
        this._seg([a[0], a[1], a[2]], [b[0], b[1], a[2]], { color: C().residual, width: 1, dash: [3, 4], alpha: 0.6 });
      }
    }
    point(p, opts) {
      opts = opts || {};
      this._dot(p, { r: opts.r || 4.5, color: role(opts.color || 'v'), alpha: opts.alpha === undefined ? 1 : opts.alpha, stroke: opts.stroke ? role(opts.stroke) : null, strokeWidth: opts.strokeWidth || 1.5 });
      if (opts.label) this.label(opts.label, p, { color: role(opts.labelColor || opts.color || 'v'), dx: opts.dx === undefined ? 9 : opts.dx, dy: opts.dy === undefined ? -9 : opts.dy, hat: opts.hat, size: opts.size });
    }
    /** Many points. opts: r, color | colors(array or fn(i,p)), alpha */
    points(arr, opts) {
      opts = opts || {};
      const r = opts.r || 3, alpha = opts.alpha === undefined ? 0.95 : opts.alpha;
      const base = role(opts.color || 'v');
      for (let i = 0; i < arr.length; i++) {
        const c = opts.colors ? (typeof opts.colors === 'function' ? role(opts.colors(i, arr[i])) : role(opts.colors[i % opts.colors.length])) : base;
        this._dot(arr[i], { r, color: c, alpha });
      }
    }
    line(a, b, opts) {
      opts = opts || {};
      this._seg(a, b, { color: role(opts.color || 'hud'), width: opts.width || 1.5, dash: opts.dashed ? (Array.isArray(opts.dashed) ? opts.dashed : [6, 5]) : opts.dash, alpha: opts.alpha === undefined ? 1 : opts.alpha, back: opts.back, bias: opts.bias });
    }
    /** A long line through p in direction d (len world units each way). */
    lineThrough(p, d, opts) {
      opts = opts || {}; const L = opts.len || 20;
      const n = Math.hypot.apply(null, d) || 1;
      const a = p.map((x, i) => x - d[i] / n * L), b = p.map((x, i) => x + d[i] / n * L);
      if (this.is3d) { const N = 10; for (let k = 0; k < N; k++) this.line(LA.lerpV(a, b, k / N), LA.lerpV(a, b, (k + 1) / N), opts); }
      else this.line(a, b, opts);
    }
    polyline(pts, opts) {
      opts = opts || {};
      const st = { color: role(opts.color || 'hud'), width: opts.width || 1.8, dash: opts.dashed ? [6, 5] : opts.dash, alpha: opts.alpha === undefined ? 1 : opts.alpha, bias: opts.bias };
      if (!this.is3d && !opts.segmented) {
        const sp = pts.map(p => this.P.toScreen(p));
        this._push({ k: 'l', pts: sp, d: 0, c: st.color, w: st.width, dash: st.dash, al: st.alpha, closed: opts.closed });
        return;
      }
      for (let i = 0; i < pts.length - 1; i++) this._seg(pts[i], pts[i + 1], st);
      if (opts.closed && pts.length > 2) this._seg(pts[pts.length - 1], pts[0], st);
    }
    /** Filled polygon. opts: fill, alpha, stroke, width, subdiv (quads only), back */
    polygon(pts, opts) {
      opts = opts || {};
      const st = { fill: opts.fill ? role(opts.fill) : null, stroke: opts.stroke ? role(opts.stroke) : null, width: opts.width || 1.2, alpha: opts.alpha === undefined ? 0.35 : opts.alpha, back: opts.back, bias: opts.bias, dash: opts.dash };
      if (this.is3d && pts.length === 4 && (opts.subdiv || opts.subdiv === undefined && bigQuad(pts))) {
        const n = opts.subdiv || 4;
        for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
          const q = [bilerp(pts, i / n, j / n), bilerp(pts, (i + 1) / n, j / n), bilerp(pts, (i + 1) / n, (j + 1) / n), bilerp(pts, i / n, (j + 1) / n)];
          this._poly(q, Object.assign({}, st, { stroke: null }));
        }
        if (st.stroke) this.polyline(pts, { color: st.stroke, width: st.width, closed: true, alpha: Math.min(1, st.alpha + 0.4) });
        return;
      }
      this._poly(pts, st);
    }
    /** Parallelogram spanned by u and v from origin o. opts as polygon (+ 'orient': fill flips color when det<0). */
    parallelogram(u, v, opts) {
      opts = opts || {};
      const o = opts.origin || u.map(() => 0);
      const pts = [o, add(o, u), add(add(o, u), v), add(o, v)];
      let fill = opts.fill || 'v';
      if (opts.orient && u.length === 2) { const det = u[0] * v[1] - u[1] * v[0]; if (det < 0) fill = 'neg'; }
      this.polygon(pts, Object.assign({}, opts, { fill, stroke: opts.stroke === undefined ? fill : opts.stroke }));
    }
    circle(c, r, opts) {
      opts = opts || {};
      const n = opts.segments || 72, pts = [];
      for (let i = 0; i < n; i++) { const a = i / n * Math.PI * 2; pts.push(c.length === 3 ? [c[0] + r * Math.cos(a), c[1] + r * Math.sin(a), c[2]] : [c[0] + r * Math.cos(a), c[1] + r * Math.sin(a)]); }
      this._closedShape(pts, opts);
    }
    /** Image of the unit circle under M (2×2 or 3×3 acting on the xy-circle), translated by c. */
    ellipse(M, c, opts) {
      opts = opts || {};
      const n = opts.segments || 72, pts = [];
      c = c || (M.length === 3 ? [0, 0, 0] : [0, 0]);
      for (let i = 0; i < n; i++) { const a = i / n * Math.PI * 2; const p = applyM(M, [Math.cos(a), Math.sin(a)], M.length === 3, 0); pts.push(p.map((x, k) => x + (c[k] || 0))); }
      this._closedShape(pts, opts);
    }
    _closedShape(pts, opts) {
      if (opts.fill) this.polygon(pts, { fill: opts.fill, alpha: opts.alpha === undefined ? 0.2 : opts.alpha, subdiv: 0 });
      if (opts.stroke !== null) this.polyline(pts, { color: opts.stroke || opts.color || 'v', width: opts.width || 1.8, closed: true, dashed: opts.dashed, alpha: opts.strokeAlpha === undefined ? 1 : opts.strokeAlpha, segmented: this.is3d });
    }
    /** Angle arc from direction a to direction b around center. opts: r (world), color, label, width */
    arc(center, a, b, opts) {
      opts = opts || {};
      const r = opts.r || 0.6, n = 24, pts = [];
      const na = Math.hypot.apply(null, a) || 1, nb = Math.hypot.apply(null, b) || 1;
      const ua = a.map(x => x / na), ub = b.map(x => x / nb);
      for (let i = 0; i <= n; i++) { const t = i / n; const d = slerp(ua, ub, t); pts.push(center.map((x, k) => x + d[k] * r)); }
      this.polyline(pts, { color: opts.color || 'hud', width: opts.width || 1.4, segmented: this.is3d });
      if (opts.right && Math.abs(dot(ua, ub)) < 0.03) { // right-angle marker
        const q = center.map((x, k) => x + (ua[k] + ub[k]) * r * 0.7);
        this.polyline([center.map((x, k) => x + ua[k] * r * 0.7), q, center.map((x, k) => x + ub[k] * r * 0.7)], { color: opts.color || 'hud', width: 1.2 });
      }
      if (opts.label) { const m = slerp(ua, ub, 0.5); this.label(opts.label, center.map((x, k) => x + m[k] * r * 1.45), { color: opts.color || 'hud', size: opts.size || 13 }); }
    }
    /** Dimension line between a and b with end ticks and a centered label. opts: offset(px), label, color */
    dim(a, b, opts) {
      opts = opts || {};
      const A = this._proj(a), B = this._proj(b); if (!A || !B) return;
      const dx = B[0] - A[0], dy = B[1] - A[1], len = Math.hypot(dx, dy) || 1, nx = -dy / len, ny = dx / len;
      const off = opts.offset === undefined ? 14 : opts.offset;
      const a2 = [A[0] + nx * off, A[1] + ny * off], b2 = [B[0] + nx * off, B[1] + ny * off];
      const color = role(opts.color || 'hud');
      this.overlay.push({ k: 'sseg', a: a2, b: b2, c: color, w: 1.2 });
      this.overlay.push({ k: 'sseg', a: [a2[0] - nx * 4, a2[1] - ny * 4], b: [a2[0] + nx * 4, a2[1] + ny * 4], c: color, w: 1.2 });
      this.overlay.push({ k: 'sseg', a: [b2[0] - nx * 4, b2[1] - ny * 4], b: [b2[0] + nx * 4, b2[1] + ny * 4], c: color, w: 1.2 });
      if (opts.label) this.overlay.push({ k: 't', text: opts.label, p: [(a2[0] + b2[0]) / 2 + nx * 12, (a2[1] + b2[1]) / 2 + ny * 12], c: color, size: opts.size || 12.5, anchor: 'center', font: 'math' });
    }
    /** Vector field: small arrows at grid points (2D). opts: bounds{x0,x1,y0,y1}, step, scale, color, alpha */
    field(fn, opts) {
      opts = opts || {};
      const b = opts.bounds || this.P.bounds(), step = opts.step || 0.5, sc = opts.scale === undefined ? 0.25 : opts.scale;
      const color = role(opts.color || 'teal');
      for (let x = Math.ceil(b.x0 / step) * step; x <= b.x1; x += step) for (let y = Math.ceil(b.y0 / step) * step; y <= b.y1; y += step) {
        const v = fn([x, y]); if (!v) continue;
        const n = Math.hypot(v[0], v[1]); if (n < 1e-9) continue;
        const L = sc * (opts.normalize ? 1 : Math.min(n, opts.maxLen || 2) / 1);
        this.vector([x, y], [x + v[0] / n * L, y + v[1] / n * L], { color, width: 1.2, head: 6, alpha: opts.alpha === undefined ? 0.8 : opts.alpha });
      }
    }
    /** Scalar field heatmap (2D): fnOrGrid is fn([x,y])→value or a matrix of values. bounds {x0,x1,y0,y1}. opts: res, range[lo,hi], ramp fn(t)→color, alpha */
    heat2d(fnOrGrid, bounds, opts) {
      opts = opts || {};
      if (this.is3d) { if (this.heat3d) this.heat3d(fnOrGrid, Object.assign({ bounds }, opts)); return; }
      const res = opts.res || 64;
      const grid = typeof fnOrGrid === 'function' ? sampleGrid(fnOrGrid, bounds, res) : fnOrGrid;
      const img = gridToImage(grid, opts.range, opts.ramp || LA.color.heat);
      const A = this.P.toScreen([bounds.x0, bounds.y1]), B = this.P.toScreen([bounds.x1, bounds.y0]);
      this._push({ k: 'img', img, x: A[0], y: A[1], w: B[0] - A[0], h: B[1] - A[1], d: 0, al: opts.alpha === undefined ? 0.85 : opts.alpha, smooth: opts.smooth !== false });
    }
    /** Grayscale/colored matrix as an image (2D). opts: ramp (default gray), range, alpha, cells(bool: draw cell borders) */
    image(grid, bounds, opts) {
      opts = opts || {};
      if (this.is3d) { this.heat3d(grid, Object.assign({ bounds, ramp: opts.ramp || grayRamp, res: Math.min(32, grid.length) }, opts)); return; }
      const img = gridToImage(grid, opts.range || [0, 1], opts.ramp || grayRamp);
      const A = this.P.toScreen([bounds.x0, bounds.y1]), B = this.P.toScreen([bounds.x1, bounds.y0]);
      this._push({ k: 'img', img, x: A[0], y: A[1], w: B[0] - A[0], h: B[1] - A[1], d: 0, al: opts.alpha === undefined ? 1 : opts.alpha, smooth: !!opts.smooth });
      if (opts.cells) {
        const n = grid.length, m = grid[0].length, st = { color: C().line, width: 0.6, alpha: 0.8 };
        for (let i = 0; i <= n; i++) { const y = bounds.y1 - (bounds.y1 - bounds.y0) * i / n; this._seg([bounds.x0, y], [bounds.x1, y], st); }
        for (let j = 0; j <= m; j++) { const x = bounds.x0 + (bounds.x1 - bounds.x0) * j / m; this._seg([x, bounds.y0], [x, bounds.y1], st); }
      }
    }
    /** Contour lines of fn over bounds at given levels. opts: res, color|colors, width, z (3D floor height), lift (3D: draw on surface) */
    contours(fn, levels, bounds, opts) {
      opts = opts || {};
      const res = opts.res || 48;
      const grid = sampleGrid(fn, bounds, res);
      levels.forEach((lv, li) => {
        const segs = marchingSquares(grid, bounds, lv);
        const color = opts.colors ? role(opts.colors[li % opts.colors.length]) : role(opts.color || 'teal');
        const st = { color, width: opts.width || 1.3, alpha: opts.alpha === undefined ? 0.9 : opts.alpha };
        for (const sg of segs) {
          let a = sg[0], b = sg[1];
          if (this.is3d) { const z = opts.lift ? lv : (opts.z || 0); a = [a[0], a[1], z]; b = [b[0], b[1], z]; }
          this._seg(a, b, st);
        }
      });
    }
    /** Bars. values array; opts: origin [x,y] world or screen:true with x,y,w,h px; width, gap, colors, labels, max, color */
    bars(values, opts) {
      opts = opts || {};
      const n = values.length; if (!n) return;
      const max = opts.max || Math.max(1e-9, ...values.map(Math.abs));
      if (opts.screen) {
        const x = opts.x, y = opts.y, w = opts.w || 160, h = opts.h || 80, gap = opts.gap === undefined ? 4 : opts.gap;
        const bw = (w - gap * (n - 1)) / n;
        this.overlay.push({ k: 'sseg', a: [x, y + h], b: [x + w, y + h], c: C().line, w: 1 });
        for (let i = 0; i < n; i++) {
          const v = values[i], bh = Math.abs(v) / max * h;
          const c = opts.colors ? role(opts.colors[i % opts.colors.length]) : role(opts.color || 'gold');
          this.overlay.push({ k: 'srect', x: x + i * (bw + gap), y: y + h - bh, w: bw, h: bh, c, al: opts.alpha === undefined ? 0.9 : opts.alpha });
          if (opts.labels) this.overlay.push({ k: 't', text: String(opts.labels[i]), p: [x + i * (bw + gap) + bw / 2, y + h + 10], c: C().muted, size: 11, anchor: 'center', font: 'ui' });
          if (opts.values) this.overlay.push({ k: 't', text: LA.fmt(v), p: [x + i * (bw + gap) + bw / 2, y + h - bh - 8], c: C().hud, size: 11, anchor: 'center', font: 'ui' });
        }
        if (opts.title) this.overlay.push({ k: 't', text: opts.title, p: [x, y - 10], c: C().muted, size: 12, anchor: 'left', font: 'ui' });
        return;
      }
      const o = opts.origin || [0, 0], bw = opts.width || 0.6, gap = opts.gap === undefined ? 0.25 : opts.gap, H = opts.height || 2;
      for (let i = 0; i < n; i++) {
        const v = values[i], bh = v / max * H, x0 = o[0] + i * (bw + gap);
        const c = opts.colors ? role(opts.colors[i % opts.colors.length]) : role(opts.color || 'gold');
        this.polygon([[x0, o[1]], [x0 + bw, o[1]], [x0 + bw, o[1] + bh], [x0, o[1] + bh]], { fill: c, alpha: 0.85, stroke: c });
        if (opts.labels) this.label(String(opts.labels[i]), [x0 + bw / 2, o[1]], { anchor: 'below', dy: 4, size: 11, color: 'muted', font: 'ui' });
        if (opts.values) this.label(LA.fmt(v), [x0 + bw / 2, o[1] + bh], { anchor: 'above', dy: -3, size: 11, color: 'hud', font: 'ui' });
      }
    }
    /** Histogram of values into `bins` bins over [lo,hi] (screen-space HUD). opts: x,y,w,h, range, color, title, marks[] */
    hist(values, bins, opts) {
      opts = opts || {};
      const lo = opts.range ? opts.range[0] : Math.min(...values), hi = opts.range ? opts.range[1] : Math.max(...values);
      const counts = new Array(bins).fill(0);
      for (const v of values) { let b = Math.floor((v - lo) / (hi - lo) * bins); if (b >= bins) b = bins - 1; if (b < 0) b = 0; counts[b]++; }
      this.bars(counts, { screen: true, x: opts.x, y: opts.y, w: opts.w || 200, h: opts.h || 80, gap: 1, color: opts.color || 'teal', title: opts.title, alpha: 0.8 });
      const x = opts.x, y = opts.y, w = opts.w || 200, h = opts.h || 80;
      this.overlay.push({ k: 't', text: LA.fmt(lo), p: [x, y + h + 10], c: C().muted, size: 10.5, anchor: 'center', font: 'ui' });
      this.overlay.push({ k: 't', text: LA.fmt(hi), p: [x + w, y + h + 10], c: C().muted, size: 10.5, anchor: 'center', font: 'ui' });
      if (opts.marks) for (const m of opts.marks) { const mx = x + (m.at - lo) / (hi - lo) * w; this.overlay.push({ k: 'sseg', a: [mx, y], b: [mx, y + h], c: role(m.color || 'gold'), w: 1.2, dash: [3, 3] }); if (m.label) this.overlay.push({ k: 't', text: m.label, p: [mx, y - 6], c: role(m.color || 'gold'), size: 11, anchor: 'center', font: 'ui' }); }
    }
    /** Small line chart (screen-space HUD). series: [{points:[[x,y]...], color, label}], opts: x,y,w,h, xRange, yRange, title, logY */
    lineChart(series, opts) {
      opts = opts || {};
      const x = opts.x, y = opts.y, w = opts.w || 220, h = opts.h || 100;
      let xs = [], ys = [];
      series.forEach(s => s.points.forEach(p => { xs.push(p[0]); ys.push(p[1]); }));
      if (!xs.length) return;
      const xr = opts.xRange || [Math.min(...xs), Math.max(...xs)], yr = opts.yRange || [Math.min(...ys), Math.max(...ys)];
      const X = v => x + (v - xr[0]) / ((xr[1] - xr[0]) || 1) * w, Y = v => y + h - (v - yr[0]) / ((yr[1] - yr[0]) || 1) * h;
      this.overlay.push({ k: 'srect', x, y, w, h, c: C().panel, al: 0.75 });
      this.overlay.push({ k: 'sseg', a: [x, y + h], b: [x + w, y + h], c: C().line, w: 1 });
      this.overlay.push({ k: 'sseg', a: [x, y], b: [x, y + h], c: C().line, w: 1 });
      series.forEach(s => {
        const pts = s.points.map(p => [X(p[0]), Y(p[1])]);
        this.overlay.push({ k: 'sline', pts, c: role(s.color || 'teal'), w: s.width || 1.8, dash: s.dash });
        if (s.label) { const last = pts[pts.length - 1]; this.overlay.push({ k: 't', text: s.label, p: [last[0] + 6, last[1]], c: role(s.color || 'teal'), size: 11, anchor: 'left', font: 'ui' }); }
      });
      if (opts.title) this.overlay.push({ k: 't', text: opts.title, p: [x, y - 8], c: C().muted, size: 12, anchor: 'left', font: 'ui' });
      if (opts.xLabel) this.overlay.push({ k: 't', text: opts.xLabel, p: [x + w / 2, y + h + 12], c: C().muted, size: 11, anchor: 'center', font: 'ui' });
      if (opts.yLabel) this.overlay.push({ k: 't', text: opts.yLabel, p: [x - 4, y - 8], c: C().muted, size: 11, anchor: 'right', font: 'ui' });
      if (opts.hline !== undefined) this.overlay.push({ k: 'sseg', a: [x, Y(opts.hline)], b: [x + w, Y(opts.hline)], c: role(opts.hlineColor || 'bad'), w: 1.2, dash: [4, 4] });
    }
    /** The n-D "strip" view: a long vector as a row of colored cells (screen-space). opts: x,y,w,h, range, ramp, title, highlight(index) */
    strip(values, opts) {
      opts = opts || {};
      const x = opts.x, y = opts.y, w = opts.w || 300, h = opts.h || 18, n = values.length;
      const r = opts.range || [Math.min(...values), Math.max(...values)];
      const ramp = opts.ramp || grayRamp;
      const cw = w / n;
      for (let i = 0; i < n; i++) {
        const t = (values[i] - r[0]) / ((r[1] - r[0]) || 1);
        this.overlay.push({ k: 'srect', x: x + i * cw, y, w: Math.max(cw, 0.8), h, c: ramp(t), al: 1 });
      }
      this.overlay.push({ k: 'srectStroke', x, y, w, h, c: C().line, lw: 1 });
      if (opts.highlight !== undefined && opts.highlight !== null) this.overlay.push({ k: 'srectStroke', x: x + opts.highlight * cw, y: y - 2, w: Math.max(cw, 2), h: h + 4, c: C().gold, lw: 1.5 });
      if (opts.title) this.overlay.push({ k: 't', text: opts.title, p: [x, y - 8], c: C().muted, size: 12, anchor: 'left', font: 'ui' });
    }
    /** Screen-aligned text at a world position (or screen position with screen:true). opts: color, size, anchor, dx, dy, hat, bg, font('math'|'ui'|'mono'), bold */
    label(text, pos, opts) {
      opts = opts || {};
      let p;
      if (opts.screen) p = [pos[0], pos[1]];
      else { const s = this._proj(pos); if (!s) return; p = [s[0], s[1]]; }
      p[0] += opts.dx || 0; p[1] += opts.dy || 0;
      const font = opts.font || (String(text).length <= 2 ? 'math' : 'ui');
      this.overlay.push({ k: 't', text: String(text), p, c: role(opts.color || 'text'), size: opts.size || 14.5, anchor: opts.anchor || 'center', hat: opts.hat, bg: opts.bg, font, bold: opts.bold, al: opts.alpha });
    }
    /** Screen-space rectangle (HUD panels, legend swatches). opts: fill, stroke, alpha, width */
    rect(x, y, w, h, opts) {
      opts = opts || {};
      if (opts.fill !== null) this.overlay.push({ k: 'srect', x, y, w, h, c: role(opts.fill || 'panel'), al: opts.alpha === undefined ? 0.85 : opts.alpha });
      if (opts.stroke) this.overlay.push({ k: 'srectStroke', x, y, w, h, c: role(opts.stroke), lw: opts.width || 1 });
    }
    /** HUD text at screen coordinates. */
    hud(text, x, y, opts) { this.label(text, [x, y], Object.assign({ screen: true, anchor: 'left', font: 'ui', size: 13, color: 'hud' }, opts || {})); }
    get marker() {
      const self = this;
      return {
        check(pos, opts) { opts = opts || {}; const s = opts.screen ? pos : self._proj(pos); if (!s) return; self.overlay.push({ k: 'mark', type: 'check', p: [s[0] + (opts.dx || 0), s[1] + (opts.dy || 0)], c: role(opts.color || 'ok'), size: opts.size || 18 }); },
        cross(pos, opts) { opts = opts || {}; const s = opts.screen ? pos : self._proj(pos); if (!s) return; self.overlay.push({ k: 'mark', type: 'cross', p: [s[0] + (opts.dx || 0), s[1] + (opts.dy || 0)], c: role(opts.color || 'bad'), size: opts.size || 18 }); },
        ring(pos, opts) { opts = opts || {}; const s = self._proj(pos); if (!s) return; self.overlay.push({ k: 'mark', type: 'ring', p: [s[0], s[1]], c: role(opts.color || 'gold'), size: opts.r || 14, w: opts.width || 2 }); },
      };
    }
    /** Draggable handle. onDrag(newWorldPos, s) is called by the input layer. opts: plane ('auto'|'xy'|'xz'|'yz'|'view'|{point,normal}), r(px), color, show(bool) */
    handle(id, pos, onDrag, opts) {
      opts = opts || {};
      const s = this._proj(pos); if (!s) return;
      const enabled = this.handlesEnabled || opts.always;
      const rec = { id, screen: [s[0], s[1]], world: pos.slice(), onDrag, onClick: opts.onClick || null, plane: opts.plane || 'auto', r: opts.r || 14, P: this.P, is3d: this.is3d, enabled };
      this.handles.push(rec);
      if (enabled && opts.show !== false) {
        const hot = this.hot === id || this.dragging === id;
        this.overlay.push({ k: 'handle', p: [s[0], s[1]], r: hot ? 11 : 8, c: role(opts.color || 'white'), hot });
      }
    }

    /* ---------------- painting ---------------- */
    end() {
      this._flush(this.records);
      this._paintOverlay(this.overlay);
      return this.stats;
    }
    _flush(recs) {
      if (this.is3d || recs.some(r => r.d !== 0)) recs.sort((a, b) => b.d - a.d);
      const ctx = this.ctx;
      for (let i = 0; i < recs.length; i++) this._paint(ctx, recs[i]);
    }
    _paint(ctx, r) {
      ctx.globalAlpha = r.al === undefined ? 1 : r.al;
      switch (r.k) {
        case 's': {
          // clip to the canvas (plus margin): near-plane segments can have huge screen coords, and dashing a
          // million-pixel line stalls Canvas 2D
          const m = 50, c = clipSeg(r.a, r.b, -m, -m, this.size.w + m, this.size.h + m);
          if (!c) break;
          ctx.strokeStyle = r.c; ctx.lineWidth = r.w; ctx.setLineDash(r.dash || []);
          ctx.beginPath(); ctx.moveTo(c[0], c[1]); ctx.lineTo(c[2], c[3]); ctx.stroke(); ctx.setLineDash([]); break;
        }
        case 'l':
          ctx.strokeStyle = r.c; ctx.lineWidth = r.w; ctx.setLineDash(r.dash || []);
          ctx.beginPath(); for (let i = 0; i < r.pts.length; i++) { if (i) ctx.lineTo(r.pts[i][0], r.pts[i][1]); else ctx.moveTo(r.pts[i][0], r.pts[i][1]); } if (r.closed) ctx.closePath(); ctx.stroke(); ctx.setLineDash([]); break;
        case 'p':
          ctx.beginPath(); for (let i = 0; i < r.pts.length; i++) { if (i) ctx.lineTo(r.pts[i][0], r.pts[i][1]); else ctx.moveTo(r.pts[i][0], r.pts[i][1]); } ctx.closePath();
          if (r.f) { ctx.fillStyle = r.f; ctx.fill(); }
          if (r.c) { ctx.globalAlpha = Math.min(1, (r.al === undefined ? 1 : r.al) + 0.45); ctx.strokeStyle = r.c; ctx.lineWidth = r.w || 1; ctx.setLineDash(r.dash || []); ctx.stroke(); ctx.setLineDash([]); }
          break;
        case 'd':
          ctx.fillStyle = r.c; ctx.beginPath(); ctx.arc(r.p[0], r.p[1], r.r, 0, Math.PI * 2); ctx.fill();
          if (r.sc) { ctx.strokeStyle = r.sc; ctx.lineWidth = r.sw || 1.5; ctx.stroke(); }
          break;
        case 'img':
          ctx.imageSmoothingEnabled = !!r.smooth; ctx.drawImage(r.img, r.x, r.y, r.w, r.h); ctx.imageSmoothingEnabled = true; break;
      }
      ctx.globalAlpha = 1;
    }
    _paintOverlay(ov) {
      const ctx = this.ctx;
      for (const r of ov) {
        if (this.noText && (r.k === 't' || r.k === 'handle')) continue;   // thumbnails: pictures only
        ctx.globalAlpha = r.al === undefined ? 1 : r.al;
        switch (r.k) {
          case 't': this._text(ctx, r); break;
          case 'sseg': ctx.strokeStyle = r.c; ctx.lineWidth = r.w || 1; ctx.setLineDash(r.dash || []); ctx.beginPath(); ctx.moveTo(r.a[0], r.a[1]); ctx.lineTo(r.b[0], r.b[1]); ctx.stroke(); ctx.setLineDash([]); break;
          case 'sline': ctx.strokeStyle = r.c; ctx.lineWidth = r.w || 1; ctx.setLineDash(r.dash || []); ctx.beginPath(); r.pts.forEach((p, i) => i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1])); ctx.stroke(); ctx.setLineDash([]); break;
          case 'srect': ctx.fillStyle = r.c; ctx.fillRect(r.x, r.y, r.w, r.h); break;
          case 'srectStroke': ctx.strokeStyle = r.c; ctx.lineWidth = r.lw || 1; ctx.strokeRect(r.x, r.y, r.w, r.h); break;
          case 'mark': this._mark(ctx, r); break;
          case 'handle':
            ctx.beginPath(); ctx.arc(r.p[0], r.p[1], r.r, 0, Math.PI * 2);
            ctx.fillStyle = LA.color.alpha(r.c, r.hot ? 0.28 : 0.12); ctx.fill();
            ctx.strokeStyle = LA.color.alpha(r.c, r.hot ? 0.95 : 0.6); ctx.lineWidth = r.hot ? 2 : 1.3; ctx.stroke();
            break;
        }
      }
      ctx.globalAlpha = 1;
    }
    _text(ctx, r) {
      const fam = r.font === 'math' ? '"Cambria Math", Cambria, "Times New Roman", serif' : r.font === 'mono' ? '"Cascadia Mono", Consolas, monospace' : 'system-ui, "Segoe UI", Roboto, sans-serif';
      const italic = r.font === 'math' && /^[A-Za-z]{1,2}$/.test(r.text) ? 'italic ' : '';
      ctx.font = (r.bold ? '600 ' : '') + italic + r.size + 'px ' + fam;
      ctx.textBaseline = 'middle';
      let x = r.p[0], y = r.p[1];
      const w = ctx.measureText(r.text).width;
      switch (r.anchor) {
        case 'left': ctx.textAlign = 'left'; break;
        case 'right': ctx.textAlign = 'right'; break;
        case 'above': ctx.textAlign = 'center'; y -= r.size * 0.75; break;
        case 'below': ctx.textAlign = 'center'; y += r.size * 0.75; break;
        default: ctx.textAlign = 'center';
      }
      if (r.bg !== false) {
        const bx = ctx.textAlign === 'left' ? x : ctx.textAlign === 'right' ? x - w : x - w / 2;
        ctx.fillStyle = LA.color.alpha(LA.color.bg, 0.72);
        roundRect(ctx, bx - 3, y - r.size * 0.62, w + 6, r.size * 1.24, 3); ctx.fill();
      }
      ctx.fillStyle = r.c; ctx.fillText(r.text, x, y);
      if (r.hat) {
        const cx = ctx.textAlign === 'left' ? x + w / 2 : ctx.textAlign === 'right' ? x - w / 2 : x;
        ctx.beginPath(); ctx.strokeStyle = r.c; ctx.lineWidth = 1.3;
        const hw = Math.max(3, r.size * 0.22), hy = y - r.size * 0.62;
        ctx.moveTo(cx - hw, hy + hw * 0.8); ctx.lineTo(cx, hy); ctx.lineTo(cx + hw, hy + hw * 0.8); ctx.stroke();
      }
    }
    _mark(ctx, r) {
      const s = r.size, x = r.p[0], y = r.p[1];
      ctx.strokeStyle = r.c; ctx.lineWidth = Math.max(2, s / 7); ctx.lineCap = 'round';
      ctx.beginPath();
      if (r.type === 'check') { ctx.moveTo(x - s * 0.45, y); ctx.lineTo(x - s * 0.1, y + s * 0.35); ctx.lineTo(x + s * 0.5, y - s * 0.4); }
      else if (r.type === 'cross') { ctx.moveTo(x - s * 0.4, y - s * 0.4); ctx.lineTo(x + s * 0.4, y + s * 0.4); ctx.moveTo(x + s * 0.4, y - s * 0.4); ctx.lineTo(x - s * 0.4, y + s * 0.4); }
      else { ctx.lineWidth = r.w || 2; ctx.arc(x, y, s, 0, Math.PI * 2); }
      ctx.stroke();
    }
  }

  /* ---------------- geometry helpers (exported on LA.geom) ---------------- */
  function add(a, b) { return a.map((x, i) => x + (b[i] || 0)); }
  function dot(a, b) { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * (b[i] || 0); return s; }
  function slerp(a, b, t) {
    const d = Math.max(-1, Math.min(1, dot(a, b))), th = Math.acos(d);
    if (th < 1e-6) return a.slice();
    if (Math.PI - th < 1e-6) { // opposite: go through a perpendicular
      const perp = a.length === 2 ? [-a[1], a[0]] : (Math.abs(a[0]) < 0.9 ? [0, -a[2], a[1]] : [-a[2], 0, a[0]]);
      const n = Math.hypot.apply(null, perp) || 1; const p = perp.map(x => x / n);
      return t < 0.5 ? slerp(a, p, t * 2) : slerp(p, b, (t - 0.5) * 2);
    }
    const s1 = Math.sin((1 - t) * th) / Math.sin(th), s2 = Math.sin(t * th) / Math.sin(th);
    return a.map((x, i) => x * s1 + b[i] * s2);
  }
  /** Apply a 2×2 or 3×3 matrix (or null) to a point; promotes 2D→3D when in3d. */
  function applyM(M, p, in3d, z) {
    if (!M) return in3d ? [p[0], p[1], p.length > 2 ? p[2] : (z || 0)] : [p[0], p[1]];
    if (M.length === 2) {
      const x = M[0][0] * p[0] + M[0][1] * p[1], y = M[1][0] * p[0] + M[1][1] * p[1];
      return in3d ? [x, y, p.length > 2 ? p[2] : (z || 0)] : [x, y];
    }
    const pz = p.length > 2 ? p[2] : (z || 0);
    return [M[0][0] * p[0] + M[0][1] * p[1] + M[0][2] * pz, M[1][0] * p[0] + M[1][1] * p[1] + M[1][2] * pz, M[2][0] * p[0] + M[2][1] * p[1] + M[2][2] * pz];
  }
  function segVisible(a, b, v) {
    const m = 1;
    if (Math.max(a[0], b[0]) < v.x0 - m || Math.min(a[0], b[0]) > v.x1 + m) return false;
    if (Math.max(a[1], b[1]) < v.y0 - m || Math.min(a[1], b[1]) > v.y1 + m) return false;
    return true;
  }
  function bigQuad(pts) { let m = 0; for (let i = 0; i < 4; i++) { const a = pts[i], b = pts[(i + 1) % 4]; m = Math.max(m, Math.hypot(a[0] - b[0], a[1] - b[1], (a[2] || 0) - (b[2] || 0))); } return m > 2.2; }
  function bilerp(q, u, v) { const a = LA.lerpV(q[0], q[1], u), b = LA.lerpV(q[3], q[2], u); return LA.lerpV(a, b, v); }
  /** Sutherland–Hodgman clip of camera-space polygon against z = near. */
  function clipNear(q, n) {
    const out = [];
    for (let i = 0; i < q.length; i++) {
      const a = q[i], b = q[(i + 1) % q.length];
      const ain = a[2] >= n, bin = b[2] >= n;
      if (ain) out.push(a);
      if (ain !== bin) { const t = (n - a[2]) / (b[2] - a[2]); out.push([a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, n]); }
    }
    return out;
  }
  function sampleGrid(fn, b, res) {
    const g = new Array(res);
    for (let i = 0; i < res; i++) { g[i] = new Array(res); const y = b.y1 - (b.y1 - b.y0) * (i + 0.5) / res; for (let j = 0; j < res; j++) { const x = b.x0 + (b.x1 - b.x0) * (j + 0.5) / res; g[i][j] = fn([x, y]); } }
    return g;
  }
  const grayRamp = t => { const v = Math.round(255 * Math.max(0, Math.min(1, t))); return 'rgb(' + v + ',' + v + ',' + v + ')'; };
  const _imgCache = { el: null, ctx: null };
  function gridToImage(grid, range, ramp) {
    const n = grid.length, m = grid[0].length;
    let lo = range ? range[0] : Infinity, hi = range ? range[1] : -Infinity;
    if (!range) for (const row of grid) for (const v of row) { if (v < lo) lo = v; if (v > hi) hi = v; }
    if (hi - lo < 1e-12) hi = lo + 1;
    const el = document.createElement('canvas'); el.width = m; el.height = n;
    const ctx = el.getContext('2d'); const img = ctx.createImageData(m, n); const d = img.data;
    const cache = new Map();
    for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) {
      const t = (grid[i][j] - lo) / (hi - lo); const key = Math.round(t * 255);
      let rgb = cache.get(key);
      if (!rgb) { rgb = parseColor(ramp(key / 255)); cache.set(key, rgb); }
      const o = (i * m + j) * 4; d[o] = rgb[0]; d[o + 1] = rgb[1]; d[o + 2] = rgb[2]; d[o + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
    return el;
  }
  function parseColor(c) {
    if (c[0] === '#') return LA.color.rgb(c);
    const m = /rgba?\(([^)]+)\)/.exec(c); if (m) return m[1].split(',').map(s => parseFloat(s));
    return [255, 255, 255];
  }
  /** Marching squares: returns array of [[x,y],[x,y]] segments at level lv. grid[i][j], i = row from top (y1) down. */
  function marchingSquares(grid, b, lv) {
    const n = grid.length, m = grid[0].length, segs = [];
    const X = j => b.x0 + (b.x1 - b.x0) * (j + 0.5) / m, Y = i => b.y1 - (b.y1 - b.y0) * (i + 0.5) / n;
    for (let i = 0; i < n - 1; i++) for (let j = 0; j < m - 1; j++) {
      const v = [grid[i][j], grid[i][j + 1], grid[i + 1][j + 1], grid[i + 1][j]];
      const P = [[X(j), Y(i)], [X(j + 1), Y(i)], [X(j + 1), Y(i + 1)], [X(j), Y(i + 1)]];
      let idx = 0; for (let k = 0; k < 4; k++) if (v[k] >= lv) idx |= (1 << k);
      if (idx === 0 || idx === 15) continue;
      const ip = (k1, k2) => { const t = (lv - v[k1]) / ((v[k2] - v[k1]) || 1e-12); return [P[k1][0] + (P[k2][0] - P[k1][0]) * t, P[k1][1] + (P[k2][1] - P[k1][1]) * t]; };
      const e = [ip(0, 1), ip(1, 2), ip(2, 3), ip(3, 0)];
      const table = { 1: [[3, 0]], 2: [[0, 1]], 3: [[3, 1]], 4: [[1, 2]], 5: [[3, 0], [1, 2]], 6: [[0, 2]], 7: [[3, 2]], 8: [[2, 3]], 9: [[0, 2]], 10: [[0, 1], [2, 3]], 11: [[1, 2]], 12: [[1, 3]], 13: [[0, 1]], 14: [[3, 0]] };
      for (const pr of table[idx]) segs.push([e[pr[0]], e[pr[1]]]);
    }
    return segs;
  }
  /** Liang–Barsky clip of segment a→b to [x0,x1]×[y0,y1]; returns [ax,ay,bx,by] or null. */
  function clipSeg(a, b, x0, y0, x1, y1) {
    let t0 = 0, t1 = 1;
    const dx = b[0] - a[0], dy = b[1] - a[1];
    const p = [-dx, dx, -dy, dy], q = [a[0] - x0, x1 - a[0], a[1] - y0, y1 - a[1]];
    for (let i = 0; i < 4; i++) {
      if (p[i] === 0) { if (q[i] < 0) return null; continue; }
      const t = q[i] / p[i];
      if (p[i] < 0) { if (t > t1) return null; if (t > t0) t0 = t; }
      else { if (t < t0) return null; if (t < t1) t1 = t; }
    }
    return [a[0] + dx * t0, a[1] + dy * t0, a[0] + dx * t1, a[1] + dy * t1];
  }
  function roundRect(ctx, x, y, w, h, r) { ctx.beginPath(); ctx.moveTo(x + r, y); ctx.arcTo(x + w, y, x + w, y + h, r); ctx.arcTo(x + w, y + h, x, y + h, r); ctx.arcTo(x, y + h, x, y, r); ctx.arcTo(x, y, x + w, y, r); ctx.closePath(); }

  LA.Gfx = Gfx;
  LA.geom = { clipSeg, applyM, slerp, sampleGrid, marchingSquares, gridToImage, grayRamp, clipNear, bilerp, add, dot };
})();
