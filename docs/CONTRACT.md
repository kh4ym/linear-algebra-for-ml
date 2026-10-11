# Engine contract for lesson authors (frozen)

Everything a lesson file may use. Names here are frozen: the engine may **add** functions and options, never rename
or remove them. If you need something that is not here, write it in your lesson with plain code (helpers inside
your IIFE) and mention it in your final report; do not edit engine files.

Read `docs/STYLE.md` (pedagogy and visual rules) and `lessons/p2-01-linear-transformations.js` (the reference
lesson) before writing. Copy `lessons/_template.js` to start.

## 1. Files and loading
- One lesson = one file `lessons/<id>.js`, a classic script (no modules, no imports), an IIFE that calls
  `LA.registerLesson({...})` once. The `id` must equal the file name and the manifest entry.
- You may use any ES2020 syntax (const/let, arrow functions, template strings, spread, optional chaining).
- Never touch the DOM, timers, `requestAnimationFrame`, `fetch`, `Math.random`, `Date`, or globals other than
  reading `LA.*`. All data is inline or generated with `LA.rng(seed)`.
- Nothing is loaded over the network. The app runs from `file://`.

## 2. Lesson object
```js
LA.registerLesson({
  id: 'p2-01-linear-transformations',   // ^p[0-6]-\d{2}-[a-z0-9-]+$
  part: 2, title: 'Linear transformations are matrices', subtitle: 'one line', minutes: 11,
  prereqs: ['p1-01-what-is-a-vector'],   // ids
  tags: ['core'],                        // 'core' | 'capstone' | 'primer' | 'extension' (may combine core+capstone)
  badge: { drawnIn: '2D, with one 3D step', realSize: 'a 784×128 weight matrix', transfers: 'the rule is the same in any n' },
  cantShow: ['…'],                       // required array (may be empty)
  terms: [{ term: 'matrix', def: 'one plain sentence' }],   // ≥1; first use in narration is auto-linked
  atlas: ['mlp'],                        // ML atlas keys (section 9)
  thumbnail: 4,                          // optional: step index for the course-home card
  state() { return { view: '2d', cam: LA.cam.defaults2d(), M: [[1,0],[0,1]], /* … */ }; },
  render(g, s, ctx) { /* draw everything from s */ },
  math(s, tex) { return tex.block(/* … */); },            // optional live notation panel
  steps: [ { title, narration, duration, setup(s), at(s, t), interactive?, misconception?, mlTie?, recap?, cards?, lockCamera? }, … ],
  playground: { intro?: 'html', setup(s), controls: [ /* ≤ 6 */ ] },
  checks: [ /* 2–3 */ ],
});
```
Validator rules (violations are listed on `#/selftest` and fail `tools\check.ps1`): 6–14 steps; each step has
`title`, `narration` (20–70 words after stripping tags) and `setup` and/or `at`; exactly one step with
`mlTie:true`, located in the last third; the last step has `recap:true`; at least one `misconception:true`;
`badge` with all three fields; `cantShow[]`; ≥1 `terms`; 2–3 checks; ≤6 playground controls; narration must not
contain `style=`, hex colors or `<script>`.

### State `s`
`state()` returns plain JSON data (numbers, strings, booleans, arrays, nested objects). Required fields:
`view` (`'2d'` or `'3d'`) and `cam` (use `LA.cam.defaults2d()` / `LA.cam.defaults3d()`; you may override fields).
The engine owns one `s` object per mounted lesson and resets it **in place**; never keep references to `s` in
closures across steps, always use the `s` you are handed. Keep `s` small (≤ a few thousand numbers); regenerate
data from a seed in `state()` rather than storing huge arrays.

### Steps: setup + at
- `setup(s)` puts `s` at the step's start configuration and stores anything `at` needs, e.g. `s._from = LA.clone(s.M)`.
  Fields starting with `_` are fine (they are snapshotted like everything else).
- `at(s, t)` with `t ∈ [0,1]` writes the animated fields as a **pure function of t** computed from values stored
  by `setup` (never from its own previous output, never incrementally). It is called many times per step,
  in any order (scrubbing), and at t = 0, 0.5, 1 by the self-test.
- The engine restores the previous step's end state before calling `setup`, so steps chain naturally.
- `duration` in seconds (default 2.5). `interactive: true` enables handles during the step. `lockCamera: true`
  forbids user orbit/pan during the step. `cards: ['…','…','…']` on the recap step shows three mini-cards.
- Camera: write `s.view` and `s.cam` in `setup`; to animate the camera in `at` use
  `LA.cam.tween(s, {yaw: 0.7}, {yaw: 1.05}, u)` which is skipped once the user has orbited (`s.cam.auto === false`).
- Narration is an HTML string or `(s) => html`. Allowed tags: `<b> <i> <sub> <sup> <br>` and color spans
  `<span class="c-i">î</span>` with classes `c-i c-j c-k c-v c-w c-b c-gold c-mag c-teal c-null c-neg c-ok c-bad c-muted`.
  Unicode: î = `î`, ĵ = `ĵ`, k̂ = `k̂`, ᵀ = `ᵀ`, ⁻¹ = `⁻¹`, · = `·`, − = `−`.

### Playground
`setup(s)` runs when Play mode starts (after the state is reset to `state()`). Controls:
```js
{ type: 'slider', bind: 'theta', min: -3.14, max: 3.14, step: 0.01, label: 'θ', color: 'gold', fmt: x => LA.fmt.deg(x) }
{ type: 'matrix', bind: 'M', label: 'A', colors: 'cols' | 'rows' | ['i','j'] | 'v', step: 0.1, digits: 2, min, max }  // also works for a flat vector
{ type: 'toggle', bind: 'showGhost', label: 'Show the original grid' }
{ type: 'preset', label: 'Presets', items: [{ label: 'Shear', values: { M: [[1,1],[0,1]] } }, { label: '3D', apply(s) { s.view = '3d'; } }] }
{ type: 'button', label: 'Retrain', onClick(s) { … }, primary: false }
{ type: 'select', bind: 'dataset', label: 'Dataset', options: [{ label: 'circles', value: 'circles' }] }
{ type: 'info', html: '<b>Tip.</b> …' }
```
`bind` is a path into `s` (`'M'`, `'cam.dist'`, `'M[0][1]'`). After any control change the engine re-renders and
refreshes the math panel. Handles (section 4) are active in Play and Check modes automatically.

### Checks
```js
{ kind: 'manipulate', prompt: 'html', setup(s) {…}, judge(s) => ({ ok, progress, hint? }) | boolean, hint: 'html', solution(s) {…} }
{ kind: 'entry', prompt: 'html', bind: 'M', answer: [[2,0],[0,1]] /* or [x,y] or number */, tol: LA.TOL.entry, setup(s) {…}, label: 'A', colors: 'cols' }
```
A manipulate check passes when `judge(s).ok` stays true for 400 ms. `progress ∈ [0,1]` drives the bar and a hint
appears when progress > 0.7 (or when `judge` returns its own `hint`). `solution(s)` must make `judge` true
(the self-test verifies `setup → judge false → solution → judge true`). Use `LA.TOL` and `LA.judge` helpers:
`LA.judge.near(p, target, tol)`, `.scalar(x, target, tol)`, `.angleDeg(a, b, targetDeg, tol)`,
`.matrix(M, target, tol)`, `.all(r1, r2, …)`. Tolerances: `LA.TOL = { angleDeg: 4, pos: 0.12, scalar: 0.05, entry: 0.05, det: 0.08, unitNorm: 0.04 }`.

### ctx (third render argument)
`{ s, mode: 'learn'|'play'|'check', step, t, size: {w,h,dpr}, dpr, hot /* hovered handle id */, time /* seconds */, instant /* true in screenshots/selftest: avoid time-based effects */ }`.

## 3. Drawing: `g` (an `LA.Gfx`)
All positions are world coordinates (`[x,y]` in 2D, `[x,y,z]` in 3D; 2D points are accepted in 3D as z=0).
Colors are role names (`'i' 'j' 'k' 'v' 'w' 'b' 'gold' 'mag' 'teal' 'neg' 'residual' 'grid' 'gridAxis' 'ghost' 'hud' 'muted' 'text' 'ok' 'bad' 'white'`) or literal `'#rrggbb'` / `'rgba(...)'` strings (prefer roles). Everything is
drawn in the order you call it in 2D; in 3D records are depth-sorted automatically. Text, markers and handles are
always drawn on top.

| call | what it draws | options |
|---|---|---|
| `g.grid(M, o)` | grid lines transformed by `M` (2×2 or 3×3; `null` = identity) | `range` (default 10 / 4 in 3D), `step`, `color`, `axisColor`, `alpha`, `width`, `axis:false`, `z` |
| `g.ghostGrid(o)` | the faint original grid | as grid |
| `g.axes(o)` | 2D: x/y axes with ticks and numbers; 3D: three labelled arrows | `ticks`, `labels`, `names:['x','y','z']`, `length` (3D), `colors` (3D) |
| `g.vector(to, o)` / `g.vector(from, to, o)` | arrow | `color`, `width`, `head` (px), `label`, `hat`, `labelAt` (0..1), `labelOffset`, `dx`, `dy`, `dashed`, `alpha`, `components: true|'labels'` (dashed i/j/k projections) |
| `g.point(p, o)` | dot | `r`, `color`, `label`, `labelColor`, `hat`, `stroke`, `alpha` |
| `g.points(arr, o)` | many dots (≤2000) | `r`, `color`, `colors: array \| (i, p) => role`, `alpha` |
| `g.line(a, b, o)` | segment | `color`, `width`, `dashed: true\|[6,5]`, `alpha` |
| `g.lineThrough(p, dir, o)` | long line through p | `len`, + line options |
| `g.polyline(pts, o)` | open/closed path | `color`, `width`, `dashed`, `closed`, `alpha` |
| `g.polygon(pts, o)` | filled polygon | `fill`, `alpha` (default 0.35), `stroke`, `width`, `subdiv` (3D quads), `back` |
| `g.parallelogram(u, v, o)` | parallelogram from origin | `origin`, `fill`, `stroke`, `alpha`, `orient:true` (fill turns violet when det<0) |
| `g.circle(c, r, o)` / `g.ellipse(M, c, o)` | circle / image of the unit circle under M | `fill`, `stroke`, `width`, `dashed`, `alpha`, `segments` |
| `g.arc(center, dirA, dirB, o)` | angle arc between two directions | `r`, `color`, `label`, `right:true` (right-angle mark) |
| `g.dim(a, b, o)` | dimension line with end ticks | `label`, `offset` (px), `color` |
| `g.field(fn, o)` | arrow field (2D), `fn([x,y]) → [vx,vy]` | `bounds`, `step`, `scale`, `color`, `normalize` |
| `g.heat2d(fnOrGrid, bounds, o)` | scalar field image (2D; becomes flat quads in 3D) | `res`, `range:[lo,hi]`, `ramp: t => color`, `alpha` |
| `g.image(grid, bounds, o)` | matrix as grayscale image (2D) | `ramp`, `range`, `alpha`, `cells:true`, `smooth` |
| `g.contours(fn, levels, bounds, o)` | contour polylines | `res`, `color`/`colors`, `width`, `z` (3D floor), `lift:true` (3D: on the surface) |
| `g.surface(fn, bounds, o)` | 3D surface z = fn([x,y]) | `res` (≤40), `mode:'wire'\|'quads'\|'both'`, `color`, `alpha`, `ramp`, `zScale`, `zRange` |
| `g.heat3d(fnOrGrid, o)` | flat-quad heatmap on the floor or a plane (3D) | `bounds`, `res` (≤32), `range`, `ramp`, `alpha`, `z`, `plane:{origin,u,v}` |
| `g.paramSurface(fn, {u0,u1,v0,v1}, o)` | parametric sheet `fn(u,v) → [x,y,z]` (e.g. a grid lifted and bent by a layer); returns the vertex grid | `res` (≤40), `mode:'wire'\|'quads'\|'both'`, `color`, `alpha`, `ramp(t)`, `colorAt(u,v)→role`, `wireStep` |
| `g.planes([{point, normal \| span:[u,v], extent, color, alpha, label}])` | translucent planes, mutually clipped (≤3) | per plane: `extentU`, `extentV`, `stroke:false`, `subdiv` |
| `g.plane(pl)` | one plane | as above |
| `g.box(M, o)` | unit cube under 3×3 M (faces + edges) | `fill`, `alpha`, `stroke`, `origin`, `size`, `faces:false` |
| `g.lattice(M, o)` | 3D lattice of lines under M | `range` (2), `step`, `color`, `alpha`, `width` |
| `g.cage(o)` | floor grid + crossfading back walls (3D) | `range`, `walls:false`, `M` |
| `g.ellipsoid(M, o)` / `g.sphere(c, r, o)` | rings + meridians of the unit sphere under M | `color`, `colors`, `center`, `meridians`, `fill:true`, `fillAlpha` |
| `g.dropLine(p, foot, o)` | dashed residual + foot point (projection picture) | `color`, `footColor`, `r` |
| `g.bars(values, o)` | bar chart in world coords, or HUD with `screen:true` | world: `origin`, `width`, `gap`, `height`, `colors`, `labels`, `values:true`; screen: `x,y,w,h`, `title`, `max` |
| `g.hist(values, bins, o)` | histogram (screen-space HUD) | `x,y,w,h`, `range`, `color`, `title`, `marks:[{at, label, color}]` |
| `g.lineChart(series, o)` | small line chart (HUD); `series = [{points:[[x,y]…], color, label, dash}]` | `x,y,w,h`, `xRange`, `yRange`, `title`, `xLabel`, `yLabel`, `hline`, `hlineColor` |
| `g.strip(values, o)` | the n-D strip view (HUD) | `x,y,w,h`, `range`, `ramp`, `title`, `highlight` |
| `g.label(text, pos, o)` | screen-aligned text at a world position | `color`, `size`, `anchor:'center'\|'left'\|'right'\|'above'\|'below'`, `dx`, `dy`, `hat`, `bg:false`, `font:'math'\|'ui'\|'mono'`, `bold`, `screen:true` (pos in px) |
| `g.hud(text, x, y, o)` | HUD text at screen px | as label |
| `g.rect(x, y, w, h, o)` | screen-space rectangle (HUD panel, swatch) | `fill` (role, default panel; `null` for none), `stroke`, `alpha`, `width` |
| `g.marker.check(pos, o)` / `g.marker.cross(pos, o)` / `g.marker.ring(pos, o)` | ✓ / ✕ / ring overlay | `size`, `color`, `dx`, `dy`, `screen:true` |
| `g.handle(id, pos, onDrag, o)` | draggable handle; `onDrag(newPos, s)` mutates `s` (called once the pointer moves ≥ 3 px) | `onClick(s)` (tap without moving), `plane:'auto'\|'xy'\|'xz'\|'yz'\|'view'\|{point,normal}` (3D), `r` (px), `color`, `show:false`, `always:true` |
| `g.viewport({x,y,w,h}, {view, cam, bg, border}, fn)` | sub-scene in a rectangle (fractions 0–1 or px) with its own camera; paints on top of everything drawn before it | `fn(g, rectPx)`; screen-space calls inside (`g.hud`, `screen:true`) use whole-canvas px, so offset by `rectPx.x/y` |
| `g.split(fnLeft, fnRight, {ratio, left:{view,cam}, right:{view,cam}})` | two side-by-side viewports | |
| `g.is3d`, `g.P` (projector: `toScreen(p)`, 2D `toWorld(sx,sy)`, `bounds()`), `g.size`, `g.time` | queries | |

Handles are honored only in Play/Check modes or in steps with `interactive:true`; registering them every frame is
the normal pattern. Keep a frame under ~6000 records (one record ≈ one segment/polygon/dot).

3D rules: the world is right-handed, z up; the 2D picture is the z = 0 floor. Curves are vertex-projected
polylines. Do not draw images on perspective planes (use `g.heat3d`/`g.contours`). Avoid two intersecting
translucent curved surfaces (painter's algorithm). Use `g.planes([...])` for intersecting planes.

## 4. Camera
`s.cam` fields — 2D: `center:[x,y]`, `halfH` (visible half-height in world units). 3D: `target:[x,y,z]`, `dist`,
`yaw`, `pitch` (radians, |pitch| ≤ 89°), `fov` (degrees), `ortho` (bool), `halfH` (for ortho). `LA.cam.defaults2d()`,
`LA.cam.defaults3d()`, `LA.cam.ensure(cam, view)`, `LA.cam.tween(s, from, to, u)`, `LA.cam.forward(cam)` (unit view
direction), `LA.cam.lookAlong(cam, 'z' | [x,y,z])` (returns a camera looking down that axis; use in setup/solution).
User interaction sets `s.cam.auto = false`; every step start resets it to true. Judge a camera with
`LA.judge.lookAlong(s.cam, [0,0,1], tolDeg)`.

## 5. Animation helpers
`LA.seg(t, a, b, ease?)` → eased progress of t inside [a,b] (0 before, 1 after; ease: `'linear' 'inOut' 'smooth' 'in' 'out' 'outCubic' 'back' 'bounce'` or a function).
`LA.seg.pulse(t, a, b)` (0→1→0), `LA.seg.after(t, a)`, `LA.seg.cycle(t, n)`.
`LA.lerp(a, b, t)`, `LA.lerpV(a, b, t)`, `LA.lerpM(A, B, t)`, `LA.lerpAngle(a, b, t)`, `LA.lerpObj(a, b, t)`, `LA.clamp`, `LA.clamp01`,
`LA.clone(x)`, `LA.getPath(s, 'M[0][1]')`, `LA.setPath(s, path, v)`, `LA.ease.*`.
Matrix interpolation: `LA.lerpM` (entrywise; standard "what does this matrix do" visual); for rotations tween an
angle and build the matrix with `LA.num.rot2(theta)`; `LA.num.matPolar(A, B, t)` (2×2) never passes through det = 0.

## 6. Numerics: `LA.num` (pure functions; matrices are arrays of rows `M[i][j]`; vectors are flat arrays)
Vectors: `add sub scale dot norm norm1 normInf normP normalize dist cross angle proj(v, onto) lerpV zeros basis(i,n) mapV sum mean1`.
Matrices: `identity(n) zerosM(r,c) shape col(M,j) row(M,i) cols fromCols transpose matMul matVec outer addM subM scaleM lerpM trace frobenius det inv solve(A,b) rank nullSpace colSpace isSymmetric chol diag diagOf hstack vstack submatrix matPow`.
Constructions: `rot2(θ) rot3(axis, θ) rotX rotY rotZ scale2 scale3 shear2(kx,ky) reflect2(lineAngle) project2(lineAngle) projectOnto(vectors)`.
Decompositions: `eig2(A)` → `{real, values, vectors}` or `{real:false, re, im}`; `eigSym(A)` → `{values, vectors}` (descending, orthonormal, sign-fixed);
`svd(A)` → `{U, S, V}` (thin, A = U diag(S) Vᵀ); `svd2rot(A)` → `{thetaU, S, thetaV}` (both proper rotations; S[1] < 0 iff det < 0) for rotate–stretch–rotate animations;
`polar2(A)`, `matPolar(A,B,t)`, `qr(A)` → `{Q,R}`, `gramSchmidt(vs)` → `{basis, steps[{input, projections, residual, normalized}]}`, `pinv(A)`, `lstsq(X,y)` (min-norm for wide X), `ridge(X,y,λ)`, `normalEq(X,y)` → `{XtX, Xty, w}`, `condNumber(A)`.
Statistics (rows = samples): `mean center(X)→{Xc, mu} cov(X,{ddof:1}) corr standardize(X)→{Xs,mu,sigma} pca(X,k)→{components, values, explained, mean, Z} mahalanobis(x, mu, covInv) whiten(X)→{Xw, W}`.
Dynamics: `powerIter(A, iters, v0)→{vector, value, history}`, `rotationAxis(R)`, `stationary(P, {rows:false})`.
Nonlinear: `softmax(v, T) sigmoid relu tanh sigmoidV reluV tanhV`. Stability: `stabilize(prevVectors, nextVectors)` (reorder/sign-align, use it in playgrounds so eigen/PC axes do not flicker), `approxEq approxEqV approxEqM clamp lerp`.

## 7. Data: `LA.rng`
`const rand = LA.rng(seed)` → `rand()` in [0,1); `LA.rng.gauss(rand)`, `gaussV(rand, n)`, `shuffle(rand, arr)`, `pick(rand, arr)`, `int(rand, lo, hi)` (inclusive), `randomUnit(rand, d)`.
Generators (deterministic): `gaussianCloud(seed, n, {mean, cov})`, `circles(seed, n, {r1, r2, noise})→{X,y}`, `spirals(seed, n, {turns, noise})`, `moons(seed, n, {noise})`, `blobs(seed, n, {centers, spread})`, `linearData(seed, n, {w, b, noise, xRange})→{x,y}`, `planeData(seed, n, {w, b, noise})→{X,y}`, `syntheticDigits(seed, n, {size:8})→[{pixels, label}]`, `smallImage(seed, {size:32})→matrix`, `wordVectors()→[{word, v}]`, `tokenVectors(seed, n, d)`, `cosineHistogram(seed, d, pairs, bins)→{bins, counts}`.

## 8. Notation: `LA.tex` (returns HTML; used in `math(s, tex)`)
`tex.num(x, {color, digits})`, `tex.sym('v', {hat, sub, sup, color})`, `tex.op('=')`, `tex.text('words', {color})`, `tex.dim(r, c)`,
`tex.mat(M, {colors:'cols'|'rows'|[...]|fn(i,j), digits, highlight:[i,j]|fn, dim:true})`, `tex.vec(v, {color})`, `tex.rowvec(v, {color})`,
`tex.eq(...parts)` (one inline equation), `tex.block(...eqs)` (wrapping row), `tex.lines(...)` (stacked), `tex.frac(a,b)`, `tex.sum(expr, from, to)`, `tex.sqrt(x)`, `tex.norm(x)`, `tex.abs(x)`, `tex.paren(x)`, `tex.T(x)`, `tex.inv(x)`, `tex.dotExpanded(a, b, {colorA, colorB, colorResult})`, `tex.ihat() tex.jhat() tex.khat()`.
`math(s, tex)` is called every frame; keep its structure stable (only numbers change) so the panel updates in place. Conditional parts are fine; the panel rebuilds when the structure changes.
Also: `LA.fmt(x)` (integers plain, else 2 decimals, real minus sign), `LA.fmt.fixed(x, d)`, `LA.fmt.vec(v)`, `LA.fmt.deg(rad)`, `LA.fmt.pct(x)`, `LA.fmt.signed(x)`.
Colors in code: `LA.color.i` … (hex strings), `LA.color.basis(j)`, `LA.color.alpha(hex, a)`, `LA.color.mix(a, b, t)`, `LA.color.heat(t)`, `LA.color.diverge(t)`.

## 9. ML atlas keys
`linear-regression logistic-regression knn pca svd gaussian mlp backprop optimization regularization embeddings attention softmax cnn normalization markov kernels tensors`.

## 10. Verifying your lesson (no Node needed)
```
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\check.ps1 <id>          # self-test: steps, checks, playground, validator
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\shot.ps1 <id> <step>     # screenshot of a step's end state → tools\shots\<id>-<step>.png
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\shot.ps1 <id> <step> -T 0.5          # mid-animation
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\shot.ps1 <id> 0 -Mode play           # playground
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\shot.ps1 <id> 0 -Mode check -Check 1 # second check
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\shot.ps1 -Url "#/selftest?gallery=1&only=<id>" -H 2400   # every step as thumbnails
```
Look at every screenshot (Read the png). A lesson is done when check.ps1 prints PASS for it with 0 spec
violations and every step's picture matches its storyboard.

## 11. Routes
`#/` home · `#/lesson/<id>/<k>?instant=1&mode=play|check&check=<i>&t=<0..1>` · `#/review/<part>` · `#/atlas` · `#/glossary` · `#/selftest?only=<id>&gallery=1`.
