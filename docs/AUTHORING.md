# Lesson author brief (read with CONTRACT.md and STYLE.md)

You implement lessons for this visual linear-algebra-for-ML course: plain JavaScript, no build step, runs from
file:// in Chrome. There is NO Node.js, npm or Python on this machine. Run all commands from the project root
`C:\Users\x\Desktop\nom`.

## Read first, in order
1. `docs/CONTRACT.md` — the frozen engine API, authoritative for every function name and option.
2. `docs/STYLE.md` — pedagogy, color roles, honesty rules, check rules.
3. Your part's storyboard `docs/curriculum/pN.md` (part intro + your sections). Implement faithfully: steps
   (titles, narration, canvas action, flags), math panel, playground, checks, badge, terms, recap cards, atlas keys.
4. `lessons/p2-01-linear-transformations.js` — the reference lesson; match its structure and quality.
   `lessons/_template.js` is the skeleton.

## If your files already exist
A previous author may have started some of your lessons; the file can be complete, partial or broken. Read it,
run check.ps1 and screenshots, and keep what is good. Finish or fix rather than rewrite from scratch, unless it is
clearly unsalvageable.

## Adapting storyboard items the contract lacks
- `ctx.fx.fly` exists as a harmless no-op.
- `components:true` draws dashed i/j(/k) components; `components:'labels'` adds numbers.
- Ghost grid: `g.ghostGrid()`. Tables: rows of `g.label` (world or `screen:true`).
- Split screens: `g.split(fnL, fnR, {left:{view,cam}, right:{view,cam}})`; keep the second camera in `s` (e.g. `s.cam2`).
- Hover/click on points or cells: `g.handle(id, pos, () => {}, {show:false})` + `ctx.hot`, or replace with a playground slider/preset.
- Arrowheads on curves: polyline + short `g.vector` at the end. Log scales: transform values before `g.lineChart`.
- Categorical palette: a ramp returning `LA.color.data` / `dataB` / `dataC`. A circle in a 3D plane: a 64-point `g.polyline`.
- Warped grids: sample each grid line (~40 points) and draw `g.polyline`. Magnifier: `g.viewport` with a zoomed 2D camera.
- Lifted / bent sheets: `g.paramSurface(fn(u,v) → [x,y,z], {u0,u1,v0,v1}, {res ≤ 24})`.
- Rotations: tween an angle and build the matrix each frame (`LA.num.rot2` / `rot3`); SVD legs via `LA.num.svd2rot`.
- Eigen / PC axes in playgrounds: pass through `LA.num.stabilize` so they do not flicker.
- Symmetric square root: `Q·diag(√λ)·Qᵀ` from `LA.num.eigSym`. Covariance ellipse: `g.ellipse(Q·diag(√λ), mean)`.
- 3D loss maps: `g.surface` + `g.contours(..., {z:0})` or `{lift:true}`; never an image on a perspective plane.
- Camera checks: `LA.judge.lookAlong(s.cam, [0,0,1])`; solutions use `LA.cam.lookAlong(s.cam, 'z')`.
If the engine truly lacks something, work around it inside your file and list it in your final report.

## Rules
- One IIFE per file calling `LA.registerLesson` once; `id` = file name = manifest id. Never edit any other file
  (engine, manifest, docs, other lessons).
- `setup(s)` puts the state at the step's start and stores what `at` needs (`s._from = LA.clone(...)`).
  `at(s, t)` is a PURE function of t: never incremental, never reading its own previous output. The engine restores
  the previous step's end state before `setup`, so steps chain. Camera in `s.cam`, view in `s.view`; animate the
  camera with `LA.cam.tween`.
- Validator (check.ps1 lists violations): 6–14 steps; narration 20–70 words; exactly one `mlTie:true` step in the
  last third; last step `recap:true` with `cards:[3 strings]`; ≥1 `misconception:true` step with a red ✕
  (`g.marker.cross`) beside the failing belief; badge {drawnIn, realSize, transfers}; `cantShow[]`; `terms[]` ≥1;
  2–3 checks; playground ≤ 6 controls; no inline styles, hex colors or `<script>` in narration.
- Data: `LA.rng` generators or small inline arrays only. Never `Math.random` or `Date`. Keep `s` small; compute
  heavy things once in `state()` and store only what render needs. Hard-code trained weights; any optional
  retraining is deterministic, ≤ 150 ms, behind a playground button.
- Performance: ≤ 6000 draw records per frame; point clouds ≤ 400; `g.surface`/`paramSurface` res ≤ 30 (≤ 24 for
  paramSurface); `g.heat3d` res ≤ 32; `g.heat2d` res ≤ 80; `g.lattice` range ≤ 2.
- Honesty: never a 4th axis; n-D as strips/heatmaps with "The picture stops at 3. The formula does not.";
  synthetic data labeled synthetic; projections labeled as shadows with % variance.
- Every colored arrow carries a letter label; narration names colors by role ("the green î").

## Verify (mandatory; one lesson at a time: write → check → screenshots → fix → next)
```
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\check.ps1 <id>
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\shot.ps1 <id> <k>            # every step, k is 0-based
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\shot.ps1 <id> <k> -T 0.5     # mid-animation where motion matters
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\shot.ps1 <id> 0 -Mode play
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\shot.ps1 <id> 0 -Mode check -Check 0   # and 1, 2
```
- check.ps1: your lesson's line must read PASS with spec-violations 0 and no step/check/playground errors. Ignore
  the planned-not-built count and other authors' lessons (written concurrently). "loaded but did not register" on
  YOUR file means a syntax error.
- shot.ps1 prints a PNG path under `tools\shots\`. Read every PNG and compare with the storyboard: right objects,
  colors by role, readable non-overlapping labels, nothing clipped (canvas ≈ 860×700 px at 1280×800; tune
  `s.cam.halfH` / `dist`), handles visible in play/check modes. The picture must teach, not merely run.

## Final report (≤ 250 words)
Per lesson: PASS/FAIL, step count, storyboard items not implemented and why, engine gaps worked around, anything a
reviewer should look at.
