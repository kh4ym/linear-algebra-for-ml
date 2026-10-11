# Reviewer brief

All 57 lessons are built and pass the automated self-test. Your job is the part automation cannot do: look at every
step as a **complete beginner who has only this app**, find what would confuse, mislead or fail them, and fix it in
place. Project root `C:\Users\x\Desktop\nom`; no Node.js. Read `docs/AUTHORING.md` (rules, tools), `docs/CONTRACT.md`
(API) and `docs/STYLE.md` (pedagogy) first, then your lessons and their storyboards in `docs/curriculum/`.

## Engine facts the authors did not have (all fixed since)
- The math panel now updates numbers correctly when `tex.num` follows `tex.mat`/`tex.vec` (authors' workarounds with
  plain spans or `tex.text` are fine; leave them unless they look wrong).
- `tex.T` / `tex.inv` superscripts render raised; `tex.text(s, {color})` shows its color; role `'null'` exists on canvas.
- `g.viewport` paints on top of everything drawn before it. New: `g.rect(x,y,w,h,{fill,stroke})` (screen space),
  `g.handle(..., {onClick(s)})` for taps.
- On screens narrower than 1500 px the sidebar collapses when a lesson opens: the canvas is ≈ 860×700 at 1280×800.
  Screenshots from `tools\shot.ps1` use 1280×800 by default; also check `-W 1700 -H 950` for wide screens.
- `shot.ps1` retries automatically when Chrome returns a black frame; if one still looks black, re-run it.

## What to check, per step (screenshot every step: `shot.ps1 <id> <k>`, plus `-T 0.5` where motion matters)
1. **Truth.** Every number in narration, labels and the math panel matches what is drawn and what the math gives.
   Recompute anything suspicious with `LA.num` in your head or a quick scratch page. Wrong storyboard numbers must
   not survive.
2. **Legibility.** No clipped, overlapping or unreadable labels; nothing important off-canvas; arrows labeled;
   colors follow STYLE.md roles (î green, ĵ red, k̂ blue, v yellow, answers gold, projections magenta, subspaces teal).
3. **Teaching.** The picture shows what the narration claims, in the order it claims it. Misconception steps must
   actually show the wrong belief failing (with a red ✕), not just assert it. A symbol is never used before the
   learner has seen it (within the lesson or a prerequisite).
4. **Honesty.** No 4th axis; n-D as strips/heatmaps; projections labeled as shadows with % variance; planted or
   synthetic data said to be planted/synthetic.
5. **Checks.** Run each check in check mode (`-Mode check -Check i`). The prompt must match the controls that
   exist in check mode (check mode shows no playground controls: only canvas handles and, for entry checks, the
   matrix input). A check must not display its own answer. It must be achievable by hand within tolerance.
6. **Playground.** Opens in a sensible state, controls do what their labels say, nothing breaks at slider extremes.

## Rules
- Fix in place, only in your assigned lesson files. Keep each lesson passing `tools\check.ps1 <id>` with 0 violations.
- Prefer small, surgical edits; do not rewrite working lessons for taste.
- If a problem needs an engine change, describe it in your report (file, function, what and why) instead of editing.

## Report (≤ 300 words)
Per lesson: issues found → fixed (one line each), anything left and why, and engine requests.
