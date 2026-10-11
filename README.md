# Linear Algebra for Machine Learning — a visual course

A self-contained, interactive course that teaches the linear algebra behind machine learning from pictures first:
animated 2D/3D scenes you can scrub, playgrounds you can drag, and checks you answer on the picture itself.
No installation, no internet, no external material.

## Open it
Double-click `index.html` (Chrome or Edge). Everything runs from the local files.

Keyboard: `←` `→` steps · `space` play/pause · `r` replay step · `1` `2` `3` Learn / Play / Check · `m` contents · `g` glossary.
Mouse: drag the canvas to pan (2D) or orbit (3D), wheel to zoom, shift-drag to pan in 3D, double-click to reset
the view, drag the ring handles in Play and Check modes.

Progress is saved in this browser (localStorage). Export/import it as JSON from the settings (⚙) drawer.

## Layout
- `index.html`, `css/app.css` — shell
- `js/core/` — colors, formatting, easing, seeded data generators (`LA.rng`), numerics (`LA.num`)
- `js/engine/` — canvas loop, cameras, drawing primitives (2D + 3D), input, notation, controls, player, checks
- `js/app/` — registry/validator, progress store, router, shell, pages, self-test
- `lessons/` — one file per lesson (`_manifest.js` is the ordered list, `_template.js` the skeleton)
- `docs/CONTRACT.md` — the frozen API for lesson files; `docs/STYLE.md` — pedagogy and visual rules;
  `docs/curriculum/p*.md` — storyboards for every lesson
- `tools/check.ps1`, `tools/shot.ps1` — headless-Chrome self-test and screenshots (no Node needed)

## Verify
```
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\check.ps1
```
runs every lesson offscreen (all steps at t = 0, 0.5, 1; every check; playground; validator; numeric library) and
prints PASS/FAIL. In the app the same report is at `#/selftest`, and `#/selftest?gallery=1` shows every step of
every lesson as a thumbnail.

## Add a lesson
Copy `lessons/_template.js` to `lessons/<id>.js` where `<id>` is an entry of `lessons/_manifest.js`, follow
`docs/CONTRACT.md` and the storyboard in `docs/curriculum/`, then run `tools\check.ps1 <id>` and
`tools\shot.ps1 <id> <step>` until the self-test passes and the pictures match.
