# STYLE.md — narration, visuals, honesty, checks (normative)

Applies to every storyboard in `docs/curriculum/p*.md` and every `lessons/<id>.js`. The registry validator enforces the countable rules; reviewers enforce the rest. Lesson ids and order come from the plan's curriculum table and are final.

## 1. Voice and narration
- 20–70 words per step, present tense, second person ("Drag the tip…").
- Picture first: the canvas moves before a new number or symbol appears; at most **one new symbol per step**; a formula appears only after the picture showed it ("Written out, that is…").
- Name colors by role: "the green î", "the teal plane", "the gold axis" — never a bare color. Every colored arrow carries a letter label on canvas.
- Allowed HTML: `<b> <i> <sub> <sup>` and `<span class="c-i">î</span>` with classes `c-i c-j c-k c-v c-w c-b c-gold c-mag c-teal c-null c-neg`. No inline styles, hex colors or `<script>`. Minus is U+2212; ≤ 2 decimals (`LA.fmt`).
- A term is used only after its definition moment (its own step or a prerequisite lesson); bold it at first use, spelled exactly as in `terms`.

## 2. Color roles
| Role | Token / class | Color |
|---|---|---|
| î, x-component, matrix column 1 | `i` / `c-i` | green |
| ĵ, y-component, column 2 | `j` / `c-j` | red |
| k̂, z-component, column 3 | `k` / `c-k` | blue |
| v: the vector under study, input x | `v` / `c-v` | yellow |
| w: second vector, weight vector | `w` / `c-w` | purple |
| b: target vector, bias, third vector u | `b` / `c-b` | orange |
| answers: eigen/singular/PC axes, fitted line, decision plane | `gold` / `c-gold` | gold |
| projections, shadows | `mag` / `c-mag` | magenta + dashed gray residual |
| subspaces: span, column space, planes | `teal` / `c-teal` | translucent teal |
| null space, lost directions | `null` / `c-null` | magenta line/plane |
| det < 0 fill (det > 0 fill is pale yellow) | `neg` / `c-neg` | violet |
| ghost grid / transformed grid | `gridGhost` / `grid` | dim dashed gray / blue-gray |
| data classes | `data` / `dataB` | sky blue / rose ("the blue points" in narration) |

Matrix columns are colored by the basis vector they record; rows get an underline only in a row view. Red ✕ (`g.marker.cross`) and green ✓ are reserved for misconceptions and checks.

## 3. Steps and flags
- 6–14 steps, each `{ setup(s), at(s,t) }` pure in t, 1.5–8 s.
- Exactly one `mlTie` step in the last third, with a mini-visual (never text only), naming the capstone it previews.
- Last step `recap`: three mini-cards (picture · rule · formula).
- At least one `misconception` step: the wrong belief is drawn and visibly fails, red ✕ beside the false claim.
- `interactive` on every step where the learner may drag (handles are otherwise honored only in play/check modes).
- A concept that generalizes gets a 3D step; one that matters beyond 3D gets an n-D strip step.

## 4. Honesty for n > 3
- Never draw a fourth axis. n-D vectors are strips (a 1×n `g.heat2d`, values printed when n ≤ 12); matrices are heatmaps; clouds are bars, histograms or labeled projections.
- Every n-D step contains the exact line **"The picture stops at 3. The formula does not."**
- Projections are labeled "shadow of ℝⁿ" with the % variance kept, plus ghost points or dashed residuals for what was lost.
- Digits and images are synthetic (`LA.rng`) and labeled so; never "MNIST".
- Every lesson has `badge: { drawnIn, realSize, transfers }` and `cantShow[]` (or "—").
- State only what holds in n-D ("rotations in n-D fix a plane, not an axis"; "the cross product exists only in 3D").

## 5. Checks
Two kinds only, 2–3 per lesson; selftest runs setup → judge must fail, solution → judge must pass.
- `manipulate`: drag, slide, orbit or press preset buttons; judged every frame with `LA.TOL = { angleDeg 4, pos 0.12, scalar 0.05, entry 0.05, det 0.08, unitNorm 0.04 }` — tighten freely, loosen at most 2×; camera checks use 2·angleDeg. Holding the condition 400 ms passes (glow); within 2× tolerance shows a near-miss hint and a faint arrow; "Show me" animates `solution(s)`.
- `entry`: type a number or matrix bound to a field of `s`; the picture updates live; judged per entry with `LA.TOL.entry`.
- No text multiple choice; "pick the picture" = `manipulate` with preset buttons. Checks never block navigation.

## 6. Playground
≤ 6 controls (slider · matrix · toggle · preset · button), 2–4 named presets plus Reset, every number live every frame. 3D: left-drag orbits, wheel zooms, right-drag pans.

## 7. Storyboard format (use verbatim for every lesson)
```
## <id> — <title>
**Part** N · **Tags** core|capstone|primer|extension · **Dims** 2D|3D|both · **Minutes** ~N · **Prereqs** id, id
**Goal.** One sentence: what the learner can do/see afterwards.
**Badge.** drawnIn: … · realSize: … (a real ML size, e.g. "W is 784×128 in a digit classifier") · transfers: …
**Can't show.** bullet list of what the 2D/3D picture cannot show (or "—")
**Terms.** term — definition; term — definition
**State.** the fields of the state object `s` the lesson needs (names + meaning), e.g. `M` (2×2), `v`, `showGhost`, `view`, `cam`
**Steps.**
### 1. <step title> · 2.5 s · flags: (none | interactive | misconception | mlTie | recap)
Narration: <20–70 words of HTML>
Canvas: <exactly what is drawn, what moves from t=0 to t=1, colors by role, which g.* primitives (g.grid, g.vector, g.points, g.polygon, g.planes, g.label, g.arc, g.dim, g.heat2d, g.contours, g.surface, g.bars, g.hist, g.marker.cross …)>
Math panel: <live notation shown, e.g. "A = [[a,b],[c,d]] with columns colored; v with its coordinates" or "—">
… (6–14 steps)
**Playground.** controls: type · bind · range · label (≤ 6); presets: …; what updates live: …
**Checks.**
1. kind: manipulate · prompt: … · setup: … · judge: <rule with LA.TOL.x> · hint: … · solution: …
2. …
(2–3 checks)
**ML tie-in.** which later capstone this previews and the one-paragraph bridge.
**Recap cards.** three bullets (text + the mini-visual each shows).
**Engine needs.** anything the storyboard requires that is not in the plan's g.* list (or "—").
```
Each part file opens with a short part intro (what it teaches, through-line, order) and ends with a "Review page" section listing 4–5 checks to sample.

## 8. Glossary
Each lesson lists `terms: [{ term, def }]`: one plain sentence each, using no symbol the learner has not yet met. The engine auto-links the first use of each term in narration to the glossary drawer, so narration spells the term exactly as listed.
