/* TEMPLATE — copy to lessons/<id>.js and fill in. Read docs/CONTRACT.md and docs/STYLE.md first, then study
   lessons/p2-01-linear-transformations.js (the reference lesson). Delete these comments in your copy.

   RULES THAT THE VALIDATOR ENFORCES (see #/selftest):
   - 6–14 steps; every step has title, narration (20–70 words of HTML), setup(s) and/or at(s,t)
   - exactly one step with mlTie:true in the last third; the LAST step has recap:true (+ cards: [3 strings])
   - at least one step with misconception:true (show the wrong belief failing, with g.marker.cross)
   - badge {drawnIn, realSize, transfers}; cantShow[]; terms[{term, def}] (≥1); 2–3 checks; ≤6 playground controls
   - narration: no inline styles, no hex colors, no <script>. Color words via <span class="c-i">î</span> etc.

   STATE MODEL: state() returns plain JSON data. Steps: setup(s) puts s at the step's START (store anything at(s,t)
   needs, e.g. s._from = LA.clone(s.M)); at(s,t) writes animated fields as a PURE function of t (never from its own
   previous output). The engine restores the previous step's end state before calling setup, so steps chain.
   render(g, s, ctx) draws everything from s, every frame, and registers handles for Play/Check modes. */
(function () {
  'use strict';
  const N = LA.num, lerpM = LA.lerpM, lerpV = LA.lerpV, seg = LA.seg;

  LA.registerLesson({
    id: 'pX-NN-slug',                    // must match the manifest id and the file name
    part: 2, title: 'Lesson title', subtitle: 'One-line promise of the lesson', minutes: 10,
    prereqs: ['p1-01-what-is-a-vector'], tags: ['core'],       // 'core' | 'capstone' | 'primer' | 'extension'
    badge: { drawnIn: '2D (and one 3D step)', realSize: 'a 784×128 weight matrix in a digit classifier', transfers: 'everything: the formula does not care about n' },
    cantShow: ['what a 4th axis would look like (there is none to draw)'],
    terms: [{ term: 'example term', def: 'a one-sentence plain-words definition shown in the glossary.' }],
    atlas: ['mlp'],                       // keys from the ML atlas (see docs/CONTRACT.md)
    thumbnail: 3,                         // step index used for the course-home card (optional)

    state() {
      return {
        view: '2d', cam: LA.cam.defaults2d(),     // camera lives in the state so steps can tween it
        M: [[1, 0], [0, 1]], v: [2, 1],
        showGhost: true, showV: false, cross: null,
      };
    },

    render(g, s, ctx) {
      if (s.showGhost) g.ghostGrid();
      g.grid(s.M);
      g.vector(N.col(s.M, 0), { color: 'i', label: 'i', hat: true });
      g.vector(N.col(s.M, 1), { color: 'j', label: 'j', hat: true });
      if (s.showV) g.vector(N.matVec(s.M, s.v), { color: 'v', label: 'v' });
      if (s.cross) g.marker.cross(s.cross);
      // handles only matter in Play/Check mode or steps marked interactive; registering them every frame is fine
      g.handle('i', N.col(s.M, 0), (p, st) => { st.M[0][0] = p[0]; st.M[1][0] = p[1]; });
      g.handle('j', N.col(s.M, 1), (p, st) => { st.M[0][1] = p[0]; st.M[1][1] = p[1]; });
    },

    math(s, tex) {                        // optional live notation panel (rebuilt only when structure changes)
      return tex.block(tex.eq(tex.sym('A'), tex.op('='), tex.mat(s.M, { colors: 'cols' })));
    },

    steps: [
      { title: 'Start', duration: 2,
        narration: 'Twenty to seventy words. Say what the learner is looking at, in plain language, naming colors by role: the green <span class="c-i">î</span> and the red <span class="c-j">ĵ</span>. One new idea per step.',
        setup(s) { s.M = [[1, 0], [0, 1]]; s.showV = false; },
        at(s, t) { /* nothing animates in this step */ } },
      { title: 'Bend the grid', duration: 3,
        narration: 'Describe what moves. Lines stay lines, evenly spaced, the origin stays put. The two arrows land somewhere new and the whole grid follows them, because every point is built from those two arrows.',
        setup(s) { s._from = LA.clone(s.M); s._to = [[2, 1], [0.5, 1.5]]; },
        at(s, t) { s.M = lerpM(s._from, s._to, seg(t, 0, 1)); } },
      // ... more steps; one with misconception:true using s.cross = [x, y]; one with mlTie:true in the last third
      { title: 'Recap', duration: 1, recap: true,
        narration: 'Three sentences that restate the lesson from the picture: what the columns are, what stays fixed, and what the learner can now read off any matrix they meet in machine learning code.',
        cards: ['Columns are landing spots of î and ĵ.', 'Lines stay lines; the origin stays put.', 'A layer’s weight matrix is exactly this picture.'],
        setup(s) { }, at(s, t) { } },
    ],

    playground: {
      intro: '<p>Drag the tips of <span class="c-i">î</span> and <span class="c-j">ĵ</span>, or edit the matrix.</p>',
      setup(s) { s.showV = true; },
      controls: [
        { type: 'matrix', bind: 'M', label: 'A', colors: 'cols', step: 0.1 },
        { type: 'toggle', bind: 'showGhost', label: 'Show the original grid' },
        { type: 'preset', label: 'Presets', items: [
          { label: 'Identity', values: { M: [[1, 0], [0, 1]] } },
          { label: 'Shear', values: { M: [[1, 1], [0, 1]] } },
        ] },
      ],
    },

    checks: [
      { kind: 'manipulate', prompt: 'Drag the landing spots so the matrix rotates the plane a quarter turn counter-clockwise.',
        setup(s) { s.M = [[1, 0], [0, 1]]; s.showV = false; },
        judge(s) { return LA.judge.matrix(s.M, [[0, -1], [1, 0]], LA.TOL.pos); },
        hint: 'Where must <span class="c-i">î</span> land after a quarter turn? Straight up.',
        solution(s) { s.M = [[0, -1], [1, 0]]; } },
      { kind: 'entry', prompt: 'Type the matrix that doubles every x-coordinate and leaves y alone.', bind: 'M', answer: [[2, 0], [0, 1]], tol: LA.TOL.entry },
    ],
  });
})();
