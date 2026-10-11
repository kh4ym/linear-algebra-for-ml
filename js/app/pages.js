/* LA.pages — non-lesson pages: home, missing lesson, part review, ML atlas, glossary. */
window.LA = window.LA || {};
(function () {
  'use strict';
  const el = (tag, cls, html) => { const e = document.createElement(tag); if (cls) e.className = cls; if (html !== undefined) e.innerHTML = html; return e; };
  const esc = s => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  const pages = {};

  pages.home = function () {
    const pg = el('div', 'page');
    pg.appendChild(el('h1', '', 'Linear algebra for machine learning, taught by pictures'));
    pg.appendChild(el('p', 'lead', 'Every idea is shown moving before it is written down: vectors, the grid a matrix bends, the directions that stay put, the shadow a data cloud casts. Each lesson has narrated steps you can scrub, a playground you can drag, and checks you answer on the picture itself. No outside material is needed.'));
    const last = LA.store.data.last;
    if (last && LA.lessons.get(last.id)) {
      const L = LA.lessons.get(last.id);
      const r = el('div', 'resume', '<span>Continue where you left off:</span><a class="btn primary" href="' + LA.router.lesson(last.id, last.step) + '">' + esc(L.title) + ' → step ' + (last.step + 1) + '</a>');
      pg.appendChild(r);
    } else {
      const first = LA.lessons.ordered().find(e => e.spec);
      if (first) pg.appendChild(el('div', 'resume', '<span>New here?</span><a class="btn primary" href="' + LA.router.lesson(first.id) + '">Start with “' + esc(first.title) + '” →</a>'));
    }
    const prog = LA.store.courseProgress();
    pg.appendChild(el('div', 'progress-bar', '<i style="width:' + (prog * 100).toFixed(1) + '%"></i>'));
    const thumbs = [];
    LA.lessons.parts().forEach(part => {
      const block = el('div', 'part-block');
      const items = LA.lessons.ordered().filter(e => e.part === part.n);
      block.appendChild(el('h2', '', 'Part ' + part.n + ' · ' + esc(part.title) + ' <small>' + items.length + ' lessons · <a href="#/review/' + part.n + '">review</a></small>'));
      block.appendChild(el('p', 'part-desc', esc(part.desc || '')));
      const cards = el('div', 'cards');
      items.forEach(e => {
        const ready = !!e.spec;
        const a = el('a', 'card' + (ready ? '' : ' planned'));
        a.href = ready ? LA.router.lesson(e.id) : '#';
        if (!ready) a.addEventListener('click', ev => ev.preventDefault());
        const status = LA.store.status(e.id);
        const kind = (e.tags || []).includes('capstone') ? '★ capstone' : (e.tags || []).includes('extension') ? 'extension' : (e.tags || []).includes('primer') ? 'primer' : 'core';
        const thumb = el('div', 'thumb');
        a.appendChild(el('div', 'card-top', '<span>' + kind + '</span><span>' + (ready ? (e.spec.minutes || 10) + ' min' : 'coming soon') + '</span>'));
        a.appendChild(thumb);
        a.appendChild(el('h3', '', esc(e.title)));
        a.appendChild(el('p', '', esc(ready ? (e.spec.subtitle || e.spec.badge && e.spec.badge.transfers || '') : (e.blurb || ''))));
        a.appendChild(el('div', 'card-foot', '<span>' + (status === 'done' ? '✓ complete' : status === 'partial' ? 'in progress' : '') + '</span><span>' + (ready ? e.spec.steps.length + ' steps' : '') + '</span>'));
        cards.appendChild(a);
        if (ready) thumbs.push({ spec: e.spec, thumb });
      });
      block.appendChild(cards);
      pg.appendChild(block);
    });
    pages.renderThumbs(thumbs);
    return pg;
  };

  /** Render lesson thumbnails lazily, a few per idle slice. */
  pages.renderThumbs = function (list) {
    let i = 0;
    const step = () => {
      const t0 = performance.now();
      while (i < list.length && performance.now() - t0 < 24) {
        const { spec, thumb } = list[i++];
        try {
          const off = LA.canvas.offscreen(300, 150, 2);
          const k = spec.thumbnail !== undefined ? Math.min(spec.thumbnail, spec.steps.length - 1) : Math.max(0, Math.min(spec.steps.length - 1, Math.floor(spec.steps.length * 0.45)));
          LA.player.renderStatic(spec, k, off.ctx, off.size, { noText: true });
          thumb.appendChild(off.el);
        } catch (e) { thumb.innerHTML = '<div class="hint-box c-bad">preview error</div>'; }
      }
      if (i < list.length) setTimeout(step, 16);
    };
    setTimeout(step, 0);
  };

  pages.missing = function (id) {
    const pg = el('div', 'page');
    const m = (LA.manifest.lessons || []).find(x => x.id === id);
    pg.appendChild(el('h1', '', m ? esc(m.title) : 'Unknown lesson'));
    pg.appendChild(el('p', 'lead', m ? 'This lesson is planned but not built yet. It will appear here automatically once its file exists.' : 'No lesson with id ' + esc(id) + '.'));
    pg.appendChild(el('p', '', '<a href="#/">← Back to the course</a>'));
    return pg;
  };

  pages.review = function (part) {
    const pg = el('div', 'page');
    const p = LA.lessons.parts().find(x => x.n === part);
    pg.appendChild(el('h1', '', 'Review · Part ' + part + (p ? ' · ' + esc(p.title) : '')));
    pg.appendChild(el('p', 'lead', 'Retrieval practice: a sample of checks from this part. Each opens the lesson in Check mode at that check. Come back after a few days and try them again.'));
    const items = LA.lessons.ordered().filter(e => e.part === part && e.spec);
    const all = [];
    items.forEach(e => (e.spec.checks || []).forEach((c, i) => all.push({ e, c, i })));
    // deterministic-but-rotating sample: pick by visit count so repeats differ
    const seed = (LA.store.data.progress && Object.values(LA.store.data.progress).reduce((a, r) => a + (r.visits || 0), 0)) || 0;
    const picked = [];
    for (let k = 0; k < Math.min(5, all.length); k++) picked.push(all[(seed * 7 + k * Math.max(1, Math.floor(all.length / 5))) % all.length]);
    const table = el('table', 'plain');
    table.innerHTML = '<tr><th>#</th><th>Check</th><th>From lesson</th><th></th></tr>' + picked.map((x, k) => '<tr><td>' + (k + 1) + '</td><td>' + x.c.prompt + '</td><td>' + esc(x.e.title) + '</td><td><a class="btn small" href="' + LA.router.lesson(x.e.id) + '?mode=check&check=' + x.i + '">Try it</a></td></tr>').join('');
    pg.appendChild(table);
    if (!picked.length) pg.appendChild(el('p', '', 'No lessons of this part are built yet.'));
    const next = LA.lessons.parts().find(x => x.n === part + 1);
    pg.appendChild(el('p', '', (next ? '<a href="#/review/' + next.n + '">Review Part ' + next.n + ' →</a> · ' : '') + '<a href="#/">Course home</a>'));
    return pg;
  };

  const ATLAS = [
    ['linear-regression', 'Linear regression / least squares', 'Fitting a line or plane is projecting the target vector onto the column space of the data matrix; the normal equations solve it.'],
    ['logistic-regression', 'Logistic regression / perceptron', 'A weight vector perpendicular to a hyperplane; the score is a dot product plus bias, squashed by a sigmoid.'],
    ['knn', 'k-nearest neighbours', 'Distances between feature vectors; the choice of norm changes the neighbours; high dimensions make distances concentrate.'],
    ['pca', 'PCA', 'Eigenvectors of the covariance matrix (or singular vectors of the centered data) give the directions of greatest variance.'],
    ['svd', 'SVD, low-rank models, recommenders, LoRA', 'Any matrix is rotate–stretch–rotate; keeping the top singular values gives the best low-rank approximation.'],
    ['gaussian', 'Gaussian models, anomaly detection', 'The covariance matrix is an ellipsoid; Mahalanobis distance measures in its units; whitening makes it a sphere.'],
    ['mlp', 'Neural networks (MLP)', 'Each layer is an affine map (grid deformation plus shift) followed by an elementwise nonlinearity (fold).'],
    ['backprop', 'Backpropagation', 'The chain rule is a product of Jacobians; weight gradients are outer products; gradients flow back through transposes.'],
    ['optimization', 'Gradient descent, learning rates, conditioning', 'The gradient is perpendicular to loss contours; Hessian eigenvalues set the usable learning rate and the zig-zag.'],
    ['regularization', 'Ridge / lasso / weight decay', 'Norm balls meeting loss contours: the L2 disk shrinks weights, the L1 diamond zeroes them.'],
    ['embeddings', 'Embeddings and retrieval', 'Words and items as vectors; similarity is the cosine of the angle; analogies are vector differences.'],
    ['attention', 'Attention / transformers', 'Scores are all dot products between queries and keys (QKᵀ, a table of pairwise dot products like a Gram matrix but between two different sets), softmax turns each row into weights, the output is a weighted sum of values.'],
    ['softmax', 'Softmax classifiers', 'Scores become a point on the probability simplex; class regions are tiled by the weight vectors.'],
    ['cnn', 'Convolution (CNNs)', 'A convolution is a banded matrix with shared weights; it is still a linear map.'],
    ['normalization', 'LayerNorm / BatchNorm', 'LayerNorm projects each vector onto the plane perpendicular to (1,…,1) and rescales; BatchNorm shifts and scales the batch cloud.'],
    ['markov', 'Markov chains / PageRank', 'A stochastic matrix maps the simplex into itself; repeated multiplication converges to the eigenvector with eigenvalue 1.'],
    ['kernels', 'Kernels / SVMs', 'A fixed lift to a higher dimension makes a plane separate what a line could not; the kernel computes dot products there.'],
    ['tensors', 'Batches, shapes, tensors', 'Rows are samples; XW transforms a whole batch at once; broadcasting repeats a vector across rows.'],
  ];
  pages.atlas = function () {
    const pg = el('div', 'page');
    pg.appendChild(el('h1', '', 'ML atlas: which lessons teach the linear algebra behind each method'));
    pg.appendChild(el('p', 'lead', 'Start from the method you care about and jump to the lessons that build its picture.'));
    const byKey = {};
    LA.lessons.ordered().forEach(e => { if (e.spec) (e.spec.atlas || []).forEach(k => { (byKey[k] = byKey[k] || []).push(e); }); });
    const table = el('table', 'plain');
    table.innerHTML = '<tr><th>Method</th><th>The linear algebra</th><th>Lessons</th></tr>' + ATLAS.map(([key, name, line]) => {
      const ls = byKey[key] || [];
      return '<tr><td><b>' + esc(name) + '</b></td><td>' + esc(line) + '</td><td>' + (ls.length ? ls.map(e => '<a href="' + LA.router.lesson(e.id) + '">' + esc(e.title) + '</a>').join('<br>') : '<span class="c-muted">coming soon</span>') + '</td></tr>';
    }).join('');
    pg.appendChild(table);
    return pg;
  };

  pages.glossary = function () {
    const pg = el('div', 'page');
    pg.appendChild(el('h1', '', 'Glossary'));
    pg.appendChild(el('p', 'lead', 'Every term the course uses, in plain words, with the lesson where it is born from a picture.'));
    const items = LA.lessons.glossary(); const seen = new Set();
    const table = el('table', 'plain');
    table.innerHTML = '<tr><th>Term</th><th>Meaning</th><th>Lesson</th></tr>' + items.filter(it => { const k = it.term.toLowerCase(); if (seen.has(k)) return false; seen.add(k); return true; }).map(it => '<tr><td><b>' + esc(it.term) + '</b></td><td>' + esc(it.def) + '</td><td><a href="' + LA.router.lesson(it.lesson) + '">' + esc(it.title) + '</a></td></tr>').join('');
    pg.appendChild(table);
    return pg;
  };

  LA.pages = pages;
})();
