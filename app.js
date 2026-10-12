(function () {
  'use strict';
  const QK = window.QK;
  const state = { data: 'circles', kernel: 'q2', noise: 0.06, seed: 11 };

  const canvas = document.getElementById('plot');
  const ctx = canvas.getContext('2d');
  const $ = (id) => document.getElementById(id);

  const TEAL = '#0e9f92', AMBER = '#e8961e', TEAL_BG = '#d5f3ef', AMBER_BG = '#fdebcb';

  function kernelFor(name) {
    if (name === 'linear') return QK.linearKernel;
    if (name === 'rbf') return QK.rbfKernel(2);
    return QK.quantumKernel(name === 'q1' ? 1 : 2);
  }

  function run() {
    const points = QK[state.data](120, state.noise, state.seed);
    const parts = QK.split(points, 2 / 3);
    const model = QK.trainSVM(parts.train, kernelFor(state.kernel), { C: 1 });
    draw(model, parts);
    const pct = (v) => (100 * v).toFixed(1) + '%';
    $('trainAcc').textContent = pct(QK.accuracy(model, parts.train));
    $('testAcc').textContent = pct(QK.accuracy(model, parts.test));
    $('sv').textContent = model.supportCount;
    const quantum = state.kernel === 'q1' || state.kernel === 'q2';
    $('pairs').textContent = quantum ? ((parts.train.length * (parts.train.length - 1)) / 2).toLocaleString('en-US') : 'none';
  }

  function draw(model, parts) {
    const S = canvas.width, B = 8;
    for (let by = 0; by < S; by += B) {
      for (let bx = 0; bx < S; bx += B) {
        const x = ((bx + B / 2) / S) * Math.PI, y = (1 - (by + B / 2) / S) * Math.PI;
        ctx.fillStyle = model.score([x, y]) >= 0 ? AMBER_BG : TEAL_BG;
        ctx.fillRect(bx, by, B, B);
      }
    }
    const dot = (p, isTest) => {
      const cx = (p.x[0] / Math.PI) * S, cy = (1 - p.x[1] / Math.PI) * S;
      const wrong = model.predict(p.x) !== p.y;
      ctx.beginPath();
      ctx.arc(cx, cy, isTest ? 8 : 6, 0, 2 * Math.PI);
      ctx.fillStyle = p.y === 1 ? AMBER : TEAL;
      ctx.globalAlpha = isTest ? 1 : 0.55;
      ctx.fill();
      ctx.globalAlpha = 1;
      ctx.lineWidth = wrong ? 3 : 2;
      ctx.strokeStyle = wrong ? '#d9457a' : isTest ? '#14122b' : '#ffffff';
      ctx.stroke();
    };
    parts.train.forEach((p) => dot(p, false));
    parts.test.forEach((p) => dot(p, true));
  }

  function segment(id, key) {
    const box = $(id);
    box.addEventListener('click', (e) => {
      const button = e.target.closest('button');
      if (!button) return;
      box.querySelectorAll('button').forEach((b) => b.classList.toggle('on', b === button));
      state[key] = button.dataset.v;
      run();
    });
  }
  segment('data', 'data');
  segment('kernel', 'kernel');

  $('noise').addEventListener('input', (e) => {
    state.noise = Number(e.target.value);
    $('noiseOut').textContent = state.noise.toFixed(2);
    run();
  });
  $('again').addEventListener('click', () => {
    state.seed = Math.floor(Math.random() * 100000);
    run();
  });

  // Slide viewer
  const total = 10;
  let current = 1;
  function show(n) {
    current = Math.min(total, Math.max(1, n));
    const img = $('slide');
    img.src = 'assets/slides/slide-' + String(current).padStart(2, '0') + '.jpg';
    img.alt = 'Slide ' + current + ' of ' + total;
    $('count').textContent = current + ' / ' + total;
  }
  $('prev').addEventListener('click', () => show(current - 1));
  $('next').addEventListener('click', () => show(current + 1));
  $('slide').addEventListener('click', () => show(current === total ? 1 : current + 1));
  document.addEventListener('keydown', (e) => {
    if (e.target.matches('input')) return;
    if (e.key === 'ArrowLeft') show(current - 1);
    if (e.key === 'ArrowRight') show(current + 1);
  });

  run();
})();
