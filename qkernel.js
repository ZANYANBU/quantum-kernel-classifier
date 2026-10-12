// Quantum kernel classifier, simulated exactly for two qubits.
// The same file runs in the browser (the live demo) and in Node (the numbers on the slides).
(function (root) {
  'use strict';

  // Small seeded random generator so every run gives the same data.
  function rng(seed) {
    let a = seed >>> 0;
    return function () {
      a = (a + 0x6d2b79f5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function gauss(rand) {
    return Math.sqrt(-2 * Math.log(1 - rand())) * Math.cos(2 * Math.PI * rand());
  }

  // Two rings: label -1 inside, +1 outside. Features are scaled to [0, pi].
  function circles(n, noise, seed) {
    const rand = rng(seed), points = [];
    for (let i = 0; i < n; i++) {
      const label = i % 2 === 0 ? -1 : 1;
      const radius = (label === -1 ? 0.35 : 0.9) + noise * gauss(rand);
      const angle = 2 * Math.PI * rand();
      points.push({
        x: [Math.PI * (0.5 + 0.5 * radius * Math.cos(angle)), Math.PI * (0.5 + 0.5 * radius * Math.sin(angle))],
        y: label,
      });
    }
    return points;
  }

  // Two interleaved half-moons, scaled to [0, pi].
  function moons(n, noise, seed) {
    const rand = rng(seed), points = [];
    for (let i = 0; i < n; i++) {
      const label = i % 2 === 0 ? -1 : 1;
      const t = Math.PI * rand();
      let px = label === -1 ? Math.cos(t) : 1 - Math.cos(t);
      let py = label === -1 ? Math.sin(t) : 0.5 - Math.sin(t);
      px += noise * gauss(rand);
      py += noise * gauss(rand);
      points.push({ x: [Math.PI * (px + 1) / 3, Math.PI * (py + 0.5) / 1.5], y: label });
    }
    return points;
  }

  function split(points, trainShare) {
    const cut = Math.round(points.length * trainShare);
    return { train: points.slice(0, cut), test: points.slice(cut) };
  }

  // ---- The quantum part -------------------------------------------------
  // Feature map: for each layer apply H to both qubits, then the phases
  // exp(i * (x1 Z1 + x2 Z2 + (pi - x1)(pi - x2) Z1 Z2)).
  // The state has four amplitudes, kept as [re, im] pairs.
  function featureState(x, layers) {
    let state = [[1, 0], [0, 0], [0, 0], [0, 0]];
    const cross = (Math.PI - x[0]) * (Math.PI - x[1]);
    for (let layer = 0; layer < layers; layer++) {
      const h = [];
      for (let k = 0; k < 4; k++) {
        let re = 0, im = 0;
        for (let j = 0; j < 4; j++) {
          const bits = k & j;
          const sign = ((bits & 1) + ((bits >> 1) & 1)) % 2 === 0 ? 1 : -1;
          re += sign * state[j][0];
          im += sign * state[j][1];
        }
        h.push([re / 2, im / 2]);
      }
      state = h.map(function (amp, k) {
        const z1 = k & 2 ? -1 : 1, z2 = k & 1 ? -1 : 1;
        const phase = x[0] * z1 + x[1] * z2 + cross * z1 * z2;
        const c = Math.cos(phase), s = Math.sin(phase);
        return [amp[0] * c - amp[1] * s, amp[0] * s + amp[1] * c];
      });
    }
    return state;
  }

  // Kernel entry = probability of measuring |00> in the overlap circuit = |<phi(a)|phi(b)>|^2.
  function quantumKernel(layers) {
    return function (a, b) {
      const sa = featureState(a, layers), sb = featureState(b, layers);
      let re = 0, im = 0;
      for (let k = 0; k < 4; k++) {
        re += sa[k][0] * sb[k][0] + sa[k][1] * sb[k][1];
        im += sa[k][0] * sb[k][1] - sa[k][1] * sb[k][0];
      }
      return re * re + im * im;
    };
  }

  // ---- Classical kernels for comparison ---------------------------------
  function linearKernel(a, b) { return a[0] * b[0] + a[1] * b[1]; }
  function rbfKernel(gamma) {
    return function (a, b) {
      const dx = a[0] - b[0], dy = a[1] - b[1];
      return Math.exp(-gamma * (dx * dx + dy * dy));
    };
  }

  // ---- Support vector machine (simplified SMO) --------------------------
  function trainSVM(train, kernel, options) {
    const C = (options && options.C) || 1, tol = 1e-3, maxPasses = 8;
    const rand = rng((options && options.seed) || 7);
    const n = train.length, alpha = new Array(n).fill(0);
    let b = 0, passes = 0, rounds = 0;
    const K = train.map(function (p) { return train.map(function (q) { return kernel(p.x, q.x); }); });
    function f(i) {
      let s = b;
      for (let j = 0; j < n; j++) if (alpha[j] !== 0) s += alpha[j] * train[j].y * K[j][i];
      return s;
    }
    while (passes < maxPasses && rounds < 400) {
      let changed = 0;
      for (let i = 0; i < n; i++) {
        const Ei = f(i) - train[i].y;
        if ((train[i].y * Ei < -tol && alpha[i] < C) || (train[i].y * Ei > tol && alpha[i] > 0)) {
          let j = Math.floor(rand() * (n - 1));
          if (j >= i) j++;
          const Ej = f(j) - train[j].y, ai = alpha[i], aj = alpha[j];
          const same = train[i].y === train[j].y;
          const L = same ? Math.max(0, ai + aj - C) : Math.max(0, aj - ai);
          const H = same ? Math.min(C, ai + aj) : Math.min(C, C + aj - ai);
          if (L === H) continue;
          const eta = 2 * K[i][j] - K[i][i] - K[j][j];
          if (eta >= 0) continue;
          alpha[j] = Math.min(H, Math.max(L, aj - train[j].y * (Ei - Ej) / eta));
          if (Math.abs(alpha[j] - aj) < 1e-5) continue;
          alpha[i] = ai + train[i].y * train[j].y * (aj - alpha[j]);
          const b1 = b - Ei - train[i].y * (alpha[i] - ai) * K[i][i] - train[j].y * (alpha[j] - aj) * K[i][j];
          const b2 = b - Ej - train[i].y * (alpha[i] - ai) * K[i][j] - train[j].y * (alpha[j] - aj) * K[j][j];
          b = alpha[i] > 0 && alpha[i] < C ? b1 : alpha[j] > 0 && alpha[j] < C ? b2 : (b1 + b2) / 2;
          changed++;
        }
      }
      passes = changed === 0 ? passes + 1 : 0;
      rounds++;
    }
    const support = [];
    for (let i = 0; i < n; i++) if (alpha[i] > 1e-6) support.push({ x: train[i].x, w: alpha[i] * train[i].y });
    function score(x) {
      let s = b;
      for (let i = 0; i < support.length; i++) s += support[i].w * kernel(support[i].x, x);
      return s;
    }
    return { score: score, predict: function (x) { return score(x) >= 0 ? 1 : -1; }, supportCount: support.length };
  }

  function accuracy(model, points) {
    let right = 0;
    for (let i = 0; i < points.length; i++) if (model.predict(points[i].x) === points[i].y) right++;
    return right / points.length;
  }

  const api = {
    rng: rng, circles: circles, moons: moons, split: split, featureState: featureState,
    quantumKernel: quantumKernel, linearKernel: linearKernel, rbfKernel: rbfKernel,
    trainSVM: trainSVM, accuracy: accuracy,
  };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.QK = api;
})(typeof self !== 'undefined' ? self : this);
