// Reproduces the numbers quoted on the slides:  node experiment.js
const QK = require('./qkernel.js');

const datasets = { circles: QK.circles(120, 0.06, 11), moons: QK.moons(120, 0.08, 11) };
const kernels = {
  'Linear (classical)': QK.linearKernel,
  'RBF (classical)': QK.rbfKernel(2),
  'Quantum, 1 layer': QK.quantumKernel(1),
  'Quantum, 2 layers': QK.quantumKernel(2),
};

for (const [name, points] of Object.entries(datasets)) {
  const { train, test } = QK.split(points, 2 / 3);
  console.log(`\n${name}: ${train.length} training points, ${test.length} test points`);
  for (const [label, kernel] of Object.entries(kernels)) {
    const model = QK.trainSVM(train, kernel, { C: 1 });
    const pct = (v) => (100 * v).toFixed(1).padStart(5) + '%';
    console.log(`  ${label.padEnd(20)} train ${pct(QK.accuracy(model, train))}  test ${pct(QK.accuracy(model, test))}  support vectors ${model.supportCount}`);
  }
}
