'use strict';
/* Unit tests for the scoring core. Loads the shipped asset bundle so the tests
   exercise exactly what the page uses. Run: node webapp/tests/score.test.js */
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const WEBAPP = path.resolve(__dirname, '..');
const sandbox = { window: {}, console, Uint8Array, Math, Array, Number, String, JSON, TextDecoder };
sandbox.globalThis = sandbox.window;
vm.createContext(sandbox);
for (const file of ['js/tasks.js', 'js/score.js', 'js/assets-data.js']) {
  vm.runInContext(fs.readFileSync(path.join(WEBAPP, file), 'utf8'), sandbox, { filename: file });
}
const { MapTasks, MapScore } = sandbox.window;
const assets = sandbox.window.MAP_ASSETS;

let failures = 0;
function check(label, ok, extra) {
  console.log(`${ok ? 'PASS' : 'FAIL'} ${label}${extra ? ' :: ' + extra : ''}`);
  if (!ok) failures++;
}

for (const key of MapTasks.TASK_ORDER) {
  const task = MapTasks.TASKS[key];
  const ref = MapScore.parseReference(assets.refs[key], task);
  check(`${key} reference shape`, ref.length === task.rows * task.cols, `${task.rows}x${task.cols}`);

  const self = MapScore.scoreMatrix(ref, ref, task);
  check(`${key} self-score is perfect`, self.errors === 0 && self.score === self.maxScore && self.hss === 1,
    `score=${self.score}/${self.maxScore} hss=${self.hss}`);
  check(`${key} normalises to 16200`, Math.abs(self.normalized - 16200) < 1e-9, String(self.normalized));

  // An inverted matrix must score worse and be flagged as a likely flip.
  const inverted = new Uint8Array(ref.length);
  for (let i = 0; i < ref.length; i++) inverted[i] = ref[i] ? 0 : 1;
  const inv = MapScore.scoreMatrix(inverted, ref, task);
  check(`${key} inverted matrix is penalised`, inv.errors > 0 && !inv.orientationOk,
    `errors=${inv.errors} orientationOk=${inv.orientationOk}`);

  // Round trips through the formats the model is asked to produce.
  const csv = MapScore.matrixToCsv(ref, task);
  const csvParsed = MapScore.parseMatrix('BEGIN_MATRIX\n' + csv + 'END_MATRIX', task);
  check(`${key} parses CSV output`, csvParsed.ok && MapScore.scoreMatrix(csvParsed.matrix, ref, task).errors === 0);

  const compact = [];
  for (let r = 0; r < task.rows; r++) {
    let line = '';
    for (let c = 0; c < task.cols; c++) line += ref[r * task.cols + c] ? '1' : '0';
    compact.push(line);
  }
  const compactParsed = MapScore.parseMatrix('```\n' + compact.join('\n') + '\n```', task);
  check(`${key} parses fenced compact output`, compactParsed.ok && MapScore.scoreMatrix(compactParsed.matrix, ref, task).errors === 0);

  // One deliberately wrong cell must cost exactly one point.
  const oneOff = Uint8Array.from(ref);
  oneOff[0] = ref[0] ? 0 : 1;
  const diff = MapScore.scoreMatrix(oneOff, ref, task);
  check(`${key} single wrong cell costs one point`, diff.errors === 1 && diff.score === diff.maxScore - 1,
    `errors=${diff.errors}`);

  // A single flattened line must still be split back into rows.
  const flat = new Uint8Array(ref.length);
  let flatText = '';
  for (let r = 0; r < task.rows; r++) {
    for (let c = 0; c < task.cols; c++) flatText += ref[r * task.cols + c] ? '1' : '0';
  }
  const flatParsed = MapScore.parseMatrix('BEGIN_MATRIX\n' + flatText + '\nEND_MATRIX', task);
  check(`${key} parses a single flattened line`, flatParsed.ok && MapScore.scoreMatrix(flatParsed.matrix, ref, task).errors === 0);

  // A space-separated body must also work.
  const spaced = compact.map((line) => line.split('').join(' ')).join('\n');
  const spacedParsed = MapScore.parseMatrix(spaced, task);
  check(`${key} parses space-separated output`, spacedParsed.ok && MapScore.scoreMatrix(spacedParsed.matrix, ref, task).errors === 0);

  // A truncated matrix must be rejected instead of silently scored.
  const short = compact.slice(0, task.rows - 1).join('\n');
  check(`${key} rejects truncated output`, MapScore.parseMatrix(short, task).ok === false);
}

console.log(failures ? `\n${failures} check(s) failed` : '\nall checks passed');
process.exit(failures ? 1 : 0);