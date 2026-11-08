/* Matrix parsing, scoring and diagnostics. Mirrors run_map_experiments.py score_matrix
   and the relevant parts of quant/quantify.py (accuracy, HSS, boundary F1@1). */
(function () {
  'use strict';

  const { GLOBAL_CELLS, cells } = window.MapTasks;

  function stripFences(text) {
    let t = String(text == null ? '' : text);
    t = t.replace(/\r\n?/g, '\n');
    // Remove fenced blocks but keep their contents.
    t = t.replace(/```[a-zA-Z0-9_-]*\n?/g, '').replace(/```/g, '');
    return t;
  }

  function extractBetweenMarkers(text) {
    const t = stripFences(text);
    const begin = /BEGIN[_\s-]*MATRIX/i;
    const end = /END[_\s-]*MATRIX/i;
    const b = t.search(begin);
    if (b < 0) return null;
    const after = t.slice(b).replace(begin, '');
    const e = after.search(end);
    if (e < 0) return null;
    return after.slice(0, e).trim();
  }

  function parseRowsFromText(text, task) {
    const src = extractBetweenMarkers(text) || stripFences(text);
    const lines = src.split('\n');
    const rows = [];
    for (const raw of lines) {
      let line = raw.trim();
      if (!line) continue;
      // Drop leading labels such as "row 1:" or bullets.
      line = line.replace(/^[\s>*\-+|]+/, '');
      line = line.replace(/^(?:row|行)\s*\d+\s*[:：]\s*/i, '');
      // Skip obvious prose / header lines.
      if (!/[01]/.test(line)) continue;
      if (/[A-Za-z\u4e00-\u9fff]/.test(line.replace(/[01]/g, ''))) {
        // Allow only separators and labels we stripped; prose is ignored.
        if (!/^[01\s,;|]+$/.test(line)) continue;
      }
      let vals;
      if (/^[01\s,;|]+$/.test(line)) {
        if (/[,\s;|]/.test(line.trim())) {
          vals = line.split(/[,\s;|]+/).filter(Boolean).map(Number);
        } else {
          vals = Array.from(line).map(Number);
        }
      } else {
        // Last resort: pull runs of 0/1 tokens.
        const tokens = line.match(/[01]+/g);
        if (!tokens) continue;
        vals = tokens.length === 1 && tokens[0].length > 2
          ? Array.from(tokens[0]).map(Number)
          : tokens.map(Number);
      }
      if (vals.length === task.cols) rows.push(vals);
      else if (vals.length > task.cols) {
        // Some models wrap long rows; split exact multiples.
        if (vals.length % task.cols === 0) {
          for (let i = 0; i < vals.length; i += task.cols) rows.push(vals.slice(i, i + task.cols));
        }
      }
    }
    return rows;
  }

  function validateShape(rows, task) {
    if (!rows || rows.length !== task.rows) {
      return { ok: false, error: `矩阵行数为 ${rows ? rows.length : 0}，应为 ${task.rows}` };
    }
    for (let r = 0; r < rows.length; r++) {
      if (!Array.isArray(rows[r]) || rows[r].length !== task.cols) {
        return { ok: false, error: `第 ${r + 1} 行有 ${rows[r] ? rows[r].length : 0} 个值，应为 ${task.cols}` };
      }
    }
    return { ok: true };
  }

  function parseMatrix(text, task) {
    const rows = parseRowsFromText(text, task);
    const shape = validateShape(rows, task);
    if (!shape.ok) return shape;
    const flat = new Uint8Array(task.rows * task.cols);
    let i = 0;
    for (const row of rows) for (const v of row) flat[i++] = v ? 1 : 0;
    return { ok: true, matrix: flat, rows };
  }

  function parseReference(csv, task) {
    const rows = [];
    for (const line of csv.split(/\r?\n/)) {
      const s = line.trim();
      if (!s) continue;
      const vals = s.split(/[,\s;]+/).filter(Boolean).map(Number);
      if (vals.length) rows.push(vals);
    }
    const shape = validateShape(rows, task);
    if (!shape.ok) throw new Error('参照矩阵形状异常：' + shape.error);
    const flat = new Uint8Array(task.rows * task.cols);
    let i = 0;
    for (const row of rows) for (const v of row) flat[i++] = v ? 1 : 0;
    return flat;
  }

  function boundarySet(m, rows, cols) {
    const b = new Uint8Array(rows * cols);
    for (let r = 1; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const a = m[(r - 1) * cols + c], x = m[r * cols + c];
        if (a !== x) { b[(r - 1) * cols + c] = 1; b[r * cols + c] = 1; }
      }
    }
    for (let c = 1; c < cols; c++) {
      for (let r = 0; r < rows; r++) {
        const a = m[r * cols + (c - 1)], x = m[r * cols + c];
        if (a !== x) { b[r * cols + (c - 1)] = 1; b[r * cols + c] = 1; }
      }
    }
    return b;
  }

  function dilate(b, rows, cols, radius) {
    let out = b;
    for (let step = 0; step < radius; step++) {
      const nxt = out.slice();
      for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
          if (!out[r * cols + c]) continue;
          if (r > 0) nxt[(r - 1) * cols + c] = 1;
          if (r + 1 < rows) nxt[(r + 1) * cols + c] = 1;
          if (c > 0) nxt[r * cols + (c - 1)] = 1;
          if (c + 1 < cols) nxt[r * cols + (c + 1)] = 1;
        }
      }
      out = nxt;
    }
    return out;
  }

  function boundaryF1(cand, ref, rows, cols, tolerance) {
    const bp = boundarySet(cand, rows, cols);
    const br = boundarySet(ref, rows, cols);
    let np = 0, nr = 0;
    for (let i = 0; i < bp.length; i++) { np += bp[i]; nr += br[i]; }
    if (!np || !nr) return 0;
    const dbp = dilate(bp, rows, cols, tolerance);
    const dbr = dilate(br, rows, cols, tolerance);
    let hitP = 0, hitR = 0;
    for (let i = 0; i < bp.length; i++) { if (bp[i] && dbr[i]) hitP++; if (br[i] && dbp[i]) hitR++; }
    const prec = hitP / np, rec = hitR / nr;
    return prec + rec ? 2 * prec * rec / (prec + rec) : 0;
  }

  function scoreMatrix(cand, ref, task) {
    const n = task.rows * task.cols;
    const diff = new Uint8Array(n);
    let bad = 0, extra = 0, missing = 0, correct = 0;
    let flipV = 0, flipH = 0;
    for (let r = 0; r < task.rows; r++) {
      for (let c = 0; c < task.cols; c++) {
        const i = r * task.cols + c;
        const p = cand[i], t = ref[i];
        if (p !== t) {
          diff[i] = 1;
          bad++;
          if (p && !t) extra++; else if (!p && t) missing++;
        } else correct++;
        if (cand[r * task.cols + c] === ref[(task.rows - 1 - r) * task.cols + c]) flipV++;
        if (cand[r * task.cols + c] === ref[r * task.cols + (task.cols - 1 - c)]) flipH++;
      }
    }
    const acc = n ? 100 * correct / n : 0;
    const fv = n ? 100 * flipV / n : 0;
    const fh = n ? 100 * flipH / n : 0;

    let tp = 0, tn = 0, fp = 0, fn = 0;
    for (let i = 0; i < n; i++) {
      if (cand[i] && ref[i]) tp++;
      else if (!cand[i] && !ref[i]) tn++;
      else if (cand[i] && !ref[i]) fp++;
      else fn++;
    }
    const expected = ((tp + fn) * (tp + fp) + (tn + fp) * (tn + fn)) / (n * n);
    const hss = expected < 1 ? (correct / n - expected) / (1 - expected) : 0;

    return {
      errors: bad,
      correct,
      score: correct,
      maxScore: n,
      cells: n,
      accuracyPct: acc,
      normalized: n ? correct * GLOBAL_CELLS / n : 0,
      extraLand: extra,
      missingLand: missing,
      orientationOk: acc + 1e-12 >= fv && acc + 1e-12 >= fh,
      flipVerticalAccuracy: fv,
      flipHorizontalAccuracy: fh,
      hss,
      boundaryF1d1: boundaryF1(cand, ref, task.rows, task.cols, 1),
      diff
    };
  }

  function matrixToCsv(matrix, task) {
    const lines = [];
    for (let r = 0; r < task.rows; r++) {
      const row = [];
      for (let c = 0; c < task.cols; c++) row.push(matrix[r * task.cols + c] ? '1' : '0');
      lines.push(row.join(','));
    }
    return lines.join('\n') + '\n';
  }

  window.MapScore = {
    parseMatrix, parseReference, scoreMatrix, matrixToCsv,
    boundarySet, dilate, boundaryF1
  };
})();