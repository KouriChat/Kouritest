/* Canvas rendering: nearest-neighbour upscale so cells stay square and crisp. */
(function () {
  'use strict';

  function drawMatrix(canvas, matrix, task, options) {
    const opts = options || {};
    const scale = opts.scale || 1;
    const w = task.cols * scale;
    const h = task.rows * scale;
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext('2d');
    ctx.imageSmoothingEnabled = false;
    const img = ctx.createImageData(w, h);
    for (let r = 0; r < task.rows; r++) {
      for (let c = 0; c < task.cols; c++) {
        const land = matrix[r * task.cols + c];
        const v = land ? 255 : 0;
        for (let dy = 0; dy < scale; dy++) {
          let o = ((r * scale + dy) * w + c * scale) * 4;
          for (let dx = 0; dx < scale; dx++) {
            img.data[o] = v; img.data[o + 1] = v; img.data[o + 2] = v; img.data[o + 3] = 255;
            o += 4;
          }
        }
      }
    }
    ctx.putImageData(img, 0, 0);
  }

  function drawDiff(canvas, cand, ref, task) {
    const scale = 2;
    const w = task.cols * scale, h = task.rows * scale;
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext('2d');
    ctx.imageSmoothingEnabled = false;
    const img = ctx.createImageData(w, h);
    for (let r = 0; r < task.rows; r++) {
      for (let c = 0; c < task.cols; c++) {
        const i = r * task.cols + c;
        const p = cand[i], t = ref[i];
        let col;
        if (p === t) col = p ? [60, 60, 60] : [0, 0, 0];
        else if (p && !t) col = [255, 60, 60];
        else col = [60, 160, 255];
        for (let dy = 0; dy < scale; dy++) {
          let o = ((r * scale + dy) * w + c * scale) * 4;
          for (let dx = 0; dx < scale; dx++) {
            img.data[o] = col[0]; img.data[o + 1] = col[1]; img.data[o + 2] = col[2]; img.data[o + 3] = 255;
            o += 4;
          }
        }
      }
    }
    ctx.putImageData(img, 0, 0);
  }

  function clearCanvas(canvas) {
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#0a0a0a';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
  }

  window.MapRender = { drawMatrix, drawDiff, clearCanvas };
})();