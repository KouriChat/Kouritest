/* PNG decoding in the browser, using the same rules as the Python harness:
   convert to grayscale (luminance), threshold at 127.5, then accept either the
   native grid, an integer upscale (block majority), or a box-filter resample. */
(function () {
  'use strict';

  function loadImage(file) {
    return new Promise((resolve, reject) => {
      const url = URL.createObjectURL(file);
      const img = new Image();
      img.onload = () => { URL.revokeObjectURL(url); resolve(img); };
      img.onerror = () => { URL.revokeObjectURL(url); reject(new Error('无法读取图片')); };
      img.src = url;
    });
  }

  function imageToGray(img) {
    const w = img.naturalWidth || img.width;
    const h = img.naturalHeight || img.height;
    const canvas = document.createElement('canvas');
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    ctx.drawImage(img, 0, 0);
    const data = ctx.getImageData(0, 0, w, h).data;
    const gray = new Uint8Array(w * h);
    for (let i = 0, j = 0; i < data.length; i += 4, j++) {
      // PIL convert('L') uses ITU-R 601-2 luma.
      gray[j] = Math.round(0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2]);
    }
    return { gray, width: w, height: h };
  }

  function decodePixels(gray, width, height, task) {
    const land = new Uint8Array(width * height);
    for (let i = 0; i < land.length; i++) land[i] = gray[i] > 127.5 ? 1 : 0;
    const h = task.rows, w = task.cols;
    if (height === h && width === w) {
      return { matrix: land, kind: 'exact', how: `${width}x${height} -> used as-is` };
    }
    if (height % h === 0 && width % w === 0) {
      const by = height / h, bx = width / w;
      const grid = new Uint8Array(h * w);
      for (let r = 0; r < h; r++) {
        for (let c = 0; c < w; c++) {
          let sum = 0;
          for (let dy = 0; dy < by; dy++) {
            for (let dx = 0; dx < bx; dx++) {
              sum += land[(r * by + dy) * width + (c * bx + dx)];
            }
          }
          grid[r * w + c] = sum * 2 >= by * bx ? 1 : 0;
        }
      }
      const kind = height === task.upscale[1] && width === task.upscale[0] ? 'upscale' : 'scaled';
      return { matrix: grid, kind, how: `${width}x${height} -> ${by}x${bx} block majority vote` };
    }
    // Generic box resample: average each destination cell's source footprint.
    const grid = new Uint8Array(h * w);
    for (let r = 0; r < h; r++) {
      const y0 = r * height / h, y1 = (r + 1) * height / h;
      for (let c = 0; c < w; c++) {
        const x0 = c * width / w, x1 = (c + 1) * width / w;
        let sum = 0, weight = 0;
        for (let y = Math.floor(y0); y < Math.ceil(y1); y++) {
          const wy = Math.min(y + 1, y1) - Math.max(y, y0);
          if (wy <= 0) continue;
          for (let x = Math.floor(x0); x < Math.ceil(x1); x++) {
            const wx = Math.min(x + 1, x1) - Math.max(x, x0);
            if (wx <= 0) continue;
            sum += land[y * width + x] * wx * wy;
            weight += wx * wy;
          }
        }
        grid[r * w + c] = weight && sum / weight >= 0.5 ? 1 : 0;
      }
    }
    return { matrix: grid, kind: 'other', how: `${width}x${height} -> box-filter resample (lossy)` };
  }

  async function decodeFile(file, task) {
    const img = await loadImage(file);
    const { gray, width, height } = imageToGray(img);
    const decoded = decodePixels(gray, width, height, task);
    return Object.assign({ width, height }, decoded);
  }

  window.MapImage = { decodeFile, decodePixels };
})();