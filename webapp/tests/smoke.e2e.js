'use strict';
/* End-to-end smoke test: real browser + fake OpenAI-compatible endpoint + the
   same local static server the page uses. Verifies: page boots, a request is
   issued, the reply is parsed and scored, the manual text path scores, and the
   PNG upload path decodes an integer upscale with block-majority voting.

   Prerequisites: a headless Edge with --remote-debugging-port=9333 and a static
   server on this page. See webapp/README.md for the exact commands. */
const fs = require('node:fs');
const http = require('node:http');
const path = require('node:path');

const WEBAPP = path.resolve(__dirname, '..');
const ROOT = path.resolve(WEBAPP, '..');
const REFS = {
  global: 'template_land_water_90x180.csv',
  csc: 'template_region_south_china_sea_111x141.csv',
  dzh: 'template_region_mediterranean_71x231.csv',
  seto: 'template_region_seto_inland_sea_64x253.csv',
  arctic01: 'template_region_arctic_nwp_61x241.csv',
  hecate: 'template_region_hecate_strait_127x127.csv'
};
const TASK = process.env.SMOKE_TASK || 'hecate';
const PAGE = process.env.SMOKE_PAGE || 'http://127.0.0.1:8765/';
const TRANSPORT = process.env.SMOKE_TRANSPORT || 'direct';
const WIRE = process.env.SMOKE_WIRE || 'chat';
const STREAM = process.env.SMOKE_STREAM === '1';
const CDP = process.env.SMOKE_CDP || 'http://127.0.0.1:9333';
const CSV = fs.readFileSync(path.join(ROOT, 'quant', 'refs', REFS[TASK]), 'utf8');

function matrixText() {
  const rows = [];
  for (const line of CSV.split(/\r?\n/)) {
    const s = line.trim();
    if (!s) continue;
    rows.push(s.split(',').map((v) => (Number(v) ? '1' : '0')).join(''));
  }
  return 'BEGIN_MATRIX\n' + rows.join('\n') + '\nEND_MATRIX\n';
}

let failures = 0;
function check(label, ok, extra) {
  console.log(`${ok ? 'PASS' : 'FAIL'} ${label}${extra ? ' :: ' + extra : ''}`);
  if (!ok) failures++;
}

const PORT = 9321;
const server = http.createServer((req, res) => {
  const cors = {
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Headers': 'Content-Type, Authorization',
    'Access-Control-Allow-Methods': 'GET, POST, OPTIONS'
  };
  if (req.method === 'OPTIONS') { res.writeHead(204, cors); res.end(); return; }
  const wantStream = STREAM;
  if (req.method === 'POST' && req.url.startsWith('/v1/chat/completions')) {
    let body = '';
    req.on('data', (c) => { body += c; });
    req.on('end', () => {
      const text = /broken/.test(body || '') ? matrixText().split(String.fromCharCode(10)).slice(0, 5).join(String.fromCharCode(10)) : matrixText();
      if (wantStream) {
        res.writeHead(200, Object.assign({ 'Content-Type': 'text/event-stream' }, cors));
        // Emit the matrix in a few delta chunks, like a real provider.
        const step = Math.ceil(text.length / 4);
        for (let i = 0; i < text.length; i += step) {
          const delta = text.slice(i, i + step);
          res.write('data: ' + JSON.stringify({ choices: [{ delta: { content: delta } }] }) + '\n\n');
        }
        res.write('data: ' + JSON.stringify({ choices: [{ delta: {}, finish_reason: 'stop' }] }) + '\n\n');
        res.write('data: [DONE]\n\n');
        res.end();
        return;
      }
      const reply = {
        choices: [{ message: { role: 'assistant', content: text }, finish_reason: 'stop' }],
        usage: { prompt_tokens: 10, completion_tokens: 5000 }
      };
      res.writeHead(200, Object.assign({ 'Content-Type': 'application/json' }, cors));
      res.end(JSON.stringify(reply));
    });
    return;
  }
  if (req.method === 'POST' && req.url.startsWith('/v1/responses')) {
    let body = '';
    req.on('data', (c) => { body += c; });
    req.on('end', () => {
      const text = /broken/.test(body || '') ? matrixText().split(String.fromCharCode(10)).slice(0, 5).join(String.fromCharCode(10)) : matrixText();
      if (wantStream) {
        res.writeHead(200, Object.assign({ 'Content-Type': 'text/event-stream' }, cors));
        const step = Math.ceil(text.length / 4);
        for (let i = 0; i < text.length; i += step) {
          const delta = text.slice(i, i + step);
          res.write('data: ' + JSON.stringify({ type: 'response.output_text.delta', delta }) + '\n\n');
        }
        res.write('data: ' + JSON.stringify({ type: 'response.completed', response: { status: 'completed', output: [] } }) + '\n\n');
        res.end();
        return;
      }
      const reply = {
        status: 'completed',
        output_text: text,
        output: [{ type: 'message', role: 'assistant', content: [{ type: 'output_text', text }] }],
        usage: { input_tokens: 10, output_tokens: 5000 }
      };
      res.writeHead(200, Object.assign({ 'Content-Type': 'application/json' }, cors));
      res.end(JSON.stringify(reply));
    });
    return;
  }
  // Stand-in for the optional Flask relay; echoes the same matrix back.
  if (req.method === 'POST' && req.url.startsWith('/api/proxy')) {
    let body = '';
    req.on('data', (c) => { body += c; });
    req.on('end', () => {
      let parsed = {};
      try { parsed = JSON.parse(body); } catch (_) {}
      const ok = parsed.baseUrl && parsed.model && typeof parsed.prompt === 'string';
      const relayText = /broken/.test(parsed.model || '') ? matrixText().split(String.fromCharCode(10)).slice(0, 5).join(String.fromCharCode(10)) : matrixText();
      res.writeHead(ok ? 200 : 400, Object.assign({ 'Content-Type': 'application/json' }, cors));
      res.end(JSON.stringify(ok
        ? { text: relayText, raw: { relay: true } }
        : { error: 'mock relay requires baseUrl, model and prompt' }));
    });
    return;
  }
  res.writeHead(404, cors);
  res.end('not found');
});

function rpc(ws, id, method, params) {
  return new Promise((resolve, reject) => {
    const onMessage = (event) => {
      let msg;
      try { msg = JSON.parse(event.data); } catch (_) { return; }
      if (msg.id !== id) return;
      ws.removeEventListener('message', onMessage);
      if (msg.error) reject(new Error(method + ': ' + JSON.stringify(msg.error)));
      else resolve(msg.result);
    };
    ws.addEventListener('message', onMessage);
    ws.send(JSON.stringify({ id, method, params: params || {} }));
  });
}

let rpcId = 0;
function evaluate(ws, expr) {
  return rpc(ws, ++rpcId, 'Runtime.evaluate', {
    expression: expr, returnByValue: true, awaitPromise: true
  }).then((r) => {
    if (r.exceptionDetails) throw new Error('page error: ' + JSON.stringify(r.exceptionDetails).slice(0, 800));
    return r.result ? r.result.value : undefined;
  });
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// Scripts are plain strings (not template literals) to keep backticks out of the file.
function manualTextScript(task) {
  return '(() => {'
    + "document.getElementById('manualTask').value = '" + task + "';"
    + "const refs = window.MAP_ASSETS.refs['" + task + "'].split(/\\r?\\n/).filter(Boolean);"
    + "const fence = String.fromCharCode(96).repeat(3);"
    + "document.getElementById('manualText').value = fence + '\\n' + refs.map(r => r.split(',').join('')).join('\\n') + '\\n' + fence;"
    + "document.getElementById('manualScore').click();"
    + 'return true;'
    + '})()';
}

async function main() {
  await new Promise((r) => server.listen(PORT, '127.0.0.1', r));
  const info = await (await fetch(`${CDP}/json/new?about:blank`, { method: 'PUT' })).json();
  const ws = new WebSocket(info.webSocketDebuggerUrl);
  await new Promise((resolve, reject) => {
    ws.addEventListener('open', resolve, { once: true });
    ws.addEventListener('error', reject, { once: true });
  });

  const consoleErrors = [];
  ws.addEventListener('message', (event) => {
    let msg; try { msg = JSON.parse(event.data); } catch (_) { return; }
    if (msg.method === 'Runtime.exceptionThrown') consoleErrors.push(JSON.stringify(msg.params).slice(0, 400));
    if (msg.method === 'Runtime.consoleAPICalled' && msg.params.type === 'error') {
      consoleErrors.push((msg.params.args || []).map((a) => a.value || a.description).join(' ').slice(0, 400));
    }
  });

  await rpc(ws, ++rpcId, 'Runtime.enable');
  await rpc(ws, ++rpcId, 'Page.enable');
  await rpc(ws, ++rpcId, 'Page.navigate', { url: PAGE });
  await sleep(1500);

  console.log('transport: ' + TRANSPORT + ' | wire: ' + WIRE + ' | stream: ' + STREAM);
  check('page title', (await evaluate(ws, 'document.title')).includes('地图海陆重建'));
  check('six task checkboxes', (await evaluate(ws, 'document.querySelectorAll("#taskList input").length')) === 6);
  check('prompt preview works', await evaluate(ws, `(() => {
    document.getElementById('promptTask').value = '${TASK}';
    document.getElementById('showPrompt').click();
    const view = document.getElementById('promptView');
    return !view.hidden && view.textContent.includes('BEGIN_MATRIX') && view.textContent.length > 1000;
  })()`));

  await evaluate(ws, `(() => {
    document.getElementById('baseUrl').value = 'http://127.0.0.1:${PORT}/v1';
    document.getElementById('apiKey').value = 'test-key';
    document.getElementById('modelId').value = 'fake/model';
    document.getElementById('transport').value = '${TRANSPORT}';
    document.getElementById('proxyUrl').value = 'http://127.0.0.1:${PORT}/api/proxy';
    document.getElementById('wire').value = '${WIRE}';
    document.getElementById('stream').checked = ${STREAM};
    for (const cb of document.querySelectorAll('#taskList input')) cb.checked = (cb.value === '${TASK}');
    document.getElementById('groups').value = '1';
    document.getElementById('startRun').click();
    return true;
  })()`);

  let rows = [];
  for (let i = 0; i < 90; i++) {
    await sleep(500);
    rows = await evaluate(ws, 'Array.from(document.querySelectorAll("#resultBody tr")).map(tr => tr.innerText)');
    if (rows.length && !rows[0].includes('尚无结果')) break;
  }
  const rowError = await evaluate(ws, 'document.querySelector("#resultBody tr") ? document.querySelector("#resultBody tr").title : ""');
  check('API run produced one scored result', rows.length === 1 && /完成/.test(rows[0]) && /100\.00%/.test(rows[0]),
    rows[0] + (rowError ? ' | title=' + rowError : ''));
  const metrics = await evaluate(ws, 'document.getElementById("metricsBox").innerText');
  check('metrics show a perfect normalised score', /16200\.00 \/ 16200/.test(metrics), metrics.replace(/\s+/g, ' '));

  // Manual text path.
  await evaluate(ws, manualTextScript(TASK));
  await sleep(800);
  check('manual text scoring', /得分 .*\/.*（100\.00%）/.test(await evaluate(ws, 'document.getElementById("manualStatus").innerText')),
    await evaluate(ws, 'document.getElementById("manualStatus").innerText'));

  // PNG upload path: an exact 8x nearest-neighbour upscale must decode identically.
  const pngResult = await evaluate(ws, `(async () => {
    const refs = window.MAP_ASSETS.refs['${TASK}'].split(/\\r?\\n/).filter(Boolean);
    const rows = refs.length, cols = refs[0].split(',').length;
    const scale = 8;
    const canvas = document.createElement('canvas');
    canvas.width = cols * scale; canvas.height = rows * scale;
    const ctx = canvas.getContext('2d');
    const img = ctx.createImageData(canvas.width, canvas.height);
    for (let r = 0; r < rows; r++) {
      const vals = refs[r].split(',');
      for (let c = 0; c < cols; c++) {
        const v = Number(vals[c]) ? 255 : 0;
        for (let dy = 0; dy < scale; dy++) {
          let o = ((r * scale + dy) * canvas.width + c * scale) * 4;
          for (let dx = 0; dx < scale; dx++) { img.data[o]=v; img.data[o+1]=v; img.data[o+2]=v; img.data[o+3]=255; o+=4; }
        }
      }
    }
    ctx.putImageData(img, 0, 0);
    const blob = await new Promise((res) => canvas.toBlob(res, 'image/png'));
    const file = new File([blob], 'upscaled.png', { type: 'image/png' });
    const dt = new DataTransfer();
    dt.items.add(file);
    const input = document.getElementById('manualFile');
    input.files = dt.files;
    document.getElementById('manualScore').click();
    return true;
  })()`);
  await sleep(1200);
  const pngStatus = await evaluate(ws, 'document.getElementById("manualStatus").innerText');
  check('PNG upload scoring at 8x upscale', /100\.00%/.test(pngStatus), pngStatus);
  // A truncated reply must be reported as a parse failure with the raw text kept.
  for (let i = 0; i < 60; i++) {
    const done = await evaluate(ws, '/结束/.test(document.getElementById("progressText").innerText)');
    if (done) break;
    await sleep(250);
  }
  await evaluate(ws, `(() => {
    document.getElementById('modelId').value = 'fake/broken';
    document.getElementById('startRun').click();
    return true;
  })()`);
  let badRow = '';
  for (let i = 0; i < 60; i++) {
    await sleep(500);
    badRow = await evaluate(ws, 'Array.from(document.querySelectorAll("#resultBody tr")).map(tr => tr.innerText).join(" ;; ")');
    if (badRow && !/请求中/.test(badRow)) break;
  }
  check('truncated reply reported as parse failure', /解析失败/.test(badRow), badRow.replace(/\t/g, ' | '));
  const logTail = await evaluate(ws, 'document.getElementById("logBox").textContent.split("\\n").slice(-4).join(" || ")');
  console.log('log tail:', logTail);
  await evaluate(ws, '(() => { const tr = Array.from(document.querySelectorAll("#resultBody tr")).find(x => /解析失败/.test(x.innerText)); if (tr) tr.click(); return !!tr; })()');
  await sleep(300);
  const failMetrics = await evaluate(ws, 'document.getElementById("metricsBox").innerText');
  const rawEnabled = await evaluate(ws, 'document.getElementById("downloadRaw").disabled === false');
  check('failure keeps the raw reply downloadable', rawEnabled && /原始回复/.test(failMetrics), failMetrics.replace(/\s+/g, ' '));

  check('no console errors', consoleErrors.length === 0, consoleErrors.join(' || '));

  console.log(failures ? `\n${failures} check(s) failed` : '\nSMOKE PASS');
  ws.close();
  await fetch(`${CDP}/json/close/${info.id}`).catch(() => {});
  server.close();
  await sleep(200);
  process.exit(failures ? 1 : 0);
}

main().catch((err) => { console.error('SMOKE ERROR', err); server.close(); process.exit(1); });