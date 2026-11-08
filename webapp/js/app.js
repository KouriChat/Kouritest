/* UI wiring and the experiment loop. */
(function () {
  'use strict';

  const { TASKS, TASK_ORDER, cells, normalized, stepText, boundsText } = window.MapTasks;
  const { parseMatrix, parseReference, scoreMatrix, matrixToCsv } = window.MapScore;
  const { drawMatrix, drawDiff, clearCanvas } = window.MapRender;
  const { decodeFile } = window.MapImage;
  const assets = window.MAP_ASSETS || { prompts: {}, refs: {} };

  const $ = (id) => document.getElementById(id);
  const state = {
    results: [],
    selected: null,
    running: false,
    controller: null,
    refs: {},
    batchStarted: ''
  };

  /* ---------------------------------------------------------------- settings */

  const SETTINGS_KEY = 'map-eval-settings-v1';
  const KEY_KEY = 'map-eval-key-v1';

  function readSettings() {
    const modelId = $('modelId').value.trim();
    return {
      modelId,
      model: modelId,
      baseUrl: $('baseUrl').value.trim(),
      apiKey: $('apiKey').value,
      wire: $('wire').value,
      transport: $('transport').value,
      proxyUrl: $('proxyUrl').value.trim(),
      maxTokens: Math.max(256, Number($('maxTokens').value) || 30000),
      tokenField: $('tokenField').value,
      reasoning: $('reasoning').value,
      timeoutMin: Math.max(1, Number($('timeoutMin').value) || 15),
      stream: $('stream').checked
    };
  }

  function applySettings(s) {
    if (!s) return;
    if (s.baseUrl) $('baseUrl').value = s.baseUrl;
    if (s.modelId) $('modelId').value = s.modelId;
    if (s.wire) $('wire').value = s.wire;
    if (s.transport) $('transport').value = s.transport;
    if (s.proxyUrl) $('proxyUrl').value = s.proxyUrl;
    if (s.maxTokens) $('maxTokens').value = s.maxTokens;
    if (s.tokenField) $('tokenField').value = s.tokenField;
    if (s.reasoning != null) $('reasoning').value = s.reasoning;
    if (s.timeoutMin) $('timeoutMin').value = s.timeoutMin;
    if (typeof s.stream === 'boolean') $('stream').checked = s.stream;
  }

  function saveSettings(showMessage) {
    const s = readSettings();
    try {
      localStorage.setItem(SETTINGS_KEY, JSON.stringify(s));
      if ($('rememberKey').checked && s.apiKey) localStorage.setItem(KEY_KEY, s.apiKey);
      else localStorage.removeItem(KEY_KEY);
      if (showMessage) setStatus($('connStatus'), '设置已保存在本机浏览器', 'ok');
    } catch (_) { /* private mode */ }
  }

  function loadSettings() {
    try {
      const raw = localStorage.getItem(SETTINGS_KEY);
      if (raw) applySettings(JSON.parse(raw));
      const key = localStorage.getItem(KEY_KEY);
      if (key) { $('apiKey').value = key; $('rememberKey').checked = true; }
    } catch (_) { /* ignore */ }
    if (!$('proxyUrl').value) $('proxyUrl').value = 'http://127.0.0.1:8788/api/proxy';
  }

  function setStatus(el, text, kind) {
    el.textContent = text || '';
    el.className = 'status' + (kind ? ' ' + kind : '');
  }

  function escapeHtml(text) {
    return String(text).replace(/[&<>"']/g, (ch) => (
      { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[ch]
    ));
  }

  /* ---------------------------------------------------------------- logging */

  function log(text) {
    const box = $('logBox');
    const stamp = new Date().toLocaleTimeString('zh-CN', { hour12: false });
    box.textContent += `[${stamp}] ${text}\n`;
    box.scrollTop = box.scrollHeight;
  }

  /* ---------------------------------------------------------------- prompts */

  function promptFor(taskKey, format) {
    const task = TASKS[taskKey];
    const base = assets.prompts[taskKey] || '';
    const n = cells(task);
    const fmt = format === 'csv'
      ? '每行用逗号分隔（例如 0,1,1,0），不要空格。'
      : '每行写成连续的 0/1 字符串（例如 0110），不要逗号或空格。';
    return `${base}

---
本次为网页版自动评测，不需要你写文件或运行代码。请只输出判断矩阵本身。

【输出格式】
- 在 BEGIN_MATRIX 和 END_MATRIX 两行标记之间给出完整矩阵。
- 共 ${task.rows} 行、每行 ${task.cols} 个值，合计 ${n} 个值；只允许 0 和 1。
- ${fmt}
- 第 1 行对应最北（${boundsText(task)} 的上边），第 1 列对应最西；自上而下、自左而右。
- 不要输出解释、标题、坐标或代码块以外的内容；标记行之外的内容会被忽略。

BEGIN_MATRIX
<在这里逐行写 ${task.rows} 行>
END_MATRIX
`;
  }

  /* ---------------------------------------------------------------- UI setup */

  function buildTaskList() {
    const box = $('taskList');
    box.innerHTML = '';
    for (const key of TASK_ORDER) {
      const task = TASKS[key];
      const label = document.createElement('label');
      const cb = document.createElement('input');
      cb.type = 'checkbox';
      cb.value = key;
      cb.checked = true;
      cb.dataset.task = key;
      const span = document.createElement('span');
      span.innerHTML = `<span class="tlabel">${task.label}</span><br><span class="tmeta">${task.rows}×${task.cols} · ${cells(task).toLocaleString()} 格</span>`;
      label.appendChild(cb);
      label.appendChild(span);
      box.appendChild(label);
    }
  }

  function selectedTasks() {
    return Array.from(document.querySelectorAll('#taskList input:checked')).map((x) => x.value);
  }

  function buildSelects() {
    for (const id of ['promptTask', 'manualTask']) {
      const sel = $(id);
      sel.innerHTML = '';
      for (const key of TASK_ORDER) {
        const o = document.createElement('option');
        o.value = key;
        o.textContent = TASKS[key].label;
        sel.appendChild(o);
      }
    }
  }

  function referenceFor(key) {
    if (!state.refs[key]) {
      state.refs[key] = parseReference(assets.refs[key] || '', TASKS[key]);
    }
    return state.refs[key];
  }

  /* ---------------------------------------------------------------- results */

  function statusLabel(status) {
    return { ok: '完成', parse: '解析失败', error: '请求失败', stopped: '已停止' }[status] || status;
  }

  function addResult(rec) {
    state.results.push(rec);
    renderResults();
  }

  function renderResults() {
    const body = $('resultBody');
    body.innerHTML = '';
    if (!state.results.length) {
      const tr = document.createElement('tr');
      tr.innerHTML = '<td class="empty" colspan="11">尚无结果</td>';
      body.appendChild(tr);
    }
    state.results.forEach((rec, index) => {
      const tr = document.createElement('tr');
      tr.dataset.index = String(index);
      if (state.selected === index) tr.className = 'selected';
      const s = rec.score;
      const scoreText = s ? String(s.score) : '—';
      const accText = s ? s.accuracyPct.toFixed(2) + '%' : '—';
      const cellsText = s ? String(s.maxScore) : '—';
      const statusClass = rec.status === 'ok' ? 'ok' : (rec.status === 'error' ? 'bad' : 'warn');
      tr.innerHTML = `
        <td>${TASKS[rec.task].label}</td>
        <td>${rec.group}</td>
        <td class="${statusClass}">${statusLabel(rec.status)}</td>
        <td>${scoreText}</td>
        <td>${cellsText}</td>
        <td>${accText}</td>
        <td>${s ? s.errors : '—'}</td>
        <td>${s ? s.extraLand : '—'}</td>
        <td>${s ? s.missingLand : '—'}</td>
        <td>${s ? (s.orientationOk ? '正常' : '可能翻转') : '—'}</td>
        <td>${rec.seconds ? rec.seconds.toFixed(1) + 's' : '—'}</td>`;
      if (rec.error) tr.title = rec.error;
      tr.addEventListener('click', () => selectResult(index));
      body.appendChild(tr);
    });
    $('resultCount').textContent = state.results.length ? `共 ${state.results.length} 条` : '';
    $('downloadJson').disabled = !state.results.length;
  }

  function selectResult(index) {
    const rec = state.results[index];
    if (!rec) return;
    state.selected = index;
    renderResults();
    if (!rec.matrix) {
      // Parsing or the request itself failed: show whatever came back so the
      // user can see why, instead of an empty preview.
      $('previewTitle').textContent = `${TASKS[rec.task].label} · 第 ${rec.group} 组 · ${rec.model || ''}`;
      $('metricsBox').innerHTML = '<div>' + (rec.error ? '原因：<b>' + escapeHtml(rec.error) + '</b>' : '没有可评分的矩阵。') + '</div>' +
        (rec.raw ? '<div>原始回复 ' + rec.raw.length.toLocaleString() + ' 字符，可下载查看。</div>' : '');
      clearCanvas($('candCanvas'));
      clearCanvas($('diffCanvas'));
      $('downloadMatrix').disabled = true;
      $('downloadDiff').disabled = true;
      $('downloadRaw').disabled = !rec.raw;
      $('downloadRaw').dataset.index = String(index);
      return;
    }
    const task = TASKS[rec.task];
    const ref = referenceFor(rec.task);
    drawMatrix($('candCanvas'), rec.matrix, task, { scale: rec.scale || 2 });
    drawDiff($('diffCanvas'), rec.matrix, ref, task);
    const s = rec.score;
    $('previewTitle').textContent = `${task.label} · 第 ${rec.group} 组 · ${rec.model || ''}`;
    $('metricsBox').innerHTML = s ? `
      <div>得分 <b>${s.score}</b> / ${s.maxScore}（归一化 <b>${s.normalized.toFixed(2)}</b> / 16200）</div>
      <div>准确率 <b>${s.accuracyPct.toFixed(3)}%</b> · HSS <b>${s.hss.toFixed(4)}</b> · 边界F1@1 <b>${s.boundaryF1d1.toFixed(4)}</b></div>
      <div>虚报陆地 <b>${s.extraLand}</b> 格 · 漏报陆地 <b>${s.missingLand}</b> 格 · 解码 ${rec.decodeHow || '—'}</div>` : '';
    $('downloadMatrix').disabled = false;
    $('downloadDiff').disabled = false;
    $('downloadRaw').disabled = !rec.raw;
    $('downloadMatrix').dataset.index = String(index);
    $('downloadDiff').dataset.index = String(index);
    $('downloadRaw').dataset.index = String(index);
  }

  /* ---------------------------------------------------------------- downloads */

  function download(name, text, type) {
    const blob = new Blob([text], { type: type || 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = name;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 2000);
  }

  function downloadCanvas(canvas, name) {
    canvas.toBlob((blob) => {
      if (!blob) return;
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = name;
      document.body.appendChild(a);
      a.click();
      a.remove();
      setTimeout(() => URL.revokeObjectURL(url), 2000);
    }, 'image/png');
  }

  function safeName(text) {
    return String(text || 'model').replace(/[^A-Za-z0-9._-]+/g, '_').replace(/^_+|_+$/g, '') || 'model';
  }

  /* ---------------------------------------------------------------- run loop */

  async function runOne(taskKey, group, cfg, format) {
    const task = TASKS[taskKey];
    const prompt = promptFor(taskKey, format);
    const started = performance.now();
    let text = '';
    try {
      const res = await window.MapApi.callModel(cfg, prompt, state.controller.signal, (partial) => {
        $('progressText').textContent = `${task.label} · 第 ${group} 组 · 已接收 ${partial.length.toLocaleString()} 字符`;
      });
      text = res.text || '';
      if (!text.trim()) throw new Error('接口返回了空内容');
      const parsed = parseMatrix(text, task);
      if (!parsed.ok) {
        return {
          task: taskKey, group, model: cfg.model, status: 'parse',
          error: parsed.error, raw: text,
          seconds: (performance.now() - started) / 1000
        };
      }
      const ref = referenceFor(taskKey);
      const score = scoreMatrix(parsed.matrix, ref, task);
      return {
        task: taskKey, group, model: cfg.model, status: 'ok',
        matrix: parsed.matrix, score,
        scale: Math.max(1, Math.min(4, Math.floor(900 / task.cols))),
        decodeHow: cfg.transport === 'proxy' ? '本地代理返回的矩阵文本' : '接口返回的矩阵文本',
        raw: text, seconds: (performance.now() - started) / 1000
      };
    } catch (err) {
      const aborted = err && err.name === 'AbortError';
      return {
        task: taskKey, group, model: cfg.model,
        status: aborted ? 'stopped' : 'error',
        error: aborted ? '用户停止' : (err.message || String(err)),
        raw: text, seconds: (performance.now() - started) / 1000
      };
    }
  }

  async function startRun() {
    if (state.running) return;
    const cfg = readSettings();
    const tasks = selectedTasks();
    if (!cfg.baseUrl && cfg.transport === 'direct') { setStatus($('progressText'), '请填写接口地址', 'bad'); return; }
    if (!cfg.modelId) { setStatus($('progressText'), '请填写模型 ID', 'bad'); return; }
    if (!tasks.length) { setStatus($('progressText'), '请至少选择一张地图', 'bad'); return; }
    saveSettings(false);

    const groups = Math.max(1, Math.min(9, Number($('groups').value) || 1));
    const format = $('submitFormat').value;
    state.running = true;
    state.controller = new AbortController();
    state.batchStarted = new Date().toISOString();
    $('startRun').disabled = true;
    $('stopRun').disabled = false;
    $('resultBody').innerHTML = '';
    state.results = [];
    state.selected = null;
    renderResults();
    log(`开始：${tasks.length} 张地图 × ${groups} 组 · ${cfg.modelId}`);

    const total = tasks.length * groups;
    let done = 0;
    for (const key of tasks) {
      for (let g = 1; g <= groups; g++) {
        if (state.controller.signal.aborted) break;
        const task = TASKS[key];
        setStatus($('progressText'), `${task.label} · 第 ${g} 组 · 请求中…`);
        log(`请求 ${task.label} 第 ${g}/${groups} 组（${task.rows}×${task.cols}）`);
        const rec = await runOne(key, g, cfg, format);
        addResult(rec);
        if (rec.status === 'ok') {
          log(`完成 ${task.label} 第 ${g} 组：${rec.score.score}/${rec.score.maxScore}，准确率 ${rec.score.accuracyPct.toFixed(2)}%`);
          if (state.selected == null) selectResult(state.results.length - 1);
        } else {
          log(`失败 ${task.label} 第 ${g} 组：${rec.error}`);
        }
        done++;
        $('progressBar').style.width = (100 * done / total).toFixed(1) + '%';
      }
      if (state.controller.signal.aborted) break;
    }

    state.running = false;
    $('startRun').disabled = false;
    $('stopRun').disabled = true;
    const okCount = state.results.filter((r) => r.status === 'ok').length;
    setStatus($('progressText'), `结束：${okCount}/${total} 组成功`, okCount === total ? 'ok' : 'warn');
    log('本批结束。');
  }

  function stopRun() {
    if (state.controller) state.controller.abort();
    setStatus($('progressText'), '正在停止…', 'warn');
  }

  /* ---------------------------------------------------------------- manual scoring */

  async function manualScore() {
    const key = $('manualTask').value;
    const task = TASKS[key];
    try {
      let matrix = null, how = '';
      const file = $('manualFile').files[0];
      if (file) {
        const decoded = await decodeFile(file, task);
        matrix = decoded.matrix;
        how = decoded.how;
      } else {
        const text = $('manualText').value;
        const parsed = parseMatrix(text, task);
        if (!parsed.ok) throw new Error(parsed.error);
        matrix = parsed.matrix;
        how = '文本矩阵';
      }
      const ref = referenceFor(key);
      const score = scoreMatrix(matrix, ref, task);
      const rec = {
        task: key, group: 0, model: 'manual', status: 'ok',
        matrix, score, decodeHow: how, scale: 2, seconds: 0
      };
      state.results.push(rec);
      renderResults();
      selectResult(state.results.length - 1);
      setStatus($('manualStatus'), `得分 ${score.score}/${score.maxScore}（${score.accuracyPct.toFixed(2)}%）`, 'ok');
    } catch (err) {
      setStatus($('manualStatus'), err.message || String(err), 'bad');
    }
  }

  /* ---------------------------------------------------------------- wiring */

  function wire() {
    $('saveSettings').addEventListener('click', () => saveSettings(true));
    $('toggleKey').addEventListener('click', () => {
      const input = $('apiKey');
      const show = input.type === 'password';
      input.type = show ? 'text' : 'password';
      $('toggleKey').textContent = show ? '隐藏' : '显示';
    });
    $('showPrompt').addEventListener('click', () => {
      const box = $('promptView');
      if (!box.hidden) { box.hidden = true; return; }
      box.textContent = promptFor($('promptTask').value, $('submitFormat').value);
      box.hidden = false;
    });
    $('startRun').addEventListener('click', startRun);
    $('stopRun').addEventListener('click', stopRun);
    $('manualScore').addEventListener('click', manualScore);
    $('testConn').addEventListener('click', testConnection);
    $('downloadJson').addEventListener('click', () => {
      const clean = state.results.map((r) => ({
        task: r.task, taskLabel: TASKS[r.task].label, group: r.group, model: r.model,
        status: r.status, seconds: r.seconds, error: r.error || null,
        decodeHow: r.decodeHow || null,
        score: r.score ? Object.assign({}, r.score, { diff: undefined }) : null
      }));
      const payload = {
        generatedAt: new Date().toISOString(),
        batchStarted: state.batchStarted,
        tasks: selectedTasks(),
        results: clean
      };
      download(`map-eval-${safeName(readSettings().modelId)}.json`, JSON.stringify(payload, null, 2), 'application/json');
    });
    $('downloadMatrix').addEventListener('click', (ev) => {
      const rec = state.results[Number(ev.currentTarget.dataset.index)];
      if (!rec || !rec.matrix) return;
      const task = TASKS[rec.task];
      download(`${safeName(rec.model)}${task.archiveSuffix}.csv`, matrixToCsv(rec.matrix, task), 'text/csv;charset=utf-8');
    });
    $('downloadRaw').addEventListener('click', (ev) => {
      const rec = state.results[Number(ev.currentTarget.dataset.index)];
      if (!rec || !rec.raw) return;
      download(`${safeName(rec.model)}${TASKS[rec.task].archiveSuffix}_last_message.txt`, rec.raw, 'text/plain;charset=utf-8');
    });
    $('downloadDiff').addEventListener('click', (ev) => {
      const rec = state.results[Number(ev.currentTarget.dataset.index)];
      if (!rec) return;
      downloadCanvas($('diffCanvas'), `${safeName(rec.model)}${TASKS[rec.task].archiveSuffix}_diff.png`);
    });
    window.addEventListener('beforeunload', (ev) => {
      if (state.running) { ev.preventDefault(); ev.returnValue = ''; }
    });
  }

  async function testConnection() {
    const cfg = readSettings();
    setStatus($('connStatus'), '测试中…');
    try {
      if (!cfg.baseUrl && cfg.transport === 'direct') throw new Error('请填写接口地址');
      if (!cfg.modelId) throw new Error('请填写模型 ID');
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), 60000);
      const res = await window.MapApi.callModel(
        Object.assign({}, cfg, { stream: false, maxTokens: 64, reasoning: '' }),
        '只回复两个字：可用', controller.signal, null
      );
      clearTimeout(timer);
      setStatus($('connStatus'), '连接成功：' + (res.text || '(空回复)').slice(0, 60), 'ok');
    } catch (err) {
      setStatus($('connStatus'), (err && err.name === 'AbortError') ? '测试超时' : (err.message || String(err)), 'bad');
    }
  }

  function init() {
    buildTaskList();
    buildSelects();
    loadSettings();
    wire();
    renderResults();
    clearCanvas($('candCanvas'));
    clearCanvas($('diffCanvas'));
  }

  document.addEventListener('DOMContentLoaded', init);
})();