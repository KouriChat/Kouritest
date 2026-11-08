/* Provider calls. All requests are issued from the browser unless the user
   selects the local Flask relay, which exists only to bypass CORS. */
(function () {
  'use strict';

  function joinUrl(base, suffix) {
    const b = String(base || '').trim().replace(/\/+$/, '');
    if (!b) throw new Error('请先填写接口地址');
    if (/\/responses$/i.test(b) || /\/chat\/completions$/i.test(b)) return b;
    return b + suffix;
  }

  function authHeaders(key) {
    const h = { 'Content-Type': 'application/json' };
    if (key) h.Authorization = 'Bearer ' + key;
    return h;
  }

  function errorText(status, body) {
    let detail = body;
    try {
      const parsed = JSON.parse(body);
      detail = (parsed.error && (parsed.error.message || parsed.error)) || parsed.message || body;
      if (typeof detail !== 'string') detail = JSON.stringify(detail);
    } catch (_) { /* keep raw */ }
    return `HTTP ${status}: ${String(detail).slice(0, 1200)}`;
  }

  function buildChatBody(cfg, prompt) {
    const body = {
      model: cfg.model,
      messages: [
        { role: 'system', content: '你是严谨的地理知识地图生成助手。严格按用户要求输出，不要联网或读取外部地图。' },
        { role: 'user', content: prompt }
      ],
      stream: !!cfg.stream
    };
    body[cfg.tokenField] = cfg.maxTokens;
    if (cfg.reasoning) body.reasoning_effort = cfg.reasoning;
    return body;
  }

  function buildResponsesBody(cfg, prompt) {
    const body = {
      model: cfg.model,
      input: [{ role: 'user', content: [{ type: 'input_text', text: prompt }] }],
      max_output_tokens: cfg.maxTokens,
      stream: !!cfg.stream
    };
    if (cfg.reasoning) body.reasoning = { effort: cfg.reasoning };
    return body;
  }

  function extractChatText(data) {
    const choice = data && data.choices && data.choices[0];
    if (!choice) return '';
    const msg = choice.message || {};
    if (typeof msg.content === 'string') return msg.content;
    if (Array.isArray(msg.content)) {
      return msg.content.map((part) => (typeof part === 'string' ? part : (part.text || ''))).join('');
    }
    return choice.text || '';
  }

  function extractResponsesText(data) {
    if (!data) return '';
    if (typeof data.output_text === 'string' && data.output_text) return data.output_text;
    const parts = [];
    for (const item of data.output || []) {
      if (item && item.type === 'message' && Array.isArray(item.content)) {
        for (const c of item.content) {
          if (c && typeof c.text === 'string') parts.push(c.text);
          else if (c && typeof c.output_text === 'string') parts.push(c.output_text);
        }
      }
    }
    return parts.join('');
  }

  function readSse(response, onDelta, signal) {
    return new Promise(async (resolve, reject) => {
      const reader = response.body.getReader();
      const decoder = new TextDecoder('utf-8');
      let buffer = '';
      let collected = '';
      let finalData = null;
      const abort = () => { try { reader.cancel(); } catch (_) {} };
      if (signal) signal.addEventListener('abort', abort, { once: true });
      try {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          let idx;
          while ((idx = buffer.indexOf('\n')) >= 0) {
            const line = buffer.slice(0, idx).replace(/\r$/, '');
            buffer = buffer.slice(idx + 1);
            if (!line.startsWith('data:')) continue;
            const payload = line.slice(5).trim();
            if (!payload || payload === '[DONE]') continue;
            let evt;
            try { evt = JSON.parse(payload); } catch (_) { continue; }
            const type = evt.type || '';
            if (type === 'response.output_text.delta' && typeof evt.delta === 'string') {
              collected += evt.delta;
              if (onDelta) onDelta(collected);
            } else if (type === 'response.completed' || type === 'response.done') {
              finalData = evt.response || evt;
            } else if (type === 'response.failed' || type === 'error') {
              reject(new Error('接口返回错误：' + JSON.stringify(evt.error || evt).slice(0, 800)));
              return;
            } else if (evt.choices && evt.choices[0]) {
              const delta = evt.choices[0].delta || {};
              if (typeof delta.content === 'string' && delta.content) {
                collected += delta.content;
                if (onDelta) onDelta(collected);
              }
              if (evt.choices[0].finish_reason) finalData = evt;
            }
          }
        }
        resolve({ text: collected, data: finalData });
      } catch (err) {
        reject(err);
      } finally {
        if (signal) signal.removeEventListener('abort', abort);
      }
    });
  }

  async function callDirect(cfg, prompt, signal, onDelta) {
    const wire = cfg.wire === 'responses' ? 'responses' : 'chat';
    const suffix = wire === 'responses' ? '/responses' : '/chat/completions';
    const url = joinUrl(cfg.baseUrl, suffix);
    const body = wire === 'responses' ? buildResponsesBody(cfg, prompt) : buildChatBody(cfg, prompt);
    let response;
    try {
      response = await fetch(url, {
        method: 'POST',
        headers: authHeaders(cfg.apiKey),
        body: JSON.stringify(body),
        signal
      });
    } catch (err) {
      if (err && err.name === 'AbortError') throw err;
      throw new Error('网络请求失败（可能是 CORS 或地址不可达）：' + (err.message || err));
    }
    if (!response.ok) {
      const text = await response.text().catch(() => '');
      throw new Error(errorText(response.status, text));
    }
    const ctype = (response.headers.get('content-type') || '').toLowerCase();
    if (cfg.stream && ctype.includes('event-stream')) {
      const { text, data } = await readSse(response, onDelta, signal);
      const finalText = text || (wire === 'responses' ? extractResponsesText(data) : extractChatText(data));
      return { text: finalText, raw: data };
    }
    const data = await response.json();
    return {
      text: wire === 'responses' ? extractResponsesText(data) : extractChatText(data),
      raw: data
    };
  }

  async function callProxy(cfg, prompt, signal) {
    const url = cfg.proxyUrl;
    if (!url) throw new Error('请填写本地代理地址');
    const body = {
      baseUrl: cfg.baseUrl,
      apiKey: cfg.apiKey,
      model: cfg.model,
      wire: cfg.wire,
      tokenField: cfg.tokenField,
      maxTokens: cfg.maxTokens,
      reasoning: cfg.reasoning,
      stream: false,
      prompt
    };
    let response;
    try {
      response = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
        signal
      });
    } catch (err) {
      if (err && err.name === 'AbortError') throw err;
      throw new Error('无法连接本地代理：' + (err.message || err) + '（请先运行 python webapp/server.py）');
    }
    const text = await response.text();
    if (!response.ok) throw new Error(errorText(response.status, text));
    let data;
    try { data = JSON.parse(text); } catch (_) { throw new Error('代理返回了非 JSON 内容'); }
    if (data.error) throw new Error(String(data.error));
    return { text: data.text || '', raw: data.raw || null, usage: data.usage || null };
  }

  async function callModel(cfg, prompt, signal, onDelta) {
    if (cfg.transport === 'proxy') return callProxy(cfg, prompt, signal);
    return callDirect(cfg, prompt, signal, onDelta);
  }

  window.MapApi = {
    callModel, buildChatBody, buildResponsesBody, extractChatText, extractResponsesText
  };
})();