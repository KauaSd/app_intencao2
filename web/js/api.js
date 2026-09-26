// api.js — HTTP client for the intent detection API

const BASE = '';

async function request(path, options = {}) {
  const url = `${BASE}${path}`;
  const response = await fetch(url, {
    headers: { 'Content-Type': 'application/json', ...options.headers },
    ...options,
  });
  const data = await response.json().catch(() => ({}));
  return { ok: response.ok, status: response.status, data };
}

export async function detect(text, opts = {}) {
  return request('/api/v1/detect', {
    method: 'POST',
    body: JSON.stringify({ text, require_supported: opts.requireSupported ?? false }),
  });
}

export async function batch(texts, opts = {}) {
  return request('/api/v1/detect/batch', {
    method: 'POST',
    body: JSON.stringify({ texts, require_supported: opts.requireSupported ?? false }),
  });
}

export async function listIntents(lang) {
  const params = lang ? `?language=${encodeURIComponent(lang)}` : '';
  return request(`/api/v1/intents${params}`);
}

export async function getRules(lang) {
  return request(`/api/v1/rules/${encodeURIComponent(lang)}`);
}

export async function health() {
  return request('/api/v1/health');
}

export async function chat(message, sessionId, opts = {}) {
  return request('/api/v1/chat', {
    method: 'POST',
    body: JSON.stringify({ message, session_id: sessionId, require_supported: opts.requireSupported ?? false }),
  });
}

export async function validateRules(document) {
  return request('/api/v1/rules/validate', {
    method: 'POST',
    body: JSON.stringify(document),
  });
}

export async function getConversations() {
  return request('/api/v1/conversations');
}