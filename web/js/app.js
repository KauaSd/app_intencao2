// app.js — Main application wiring

import * as api from './api.js';
import * as intents from './intents.js';
import { mountChat } from './chat.js';
import { formatConfidence, escapeText } from './format.js';

const state = {
  sessionId: null,
  log: [],
};

const elements = {
  message: document.getElementById('message'),
  expectedIntent: document.getElementById('expected-intent'),
  samples: document.getElementById('samples'),
  requireSupported: document.getElementById('require-supported'),
  detectBtn: document.getElementById('detect'),
  clearBtn: document.getElementById('clear'),
  result: document.getElementById('result-content'),
  explanation: document.getElementById('explanation-content'),
  tokens: document.getElementById('tokens-content'),
  trace: document.getElementById('trace-content'),
  language: document.getElementById('language-content'),
  chat: document.getElementById('chat'),
  log: document.getElementById('log-content'),
};

elements.samples.addEventListener('change', () => {
  if (elements.samples.value) {
    elements.message.value = elements.samples.value;
    elements.samples.value = '';
  }
});

elements.detectBtn.addEventListener('click', detect);
elements.clearBtn.addEventListener('click', clearAll);

async function detect() {
  const text = elements.message.value.trim();
  if (!text) return;
  const requireSupported = elements.requireSupported.checked;

  elements.detectBtn.disabled = true;
  elements.detectBtn.textContent = 'Detectando...';

  try {
    const response = await api.detect(text, { requireSupported });
    if (!response.ok) {
      throw new Error(response.data?.error?.message || `HTTP ${response.status}`);
    }
    const data = response.data;
    renderResult(data);
    renderExplanation(data);
    intents.renderTokens(data.tokens, elements.tokens);
    intents.renderTrace(data.trace, elements.trace);
    intents.renderLanguage(data.language, elements.language);
    logExecution(text, data, elements.expectedIntent.value || null);
  } catch (err) {
    elements.result.replaceChildren();
    elements.result.appendChild(document.createElement('p'));
    elements.result.firstChild.style.color = 'var(--error)';
    elements.result.firstChild.textContent = `Erro: ${escapeText(err.message)}`;
  } finally {
    elements.detectBtn.disabled = false;
    elements.detectBtn.textContent = 'Detectar';
  }
}

function renderResult(data) {
  elements.result.replaceChildren();
  const primary = document.createElement('span');
  primary.className = `badge badge-${data.primary}`;
  primary.textContent = data.primary;

  const confidence = document.createElement('span');
  confidence.style.cssText = 'margin-left: 0.5rem; color: var(--fg-muted);';
  confidence.textContent = `Confiança: ${formatConfidence(data.confidence)}`;

  const multi = data.multi_intent ? document.createElement('span') : null;
  if (multi) {
    multi.style.cssText = 'margin-left: 0.5rem; color: var(--warning);';
    multi.textContent = 'MULTI-INTENT';
  }

  const container = document.createElement('div');
  container.style.cssText = 'display: flex; align-items: center; gap: 0.5rem;';
  container.appendChild(primary);
  container.appendChild(confidence);
  if (multi) container.appendChild(multi);
  elements.result.appendChild(container);

  if (data.intents && data.intents.length) {
    const list = document.createElement('div');
    list.style.cssText = 'margin-top: 0.75rem; display: flex; flex-wrap: wrap; gap: 0.5rem;';
    data.intents.forEach(i => {
      const badge = document.createElement('span');
      badge.className = `badge badge-${i.intent}`;
      badge.textContent = `${escapeText(i.label)} `;
      const scoreSpan = document.createElement('span');
      scoreSpan.style.cssText = 'font-weight:400;opacity:0.8;';
      scoreSpan.textContent = `${i.score.toFixed(1)} / ${i.min_score.toFixed(1)}`;
      badge.appendChild(scoreSpan);
      list.appendChild(badge);
    });
    elements.result.appendChild(list);
  }
}

function renderExplanation(data) {
  if (!data.explanation) {
    elements.explanation.textContent = 'Sem explicação disponível';
    return;
  }
  elements.explanation.textContent = data.explanation;
}

function logExecution(text, data, expected) {
  const correct = expected ? data.primary === expected : null;
  state.log.unshift({
    text: text.length > 80 ? text.slice(0, 80) + '…' : text,
    primary: data.primary,
    correct,
    expected,
  });
  if (state.log.length > 20) state.log.pop();
  intents.renderLog(state.log, elements.log);
}

function clearAll() {
  elements.message.value = '';
  elements.result.replaceChildren();
  elements.result.textContent = 'Clique em "Detectar" para analisar uma mensagem';
  elements.explanation.textContent = '';
  elements.tokens.replaceChildren();
  elements.trace.replaceChildren();
  elements.language.replaceChildren();
  elements.expectedIntent.value = '';
}

// Initialize chat
mountChat(elements.chat, async (message) => {
  const response = await api.chat(message, state.sessionId);
  if (!response.ok) {
    throw new Error(response.data?.error?.message || `HTTP ${response.status}`);
  }
  if (response.data.session_id) {
    state.sessionId = response.data.session_id;
  }
  return response.data;
});

// Load health on startup
api.health().then(response => {
  if (response.ok) {
    console.log('API saudável:', response.data);
  }
});

// Initial render
clearAll();