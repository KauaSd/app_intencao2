// chat.js — Chatbot UI component

import { escapeText, badgeClass } from './format.js';

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  Object.entries(attrs).forEach(([k, v]) => {
    if (k === 'class') node.className = v;
    else if (k === 'text') node.textContent = v;
    else node.setAttribute(k, v);
  });
  children.flat().forEach(c => {
    if (c instanceof Node) node.appendChild(c);
    else if (c !== null && c !== undefined) node.appendChild(document.createTextNode(String(c)));
  });
  return node;
}

export function mountChat(root, sendFn) {
  root.replaceChildren();
  const messages = el('div', { id: 'chat-messages', class: 'chat-messages' });
  const inputRow = el('div', { class: 'chat-input-row' });
  const input = el('input', { id: 'chat-input', type: 'text', placeholder: 'Digite uma mensagem para o chatbot...' });
  const sendBtn = el('button', { id: 'chat-send', text: 'Enviar' });
  inputRow.appendChild(input);
  inputRow.appendChild(sendBtn);
  root.appendChild(messages);
  root.appendChild(inputRow);

  function send() {
    const text = input.value.trim();
    if (!text) return;
    input.value = '';
    appendMessage('user', text);
    sendFn(text).then(response => {
      appendMessage('bot', response.reply, response);
    }).catch(err => {
      appendMessage('bot', `Erro: ${err.message}`);
    });
  }

  sendBtn.addEventListener('click', send);
  input.addEventListener('keydown', e => {
    if (e.key === 'Enter') send();
  });

  function appendMessage(role, text, meta = null) {
    const msg = el('div', { class: `chat-message ${role}` });
    msg.appendChild(el('div', { text: escapeText(text) }));
    if (meta && role === 'bot') {
      const metaDiv = el('div', { class: 'meta' });
      const parts = [];
      parts.push(`intent: ${meta.primary}`);
      if (meta.multi_intent) parts.push('multi_intent');
      if (meta.strategy) parts.push(`strategy: ${meta.strategy}`);
      if (meta.pending_step) parts.push(`passo: ${meta.pending_step}/3`);
      metaDiv.textContent = parts.join(' | ');
      msg.appendChild(metaDiv);
    }
    messages.appendChild(msg);
    messages.scrollTop = messages.scrollHeight;
  }

  return { appendMessage };
}