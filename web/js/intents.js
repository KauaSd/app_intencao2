// intents.js — DOM rendering for intent detection results

import { escapeText, badgeClass, formatConfidence, formatScore, explainUnsupported } from './format.js';

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

export function renderIntents(data, container) {
  container.replaceChildren();
  if (!data || !data.intents) return;
  const list = document.createElement('div');
  list.style.display = 'flex';
  list.style.flexWrap = 'wrap';
  list.style.gap = '0.5rem';
  data.intents.forEach(i => {
    const badge = el('span', { class: `badge ${badgeClass(i.id)}` }, escapeText(i.label || i.id));
    list.appendChild(badge);
  });
  container.appendChild(list);
}

export function renderTokens(tokens, container) {
  container.replaceChildren();
  if (!tokens || !tokens.length) {
    container.appendChild(el('p', { class: 'empty-state' }, 'Nenhum token'));
    return;
  }
  const table = el('table', { class: 'token-table' });
  const thead = el('thead', {},
    el('tr', {},
      el('th', { text: 'Texto' }),
      el('th', { text: 'Lema' }),
      el('th', { text: 'POS' }),
      el('th', { text: 'Início sentença' }),
      el('th', { text: 'Match' }),
      el('th', { text: 'Negado' })
    )
  );
  const tbody = el('tbody');
  tokens.forEach(t => {
    const tr = el('tr', { class: `${t.matched ? 'matched' : ''} ${t.negated ? 'negated' : ''}` });
    tr.appendChild(el('td', { text: escapeText(t.text) }));
    tr.appendChild(el('td', { text: escapeText(t.lemma) }));
    tr.appendChild(el('td', { text: escapeText(t.pos) }));
    tr.appendChild(el('td', { text: t.is_sentence_start ? 'Sim' : 'Não' }));
    tr.appendChild(el('td', { text: t.matched ? 'Sim' : 'Não' }));
    tr.appendChild(el('td', { text: t.negated ? 'Sim' : 'Não' }));
    tbody.appendChild(tr);
  });
  table.appendChild(thead);
  table.appendChild(tbody);
  container.appendChild(table);
}

export function renderTrace(trace, container) {
  container.replaceChildren();
  if (!trace || !trace.length) {
    container.appendChild(el('p', { class: 'empty-state' }, 'Nenhum trace de regras'));
    return;
  }
  trace.forEach(hit => {
    const div = el('div', { class: `rule ${hit.negated ? 'negated' : ''} ${hit.dropped ? 'dropped' : ''}` });
    div.appendChild(el('div', { class: 'rule-header' },
      el('span', { class: 'rule-id', text: escapeText(hit.rule_id) }),
      el('span', { class: 'rule-weight', text: `peso: ${hit.weight}` })
    ));
    const details = el('div', { class: 'rule-details' });
    details.appendChild(el('span', { text: `tipo: ${escapeText(hit.type)}` }));
    details.appendChild(el('span', { text: `valor: ${escapeText(hit.value)}` }));
    details.appendChild(el('span', { text: `posição: ${hit.char_start}–${hit.char_end}` }));
    details.appendChild(el('span', { text: `texto: "${escapeText(hit.matched_text)}"` }));
    details.appendChild(el('span', { text: hit.negated ? 'NEGADO' : '' }));
    details.appendChild(el('span', { text: hit.offsets_exact ? 'offsets exatos' : 'offsets aproximados' }));
    div.appendChild(details);
    container.appendChild(div);
  });
}

export function renderLanguage(language, container) {
  container.replaceChildren();
  if (!language) {
    container.appendChild(el('p', { class: 'empty-state' }, 'Sem informação de idioma'));
    return;
  }
  const grid = el('div', { class: 'language-info' });
  const items = [
    { label: 'Idioma', value: language.language || '—' },
    { label: 'Suportado', value: language.supported ? 'Sim' : 'Não' },
    { label: 'Motivo', value: language.reason || '—' },
    { label: 'Roteiro', value: language.script || '—' },
    { label: 'Offsets preservados', value: language.offset_preserved ? 'Sim' : 'Não' },
  ];
  if (language.min_share !== null && language.min_share !== undefined) {
    items.push({ label: 'Min share', value: language.min_share.toFixed(3) });
  }
  if (language.margin !== null && language.margin !== undefined) {
    items.push({ label: 'Margem', value: language.margin.toFixed(3) });
  }
  items.forEach(item => {
    grid.appendChild(el('div', { class: 'item' },
      el('div', { class: 'label', text: item.label }),
      el('div', { class: 'value', text: item.value })
    ));
  });
  container.appendChild(grid);

  if (language.candidates && language.candidates.length) {
    const candDiv = el('div');
    candDiv.appendChild(el('h3', { text: 'Candidatos' }));
    language.candidates.forEach((c, idx) => {
      const cand = el('div', { class: `candidate ${idx === 0 ? 'top' : ''}` });
      cand.appendChild(el('strong', { text: `${escapeText(c.language)}: ${(c.share * 100).toFixed(1)}%` }));
      if (c.markers && c.markers.length) {
        cand.appendChild(el('span', { text: ` [${c.markers.map(escapeText).join(', ')}]` }));
      }
      candDiv.appendChild(cand);
    });
    container.appendChild(candDiv);
  }
}

export function renderResult(result, container) {
  container.replaceChildren();
  if (!result) {
    container.appendChild(el('p', { class: 'empty-state' }, 'Nenhum resultado'));
    return;
  }
  const primary = el('span', { class: `badge ${badgeClass(result.primary)}` }, escapeText(result.primary));
  const confidence = el('span', { style: 'margin-left: 0.5rem; color: var(--fg-muted);' },
    `Confiança: ${formatConfidence(result.confidence)}`);
  const multi = result.multi_intent ? el('span', { style: 'margin-left: 0.5rem; color: var(--warning);' }, 'MULTI-INTENT') : null;
  container.appendChild(el('div', { style: 'display: flex; align-items: center; gap: 0.5rem;' }, primary, confidence, multi));

  if (result.intents && result.intents.length) {
    const list = el('div', { style: 'margin-top: 0.75rem; display: flex; flex-wrap: wrap; gap: 0.5rem;' });
    result.intents.forEach(i => {
      const badge = el('span', { class: `badge ${badgeClass(i.intent)}` },
        `${escapeText(i.label)} `,
        el('span', { style: 'font-weight: 400; opacity: 0.8;' }, formatScore(i))
      );
      list.appendChild(badge);
    });
    container.appendChild(list);
  }
}

export function renderLog(entries, container) {
  container.replaceChildren();
  if (!entries || !entries.length) {
    container.appendChild(el('p', { class: 'empty-state' }, 'Nenhuma execução registrada'));
    return;
  }
  entries.slice().reverse().forEach(e => {
    const div = el('div', { class: `log-entry ${e.correct ? 'correct' : e.incorrect ? 'incorrect' : ''}` });
    div.appendChild(el('span', { class: 'text', text: escapeText(e.text) }));
    div.appendChild(el('span', { class: `badge ${badgeClass(e.primary)}` }, escapeText(e.primary)));
    container.appendChild(div);
  });
  if (entries.length) {
    const correct = entries.filter(e => e.correct).length;
    const total = entries.length;
    const acc = total ? (correct / total * 100).toFixed(1) : 0;
    container.appendChild(el('div', { class: 'accuracy' }, `Acurácia: ${acc}% (${correct}/${total})`));
  }
}

export function accuracy(entries) {
  if (!entries || !entries.length) return '—';
  const correct = entries.filter(e => e.correct).length;
  return `${(correct / entries.length * 100).toFixed(1)}%`;
}