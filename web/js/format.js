// format.js - Pure formatting helpers, no external deps, no side effects.
// This file is evaluated by Python tests via exec(), so it must remain free of external deps.

export function formatConfidence(n) {
  if (n === undefined || n === null) return '--';
  const pct = Math.round(n * 100);
  return `${pct}%`;
}

export function formatScore(match) {
  if (!match) return '--';
  const score = match.score ?? 0;
  const minScore = match.min_score ?? 0;
  return `${score.toFixed(1)} / ${minScore.toFixed(1)}`;
}

export function explainUnsupported(languageInfo) {
  if (!languageInfo) return 'Idioma nao identificado';
  switch (languageInfo.reason) {
    case 'unsupported_script':
      return `Idioma nao suportado (roteiro: ${languageInfo.script || 'desconhecido'})`;
    case 'empty_text':
      return 'Texto sem caracteres alfabeticos';
    case 'insufficient_signal':
      return 'Sinal insuficiente para identificar o idioma';
    case 'ambiguous_language':
      return 'Idioma ambiguo';
    default:
      return `Nao suportado: ${languageInfo.reason}`;
  }
}

export function badgeClass(intent) {
  if (!intent) return 'badge-desconhecido';
  return `badge-${intent}`;
}

export function escapeText(s) {
  if (s === undefined || s === null) return '';
  return String(s)
    .replace(/&/g, '&')
    .replace(/</g, '<')
    .replace(/>/g, '>')
    .replace(/"/g, '"')
    .replace(/'/g, '&#039;');
}