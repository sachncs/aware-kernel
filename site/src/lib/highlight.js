export function highlight(code) {
  const escape = (s) =>
    s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

  const escaped = escape(code);

  return escaped
    .replace(/(#[^\n]*)/g, '<span class="hl-c">$1</span>')
    .replace(/("[^"\\]*(?:\\.[^"\\]*)*")/g, '<span class="hl-s">$1</span>')
    .replace(
      /\b(import|from|as|def|return|class|with|if|else|elif|while|for|in|None|True|False|and|or|not)\b/g,
      '<span class="hl-k">$1</span>'
    )
    .replace(/\b(\d+\.?\d*(?:e-?\d+)?)\b/g, '<span class="hl-n">$1</span>')
    .replace(
      /\b(Kernos|Buffer|GridSearchCV)\b/g,
      '<span class="hl-t">$1</span>'
    )
    .replace(
      /\b(X|y|model|search|param_grid|stream|x_batch|y_batch|X_train|y_train|X_test|seed|steps|dim|mbasis|abasis|lk|ridge|mode|cool|warm|gain|drift_hi|score|fit|predict|partial_fit|standard_normal|default_rng|print|best_params_|best_score_|rng)\b(?=\s*[=,.()\]])/g,
      '<span class="hl-v">$1</span>'
    );
}