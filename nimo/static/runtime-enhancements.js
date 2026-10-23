(() => {
  const originalFetch = window.fetch.bind(window);

  window.fetch = (input, init = {}) => {
    const url = typeof input === 'string' ? input : (input && input.url) || '';
    if ((url.endsWith('/api/chat') || url.endsWith('/api/chat/stream')) && typeof init.body === 'string') {
      try {
        const body = JSON.parse(init.body);
        const selector = document.getElementById('reasoningLevel');
        if (selector && !body.reasoning) body.reasoning = selector.value;
        init = {...init, body: JSON.stringify(body)};
      } catch { /* Leave non-JSON requests unchanged. */ }
    }
    return originalFetch(input, init);
  };

  document.addEventListener('DOMContentLoaded', () => {
    const chatMode = document.getElementById('chatMode');
    if (chatMode && !document.getElementById('reasoningLevel')) {
      const label = document.createElement('label');
      label.className = 'sr-only';
      label.htmlFor = 'reasoningLevel';
      label.textContent = 'Reasoning level';
      const select = document.createElement('select');
      select.id = 'reasoningLevel';
      select.setAttribute('aria-label', 'Reasoning level');
      select.innerHTML = '<option value="fast">⚡ Fast</option><option value="balanced" selected>◐ Balanced</option><option value="deep">✦ Deep</option>';
      try {
        const saved = localStorage.getItem('nimo-reasoning');
        if (['fast','balanced','deep'].includes(saved)) select.value = saved;
      } catch { /* localStorage may be unavailable */ }
      select.addEventListener('change', () => {
        try { localStorage.setItem('nimo-reasoning', select.value); } catch { /* ignore */ }
      });
      chatMode.insertAdjacentElement('afterend', label);
      label.insertAdjacentElement('afterend', select);
    }

    const modeHelp = document.getElementById('chatModeHelp');
    if (modeHelp) {
      modeHelp.textContent = 'Generative local chat uses cached inference, optional local memory, and controlled web sources when online mode is enabled.';
    }

    const welcome = document.getElementById('chatWelcome');
    if (welcome) {
      const note = document.createElement('p');
      note.className = 'muted';
      note.textContent = 'Reasoning: Fast / Balanced / Deep · Memory stays in local SQLite and filters sensitive data.';
      welcome.appendChild(note);
    }
  });
})();
