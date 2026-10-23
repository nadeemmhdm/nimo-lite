(() => {
  const originalFetch = window.fetch.bind(window);

  window.fetch = (input, init = {}) => {
    const url = typeof input === 'string' ? input : (input && input.url) || '';
    if ((url.endsWith('/api/chat') || url.endsWith('/api/chat/stream')) && typeof init.body === 'string') {
      try {
        const body = JSON.parse(init.body);
        const selector = document.getElementById('reasoningLevel');
        const memoryToggle = document.getElementById('autoMemoryToggle');
        if (selector && !body.reasoning) body.reasoning = selector.value;
        if (memoryToggle && body.auto_memory === undefined) body.auto_memory = memoryToggle.checked;
        init = {...init, body: JSON.stringify(body)};
      } catch { /* Leave non-JSON requests unchanged. */ }
    }
    return originalFetch(input, init);
  };

  const enhance = () => {
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

    if (chatMode && !document.getElementById('autoMemoryToggle')) {
      const memoryLabel = document.createElement('label');
      memoryLabel.className = 'check-label';
      memoryLabel.title = 'Automatically remember low-risk preferences and project details in local SQLite';
      const memoryToggle = document.createElement('input');
      memoryToggle.id = 'autoMemoryToggle';
      memoryToggle.type = 'checkbox';
      memoryToggle.checked = true;
      try {
        const saved = localStorage.getItem('nimo-auto-memory');
        if (saved === 'off') memoryToggle.checked = false;
      } catch { /* localStorage may be unavailable */ }
      memoryToggle.addEventListener('change', () => {
        try { localStorage.setItem('nimo-auto-memory', memoryToggle.checked ? 'on' : 'off'); } catch { /* ignore */ }
      });
      memoryLabel.append(memoryToggle, document.createTextNode(' Memory'));
      const reasoning = document.getElementById('reasoningLevel');
      (reasoning || chatMode).insertAdjacentElement('afterend', memoryLabel);
    }

    const updateHelp = () => {
      const modeHelp = document.getElementById('chatModeHelp');
      if (!modeHelp) return;
      const mode = chatMode ? chatMode.value : 'assistant';
      const messages = {
        assistant: 'Generative local chat uses cached inference, optional local memory, and controlled web sources when online mode is enabled.',
        knowledge: 'Returns matching local notes only. Model weights are not changed.',
        model: 'Uses the active local text model directly. Output quality depends on its training data and evaluation.'
      };
      modeHelp.textContent = messages[mode] || messages.assistant;
    };
    updateHelp();
    if (chatMode) chatMode.addEventListener('change', () => setTimeout(updateHelp, 0));

    const welcome = document.getElementById('chatWelcome');
    if (welcome && !document.getElementById('runtimeFeatureNote')) {
      const note = document.createElement('p');
      note.id = 'runtimeFeatureNote';
      note.className = 'muted';
      note.textContent = 'Reasoning: Fast / Balanced / Deep · Memory stays in local SQLite and filters sensitive data.';
      welcome.appendChild(note);
    }
  };

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', enhance, {once: true});
  else enhance();
})();
