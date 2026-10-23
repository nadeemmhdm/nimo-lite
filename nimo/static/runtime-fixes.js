(() => {
  'use strict';

  let regenerateNext = false;
  const originalFetch = window.fetch.bind(window);

  window.fetch = (input, init = {}) => {
    const url = typeof input === 'string' ? input : (input && input.url) || '';
    if (regenerateNext && url.endsWith('/api/chat/stream') && typeof init.body === 'string') {
      try {
        const body = JSON.parse(init.body);
        body.regenerate = true;
        init = {...init, body: JSON.stringify(body)};
      } finally {
        regenerateNext = false;
      }
    } else if (url.endsWith('/api/chat/stream')) {
      regenerateNext = false;
    }
    return originalFetch(input, init);
  };

  document.addEventListener('click', event => {
    const button = event.target.closest?.('button[aria-label="Regenerate"]');
    if (button) regenerateNext = true;
  }, true);

  function icon(path) {
    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('viewBox', '0 0 24 24');
    svg.setAttribute('fill', 'none');
    svg.setAttribute('stroke', 'currentColor');
    svg.setAttribute('stroke-width', '1.8');
    svg.setAttribute('stroke-linecap', 'round');
    svg.setAttribute('stroke-linejoin', 'round');
    svg.setAttribute('aria-hidden', 'true');
    svg.classList.add('ui-svg');
    const p = document.createElementNS(svg.namespaceURI, 'path');
    p.setAttribute('d', path);
    svg.append(p);
    return svg;
  }

  async function bootstrapToken() {
    const response = await originalFetch('/api/bootstrap');
    if (!response.ok) throw new Error('Could not initialize local session');
    return (await response.json()).token;
  }

  async function call(path, body, method = body === undefined ? 'GET' : 'POST') {
    const token = await bootstrapToken();
    const options = {method, headers: {'X-Nimo-Token': token}};
    if (body !== undefined) {
      options.headers['Content-Type'] = 'application/json';
      options.body = JSON.stringify(body);
    }
    const response = await originalFetch('/api/' + path, options);
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.detail || 'Request failed');
    return data;
  }

  function showNotice(text, error = false) {
    const notice = document.getElementById('notice');
    if (!notice) return;
    notice.textContent = text;
    notice.hidden = false;
    notice.classList.toggle('error', error);
  }

  function addUpdateControls() {
    const security = document.getElementById('security');
    if (!security || document.getElementById('nimoUpdateCard')) return;
    const card = document.createElement('div');
    card.className = 'card';
    card.id = 'nimoUpdateCard';
    const title = document.createElement('h2');
    title.textContent = 'Nimo updates';
    const note = document.createElement('p');
    note.className = 'muted';
    note.textContent = 'Update checks contact only the official GitHub repository and run only when you request them here or enable auto-update at launch.';
    const row = document.createElement('div');
    row.className = 'row';
    const check = document.createElement('button');
    check.type = 'button';
    check.className = 'button-secondary';
    check.append(icon('M4 12a8 8 0 0 1 14-5l2 2M20 4v5h-5M20 12a8 8 0 0 1-14 5l-2-2M4 20v-5h5'));
    check.append(document.createTextNode('Check for update'));
    const install = document.createElement('button');
    install.type = 'button';
    install.hidden = true;
    install.append(icon('M12 3v12m-5-5 5 5 5-5M5 20h14'));
    install.append(document.createTextNode('Install update'));
    const state = document.createElement('span');
    state.className = 'muted';
    check.onclick = async () => {
      check.disabled = true;
      state.textContent = 'Checking official releases...';
      try {
        const info = await call('update');
        state.textContent = info.available ? `Update available: ${info.latest}` : `Current installation is up to date (${info.current}).`;
        install.hidden = !info.available;
      } catch (error) {
        state.textContent = 'Update check failed.';
        showNotice(error.message, true);
      } finally {
        check.disabled = false;
      }
    };
    install.onclick = async () => {
      if (!confirm('Install the newer official Nimo Lite release? Restart Nimo after installation.')) return;
      install.disabled = true;
      state.textContent = 'Installing verified release source...';
      try {
        const result = await call('update', {confirm: true});
        state.textContent = result.message || 'Update installed. Restart Nimo.';
      } catch (error) {
        state.textContent = 'Update installation failed.';
        showNotice(error.message, true);
      } finally {
        install.disabled = false;
      }
    };
    row.append(check, install, state);
    card.append(title, note, row);
    security.append(card);
  }

  function enhance() {
    addUpdateControls();
    // Accessible state labels are text, not decorative emoji. Generated assistant
    // content is intentionally left unchanged because emoji may be semantically relevant.
    document.querySelectorAll('[aria-label]').forEach(node => {
      if (node.getAttribute('aria-label') === 'Send message') node.title = 'Send message';
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', enhance, {once: true});
  } else {
    enhance();
  }
})();
