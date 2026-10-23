// Apply before first paint. Theme preference never leaves this browser.
(() => {
  let choice;
  try { choice = localStorage.getItem('nimo-theme'); } catch { /* restricted browser storage */ }
  const theme = ['dark','light'].includes(choice)
    ? choice
    : (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
  document.documentElement.dataset.theme = theme;

  // Load local UI enhancements in a deterministic order. No external scripts/CDNs.
  const enhancements = document.createElement('script');
  enhancements.src = '/static/runtime-enhancements.js?v=20261023-domfix1';
  enhancements.defer = true;
  enhancements.async = false;
  enhancements.onload = () => {
    const fixes = document.createElement('script');
    fixes.src = '/static/runtime-fixes.js?v=20261023-domfix1';
    fixes.defer = true;
    fixes.async = false;
    document.head.appendChild(fixes);
  };
  document.head.appendChild(enhancements);
})();
