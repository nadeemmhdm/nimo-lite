// Apply before the first paint. Theme preference never leaves this browser.
(() => {
  let choice;
  try { choice = localStorage.getItem('nimo-theme'); } catch { /* restricted browser storage */ }
  const theme = ['dark','light'].includes(choice) ? choice : (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
  document.documentElement.dataset.theme = theme;
})();
