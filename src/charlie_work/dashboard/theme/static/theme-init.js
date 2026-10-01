/* Pre-paint theme: loaded synchronously in <head> (same-origin, so CSP script-src 'self'
   allows it) so a stored light/dark choice is on <html> before first paint.
   A ?theme=light|dark query parameter overrides the stored choice for this page view
   only (screenshots, sharing a view); it is never persisted. */
(function () {
  "use strict";
  var root = document.documentElement;
  root.classList.add("js"); /* lets CSS hide JS-revealable rows without a flash */
  var v = null;
  try {
    var q = new URLSearchParams(window.location.search).get("theme");
    if (q === "light" || q === "dark") { v = q; root.setAttribute("data-theme-src", "query"); }
  } catch (err) { /* no URLSearchParams: ignore the override */ }
  if (v === null) {
    try { v = localStorage.getItem("cw-dash-theme"); } catch (err) { /* storage blocked */ }
  }
  if (v === "light" || v === "dark") {
    root.setAttribute("data-theme", v);
  }
})();
