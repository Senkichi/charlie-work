/* Pre-paint theme: loaded synchronously in <head> (same-origin, so CSP script-src 'self'
   allows it) so a stored light/dark choice is on <html> before first paint. */
(function () {
  "use strict";
  try {
    var v = localStorage.getItem("cw-dash-theme");
    if (v === "light" || v === "dark") {
      document.documentElement.setAttribute("data-theme", v);
    }
  } catch (err) { /* storage blocked: follow the system theme */ }
})();
