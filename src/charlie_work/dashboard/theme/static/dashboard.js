/* Dashboard behaviour (CSP script-src 'self': this file is the only script besides htmx).
   - copy buttons: button[data-copy] puts its exact command on the clipboard
   - <time class="js-local" datetime=ISO>: re-rendered in the viewer's local zone
   - keys: j/k move through Needs-me rows, Enter opens, c copies, g n / g h navigate
   Everything is event-delegated on document so htmx swaps of #now need no re-binding. */
(function () {
  "use strict";

  function pad(n) { return (n < 10 ? "0" : "") + n; }

  function localise(root) {
    var nodes = (root || document).querySelectorAll("time.js-local[datetime]");
    for (var i = 0; i < nodes.length; i++) {
      var d = new Date(nodes[i].getAttribute("datetime"));
      if (!isNaN(d.getTime())) {
        nodes[i].textContent = pad(d.getHours()) + ":" + pad(d.getMinutes()) + ":" + pad(d.getSeconds());
      }
    }
  }

  function copy(button) {
    var text = button.getAttribute("data-copy") || "";
    var done = function () {
      button.textContent = "copied";
      setTimeout(function () { button.textContent = "copy"; }, 1200);
    };
    try {
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(done, function () {});
      }
    } catch (err) { /* clipboard unavailable: the command stays visible in the title */ }
  }

  document.addEventListener("click", function (e) {
    var b = e.target && e.target.closest ? e.target.closest("button[data-copy]") : null;
    if (b) { copy(b); }
  });

  var pendingG = false;
  function rows() { return Array.prototype.slice.call(document.querySelectorAll("#needs-list .row")); }
  function current(list) {
    var a = document.activeElement;
    for (var i = 0; i < list.length; i++) { if (list[i].contains(a)) { return i; } }
    return -1;
  }
  function focusRow(row) {
    var target = row.querySelector(".why a");
    if (target) { target.focus(); }
  }

  document.addEventListener("keydown", function (e) {
    var t = e.target;
    if (e.ctrlKey || e.metaKey || e.altKey) { return; }
    if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.isContentEditable)) { return; }
    if (pendingG) {
      pendingG = false;
      if (e.key === "n") { location.href = "/now"; }
      else if (e.key === "h") { location.href = "/history"; }
      return;
    }
    var list = rows();
    var i = current(list);
    if (e.key === "g") { pendingG = true; return; }
    if (!list.length) { return; }
    if (e.key === "j") { focusRow(list[Math.min(list.length - 1, i + 1)]); e.preventDefault(); }
    else if (e.key === "k") { focusRow(list[Math.max(0, i - 1)]); e.preventDefault(); }
    else if (e.key === "c" && i >= 0) {
      var b = list[i].querySelector("button[data-copy]");
      if (b) { copy(b); }
    }
  });

  document.addEventListener("DOMContentLoaded", function () { localise(document); });
  document.body && localise(document);
  document.addEventListener("htmx:afterSwap", function (e) { localise(e.target); });
  document.addEventListener("htmx:afterSettle", function () { localise(document); });
})();
