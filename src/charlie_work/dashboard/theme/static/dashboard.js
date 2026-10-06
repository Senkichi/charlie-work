/* Dashboard behaviour (CSP script-src 'self': same-origin files only, no inline script).
   Vanilla, no build. Loaded deferred; theme-init.js handles the pre-paint theme part.
   - copy buttons (clipboard API, textarea fallback, polite live announcement)
   - theme toggle: system -> light -> dark, stored under 'cw-dash-theme'
   - <time class="js-local">: re-rendered as local HH:MM:SS
   - keys: g <view> (via the nav's data-go links), '/' filter, j/k select, Enter open,
     c copy, '?' help
   - staleness: no successful swap for 2x the poll interval => "Not updating" banner in
     #client-status + html.is-stale (muted numbers); cleared by the next success
   - server banners ([data-alert]) are announced once, when their set changes
   Key and click handling is delegated on document, so an htmx swap of #now needs no
   re-binding; the afterSwap hook only restores selection (by row id), filter and labels. */
(function () {
  "use strict";

  var THEME_KEY = "cw-dash-theme";
  var THEMES = ["system", "light", "dark"];
  var reduceMotion = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  var selectedId = null;
  var filterText = "";
  var expanded = {}; /* group slug -> true once "+N more" was opened (survives swaps) */
  var pendingG = false;
  var live = null;
  var lastOk = Date.now();
  var lastFail = null;
  var alertKeys = "";

  function pad(n) { return (n < 10 ? "0" : "") + n; }

  /* ---- local time ---- */
  function localise(root) {
    var nodes = (root || document).querySelectorAll("time.js-local[datetime]");
    for (var i = 0; i < nodes.length; i++) {
      var d = new Date(nodes[i].getAttribute("datetime"));
      if (!isNaN(d.getTime())) {
        nodes[i].textContent = pad(d.getHours()) + ":" + pad(d.getMinutes()) + ":" + pad(d.getSeconds());
      }
    }
  }

  /* ---- announcements ---- */
  function announce(msg) {
    if (!live) {
      live = document.createElement("span");
      live.className = "sr";
      live.setAttribute("role", "status");
      live.setAttribute("aria-live", "polite");
      document.body.appendChild(live);
    }
    live.textContent = "";
    setTimeout(function () { live.textContent = msg; }, 30);
  }

  /* ---- copy ---- */
  function fallbackCopy(text) {
    var ta = document.createElement("textarea");
    ta.value = text;
    ta.setAttribute("readonly", "");
    ta.className = "sr";
    document.body.appendChild(ta);
    ta.select();
    var ok = false;
    try { ok = document.execCommand("copy"); } catch (err) { ok = false; }
    document.body.removeChild(ta);
    return ok;
  }

  function copy(button) {
    var text = button.getAttribute("data-copy") || "";
    var original = button.getAttribute("data-label") || button.textContent;
    button.setAttribute("data-label", original);
    var done = function (ok) {
      button.textContent = ok ? "copied" : "failed";
      announce(ok ? "Command copied" : "Copy failed");
      setTimeout(function () { button.textContent = original; }, 1200);
    };
    var viaFallback = function () { done(fallbackCopy(text)); };
    try {
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(function () { done(true); }, viaFallback);
        return;
      }
    } catch (err) { /* fall through to the textarea path */ }
    viaFallback();
  }

  /* ---- theme ---- */
  function storedTheme() {
    var root = document.documentElement;
    if (root.getAttribute("data-theme-src") === "query") {
      var q = root.getAttribute("data-theme");
      if (q === "light" || q === "dark") { return q; }
    }
    try {
      var v = localStorage.getItem(THEME_KEY);
      return v === "light" || v === "dark" ? v : "system";
    } catch (err) { return "system"; }
  }

  function syncThemeButton(name) {
    var b = document.getElementById("theme-toggle");
    if (b) {
      b.textContent = "theme: " + name;
      b.setAttribute("aria-label", "Theme: " + name + ". Activate to change.");
    }
  }

  function applyTheme(name) {
    document.documentElement.setAttribute("data-theme", name === "system" ? "auto" : name);
    syncThemeButton(name);
  }

  function cycleTheme() {
    document.documentElement.removeAttribute("data-theme-src");
    var next = THEMES[(THEMES.indexOf(storedTheme()) + 1) % THEMES.length];
    try {
      if (next === "system") { localStorage.removeItem(THEME_KEY); }
      else { localStorage.setItem(THEME_KEY, next); }
    } catch (err) { /* choice applies for this page view only */ }
    applyTheme(next);
    announce("Theme " + next);
  }

  /* ---- needs-me rows: selection and filter ---- */
  function allRows() { return Array.prototype.slice.call(document.querySelectorAll("#needs-list .row")); }
  function visibleRows() {
    return allRows().filter(function (r) { return !r.hidden && r.getClientRects().length > 0; });
  }

  function applyExpanded() {
    var items = document.querySelectorAll("#needs-list > [data-grp]");
    for (var i = 0; i < items.length; i++) {
      var li = items[i], open = !!expanded[li.getAttribute("data-grp")];
      if (li.classList.contains("row")) { li.classList.toggle("is-open", open); }
      var b = li.querySelector(".more-btn");
      if (b) {
        b.setAttribute("aria-expanded", open ? "true" : "false");
        b.textContent = open ? "show fewer" : "+" + b.getAttribute("data-more") + " more";
      }
    }
  }

  function paintSelection() {
    var rows = allRows();
    for (var i = 0; i < rows.length; i++) {
      var on = rows[i].id === selectedId;
      rows[i].classList.toggle("is-sel", on);
      if (on) { rows[i].setAttribute("aria-current", "true"); }
      else { rows[i].removeAttribute("aria-current"); }
    }
  }

  function select(row) {
    selectedId = row ? row.id : null;
    paintSelection();
    if (row && row.scrollIntoView) {
      row.scrollIntoView({ block: "nearest", behavior: reduceMotion ? "auto" : "smooth" });
    }
  }

  function applyFilter() {
    var q = filterText.trim().toLowerCase();
    var list = document.getElementById("needs-list");
    if (list) { list.classList.toggle("is-filtering", !!q); }
    var items = document.querySelectorAll("#needs-list > li");
    var groupHead = null, groupHasRow = false;
    var closeGroup = function () { if (groupHead) { groupHead.hidden = !!q && !groupHasRow; } };
    for (var i = 0; i < items.length; i++) {
      var li = items[i];
      if (li.classList.contains("grp")) {
        closeGroup();
        groupHead = li; groupHasRow = false;
      } else if (li.classList.contains("row")) {
        var repo = (li.querySelector(".repo") || {}).title || "";
        var hit = !q || li.textContent.toLowerCase().indexOf(q) !== -1 ||
          repo.toLowerCase().indexOf(q) !== -1;
        li.hidden = !hit;
        if (hit) { groupHasRow = true; }
      }
    }
    closeGroup();
    var sel = selectedId && document.getElementById(selectedId);
    if (sel && (sel.hidden || !sel.getClientRects().length)) { select(null); }
  }

  function move(delta) {
    var rows = visibleRows();
    if (!rows.length) { return; }
    var i = -1;
    for (var k = 0; k < rows.length; k++) { if (rows[k].id === selectedId) { i = k; } }
    var next = i === -1 ? (delta > 0 ? 0 : rows.length - 1)
                        : Math.max(0, Math.min(rows.length - 1, i + delta));
    select(rows[next]);
  }

  function selectedRow() { return selectedId ? document.getElementById(selectedId) : null; }

  function toggleHelp(force) {
    var p = document.getElementById("keyhelp");
    if (!p) { return; }
    p.hidden = typeof force === "boolean" ? !force : !p.hidden;
  }

  /* ---- client-side staleness (rule in staleness.js) ---- */
  function pollSeconds() {
    var n = document.getElementById("now");
    var v = n ? parseInt(n.getAttribute("data-poll"), 10) : NaN;
    return v > 0 ? v : 20;
  }

  function clock(ms) {
    var d = new Date(ms);
    return pad(d.getHours()) + ":" + pad(d.getMinutes()) + ":" + pad(d.getSeconds());
  }

  function checkStale() {
    var rule = window.CwStale;
    if (!rule) { return; }
    var st = rule.staleState(Date.now(), lastOk, pollSeconds(), lastFail);
    document.documentElement.classList.toggle("is-stale", !!st);
    var region = document.getElementById("client-status");
    if (!region) { return; }
    var p = region.querySelector("p");
    if (!st) {
      if (p) { region.removeChild(p); }
      return;
    }
    var msg = rule.message(st, clock(st.since));
    if (!p) {
      p = document.createElement("p");
      p.className = "banner offline text-warn";
      region.appendChild(p);
    }
    if (p.textContent !== msg) { p.textContent = msg; } /* unchanged text: not re-announced */
  }

  function pollOk() { lastOk = Date.now(); checkStale(); }
  function pollFailed() { lastFail = Date.now(); checkStale(); }

  /* Server banners sit inside the swapped #now, so a role=alert there would be re-read on
     every poll; instead each banner is spoken once, when the set of banners changes. */
  function announceAlerts() {
    var nodes = document.querySelectorAll("#now [data-alert]");
    var keys = [], fresh = [];
    for (var i = 0; i < nodes.length; i++) {
      var k = nodes[i].getAttribute("data-alert");
      keys.push(k);
      if (alertKeys.split(" ").indexOf(k) === -1) { fresh.push(nodes[i].textContent); }
    }
    alertKeys = keys.join(" ");
    if (fresh.length) { announce(fresh.join(" ")); }
  }

  /* ---- events ---- */
  document.addEventListener("click", function (e) {
    var t = e.target && e.target.closest ? e.target : null;
    if (!t) { return; }
    var b = t.closest("button[data-copy]");
    if (b) { copy(b); return; }
    if (t.closest("#theme-toggle")) { cycleTheme(); return; }
    var more = t.closest(".more-btn");
    if (more) {
      var g = more.getAttribute("data-grp");
      expanded[g] = !expanded[g];
      applyExpanded();
      applyFilter();
      return;
    }
    var row = t.closest("#needs-list .row");
    if (row && !t.closest("a, button")) { select(row); }
  });

  document.addEventListener("input", function (e) {
    if (e.target && e.target.id === "needs-filter") {
      filterText = e.target.value;
      applyFilter();
    }
  });

  document.addEventListener("keydown", function (e) {
    var t = e.target;
    var typing = t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" ||
      t.tagName === "SELECT" || t.isContentEditable);
    if (typing) {
      if (e.key === "Escape" && t.id === "needs-filter") {
        t.value = ""; filterText = ""; applyFilter(); t.blur();
      }
      return;
    }
    if (e.ctrlKey || e.metaKey || e.altKey) { return; }
    if (pendingG) {
      pendingG = false;
      var view = /^[a-z]$/.test(e.key) && document.querySelector('nav.views a[data-go="' + e.key + '"]');
      if (view) { location.href = view.href; e.preventDefault(); }
      return;
    }
    switch (e.key) {
      case "g": pendingG = true; setTimeout(function () { pendingG = false; }, 1500); break;
      case "/":
        var f = document.getElementById("needs-filter");
        if (f) { f.focus(); f.select(); e.preventDefault(); }
        break;
      case "j": move(1); e.preventDefault(); break;
      case "k": move(-1); e.preventDefault(); break;
      case "Enter":
        var r = selectedRow();
        var a = r && r.querySelector(".why a");
        if (a && !(t && t.closest && t.closest("a, button"))) { location.href = a.href; e.preventDefault(); }
        break;
      case "c":
        var sr = selectedRow();
        var cb = sr && sr.querySelector("button[data-copy]");
        if (cb) { copy(cb); }
        break;
      case "?": toggleHelp(); e.preventDefault(); break;
      case "Escape": toggleHelp(false); break;
      default: break;
    }
  });

  /* ---- (re)apply state: first load and after every htmx swap ---- */
  function restore() {
    var f = document.getElementById("needs-filter");
    if (f && f.value !== filterText) { f.value = filterText; }
    applyExpanded();
    applyFilter();
    paintSelection();
    syncThemeButton(storedTheme());
    localise(document);
  }

  function start() { restore(); announceAlerts(); setInterval(checkStale, 1000); }

  document.addEventListener("htmx:afterSwap", function () { pollOk(); restore(); announceAlerts(); });
  document.addEventListener("htmx:afterSettle", restore);
  document.addEventListener("htmx:responseError", pollFailed);
  document.addEventListener("htmx:sendError", pollFailed);
  document.addEventListener("htmx:timeout", pollFailed);
  if (document.readyState === "loading") { document.addEventListener("DOMContentLoaded", start); }
  else { start(); }
})();
