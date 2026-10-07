/* History page behaviour (CSP script-src 'self': same-origin file, no inline script, no
   HTML-string injection). The server renders one range with every tab's cards, every
   metric's headline and window drawers, and all chart data in #hchart[data-hist]; this
   file only decides what is showing and keeps that choice in the URL hash so it survives
   the htmx refresh of #hist and a reload:

     #t=<tab>&m=<metric>&r=<range>&pick=<bucket ISO>

   Defaults (the first tab, its first metric, 7d, nothing picked) are left out. The range
   is also mirrored into ?range= so a reload paints the right range server-side; picking
   another range fetches that range's #hist fragment. Every handler is delegated on
   document, so a swap needs no re-binding; after each swap (and settle: htmx re-applies
   server attributes then) apply() re-asserts the state. State is driven through data-* /
   aria-* / hidden, never `class`, which htmx settles back to the server's value.
   Keys: left/right pick a bucket while the chart is focused, Escape clears. */
(function () {
  "use strict";

  var NS = "http://www.w3.org/2000/svg";
  var SVG_TAGS = { svg: 1, rect: 1 };
  var DEFAULTS = { t: "flow", m: null, r: "7d", pick: null };
  var ui = { t: "flow", m: null, r: "7d", pick: null };
  var cacheKey = null, cacheData = null;
  var observer = null, observed = null, frame = 0;
  var fetching = null;

  function $(id) { return document.getElementById(id); }
  function all(sel, root) { return Array.prototype.slice.call((root || document).querySelectorAll(sel)); }

  function h(tag, attrs, kids) {
    var el = SVG_TAGS[tag] ? document.createElementNS(NS, tag) : document.createElement(tag);
    for (var k in attrs || {}) { if (Object.prototype.hasOwnProperty.call(attrs, k)) { el.setAttribute(k, attrs[k]); } }
    (kids || []).forEach(function (c) { el.appendChild(typeof c === "string" ? document.createTextNode(c) : c); });
    return el;
  }

  /* ---- state <-> URL ---- */
  function load() {
    var p = new URLSearchParams(location.hash.replace(/^#/, ""));
    var hist = $("hist");
    ui.t = p.get("t") || (hist && hist.getAttribute("data-tab")) || DEFAULTS.t;
    ui.m = p.get("m");
    ui.r = p.get("r") || (hist && hist.getAttribute("data-range")) || DEFAULTS.r;
    ui.pick = p.get("pick");
  }

  function save() {
    var p = new URLSearchParams();
    if (ui.t !== DEFAULTS.t) { p.set("t", ui.t); }
    var list = currentList();
    if (ui.m && (!list || ui.m !== list.getAttribute("data-default"))) { p.set("m", ui.m); }
    if (ui.r !== DEFAULTS.r) { p.set("r", ui.r); }
    if (ui.pick) { p.set("pick", ui.pick); }
    var q = p.toString();
    var search = ui.r === DEFAULTS.r ? "" : "?range=" + encodeURIComponent(ui.r);
    try { history.replaceState(null, "", location.pathname + search + (q ? "#" + q : "")); }
    catch (err) { /* state then lives for this page view only */ }
  }

  /* ---- data ---- */
  function data() {
    var box = $("hchart");
    if (!box) { return null; }
    var raw = box.getAttribute("data-hist");
    if (raw !== cacheKey) {
      cacheKey = raw;
      try { cacheData = JSON.parse(raw); } catch (err) { cacheData = null; }
    }
    return cacheData;
  }

  function currentList() {
    return all(".cardlist").filter(function (l) { return l.getAttribute("data-tab") === ui.t; })[0] || null;
  }

  function pickIndex(d) {
    if (!d || !ui.pick) { return -1; }
    return d.t.indexOf(ui.pick);
  }

  /* ---- range: the one control that needs the server ---- */
  function ensureRange() {
    var hist = $("hist");
    if (!hist) { return false; }
    var have = hist.getAttribute("data-range");
    if (!$("range-" + ui.r)) { ui.r = have; }
    if (have === ui.r) { hist.removeAttribute("aria-busy"); return false; }
    hist.setAttribute("aria-busy", "true");
    if (fetching !== ui.r && window.htmx) {
      fetching = ui.r;
      window.htmx.ajax("GET", "/history/fragment?range=" + encodeURIComponent(ui.r), { target: "#hist", swap: "outerHTML" });
    }
    return true;
  }

  /* ---- the visible parts ---- */
  function applyTabs() {
    var tabs = all(".cardlist").map(function (l) { return l.getAttribute("data-tab"); });
    if (tabs.indexOf(ui.t) === -1) { ui.t = tabs[0] || DEFAULTS.t; }
    all("button[data-tab]").forEach(function (b) { b.setAttribute("aria-pressed", b.getAttribute("data-tab") === ui.t ? "true" : "false"); });
    all("button[data-range]").forEach(function (b) { b.setAttribute("aria-pressed", b.getAttribute("data-range") === ui.r ? "true" : "false"); });
    all(".cardlist").forEach(function (l) { l.hidden = l.getAttribute("data-tab") !== ui.t; });
  }

  function applyMetric() {
    var list = currentList();
    var keys = list ? all(".card", list).map(function (c) { return c.getAttribute("data-k"); }) : [];
    if (keys.indexOf(ui.m) === -1) { ui.m = list ? list.getAttribute("data-default") || keys[0] || null : null; }
    all(".card").forEach(function (c) { c.setAttribute("aria-pressed", c.getAttribute("data-k") === ui.m ? "true" : "false"); });
    all(".mhead, .mlower").forEach(function (el) { el.hidden = el.getAttribute("data-k") !== ui.m; });
  }

  function entry() {
    var d = data();
    return d && ui.m ? d.metrics[ui.m] || null : null;
  }

  function repoLabel(d, repo) {
    var url = d.urls[repo], name = d.names[repo] || repo;
    return url ? h("a", { href: url, title: repo }, [name]) : h("span", { title: repo }, [name]);
  }

  function barSection(title, rows, label) {
    if (!rows.length) { return null; }
    var top = rows[0][1] || 1;
    var sec = h("section", { "class": "hdrawer" }, [h("h3", {}, [title])]);
    rows.slice(0, 8).forEach(function (r, i) {
      sec.appendChild(h("div", { "class": "hbar" + (i === 0 ? " lead" : "") }, [
        h("span", { "class": "hl" }, [label(r[0])]),
        h("svg", { "class": "hb", viewBox: "0 0 100 8", preserveAspectRatio: "none", "aria-hidden": "true", focusable: "false" },
          [h("rect", { "class": "hb-fill", x: 0, y: 0, width: Math.max(0, Math.min(100, 100 * r[1] / top)).toFixed(2), height: 8 })]),
        h("b", {}, [r[2]])
      ]));
    });
    if (rows.length > 8) { sec.appendChild(h("p", { "class": "dtitle" }, ["and " + (rows.length - 8) + " more"])); }
    return sec;
  }

  function rowsAt(cells, idx) {
    var rows = [];
    Object.keys(cells || {}).forEach(function (k) {
      cells[k].forEach(function (c) { if (c[0] === idx) { rows.push([k, c[1], c[2]]); } });
    });
    return rows.sort(function (a, b) { return b[1] - a[1] || (a[0] < b[0] ? -1 : 1); });
  }

  function applyBucket() {
    var drawer = $("hbucket"), d = data(), e = entry();
    if (!drawer) { return; }
    while (drawer.firstChild) { drawer.removeChild(drawer.firstChild); }
    var idx = pickIndex(d);
    if (idx < 0 || !e || e.chart === "none") { ui.pick = null; idx = -1; }
    drawer.hidden = idx < 0;
    all(".mlower .lower").forEach(function (el) { el.hidden = idx >= 0; });
    if (idx < 0) { return; }
    var when = window.CwHistChart.fmtTime(d.t[idx], d.range, d.range === "7d");
    var prsCell = (e.prs || []).filter(function (c) { return c[0] === idx; })[0];
    var prsBit = prsCell ? " (" + prsCell[2] + " PRs)" : "";
    drawer.appendChild(h("p", { "class": "dtitle" }, [
      "Selected bucket: " + when + " · " + (e.f[idx] === null ? "no data" : e.f[idx] + prsBit) + " · "
    ]));
    drawer.firstChild.appendChild(h("button", { type: "button", "class": "linkish", "data-clear": "1" }, ["back to the whole window"]));
    var box = h("div", { "class": "lower" });
    var repos = barSection("By repo in the selected bucket", rowsAt(e.repo, idx), function (r) { return repoLabel(d, r); });
    var parts = barSection("By reason in the selected bucket", rowsAt(e.parts, idx), function (k) { return h("span", {}, [k]); });
    var causes = barSection("By cause in the selected bucket", rowsAt(e.causes, idx), function (k) { return h("span", {}, [k]); });
    if (repos) { box.appendChild(repos); }
    if (parts) { box.appendChild(parts); }
    if (causes) { box.appendChild(causes); }
    if (!repos && !parts && !causes) { box.appendChild(h("p", { "class": "dim" }, ["Nothing attributed to a repo, reason or cause in this bucket."])); }
    drawer.appendChild(box);
  }

  function drawChart() {
    var box = $("hchart"), d = data();
    if (!box || !d || !window.CwHistChart) { return; }
    var idx = pickIndex(d);
    window.CwHistChart.draw(box, entry(), d.t, {
      range: d.range, picked: idx < 0 ? null : idx,
      onPick: function (k) { ui.pick = k === null ? null : d.t[k]; commit(); }
    });
  }

  function observeChart() {
    var box = $("hchart");
    if (observed === box) { return; }
    if (observer) { observer.disconnect(); }
    observed = box;
    if (box && window.ResizeObserver) {
      observer = new ResizeObserver(function () {
        if (frame) { return; }
        frame = requestAnimationFrame(function () { frame = 0; drawChart(); });
      });
      observer.observe(box);
    }
  }

  function apply() {
    if (!$("hist")) { return; }
    applyTabs();
    if (ensureRange()) { return; }
    applyMetric();
    applyBucket();
    drawChart();
    observeChart();
  }

  function commit() { save(); apply(); }

  /* ---- events ---- */
  document.addEventListener("click", function (e) {
    var t = e.target && e.target.closest ? e.target : null;
    if (!t || !t.closest("#hist")) { return; }
    var hit;
    if ((hit = t.closest("button[data-tab]"))) { ui.t = hit.getAttribute("data-tab"); ui.m = null; ui.pick = null; }
    else if ((hit = t.closest("button[data-range]"))) { ui.r = hit.getAttribute("data-range"); ui.pick = null; }
    else if ((hit = t.closest("button.card"))) { ui.m = hit.getAttribute("data-k"); ui.pick = null; }
    else if (t.closest("button[data-clear]")) { ui.pick = null; }
    else { return; }
    commit();
  });

  function pickBar(delta) {
    var d = data(), e = entry();
    if (!d || !e || e.chart === "none" || !d.t.length) { return; }
    var n = d.t.length, i = pickIndex(d);
    var next = i === -1 ? (delta > 0 ? 0 : n - 1) : Math.max(0, Math.min(n - 1, i + delta));
    ui.pick = d.t[next];
    commit();
  }

  document.addEventListener("keydown", function (e) {
    if (e.ctrlKey || e.metaKey || e.altKey) { return; }
    var t = e.target;
    if (!t || t.id !== "hchart") { return; }
    if (e.key === "ArrowLeft") { pickBar(-1); e.preventDefault(); }
    else if (e.key === "ArrowRight") { pickBar(1); e.preventDefault(); }
    else if (e.key === "Escape" && ui.pick) { ui.pick = null; commit(); }
  });

  function afterSwap() {
    var hist = $("hist");
    if (hist && hist.getAttribute("data-range") === fetching) { fetching = null; }
    apply();
  }
  function fetchFailed() {  /* stay on the range we have rather than a blank, busy page */
    var hist = $("hist");
    if (!fetching || !hist) { return; }
    fetching = null;
    ui.r = hist.getAttribute("data-range");
    commit();
  }
  document.addEventListener("htmx:responseError", fetchFailed);
  document.addEventListener("htmx:sendError", fetchFailed);
  document.addEventListener("htmx:afterSwap", afterSwap);
  document.addEventListener("htmx:afterSettle", afterSwap);
  window.addEventListener("hashchange", function () { load(); apply(); });

  function start() { load(); save(); apply(); }
  if (document.readyState === "loading") { document.addEventListener("DOMContentLoaded", start); }
  else { start(); }
})();
