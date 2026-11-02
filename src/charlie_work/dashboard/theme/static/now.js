/* Now page behaviour (CSP script-src 'self': same-origin file, no inline script, no HTML-string injection).
   The server renders every panel with all of its detail (every group's rows, every
   drawer, all eight chart series in one data attribute); this file only decides what is
   showing, and keeps that choice in the URL hash so it survives the htmx refresh of #now:

     #g=<needs group>&open=<row ids>&more=<groups showing all rows>
      &m=<metric>&r=<range>&pick=<bucket ISO>&pipe=<stage drawer>&cap=<meter drawer>&health=1

   Every handler is delegated on document, so swapping #now needs no re-binding. After each
   swap (and settle: htmx re-applies server attributes then) apply() re-asserts the state
   from `ui`, restores the Needs list's scroll position and re-attaches the chart's
   ResizeObserver to the new node. State is driven through data-* / aria-* / hidden, never
   `class`, because htmx settles `class` back to the server's value.
   Keys: j/k focus a row, c copy its command, o open its drill-down, left/right pick a bar. */
(function () {
  "use strict";

  var NS = "http://www.w3.org/2000/svg";
  var SVG_TAGS = { svg: 1, rect: 1, line: 1 };
  var DEFAULTS = { g: null, m: "merged", r: "7d", pick: null, pipe: null, cap: null, health: false };
  var ui = { g: null, m: "merged", r: "7d", pick: null, pipe: null, cap: null, health: false, open: [], more: [] };
  var listScroll = 0;
  var cacheKey = null, cacheData = null;
  var observer = null, observed = null, frame = 0;

  function $(id) { return document.getElementById(id); }
  function all(sel, root) { return Array.prototype.slice.call((root || document).querySelectorAll(sel)); }
  function csv(v) { return v ? v.split(",").filter(Boolean) : []; }
  function has(list, v) { return list.indexOf(v) !== -1; }
  function short(repo) { return repo.replace(/^[^/]*\//, ""); }

  function h(tag, attrs, kids) {
    var el = SVG_TAGS[tag] ? document.createElementNS(NS, tag) : document.createElement(tag);
    for (var k in attrs || {}) { if (Object.prototype.hasOwnProperty.call(attrs, k)) { el.setAttribute(k, attrs[k]); } }
    (kids || []).forEach(function (c) { el.appendChild(typeof c === "string" ? document.createTextNode(c) : c); });
    return el;
  }

  /* ---- state <-> URL hash ---- */
  function load() {
    var p = new URLSearchParams(location.hash.replace(/^#/, ""));
    ui.g = p.get("g");
    ui.m = p.get("m") || DEFAULTS.m;
    ui.r = p.get("r") || DEFAULTS.r;
    ui.pick = p.get("pick");
    ui.pipe = p.get("pipe");
    ui.cap = p.get("cap");
    ui.health = p.get("health") === "1";
    ui.open = csv(p.get("open"));
    ui.more = csv(p.get("more"));
  }

  function save() {
    var p = new URLSearchParams();
    ["g", "m", "r", "pick", "pipe", "cap"].forEach(function (k) {
      if (ui[k] && ui[k] !== DEFAULTS[k]) { p.set(k, ui[k]); }
    });
    if (ui.health) { p.set("health", "1"); }
    if (ui.open.length) { p.set("open", ui.open.join(",")); }
    if (ui.more.length) { p.set("more", ui.more.join(",")); }
    var q = p.toString();
    try { history.replaceState(null, "", location.pathname + location.search + (q ? "#" + q : "")); }
    catch (err) { /* state then lives for this page view only */ }
  }

  function toggle(list, v) { return has(list, v) ? list.filter(function (x) { return x !== v; }) : list.concat([v]); }

  /* ---- Needs you ---- */
  function applyNeeds() {
    var list = $("needs-list");
    if (!list) { return; }
    var groups = all(".grp", list).map(function (s) { return s.getAttribute("data-g"); });
    var g = has(groups, ui.g) ? ui.g : list.getAttribute("data-default");
    list.setAttribute("data-active", g);
    all(".tile").forEach(function (t) { t.setAttribute("aria-pressed", t.getAttribute("data-g") === g ? "true" : "false"); });
    all(".grp", list).forEach(function (s) {
      var slug = s.getAttribute("data-g"), on = slug === g, every = has(ui.more, slug);
      if (on) { s.setAttribute("data-on", "1"); } else { s.removeAttribute("data-on"); }
      s.setAttribute("data-all", every ? "1" : "0");
      var more = s.querySelector(".more");
      if (more) {
        more.setAttribute("aria-expanded", every ? "true" : "false");
        more.textContent = every ? "show fewer" : more.getAttribute("data-n") + " more";
      }
    });
    var ids = all(".row", list).map(function (r) { return r.id; });
    ui.open = ui.open.filter(function (id) { return has(ids, id); });
    all(".row", list).forEach(function (r) {
      var on = r.closest(".grp").getAttribute("data-on") === "1" && has(ui.open, r.id);
      r.querySelector(".rowbtn").setAttribute("aria-expanded", on ? "true" : "false");
      r.querySelector(".detail").hidden = !on;
    });
  }

  /* ---- Progress chart ---- */
  function data() {
    var box = $("chart");
    if (!box) { return null; }
    var raw = box.getAttribute("data-progress");
    if (raw !== cacheKey) {
      cacheKey = raw;
      try { cacheData = JSON.parse(raw); } catch (err) { cacheData = null; }
    }
    return cacheData;
  }

  function current() {
    var d = data();
    if (!d) { return null; }
    var metric = d.metrics[ui.m] ? ui.m : DEFAULTS.m;
    var ranges = d.metrics[metric].ranges;
    var range = ranges[ui.r] ? ui.r : DEFAULTS.r;
    return { d: d, metric: metric, range: range, meta: d.metrics[metric], s: ranges[range] };
  }

  function pickIndex(c) {
    for (var i = 0; i < c.s.points.length; i++) { if (c.s.points[i][0] === ui.pick) { return i; } }
    return -1;
  }

  function setText(id, text) { var el = $(id); if (el && el.textContent !== text) { el.textContent = text; } }

  function repoLink(c, repo) {
    var url = c.d.urls[repo];
    return url ? h("a", { href: url, "class": "rl" }, [short(repo)]) : h("span", { "class": "rl" }, [short(repo)]);
  }

  function drawDrawer(c, idx) {
    var drawer = $("chart-drawer");
    if (!drawer) { return; }
    while (drawer.firstChild) { drawer.removeChild(drawer.firstChild); }
    drawer.hidden = idx < 0;
    if (idx < 0) { return; }
    var rows = [];
    Object.keys(c.s.repo).forEach(function (repo) {
      c.s.repo[repo].forEach(function (e) { if (e[0] === idx) { rows.push([repo, e[1]]); } });
    });
    rows.sort(function (a, b) { return b[1] - a[1] || (a[0] < b[0] ? -1 : 1); });
    var top = rows.length ? rows[0][1] : 1;
    drawer.appendChild(h("p", { "class": "dtitle" }, [window.CwChart.fmtTime(c.s.points[idx][0], c.range, c.range === "24h" || c.range === "7d") + " by repo"]));
    if (!rows.length) { drawer.appendChild(h("p", { "class": "dim" }, ["none in this bucket"])); }
    rows.forEach(function (r) {
      drawer.appendChild(h("div", { "class": "hbar" }, [
        h("span", { "class": "hl" }, [repoLink(c, r[0])]),
        h("svg", { "class": "hb", viewBox: "0 0 100 8", preserveAspectRatio: "none", "aria-hidden": "true" },
          [h("rect", { "class": "hb-fill", x: 0, y: 0, width: 100 * r[1] / top, height: 8 })]),
        h("b", {}, [String(Math.round(r[1]))])
      ]));
    });
  }

  function drawChart() {
    var c = current(), box = $("chart");
    if (!c || !box || !window.CwChart) { return; }
    var idx = pickIndex(c);
    window.CwChart.draw(box, c.s, {
      range: c.range, unit: c.meta.unit, picked: idx < 0 ? null : ui.pick,
      onPick: function (iso) { ui.pick = iso; save(); applyProgress(); }
    });
  }

  function applyProgress() {
    var c = current();
    if (!c) { return; }
    ui.m = c.metric; ui.r = c.range;
    var idx = pickIndex(c);
    if (idx < 0) { ui.pick = null; }
    all("[data-m]", $("progress")).forEach(function (b) { b.setAttribute("aria-pressed", b.getAttribute("data-m") === c.metric ? "true" : "false"); });
    all("[data-r]", $("progress")).forEach(function (b) { b.setAttribute("aria-pressed", b.getAttribute("data-r") === c.range ? "true" : "false"); });
    setText("prog-head", c.s.head);
    setText("prog-vs", c.s.vs);
    setText("prog-sub", c.s.sub);
    var tip = document.querySelector("#chart .tip");
    if (tip) { tip.hidden = true; }
    drawChart();
    drawDrawer(c, idx);
  }

  function observeChart() {
    var box = $("chart");
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

  /* ---- Pipeline, Capacity, health ---- */
  function applyDrawers(attr) {
    var drawers = all(".drawer[data-" + attr + "]");
    var keys = drawers.map(function (d) { return d.getAttribute("data-" + attr); });
    if (!has(keys, ui[attr])) { ui[attr] = null; }
    drawers.forEach(function (d) { d.hidden = d.getAttribute("data-" + attr) !== ui[attr]; });
    all("button[data-" + attr + "]").forEach(function (b) {
      b.setAttribute("aria-pressed", b.getAttribute("data-" + attr) === ui[attr] ? "true" : "false");
    });
  }

  function applyHealth() {
    var btn = $("health-btn"), pop = $("health-pop");
    if (!btn || !pop) { return; }
    btn.setAttribute("aria-expanded", ui.health ? "true" : "false");
    pop.hidden = !ui.health;
  }

  function apply() {
    if (!$("now")) { return; }
    applyNeeds();
    applyProgress();
    applyDrawers("pipe");
    applyDrawers("cap");
    applyHealth();
    observeChart();
  }

  function commit() { save(); apply(); }

  /* ---- events ---- */
  function onRow(btn) {
    var row = btn.closest(".row"), slug = row.closest(".grp").getAttribute("data-g");
    if (row.closest(".grp").getAttribute("data-on") !== "1") { ui.g = slug; }
    else { ui.open = toggle(ui.open, row.id); }
  }

  document.addEventListener("click", function (e) {
    var t = e.target && e.target.closest ? e.target : null;
    if (!t || !t.closest("#now")) { return; }
    var hit;
    if ((hit = t.closest(".tile, .ghead"))) {
      ui.g = hit.getAttribute("data-g");
      var list = $("needs-list");
      if (list) { list.scrollTop = 0; }
    } else if ((hit = t.closest(".rowbtn"))) { onRow(hit); }
    else if ((hit = t.closest(".more"))) { ui.more = toggle(ui.more, hit.getAttribute("data-more")); }
    else if ((hit = t.closest("button[data-m]"))) { ui.m = hit.getAttribute("data-m"); ui.pick = null; }
    else if ((hit = t.closest("button[data-r]"))) { ui.r = hit.getAttribute("data-r"); ui.pick = null; }
    else if ((hit = t.closest("button[data-pipe]"))) {
      var k = hit.getAttribute("data-pipe");
      ui.pipe = ui.pipe === k ? null : k;
    } else if ((hit = t.closest("button[data-cap]"))) {
      var c = hit.getAttribute("data-cap");
      ui.cap = ui.cap === c ? null : c;
    } else if (t.closest("#health-btn")) { ui.health = !ui.health; }
    else if (ui.health && !t.closest("#health-pop")) { ui.health = false; }
    else { return; }
    commit();
  });

  function rowButtons() {
    return all('#needs-list .grp[data-on="1"] .rowbtn').filter(function (b) { return b.getClientRects().length > 0; });
  }

  function moveRow(delta) {
    var rows = rowButtons();
    if (!rows.length) { return; }
    var i = rows.indexOf(document.activeElement);
    var next = i === -1 ? (delta > 0 ? 0 : rows.length - 1) : Math.max(0, Math.min(rows.length - 1, i + delta));
    rows[next].focus();
  }

  function pickBar(delta) {
    var c = current();
    if (!c || !c.s.points.length) { return; }
    var n = c.s.points.length, i = pickIndex(c);
    var next = i === -1 ? (delta > 0 ? 0 : n - 1) : Math.max(0, Math.min(n - 1, i + delta));
    ui.pick = c.s.points[next][0];
    commit();
  }

  document.addEventListener("keydown", function (e) {
    if (e.ctrlKey || e.metaKey || e.altKey) { return; }
    var t = e.target, tag = t && t.tagName;
    if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") { return; }
    if (t && t.id === "chart") {
      if (e.key === "ArrowLeft") { pickBar(-1); e.preventDefault(); }
      else if (e.key === "ArrowRight") { pickBar(1); e.preventDefault(); }
      else if (e.key === "Escape" && ui.pick) { ui.pick = null; commit(); }
      return;
    }
    if (e.key === "Escape" && ui.health) { ui.health = false; commit(); return; }
    var row = document.activeElement && document.activeElement.closest ? document.activeElement.closest(".row") : null;
    switch (e.key) {
      case "j": moveRow(1); e.preventDefault(); break;
      case "k": moveRow(-1); e.preventDefault(); break;
      case "c":
        var cb = row && row.querySelector("button[data-copy]");
        if (cb) { cb.click(); }
        break;
      case "o":
        var a = row && row.querySelector("a.drill");
        if (a) { location.href = a.href; e.preventDefault(); }
        break;
      default: break;
    }
  });

  document.addEventListener("htmx:beforeSwap", function () {
    var list = $("needs-list");
    listScroll = list ? list.scrollTop : 0;
  });
  function afterSwap() {
    apply();
    var list = $("needs-list");
    if (list && listScroll) { list.scrollTop = listScroll; }
  }
  document.addEventListener("htmx:afterSwap", afterSwap);
  document.addEventListener("htmx:afterSettle", afterSwap);
  window.addEventListener("hashchange", function () { load(); apply(); });

  function start() { load(); apply(); }
  if (document.readyState === "loading") { document.addEventListener("DOMContentLoaded", start); }
  else { start(); }
})();
