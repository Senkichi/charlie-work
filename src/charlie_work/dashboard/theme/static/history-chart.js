/* The History page's chart (CSP script-src 'self': same-origin, no inline script).
   Pure drawing: CwHistChart.draw(box, entry, grid, view) fills box's <svg> with DOM nodes
   (never HTML strings) in pixel units read from the box, so text is never scaled. Counts
   are bars, levels and durations a line with a gap where a bucket has no value. Every
   number arrives pre-formatted from the server (pages/history_detail.chart_entry); this
   file only places it. Colour comes from classes in now-page.css / history.css: ink for
   the data, grey for context. history.js owns the state and calls draw() on load, resize
   and after each swap.

   entry: {chart: "bars"|"line"|"none", approx, v: [n|null], f: [text|null], top,
           ticks: [[n, text]], avg: [n, text]|null, msg}
   view:  {range, picked: index|null, onPick(index|null)} */
(function (root) {
  "use strict";

  var NS = "http://www.w3.org/2000/svg";

  function node(tag, attrs, text) {
    var el = document.createElementNS(NS, tag);
    for (var k in attrs) { if (Object.prototype.hasOwnProperty.call(attrs, k)) { el.setAttribute(k, attrs[k]); } }
    if (text !== undefined) { el.textContent = text; }
    return el;
  }

  function clear(el) { while (el.firstChild) { el.removeChild(el.firstChild); } }

  function fmtTime(iso, range, withHour) {
    var d = new Date(iso);
    if (range === "24h") { return d.toLocaleTimeString([], { hour: "numeric" }); }
    var day = d.toLocaleDateString([], { month: "short", day: "numeric" });
    return withHour ? day + " " + d.toLocaleTimeString([], { hour: "numeric" }) : day;
  }

  function message(svg, W, H, text) {
    svg.appendChild(node("text", { x: W / 2, y: H / 2, "text-anchor": "middle", "class": "ct" }, text));
  }

  function draw(box, entry, grid, view) {
    var svg = box.querySelector("svg");
    var tip = box.querySelector(".tip");
    if (!svg) { return; }
    clear(svg);
    if (tip) { tip.hidden = true; }
    var W = box.clientWidth, H = box.clientHeight;
    if (W < 40 || H < 40) { return; }
    svg.setAttribute("viewBox", "0 0 " + W + " " + H);
    if (!entry || entry.chart === "none") { message(svg, W, H, (entry && entry.msg) || "Nothing to draw"); return; }
    var vals = entry.v, n = vals.length, any = false, i;
    for (i = 0; i < n; i++) { if (vals[i] !== null) { any = true; break; } }
    if (!n || !any) { message(svg, W, H, "No data in this window"); return; }
    var fs = parseFloat(getComputedStyle(box).fontSize) * 0.85;
    var m = { l: fs * 3.6, r: fs * 5.5, t: 14, b: fs * 2.4 };
    var iw = W - m.l - m.r, ih = H - m.t - m.b;
    var top = entry.top || 1;
    var y = function (v) { return m.t + ih - (v / top) * ih; };
    var bw = iw / n, gap = Math.min(2, bw * 0.2);
    var cx = function (k) { return m.l + (k + 0.5) * bw; };

    entry.ticks.forEach(function (t) {
      svg.appendChild(node("line", { x1: m.l, x2: m.l + iw, y1: y(t[0]), y2: y(t[0]), "class": "cgrid" }));
      svg.appendChild(node("text", { x: m.l - 8, y: y(t[0]) + 4, "text-anchor": "end", "class": "ct", "font-size": fs }, t[1]));
    });

    if (entry.chart === "bars") {
      vals.forEach(function (v, k) {
        if (v === null) { return; }
        var h = Math.max(v > 0 ? 2 : 0, y(0) - y(v));
        var w = Math.max(1, bw - gap);
        var dim = view.picked !== null && view.picked !== k;
        svg.appendChild(node("rect", {
          x: m.l + k * bw + gap / 2, y: y(0) - h, width: w, height: h, rx: Math.min(4, w / 2),
          "class": "cbar" + (dim ? " dim" : "")
        }));
      });
    } else {
      var d = "", pen = false, lone = [];
      vals.forEach(function (v, k) {
        if (v === null) { pen = false; return; }
        var next = k + 1 < n ? vals[k + 1] : null;
        if (!pen && next === null) { lone.push(k); }
        d += (pen ? "L" : "M") + cx(k).toFixed(1) + " " + y(v).toFixed(1);
        pen = true;
      });
      svg.appendChild(node("path", { d: d, "class": "cline" + (entry.approx ? " approx" : "") }));
      lone.forEach(function (k) { svg.appendChild(node("circle", { cx: cx(k), cy: y(vals[k]), r: 2.5, "class": "cdot" })); });
      if (view.picked !== null && vals[view.picked] !== null) {
        svg.appendChild(node("line", { x1: cx(view.picked), x2: cx(view.picked), y1: m.t, y2: m.t + ih, "class": "cpick" }));
        svg.appendChild(node("circle", { cx: cx(view.picked), cy: y(vals[view.picked]), r: 5, "class": "cdot" }));
      }
    }

    var every = Math.ceil(n / Math.max(2, Math.floor(iw / (fs * 7))));
    var sevenDay = view.range === "7d";
    grid.forEach(function (iso, k) {
      var newDay = k === 0 || new Date(grid[k - 1]).getDate() !== new Date(iso).getDate();
      if (sevenDay ? newDay : k % every === 0) {
        svg.appendChild(node("text", { x: cx(k), y: H - 6, "text-anchor": "middle", "class": "ct", "font-size": fs },
          fmtTime(iso, view.range, false)));
      }
    });

    if (entry.avg) {
      svg.appendChild(node("line", { x1: m.l, x2: m.l + iw, y1: y(entry.avg[0]), y2: y(entry.avg[0]), "class": "cavg" }));
      svg.appendChild(node("text", { x: m.l + iw + 6, y: y(entry.avg[0]) + 4, "class": "ct", "font-size": fs }, entry.avg[1]));
    }

    grid.forEach(function (iso, k) {
      var hit = node("rect", { x: m.l + k * bw, y: m.t, width: bw, height: ih, "class": "chit" });
      hit.addEventListener("mousemove", function () {
        if (!tip) { return; }
        clear(tip);
        var b = document.createElement("b");
        b.textContent = entry.f[k] === null ? "no data" : entry.f[k];
        tip.appendChild(b);
        tip.appendChild(document.createTextNode(" · " + fmtTime(iso, view.range, view.range === "7d")));
        tip.hidden = false;
        tip.style.left = cx(k) + "px";
        tip.style.top = y(vals[k] === null ? 0 : vals[k]) + "px";
      });
      hit.addEventListener("mouseleave", function () { if (tip) { tip.hidden = true; } });
      hit.addEventListener("click", function () { view.onPick(view.picked === k ? null : k); });
      svg.appendChild(hit);
    });
  }

  root.CwHistChart = { draw: draw, fmtTime: fmtTime };
})(this);
