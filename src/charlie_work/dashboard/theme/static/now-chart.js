/* The Now page's progress bar chart (CSP script-src 'self': same-origin, no inline script).
   Pure drawing: CwChart.draw(box, series, view) fills box's <svg> with DOM nodes (never
   HTML strings) in pixel units read from the box, so text is never scaled and the chart
   redraws crisply at any size. All colour comes from classes in now-page.css (ink for the
   bars, grey for context); the avg line is dashed and labelled directly. now.js owns the
   state (metric, range, picked bar) and calls draw() on load, resize and after each swap.

   series: {points: [[iso, value], ...], approx, notInstrumented}
   view:   {range: "24h"|"7d"|"30d"|"90d", unit: "merged", picked: iso|null, onPick(iso|null)} */
(function (root) {
  "use strict";

  var NS = "http://www.w3.org/2000/svg";

  function node(tag, attrs, text) {
    var el = document.createElementNS(NS, tag);
    for (var k in attrs) { if (Object.prototype.hasOwnProperty.call(attrs, k)) { el.setAttribute(k, attrs[k]); } }
    if (text !== undefined) { el.textContent = text; }
    return el;
  }

  function fmtTime(iso, range, withHour) {
    var d = new Date(iso);
    if (range === "24h") { return d.toLocaleTimeString([], { hour: "numeric" }); }
    var day = d.toLocaleDateString([], { month: "short", day: "numeric" });
    return withHour ? day + " " + d.toLocaleTimeString([], { hour: "numeric" }) : day;
  }

  function niceTop(max) {
    var step = Math.pow(10, Math.floor(Math.log10(max)));
    return Math.ceil(max / step) * step;
  }

  function clear(el) { while (el.firstChild) { el.removeChild(el.firstChild); } }

  function tipFill(tip, value, view, iso) {
    clear(tip);
    var b = document.createElement("b");
    b.textContent = String(Math.round(value));
    tip.appendChild(b);
    tip.appendChild(document.createTextNode(" " + view.unit + " · " + fmtTime(iso, view.range, view.range !== "30d" && view.range !== "90d")));
  }

  function draw(box, series, view) {
    var svg = box.querySelector("svg");
    var tip = box.querySelector(".tip");
    if (!svg) { return; }
    clear(svg);
    var W = box.clientWidth, H = box.clientHeight;
    if (W < 40 || H < 40) { return; }
    svg.setAttribute("viewBox", "0 0 " + W + " " + H);
    var pts = series.points;
    if (!pts.length || series.notInstrumented) {
      svg.appendChild(node("text", { x: W / 2, y: H / 2, "text-anchor": "middle", "class": "ct" },
        series.notInstrumented ? "Not instrumented yet" : "No data in this window"));
      return;
    }
    var fs = parseFloat(getComputedStyle(box).fontSize) * 0.85;
    var m = { l: fs * 3.2, r: fs * 5.5, t: 14, b: fs * 2.4 };
    var iw = W - m.l - m.r, ih = H - m.t - m.b;
    var total = 0, max = 1, i;
    for (i = 0; i < pts.length; i++) { total += pts[i][1]; max = Math.max(max, pts[i][1]); }
    var top = niceTop(max);
    var y = function (v) { return m.t + ih - (v / top) * ih; };
    var bw = iw / pts.length, gap = Math.min(2, bw * 0.2);
    var avg = total / pts.length;
    var ticks = [0];
    if (top / 2 === Math.floor(top / 2)) { ticks.push(top / 2); }
    ticks.push(top);
    ticks.forEach(function (t) {
      svg.appendChild(node("line", { x1: m.l, x2: m.l + iw, y1: y(t), y2: y(t), "class": "cgrid" }));
      svg.appendChild(node("text", { x: m.l - 8, y: y(t) + 4, "text-anchor": "end", "class": "ct", "font-size": fs }, String(t)));
    });
    var every = Math.ceil(pts.length / Math.max(2, Math.floor(iw / 90)));
    var sevenDay = view.range === "7d";
    pts.forEach(function (p, k) {
      var h = Math.max(p[1] > 0 ? 2 : 0, y(0) - y(p[1]));
      var x = m.l + k * bw + gap / 2, w = Math.max(1, bw - gap);
      var dim = view.picked !== null && view.picked !== p[0];
      svg.appendChild(node("rect", {
        x: x, y: y(0) - h, width: w, height: h, rx: Math.min(4, w / 2),
        "class": "cbar" + (dim ? " dim" : "")
      }));
      var newDay = k === 0 || new Date(pts[k - 1][0]).getDate() !== new Date(p[0]).getDate();
      if (sevenDay ? newDay : k % every === 0) {
        svg.appendChild(node("text", { x: x + w / 2, y: H - 6, "text-anchor": "middle", "class": "ct", "font-size": fs },
          fmtTime(p[0], view.range, false)));
      }
    });
    svg.appendChild(node("line", { x1: m.l, x2: m.l + iw, y1: y(avg), y2: y(avg), "class": "cavg" }));
    svg.appendChild(node("text", { x: m.l + iw + 6, y: y(avg) + 4, "class": "ct", "font-size": fs },
      "avg " + (avg < 10 ? avg.toFixed(1) : avg.toFixed(0))));
    pts.forEach(function (p, k) {
      var hit = node("rect", { x: m.l + k * bw, y: m.t, width: bw, height: ih, "class": "chit" });
      hit.addEventListener("mousemove", function () {
        if (!tip) { return; }
        tipFill(tip, p[1], view, p[0]);
        tip.hidden = false;
        tip.style.left = (m.l + (k + 0.5) * bw) + "px";
        tip.style.top = y(p[1]) + "px";
      });
      hit.addEventListener("mouseleave", function () { if (tip) { tip.hidden = true; } });
      hit.addEventListener("click", function () { view.onPick(view.picked === p[0] ? null : p[0]); });
      svg.appendChild(hit);
    });
  }

  root.CwChart = { draw: draw, fmtTime: fmtTime };
})(this);
