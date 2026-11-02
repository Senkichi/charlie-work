/* Client-side staleness rule for the Now page (pure: no DOM). dashboard.js feeds it the
   time of the last successful #now swap and of the last failed poll; it decides whether
   the page is showing numbers that are no longer being refreshed. A server that stopped
   answering leaves htmx's old DOM in place, so without this a dead page looks live.
   Loaded as a classic script (window.CwStale); node's require() gets module.exports. */
(function (root) {
  "use strict";

  var FACTOR = 2; /* no success for more than FACTOR poll intervals => stale */

  function staleState(nowMs, lastOkMs, pollSeconds, lastFailMs) {
    var poll = pollSeconds > 0 ? pollSeconds : 20;
    if (nowMs - lastOkMs <= FACTOR * poll * 1000) { return null; }
    return {
      since: lastOkMs,
      unreachable: typeof lastFailMs === "number" && lastFailMs > lastOkMs
    };
  }

  function message(state, clock) {
    return "Not updating — last update " + clock +
      (state.unreachable ? " · the dashboard server is not answering" : "");
  }

  var api = { FACTOR: FACTOR, staleState: staleState, message: message };
  if (typeof module !== "undefined" && module.exports) { module.exports = api; }
  else { root.CwStale = api; }
})(this);
