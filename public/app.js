/**
 * NeuroTropic UI.
 *
 * Two modes, chosen automatically:
 *  - SERVER: page was served by `node server.js`. The server owns the game,
 *            so external clients (e.g. a neural network) can fetch the board
 *            from GET /api/state and play via POST /api/input. The page polls
 *            the server and reflects those moves live.
 *  - LOCAL:  page opened directly (file:// or static host). The game runs in
 *            the browser; use window.NeuroTropic.getState() / sendInput().
 */
(() => {
  "use strict";

  const { TicTacToe } = window.TicTacToeEngine;
  const POLL_MS = 500;

  const $ = (id) => document.getElementById(id);
  const boardEl = $("board");
  const statusEl = $("status");
  const stateJsonEl = $("state-json");
  const inputEl = $("input-json");
  const inputResultEl = $("input-result");

  let mode = "local";
  let localGame = new TicTacToe();
  let state = localGame.toJSON();
  let lastRendered = "";
  const scores = { X: 0, O: 0, draw: 0 };
  const scoredGames = new Set();

  // ---------- transport (local or server) ----------
  async function apiGetState() {
    if (mode === "local") return localGame.toJSON();
    const res = await fetch("/api/state", { cache: "no-store" });
    return res.json();
  }

  async function apiSendInput(input) {
    if (mode === "local") return localGame.applyInput(input);
    const body = typeof input === "string" ? input : JSON.stringify(input);
    const res = await fetch("/api/input", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body,
    });
    return res.json();
  }

  async function detectMode() {
    if (location.protocol === "file:") return "local";
    try {
      const res = await fetch("/api/state", { cache: "no-store" });
      if (res.ok && (res.headers.get("content-type") || "").includes("json")) return "server";
    } catch (_) { /* no server */ }
    return "local";
  }

  // ---------- rendering ----------
  function buildBoard() {
    for (let i = 0; i < 9; i++) {
      const cell = document.createElement("button");
      cell.className = "cell";
      cell.dataset.index = i;
      cell.setAttribute("role", "gridcell");
      cell.innerHTML = `<span class="idx">${i}</span><span class="mark"></span>`;
      cell.addEventListener("click", () => send({ action: "move", position: i }));
      boardEl.appendChild(cell);
    }
  }

  function updateScores(s) {
    if (s.status === "playing") return;
    const key = `${s.gameId}:${s.moveCount}`;
    if (scoredGames.has(key)) return;
    scoredGames.add(key);
    if (s.status === "win") scores[s.winner]++;
    else scores.draw++;
    $("score-x").textContent = scores.X;
    $("score-o").textContent = scores.O;
    $("score-d").textContent = scores.draw;
  }

  function render(s) {
    const json = JSON.stringify(s, null, 2);
    if (json === lastRendered) return;
    lastRendered = json;
    state = s;

    [...boardEl.children].forEach((cell, i) => {
      const v = s.board[i];
      cell.className = "cell" + (v ? " " + v.toLowerCase() : "") +
        (s.winningLine && s.winningLine.includes(i) ? " win" : "");
      cell.querySelector(".mark").textContent = v || "";
      cell.disabled = !!v || s.status !== "playing";
      cell.setAttribute("aria-label", `Célula ${i}: ${v || "vazia"}`);
    });

    statusEl.className = "status";
    if (s.status === "win") {
      statusEl.textContent = `▲ ${s.winner} WINS ▲`;
    } else if (s.status === "draw") {
      statusEl.textContent = "◆ DRAW ◆";
      statusEl.classList.add("draw");
    } else {
      statusEl.textContent = `${s.currentPlayer} TO MOVE`;
      statusEl.classList.add(`turn-${s.currentPlayer.toLowerCase()}`);
    }

    stateJsonEl.textContent = json;
    updateScores(s);
  }

  function showResult(result) {
    inputResultEl.textContent = JSON.stringify(
      result.ok ? { ok: true } : { ok: false, error: result.error }, null, 2);
    inputResultEl.classList.toggle("error", !result.ok);
  }

  async function send(input) {
    let result;
    try {
      result = await apiSendInput(input);
    } catch (err) {
      result = { ok: false, error: err.message, state };
    }
    if (result.state) render(result.state);
    showResult(result);
    return result;
  }

  async function refresh() {
    try { render(await apiGetState()); } catch (_) { /* server down: keep last state */ }
  }

  // ---------- controls ----------
  $("btn-reset").addEventListener("click", () => send({ action: "reset" }));
  $("btn-undo").addEventListener("click", () => send({ action: "undo" }));
  $("btn-send").addEventListener("click", () => send(inputEl.value));
  inputEl.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) send(inputEl.value);
  });
  $("btn-copy").addEventListener("click", async () => {
    try { await navigator.clipboard.writeText(JSON.stringify(state)); $("btn-copy").textContent = "Copied"; }
    catch (_) { $("btn-copy").textContent = "Failed"; }
    setTimeout(() => ($("btn-copy").textContent = "Copy"), 1200);
  });

  // ---------- public API (for scripts / neural network in the browser) ----------
  window.NeuroTropic = {
    /** Promise<state JSON> */
    getState: apiGetState,
    /** Promise<{ ok, state, error? }> — accepts object or JSON string */
    sendInput: send,
    get mode() { return mode; },
  };

  // ---------- boot ----------
  (async () => {
    buildBoard();
    mode = await detectMode();
    $("mode").textContent = mode === "server" ? "SERVER (/api)" : "LOCAL (browser)";
    await refresh();
    if (mode === "server") setInterval(refresh, POLL_MS);
  })();
})();
