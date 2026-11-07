/**
 * NeuroTropic — tiny zero-dependency server.
 *   node server.js            -> http://localhost:3000
 *
 * API (JSON):
 *   GET  /api/state           current board state
 *   POST /api/input           any input accepted by TicTacToe.applyInput
 *   POST /api/move            { "position": 4 } | { "row": 1, "col": 1 } | { "probabilities": [...] }
 *   POST /api/reset           { "startingPlayer": "X" }   (body optional)
 *   POST /api/undo
 *
 * Neural network telemetry (sent by ai/neurotropic_ai.py, shown live on the page):
 *   GET  /api/nn              latest forward pass, training progress, stats, events
 *   POST /api/nn              { forward?, training?, stats?, event? }  (partial updates)
 */
const http = require("http");
const fs = require("fs");
const path = require("path");
const { TicTacToe } = require("./public/game.js");

const PORT = Number(process.env.PORT) || 3000;
const PUBLIC_DIR = path.join(__dirname, "public");
const game = new TicTacToe();

const MAX_EVENTS = 40;
const nn = { seq: 0, updatedAt: null, forward: null, training: null, stats: null, events: [] };

function updateNN(body) {
  for (const key of ["forward", "training", "stats"]) {
    if (body[key] !== undefined) nn[key] = body[key];
  }
  if (body.event) {
    nn.events.push({ t: Date.now(), ...(typeof body.event === "string" ? { text: body.event } : body.event) });
    if (nn.events.length > MAX_EVENTS) nn.events.splice(0, nn.events.length - MAX_EVENTS);
  }
  nn.seq += 1;
  nn.updatedAt = Date.now();
  return { ok: true, seq: nn.seq };
}

const MIME = {
  ".html": "text/html; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".js": "application/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".svg": "image/svg+xml",
};

function sendJSON(res, status, data) {
  res.writeHead(status, {
    "Content-Type": "application/json; charset=utf-8",
    "Access-Control-Allow-Origin": "*",
    "Cache-Control": "no-store",
  });
  res.end(JSON.stringify(data));
}

function readBody(req) {
  return new Promise((resolve, reject) => {
    let raw = "";
    req.on("data", (chunk) => {
      raw += chunk;
      if (raw.length > 1e6) reject(new Error("Body too large"));
    });
    req.on("end", () => {
      if (!raw.trim()) return resolve({});
      try { resolve(JSON.parse(raw)); } catch { reject(new Error("Invalid JSON body")); }
    });
    req.on("error", reject);
  });
}

async function handleApi(req, res, pathname) {
  if (req.method === "GET" && pathname === "/api/state") return sendJSON(res, 200, game.toJSON());
  if (req.method === "GET" && pathname === "/api/nn") return sendJSON(res, 200, nn);
  if (req.method !== "POST") return sendJSON(res, 405, { ok: false, error: "Method not allowed" });

  let body;
  try { body = await readBody(req); } catch (e) { return sendJSON(res, 400, { ok: false, error: e.message }); }

  if (pathname === "/api/nn") return sendJSON(res, 200, updateNN(body));

  const actions = { "/api/input": null, "/api/move": "move", "/api/reset": "reset", "/api/undo": "undo" };
  if (!(pathname in actions)) return sendJSON(res, 404, { ok: false, error: "Unknown endpoint" });
  if (actions[pathname]) body = { ...body, action: actions[pathname] };

  const result = game.applyInput(body);
  sendJSON(res, result.ok ? 200 : 400, result);
}

function serveStatic(req, res, pathname) {
  const file = path.normalize(path.join(PUBLIC_DIR, pathname === "/" ? "index.html" : pathname));
  if (!file.startsWith(PUBLIC_DIR)) { res.writeHead(403); return res.end("Forbidden"); }
  fs.readFile(file, (err, data) => {
    if (err) { res.writeHead(404); return res.end("Not found"); }
    res.writeHead(200, { "Content-Type": MIME[path.extname(file)] || "application/octet-stream" });
    res.end(data);
  });
}

const server = http.createServer((req, res) => {
  const { pathname } = new URL(req.url, `http://${req.headers.host}`);
  if (req.method === "OPTIONS") {
    res.writeHead(204, {
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
      "Access-Control-Allow-Headers": "Content-Type",
    });
    return res.end();
  }
  if (pathname.startsWith("/api/")) return handleApi(req, res, pathname);
  serveStatic(req, res, pathname);
});

if (require.main === module) {
  server.listen(PORT, () => console.log(`NeuroTropic running at http://localhost:${PORT}`));
}

module.exports = { server, game };
