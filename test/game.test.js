const test = require("node:test");
const assert = require("node:assert");
const { TicTacToe, checkWin, checkDraw } = require("../public/game.js");

test("3x3 empty board, X starts", () => {
  const s = new TicTacToe().toJSON();
  assert.strictEqual(s.board.length, 9);
  assert.strictEqual(s.currentPlayer, "X");
  assert.deepStrictEqual(s.legalMoves, [0, 1, 2, 3, 4, 5, 6, 7, 8]);
});

test("row, column and diagonal wins", () => {
  for (const line of [[0, 1, 2], [2, 5, 8], [0, 4, 8], [2, 4, 6]]) {
    const b = Array(9).fill(null);
    line.forEach((i) => (b[i] = "O"));
    assert.deepStrictEqual(checkWin(b), { winner: "O", line });
  }
});

test("game detects win and blocks further moves", () => {
  const g = new TicTacToe();
  [0, 3, 1, 4, 2].forEach((p) => g.play(p));
  const s = g.toJSON();
  assert.strictEqual(s.status, "win");
  assert.strictEqual(s.winner, "X");
  assert.deepStrictEqual(s.winningLine, [0, 1, 2]);
  assert.throws(() => g.play(5));
});

test("draw detection", () => {
  const g = new TicTacToe();
  [0, 1, 2, 4, 3, 5, 7, 6, 8].forEach((p) => g.play(p));
  assert.strictEqual(g.status, "draw");
  assert.ok(checkDraw(g.board));
});

test("applyInput formats", () => {
  const g = new TicTacToe();
  assert.ok(g.applyInput({ position: 4 }).ok);
  assert.ok(g.applyInput('{"action":"move","row":0,"col":0}').ok);
  assert.strictEqual(g.board[0], "O");
  const bad = g.applyInput({ position: 4 });
  assert.strictEqual(bad.ok, false);
  assert.match(bad.error, /taken/);
  assert.strictEqual(g.applyInput({ position: 1, player: "O" }).ok, false); // wrong turn
  // probabilities: picks best legal cell (4 is taken, so 8)
  const r = g.applyInput({ probabilities: [0, 0, 0, 0, 9, 0, 0, 0, 5] });
  assert.ok(r.ok);
  assert.strictEqual(g.board[8], "X");
  assert.ok(g.applyInput({ action: "undo" }).ok);
  assert.strictEqual(g.board[8], null);
  assert.ok(g.applyInput({ action: "reset" }).ok);
  assert.ok(g.board.every((c) => c === null));
});

test("load board and nn encodings", () => {
  const g = new TicTacToe();
  const r = g.applyInput({ action: "load", board: [1, -1, 0, 0, 1, 0, 0, 0, 0] });
  assert.ok(r.ok, r.error);
  assert.strictEqual(r.state.currentPlayer, "O");
  assert.deepStrictEqual(r.state.nn.absolute, [1, -1, 0, 0, 1, 0, 0, 0, 0]);
  assert.deepStrictEqual(r.state.nn.relative, [-1, 1, 0, 0, -1, 0, 0, 0, 0]);
  assert.deepStrictEqual(r.state.nn.legalMask, [0, 0, 1, 1, 0, 1, 1, 1, 1]);
});

test("HTTP API", async () => {
  const { server } = require("../server.js");
  await new Promise((r) => server.listen(0, r));
  const base = `http://localhost:${server.address().port}`;
  const post = (p, b) => fetch(base + p, { method: "POST", body: JSON.stringify(b ?? {}) }).then((r) => r.json());
  try {
    assert.strictEqual((await (await fetch(base + "/api/state")).json()).status, "playing");
    assert.ok((await post("/api/move", { position: 4 })).ok);
    assert.strictEqual((await post("/api/input", { position: 4 })).ok, false);
    assert.strictEqual((await (await fetch(base + "/api/state")).json()).board[4], "X");
    assert.ok((await post("/api/reset")).ok);
    assert.ok((await (await fetch(base + "/")).text()).includes("NEUROTROPIC"));
  } finally {
    server.close();
  }
});
