/**
 * NeuroTropic — Tic Tac Toe engine (no DOM, no dependencies).
 * Works in the browser (window.TicTacToeEngine) and in Node (require).
 *
 * Board: array of 9 cells, index = row * 3 + col
 *   0 | 1 | 2
 *   3 | 4 | 5
 *   6 | 7 | 8
 * Cell values: "X", "O" or null.
 */
(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.TicTacToeEngine = factory();
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  const SIZE = 3;
  const PLAYERS = ["X", "O"];
  const WIN_LINES = [
    [0, 1, 2], [3, 4, 5], [6, 7, 8], // rows
    [0, 3, 6], [1, 4, 7], [2, 5, 8], // columns
    [0, 4, 8], [2, 4, 6],            // diagonals
  ];

  /** Returns { winner, line } or null. */
  function checkWin(board) {
    for (const line of WIN_LINES) {
      const [a, b, c] = line;
      if (board[a] && board[a] === board[b] && board[a] === board[c]) {
        return { winner: board[a], line: line.slice() };
      }
    }
    return null;
  }

  /** Draw = board full and nobody won. */
  function checkDraw(board) {
    return board.every((c) => c !== null) && !checkWin(board);
  }

  function legalMoves(board) {
    const moves = [];
    board.forEach((c, i) => { if (c === null) moves.push(i); });
    return moves;
  }

  class TicTacToe {
    constructor(startingPlayer = "X") {
      this.reset(startingPlayer);
    }

    reset(startingPlayer = "X") {
      if (!PLAYERS.includes(startingPlayer)) throw new Error('startingPlayer must be "X" or "O"');
      this.board = Array(SIZE * SIZE).fill(null);
      this.startingPlayer = startingPlayer;
      this.currentPlayer = startingPlayer;
      this.status = "playing"; // "playing" | "win" | "draw"
      this.winner = null;
      this.winningLine = null;
      this.history = [];       // [{ player, position, row, col }]
      this.gameId = (this.gameId || 0) + 1;
      return this.toJSON();
    }

    /**
     * Play a move. position is 0..8 (or pass row/col via applyInput).
     * Throws an Error with a readable message on illegal moves.
     */
    play(position, player = this.currentPlayer) {
      if (this.status !== "playing") throw new Error("Game is over; send a reset first");
      if (!Number.isInteger(position) || position < 0 || position > 8) {
        throw new Error("position must be an integer 0..8");
      }
      if (player !== this.currentPlayer) throw new Error(`It is ${this.currentPlayer}'s turn`);
      if (this.board[position] !== null) throw new Error(`Cell ${position} is already taken`);

      this.board[position] = player;
      this.history.push({ player, position, row: Math.floor(position / SIZE), col: position % SIZE });
      this._evaluate();
      return this.toJSON();
    }

    undo() {
      const last = this.history.pop();
      if (!last) throw new Error("Nothing to undo");
      this.board[last.position] = null;
      this.currentPlayer = last.player;
      this.status = "playing";
      this.winner = null;
      this.winningLine = null;
      return this.toJSON();
    }

    /** Replace the whole board (e.g. to set up a training position). */
    loadBoard(board, currentPlayer) {
      if (!Array.isArray(board) || board.length !== 9) throw new Error("board must be an array of 9 cells");
      const norm = board.map((c) => {
        if (c === "X" || c === 1) return "X";
        if (c === "O" || c === -1) return "O";
        if (c === null || c === 0 || c === "" || c === " ") return null;
        throw new Error(`Invalid cell value: ${JSON.stringify(c)}`);
      });
      const x = norm.filter((c) => c === "X").length;
      const o = norm.filter((c) => c === "O").length;
      if (Math.abs(x - o) > 1) throw new Error("Invalid board: piece counts differ by more than 1");
      this.board = norm;
      this.history = [];
      this.currentPlayer = currentPlayer || (x === o ? this.startingPlayer : x > o ? "O" : "X");
      this._evaluate(false);
      return this.toJSON();
    }

    _evaluate(switchTurn = true) {
      const win = checkWin(this.board);
      if (win) {
        this.status = "win";
        this.winner = win.winner;
        this.winningLine = win.line;
      } else if (checkDraw(this.board)) {
        this.status = "draw";
        this.winner = null;
        this.winningLine = null;
      } else {
        this.status = "playing";
        this.winner = null;
        this.winningLine = null;
        if (switchTurn) this.currentPlayer = this.currentPlayer === "X" ? "O" : "X";
      }
    }

    /**
     * Accepts a JSON object (or JSON string) and applies it. Supported shapes:
     *   { "action": "move", "position": 4 }
     *   { "action": "move", "row": 1, "col": 2, "player": "O" }
     *   { "position": 4 }                       (action defaults to "move")
     *   { "action": "reset", "startingPlayer": "O" }
     *   { "action": "undo" }
     *   { "action": "load", "board": ["X",null,...], "currentPlayer": "O" }
     *   { "action": "move", "probabilities": [9 numbers] }  -> plays the
     *       highest-scoring legal cell (handy for neural-network output)
     * Returns { ok: true, state } or { ok: false, error, state }.
     */
    applyInput(input) {
      try {
        const msg = typeof input === "string" ? JSON.parse(input) : input;
        if (!msg || typeof msg !== "object") throw new Error("Input must be a JSON object");
        const action = msg.action || (msg.board ? "load" : "move");

        switch (action) {
          case "move": {
            let pos = msg.position;
            if (pos === undefined && msg.row !== undefined && msg.col !== undefined) {
              if (![0, 1, 2].includes(msg.row) || ![0, 1, 2].includes(msg.col)) {
                throw new Error("row and col must be 0, 1 or 2");
              }
              pos = msg.row * SIZE + msg.col;
            }
            if (pos === undefined && Array.isArray(msg.probabilities)) {
              pos = this.bestFromScores(msg.probabilities);
            }
            if (pos === undefined) throw new Error("move needs position, row/col or probabilities");
            this.play(pos, msg.player || this.currentPlayer);
            break;
          }
          case "reset": this.reset(msg.startingPlayer || "X"); break;
          case "undo": this.undo(); break;
          case "load": this.loadBoard(msg.board, msg.currentPlayer); break;
          default: throw new Error(`Unknown action: ${action}`);
        }
        return { ok: true, state: this.toJSON() };
      } catch (err) {
        return { ok: false, error: err.message, state: this.toJSON() };
      }
    }

    /** Index of the highest score among legal cells. */
    bestFromScores(scores) {
      if (!Array.isArray(scores) || scores.length !== 9) throw new Error("probabilities must have 9 numbers");
      const moves = legalMoves(this.board);
      if (!moves.length) throw new Error("No legal moves");
      return moves.reduce((best, i) => (scores[i] > scores[best] ? i : best), moves[0]);
    }

    /** Full JSON snapshot of the current game. */
    toJSON() {
      const toNum = (c) => (c === "X" ? 1 : c === "O" ? -1 : 0);
      const me = this.currentPlayer;
      return {
        gameId: this.gameId,
        board: this.board.slice(),
        grid: [0, 1, 2].map((r) => this.board.slice(r * 3, r * 3 + 3)),
        currentPlayer: this.currentPlayer,
        status: this.status,
        winner: this.winner,
        winningLine: this.winningLine,
        moveCount: this.history.length,
        legalMoves: this.status === "playing" ? legalMoves(this.board) : [],
        history: this.history.map((h) => ({ ...h })),
        // Neural-network friendly encodings
        nn: {
          // X = 1, O = -1, empty = 0
          absolute: this.board.map(toNum),
          // current player = 1, opponent = -1, empty = 0
          relative: this.board.map((c) => (c === null ? 0 : c === me ? 1 : -1)),
          // 1 where a move is allowed, 0 otherwise
          legalMask: this.board.map((c) => (c === null && this.status === "playing" ? 1 : 0)),
        },
      };
    }
  }

  return { TicTacToe, checkWin, checkDraw, legalMoves, WIN_LINES };
});
