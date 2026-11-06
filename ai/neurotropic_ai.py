"""
NeuroTropic AI — rede neural que joga Tic Tac Toe contra você.

Fluxo:
  1. GET /api/state              -> pega os dados úteis (nn.relative, nn.legalMask...)
  2. Rede neural (forward)       -> 9 probabilidades, uma por célula
  3. POST /api/input             -> {"probabilities": [...], "player": "O"}
  4. Fim de jogo                 -> recompensa: vitória reforça, derrota pune
  5. Memória de "melhores jogadas" por estado é atualizada
  6. model.fit(...)              -> treino por backpropagation nas melhores jogadas

Uso:
  npm start                                   # em outro terminal (servidor do jogo)
  python ai/neurotropic_ai.py play            # IA joga de O; você joga de X no navegador
  python ai/neurotropic_ai.py play --ai X     # IA joga de X
  python ai/neurotropic_ai.py pretrain --games 3000   # treino offline contra jogador aleatório
  python ai/neurotropic_ai.py stats           # mostra o que a IA já aprendeu

Dependência: numpy  (pip install numpy)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "model.npz")
MEMORY_PATH = os.path.join(HERE, "memory.json")

# Recompensas
REWARD_WIN = 1.0
REWARD_DRAW = 0.5
REWARD_LOSS = -1.0
GAMMA = 0.8  # desconto: jogadas mais perto do fim recebem mais crédito/culpa

WIN_LINES = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]


# ======================================================================
# Rede neural (MLP) com backpropagation escrito à mão
# ======================================================================
class NeuralNetwork:
    """
    Entrada: 18 valores
       - 9 do tabuleiro relativo (minha peça = 1, oponente = -1, vazio = 0)
       - 9 da máscara de jogadas legais (1 = pode jogar)
    Camadas ocultas: ReLU
    Saída: 9 valores -> softmax mascarada (células ocupadas ficam com prob. 0)
    """

    def __init__(self, layers=(18, 128, 64, 9), lr=0.01, seed=None):
        rng = np.random.default_rng(seed)
        self.layers = list(layers)
        self.lr = lr
        # Inicialização He (boa para ReLU)
        self.W = [rng.normal(0, np.sqrt(2 / n_in), (n_in, n_out))
                  for n_in, n_out in zip(layers[:-1], layers[1:])]
        self.b = [np.zeros(n_out) for n_out in layers[1:]]
        # Estado do otimizador Adam
        self._t = 0
        self._m = [np.zeros_like(p) for p in self.W + self.b]
        self._v = [np.zeros_like(p) for p in self.W + self.b]

    # ---------- forward ----------
    @staticmethod
    def _masked_softmax(z, mask):
        z = np.where(mask > 0, z, -1e9)
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z) * (mask > 0)
        return e / e.sum(axis=1, keepdims=True)

    def _forward(self, X, mask):
        """Retorna as ativações de cada camada (necessárias no backprop)."""
        acts = [X]
        a = X
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = a @ W + b
            a = np.maximum(0, z) if i < len(self.W) - 1 else z  # ReLU nas ocultas
            acts.append(a)
        probs = self._masked_softmax(acts[-1], mask)
        return acts, probs

    def predict(self, X, mask):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        mask = np.atleast_2d(np.asarray(mask, dtype=float))
        return self._forward(X, mask)[1]

    # ---------- backpropagation ----------
    def _backward(self, acts, probs, Y, weights):
        """
        Perda: entropia cruzada ponderada  L = -sum_i w_i * sum_k Y_ik * log(p_ik)
        Para softmax + entropia cruzada, o gradiente na saída é (p - y).
        Depois propagamos camada por camada usando a regra da cadeia.
        """
        n = len(Y)
        delta = (probs - Y) * weights[:, None] / n           # dL/dz da saída
        grads_W, grads_b = [None] * len(self.W), [None] * len(self.b)
        for i in reversed(range(len(self.W))):
            grads_W[i] = acts[i].T @ delta                   # dL/dW = a_anterior^T · delta
            grads_b[i] = delta.sum(axis=0)
            if i > 0:
                delta = (delta @ self.W[i].T) * (acts[i] > 0)  # derivada da ReLU
        return grads_W, grads_b

    def _adam_step(self, grads, beta1=0.9, beta2=0.999, eps=1e-8):
        self._t += 1
        params = self.W + self.b
        for j, (p, g) in enumerate(zip(params, grads)):
            self._m[j] = beta1 * self._m[j] + (1 - beta1) * g
            self._v[j] = beta2 * self._v[j] + (1 - beta2) * g * g
            m_hat = self._m[j] / (1 - beta1 ** self._t)
            v_hat = self._v[j] / (1 - beta2 ** self._t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + eps)  # atualização in-place

    # ---------- fit (estilo Keras) ----------
    def trace(self, relative, legal_mask, top_edges=48):
        """
        Forward pass com tudo o que acontece dentro da rede, para a visualização:
        valores de cada camada e as conexões que mais contribuíram (|a_i * W_ij|).
        """
        x = encode(relative, legal_mask)
        acts, probs = self._forward(x[None, :], np.asarray(legal_mask, float)[None, :])
        acts = [a[0] for a in acts]
        edges = []
        for i, W in enumerate(self.W):
            contrib = acts[i][:, None] * W                     # contribuição de cada conexão
            flat = np.argsort(-np.abs(contrib), axis=None)[:top_edges]
            rows, cols = np.unravel_index(flat, contrib.shape)
            edges.append([[int(r), int(c), round(float(contrib[r, c]), 4)]
                          for r, c in zip(rows, cols) if contrib[r, c] != 0])
        names = ["input"] + [f"hidden{i}" for i in range(1, len(self.W))] + ["output"]
        return {
            "layers": [{"name": n, "size": len(a), "values": [round(float(v), 4) for v in a]}
                       for n, a in zip(names, acts)],
            "edges": edges,
            "probs": [round(float(p), 4) for p in probs[0]],
            "architecture": self.layers,
        }

    def fit(self, X, Y, mask, sample_weight=None, epochs=50, batch_size=32, verbose=True,
            on_epoch=None):
        X, Y, mask = (np.asarray(a, dtype=float) for a in (X, Y, mask))
        w = np.ones(len(X)) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        n = len(X)
        loss = 0.0
        for epoch in range(1, epochs + 1):
            order = np.random.permutation(n)
            for start in range(0, n, batch_size):
                idx = order[start:start + batch_size]
                acts, probs = self._forward(X[idx], mask[idx])
                gW, gb = self._backward(acts, probs, Y[idx], w[idx])
                self._adam_step(gW + gb)
            _, probs = self._forward(X, mask)
            loss = float(-(w * (Y * np.log(probs + 1e-12)).sum(axis=1)).mean())
            acc = float((probs.argmax(1) == Y.argmax(1)).mean())
            if on_epoch:
                on_epoch(epoch, epochs, loss, acc)
            if verbose and (epoch == 1 or epoch == epochs or epoch % max(1, epochs // 5) == 0):
                print(f"    epoch {epoch:4d}/{epochs}  loss={loss:.4f}  acerto_melhor_jogada={acc:.0%}")
        return loss

    # ---------- persistência ----------
    def save(self, path=MODEL_PATH):
        np.savez(path, layers=np.array(self.layers),
                 **{f"W{i}": W for i, W in enumerate(self.W)},
                 **{f"b{i}": b for i, b in enumerate(self.b)})

    @classmethod
    def load(cls, path=MODEL_PATH, lr=0.01):
        data = np.load(path)
        net = cls(tuple(int(x) for x in data["layers"]), lr=lr)
        net.W = [data[f"W{i}"] for i in range(len(net.W))]
        net.b = [data[f"b{i}"] for i in range(len(net.b))]
        return net


# ======================================================================
# Memória das melhores jogadas (o "modelo fit" de cada estado)
# ======================================================================
class BestMoveMemory:
    """
    Para cada estado visto guarda, por célula, a soma das recompensas e quantas
    vezes foi jogada. Assim sabemos qual foi a melhor jogada em cada estado.
    Chave do estado: o tabuleiro relativo, ex.: "1,0,-1,0,0,0,0,0,0".
    """

    def __init__(self):
        self.table: dict[str, dict[str, list[float]]] = {}  # estado -> {célula: [soma, contagem]}

    @staticmethod
    def key(relative):
        return ",".join(str(int(v)) for v in relative)

    def record(self, relative, move, reward):
        cell = self.table.setdefault(self.key(relative), {}).setdefault(str(move), [0.0, 0])
        cell[0] += reward
        cell[1] += 1

    def dataset(self, temperature=0.25):
        """
        Monta (X, Y, mask, peso) para o fit:
          - Y = distribuição "alvo" sobre as células, a partir do valor médio de cada jogada.
            Jogadas que levaram à vitória ganham probabilidade alta (reforço);
            jogadas que levaram à derrota ficam com probabilidade ~0 (punição).
            Células legais nunca testadas ficam com valor neutro 0 (exploração).
          - peso = confiança do estado (mais visitas -> mais peso).
        """
        X, Y, M, W = [], [], [], []
        for k, moves in self.table.items():
            rel = np.array([float(v) for v in k.split(",")])
            legal = (rel == 0).astype(float)
            values = np.zeros(9)
            visits = 0
            for m, (total, count) in moves.items():
                values[int(m)] = total / count
                visits += count
            z = np.where(legal > 0, values / temperature, -1e9)
            target = np.exp(z - z.max()) * legal
            target /= target.sum()
            X.append(encode(rel, legal))
            Y.append(target)
            M.append(legal)
            W.append(min(1.0 + np.log1p(visits), 4.0))
        return np.array(X), np.array(Y), np.array(M), np.array(W)

    def best_move(self, relative):
        moves = self.table.get(self.key(relative))
        if not moves:
            return None
        return max(moves.items(), key=lambda kv: kv[1][0] / kv[1][1])

    def save(self, path=MEMORY_PATH):
        with open(path, "w") as f:
            json.dump(self.table, f)

    @classmethod
    def load(cls, path=MEMORY_PATH):
        mem = cls()
        if os.path.exists(path):
            with open(path) as f:
                mem.table = json.load(f)
        return mem


# ======================================================================
# Agente: junta rede + memória + regras de recompensa/punição
# ======================================================================
def encode(relative, legal_mask):
    """Dados úteis do GET -> vetor de entrada da rede (18 valores)."""
    return np.concatenate([np.asarray(relative, float), np.asarray(legal_mask, float)])


class Reporter:
    """Envia a atividade da rede para o servidor (POST /api/nn) para a página mostrar ao vivo."""

    def __init__(self, base_url=None):
        self.base = base_url.rstrip("/") if base_url else None
        self._warned = False

    def send(self, **payload):
        if not self.base:
            return
        req = urllib.request.Request(self.base + "/api/nn", data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
        try:
            urllib.request.urlopen(req, timeout=2).read()
        except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
            if not self._warned:
                print(f"  (visualização indisponível: {e})")
                self._warned = True


class Agent:
    def __init__(self, epsilon=0.1, fit_epochs=60, reporter=None):
        self.net = NeuralNetwork.load() if os.path.exists(MODEL_PATH) else NeuralNetwork()
        self.memory = BestMoveMemory.load()
        self.epsilon = epsilon        # chance de explorar uma jogada aleatória
        self.fit_epochs = fit_epochs
        self.episode: list[tuple[list[float], int]] = []  # (estado relativo, jogada) desta partida
        self.stats = {"win": 0, "loss": 0, "draw": 0}
        self.reporter = reporter or Reporter()
        self.loss_history: list[float] = []  # loss final de cada fit

    def report_stats(self):
        self.reporter.send(stats={**self.stats, "statesKnown": len(self.memory.table),
                                  "epsilon": self.epsilon, "fits": len(self.loss_history),
                                  "lossHistory": self.loss_history[-60:]})

    def analyse(self, relative, legal_mask, mode, **extra):
        """Forward pass sem jogar (ex.: prever a sua jogada) e envio para a visualização."""
        trace = self.net.trace(relative, legal_mask)
        self.reporter.send(forward={"mode": mode, "chosen": None, **trace, **extra})
        return trace

    def choose(self, relative, legal_mask, explore=True, report=False, **extra):
        trace = self.net.trace(relative, legal_mask)
        probs = np.array(trace["probs"])
        legal = [i for i, v in enumerate(legal_mask) if v]
        explored = explore and random.random() < self.epsilon
        move = random.choice(legal) if explored else int(np.argmax(probs))
        self.episode.append((list(relative), move))
        if report:
            self.reporter.send(
                forward={"mode": "thinking", "chosen": move, "explored": explored, **trace, **extra},
                event={"type": "move", "text": f"IA joga na célula {move}"
                       + (" (exploração aleatória)" if explored else f" ({probs[move]:.0%})")})
        return move, probs

    def end_game(self, result, verbose=True, fit=True):
        """result: 'win' | 'loss' | 'draw' do ponto de vista da IA."""
        reward = {"win": REWARD_WIN, "loss": REWARD_LOSS, "draw": REWARD_DRAW}[result]
        self.stats[result] += 1
        # Recompensa com desconto: última jogada recebe a recompensa cheia,
        # jogadas anteriores recebem reward * GAMMA^distância.
        n = len(self.episode)
        for i, (relative, move) in enumerate(self.episode):
            self.memory.record(relative, move, reward * GAMMA ** (n - 1 - i))
        self.episode = []
        tag = {"win": "VITÓRIA → reforçando", "loss": "DERROTA → punindo", "draw": "EMPATE"}[result]
        if verbose:
            print(f"\n  {tag} (recompensa {reward:+.1f}). Estados conhecidos: {len(self.memory.table)}")
            self.reporter.send(event={"type": result, "text": f"{tag} ({n} jogadas, recompensa {reward:+.1f})"})
        if fit:
            self.fit(verbose=verbose)

    def fit(self, verbose=True):
        X, Y, M, W = self.memory.dataset()
        if len(X) == 0:
            return
        if verbose:
            print(f"  fit: {len(X)} estados com melhores jogadas, {self.fit_epochs} épocas")
        curve = {"loss": [], "acc": []}

        def on_epoch(epoch, epochs, loss, acc):
            curve["loss"].append(round(loss, 4))
            curve["acc"].append(round(acc, 4))
            self.reporter.send(training={"active": epoch < epochs, "epoch": epoch, "epochs": epochs,
                                         "samples": len(X), **curve})

        loss = self.net.fit(X, Y, M, sample_weight=W, epochs=self.fit_epochs, verbose=verbose,
                            on_epoch=on_epoch)
        self.loss_history.append(round(loss, 4))
        if verbose:
            self.reporter.send(event={"type": "fit", "text": f"fit: {len(X)} estados, {self.fit_epochs} "
                                      f"épocas, loss {loss:.3f}, acerto {curve['acc'][-1]:.0%}"})
        self.report_stats()
        self.save()

    def save(self):
        self.net.save()
        self.memory.save()


# ======================================================================
# Comunicação com o servidor do jogo
# ======================================================================
class GameClient:
    def __init__(self, base_url):
        self.base = base_url.rstrip("/")

    def _request(self, path, body=None):
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(self.base + path, data=data,
                                     headers={"Content-Type": "application/json"},
                                     method="GET" if body is None else "POST")
        try:
            with urllib.request.urlopen(req, timeout=5) as res:
                return json.loads(res.read())
        except urllib.error.HTTPError as e:  # 400 = jogada inválida, ainda é JSON
            return json.loads(e.read())

    def state(self):
        return self._request("/api/state")

    def send(self, body):
        return self._request("/api/input", body)


def result_for(state, ai_player):
    if state["status"] == "draw":
        return "draw"
    return "win" if state["winner"] == ai_player else "loss"


def show_probs(probs):
    rows = []
    for r in range(3):
        rows.append("   ".join(f"{probs[r * 3 + c] * 100:5.1f}%" for c in range(3)))
    return "\n    ".join(rows)


# ======================================================================
# Modos
# ======================================================================
def play(args):
    client = GameClient(args.url)
    agent = Agent(epsilon=args.epsilon, fit_epochs=args.epochs, reporter=Reporter(args.url))
    self_play = args.ai == "both"
    if args.delay is None:
        args.delay = 1.2 if self_play else 0.4
    sides = ["X", "O"] if self_play else [args.ai]
    episodes = {p: [] for p in sides}  # jogadas de cada lado controlado pela IA
    if self_play:
        print(f"IA jogando contra ela mesma em {args.url} (reinicia sozinha). Ctrl+C para sair.\n")
    else:
        print(f"IA jogando de {args.ai} em {args.url}. Jogue no navegador. Ctrl+C para sair.")
        print("Depois de cada partida clique em RESET para jogar de novo.\n")
    agent.reporter.send(event={"type": "start", "text": "IA conectada: " + (
        "jogando contra ela mesma" if self_play else f"jogando de {args.ai}")})
    agent.report_stats()

    last_game = None
    finished_game = None
    last_analysed = None
    while True:
        try:
            s = client.state()
        except (urllib.error.URLError, ConnectionError):
            print("Servidor fora do ar? Rode `npm start`. Tentando de novo...")
            time.sleep(2)
            continue

        game_key = s["gameId"]
        if game_key != last_game:  # novo jogo (reset)
            episodes = {p: [] for p in sides}
            last_game, finished_game = game_key, None
            print(f"--- Jogo #{game_key} ---")

        position_key = (game_key, s["moveCount"], tuple(s["board"]))
        if s["status"] != "playing":
            # Undo depois do fim reabre o jogo; só aprende uma vez por final
            fin = (game_key, s["moveCount"])
            if fin != finished_game:
                for p in sides:
                    if episodes[p]:
                        agent.episode = episodes[p]
                        agent.end_game(result_for(s, p), fit=(p == sides[-1]))
                        episodes[p] = []
                w, l, d = agent.stats["win"], agent.stats["loss"], agent.stats["draw"]
                print(f"  Placar da IA nesta sessão: {w}V {l}D {d}E."
                      + ("" if self_play else " Clique em RESET.") + "\n")
                agent.report_stats()
                if self_play:
                    time.sleep(args.delay * 4)
                    client.send({"action": "reset"})
            finished_game = fin
        elif s["currentPlayer"] in sides:
            player = s["currentPlayer"]
            # Mostra a rede "pensando" antes de jogar
            agent.analyse(s["nn"]["relative"], s["nn"]["legalMask"], "thinking",
                          player=player, board=s["board"])
            time.sleep(args.delay)
            agent.episode = episodes[player]
            move, probs = agent.choose(s["nn"]["relative"], s["nn"]["legalMask"], report=True,
                                       player=player, board=s["board"])
            # Envia a jogada escolhida; se outro cliente mexeu antes, descarta
            res = client.send({"position": move, "player": player})
            if res.get("ok"):
                print(f"  IA ({player}) joga na célula {move}\n    {show_probs(probs)}")
            else:
                agent.episode.pop()
        elif position_key != last_analysed:
            # Vez do humano: a rede tenta prever a sua jogada (visão de quem vai jogar)
            agent.analyse(s["nn"]["relative"], s["nn"]["legalMask"], "predicting",
                          player=s["currentPlayer"], board=s["board"])
            last_analysed = position_key
        time.sleep(0.25)


def simulate(agent, ai_player, opponent="random"):
    """Uma partida local (sem servidor) para pré-treino."""
    board = [0] * 9  # 1 = X, -1 = O
    me = 1 if ai_player == "X" else -1
    turn = 1
    while True:
        legal = [0 if v else 1 for v in board]
        if turn == me:
            rel = [v * me for v in board]
            move, _ = agent.choose(rel, legal)
        else:
            move = random.choice([i for i in range(9) if legal[i]])
        board[move] = turn
        for a, b, c in WIN_LINES:
            if board[a] and board[a] == board[b] == board[c]:
                return "win" if board[a] == me else "loss"
        if all(board):
            return "draw"
        turn = -turn


def pretrain(args):
    agent = Agent(epsilon=0.3, fit_epochs=args.epochs, reporter=Reporter(args.url))
    agent.reporter.send(event={"type": "start", "text": f"Pré-treino: {args.games} partidas"})
    print(f"Pré-treino: {args.games} partidas contra jogador aleatório "
          f"(fit a cada {args.fit_every} partidas)")
    for g in range(1, args.games + 1):
        side = random.choice("XO")
        agent.end_game(simulate(agent, side), verbose=False, fit=False)
        if g % args.fit_every == 0:
            agent.fit(verbose=False)
            w, l, d = agent.stats["win"], agent.stats["loss"], agent.stats["draw"]
            print(f"  {g:5d} partidas | V {w} D {l} E {d} | estados: {len(agent.memory.table)}")
            agent.reporter.send(event={"type": "fit", "text": f"pré-treino {g}/{args.games}: "
                                       f"{w}V {l}D {d}E, loss {agent.loss_history[-1]:.3f}"})
            agent.stats = {"win": 0, "loss": 0, "draw": 0}
    agent.fit(verbose=True)
    # Avaliação sem exploração
    agent.epsilon = 0
    res = {"win": 0, "loss": 0, "draw": 0}
    for _ in range(500):
        res[simulate(agent, random.choice("XO"))] += 1
        agent.episode = []
    print(f"\nAvaliação (500 jogos vs aleatório, sem exploração): {res}")
    print(f"Modelo salvo em {MODEL_PATH}")


def stats(_args):
    mem = BestMoveMemory.load()
    print(f"Estados conhecidos: {len(mem.table)}")
    empty = "0,0,0,0,0,0,0,0,0"
    if os.path.exists(MODEL_PATH):
        net = NeuralNetwork.load()
        rel = [0] * 9
        probs = net.predict(encode(rel, [1] * 9), [1] * 9)[0]
        print(f"Abertura preferida da rede (tabuleiro vazio):\n    {show_probs(probs)}")
    best = mem.best_move([0] * 9) if empty in mem.table else None
    if best:
        m, (total, count) = best
        print(f"Melhor abertura na memória: célula {m} (valor médio {total / count:+.2f}, {count} jogos)")


def main(argv=None):
    p = argparse.ArgumentParser(description="IA de rede neural para o NeuroTropic Tic Tac Toe")
    sub = p.add_subparsers(dest="cmd", required=True)

    pp = sub.add_parser("play", help="joga contra você pelo servidor")
    pp.add_argument("--url", default="http://localhost:3000")
    pp.add_argument("--ai", choices=["X", "O", "both"], default="O",
                    help="lado da IA (padrão O); 'both' = IA contra ela mesma, para demonstração")
    pp.add_argument("--epsilon", type=float, default=0.05, help="taxa de exploração")
    pp.add_argument("--epochs", type=int, default=60, help="épocas de fit após cada partida")
    pp.add_argument("--delay", type=float, default=None,
                    help="segundos antes de jogar (padrão 0.4; 1.2 com --ai both)")
    pp.set_defaults(func=play)

    pt = sub.add_parser("pretrain", help="treino offline contra jogador aleatório")
    pt.add_argument("--games", type=int, default=3000)
    pt.add_argument("--fit-every", type=int, default=250)
    pt.add_argument("--epochs", type=int, default=30)
    pt.add_argument("--url", default=None,
                    help="servidor para mostrar o treino ao vivo (ex.: http://localhost:3000)")
    pt.set_defaults(func=pretrain)

    ps = sub.add_parser("stats", help="mostra o que a IA aprendeu")
    ps.set_defaults(func=stats)

    args = p.parse_args(argv)
    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\nSaindo. Modelo e memória já estão salvos.")


if __name__ == "__main__":
    sys.exit(main())
