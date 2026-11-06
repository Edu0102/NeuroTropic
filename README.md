# NeuroTropic

Jogo da velha (Tic Tac Toe) 3x3 com visual sci‑fi neon, feito para ser jogado
por humanos **e** por programas (ex.: uma rede neural) via JSON.

## Rodar

```bash
npm start          # ou: node server.js  -> http://localhost:3000
npm test           # testes (node --test, sem dependências)
```

Sem Node, dá para abrir `public/index.html` direto no navegador (modo LOCAL:
o jogo roda só no navegador, sem API HTTP).

## Estrutura

| Arquivo            | Função                                                        |
|--------------------|---------------------------------------------------------------|
| `public/game.js`   | Motor do jogo puro (sem DOM): jogadas, vitória, empate, JSON  |
| `public/app.js`    | Interface; sincroniza com o servidor                          |
| `public/style.css` | Tema neon                                                     |
| `server.js`        | Servidor HTTP sem dependências + API JSON                     |

## API JSON

Tabuleiro: índices `0..8` (`index = row * 3 + col`).

```
0 | 1 | 2
3 | 4 | 5
6 | 7 | 8
```

### `GET /api/state` — estado atual

```json
{
  "gameId": 1,
  "board": ["X", null, null, null, "O", null, null, null, null],
  "grid": [["X", null, null], [null, "O", null], [null, null, null]],
  "currentPlayer": "X",
  "status": "playing",            // "playing" | "win" | "draw"
  "winner": null,                 // "X" | "O" | null
  "winningLine": null,            // ex.: [0, 4, 8]
  "moveCount": 2,
  "legalMoves": [1, 2, 3, 5, 6, 7, 8],
  "history": [{ "player": "X", "position": 0, "row": 0, "col": 0 }, ...],
  "nn": {
    "absolute":  [1, 0, 0, 0, -1, 0, 0, 0, 0],   // X=1, O=-1, vazio=0
    "relative":  [1, 0, 0, 0, -1, 0, 0, 0, 0],   // jogador da vez=1, oponente=-1
    "legalMask": [0, 1, 1, 1, 0, 1, 1, 1, 1]     // 1 = jogada permitida
  }
}
```

### `POST /api/input` — enviar uma entrada

| Corpo                                                     | Efeito                                   |
|-----------------------------------------------------------|------------------------------------------|
| `{ "position": 4 }`                                       | joga na célula 4                         |
| `{ "row": 1, "col": 2 }`                                  | joga por linha/coluna                    |
| `{ "position": 4, "player": "O" }`                        | joga validando de quem é a vez           |
| `{ "probabilities": [9 números] }`                        | joga a célula legal de maior valor (saída da rede) |
| `{ "action": "reset", "startingPlayer": "O" }`            | novo jogo                                |
| `{ "action": "undo" }`                                    | desfaz a última jogada                   |
| `{ "action": "load", "board": [1,-1,0, 0,1,0, 0,0,0] }`   | carrega uma posição (aceita `"X"/"O"/null` ou `1/-1/0`) |

Resposta: `{ "ok": true, "state": {...} }` ou `{ "ok": false, "error": "...", "state": {...} }`
(HTTP 400 em jogada inválida). Atalhos: `POST /api/move`, `/api/reset`, `/api/undo`.

```bash
curl localhost:3000/api/state
curl -X POST localhost:3000/api/input -d '{"position": 4}'
```

A página consulta o servidor a cada 500 ms, então jogadas enviadas pela API
aparecem na tela automaticamente.

### No navegador / JavaScript

```js
await NeuroTropic.getState();                 // estado JSON
await NeuroTropic.sendInput({ position: 4 }); // mesmos formatos acima
```

O motor também pode ser usado direto em Node para treinar sem HTTP:

```js
const { TicTacToe } = require("./public/game.js");
const g = new TicTacToe();
g.applyInput({ probabilities: net.predict(g.toJSON().nn.relative) });
```

## IA (rede neural em Python)

`ai/neurotropic_ai.py` joga contra você pelo servidor e aprende com cada partida.
Só depende do `numpy` (`pip install -r ai/requirements.txt`).

```bash
npm start                                          # terminal 1: servidor do jogo
python ai/neurotropic_ai.py pretrain --games 3000  # opcional: treino offline (~10 s)
python ai/neurotropic_ai.py play                   # terminal 2: IA joga de O, você de X no navegador
python ai/neurotropic_ai.py play --ai X            # IA começa
python ai/neurotropic_ai.py stats                  # o que a IA já aprendeu
```

Como funciona:

1. **Entrada:** do `GET /api/state` usa `nn.relative` e `nn.legalMask` (18 números).
2. **Rede:** MLP 18 → 128 → 64 → 9 (ReLU, softmax mascarada), escrita em NumPy.
   A saída são 9 probabilidades, enviadas como `{"probabilities": [...]}`, e o jogo escolhe a célula.
3. **Recompensa:** vitória `+1` (reforça), empate `+0.5`, derrota `-1` (pune).
   Cada jogada da partida recebe `recompensa × 0.8^(distância até o fim)`.
4. **Melhores jogadas:** uma memória guarda, para cada estado do tabuleiro, o valor
   médio de cada jogada. Disso sai uma distribuição alvo por estado (jogadas boas
   com probabilidade alta, ruins perto de zero).
5. **Fit:** `model.fit(X, Y, ...)` treina a rede nessas melhores jogadas com
   backpropagation (entropia cruzada + Adam), depois de cada partida.

Modelo e memória ficam em `ai/model.npz` e `ai/memory.json` (fora do git); apague-os para recomeçar do zero.
