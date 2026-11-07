/**
 * NEURAL CORE — visualização ao vivo da rede neural (ai/neurotropic_ai.py).
 *
 * O script Python envia cada forward pass e o progresso do fit para
 * POST /api/nn; esta página lê GET /api/nn e desenha:
 *   entradas (18) -> oculta 1 -> oculta 2 -> saída (9)
 * com as conexões que mais contribuíram para a decisão.
 */
(() => {
  "use strict";

  const POLL_MS = 300;
  const WAVE_MS = 220;         // duração da onda em cada transição de camada
  const STALE_MS = 20000;      // sem notícias da IA por esse tempo = ociosa
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  const $ = (id) => document.getElementById(id);
  const canvas = $("nn-canvas");
  const ctx = canvas.getContext("2d");
  const tooltip = $("nn-tooltip");

  const css = getComputedStyle(document.documentElement);
  const C = {
    cyan: css.getPropertyValue("--cyan").trim() || "#00f0ff",
    magenta: css.getPropertyValue("--magenta").trim() || "#ff2bd6",
    lime: css.getPropertyValue("--lime").trim() || "#9dff00",
    amber: css.getPropertyValue("--amber").trim() || "#ffcc00",
    muted: css.getPropertyValue("--muted").trim() || "#7a8bb5",
    text: css.getPropertyValue("--text").trim() || "#d8f7ff",
  };
  const rgba = (hex, a) => {
    const n = parseInt(hex.slice(1), 16);
    return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, ${a})`;
  };

  let data = null;           // último GET /api/nn
  let forwardKey = "";       // detecta um forward pass novo
  let waveStart = -1e9;
  let geometry = null;       // posições calculadas dos neurônios
  let online = false;

  // ================= dados de exemplo enquanto a IA não conecta =================
  function placeholderForward() {
    const arch = [18, 128, 64, 9];
    return {
      mode: "offline",
      architecture: arch,
      layers: arch.map((n, i) => ({
        name: ["input", "hidden1", "hidden2", "output"][i],
        size: n,
        values: Array(n).fill(0).map((_, j) => (i === 0 && j >= 9 ? 1 : 0)),
      })),
      edges: [[], [], []],
      probs: Array(9).fill(0),
      chosen: null,
      board: Array(9).fill(null),
    };
  }

  // ================= layout =================
  function gridShape(layerIdx, n, last, vertical) {
    if (layerIdx === 0 && n === 18) {
      // tabuleiro (9) + máscara de jogadas legais (9)
      return vertical ? { cols: 6, rows: 3, gapCol: 3 } : { cols: 3, rows: 6, gapRow: 3 };
    }
    if (layerIdx === last && n === 9) return { cols: 3, rows: 3 };
    const sq = Math.sqrt(n);
    if (Number.isInteger(sq)) return { cols: sq, rows: sq };
    const a = Math.ceil(Math.sqrt(n / 2));
    const b = Math.ceil(n / a);
    return vertical ? { cols: b, rows: a } : { cols: a, rows: b };
  }

  function computeGeometry(fwd) {
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    const vertical = w < 640;
    const sizes = fwd.layers.map((l) => l.size);
    const last = sizes.length - 1;
    const weights = sizes.map((n, i) => (i === 0 || i === last ? 1 : 1.35));
    const total = weights.reduce((a, b) => a + b, 0);
    const labelSpace = 34;
    const pad = 18;

    let offset = pad;
    const span = (vertical ? h : w) - pad * 2;
    const layers = sizes.map((n, i) => {
      const len = (span * weights[i]) / total;
      const rect = vertical
        ? { x: pad, y: offset, w: w - pad * 2, h: len - labelSpace }
        : { x: offset, y: pad, w: len, h: h - pad * 2 - labelSpace };
      offset += len;

      const shape = gridShape(i, n, last, vertical);
      const gapUnits = shape.gapRow || shape.gapCol ? 0.8 : 0;
      const unitsX = shape.cols + (shape.gapCol ? gapUnits : 0);
      const unitsY = shape.rows + (shape.gapRow ? gapUnits : 0);
      const pitch = Math.min((rect.w * 0.82) / unitsX, (rect.h * 0.92) / unitsY, 46);
      const gw = unitsX * pitch;
      const gh = unitsY * pitch;
      const ox = rect.x + (rect.w - gw) / 2;
      const oy = rect.y + (rect.h - gh) / 2;

      const nodes = [];
      for (let j = 0; j < n; j++) {
        let r, c;
        if (i === 0 && n === 18) {
          // célula k do tabuleiro (j<9) ou da máscara (j>=9)
          const k = j % 9, part = j < 9 ? 0 : 1;
          r = Math.floor(k / 3); c = k % 3;
          if (vertical) c += part * 3; else r += part * 3;
        } else {
          r = Math.floor(j / shape.cols); c = j % shape.cols;
        }
        const gx = c + (shape.gapCol && c >= shape.gapCol ? gapUnits : 0);
        const gy = r + (shape.gapRow && r >= shape.gapRow ? gapUnits : 0);
        nodes.push({ x: ox + (gx + 0.5) * pitch, y: oy + (gy + 0.5) * pitch, s: pitch * 0.78 });
      }
      const label = vertical
        ? { x: w / 2, y: rect.y + rect.h + 16 }
        : { x: rect.x + rect.w / 2, y: rect.y + rect.h + 22 };
      return { nodes, label, pitch, shape, box: { x: ox, y: oy, w: gw, h: gh } };
    });
    return { w, h, vertical, layers };
  }

  // ================= desenho da rede =================
  function layerLabel(i, n, last) {
    if (i === 0) return `ENTRADA · ${n}`;
    if (i === last) return `SAÍDA · ${n} softmax`;
    return `OCULTA ${i} · ${n} ReLU`;
  }

  function roundRect(x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r);
    ctx.arcTo(x, y, x + w, y, r);
    ctx.closePath();
  }

  const hash = (a, b, c) => ((a * 73856093) ^ (b * 19349663) ^ (c * 83492791)) >>> 0;

  function draw(now) {
    const fwd = (data && data.forward) || placeholderForward();
    geometry = computeGeometry(fwd);
    const { w, h, layers } = geometry;
    const last = layers.length - 1;
    ctx.clearRect(0, 0, w, h);

    const elapsed = reduceMotion ? 1e9 : now - waveStart;
    const progress = (k) => Math.min(1, Math.max(0, elapsed / WAVE_MS - k)); // 0..1 por transição

    // ---- conexões ----
    ctx.globalCompositeOperation = "lighter";
    (fwd.edges || []).forEach((list, k) => {
      if (!list.length || !layers[k + 1]) return;
      const max = Math.max(...list.map((e) => Math.abs(e[2]))) || 1;
      const p = progress(k);
      list.forEach(([from, to, v], idx) => {
        const a = layers[k].nodes[from], b = layers[k + 1].nodes[to];
        if (!a || !b) return;
        const strength = Math.abs(v) / max;
        const color = v >= 0 ? C.cyan : C.magenta;
        ctx.strokeStyle = rgba(color, 0.06 + 0.4 * strength * (0.35 + 0.65 * p));
        ctx.lineWidth = 0.5 + 1.6 * strength;
        ctx.beginPath();
        ctx.moveTo(a.x, a.y);
        ctx.lineTo(b.x, b.y);
        ctx.stroke();

        if (reduceMotion) return;
        // pulso da onda (forward pass) e pulsos contínuos mais fracos
        let t = null, alpha = 0;
        if (p > 0 && p < 1) { t = p; alpha = 0.9; }
        else if (strength > 0.35) {
          t = ((now / 1600) + (hash(k, from, to) % 1000) / 1000) % 1;
          alpha = 0.35 * strength;
        }
        if (t !== null) {
          const x = a.x + (b.x - a.x) * t, y = a.y + (b.y - a.y) * t;
          ctx.fillStyle = rgba(color, alpha);
          ctx.beginPath();
          ctx.arc(x, y, 1.4 + 1.6 * strength, 0, Math.PI * 2);
          ctx.fill();
        }
      });
    });
    ctx.globalCompositeOperation = "source-over";

    // ---- neurônios ----
    layers.forEach((L, i) => {
      const values = i === last && fwd.probs ? fwd.probs : fwd.layers[i].values;
      const maxAct = i > 0 && i < last ? Math.max(...values, 1e-6) : 1;
      const reveal = i === 0 ? 1 : 0.3 + 0.7 * progress(i - 1); // nunca fica totalmente apagada
      L.nodes.forEach((nd, j) => {
        const v = values[j] || 0;
        const half = nd.s / 2;
        let fill, glow = 0, stroke = "rgba(122, 139, 181, 0.35)";
        if (i === 0) {
          if (j < 9) {
            fill = v > 0 ? C.cyan : v < 0 ? C.magenta : "rgba(255,255,255,0.04)";
            if (v !== 0) glow = 10;
          } else {
            fill = v > 0 ? rgba(C.lime, 0.45) : "rgba(255,255,255,0.03)";
          }
        } else if (i === last) {
          const occupied = fwd.board && fwd.board[j];
          fill = occupied ? "rgba(255,255,255,0.03)" : rgba(C.lime, 0.08 + 0.85 * v * reveal);
          glow = v > 0.25 ? 14 * v * reveal : 0;
          if (fwd.chosen === j && reveal >= 1) stroke = C.lime;
        } else {
          const a = (v / maxAct) * reveal;
          fill = a > 0.01 ? rgba("#7ff8ff", 0.12 + 0.85 * a) : "rgba(255,255,255,0.035)";
          glow = a > 0.55 ? 10 * a : 0;
        }
        // base opaca: as conexões não aparecem por dentro das células
        ctx.fillStyle = "#0a0420";
        roundRect(nd.x - half, nd.y - half, nd.s, nd.s, Math.min(6, nd.s * 0.22));
        ctx.fill();
        ctx.shadowColor = i === last ? C.lime : (i === 0 && j < 9 && v < 0 ? C.magenta : C.cyan);
        ctx.shadowBlur = glow;
        ctx.fillStyle = fill;
        roundRect(nd.x - half, nd.y - half, nd.s, nd.s, Math.min(6, nd.s * 0.22));
        ctx.fill();
        ctx.shadowBlur = 0;
        ctx.strokeStyle = stroke;
        ctx.lineWidth = stroke === C.lime ? 2.5 : 1;
        ctx.stroke();

        // textos dentro das células grandes
        if (i === last && nd.s > 26) {
          const piece = fwd.board && fwd.board[j];
          ctx.textAlign = "center"; ctx.textBaseline = "middle";
          if (piece) {
            ctx.font = `800 ${Math.round(nd.s * 0.45)}px Orbitron, sans-serif`;
            ctx.fillStyle = piece === "X" ? C.cyan : C.magenta;
            ctx.fillText(piece, nd.x, nd.y + 1);
          } else {
            ctx.font = `${Math.round(nd.s * 0.26)}px "Share Tech Mono", monospace`;
            ctx.fillStyle = C.text;
            ctx.fillText(`${Math.round(v * 100)}%`, nd.x, nd.y + 1);
          }
        }
        if (i === 0 && j < 9 && v !== 0 && nd.s > 16) {
          ctx.textAlign = "center"; ctx.textBaseline = "middle";
          ctx.font = `700 ${Math.round(nd.s * 0.5)}px "Share Tech Mono", monospace`;
          ctx.fillStyle = "#05010f";
          ctx.fillText(v > 0 ? "+" : "−", nd.x, nd.y + 1);
        }
      });

      // rótulo da camada
      ctx.textAlign = "center"; ctx.textBaseline = "alphabetic";
      ctx.font = `500 11px Orbitron, sans-serif`;
      ctx.fillStyle = C.muted;
      ctx.fillText(layerLabel(i, fwd.layers[i].size, last), L.label.x, L.label.y);
      if (i === 0) {
        const { box } = L;
        ctx.font = `10px "Share Tech Mono", monospace`;
        ctx.fillStyle = rgba(C.muted, 0.8);
        if (geometry.vertical) {
          ctx.fillText("tabuleiro", box.x + box.w * 0.23, box.y - 6);
          ctx.fillText("legais", box.x + box.w * 0.77, box.y - 6);
        } else {
          ctx.fillText("tabuleiro", box.x + box.w / 2, box.y - 6);
          ctx.fillText("jogadas legais", box.x + box.w / 2, box.y + 3.4 * L.pitch + 4);
        }
      }
    });

    requestAnimationFrame(draw);
  }

  // ---- tooltip dos neurônios ----
  canvas.addEventListener("mousemove", (e) => {
    if (!geometry) return;
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left, my = e.clientY - rect.top;
    const fwd = (data && data.forward) || placeholderForward();
    const last = geometry.layers.length - 1;
    let hit = null;
    geometry.layers.forEach((L, i) => L.nodes.forEach((nd, j) => {
      const r = Math.max(nd.s / 2, 6); // alvo maior que a marca
      if (Math.abs(mx - nd.x) <= r && Math.abs(my - nd.y) <= r) hit = { i, j, nd };
    }));
    if (!hit) { tooltip.hidden = true; return; }
    const { i, j, nd } = hit;
    const v = fwd.layers[i].values[j];
    let text;
    if (i === 0) {
      text = j < 9
        ? `Entrada ${j} · célula ${j} = ${v > 0 ? "+1 (quem joga)" : v < 0 ? "−1 (oponente)" : "0 (vazia)"}`
        : `Entrada ${j} · célula ${j - 9} ${v ? "permitida" : "bloqueada"}`;
    } else if (i === last) {
      text = `Saída · célula ${j} = ${((fwd.probs[j] || 0) * 100).toFixed(1)}% (logit ${(+v).toFixed(2)})`;
    } else {
      text = `Oculta ${i} · neurônio ${j} · ativação ${(+v).toFixed(3)}`;
    }
    tooltip.textContent = text;
    tooltip.hidden = false;
    const left = Math.min(nd.x + 12, rect.width - tooltip.offsetWidth - 4);
    tooltip.style.left = `${Math.max(4, left)}px`;
    tooltip.style.top = `${nd.y - 34}px`;
  });
  canvas.addEventListener("mouseleave", () => (tooltip.hidden = true));

  // ================= painéis HTML =================
  const outputEl = $("nn-output");
  const outCells = [...Array(9)].map(() => {
    const el = document.createElement("div");
    el.className = "out-cell";
    el.innerHTML = `<div class="fill"></div><span class="piece"></span><span class="pct"></span>`;
    outputEl.appendChild(el);
    return el;
  });

  function renderOutput(fwd) {
    const probs = (fwd && fwd.probs) || Array(9).fill(0);
    const board = (fwd && fwd.board) || Array(9).fill(null);
    outCells.forEach((el, i) => {
      const piece = board[i];
      el.className = "out-cell" + (piece ? " " + piece.toLowerCase() : "") +
        (fwd && fwd.chosen === i ? " chosen" : "");
      el.querySelector(".fill").style.height = piece ? "0%" : `${probs[i] * 100}%`;
      el.querySelector(".piece").textContent = piece || "";
      el.querySelector(".pct").textContent = piece ? "" : `${(probs[i] * 100).toFixed(1)}%`;
      el.title = `Célula ${i}`;
    });

    const dec = $("nn-decision");
    if (!fwd) { dec.textContent = "—"; return; }
    const best = probs.indexOf(Math.max(...probs));
    if (fwd.mode === "predicting") {
      dec.textContent = `Prevendo a jogada de ${fwd.player}: célula ${best} (${(probs[best] * 100).toFixed(0)}%)`;
    } else if (fwd.chosen !== null && fwd.chosen !== undefined) {
      dec.textContent = `IA (${fwd.player}) escolheu a célula ${fwd.chosen}` +
        (fwd.explored ? " · exploração aleatória" : ` · ${(probs[fwd.chosen] * 100).toFixed(0)}%`);
    } else {
      dec.textContent = `IA (${fwd.player}) pensando…`;
    }
  }

  function renderMode() {
    const badge = $("nn-mode");
    let cls = "offline", label = online ? "AGUARDANDO IA" : "SEM SERVIDOR";
    if (data && data.updatedAt) {
      const stale = Date.now() - data.updatedAt > STALE_MS;
      if (data.training && data.training.active && !stale) { cls = "training"; label = "TREINANDO"; }
      else if (stale) { cls = "idle"; label = "OCIOSA"; }
      else if (data.forward && data.forward.mode === "predicting") { cls = "predicting"; label = "PREVENDO SUA JOGADA"; }
      else if (data.forward && data.forward.mode === "thinking") {
        cls = "thinking"; label = data.forward.chosen != null ? "JOGOU" : "PENSANDO";
      } else { cls = "idle"; label = "ONLINE"; }
    }
    badge.className = `badge ${cls}`;
    badge.textContent = label;
    $("nn-hint").hidden = !!(data && data.updatedAt);
    if (!online) {
      $("nn-hint").innerHTML = "Sem servidor. Rode <code>npm start</code> e depois " +
        "<code>python ai/neurotropic_ai.py play</code> (ou <code>play --ai both</code>).";
    }
  }

  // ---- gráficos de linha (uma série por gráfico, hover com tooltip) ----
  const charts = {};
  document.querySelectorAll(".chart").forEach((fig) => {
    const c = fig.querySelector("canvas");
    const tip = document.createElement("div");
    tip.className = "tooltip"; tip.hidden = true;
    fig.appendChild(tip);
    const chart = { canvas: c, tip, values: [], opts: {}, hover: null };
    c.addEventListener("mousemove", (e) => {
      const r = c.getBoundingClientRect();
      chart.hover = e.clientX - r.left;
      drawChart(chart);
    });
    c.addEventListener("mouseleave", () => { chart.hover = null; tip.hidden = true; drawChart(chart); });
    charts[fig.dataset.chart] = chart;
  });

  function drawChart(chart) {
    const { canvas: c, values, opts, tip } = chart;
    const dpr = window.devicePixelRatio || 1;
    const w = c.clientWidth, h = c.clientHeight;
    c.width = Math.round(w * dpr); c.height = Math.round(h * dpr);
    const g = c.getContext("2d");
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, w, h);

    const left = 38, right = 34, top = 8, bottom = 14;
    const pw = w - left - right, ph = h - top - bottom;
    g.font = `10px "Share Tech Mono", monospace`;

    if (!values.length) {
      g.fillStyle = C.muted; g.textAlign = "center";
      g.fillText("sem dados ainda", w / 2, h / 2);
      return;
    }
    let lo = opts.min ?? Math.min(...values), hi = opts.max ?? Math.max(...values);
    if (hi - lo < 1e-6) { hi += 0.01; lo -= 0.01; }
    const x = (i) => left + (values.length === 1 ? pw / 2 : (i / (values.length - 1)) * pw);
    const y = (v) => top + (1 - (v - lo) / (hi - lo)) * ph;

    // grade e eixo recessivos
    g.strokeStyle = "rgba(122, 139, 181, 0.18)"; g.lineWidth = 1;
    [lo, hi].forEach((v) => { g.beginPath(); g.moveTo(left, y(v)); g.lineTo(left + pw, y(v)); g.stroke(); });
    g.fillStyle = C.muted; g.textAlign = "right"; g.textBaseline = "middle";
    g.fillText(opts.format(hi), left - 5, y(hi));
    g.fillText(opts.format(lo), left - 5, y(lo));
    g.textAlign = "left"; g.textBaseline = "alphabetic";
    g.fillText(opts.xLabel(1), left, h - 2);
    g.textAlign = "right";
    g.fillText(opts.xLabel(values.length), left + pw, h - 2);

    // linha de 2px
    g.strokeStyle = opts.color; g.lineWidth = 2; g.lineJoin = "round";
    g.shadowColor = opts.color; g.shadowBlur = 6;
    g.beginPath();
    values.forEach((v, i) => (i ? g.lineTo(x(i), y(v)) : g.moveTo(x(i), y(v))));
    g.stroke();
    g.shadowBlur = 0;

    // último ponto com rótulo direto
    const li = values.length - 1;
    g.fillStyle = opts.color;
    g.beginPath(); g.arc(x(li), y(values[li]), 4, 0, Math.PI * 2); g.fill();
    g.fillStyle = C.text; g.textAlign = "left"; g.textBaseline = "middle";
    g.fillText(opts.format(values[li]), x(li) + 7, y(values[li]));

    // hover: crosshair + tooltip
    if (chart.hover !== null) {
      const i = Math.round(((chart.hover - left) / pw) * (values.length - 1));
      const idx = Math.max(0, Math.min(values.length - 1, isNaN(i) ? 0 : i));
      g.strokeStyle = "rgba(216, 247, 255, 0.35)"; g.lineWidth = 1;
      g.beginPath(); g.moveTo(x(idx), top); g.lineTo(x(idx), top + ph); g.stroke();
      g.fillStyle = opts.color;
      g.beginPath(); g.arc(x(idx), y(values[idx]), 4, 0, Math.PI * 2); g.fill();
      tip.textContent = `${opts.xLabel(idx + 1)} · ${opts.format(values[idx])}`;
      tip.hidden = false;
      tip.style.left = `${Math.min(Math.max(0, x(idx) - tip.offsetWidth / 2), w - tip.offsetWidth)}px`;
      tip.style.top = `${-8}px`;
    }
  }

  function renderTraining(tr, stats) {
    const bar = $("fit-bar"), label = $("fit-label");
    if (tr) {
      bar.style.width = `${(tr.epoch / tr.epochs) * 100}%`;
      const acc = tr.acc[tr.acc.length - 1], loss = tr.loss[tr.loss.length - 1];
      label.textContent = `${tr.active ? "Treinando" : "Último fit"}: época ${tr.epoch}/${tr.epochs} · ` +
        `${tr.samples} estados · loss ${loss.toFixed(3)} · acerto ${(acc * 100).toFixed(0)}%`;
    }
    const fmt3 = (v) => v.toFixed(3);
    Object.assign(charts.loss, { values: (tr && tr.loss) || [],
      opts: { color: C.amber, format: fmt3, xLabel: (n) => `época ${n}` } });
    Object.assign(charts.acc, { values: (tr && tr.acc) || [],
      opts: { color: C.lime, min: 0, max: 1, format: (v) => `${Math.round(v * 100)}%`, xLabel: (n) => `época ${n}` } });
    Object.assign(charts.history, { values: (stats && stats.lossHistory) || [],
      opts: { color: C.cyan, format: fmt3, xLabel: (n) => `fit ${n}` } });
    Object.values(charts).forEach(drawChart);
  }

  function renderStats(s) {
    if (!s) return;
    $("st-win").textContent = s.win;
    $("st-loss").textContent = s.loss;
    $("st-draw").textContent = s.draw;
    $("st-states").textContent = s.statesKnown;
  }

  let lastEventsKey = "";
  function renderEvents(events) {
    const key = events.length ? events[events.length - 1].t + ":" + events.length : "";
    if (key === lastEventsKey) return;
    lastEventsKey = key;
    const ol = $("nn-events");
    ol.innerHTML = "";
    events.slice().reverse().forEach((ev) => {
      const li = document.createElement("li");
      li.className = ev.type || "";
      const time = document.createElement("time");
      time.textContent = new Date(ev.t).toLocaleTimeString();
      li.append(time, document.createTextNode(ev.text));
      ol.appendChild(li);
    });
  }

  // ================= polling =================
  let lastSeq = -1;
  async function poll() {
    try {
      const res = await fetch("/api/nn", { cache: "no-store" });
      if (!res.ok) throw new Error(res.status);
      const next = await res.json();
      online = true;
      if (next.seq !== lastSeq) {
        lastSeq = next.seq;
        data = next;
        const f = next.forward;
        // a chegada da jogada escolhida só destaca a célula; a onda roda para um forward novo
        const key = f ? JSON.stringify([f.board, f.probs]) : "";
        if (key !== forwardKey) { forwardKey = key; waveStart = performance.now(); }
        renderOutput(f);
        renderTraining(next.training, next.stats);
        renderStats(next.stats);
        renderEvents(next.events || []);
        if (f && f.architecture) $("nn-arch").textContent = f.architecture.join(" → ");
      }
    } catch (_) {
      online = false;
    }
    renderMode();
  }

  renderOutput(null);
  renderTraining(null, null);
  window.addEventListener("resize", () => Object.values(charts).forEach(drawChart));
  requestAnimationFrame(draw);
  if (location.protocol === "file:") { online = false; renderMode(); }
  else { poll(); setInterval(poll, POLL_MS); }
})();
