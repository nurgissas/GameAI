"""Playable MNK/Gomoku UI with online Q-value based DDA.

Usage:
    python experiments/play_mnk.py 15 15 5

The three positional numbers are board width, board height, and k-in-a-row.
After each game, the player's moves are converted into an estimated difficulty
using Q-value percentiles, and the DDA controller updates the next opponent
difficulty level.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.mnk_game import OPPONENT, PLAYER, _check_winner, legal_moves
from rl_training.difficulty_scaling import (
    QLearningDDA,
    QValueDDAMetrics,
    QValueDifficultyEstimator,
    RewardConfig,
)
from rl_training.mnk_q_agent import (
    EPSILON_LEVELS,
    MNKQLearningAgent,
    build_opponent_pool,
    train_mnk_agent,
)


ROOT = Path(__file__).resolve().parents[1]


def has_valid_model(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        agent = MNKQLearningAgent()
        agent.load(str(path))
        return agent.is_healthy()
    except Exception as exc:
        print(f"Ignoring invalid trained model at {path}: {exc}")
        try:
            path.unlink()
            print(f"Removed invalid model: {path}")
        except OSError as remove_exc:
            print(f"Could not remove invalid model: {remove_exc}")
        return False


def model_path(width: int, height: int, k: int) -> Path:
    if width == height:
        return ROOT / "agents" / f"mnk_{width}x{width}_k{k}_trained.pkl"
    return ROOT / "agents" / f"mnk_{width}x{height}_k{k}_trained.pkl"


def train_rectangular_agent(
    width: int,
    height: int,
    k: int,
    episodes: int,
    save_path: Path,
    use_tactical_rules: bool,
) -> None:
    """Train and save an MNKQLearningAgent for rectangular boards."""
    from envs.mnk_game import MNKGame

    game = MNKGame(height, width, k)
    agent = MNKQLearningAgent(
        learning_rate=0.01,
        epsilon=0.3,
        use_tactical_rules=use_tactical_rules,
    )
    log_interval = max(episodes // 5, 1)

    print(f"Training MNK({height},{width},{k}) for {episodes:,} episodes")
    print(f"Tactical rules: {'on' if use_tactical_rules else 'off'}")
    print("-" * 60)
    for episode in range(episodes):
        agent.epsilon = 0.3 - 0.25 * (episode / max(episodes, 1))
        agent.train_episode(game)
        if (episode + 1) % log_interval == 0:
            print(
                f"  Episode {episode + 1:,} / {episodes:,} | "
                f"Win rate: {agent.win_rate():.1%} | "
                f"Clipped: {agent.stats['clipped_updates']:,} | "
                f"Skipped: {agent.stats['skipped_updates']:,}"
            )

    agent.save(str(save_path))
    print("-" * 60)
    print(f"Done. Saved to {save_path}")


class PlaySession:
    def __init__(
        self,
        width: int,
        height: int,
        k: int,
        opponent_pool: list[MNKQLearningAgent],
        seed: int,
    ) -> None:
        self.width = width
        self.height = height
        self.k = k
        self.opponent_pool = opponent_pool
        self.rng = random.Random(seed)
        self.estimator = QValueDifficultyEstimator(opponent_pool)
        self.metrics = QValueDDAMetrics(
            window=8,
            reward_config=RewardConfig(),
            n_levels=len(opponent_pool),
        )
        self.dda = QLearningDDA(
            n_levels=len(opponent_pool),
            rng=self.rng,
            epsilon=0.20,
            min_epsilon=0.04,
        )
        self.history: list[dict] = []
        self.ai_starts = False
        self.manual_difficulty: int | None = None
        self.lock = threading.Lock()
        self.reset_match()

    def reset_match(self) -> None:
        state = self.metrics.state()
        self.current_state = state
        self.difficulty = self._select_starting_difficulty(state)
        self.board = [[0] * self.width for _ in range(self.height)]
        self.moves = 0
        self.game_over = False
        self.winner = None
        self.player_estimates: list[float] = []
        self.player_qualities: list[float] = []
        self.last_update: dict | None = None
        if self.ai_starts:
            self._play_ai_turn()

    def snapshot(self) -> dict:
        eps = self.opponent_pool[self.difficulty].epsilon
        return {
            "width": self.width,
            "height": self.height,
            "k": self.k,
            "board": self.board,
            "moves": self.moves,
            "gameOver": self.game_over,
            "winner": self.winner,
            "difficultyIndex": self.difficulty,
            "difficultyLevel": self.difficulty + 1,
            "nLevels": len(self.opponent_pool),
            "epsilon": eps,
            "estimatedDifficulty": self._mean(self.player_estimates, 0.0),
            "moveQuality": self._mean(self.player_qualities, 0.0),
            "lastUpdate": self.last_update,
            "history": self.history[-8:],
            "ddaEpsilon": self.dda.epsilon,
            "aiStarts": self.ai_starts,
            "manualDifficulty": (
                None if self.manual_difficulty is None else self.manual_difficulty + 1
            ),
        }

    def play_human_move(self, row: int, col: int) -> dict:
        with self.lock:
            if self.game_over:
                return {"ok": False, "error": "game is already over", "state": self.snapshot()}
            if not (0 <= row < self.height and 0 <= col < self.width):
                return {"ok": False, "error": "move is outside the board", "state": self.snapshot()}
            if self.board[row][col] != 0:
                return {"ok": False, "error": "cell is occupied", "state": self.snapshot()}

            board_before = [r[:] for r in self.board]
            obs = self.estimator.estimate_move(
                board_before,
                (row, col),
                PLAYER,
                OPPONENT,
                self.k,
            )
            self.player_estimates.append(obs["difficulty"])
            self.player_qualities.append(obs["quality"])

            self._place((row, col), PLAYER)
            if self._finish_if_terminal((row, col), PLAYER):
                return {"ok": True, "state": self.snapshot()}

            ai_move = self._play_ai_turn()
            return {"ok": True, "aiMove": ai_move, "state": self.snapshot()}

    def next_game(self) -> dict:
        with self.lock:
            self.reset_match()
            return self.snapshot()

    def set_ai_starts(self, ai_starts: bool) -> dict:
        with self.lock:
            self.ai_starts = ai_starts
            self.reset_match()
            return self.snapshot()

    def set_difficulty(self, level: int | None) -> dict:
        with self.lock:
            if level is None:
                self.manual_difficulty = None
                self.difficulty = self.dda.select_difficulty(self.metrics.state())
            else:
                self.manual_difficulty = max(0, min(level - 1, len(self.opponent_pool) - 1))
                self.difficulty = self.manual_difficulty
            return self.snapshot()

    def _select_starting_difficulty(self, state) -> int:
        if self.manual_difficulty is not None:
            return self.manual_difficulty
        return self.dda.select_difficulty(state)

    def _play_ai_turn(self) -> tuple[int, int]:
        ai_move = self.opponent_pool[self.difficulty].choose_move(
            self.board,
            OPPONENT,
            PLAYER,
            self.k,
        )
        self._place(ai_move, OPPONENT)
        self._finish_if_terminal(ai_move, OPPONENT)
        return ai_move

    def _place(self, move: tuple[int, int], marker: int) -> None:
        row, col = move
        self.board[row][col] = marker
        self.moves += 1

    def _finish_if_terminal(self, move: tuple[int, int], marker: int) -> bool:
        winner = None
        if _check_winner(self.board, self.height, self.width, self.k, move, marker):
            winner = marker
        elif not legal_moves(self.board):
            winner = 0

        if winner is None:
            return False

        self.game_over = True
        self.winner = winner
        result = SimpleResult(
            winner=winner,
            moves=self.moves,
            normalized_duration=self.moves / float(self.width * self.height),
            estimated_player_difficulty=self._mean(self.player_estimates, 0.0),
            player_move_quality=self._mean(self.player_qualities, 0.0),
        )
        update = self.metrics.update(result, self.difficulty)
        next_state = self.metrics.state()
        self.dda.observe(self.current_state, self.difficulty, update["reward"], next_state)
        self.last_update = {
            "playedDifficulty": self.difficulty + 1,
            "nextState": next_state,
            "winner": winner,
            "moves": self.moves,
            "estimatedPlayerDifficulty": update["estimated_difficulty"],
            "playerMoveQuality": update["player_move_quality"],
            "difficultyError": update["difficulty_error"],
            "duration": update["duration"],
            "reward": update["reward"],
        }
        self.history.append(self.last_update)
        return True

    @staticmethod
    def _mean(values: list[float], default: float) -> float:
        return sum(values) / len(values) if values else default


class SimpleResult:
    def __init__(
        self,
        winner: int,
        moves: int,
        normalized_duration: float,
        estimated_player_difficulty: float,
        player_move_quality: float,
    ) -> None:
        self.winner = winner
        self.moves = moves
        self.normalized_duration = normalized_duration
        self.estimated_player_difficulty = estimated_player_difficulty
        self.player_move_quality = player_move_quality


HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Playable MNK DDA</title>
  <style>
    :root {
      --bg: #f5f1e7;
      --board: #c9944f;
      --line: #3c2a17;
      --panel: #ffffff;
      --ink: #172026;
      --muted: #6b7280;
      --human: #1d4ed8;
      --ai: #b91c1c;
      --accent: #237857;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    main {
      display: grid;
      grid-template-columns: minmax(360px, 760px) minmax(280px, 380px);
      gap: 24px;
      padding: 24px;
      align-items: start;
      justify-content: center;
    }
    #board {
      display: grid;
      width: min(88vmin, 760px);
      aspect-ratio: var(--cols) / var(--rows);
      background: var(--line);
      border: 2px solid var(--line);
      border-radius: 8px;
      gap: 1px;
      overflow: hidden;
    }
    .cell {
      min-width: 0;
      min-height: 0;
      border: 0;
      background: var(--board);
      display: grid;
      place-items: center;
      cursor: pointer;
      padding: 0;
    }
    .cell:disabled { cursor: default; }
    .stone {
      width: 72%;
      height: 72%;
      border-radius: 50%;
      box-shadow: inset 0 2px 4px rgba(255,255,255,.35), inset 0 -3px 6px rgba(0,0,0,.24);
    }
    .human { background: var(--human); }
    .ai { background: var(--ai); }
    aside {
      background: var(--panel);
      border: 1px solid rgba(23,32,38,.14);
      border-radius: 8px;
      padding: 18px;
      box-shadow: 0 8px 28px rgba(23,32,38,.08);
    }
    h1 { margin: 0 0 4px; font-size: 22px; }
    .sub { color: var(--muted); margin-bottom: 14px; }
    .metrics {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
      margin-bottom: 14px;
    }
    .metric {
      border-top: 1px solid rgba(23,32,38,.14);
      padding-top: 8px;
      min-width: 0;
    }
    .metric b { display: block; font-size: 20px; overflow-wrap: anywhere; }
    .metric span { color: var(--muted); font-size: 12px; }
    .actions { display: flex; gap: 8px; margin-bottom: 14px; }
    button.action {
      border: 1px solid rgba(23,32,38,.18);
      border-radius: 6px;
      background: #fff;
      min-height: 38px;
      padding: 0 12px;
      cursor: pointer;
    }
    button.primary { background: var(--accent); color: #fff; border-color: var(--accent); }
    .controls {
      display: grid;
      gap: 10px;
      margin-bottom: 14px;
    }
    label.control {
      display: grid;
      gap: 4px;
      color: var(--muted);
      font-size: 12px;
    }
    select {
      min-height: 36px;
      border-radius: 6px;
      border: 1px solid rgba(23,32,38,.18);
      background: #fff;
      padding: 0 8px;
      color: var(--ink);
    }
    .toggle {
      display: flex;
      align-items: center;
      gap: 8px;
      color: var(--ink);
      font-size: 14px;
    }
    #log {
      border-top: 1px solid rgba(23,32,38,.14);
      padding-top: 10px;
      color: var(--muted);
      display: grid;
      gap: 6px;
      max-height: 36vh;
      overflow: auto;
      font-size: 13px;
    }
    @media (max-width: 920px) {
      main { grid-template-columns: 1fr; padding: 14px; }
      #board { width: 100%; }
    }
  </style>
</head>
<body>
  <main>
    <section id="board"></section>
    <aside>
      <h1>MNK DDA Playtest</h1>
      <div class="sub">You are blue. The opponent is red. Difficulty updates after each finished game.</div>
      <div class="actions">
        <button class="action primary" id="next">Next game</button>
      </div>
      <div class="controls">
        <label class="toggle">
          <input type="checkbox" id="aiFirst">
          AI first
        </label>
        <label class="control">
          Difficulty
          <select id="difficulty"></select>
        </label>
      </div>
      <div class="metrics" id="metrics"></div>
      <div id="log"></div>
    </aside>
  </main>
  <script>
    let state = null;
    const boardEl = document.getElementById("board");
    const metricsEl = document.getElementById("metrics");
    const logEl = document.getElementById("log");
    const aiFirstEl = document.getElementById("aiFirst");
    const difficultyEl = document.getElementById("difficulty");

    async function api(path, body) {
      const res = await fetch(path, {
        method: body ? "POST" : "GET",
        headers: {"Content-Type": "application/json"},
        body: body ? JSON.stringify(body) : undefined
      });
      return await res.json();
    }

    async function load() {
      state = await api("/state");
      render();
    }

    async function play(row, col) {
      const result = await api("/move", {row, col});
      state = result.state;
      render(result.error);
    }

    async function nextGame() {
      state = await api("/next", {});
      render();
    }

    async function setAiFirst() {
      state = await api("/ai-first", {aiFirst: aiFirstEl.checked});
      render();
    }

    async function setDifficulty() {
      const value = difficultyEl.value;
      state = await api("/difficulty", {level: value === "auto" ? null : Number(value)});
      render();
    }

    function statusText() {
      if (!state.gameOver) return "Your turn";
      if (state.winner === 1) return "You win";
      if (state.winner === -1) return "AI wins";
      return "Draw";
    }

    function metric(label, value) {
      return `<div class="metric"><b>${value}</b><span>${label}</span></div>`;
    }

    function render(error) {
      boardEl.style.setProperty("--cols", state.width);
      boardEl.style.setProperty("--rows", state.height);
      boardEl.style.gridTemplateColumns = `repeat(${state.width}, 1fr)`;
      boardEl.innerHTML = "";

      for (let r = 0; r < state.height; r++) {
        for (let c = 0; c < state.width; c++) {
          const btn = document.createElement("button");
          btn.className = "cell";
          btn.disabled = state.gameOver || state.board[r][c] !== 0;
          if (state.board[r][c] !== 0) {
            const stone = document.createElement("div");
            stone.className = "stone " + (state.board[r][c] === 1 ? "human" : "ai");
            btn.appendChild(stone);
          }
          btn.onclick = () => play(r, c);
          boardEl.appendChild(btn);
        }
      }

      const est = state.estimatedDifficulty.toFixed(2);
      const quality = state.moveQuality.toFixed(2);
      const reward = state.lastUpdate ? state.lastUpdate.reward.toFixed(3) : "-";
      renderDifficultySelect();
      aiFirstEl.checked = state.aiStarts;
      metricsEl.innerHTML = [
        metric("Status", statusText()),
        metric("Board", `${state.width}x${state.height} / ${state.k}`),
        metric("Current difficulty", `${state.difficultyLevel} / ${state.nLevels}`),
        metric("AI epsilon", state.epsilon.toFixed(2)),
        metric("Player estimate", est),
        metric("Move quality", quality),
        metric("Moves", state.moves),
        metric("Last reward", reward)
      ].join("");

      const lines = [];
      if (error) lines.push(`<div>${error}</div>`);
      if (state.lastUpdate) {
        lines.push(`<div>Finished: difficulty ${state.lastUpdate.playedDifficulty}, estimated player ${state.lastUpdate.estimatedPlayerDifficulty.toFixed(2)}, duration ${state.lastUpdate.duration.toFixed(2)}.</div>`);
      }
      for (const item of state.history.slice().reverse()) {
        const winner = item.winner === 1 ? "you" : item.winner === -1 ? "AI" : "draw";
        lines.push(`<div>Game: ${winner}, diff ${item.playedDifficulty}, est ${item.estimatedPlayerDifficulty.toFixed(2)}, reward ${item.reward.toFixed(3)}</div>`);
      }
      logEl.innerHTML = lines.join("");
    }

    function renderDifficultySelect() {
      const selected = state.manualDifficulty === null ? "auto" : String(state.difficultyLevel);
      const options = [`<option value="auto">Auto DDA</option>`];
      for (let level = 1; level <= state.nLevels; level++) {
        const label = `Level ${level}` + (level === state.difficultyLevel ? " (current)" : "");
        options.push(`<option value="${level}">${label}</option>`);
      }
      difficultyEl.innerHTML = options.join("");
      difficultyEl.value = selected;
    }

    document.getElementById("next").onclick = nextGame;
    aiFirstEl.onchange = setAiFirst;
    difficultyEl.onchange = setDifficulty;
    load();
  </script>
</body>
</html>
"""


def make_handler(session: PlaySession):
    class Handler(BaseHTTPRequestHandler):
        def _json(self, payload, status=200):
            raw = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path == "/":
                raw = HTML.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
                return
            if parsed.path == "/state":
                self._json(session.snapshot())
                return
            self.send_error(404)

        def do_POST(self):
            parsed = urlparse(self.path)
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
            if parsed.path == "/move":
                self._json(session.play_human_move(int(body["row"]), int(body["col"])))
                return
            if parsed.path == "/next":
                self._json(session.next_game())
                return
            if parsed.path == "/ai-first":
                self._json(session.set_ai_starts(bool(body.get("aiFirst"))))
                return
            if parsed.path == "/difficulty":
                level = body.get("level")
                self._json(session.set_difficulty(None if level is None else int(level)))
                return
            self.send_error(404)

        def log_message(self, fmt, *args):
            return

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description="Play MNK/Gomoku against a DDA opponent.")
    parser.add_argument("width", type=int, help="Board width")
    parser.add_argument("height", type=int, help="Board height")
    parser.add_argument("k", type=int, help="Connected stones needed to win")
    parser.add_argument("--episodes", type=int, default=30_000, help="Training episodes if model is missing")
    parser.add_argument("--levels", type=int, default=len(EPSILON_LEVELS), help="Number of difficulty levels")
    parser.add_argument("--port", type=int, default=8765, help="Local web server port")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--force-train", action="store_true", help="Retrain even if a saved model exists")
    parser.add_argument("--no-tactical-rules", action="store_true", help="Disable tactical rules during training and play")
    parser.add_argument("--no-open", action="store_true", help="Do not open the browser automatically")
    args = parser.parse_args()

    if args.k > max(args.width, args.height):
        raise SystemExit("k must be achievable on the board.")

    path = model_path(args.width, args.height, args.k)
    use_tactical_rules = not args.no_tactical_rules
    if args.force_train or not has_valid_model(path):
        path.parent.mkdir(parents=True, exist_ok=True)
        if args.width == args.height:
            train_mnk_agent(
                n=args.width,
                k=args.k,
                num_episodes=args.episodes,
                save_dir=str(path.parent),
                name="mnk",
                use_tactical_rules=use_tactical_rules,
            )
        else:
            train_rectangular_agent(
                args.width,
                args.height,
                args.k,
                args.episodes,
                path,
                use_tactical_rules=use_tactical_rules,
            )

    pool = build_opponent_pool(
        str(path),
        num_levels=args.levels,
        use_tactical_rules=use_tactical_rules,
    )
    session = PlaySession(args.width, args.height, args.k, pool, args.seed)

    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(session))
    url = f"http://127.0.0.1:{args.port}/"
    print(f"Playable MNK DDA is running: {url}")
    print("Press Ctrl+C to stop.")
    if not args.no_open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
