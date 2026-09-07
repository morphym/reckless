"""Minimal synchronous UCI client tailored to the Reckless experiment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import queue
import subprocess
import threading
import time


MATE_BASE = 100_000


@dataclass(frozen=True)
class SearchInfo:
    depth: int
    multipv: int
    score_kind: str
    score_raw: int
    score: int
    bound: str | None
    nodes: int
    time_ms: int
    pv: tuple[str, ...]


@dataclass(frozen=True)
class SearchResult:
    bestmove: str
    infos: tuple[SearchInfo, ...]
    history: tuple[SearchInfo, ...] = ()


def normalize_score(kind: str, raw: int) -> int:
    if kind == "cp":
        return raw
    if kind == "mate":
        sign = 1 if raw > 0 else -1
        return sign * (MATE_BASE - abs(raw) * 100)
    raise ValueError(f"unknown score kind: {kind}")


def parse_info(line: str) -> SearchInfo | None:
    tokens = line.split()
    if not tokens or tokens[0] != "info" or "depth" not in tokens or "score" not in tokens:
        return None

    def value_after(name: str, default: int) -> int:
        try:
            return int(tokens[tokens.index(name) + 1])
        except (ValueError, IndexError):
            return default

    score_index = tokens.index("score")
    try:
        kind = tokens[score_index + 1]
        raw = int(tokens[score_index + 2])
    except (IndexError, ValueError):
        return None

    pv = tuple(tokens[tokens.index("pv") + 1 :]) if "pv" in tokens else ()
    bound = "lowerbound" if "lowerbound" in tokens else "upperbound" if "upperbound" in tokens else None
    return SearchInfo(
        depth=value_after("depth", 0),
        multipv=value_after("multipv", 1),
        score_kind=kind,
        score_raw=raw,
        score=normalize_score(kind, raw),
        bound=bound,
        nodes=value_after("nodes", 0),
        time_ms=value_after("time", 0),
        pv=pv,
    )


class RecklessUci:
    def __init__(self, executable: Path, timeout_seconds: float = 60.0) -> None:
        executable = executable.resolve()
        if not executable.is_file():
            raise FileNotFoundError(executable)
        self.timeout_seconds = timeout_seconds
        self.process = subprocess.Popen(
            [str(executable)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert self.process.stdin is not None
        assert self.process.stdout is not None
        self._lines: queue.Queue[str | None] = queue.Queue()

        def read_lines() -> None:
            assert self.process.stdout is not None
            for line in self.process.stdout:
                self._lines.put(line.rstrip("\r\n"))
            self._lines.put(None)

        self._reader = threading.Thread(target=read_lines, name="reckless-uci-reader", daemon=True)
        self._reader.start()
        self._send("uci")
        self._read_until(lambda line: line == "uciok")
        self._send("setoption name Threads value 1")
        self._send("setoption name Hash value 32")
        self._ready()

    def _send(self, command: str) -> None:
        if self.process.poll() is not None:
            raise RuntimeError(f"engine exited with status {self.process.returncode}")
        assert self.process.stdin is not None
        self.process.stdin.write(command + "\n")
        self.process.stdin.flush()

    def _send_many(self, commands: list[str]) -> None:
        if self.process.poll() is not None:
            raise RuntimeError(f"engine exited with status {self.process.returncode}")
        assert self.process.stdin is not None
        self.process.stdin.write("".join(command + "\n" for command in commands))
        self.process.stdin.flush()

    def _read_until(self, predicate) -> list[str]:
        deadline = time.monotonic() + self.timeout_seconds
        lines: list[str] = []
        while time.monotonic() < deadline:
            try:
                line = self._lines.get(timeout=deadline - time.monotonic())
            except queue.Empty:
                break
            if line is None:
                raise RuntimeError(f"engine closed output with status {self.process.poll()}")
            lines.append(line)
            if predicate(line):
                return lines
        raise TimeoutError(f"engine did not produce expected output; tail={lines[-10:]}")

    def _ready(self) -> None:
        self._send("isready")
        self._read_until(lambda line: line == "readyok")

    def new_game(self) -> None:
        """Reset Reckless once at an episode boundary.

        Branch computations inside an episode deliberately do not call this:
        the engine keeps its transposition table and native search histories so
        later computation can reuse earlier work.
        """
        self._send("ucinewgame")
        self._ready()

    def legal_moves(self, fen: str) -> tuple[str, ...]:
        """Return legal moves from Reckless's native move generator."""
        self._send(f"position fen {fen}")
        self._send("legalmoves")
        lines = self._read_until(lambda line: line == "legalmoves" or line.startswith("legalmoves "))
        return tuple(lines[-1].split()[1:])

    def fen_after_move(self, fen: str, move: str) -> str:
        """Apply a move on Reckless's native board and return its FEN."""
        self._send(f"position fen {fen} moves {move}")
        self._send("fen")
        lines = self._read_until(lambda line: line.startswith("fen "))
        return lines[-1][4:]

    def _search_current_position(self, depth: int) -> SearchResult:
        self._send(f"go depth {depth}")
        lines = self._read_until(lambda line: line.startswith("bestmove "))

        bestmove = lines[-1].split(maxsplit=1)[1]
        parsed = [info for line in lines if (info := parse_info(line)) is not None]
        if not parsed:
            raise RuntimeError(f"search returned no parseable info: {lines[-20:]}")
        completed_depth = max(info.depth for info in parsed)
        latest: dict[int, SearchInfo] = {}
        for info in parsed:
            if info.depth == completed_depth:
                latest[info.multipv] = info
        return SearchResult(
            bestmove,
            tuple(latest[index] for index in sorted(latest)),
            tuple(parsed),
        )

    def analyze(
        self,
        fen: str,
        depth: int,
        multipv: int = 1,
        moves: tuple[str, ...] = (),
    ) -> SearchResult:
        if depth <= 0:
            raise ValueError("depth must be positive")
        self.new_game()
        self._send(f"setoption name MultiPV value {multipv}")
        self._ready()

        position = f"position fen {fen}"
        if moves:
            position += " moves " + " ".join(moves)
        self._send(position)
        return self._search_current_position(depth)

    def analyze_branch(self, fen: str, move: str, depth: int) -> SearchResult:
        """Search one root branch while preserving native TT/history state."""
        if depth <= 0:
            raise ValueError("depth must be positive")
        self._send("setoption name MultiPV value 1")
        self._send(f"position fen {fen} moves {move}")
        return self._search_current_position(depth)

    def static_evaluate(self, fens: list[str]) -> list[int]:
        """Return frozen NNUE scores from side-to-move, without running search."""
        commands: list[str] = []
        for fen in fens:
            commands.extend((f"position fen {fen}", "staticeval"))
        self._send_many(commands)

        scores: list[int] = []
        for _ in fens:
            lines = self._read_until(lambda line: line.startswith("staticeval "))
            scores.append(int(lines[-1].split()[1]))
        return scores

    def static_evaluate_after_moves(self, fen: str, moves: tuple[str, ...]) -> list[int]:
        """Evaluate native child positions without search, in child POV."""
        commands: list[str] = []
        for move in moves:
            commands.extend((f"position fen {fen} moves {move}", "staticeval"))
        self._send_many(commands)

        scores: list[int] = []
        for _ in moves:
            lines = self._read_until(lambda line: line.startswith("staticeval "))
            scores.append(int(lines[-1].split()[1]))
        return scores

    def close(self) -> None:
        if self.process.poll() is None:
            try:
                self._send("quit")
                self.process.wait(timeout=5)
            except (BrokenPipeError, subprocess.TimeoutExpired):
                self.process.terminate()
                self.process.wait(timeout=5)
        self._reader.join(timeout=1)

    def __enter__(self) -> "RecklessUci":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
