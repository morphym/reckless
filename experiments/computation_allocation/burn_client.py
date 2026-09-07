"""Binary client for the resident Rust/Burn CS actor."""

from __future__ import annotations

from pathlib import Path
import struct
import subprocess
from typing import Sequence

import torch

from controller_state import ControllerObservation


class BurnCsInference:
    def __init__(self, binary: Path, weights: Path) -> None:
        self.process = subprocess.Popen(
            [str(binary), "serve", str(weights)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )

    def logits(self, observation: ControllerObservation) -> list[float]:
        if self.process.stdin is None or self.process.stdout is None:
            raise RuntimeError("Burn process pipes are unavailable")
        count = len(observation.candidates)
        flat_candidates = [value for row in observation.candidates for value in row]
        request = bytearray(struct.pack("<I", count))
        request.extend(struct.pack(f"<{len(flat_candidates)}f", *flat_candidates))
        request.extend(struct.pack("<12f", *observation.global_features))
        request.extend(bytes(observation.action_mask))
        self.process.stdin.write(request)
        self.process.stdin.flush()
        response = self.process.stdout.read(4 * (count + 1))
        if len(response) != 4 * (count + 1):
            raise RuntimeError(f"short Burn response; process status={self.process.poll()}")
        return list(struct.unpack(f"<{count + 1}f", response))

    def close(self) -> None:
        if self.process.stdin is not None:
            self.process.stdin.close()
        self.process.wait(timeout=10)
        if self.process.returncode != 0:
            raise RuntimeError(f"Burn process exited with {self.process.returncode}")

    def __enter__(self) -> "BurnCsInference":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


def pytorch_logits(model, observation: ControllerObservation) -> Sequence[float]:
    from controller_model import batch_observations

    with torch.no_grad():
        logits, _ = model(batch_observations([observation], "cpu"))
    return logits[0].tolist()
