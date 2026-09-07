"""Export the inference-only CS actor to PyTorch-layout SafeTensors for Burn."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import struct

import torch


KEYS = {
    "candidate_encoder.0.weight": "candidate_linear_1.weight",
    "candidate_encoder.0.bias": "candidate_linear_1.bias",
    "candidate_encoder.2.weight": "candidate_norm.weight",
    "candidate_encoder.2.bias": "candidate_norm.bias",
    "candidate_encoder.3.weight": "candidate_linear_2.weight",
    "candidate_encoder.3.bias": "candidate_linear_2.bias",
    "global_encoder.0.weight": "global_linear.weight",
    "global_encoder.0.bias": "global_linear.bias",
    "global_encoder.2.weight": "global_norm.weight",
    "global_encoder.2.bias": "global_norm.bias",
    "context_encoder.0.weight": "context_linear.weight",
    "context_encoder.0.bias": "context_linear.bias",
    "context_encoder.2.weight": "context_norm.weight",
    "context_encoder.2.bias": "context_norm.bias",
    "candidate_actor.0.weight": "candidate_actor_hidden.weight",
    "candidate_actor.0.bias": "candidate_actor_hidden.bias",
    "candidate_actor.2.weight": "candidate_actor_output.weight",
    "candidate_actor.2.bias": "candidate_actor_output.bias",
    "stop_actor.0.weight": "stop_actor_hidden.weight",
    "stop_actor.0.bias": "stop_actor_hidden.bias",
    "stop_actor.2.weight": "stop_actor_output.weight",
    "stop_actor.2.bias": "stop_actor_output.bias",
}


def save_safetensors(tensors: dict[str, torch.Tensor], output: Path, metadata: dict[str, str]) -> None:
    """Write the small F32 subset of SafeTensors without another Python dependency."""
    header: dict[str, object] = {"__metadata__": metadata}
    payload = bytearray()
    for name, tensor in tensors.items():
        value = tensor.detach().to(dtype=torch.float32, device="cpu").contiguous()
        encoded = value.numpy().tobytes(order="C")
        start = len(payload)
        payload.extend(encoded)
        header[name] = {
            "dtype": "F32",
            "shape": list(value.shape),
            "data_offsets": [start, len(payload)],
        }
    encoded_header = json.dumps(header, separators=(",", ":")).encode()
    encoded_header += b" " * ((8 - len(encoded_header) % 8) % 8)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as handle:
        handle.write(struct.pack("<Q", len(encoded_header)))
        handle.write(encoded_header)
        handle.write(payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    state = checkpoint["model_state_dict"]
    missing = set(KEYS) - set(state)
    if missing:
        raise ValueError(f"checkpoint is missing actor tensors: {sorted(missing)}")
    exported = {target: state[source] for source, target in KEYS.items()}
    actor_parameters = sum(tensor.numel() for tensor in exported.values())
    save_safetensors(
        exported,
        args.output,
        {
            "format": "pytorch",
            "model": "reckless-cs-controller-actor",
            "completed_updates": str(checkpoint["completed_updates"]),
            "actor_parameters": str(actor_parameters),
            "candidate_features": "18",
            "global_features": "12",
        },
    )
    print(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint),
                "output": str(args.output),
                "completed_updates": checkpoint["completed_updates"],
                "actor_parameters": actor_parameters,
                "tensors": len(exported),
            }
        )
    )


if __name__ == "__main__":
    main()
