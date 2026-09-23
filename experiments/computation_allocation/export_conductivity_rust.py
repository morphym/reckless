#!/usr/bin/env python3
"""Export a trained conductivity head to the compact, versioned Rust format."""
import argparse
from pathlib import Path
import struct

import torch


KEYS = (
    'encoder.0.weight', 'encoder.0.bias',
    'encoder.2.weight', 'encoder.2.bias',
    'output.0.weight', 'output.0.bias',
    'output.2.weight', 'output.2.bias',
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if state['feature_version'] != 2:
        raise ValueError('Rust search requires conductivity feature version 2')
    width = state['width']
    model = state['model']
    expected = ((width, 924), (width,), (width, width), (width,),
                (width, 2 * width), (width,), (1, width), (1,))
    data = bytearray(b'PCNDv2\0\0')
    data.extend(struct.pack('<II', width, state.get('update') or 0))
    for key, shape in zip(KEYS, expected):
        tensor = model[key].detach().cpu().contiguous()
        if tuple(tensor.shape) != shape:
            raise ValueError(f'{key}: expected {shape}, got {tuple(tensor.shape)}')
        data.extend(tensor.float().numpy().astype('<f4', copy=False).tobytes())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(data)
    print(f'{args.output}: {len(data)} bytes, width {width}, update {state.get("update")}')


if __name__ == '__main__':
    main()
