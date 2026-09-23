"""FEN-only input and resumable games. Dataset results/WDL never enter training."""
import hashlib
import json
import gc

import chess


DEFAULT_REVISION = '0040de16823530c90b5ab31ba187a268ed7f5396'


def board_position(board):
    return board.root().fen(), tuple(move.uci() for move in board.move_stack)


def restore_board(state):
    if state is None:
        return None
    board = chess.Board(state['fen'])
    for move in state['moves']:
        board.push_uci(move)
    return board


def board_state(board):
    if board is None:
        return None
    fen, moves = board_position(board)
    return {'fen': fen, 'moves': moves}


def held_out(fen):
    # Ignore clocks when partitioning, preventing the same position at another
    # game clock leaking across the root train/eval split. Not a game-ID split.
    key = ' '.join(fen.split()[:4])
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'big') % 20 == 0


class FenSource:
    def __init__(self, args, consumed=0):
        self.consumed = 0
        self.evaluate = args.evaluate_only
        self.local = args.positions is not None
        if self.local:
            rows = json.loads(args.positions.read_text())
            if not rows:
                raise ValueError('empty local FEN source')
            self.rows = iter(rows)
            self.dataset = None
        else:
            import datasets
            from datasets import load_dataset
            # This is a single-process iterator. Datasets 5 shares its epoch
            # scalar via Torch even without workers; restricted macOS runtimes
            # may deny the POSIX shared-memory allocation. An ordinary integer
            # is sufficient here because no DataLoader workers are launched.
            datasets.config.TORCH_AVAILABLE = False
            dataset = load_dataset('Pawitt/zero-evaluator', 'lc0_selfplay',
                                   split=args.dataset_split, revision=args.dataset_revision,
                                   streaming=True, columns=['fen'])
            self.dataset = dataset.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)
            self.rows = iter(self.dataset)
        # Deterministic stream replay restores shuffled order without silently
        # losing Hugging Face's shuffle buffer on state_dict/load_state_dict.
        for _ in range(consumed):
            next(self.rows)
            self.consumed += 1

    def next_board(self):
        for row in self.rows:
            self.consumed += 1
            fen = row['fen']  # Deliberately the ONLY accessed data column.
            if not self.local and held_out(fen) != self.evaluate:
                continue
            board = chess.Board(fen)
            if not board.is_valid():
                raise ValueError(f'invalid dataset FEN: {fen}')
            if board.outcome(claim_draw=False) is None:
                return board
        raise StopIteration('FEN source exhausted')

    def close(self):
        close = getattr(self.rows, 'close', None)
        if close is not None:
            close()
        self.rows = None
        self.dataset = None
        # Close Arrow generators while the Arrow module is still alive, before
        # interpreter teardown. Datasets 5 + Python 3.14 otherwise segfaults.
        gc.collect()
