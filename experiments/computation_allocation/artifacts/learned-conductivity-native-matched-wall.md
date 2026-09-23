# Learned conductivity vs native Reckless, matched observed time

Checkpoint: `hf://buckets/Pawitt/temporary/latest.pt`, feature version 2,
width 64, training update 29. Tested with the Python/UCI Physarum training
search in inference mode, one stochastic trajectory per trial, explicit budget
256, explicit depth cap 4, quiescence cap 4096 nodes per frontier child.

Eight fixed chess positions were tested three times each with independent
sampling seeds. A separate native Reckless process then searched the same FEN
with `go movetime` set to the measured wall time of the Python search. The
actual elapsed time of both sides is recorded for every pair. The existing
native MultiPV depth-16 score cache adjudicates each root move. The script
validates every FEN and the legal move set against that cache.

| Position | Learned mean regret (cp) | Native mean regret (cp) | Mean learned time (ms) |
|---|---:|---:|---:|
| Start position | 6.0 | 0.0 | 2876 |
| Poisoned queen | 31.7 | 0.0 | 2786 |
| Tactical attack | 0.0 | 0.0 | 2802 |
| Quiet middlegame | 46.0 | 1.7 | 2011 |
| Kiwipete | 40.0 | 0.0 | 3083 |
| Complex middlegame | 10.0 | 10.0 | 2829 |
| Rook endgame | 76.0 | 0.0 | 985 |
| Minor-piece endgame | 351.0 | 0.0 | 424 |

Overall mean regret: learned 70.08 cp; native 1.46 cp. The learned search
selected a cached reference-best move in 12.5% of trials; native in 83.3%.
Native was ahead on 18 of 24 matched pairs, tied on six, and behind on none.
Mean observed time was 2224 ms for learned search and 2213 ms for native.

These are **eight positions**, not 24 independent chess tests. The depth-16
cache is a finite adjudicator; native reached depths 19–38 in the paired runs,
so the cached rankings are not ground truth. The learned implementation runs
through Python and UCI and includes that overhead, while native search is
compiled Rust. These results establish that this checkpoint and implementation
do not overtake native Reckless on this suite at matched observed time. They do
not isolate whether the limiting factor is the learned policy, flow rule,
evaluation, or runtime implementation.

Reproduce from repository root:

```bash
python experiments/computation_allocation/compare_learned_conductivity_native.py \
  --checkpoint local/latest.pt --engine target/release/reckless \
  --budget 256 --max-depth 4 --qnodes 4096 --repeats 3 \
  --output experiments/computation_allocation/artifacts/learned-conductivity-native-matched-wall-3x.json
```

Raw observations and exact per-pair timings are in
`learned-conductivity-native-matched-wall-3x.json`. An earlier one-repeat run is
in `learned-conductivity-native-matched-wall.json`.
