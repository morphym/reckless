# Progressive conductivity training (experimental)

This is a standalone, small PyTorch policy head and a Python Physarum-style
training search. It does **not** replace the deployed Rust search or use the old
CS allocation controller. Native Reckless generates legal moves, evaluates
frontier children with full-window quiescence search, and, in a separate
process, supplies deeper reference values. Roots come from the `fen` column of
the `Pawitt/zero-evaluator` `lc0_selfplay` configuration. No WDL, game result,
or other dataset field is used as a label or feature.

## Model and objective

The default head has 71,681 parameters. A shared edge MLP consumes the parent
board, move, depth, consumed budget, initial/current child evaluation, parent
evaluation, evaluation change, visits, frontier/expanded flags, and transported
evidence memory. Mean pooling over siblings provides local branch context.
Variable move counts need no padding. All evaluations in search state use root
POV, with actor-relative features for local decisions. Board planes use absolute
colors. Deeper labels NEVER enter these features.

For each represented sibling set, the head outputs

    d_theta(s,a,h) = 0.01 + softmax(MLP(board, move, observed history))[a].

The actual edge conductivity is D = d_theta + M. M starts at zero and retains
0.97 of its previous value after each successful expansion. A selected frontier
transports bounded evidence utility to every ancestor edge, divided by path
length. Because the frontier is sampled proportional to conserved current, its
deposit is a stochastic estimate of flow-weighted evidence transport. This is
not the previous deterministic top-k flow implementation. The network is
re-evaluated as the search state progresses, rather than predicting a full line.

Unit current enters the root; frontier sinks have equal pressure. Series edge
conductance is D*G/(D+G); parallel conductances add. Differentiable flow defines
the categorical distribution of the next frontier to expand. Values back up
with alternating max/min; the final move maximizes the backed-up root value.

The external reference uses native Reckless MultiPV at a depth strictly greater
than the explicit tree cap. Quiescence may extend tactically past that cap; the
reference is therefore deeper in normal search plies, not guaranteed to inspect
more total plies on every line. Every legal root move must have a completed exact
(not bound-only) score. Let u(cp)=tanh(cp/600), u(winning mate)=1, and
u(losing mate)=-1. The ONLY optimization objective is expected terminal regret:

    J(theta) = E[max_a u(V_ref(s,a)) - u(V_ref(s, a_search))].

Regret is in bounded utility units [0,2], **not centipawns**. This monotone scale
preserves ranking of finite cp values but changes their relative penalty and
saturates extreme scores. Mate distance is not optimized. The deeper engine is
an approximate teacher, not ground truth.

Bound-only/missing MultiPV entries are re-searched independently from the child
at reference-depth minus one, with their side-to-move score negated back to root
POV. A reference that still fails the exact-score check stops training rather
than silently supplying an invalid label.

Discrete expansions use a score-function gradient through the actual flow:

    gradient estimate = mean_i [(regret_i - baseline_i) * grad log P(trajectory_i)]

The baseline is the mean regret of OTHER independent rollouts of the same root.
There is no critic, imitation term, entropy bonus, or self-reported-score reward.
Same-regret groups produce zero gradients; log gradient norms to detect this.
The logged surrogate loss itself is not a performance metric.

## Run

From the repository root, using a Python environment with runnable PyTorch:

```bash
python -m pip install -r experiments/computation_allocation/requirements-conductivity.txt
cargo build --release --no-default-features --bin reckless
python experiments/computation_allocation/train_conductivity.py \
  --engine target/release/reckless --device cuda \
  --reference-depth 12 --max-depth 4 --budget 256 \
  --qnodes 4096 --rollouts 4 --updates 10000 --output outputs/conductivity
```

The dataset stream mixes `strong`, `mid`, `low`, and `early` splits with a seeded
shuffle buffer and reads only `fen`. A verified Hub commit SHA is pinned by
default; override it only to start a new run. A deterministic FEN hash reserves
about 5% of roots for `--evaluate-only`. This is a position split, not a game
identity split. For offline smoke tests, `--positions some.json` accepts a list
of objects containing `fen` and ignores any other fields.
The four named splits are loaded separately and concatenated before shuffling;
the `--dataset-split strong+mid+low+early` setting is parsed by this trainer,
not passed to Hugging Face as one split name.

Each sampled nonterminal FEN begins a game. The first of the independent search
rollouts chooses the played move, then both colors use the same policy at the
next position. Training and play continue through legal moves until checkmate,
stalemate, insufficient material, fivefold repetition, or the automatic
75-move draw. A FEN lacks earlier move history; repetition tracking begins at
the sampled root. Checkpoints retain the subsequent complete game history.

Use `--device mps` on Apple Silicon, or `cpu`. Existing runnable Torch satisfies
the requirement; no CUDA toolkit installation is requested. CUDA only runs the
small network, not native Reckless. This sequential correctness prototype will
remain CPU/UCI-bound; it is NOT yet a GPU-saturating large-scale trainer.

Checkpoints save atomically after each update to `outputs/conductivity/latest.pt`,
including model, optimizer, architecture/config, dataset stream position,
in-progress game history, update and CPU sampler RNG.
SIGINT saves current weights too; an interrupted rollout is discarded. Resume
trusted checkpoints with `--resume outputs/conductivity/latest.pt` and the same
architecture/search and dataset settings. `--updates` is the total target update
number. Terminal dataset roots are skipped. Resume with the same dataset split,
revision, seed, and shuffle buffer so the stream replays identically.
Checkpoints from the earlier static-evaluation prototype use feature version 1
and cannot resume this quiescence version; start a new run with a new output
directory.

For held-out evaluation, keep weights frozen and use a different root seed:

```bash
python experiments/computation_allocation/train_conductivity.py \
  --engine target/release/reckless --device cuda \
  --resume outputs/conductivity/latest.pt --evaluate-only --seed 1042 \
  --updates 100 --reference-depth 12 --max-depth 4 --budget 256 \
  --output outputs/conductivity-heldout
tensorboard --logdir outputs
```

Watch held-out `eval/mean_regret` (lower is better), versus an untrained head with
the same seed and settings. Training regret alone does not establish improvement.
Metrics are also written to JSONL. No best checkpoint is selected from noisy
training regret. Use separate output directories for new runs and evaluation.

TensorBoard event files live in `<output>/tensorboard`. The writer flushes after
every completed update. For side-by-side train/eval runs:

```bash
tensorboard --logdir_spec train:outputs/conductivity/tensorboard,eval:outputs/conductivity-heldout/tensorboard
```

The most useful curves are `train/mean_regret` and `eval/mean_regret` (lower is
better), `*/optimal_fraction` (higher is better), and `*/regret_std` (variation
between same-root rollouts). `*/gradient_norm` shows whether training updates
have a usable signal; a group whose moves all have the same regret has zero
gradient. `*/search/qsearch_truncation_rate` should stay low: frequent capping
means frontier scores may be unreliable. `*/search/qsearch_nodes` and
`*/search/frontier_evaluations` distinguish tactical work from explicit tree
width. `*/reference_seconds`, `*/search_seconds`, and `*/optimizer_seconds`
show where the runtime goes. `*/game/ply`, `*/game/completed`, and terminal-only
`*/game/result_white` and `*/game/length_plies` describe self-play progress.
The game result is observational and is not the regret training target.

## Limits and checks

- The strict budget counts frontier child evaluations. Complete sibling sets are evaluated
  before expansion; a branch that cannot fit closes for this episode. The count
  is now `frontier_evaluations`: a full-window native quiescence search may visit
  many nodes per frontier. `qsearch_nodes` is logged separately. Policy inference,
  move generation and flow costs are additional. This is not an equal-wall-time
  baseline comparison.
- Each native quiescence call has a 4,096-node cap by default (`--qnodes`). At
  the cap, the unresolved leaf uses frozen NNUE (or a neutral score if in
  check), and the result is marked `qsearch_truncated`. Truncated values are
  not inserted into the native TT as exact scores. Monitor that count: a high
  rate means the values are still horizon-limited; increasing `--qnodes` raises
  CPU cost.
- Native quiescence stabilizes captures and checks but can still have horizon
  effects. A deeper reference penalizes resulting bad decisions but does not
  guarantee their elimination.
- Self-play from an Lc0 FEN is a new continuation generated by the current
  policy. It need not match the original Lc0 game; the dataset `result` is ignored.
- A completed terminal game is recorded, but the optimization signal at each
  position remains deeper-search regret, not the eventual game result.
- CPU/GPU parity, policy learning efficacy, production Rust/Burn export, and
  matched-cost MCTS comparisons are separate follow-up validations. Neither this
  implementation nor a successful gradient smoke test proves stronger play.

```bash
python -m unittest discover -s experiments/computation_allocation/tests \
  -p test_conductivity.py -v
```

## Local verification (2026-09-23)

- Thirteen unit tests pass after the quiescence/dataset changes, including flow conservation/autograd, closed branches,
  sibling permutation equivariance, progressive evidence input, regret gradient
  direction, equal-cost zero gradient, terminal/mate handling, strict budget,
  checkpoint roundtrip, and bound-reference POV reversal.
- Earlier static-evaluation prototype: native release build succeeded. Two real training updates with seed 42,
  depth-5 reference, depth-3 cap, 256 NNUE-call budget and four rollouts produced
  finite gradient norms 0.02393 and 0.02912. Saved parameters differed from
  initialization (total absolute change 9.7153). These are gradient smoke tests,
  NOT evidence of improved held-out strength.
- The next root exposed a bound-only MultiPV label; after implementing fallback,
  all 27 moves on that exact root received valid references.
- Resume completed update 3. Frozen evaluation with independent seed 1042 also
  completed. Local tests used CPU; MPS was unavailable in this runtime and CUDA
  was not tested. Logs/checkpoints from these runs are under
  `/private/tmp/conductivity-*-smoke` and are disposable, not trained deliverables.
- Quiescence version: native release build and 13 unit tests pass. A local
  two-update end-to-end game from the initial position played `d4` and `...Nf6`;
  its first gradient was finite/nonzero and `qsearch_nodes` exceeded frontier
  counts. The pinned `lc0_selfplay` stream returned a legal FEN while accessing
  only `fen`; one real FEN training update completed and saved its played move.
  That update had zero regret for both rollouts, so no policy gradient arose.
  This remains a functional smoke test, not a strength result.
- A second ply from that same streamed game initially exceeded the UCI client
  timeout in an uncapped quiescence call. With the per-call node cap, two
  consecutive streamed training updates completed from the same root; the
  second continued at black's reply. Both reported zero truncated frontier
  evaluations on that particular game. This establishes progress through the
  previously stalled position, not search strength or cap safety on every FEN.
- TensorBoard logging was checked with a full local update and the event file
  reader: 16 scalar tags appeared under `train/`, including regret, optimal
  fraction, quiescence cost/truncation, timing, and game progress. Fourteen unit
  tests pass, including scalar tag/step coverage.
