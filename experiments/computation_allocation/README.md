# CS computation controller

This experiment now contains the first complete reinforcement-learning path
for computation allocation.  It keeps three roles separate:

1. Reckless owns chess state, native legal move generation, `MovePicker`,
   alpha-beta search, TT/history state, and frozen NNUE evaluation.
2. The CS controller chooses which root branch receives the next depth unit,
   or chooses a real terminal `STOP` action.
3. Deeper Reckless values are privileged reward data only. They never enter the
   controller observation and never provide target computation actions.

The previously trained move-ordering network is a separate alpha-beta
component. It is not the actor and receives no PPO gradients here.

## Online large-scale path

`online_train.py` is the corpus-free training runner. Every episode is created
at runtime:

1. Starting from the normal initial position, it asks Reckless for native legal
   moves and frozen NNUE values. A Boltzmann sampler turns those deterministic
   numbers into unusual move choices. Root-move temperature is measured in
   centipawns and anneals during training.
2. A separate Reckless process searches the generated root above the CS depth
   cap with full MultiPV. Its numerical value for every legal root move is the
   hidden reward reference; its preferred action and tree are never exposed.
3. A fresh live CS process receives a smaller sampled budget. GPU-batched actor
   inference selects branch computations or `STOP`, while CPU workers execute
   Reckless-native searches concurrently.
4. PPO uses `r_t = L_t - L_(t+1)` and `gamma=1`. Controller temperature and the
   entropy bonus provide early exploration, then anneal toward concentrated
   allocation.

The NNUE itself has no temperature and remains deterministic. Temperature is
applied only to sampling from its move-evaluation numbers and to the controller
logits. High-depth reference search remains deterministic. Reference and CS use
separate engine processes so reference TT contents cannot leak into the
lower-cap search.

## Components

- `diagnostic.py`: exact Bellman toy model, delayed payoff, budget dependence,
  reliability thresholds, and reward identities.
- `allocation_env.py`: original deterministic cached branch environment and
  privileged Bellman diagnostic.
- `live_env.py`: live CS episode using Reckless-native legal actions and search.
- `controller_state.py`: leak-free candidate/global features and legal masks.
- `controller_model.py`: variable-frontier actor plus `STOP`, and a global
  critic. The default network has 179,683 parameters.
- `ppo.py`: masked PPO with `gamma=1`; raw rewards are retained and advantages
  alone are normalized. Rollouts are batched across environments for GPU use.
- `build_controller_corpus.py`: converts the existing native depth-6 Reckless
  labels into branch-local depth trajectories.
- `train_controller.py`: mixed-budget PPO training on cached trajectories.
- `evaluate_controller.py`: equal-budget held-out comparisons with current-best,
  round-robin, and random allocation.
- `run_live_cs.py`: runs either a heuristic or trained controller against a
  persistent native Reckless process.
- `online_train.py`: generates roots and rewards online, batches controller
  inference on CUDA/MPS, and runs live Reckless search workers concurrently.

## Native Reckless boundary

Reckless exposes a search-free `legalmoves` command. It calls
`Board::generate_all_moves`, so Python maintains no second chess move generator.
A live computation action sends the chosen root move back to Reckless and runs
the child search at the requested depth. All internal descendants use the
engine's normal `MovePicker` and native alpha-beta implementation.

`LiveCsEnv` calls `ucinewgame` once at episode start, not before each action.
Its TT and history state therefore persist across actions. The current
controller observes branch estimates and aggregate cost/state features, not
the complete TT contents; this live version is consequently a practical
partially observable environment. A recurrent controller or an explicit cache
summary is the clean next refinement.

For cached PPO training, each root branch receives an independent Reckless
cache trajectory. Successive depths within that branch reuse work, while the
order of unrelated branch actions cannot invisibly change the transition. The
visible branch-depth vector is therefore a deterministic Markov approximation.

## Observation and actions

Every legal root branch has candidate features for its current score, score
gap, revealed depth, remaining depth, bound kind, recent score change, node/time
cost, move squares, and whether it is currently selected. Global features
include root gaps, the depth profile, total charged work, decision changes, and
both remaining and original budget.

The actor scores the variable candidate set plus `STOP`. The critic pools the
candidate set and predicts remaining return. Reference depth-6 values are
absent from both feature groups.

## Reward

For reference regret `L_t`, every transition receives

```text
r_t = L_t - L_(t+1)
gamma = 1
```

Thus every tested episode satisfies
`sum(reward) = initial_loss - terminal_loss`. Rewards are not clipped or
nonlinearly transformed.

## Commands

### One-command NVIDIA setup

On an Ubuntu/Debian NVIDIA host, the installer creates an isolated Python
environment, installs the Python and system dependencies, compiles Reckless in
release mode, and runs both Rust and Python tests:

```sh
./experiments/computation_allocation/install.sh
```

The NVIDIA driver must already be installed on the host. For non-Debian Linux
systems, install Python 3, Clang, and Rust 1.88+ first; the remainder of the
script is portable. Set `REQUIRE_CUDA=0` only when intentionally preparing a
CPU-only development machine. If the host requires a particular PyTorch CUDA
wheel channel, set `TORCH_INDEX_URL` to that channel before running the
installer. An existing PyTorch installation is preserved when it successfully
executes a tensor operation on the requested device.

Build and test Reckless from the repository root:

```sh
cargo test --release
cargo build --release
```

Build a native CS corpus from the existing depth-6 labels:

```sh
python build_controller_corpus.py \
  --input ../../local/reckless-ordering-native-d6 \
  --engine ../../target/release/reckless \
  --output ../../local/cs-controller-native-1k.jsonl \
  --manifest ../../outputs/cs_controller/corpus-1k-manifest.json \
  --target 1000 --depths 1,2,3,4 --workers 4 --split all
```

Train and evaluate:

```sh
python train_controller.py \
  --input ../../local/cs-controller-native-1k.jsonl \
  --updates 150 --episodes-per-update 128 \
  --min-budget 1 --max-budget 8 --device mps \
  --checkpoint ../../outputs/cs_controller/controller-1k.pt \
  --summary ../../outputs/cs_controller/training-1k.json

python evaluate_controller.py \
  --input ../../local/cs-controller-native-1k.jsonl \
  --checkpoint ../../outputs/cs_controller/controller-1k.pt \
  --output ../../outputs/cs_controller/evaluation-1k.json
```

Run the trained controller against live native search:

```sh
python run_live_cs.py \
  --engine ../../target/release/reckless \
  --strategy controller \
  --checkpoint ../../outputs/cs_controller/controller-1k.pt \
  --budget 8 --depths 1,2,3,4
```

Launch corpus-free training on an NVIDIA machine from the repository root:

```sh
./experiments/computation_allocation/train_nvidia.sh
```

This command consumes no position corpus. Checkpoints are written after every
update so a long remote run is fully resumable, including optimizer and random
number generator state. Resume the same run with:

```sh
./experiments/computation_allocation/train_nvidia.sh --resume
```

The launcher's large-scale defaults match the documented experiment. They can
be changed through environment variables, for example
`WORKERS=16 REFERENCE_DEPTH=10 ./experiments/computation_allocation/train_nvidia.sh`.

TensorBoard events are written to `outputs/cs_online/tensorboard`. They include
PPO losses and entropy, initial and terminal reference regret, regret reduction,
node use, temperatures, throughput, reward-identity checks, and per-episode
histograms. View them from the repository root with:

```sh
.venv/bin/tensorboard --logdir outputs/cs_online/tensorboard --port 6006
```

On a remote trainer, forward port 6006 over SSH rather than exposing it
publicly. Set `TENSORBOARD_DIR` to relocate the logs, or pass
`--no-tensorboard` to disable them.

## Current boundary

The 1,000-root run demonstrates that the complete RL path works and that a
controller can reach current-best-like regret with fewer cached search nodes.
It is not yet a strength claim. The next training run needs substantially more
roots, hard-case mining, repeated random seeds, inference cost charged in the
time budget, and match-level evaluation against conventional Reckless search.
