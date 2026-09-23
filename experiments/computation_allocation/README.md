# Heuristic Physarum search

This branch keeps one experimental search path: Physarum flow over an explicit
tree, with Reckless quiescence evaluation and a conservative built-in heuristic
prior. There is no learned policy head, checkpoint, controller, or training
runtime.

Build the engine from the repository root:

```sh
cargo build --release
```

The normal UCI defaults are a 4,096 evaluation budget and no artificial depth
cap. `PhysarumBudget`, `PhysarumQNodes`, and `PhysarumSeed` remain available
for controlled experiments. A `go depth N` command still supplies an explicit
depth limit; time- and node-limited searches grow until their active limit.

The match harness compares heuristic Physarum with a native Reckless binary:

```sh
.venv/bin/python experiments/computation_allocation/match_rust_conductivity_native.py \
  --flow-engine target/release/reckless \
  --native-engine /path/to/native/reckless \
  --native-depth 4 \
  --position tactical-attack \
  --output experiments/computation_allocation/artifacts/match.json
```
