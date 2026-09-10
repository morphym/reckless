#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENGINE="${1:-$REPO_ROOT/target/release/reckless}"

if [[ ! -x "$ENGINE" ]]; then
    echo "Physarum engine is not executable: $ENGINE" >&2
    echo "Build it with: cargo build --release --features physarum-search" >&2
    exit 1
fi

OUTPUT="$({
    printf '%s\n' \
        'setoption name PhysarumBatch value 8' \
        'setoption name PhysarumDiagnosticPriorMove value d8d4' \
        'setoption name PhysarumDiagnosticPriorMass value 990' \
        'position fen rnbqkb1r/ppp1pppp/5n2/8/2PP4/2P2N2/PPQ1PPPP/RNB1KB1R b KQkq - 0 1' \
        'go nodes 1024' \
        'quit'
} | "$ENGINE")"

printf '%s\n' "$OUTPUT"

BEST_MOVE="$(awk '/^bestmove / { move=$2 } END { print move }' <<<"$OUTPUT")"
LAST_INFO="$(awk '/^info depth / { line=$0 } END { print line }' <<<"$OUTPUT")"
PV_MOVE="$(awk '{ for (i=1; i<=NF; ++i) if ($i == "pv") { print $(i+1); exit } }' <<<"$LAST_INFO")"

if [[ "$BEST_MOVE" == "d8d4" ]]; then
    echo "FAIL: the 99% misleading queen-sacrifice prior became bestmove" >&2
    exit 1
fi
case " b8d7 c7c5 b8c6 f6d7 e7e5 g7g6 " in
    *" $BEST_MOVE "*) ;;
    *)
        echo "FAIL: bestmove '$BEST_MOVE' is outside the native Reckless depth-8 safe set" >&2
        exit 1
        ;;
esac
if [[ -z "$BEST_MOVE" || "$PV_MOVE" != "$BEST_MOVE" ]]; then
    echo "FAIL: final PV root '$PV_MOVE' does not match bestmove '$BEST_MOVE'" >&2
    exit 1
fi

echo "PASS: 99% prior d8d4 was refuted; native-safe bestmove=$BEST_MOVE and PV agrees."
