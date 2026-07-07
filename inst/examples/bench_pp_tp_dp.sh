#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Full PP/TP/DP benchmark sweep for llamaR on 4x P100, driven by replica.R.
# Each config runs in its own process. DP configs launch replicas CONCURRENTLY
# (background + wait) so the summed decode t/s reflects real parallel throughput.
#
# Usage: ./run_bench.sh <model.gguf> [n_gen] [n_rep]
# ---------------------------------------------------------------------------
set -u
MODEL="${1:?usage: run_bench.sh <model.gguf> [n_gen] [n_rep]}"
NGEN="${2:-128}"
NREP="${3:-3}"
R="$(dirname "$0")/bench_replica.R"

run() { echo; echo ">>> $1"; shift; Rscript "$R" "$MODEL" "$@" "$NGEN" "$NREP"; }

echo "=== SINGLE-CONTEXT MODES (sequential) ==="
run "baseline 1 GPU"      base       Vulkan0                       none
run "PP layer-split 2GPU" pp2        Vulkan0,Vulkan1               layer
run "TP row-split 2GPU"   tp2        Vulkan0,Vulkan1               row
run "PP layer-split 4GPU" pp4        Vulkan0,Vulkan1,Vulkan2,Vulkan3 layer
run "TP row-split 4GPU"   tp4        Vulkan0,Vulkan1,Vulkan2,Vulkan3 row

echo
echo "=== TP=2 x DP=2  (two replicas CONCURRENTLY) ==="
echo ">>> replica A on {Vulkan0,Vulkan1}  +  replica B on {Vulkan2,Vulkan3}  in parallel"
Rscript "$R" "$MODEL" A Vulkan0,Vulkan1 row "$NGEN" "$NREP" &
PID_A=$!
Rscript "$R" "$MODEL" B Vulkan2,Vulkan3 row "$NGEN" "$NREP" &
PID_B=$!
wait $PID_A; wait $PID_B

echo
echo "=== DP=4 (four single-GPU replicas CONCURRENTLY, no TP) ==="
for g in 0 1 2 3; do
    Rscript "$R" "$MODEL" "dp$g" "Vulkan$g" none "$NGEN" "$NREP" &
done
wait

echo
echo "Done. For DP rows: throughput = sum of the concurrent replicas' decode_tps."
echo "Grep the RESULT lines above; single-context rows are per-config decode t/s."
