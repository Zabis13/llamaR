#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Full PP/TP/DP benchmark sweep for llamaR, driven by bench_replica.R.
# Each config runs in its own process. DP configs launch replicas CONCURRENTLY
# (background + wait) so the summed decode t/s reflects real parallel throughput.
#
# Scales to any GPU count. NGPU defaults to the auto-detected device count
# (nvidia-smi -L); pass it explicitly to restrict to a subset (e.g. when some
# cards are busy). TP_SIZE selects the tensor-parallel width for the TPxDP and
# single-context TP/PP rows; N_REPLICAS = NGPU / TP_SIZE.
#
# Usage: ./bench_pp_tp_dp.sh <model.gguf> [NGPU] [TP_SIZE] [n_gen] [n_rep]
#   NGPU     number of GPUs to use          (default: nvidia-smi -L | wc -l)
#   TP_SIZE  GPUs per replica for TPxDP      (default: 2)
#   n_gen    tokens generated per rep        (default: 128)
#   n_rep    timed reps per config          (default: 3)
#
# Examples:
#   ./bench_pp_tp_dp.sh model.gguf                 # auto GPUs, TP=2
#   ./bench_pp_tp_dp.sh model.gguf 8 2             # 8 GPUs, TP=2 x DP=4
#   ./bench_pp_tp_dp.sh model.gguf 8 4 128 3       # 8 GPUs, TP=4 x DP=2
# ---------------------------------------------------------------------------
set -u
MODEL="${1:?usage: bench_pp_tp_dp.sh <model.gguf> [NGPU] [TP_SIZE] [n_gen] [n_rep]}"
NGPU="${2:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
TP_SIZE="${3:-2}"
NGEN="${4:-128}"
NREP="${5:-3}"
R="$(dirname "$0")/bench_replica.R"

[ "$NGPU" -ge 1 ] 2>/dev/null || { echo "ERROR: could not determine NGPU (got '$NGPU'). Pass it explicitly." >&2; exit 1; }
if [ $((NGPU % TP_SIZE)) -ne 0 ]; then
    echo "ERROR: NGPU=$NGPU not divisible by TP_SIZE=$TP_SIZE." >&2; exit 1
fi
N_REPLICAS=$((NGPU / TP_SIZE))

echo "=== config: NGPU=$NGPU  TP_SIZE=$TP_SIZE  DP=$N_REPLICAS  n_gen=$NGEN  n_rep=$NREP ==="

# devs A B  -> comma-separated "Vulkan<A>,...,Vulkan<B-1>" for GPUs [A, B).
devs() {
    local from=$1 to=$2 out="" g
    for ((g = from; g < to; g++)); do out="${out:+$out,}Vulkan$g"; done
    echo "$out"
}

run() { echo; echo ">>> $1"; shift; Rscript "$R" "$MODEL" "$@" "$NGEN" "$NREP"; }

echo
echo "=== SINGLE-CONTEXT MODES (sequential) ==="
run "baseline 1 GPU" base "$(devs 0 1)" none
# PP (layer-split) and TP (row-split) on 2, 4, ... up to NGPU (powers of two
# that don't exceed NGPU). Single-context uses ALL listed GPUs in one process.
for ((k = 2; k <= NGPU; k *= 2)); do
    d="$(devs 0 "$k")"
    run "PP layer-split ${k}GPU" "pp$k" "$d" layer
    run "TP row-split ${k}GPU"   "tp$k" "$d" row
done

echo
echo "=== TP=$TP_SIZE x DP=$N_REPLICAS  ($N_REPLICAS replicas CONCURRENTLY) ==="
if [ "$N_REPLICAS" -gt 1 ] || [ "$TP_SIZE" -gt 1 ]; then
    pids=()
    for ((r = 0; r < N_REPLICAS; r++)); do
        from=$((r * TP_SIZE)); to=$(((r + 1) * TP_SIZE))
        d="$(devs "$from" "$to")"
        echo ">>> replica $r on {$d}"
        Rscript "$R" "$MODEL" "tp${TP_SIZE}dp$r" "$d" row "$NGEN" "$NREP" &
        pids+=($!)
    done
    for p in "${pids[@]}"; do wait "$p"; done
fi

echo
echo "=== DP=$NGPU (NGPU single-GPU replicas CONCURRENTLY, no TP) ==="
pids=()
for ((g = 0; g < NGPU; g++)); do
    Rscript "$R" "$MODEL" "dp$g" "Vulkan$g" none "$NGEN" "$NREP" &
    pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done

echo
echo "Done. For DP rows: throughput = sum of the concurrent replicas' decode_tps."
echo "Grep the RESULT lines above; single-context rows are per-config decode t/s."
