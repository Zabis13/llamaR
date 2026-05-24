#!/usr/bin/env bash
# Side-by-side profile: llama-bench vs llamaR on the same model/params.
# Captures VK_PERF_LOGGER (both), SCHED_PROF + VKG_PROF (llamaR only).
# Run as:  bash profile_vs_llamacpp.sh
set -u

MODEL="/mnt/Data2/DS_projects/llm_models/Ministral-3-3B-Instruct-2512-Q8_0.gguf"
LLAMA_BENCH="/mnt/Data2/DS_projects/llama-b9245/llama-bench"
N_GEN=30
N_RUNS=2
OUTDIR="/tmp/llamaR_vs_bench_$(date +%H%M%S)"
mkdir -p "$OUTDIR"

if [[ ! -f "$MODEL" ]]; then echo "missing model $MODEL"; exit 1; fi
if [[ ! -x "$LLAMA_BENCH" ]]; then echo "missing llama-bench $LLAMA_BENCH"; exit 1; fi

echo "=== outdir: $OUTDIR ==="

echo ""
echo "### 1/2  llama-bench (upstream) -- VK_PERF_LOGGER ###"
GGML_VK_PERF_LOGGER=1 \
  "$LLAMA_BENCH" -m "$MODEL" -ngl 99 -p 0 -n "$N_GEN" -r "$N_RUNS" -fa 1 \
  2>&1 | tee "$OUTDIR/upstream.log"

echo ""
echo "### 2/2  llamaR -- VK_PERF_LOGGER + SCHED_PROF + VKG_PROF ###"
GGML_VK_PERF_LOGGER=1 GGML_SCHED_PROF=1 GGML_VKG_PROF=1 \
  Rscript /mnt/Data2/DS_projects/llamaR/inst/scripts/profile_vs_llamacpp.R \
  "$MODEL" "$N_GEN" "$N_RUNS" \
  2>&1 | tee "$OUTDIR/llamaR.log"

echo ""
echo "=== logs in $OUTDIR ==="
echo "Compare with:"
echo "  diff <(grep -E 'tok/s|kernel_time|PERF' '$OUTDIR/upstream.log') <(grep -E 'tok/s|kernel_time|PERF|CS_|VKG_|SCHED_' '$OUTDIR/llamaR.log')"
