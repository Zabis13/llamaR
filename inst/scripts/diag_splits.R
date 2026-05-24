#!/usr/bin/env Rscript
# Diagnostic: which ops break the decode graph into 341 splits?
# GGML_SCHED_DEBUG=2 makes ggml_backend_sched_print_assignments dump every
# node, its assigned backend, and SPLIT boundaries.
#
# Run:
#   GGML_SCHED_DEBUG=2 Rscript inst/scripts/diag_splits.R 2>/tmp/splits.log

library(llamaR)

MODEL_PATH <- "/mnt/Data2/DS_projects/llm_models/Ministral-3-3B-Instruct-2512-Q8_0.gguf"

llama_set_verbosity(3L)   # let DEBUG through llamaR's log filter

model <- llama_load_model(MODEL_PATH, n_gpu_layers = -1L)
ctx   <- llama_new_context(model, n_ctx = 512L, n_threads = 8L)

invisible(llama_generate(ctx, "Hi", max_new_tokens = 1L, temp = 0))  # warmup
cat("=== measured: 2 decode tokens ===\n")
res <- llama_generate(ctx, "Explain quantum computing:", max_new_tokens = 2L, temp = 0)
cat(res, "\n")

llama_free_context(ctx)
llama_free_model(model)
