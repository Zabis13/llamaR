#!/usr/bin/env Rscript
# Diagnostic: why does graph reuse fail (reused=0)?
#
# Sets verbosity=3 so LLAMA_LOG_DEBUG passes the llamaR log filter, and
# LLAMA_GRAPH_RESULT_DEBUG=2 so llm_graph_result::can_reuse prints, per
# decode step, exactly which check rejects graph reuse.
#
# Run:
#   LLAMA_GRAPH_RESULT_DEBUG=2 Rscript inst/scripts/diag_graph_reuse.R 2>/tmp/reuse_debug.log
# Then:
#   grep -m20 "cannot reuse\|can reuse graph\|can_reuse" /tmp/reuse_debug.log

library(llamaR)

MODEL_PATH <- "/mnt/Data2/DS_projects/llm_models/Ministral-3-3B-Instruct-2512-Q8_0.gguf"
PROMPT     <- "Explain quantum computing in simple terms:"
MAX_TOKENS <- 8L          # few tokens are enough to see the reuse pattern

llama_set_verbosity(3L)   # 3 = show everything incl. DEBUG (default is 1 = errors only)

model <- llama_load_model(MODEL_PATH, n_gpu_layers = -1L)
ctx   <- llama_new_context(model, n_ctx = 512L, n_threads = 8L)

cat("=== warmup ===\n")
invisible(llama_generate(ctx, "Hi", max_new_tokens = 2L, temp = 0))

cat("=== measured generate (watch stderr for can_reuse) ===\n")
res <- llama_generate(ctx, PROMPT, max_new_tokens = MAX_TOKENS, temp = 0)
cat(res, "\n")

llama_free_context(ctx)
llama_free_model(model)
