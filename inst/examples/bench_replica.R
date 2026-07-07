#!/usr/bin/env Rscript
# ---------------------------------------------------------------------------
# One llamaR replica for the TP/DP benchmark. Runs in its OWN process, so its
# Vulkan / ggml singleton state is isolated from any other replica (required
# for correct concurrent DP — see ggmlR TODO note on per-process device state).
#
# Loads the model on the given Vulkan devices with the given split mode, times
# N decode passes, and prints ONE machine-parseable result line:
#
#   RESULT <tag> devices=<..> split=<..> decode_tps=<..> prefill_ms=<..> total_ms=<..>
#
# Usage:
#   Rscript replica.R <model.gguf> <tag> <dev_csv> <split_mode> [n_gen] [n_rep]
# e.g.
#   Rscript replica.R model.gguf A Vulkan0,Vulkan1 row 128 3
# ---------------------------------------------------------------------------

suppressMessages(library(llamaR))

a <- commandArgs(trailingOnly = TRUE)
if (length(a) < 4)
    stop("usage: replica.R <model.gguf> <tag> <dev_csv> <split> [n_gen] [n_rep]")
MODEL <- path.expand(a[[1]]); TAG <- a[[2]]
DEVS  <- strsplit(a[[3]], ",", fixed = TRUE)[[1]]
SPLIT <- a[[4]]
N_GEN <- if (length(a) >= 5) as.integer(a[[5]]) else 128L
N_REP <- if (length(a) >= 6) as.integer(a[[6]]) else 3L
stopifnot(file.exists(MODEL), llama_supports_gpu())

PROMPT <- list(list(role = "user",
    content = "Write a detailed paragraph about the history of computing."))

model <- llama_load_model(MODEL, n_gpu_layers = -1L, devices = DEVS, split_mode = SPLIT)
ctx   <- llama_new_context(model, n_ctx = 2048L)
built <- llama_chat_build(model, PROMPT)

# Warm-up (shader compile / cache priming) — not measured.
invisible(llama_generate(ctx, built$prompt, max_new_tokens = 8L, temp = 0.0, seed = 1L))

dec_tps <- numeric(N_REP); pre_ms <- numeric(N_REP); tot_ms <- numeric(N_REP)
for (i in seq_len(N_REP)) {
    out <- llama_generate(ctx, built$prompt, max_new_tokens = N_GEN,
                          temp = 0.0, seed = 42L, with_timings = TRUE)
    tm <- attr(out, "timings")
    dec_s <- tm[["t_decode_dispatch_ms"]] / 1000
    dec_tps[i] <- if (dec_s > 0) tm[["n_iterations"]] / dec_s else NA_real_
    pre_ms[i] <- tm[["t_prefill_dispatch_ms"]]; tot_ms[i] <- tm[["t_total_ms"]]
}

llama_free_context(ctx); llama_free_model(model)

cat(sprintf("RESULT %s devices=%s split=%s decode_tps=%.1f prefill_ms=%.1f total_ms=%.1f\n",
            TAG, paste(DEVS, collapse = ","), SPLIT,
            median(dec_tps, na.rm = TRUE), median(pre_ms), median(tot_ms)))

ggml_vulkan_shutdown(hard = TRUE, status = 0L)   # clean exit, no f1ba0
