#!/usr/bin/env Rscript
# Helper invoked by profile_vs_llamacpp.sh. Runs the same bench setup as
# benchmark_compare.R but expects model path / n_gen / n_runs from argv.
args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) >= 3)
MODEL_PATH <- args[[1]]
N_GEN      <- as.integer(args[[2]])
N_RUNS     <- as.integer(args[[3]])

library(llamaR)

N_CTX     <- 4096L
N_BATCH   <- 2048L
N_UBATCH  <- 512L
N_THREADS <- {
  out <- tryCatch(system2("lscpu", "-p=CORE", stdout = TRUE, stderr = FALSE),
                  error = function(e) character())
  cores <- out[!startsWith(out, "#")]
  if (length(cores)) length(unique(cores)) else parallel::detectCores(logical = FALSE)
}

cat(sprintf("=== llamaR profile run ===\nmodel: %s\nn_gen: %d  n_runs: %d  n_threads: %d\n\n",
            basename(MODEL_PATH), N_GEN, N_RUNS, N_THREADS))

model <- llama_load_model(MODEL_PATH, n_gpu_layers = -1L)
ctx   <- llama_new_context(model,
                           n_ctx      = N_CTX,
                           n_batch    = N_BATCH,
                           n_ubatch   = N_UBATCH,
                           n_threads  = N_THREADS,
                           flash_attn = "on")

# warmup
invisible(llama_generate(ctx, " ", max_new_tokens = 1L, temp = 0))

for (i in seq_len(N_RUNS)) {
  llama_perf_reset(ctx)
  t0 <- Sys.time()
  invisible(llama_generate(ctx, " ", max_new_tokens = N_GEN, temp = 0))
  wall_ms <- as.numeric(difftime(Sys.time(), t0, units = "secs")) * 1000
  p <- llama_perf(ctx)
  tg <- if (p$t_eval_ms > 0) p$n_eval / (p$t_eval_ms / 1000) else NA_real_
  cat(sprintf("run %d: tg=%.2f t/s  wall=%.1fms  t_eval=%.1fms  n_g=%d  reused=%s\n",
              i, tg, wall_ms, p$t_eval_ms, p$n_eval,
              if (is.null(p$n_reused)) "?" else as.character(p$n_reused)))
}

llama_free_context(ctx)
llama_free_model(model)
