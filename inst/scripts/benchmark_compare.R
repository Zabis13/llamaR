#!/usr/bin/env Rscript
# Mirrors `llama-bench -m <model> -ngl 99 -p 0 -n 30 -r 2 -fa 0,1`
# using llama_perf() so the tg t/s metric excludes tokenize+sample
# (same as llama-bench internally).

library(llamaR)

MODEL_PATH <- "/mnt/Data2/DS_projects/llm_models/Qwen3-14B-Q8_0.gguf"
N_GEN      <- 30L
N_RUNS     <- 2L
N_CTX      <- 4096L
N_BATCH    <- 2048L
N_UBATCH   <- 512L
FA_VALUES  <- c("on")

# llama-bench defaults to physical cores; match it.
# parallel::detectCores(logical=FALSE) is unreliable on Linux (often returns
# logical CPUs), so parse lscpu when available and fall back if not.
detect_physical_cores <- function() {
  if (.Platform$OS.type == "unix" && nzchar(Sys.which("lscpu"))) {
    out <- tryCatch(system2("lscpu", "-p=CORE", stdout = TRUE, stderr = FALSE),
                    error = function(e) character())
    cores <- out[!startsWith(out, "#")]
    if (length(cores)) return(length(unique(cores)))
  }
  n <- parallel::detectCores(logical = FALSE)
  if (is.na(n)) n <- parallel::detectCores(logical = TRUE)
  if (is.na(n)) n <- 4L
  as.integer(n)
}
N_THREADS_MAX <- detect_physical_cores()

THREAD_SWEEP <- N_THREADS_MAX

stopifnot(file.exists(MODEL_PATH))

cat("=== llamaR vs llama-bench (mirrored) ===\n")
cat(sprintf("Model:    %s\n", basename(MODEL_PATH)))
cat(sprintf("n_gen:    %d   n_runs: %d   n_ctx: %d   n_batch/ub: %d/%d\n",
            N_GEN, N_RUNS, N_CTX, N_BATCH, N_UBATCH))
cat(sprintf("threads sweep: %s   fa: %s\n\n",
            paste(THREAD_SWEEP, collapse = ","),
            paste(FA_VALUES,    collapse = ",")))

run_one <- function(ctx) {
  llama_perf_reset(ctx)
  # short greedy generation; llama_perf separates prefill/decode timings,
  # so the prompt size doesn't pollute the tg measurement.
  invisible(llama_generate(ctx, " ", max_new_tokens = N_GEN, temp = 0))
  p <- llama_perf(ctx)
  list(
    pp_tps   = if (p$t_p_eval_ms > 0) p$n_p_eval / (p$t_p_eval_ms / 1000) else NA_real_,
    tg_tps   = if (p$t_eval_ms  > 0) p$n_eval   / (p$t_eval_ms  / 1000) else NA_real_,
    n_p      = p$n_p_eval,
    n_g      = p$n_eval,
    t_eval   = p$t_eval_ms,
    n_reused = if (!is.null(p$n_reused)) p$n_reused else NA_integer_
  )
}

bench_one <- function(model, fa_label, n_threads) {
  ctx <- llama_new_context(model,
                           n_ctx      = N_CTX,
                           n_batch    = N_BATCH,
                           n_ubatch   = N_UBATCH,
                           n_threads  = n_threads,
                           flash_attn = fa_label)

  # warmup: one decode to prime kernels/pipelines (llama-bench does the same)
  invisible(llama_generate(ctx, " ", max_new_tokens = 1L, temp = 0))

  tg <- numeric(N_RUNS)
  for (i in seq_len(N_RUNS)) {
    r <- run_one(ctx)
    tg[i] <- r$tg_tps
    cat(sprintf("  run %d: tg=%.2f t/s  (n_p=%d, n_g=%d, t_eval=%.1fms, reused=%s)\n",
                i, r$tg_tps, r$n_p, r$n_g, r$t_eval,
                if (is.na(r$n_reused)) "NA" else as.character(r$n_reused)))
  }

  llama_free_context(ctx)

  avg <- mean(tg); sdv <- if (N_RUNS > 1) sd(tg) else 0
  cat(sprintf("=> fa=%-3s  n_threads=%d  tg %d : %.2f ± %.2f t/s\n\n",
              fa_label, n_threads, N_GEN, avg, sdv))
  list(fa = fa_label, n_threads = n_threads, avg = avg, sd = sdv)
}

model <- llama_load_model(MODEL_PATH, n_gpu_layers = -1L)

results <- list()
for (fa in FA_VALUES) {
  for (nt in THREAD_SWEEP) {
    cat(sprintf("--- flash_attn = %s   n_threads = %d ---\n", fa, nt))
    results[[length(results) + 1L]] <- bench_one(model, fa, nt)
  }
}

llama_free_model(model)

cat("=== summary ===\n")
cat(sprintf("%-3s  %-9s  %-8s  %s\n", "fa", "n_threads", "tg t/s", "± sd"))
for (r in results) {
  cat(sprintf("%-3s  %-9d  %8.2f  ± %.2f\n",
              r$fa, r$n_threads, r$avg, r$sd))
}
