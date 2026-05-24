suppressPackageStartupMessages(library(llamaR))

MODEL_PATH <- "/mnt/Data2/DS_projects/llm_models/Ministral-3-3B-Instruct-2512-Q4_K_M.gguf"
stopifnot(file.exists(MODEL_PATH))

llama_set_verbosity(0L)

BATCH_SIZES <- c(1L, 2L, 4L, 8L, 16L)
MAX_NEW     <- 128L
PROMPT      <- "Write a short story about a robot:"
N_RUNS      <- 3L

PROMPT_TOKENS_GUESS <- 12L  # rough; n_ctx is sized generously below

cat(sprintf("Model         : %s\n", basename(MODEL_PATH)))
cat(sprintf("Batch sizes   : %s\n", paste(BATCH_SIZES, collapse = ", ")))
cat(sprintf("max_new_tokens: %d\n", MAX_NEW))
cat(sprintf("Prompt        : %s\n", PROMPT))
cat(sprintf("N runs        : %d (median reported)\n\n", N_RUNS))

cat("=== loading model ===\n")
model <- llama_load_model(MODEL_PATH, n_gpu_layers = -1L)

# ---- baseline: sequential N calls via llama_generate ----
ctx_seq <- llama_new_context(model, n_ctx = 512L, n_seq_max = 1L,
                             flash_attn = "on")
invisible(llama_generate(ctx_seq, PROMPT, max_new_tokens = 8L,
                         temp = 0, seed = 1L, mirostat = 0L))

run_sequential <- function(n_par) {
    n_tokens_total <- 0L
    finish_times   <- numeric(n_par)
    decode_total   <- 0
    t_start <- Sys.time()
    for (s in seq_len(n_par)) {
        out <- llama_generate(ctx_seq, PROMPT, max_new_tokens = MAX_NEW,
                              temp = 0, seed = 42L + s - 1L,
                              mirostat = 0L, with_timings = TRUE)
        tim <- attr(out, "timings")
        n_tokens_total <- n_tokens_total + as.integer(tim["n_iterations"])
        decode_total <- decode_total +
            tim["t_decode_dispatch_ms"] + tim["t_post_decode_sync_ms"] +
            tim["t_gpu_sync_ms"] + tim["t_sample_ms"]
        finish_times[s] <- as.numeric(difftime(Sys.time(), t_start, units = "secs"))
    }
    t_total <- as.numeric(difftime(Sys.time(), t_start, units = "secs"))
    list(wall = t_total, last_finish = max(finish_times),
         n_tokens = n_tokens_total,
         decode_ms_sum = as.numeric(decode_total))
}

run_batch <- function(ctx, n_par) {
    t0 <- Sys.time()
    out <- llama_generate_batch(ctx, rep(PROMPT, n_par),
                                max_new_tokens = MAX_NEW,
                                temp = 0, seed = 42L)
    t_total <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
    n_tokens_total <- sum(vapply(out, `[[`, integer(1), "n_tokens"))
    list(wall = t_total, last_finish = t_total, n_tokens = n_tokens_total)
}

median_field <- function(xs, f) median(vapply(xs, function(r) r[[f]], numeric(1)))

# ---- collect: sequential baseline + batch for each N ----
results <- list()

for (N in BATCH_SIZES) {
    cat(sprintf("\n=== N = %d ===\n", N))

    cat("  sequential:\n")
    seq_runs <- replicate(N_RUNS, run_sequential(N), simplify = FALSE)
    for (i in seq_along(seq_runs)) {
        r <- seq_runs[[i]]
        cat(sprintf("    run %d: wall=%.3fs  tokens=%d  tok/s=%.2f\n",
                    i, r$wall, r$n_tokens, r$n_tokens / r$wall))
    }

    n_ctx_batch <- as.integer(N * (PROMPT_TOKENS_GUESS + MAX_NEW + 16L))
    ctx_batch <- llama_new_context(model, n_ctx = n_ctx_batch, n_seq_max = N,
                                   flash_attn = "on")
    invisible(llama_generate_batch(ctx_batch, rep(PROMPT, N),
                                   max_new_tokens = 8L, temp = 0, seed = 1L))

    cat("  batch:\n")
    batch_runs <- replicate(N_RUNS, run_batch(ctx_batch, N), simplify = FALSE)
    for (i in seq_along(batch_runs)) {
        r <- batch_runs[[i]]
        cat(sprintf("    run %d: wall=%.3fs  tokens=%d  tok/s=%.2f\n",
                    i, r$wall, r$n_tokens, r$n_tokens / r$wall))
    }
    llama_free_context(ctx_batch)

    results[[as.character(N)]] <- list(
        N            = N,
        seq_wall     = median_field(seq_runs,   "wall"),
        seq_tok      = median_field(seq_runs,   "n_tokens"),
        seq_decode_s = median_field(seq_runs,   "decode_ms_sum") / 1000,
        seq_last     = median_field(seq_runs,   "last_finish"),
        bat_wall     = median_field(batch_runs, "wall"),
        bat_tok      = median_field(batch_runs, "n_tokens"),
        bat_last     = median_field(batch_runs, "last_finish")
    )
}

llama_free_context(ctx_seq)
llama_free_model(model)

# ============================================================
# Summary table
# ============================================================
cat("\n=== summary (median over runs) ===\n\n")

cat(sprintf("%4s | %16s | %14s | %8s\n",
            "N", "seq tok/s (wall)", "batch tok/s", "speedup"))
cat(strrep("-", 56), "\n", sep = "")
for (r in results) {
    seq_tps <- r$seq_tok / r$seq_wall
    bat_tps <- r$bat_tok / r$bat_wall
    cat(sprintf("%4d | %16.2f | %14.2f | %7.2fx\n",
                r$N, seq_tps, bat_tps, bat_tps / seq_tps))
}

cat("\nLatency-to-last (seconds):\n")
cat(sprintf("%4s | %12s | %12s | %8s\n", "N", "sequential", "batch", "speedup"))
cat(strrep("-", 48), "\n", sep = "")
for (r in results) {
    cat(sprintf("%4d | %12.3f | %12.3f | %7.2fx\n",
                r$N, r$seq_last, r$bat_last, r$seq_last / r$bat_last))
}

cat("\nDecode-only tok/s (sequential, excl. prefill):\n")
for (r in results) {
    cat(sprintf("  N=%d : %.2f tok/s\n", r$N, r$seq_tok / r$seq_decode_s))
}

# ============================================================
# Final combined table — one row per N with everything visible
# ============================================================
cat("\n=== FINAL TABLE ===\n\n")

# baseline tok/s for scaling efficiency = sequential single-prompt rate (N=1 batch)
baseline <- results[[1]]
baseline_tps <- baseline$bat_tok / baseline$bat_wall

cat(sprintf("%3s | %10s | %10s | %8s | %10s | %10s | %8s | %10s\n",
            "N", "seq tok/s", "bat tok/s", "speedup",
            "seq wall", "bat wall", "lat x", "scal eff"))
cat(strrep("-", 90), "\n", sep = "")
for (r in results) {
    seq_tps  <- r$seq_tok / r$seq_wall
    bat_tps  <- r$bat_tok / r$bat_wall
    speedup  <- bat_tps / seq_tps
    lat_x    <- r$seq_last / r$bat_last
    # scaling efficiency: how close batch tok/s is to N x baseline
    scal_eff <- bat_tps / (r$N * baseline_tps)
    cat(sprintf("%3d | %10.2f | %10.2f | %7.2fx | %9.2fs | %9.2fs | %7.2fx | %8.0f%%\n",
                r$N, seq_tps, bat_tps, speedup,
                r$seq_wall, r$bat_wall, lat_x, 100 * scal_eff))
}

cat("\nColumns:\n")
cat("  seq tok/s : sequential N x llama_generate, total tokens / wall\n")
cat("  bat tok/s : single llama_generate_batch(N prompts), total tokens / wall\n")
cat("  speedup   : bat_tps / seq_tps\n")
cat("  seq/bat wall: median wall-clock seconds for N prompts\n")
cat("  lat x     : how much sooner the last-finishing prompt completes in batch\n")
cat("  scal eff  : bat_tps / (N x N=1 baseline) -- 100% = perfect linear scaling\n")
