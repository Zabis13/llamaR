#!/usr/bin/env Rscript
# llamaR GPU Profiling: per-op Vulkan timings inside llama_generate_batch.
# Run: Rscript /mnt/Data2/DS_projects/llamaR/inst/scripts/profile_gpu.R
#
# vk_perf_logger writes via Rprintf -> stdout (not stderr). We capture
# stdout into a log file with sink() during the generate_batch call.

Sys.setenv(GGML_VK_PERF_LOGGER = "1")
LOG_PATH <- "/tmp/llamaR_profile.log"

library(llamaR)

MODEL_PATH <- "/mnt/Data2/DS_projects/llm_models/Ministral-3-3B-Instruct-2512-Q8_0.gguf"
ALL_PROMPTS <- c(
    "Explain quantum computing in simple terms:",
    "Write a short poem about the sea:",
    "List three reasons R is useful for data analysis:",
    "Describe the taste of a ripe mango:"
)
N_PARALLEL <- 1L
PROMPTS    <- ALL_PROMPTS[seq_len(N_PARALLEL)]
MAX_TOKENS <- 50L
N_THREADS  <- 8L

PROMPT_TOK_GUESS <- 10L
N_CTX <- as.integer(N_PARALLEL * (PROMPT_TOK_GUESS + MAX_TOKENS) + 64L)

# Each llama_decode prints one block delimited by "----------------\nVulkan Timings:".
# In llama_generate_batch, the first block is prefill (all prompt tokens at once),
# and the remaining blocks are decode steps (one new token per active sequence).
split_sections <- function(lines) {
    sep_idx <- which(grepl("^-{4,}$", lines))
    if (length(sep_idx) == 0) return(list(all = lines))
    sep_idx <- c(sep_idx, length(lines) + 1L)

    secs <- list()
    for (i in seq_len(length(sep_idx) - 1L)) {
        block <- lines[seq(sep_idx[i] + 1L, sep_idx[i + 1L] - 1L)]
        block <- block[nzchar(trimws(block))]
        if (length(block) == 0) next
        # Drop the "Vulkan Timings:" header and "Total time:" footer
        block <- block[!grepl("^Vulkan Timings:", block)]
        block <- block[!grepl("^Total time:", block)]
        if (length(block) == 0) next
        nm <- if (i == 1L) "prefill" else sprintf("decode_%03d", i - 1L)
        secs[[nm]] <- block
    }
    secs
}

# "OP NAME: 72 x 1523.12 us = 109664.64 us [ (123.4 GFLOPS/s)]"
parse_section <- function(lines) {
    pat  <- "^(.+?):\\s+(\\d+) x ([0-9.]+) us = ([0-9.]+) us"
    hits <- regmatches(lines, regexec(pat, lines))
    rows <- lapply(hits, function(m) {
        if (length(m) < 5) return(NULL)
        data.frame(op       = m[[2]],
                   count    = as.integer(m[[3]]),
                   avg_us   = as.numeric(m[[4]]),
                   total_us = as.numeric(m[[5]]),
                   stringsAsFactors = FALSE)
    })
    do.call(rbind, Filter(Negate(is.null), rows))
}

# Strip mul_mat shape suffix so steps with the same op family aggregate.
op_family <- function(op) {
    sub("\\s+m=.*$", "", op)
}

print_top <- function(df, top_n, label) {
    if (is.null(df) || nrow(df) == 0) return(invisible(NULL))
    df$family <- op_family(df$op)
    agg <- aggregate(cbind(total_us, count) ~ family, data = df, FUN = sum)
    agg <- agg[order(-agg$total_us), ]
    agg$pct <- agg$total_us / sum(agg$total_us) * 100
    total_ms <- sum(agg$total_us) / 1e3
    cat(sprintf("\n--- %s  (total %.1f ms over %d ops) ---\n",
                label, total_ms, sum(agg$count)))
    cat(sprintf("  %-50s %8s %8s %6s\n", "Op family", "ms", "calls", "%"))
    for (i in seq_len(min(top_n, nrow(agg)))) {
        cat(sprintf("  %-50s %8.2f %8d %5.1f%%\n",
                    agg$family[i], agg$total_us[i] / 1e3,
                    agg$count[i], agg$pct[i]))
    }
}

parse_vk_timings <- function(lines, top_n = 10) {
    secs <- split_sections(lines)
    if (length(secs) == 0) {
        cat("No Vulkan timing blocks found in captured stderr.\n")
        return(invisible(NULL))
    }

    # Per-section roll-up: prefill in detail, decode aggregated across all steps
    prefill_lines <- secs$prefill
    decode_names  <- grep("^decode_", names(secs), value = TRUE)
    decode_lines  <- unlist(secs[decode_names], use.names = FALSE)

    if (!is.null(prefill_lines)) {
        print_top(parse_section(prefill_lines), top_n, "prefill (1 decode call)")
    }
    if (length(decode_lines) > 0) {
        n_steps <- length(decode_names)
        df <- parse_section(decode_lines)
        print_top(df, top_n,
                  sprintf("decode aggregate (%d decode calls)", n_steps))

        # Average per-step cost for the heaviest families
        if (!is.null(df) && nrow(df) > 0) {
            df$family <- op_family(df$op)
            agg <- aggregate(total_us ~ family, data = df, FUN = sum)
            agg$avg_per_step_ms <- agg$total_us / 1e3 / n_steps
            agg <- agg[order(-agg$avg_per_step_ms), ]
            cat(sprintf("\n--- decode per-step average (top %d, %d steps) ---\n",
                        top_n, n_steps))
            cat(sprintf("  %-50s %12s\n", "Op family", "ms / step"))
            for (i in seq_len(min(top_n, nrow(agg)))) {
                cat(sprintf("  %-50s %12.3f\n",
                            agg$family[i], agg$avg_per_step_ms[i]))
            }
        }
    }

    # Global summary: prefill + all decode steps combined
    all_df <- do.call(rbind, lapply(secs, parse_section))
    if (!is.null(all_df) && nrow(all_df) > 0) {
        all_df$family <- op_family(all_df$op)
        agg <- aggregate(total_us ~ family, data = all_df, FUN = sum)
        agg <- agg[order(-agg$total_us), ]
        agg$pct <- agg$total_us / sum(agg$total_us) * 100
        cat(sprintf("\n=== GLOBAL TOP %d (%.1f ms GPU total) ===\n",
                    top_n, sum(agg$total_us) / 1e3))
        cat(sprintf("  %-50s %8s %6s\n", "Op family", "ms", "%"))
        for (i in seq_len(min(top_n, nrow(agg)))) {
            cat(sprintf("  %-50s %8.2f %5.1f%%\n",
                        agg$family[i], agg$total_us[i] / 1e3, agg$pct[i]))
        }
        invisible(sum(agg$total_us) / 1e3)  # gpu total in ms
    }
}

# ---- run ---------------------------------------------------------------------

cat(sprintf("=== llamaR GPU Profile (Vulkan per-op, batch N=%d) ===\n",
            N_PARALLEL))
cat(sprintf("Model: %s\n", basename(MODEL_PATH)))
cat(sprintf("Max new tokens (per seq): %d\n", MAX_TOKENS))
cat(sprintf("n_ctx: %d\n", N_CTX))
for (i in seq_along(PROMPTS)) {
    cat(sprintf("  prompt %d: %s\n", i, PROMPTS[i]))
}

if (!llama_supports_gpu()) {
    cat("GPU not available. Install with Vulkan support:\n")
    cat("  R CMD INSTALL --configure-args=\"--with-vulkan\" .\n")
    quit(status = 1L)
}

model_gpu <- llama_load_model(MODEL_PATH, n_gpu_layers = -1L)
ctx_gpu <- llama_new_context(model_gpu,
                             n_ctx     = N_CTX,
                             n_seq_max = N_PARALLEL,
                             n_threads = N_THREADS,
                             flash_attn = "on")

# Warmup: the very first llama_decode pays a one-time cold-start cost
# (lazy Vulkan pipeline creation + weight upload to GPU). Without this
# the measured prefill is dominated by cold-start, not real compute.
# Run a tiny generate first, then reset perf so timings reflect steady state.
cat("Warming up...\n")
invisible(llama_generate_batch(ctx_gpu, PROMPTS,
                               max_new_tokens = 1L,
                               temp = 0, seed = 42L))

llama_perf_reset(ctx_gpu)

# Redirect stdout (where Rprintf-based vk_perf_logger writes) to log file.
log_con <- file(LOG_PATH, open = "wt")
sink(log_con, type = "output")

t0 <- Sys.time()
results <- llama_generate_batch(ctx_gpu, PROMPTS,
                                max_new_tokens = MAX_TOKENS,
                                temp = 0, seed = 42L)
wall_s <- as.numeric(difftime(Sys.time(), t0, units = "secs"))

sink(NULL, type = "output")
close(log_con)

perf <- llama_perf(ctx_gpu)
vk_out <- readLines(LOG_PATH, warn = FALSE)
cat(sprintf("\n[log: %s, %d lines]\n", LOG_PATH, length(vk_out)))

prefill_s <- perf$t_p_eval_ms / 1000
decode_s  <- perf$t_eval_ms   / 1000

cat(sprintf("\n=== generate_batch wall: %.3f s ===\n", wall_s))
cat(sprintf("  llama prefill (n=%d):   %7.3f s  (%.1f tok/s)\n",
            perf$n_p_eval, prefill_s,
            if (prefill_s > 0) perf$n_p_eval / prefill_s else NaN))
cat(sprintf("  llama decode  (n=%d):   %7.3f s  (%.1f tok/s)\n",
            perf$n_eval, decode_s,
            if (decode_s > 0) perf$n_eval / decode_s else NaN))

gpu_total_ms <- parse_vk_timings(vk_out, top_n = 12)

if (!is.null(gpu_total_ms)) {
    overhead_s <- wall_s - gpu_total_ms / 1000
    cat(sprintf("\n=== Wall vs GPU ===\n"))
    cat(sprintf("  Wall:        %7.3f s\n", wall_s))
    cat(sprintf("  GPU (Vulkan): %6.3f s  (sum of all op timings)\n",
                gpu_total_ms / 1000))
    cat(sprintf("  Non-GPU:     %7.3f s  (tokenize + sampling + detok + dispatch)\n",
                overhead_s))
}

cat("\n--- Per-sequence ---\n")
for (i in seq_along(results)) {
    r <- results[[i]]
    cat(sprintf("  seq %d: gen_tok=%d  reason=%s\n",
                i, r$n_tokens, r$finished_reason))
}

llama_free_context(ctx_gpu)
llama_free_model(model_gpu)
