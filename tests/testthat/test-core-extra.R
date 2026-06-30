# ============================================================
# Core engine functions not exercised elsewhere (HEAVY).
#   llama_generate_batch()    — multi-prompt generation
#   llama_memory_seq_add()    — position shift (context sliding)
#   llama_get_embeddings_seq()— pooled per-sequence embedding
# Needs a model; listed in tests/testthat.R `heavy` so it is
# skipped on CRAN. Run locally with NOT_CRAN=true.
# ============================================================

MODEL_PATH <- "/mnt/Data2/DS_projects/llm_models/tiny-mistral-test-Q2_K.gguf"

skip_if_no_model <- function() {
    if (!file.exists(MODEL_PATH)) skip("test model not available")
}

test_that("llama_generate_batch returns one completion per prompt", {
    skip_if_no_model()
    model <- llama_load_model(MODEL_PATH)
    # batch generation needs one sequence slot per prompt.
    ctx   <- llama_new_context(model, n_ctx = 256L, n_threads = 2L,
                               n_seq_max = 2L)
    on.exit({ llama_free_context(ctx); llama_free_model(model) }, add = TRUE)

    prompts <- c("The capital of France is", "Two plus two equals")
    out <- llama_generate_batch(ctx, prompts, max_new_tokens = 8L, temp = 0.0)

    # one result list per prompt, each carrying text / n_tokens / finished_reason.
    expect_type(out, "list")
    expect_length(out, length(prompts))
    for (r in out) {
        expect_true(all(c("text", "n_tokens", "finished_reason") %in% names(r)))
        expect_type(r$text, "character")
    }
})

test_that("llama_generate_batch rejects an empty prompt vector", {
    skip_if_no_model()
    model <- llama_load_model(MODEL_PATH)
    ctx   <- llama_new_context(model, n_ctx = 128L, n_threads = 2L)
    on.exit({ llama_free_context(ctx); llama_free_model(model) }, add = TRUE)

    # stopifnot(length(prompts) >= 1) guards this.
    expect_error(llama_generate_batch(ctx, character(0)))
})

test_that("llama_memory_seq_add shifts positions without error", {
    skip_if_no_model()
    model <- llama_load_model(MODEL_PATH)
    ctx   <- llama_new_context(model, n_ctx = 256L, n_threads = 2L)
    on.exit({ llama_free_context(ctx); llama_free_model(model) }, add = TRUE)

    # Prime the KV cache with a short generation, then shift seq 0 positions.
    llama_generate(ctx, "Hello world", max_new_tokens = 4L, temp = 0.0)
    expect_no_error(
        llama_memory_seq_add(ctx, seq_id = 0L, p0 = 0L, p1 = -1L, delta = 1L))
    # invisibly returns NULL.
    expect_null(
        llama_memory_seq_add(ctx, seq_id = 0L, p0 = 0L, p1 = -1L, delta = -1L))
})

test_that("llama_get_embeddings_seq yields a pooled vector in embedding mode", {
    skip_if_no_model()
    model <- llama_load_model(MODEL_PATH)
    # embedding=TRUE enables pooling so seq-level embeddings are available.
    ctx   <- llama_new_context(model, n_ctx = 256L, n_threads = 2L,
                               embedding = TRUE)
    on.exit({ llama_free_context(ctx); llama_free_model(model) }, add = TRUE)

    # llama_embed_batch decodes with per-sequence pooling; seq 0 is the first text.
    emb <- tryCatch(
        llama_embed_batch(ctx, "Hello world"),
        error = function(e) skip(paste("model lacks pooled embeddings:",
                                       conditionMessage(e))))

    seq_emb <- llama_get_embeddings_seq(ctx, 0L)
    expect_type(seq_emb, "double")
    expect_gt(length(seq_emb), 0L)
    expect_false(anyNA(seq_emb))
})
