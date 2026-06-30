# ============================================================
# LoRA multi-adapter contract (HEAVY).
# Release 0.2.5 rewrote apply/remove/clear on top of master's
# llama_set_adapters_lora(ctx, adapters**, n, scales*) with a
# per-ctx tracking map. The documented contract:
#   apply(ctx, lora, scale)  -> add/update entry (NULL)
#   remove(ctx, lora)        -> 0 if it was applied, -1 otherwise
#   clear(ctx)               -> drop all; a later remove() -> -1
# Needs a model + a LoRA adapter; listed in tests/testthat.R
# `heavy` so it is skipped on CRAN. Skips locally until the
# adapter file exists.
# ============================================================

MODEL_PATH <- "/mnt/Data2/DS_projects/llm_models/tiny-mistral-test-Q2_K.gguf"
LORA_PATH  <- "/mnt/Data2/DS_projects/llm_models/test-lora-adapter.gguf"

skip_if_no_lora <- function() {
    if (!file.exists(MODEL_PATH)) skip("test model not available")
    if (!file.exists(LORA_PATH))  skip("test LoRA adapter not available")
}

test_that("remove returns -1 for an adapter that was never applied", {
    skip_if_no_lora()
    model <- llama_load_model(MODEL_PATH)
    ctx   <- llama_new_context(model, n_ctx = 128L, n_threads = 2L)
    on.exit({ llama_free_context(ctx); llama_free_model(model) }, add = TRUE)

    lora <- llama_lora_load(model, LORA_PATH)
    # never applied to this ctx -> -1, and it must NOT error.
    expect_identical(llama_lora_remove(ctx, lora), -1L)
})

test_that("apply then remove returns 0, and a second remove returns -1", {
    skip_if_no_lora()
    model <- llama_load_model(MODEL_PATH)
    ctx   <- llama_new_context(model, n_ctx = 128L, n_threads = 2L)
    on.exit({ llama_free_context(ctx); llama_free_model(model) }, add = TRUE)

    lora <- llama_lora_load(model, LORA_PATH)
    llama_lora_apply(ctx, lora, scale = 1.0)

    expect_identical(llama_lora_remove(ctx, lora), 0L)   # was active
    expect_identical(llama_lora_remove(ctx, lora), -1L)  # now gone
})

test_that("re-applying the same adapter updates the scale (single entry)", {
    skip_if_no_lora()
    model <- llama_load_model(MODEL_PATH)
    ctx   <- llama_new_context(model, n_ctx = 128L, n_threads = 2L)
    on.exit({ llama_free_context(ctx); llama_free_model(model) }, add = TRUE)

    lora <- llama_lora_load(model, LORA_PATH)
    llama_lora_apply(ctx, lora, scale = 0.5)
    llama_lora_apply(ctx, lora, scale = 1.0)   # update, not a duplicate

    # a single remove clears the (single) tracked entry -> 0, then -1.
    expect_identical(llama_lora_remove(ctx, lora), 0L)
    expect_identical(llama_lora_remove(ctx, lora), -1L)
})

test_that("clear drops all adapters; a later remove returns -1", {
    skip_if_no_lora()
    model <- llama_load_model(MODEL_PATH)
    ctx   <- llama_new_context(model, n_ctx = 128L, n_threads = 2L)
    on.exit({ llama_free_context(ctx); llama_free_model(model) }, add = TRUE)

    lora <- llama_lora_load(model, LORA_PATH)
    llama_lora_apply(ctx, lora, scale = 1.0)
    expect_no_error(llama_lora_clear(ctx))

    # everything cleared -> the previously-active adapter is gone.
    expect_identical(llama_lora_remove(ctx, lora), -1L)
})
