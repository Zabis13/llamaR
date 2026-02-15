MODEL_PATH <- "/mnt/Data2/DS_projects/llm_models/tiny-mistral-test-Q2_K.gguf"
LORA_PATH  <- "/mnt/Data2/DS_projects/llm_models/test-lora-adapter.gguf"

# ============================================================
# Shared fixtures: load model & context once
# ============================================================

HAS_MODEL <- file.exists(MODEL_PATH)

if (HAS_MODEL) {
    shared_model <- llama_load_model(MODEL_PATH)
    shared_info  <- llama_model_info(shared_model)
    shared_ctx   <- llama_new_context(shared_model, n_ctx = 256L, n_threads = 2L)

    withr::defer(llama_free_context(shared_ctx), teardown_env())
    withr::defer(llama_free_model(shared_model), teardown_env())
}

skip_if_no_model <- function() {
    if (!HAS_MODEL) skip("test model not available")
}

# ============================================================
# Package load (no model required)
# ============================================================

test_that("package loads correctly", {
    expect_true(require(llamaR, quietly = TRUE))
})

# ============================================================
# Verbosity (no model required)
# ============================================================

test_that("verbosity can be set and retrieved", {
    old <- llama_get_verbosity()

    llama_set_verbosity(0L)
    expect_equal(llama_get_verbosity(), 0L)

    llama_set_verbosity(3L)
    expect_equal(llama_get_verbosity(), 3L)

    llama_set_verbosity(old)
    expect_equal(llama_get_verbosity(), old)
})

# ============================================================
# Hardware / System info (no model required)
# ============================================================

test_that("llama_supports_gpu returns logical", {
    result <- llama_supports_gpu()
    expect_true(is.logical(result))
    expect_equal(length(result), 1L)
})

test_that("system_info returns non-empty string", {
    info <- llama_system_info()
    expect_true(is.character(info))
    expect_true(nchar(info) > 0)
})

test_that("supports_mmap returns logical", {
    result <- llama_supports_mmap()
    expect_true(is.logical(result))
    expect_equal(length(result), 1L)
})

test_that("supports_mlock returns logical", {
    result <- llama_supports_mlock()
    expect_true(is.logical(result))
    expect_equal(length(result), 1L)
})

test_that("max_devices returns positive integer", {
    result <- llama_max_devices()
    expect_true(is.integer(result))
    expect_true(result >= 1L)
})

test_that("chat_builtin_templates returns character vector", {
    templates <- llama_chat_builtin_templates()
    expect_true(is.character(templates))
    expect_true(length(templates) > 0)
})

# ============================================================
# Model: load + info
# ============================================================

test_that("model loads and info is returned", {
    skip_if_no_model()

    expect_false(is.null(shared_model))

    expect_true(is.list(shared_info))
    expect_true(shared_info$n_vocab > 0)
    expect_true(shared_info$n_embd  > 0)
    expect_true(shared_info$n_layer > 0)
    expect_true(shared_info$n_head  > 0)
    expect_true(nchar(shared_info$desc) > 0)
})

test_that("model_info returns extended fields", {
    skip_if_no_model()

    expect_true(is.numeric(shared_info$size))
    expect_true(shared_info$size > 0)
    expect_true(is.numeric(shared_info$n_params))
    expect_true(shared_info$n_params > 0)
    expect_true(is.logical(shared_info$has_encoder))
    expect_true(is.logical(shared_info$has_decoder))
    expect_true(is.logical(shared_info$is_recurrent))
})

# ============================================================
# Model metadata
# ============================================================

test_that("model_meta returns named character vector", {
    skip_if_no_model()

    meta <- llama_model_meta(shared_model)
    expect_true(is.character(meta))
    expect_true(length(meta) > 0)
    expect_false(is.null(names(meta)))
})

test_that("model_meta_val retrieves values by key", {
    skip_if_no_model()

    arch <- llama_model_meta_val(shared_model, "general.architecture")
    expect_true(is.character(arch) || is.null(arch))

    val <- llama_model_meta_val(shared_model, "nonexistent.key.12345")
    expect_null(val)
})

# ============================================================
# Vocabulary info
# ============================================================

test_that("vocab_info returns named integer vector", {
    skip_if_no_model()

    vocab <- llama_vocab_info(shared_model)
    expect_true(is.integer(vocab))
    expect_equal(length(vocab), 11L)
    expect_true(all(c("bos", "eos", "eot", "sep", "nl", "pad",
                       "fim_pre", "fim_suf", "fim_mid", "fim_rep", "fim_sep")
                     %in% names(vocab)))
})

# ============================================================
# Chat templates
# ============================================================

test_that("chat template can be retrieved from model", {
    skip_if_no_model()

    tmpl <- llama_chat_template(shared_model)
    expect_true(is.null(tmpl) || is.character(tmpl))
})

test_that("chat_apply_template formats messages", {
    skip_if_no_model()

    tmpl <- llama_chat_template(shared_model)
    if (is.null(tmpl)) skip("model has no built-in chat template")

    messages <- list(list(role = "user", content = "Hello"))
    prompt <- llama_chat_apply_template(messages, template = tmpl)

    expect_true(is.character(prompt))
    expect_true(nchar(prompt) > 0)
    expect_true(grepl("Hello", prompt, fixed = TRUE))
})

# ============================================================
# Context: create + config
# ============================================================

test_that("context can be created", {
    skip_if_no_model()
    expect_false(is.null(shared_ctx))
})

test_that("n_ctx returns correct context size", {
    skip_if_no_model()

    n <- llama_n_ctx(shared_ctx)
    expect_true(is.integer(n))
    expect_equal(n, 256L)
})

test_that("set_threads does not error", {
    skip_if_no_model()

    expect_no_error(llama_set_threads(shared_ctx, n_threads = 4L))
    expect_no_error(llama_set_threads(shared_ctx, n_threads = 2L, n_threads_batch = 4L))
    # restore
    llama_set_threads(shared_ctx, n_threads = 2L)
})

test_that("set_causal_attn does not error", {
    skip_if_no_model()

    expect_no_error(llama_set_causal_attn(shared_ctx, FALSE))
    expect_no_error(llama_set_causal_attn(shared_ctx, TRUE))
})

# ============================================================
# Tokenize / Detokenize
# ============================================================

test_that("tokenize and detokenize are inverse operations", {
    skip_if_no_model()

    text   <- "Hello, world!"
    tokens <- llama_tokenize(shared_ctx, text)

    expect_true(is.integer(tokens))
    expect_true(length(tokens) > 0)

    recovered <- llama_detokenize(shared_ctx, tokens)
    expect_true(is.character(recovered))
    expect_equal(recovered, text)
})

# ============================================================
# Generation
# ============================================================

test_that("generation produces non-empty output", {
    skip_if_no_model()

    result <- llama_generate(shared_ctx, "The capital of France is",
                             max_new_tokens = 20L, temp = 0.1)
    expect_true(is.character(result))
    expect_true(nchar(result, type = "bytes") > 0)
})

test_that("greedy generation is deterministic", {
    skip_if_no_model()

    r1 <- llama_generate(shared_ctx, "Once upon a time", max_new_tokens = 30L, temp = 0.0)
    r2 <- llama_generate(shared_ctx, "Once upon a time", max_new_tokens = 30L, temp = 0.0)
    expect_equal(r1, r2)
})

# ============================================================
# Advanced sampling
# ============================================================

test_that("generation with min_p produces output", {
    skip_if_no_model()

    result <- llama_generate(shared_ctx, "Hello", max_new_tokens = 10L,
                             temp = 0.8, min_p = 0.05)
    expect_true(is.character(result))
    expect_true(nchar(result, type = "bytes") > 0)
})

test_that("generation with repeat_penalty produces output", {
    skip_if_no_model()

    result <- llama_generate(shared_ctx, "Hello", max_new_tokens = 10L,
                             temp = 0.8, repeat_penalty = 1.1,
                             repeat_last_n = 32L)
    expect_true(is.character(result))
    expect_true(nchar(result, type = "bytes") > 0)
})

test_that("generation with mirostat v2 produces output", {
    skip_if_no_model()

    result <- llama_generate(shared_ctx, "Hello", max_new_tokens = 10L,
                             mirostat = 2L, mirostat_tau = 5.0,
                             mirostat_eta = 0.1)
    expect_true(is.character(result))
    expect_true(nchar(result, type = "bytes") > 0)
})

test_that("generation with typical_p produces output", {
    skip_if_no_model()

    result <- llama_generate(shared_ctx, "Hello", max_new_tokens = 10L,
                             temp = 0.8, typical_p = 0.9)
    expect_true(is.character(result))
    expect_true(nchar(result, type = "bytes") > 0)
})

# ============================================================
# Embeddings
# ============================================================

test_that("embeddings have correct dimensionality", {
    skip_if_no_model()

    emb <- llama_embeddings(shared_ctx, "Hello")

    expect_true(is.numeric(emb))
    expect_equal(length(emb), shared_info$n_embd)
    expect_true(any(emb != 0))
})

# ============================================================
# Logits
# ============================================================

test_that("get_logits returns numeric vector of n_vocab length", {
    skip_if_no_model()

    llama_generate(shared_ctx, "Hello", max_new_tokens = 1L, temp = 0)

    logits <- llama_get_logits(shared_ctx)
    expect_true(is.numeric(logits))
    expect_equal(length(logits), shared_info$n_vocab)
    expect_true(any(logits != 0))
})

# ============================================================
# KV Cache operations
# ============================================================

test_that("memory_clear works", {
    skip_if_no_model()

    llama_generate(shared_ctx, "Hello", max_new_tokens = 5L, temp = 0)
    expect_no_error(llama_memory_clear(shared_ctx))
})

test_that("memory_seq_rm works", {
    skip_if_no_model()

    llama_generate(shared_ctx, "Hello", max_new_tokens = 5L, temp = 0)
    result <- llama_memory_seq_rm(shared_ctx, seq_id = 0L, p0 = -1L, p1 = -1L)
    expect_true(is.logical(result))
})

test_that("memory_seq_keep works", {
    skip_if_no_model()

    llama_generate(shared_ctx, "Hello", max_new_tokens = 5L, temp = 0)
    expect_no_error(llama_memory_seq_keep(shared_ctx, seq_id = 0L))
})

test_that("memory_seq_pos_range returns named integer", {
    skip_if_no_model()

    range <- llama_memory_seq_pos_range(shared_ctx, seq_id = 0L)
    expect_true(is.integer(range))
    expect_equal(length(range), 2L)
    expect_true(all(c("min", "max") %in% names(range)))
})

test_that("memory_can_shift returns logical", {
    skip_if_no_model()

    result <- llama_memory_can_shift(shared_ctx)
    expect_true(is.logical(result))
    expect_equal(length(result), 1L)
})

# ============================================================
# State save/load
# ============================================================

test_that("state save and load round-trip", {
    skip_if_no_model()

    llama_generate(shared_ctx, "Hello world", max_new_tokens = 5L, temp = 0)

    state_file <- tempfile(fileext = ".bin")
    on.exit(unlink(state_file), add = TRUE)

    result <- llama_state_save(shared_ctx, state_file)
    expect_true(result)
    expect_true(file.exists(state_file))
    expect_true(file.info(state_file)$size > 0)

    ctx2 <- llama_new_context(shared_model, n_ctx = 256L, n_threads = 2L)
    result2 <- llama_state_load(ctx2, state_file)
    expect_true(result2)

    llama_free_context(ctx2)
})

test_that("state_load errors on non-existent file", {
    skip_if_no_model()
    expect_error(llama_state_load(shared_ctx, "nonexistent_state.bin"))
})

# ============================================================
# Performance counters
# ============================================================

test_that("perf returns named list with expected fields", {
    skip_if_no_model()

    llama_generate(shared_ctx, "Hello", max_new_tokens = 5L, temp = 0)

    perf <- llama_perf(shared_ctx)
    expect_true(is.list(perf))
    expect_true(all(c("t_load_ms", "t_p_eval_ms", "t_eval_ms",
                       "n_p_eval", "n_eval", "n_reused") %in% names(perf)))
    expect_true(perf$n_eval > 0)

    expect_no_error(llama_perf_reset(shared_ctx))
})

# ============================================================
# LoRA adapters (separate model load — LoRA modifies model)
# ============================================================

test_that("lora_load returns handle or errors on missing file", {
    skip_if_no_model()

    expect_error(llama_lora_load(shared_model, "nonexistent.gguf"))

    if (file.exists(LORA_PATH)) {
        lora <- llama_lora_load(shared_model, LORA_PATH)
        expect_false(is.null(lora))
    }
})

test_that("lora_apply and lora_remove work on context", {
    skip_if_no_model()
    if (!file.exists(LORA_PATH)) skip("test LoRA adapter not available")

    model <- llama_load_model(MODEL_PATH)
    lora <- llama_lora_load(model, LORA_PATH)
    ctx <- llama_new_context(model, n_ctx = 128L, n_threads = 2L)

    expect_no_error(llama_lora_apply(ctx, lora, scale = 1.0))

    result <- llama_lora_remove(ctx, lora)
    expect_equal(result, 0L)

    llama_free_context(ctx)
    llama_free_model(model)
})

test_that("lora_clear works on context", {
    skip_if_no_model()
    if (!file.exists(LORA_PATH)) skip("test LoRA adapter not available")

    model <- llama_load_model(MODEL_PATH)
    lora <- llama_lora_load(model, LORA_PATH)
    ctx <- llama_new_context(model, n_ctx = 128L, n_threads = 2L)

    llama_lora_apply(ctx, lora, scale = 0.5)
    expect_no_error(llama_lora_clear(ctx))

    result <- llama_lora_remove(ctx, lora)
    expect_equal(result, -1L)

    llama_free_context(ctx)
    llama_free_model(model)
})
