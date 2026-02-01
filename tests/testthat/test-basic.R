MODEL_PATH <- "/mnt/Data2/DS_projects/llm_models/tiny-mistral-test-Q2_K.gguf"

test_that("package loads correctly", {
    expect_true(require(llamaR, quietly = TRUE))
})

test_that("model loads and info is returned", {
    if (!file.exists(MODEL_PATH)) skip("test model not available")

    model <- llama_load_model(MODEL_PATH)
    expect_false(is.null(model))

    info <- llama_model_info(model)
    expect_true(is.list(info))
    expect_true(info$n_vocab  > 0)
    expect_true(info$n_embd   > 0)
    expect_true(info$n_layer  > 0)
    expect_true(info$n_head   > 0)
    expect_true(nchar(info$desc) > 0)

    llama_free_model(model)
})

test_that("context can be created and freed", {
    if (!file.exists(MODEL_PATH)) skip("test model not available")

    model <- llama_load_model(MODEL_PATH)
    ctx <- llama_new_context(model, n_ctx = 128L, n_threads = 2L)
    expect_false(is.null(ctx))

    llama_free_context(ctx)
    llama_free_model(model)
})

test_that("tokenize and detokenize are inverse operations", {
    if (!file.exists(MODEL_PATH)) skip("test model not available")

    model <- llama_load_model(MODEL_PATH)
    ctx <- llama_new_context(model, n_ctx = 128L, n_threads = 2L)

    text   <- "Hello, world!"
    tokens <- llama_tokenize(ctx, text)

    expect_true(is.integer(tokens))
    expect_true(length(tokens) > 0)

    recovered <- llama_detokenize(ctx, tokens)
    expect_true(is.character(recovered))
    expect_equal(recovered, text)

    llama_free_context(ctx)
    llama_free_model(model)
})

test_that("generation produces non-empty output", {
    if (!file.exists(MODEL_PATH)) skip("test model not available")

    model <- llama_load_model(MODEL_PATH)
    ctx <- llama_new_context(model, n_ctx = 256L, n_threads = 2L)

    result <- llama_generate(ctx, "The capital of France is",
                             max_new_tokens = 20L, temp = 0.1)

    expect_true(is.character(result))
    # use type="bytes" — tiny test model (1 layer) produces garbage that may not be valid UTF-8
    expect_true(nchar(result, type = "bytes") > 0)

    llama_free_context(ctx)
    llama_free_model(model)
})

test_that("greedy generation is deterministic", {
    if (!file.exists(MODEL_PATH)) skip("test model not available")

    model <- llama_load_model(MODEL_PATH)
    ctx <- llama_new_context(model, n_ctx = 256L, n_threads = 2L)

    r1 <- llama_generate(ctx, "Once upon a time", max_new_tokens = 30L, temp = 0.0)
    r2 <- llama_generate(ctx, "Once upon a time", max_new_tokens = 30L, temp = 0.0)

    expect_equal(r1, r2)

    llama_free_context(ctx)
    llama_free_model(model)
})

test_that("embeddings have correct dimensionality", {
    if (!file.exists(MODEL_PATH)) skip("test model not available")

    model <- llama_load_model(MODEL_PATH)
    info  <- llama_model_info(model)
    ctx   <- llama_new_context(model, n_ctx = 128L, n_threads = 2L)

    emb <- llama_embeddings(ctx, "Hello")

    expect_true(is.numeric(emb))
    expect_equal(length(emb), info$n_embd)
    # embeddings should not be all zeros
    expect_true(any(emb != 0))

    llama_free_context(ctx)
    llama_free_model(model)
})
