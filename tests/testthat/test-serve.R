MODEL_PATH <- "/mnt/Data2/DS_projects/llm_models/tiny-mistral-test-Q2_K.gguf"

# ============================================================
# OpenAI server: response shapes and input guards.
# Fast unit tests only — no model load, no network. End-to-end
# serving is exercised by inst/examples/serve_openai.R.
# ============================================================

test_that(".openai_completion has the expected OpenAI shape", {
    out <- llamaR:::.openai_completion(
        id = "chatcmpl-x", model_id = "m", text = "hello",
        created = 123L, prompt_tokens = 5L, completion_tokens = 2L,
        finish_reason = "stop")

    expect_identical(out$object, "chat.completion")
    expect_identical(out$id, "chatcmpl-x")
    expect_identical(out$model, "m")
    expect_length(out$choices, 1L)
    expect_identical(out$choices[[1]]$message$role, "assistant")
    expect_identical(out$choices[[1]]$message$content, "hello")
    expect_identical(out$choices[[1]]$finish_reason, "stop")
    expect_identical(out$usage$prompt_tokens, 5L)
    expect_identical(out$usage$completion_tokens, 2L)
    expect_identical(out$usage$total_tokens, 7L)
})

test_that(".openai_chunk content chunk serialises with finish_reason null", {
    skip_if_not_installed("jsonlite")
    obj <- llamaR:::.openai_chunk("id1", "m", 1L, list(content = "hi"))
    js  <- as.character(jsonlite::toJSON(obj, auto_unbox = TRUE))

    expect_identical(obj$object, "chat.completion.chunk")
    expect_identical(obj$choices[[1]]$delta$content, "hi")
    # non-final chunk: finish_reason must be JSON null, not {} or missing
    expect_match(js, '"finish_reason":null')
})

test_that(".openai_chunk role chunk carries assistant role", {
    skip_if_not_installed("jsonlite")
    obj <- llamaR:::.openai_chunk("id1", "m", 1L, list(role = "assistant"))
    js  <- as.character(jsonlite::toJSON(obj, auto_unbox = TRUE))

    expect_match(js, '"delta":\\{"role":"assistant"\\}')
    expect_match(js, '"finish_reason":null')
})

test_that(".openai_chunk finishing chunk serialises delta as empty object", {
    skip_if_not_installed("jsonlite")
    obj <- llamaR:::.openai_chunk("id1", "m", 1L,
                                  structure(list(), names = character()),
                                  finish_reason = "length")
    js  <- as.character(jsonlite::toJSON(obj, auto_unbox = TRUE))

    expect_match(js, '"delta":\\{\\}')
    expect_match(js, '"finish_reason":"length"')
})

test_that("llama_serve_openai errors clearly when drogonR is missing", {
    # Only meaningful when drogonR is genuinely absent; otherwise skip.
    if (requireNamespace("drogonR", quietly = TRUE)) {
        skip("drogonR is installed; cannot exercise the missing-package guard")
    }
    expect_error(llama_serve_openai(MODEL_PATH), "drogonR")
})

test_that("llama_serve_openai errors on a missing model file", {
    skip_if_not_installed("drogonR")
    expect_error(
        llama_serve_openai("/no/such/model.gguf"),
        "model file not found")
})
