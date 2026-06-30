# ============================================================
# Tool-aware chat layer — round-trip (HEAVY).
# Builds a prompt with llama_chat_build(), generates with the
# model, and parses the output back with llama_chat_parse().
# Needs a real model; listed in tests/testthat.R `heavy` so it
# is skipped on CRAN. Run locally with NOT_CRAN=true.
# ============================================================

MODEL_PATH <- "/mnt/Data2/DS_projects/llm_models/tiny-mistral-test-Q2_K.gguf"

skip_if_no_model <- function() {
    if (!file.exists(MODEL_PATH)) skip("test model not available")
}

test_that("llama_chat_build returns a prompt and a format id", {
    skip_if_no_model()
    model <- llama_load_model(MODEL_PATH)
    on.exit(llama_free_model(model), add = TRUE)

    messages <- list(
        list(role = "system", content = "You are a helpful assistant."),
        list(role = "user",   content = "Say hi."))
    built <- llama_chat_build(model, messages)

    expect_type(built, "list")
    expect_true("prompt" %in% names(built))
    expect_type(built$prompt, "character")
    expect_gt(nchar(built$prompt), 0L)
    # format id drives llama_chat_parse(); must be an integer scalar.
    expect_true("format" %in% names(built))
    expect_length(built$format, 1L)
    # both system and user content must reach the prompt.
    expect_true(grepl("helpful assistant", built$prompt, fixed = TRUE))
    expect_true(grepl("Say hi", built$prompt, fixed = TRUE))
})

test_that("build -> generate -> parse yields parseable content", {
    skip_if_no_model()
    model <- llama_load_model(MODEL_PATH)
    ctx   <- llama_new_context(model, n_ctx = 512L, n_threads = 2L)
    on.exit({ llama_free_context(ctx); llama_free_model(model) }, add = TRUE)

    messages <- list(list(role = "user", content = "Reply with a short greeting."))
    built <- llama_chat_build(model, messages)

    out <- llama_generate(ctx, built$prompt, max_new_tokens = 16L, temp = 0.0)
    expect_type(out, "character")

    parsed <- llama_chat_parse(out, format = built$format,
                               parser = built$parser)
    expect_type(parsed, "list")
    expect_true(all(c("content", "tool_calls") %in% names(parsed)))
    expect_s3_class(parsed$tool_calls, "data.frame")
    # plain chat (no tools) -> no tool calls.
    expect_identical(nrow(parsed$tool_calls), 0L)
})

test_that("llama_chat_build accepts tool definitions and a grammar", {
    skip_if_no_model()
    model <- llama_load_model(MODEL_PATH)
    on.exit(llama_free_model(model), add = TRUE)

    tools <- list(list(
        type = "function",
        "function" = list(
            name = "get_weather",
            description = "Get the weather for a city.",
            parameters = list(
                type = "object",
                properties = list(city = list(type = "string")),
                required = list("city")))))
    messages <- list(list(role = "user", content = "Weather in Paris?"))

    built <- llama_chat_build(model, messages, tools = tools,
                              tool_choice = "auto")
    expect_type(built$prompt, "character")
    expect_gt(nchar(built$prompt), 0L)
    # a tool-enabled build should expose grammar / lazy-trigger metadata.
    expect_true("grammar" %in% names(built))
})
