# ============================================================
# Tool-aware chat layer: llama_chat_build() / llama_chat_parse().
# Fast guard/shape tests need no model. The round-trip
# (build -> generate -> parse) lives in test-chat-build-roundtrip.R
# (heavy: needs a model, skipped on CRAN via tests/testthat.R).
# ============================================================

test_that("llama_chat_build rejects a non-list messages argument", {
    # stopifnot(is.list(messages)) fires before any .Call, so no model needed.
    expect_error(llama_chat_build(NULL, messages = "not a list"))
})

test_that("llama_chat_parse returns content + a tool_calls data frame shape", {
    # The empty-input / no-tool case must still yield the documented shape:
    # a list with content, reasoning_content and a zero-row tool_calls frame.
    # Uses format 0 (GENERIC-ish) which needs no model handle.
    res <- llama_chat_parse("hello world", format = 0L)
    expect_type(res, "list")
    expect_true(all(c("content", "reasoning_content", "tool_calls") %in% names(res)))
    expect_s3_class(res$tool_calls, "data.frame")
    expect_identical(names(res$tool_calls), c("name", "arguments", "id"))
    expect_identical(nrow(res$tool_calls), 0L)
})

test_that("llama_chat_parse coerces format and is_partial without error", {
    # numeric format and logical is_partial are coerced inside the wrapper.
    res <- llama_chat_parse("plain text", format = 0, is_partial = TRUE)
    expect_type(res$content, "character")
})
