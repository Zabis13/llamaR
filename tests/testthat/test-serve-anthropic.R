# ============================================================
# Anthropic server: request translation and response shaping.
# Fast unit tests only — no model load, no network. End-to-end
# serving is exercised by inst/examples/claude_code_launcher.sh.
# Covers the bugs fixed while stabilising Claude Code:
#   - merging multiple system blocks into one (avoid silent loss)
#   - stripping <think> reasoning (full / dangling / unclosed)
#   - content-block / tool-call translation both directions.
# ============================================================

# --- .strip_thinking ----------------------------------------------------------

test_that(".strip_thinking removes a complete <think>...</think> block", {
    expect_identical(
        llamaR:::.strip_thinking("<think>\nreasoning\n</think>\nHello!"),
        "Hello!")
})

test_that(".strip_thinking removes a dangling closing tag (opener in template)", {
    expect_identical(
        llamaR:::.strip_thinking("reasoning with no opener\n</think>\nAnswer"),
        "Answer")
})

test_that(".strip_thinking holds back an unclosed <think> (still streaming)", {
    # While reasoning is open we must emit nothing — the closing tag arrives later.
    expect_identical(llamaR:::.strip_thinking("<think>\nstill thinking"), "")
})

test_that(".strip_thinking is a no-op on plain text", {
    expect_identical(llamaR:::.strip_thinking("Just an answer"), "Just an answer")
})

test_that(".strip_thinking removes multiple think blocks", {
    expect_identical(
        llamaR:::.strip_thinking("<think>a</think>text<think>b</think>more"),
        "textmore")
})

test_that(".strip_thinking leaves an empty string empty", {
    expect_identical(llamaR:::.strip_thinking(""), "")
})

# --- .anthropic_to_messages: system merging (the silent-loss bug) -------------

test_that(".anthropic_to_messages merges top-level and in-messages system", {
    body <- list(
        system = list(list(type = "text", text = "TOP_A"),
                      list(type = "text", text = "TOP_B")),
        messages = list(
            list(role = "user",   content = "hello USR"),
            list(role = "system", content = "INNER_C")))
    msgs <- llamaR:::.anthropic_to_messages(body)

    sys <- Filter(function(m) identical(m$role, "system"), msgs)
    expect_length(sys, 1L)                       # exactly one system message
    all_sys <- sys[[1]]$content
    expect_true(grepl("TOP_A",   all_sys, fixed = TRUE))
    expect_true(grepl("TOP_B",   all_sys, fixed = TRUE))
    expect_true(grepl("INNER_C", all_sys, fixed = TRUE))
    expect_identical(msgs[[1]]$role, "system")   # system leads
})

test_that(".anthropic_to_messages handles a plain string system field", {
    body <- list(system = "ONLY_SYS",
                 messages = list(list(role = "user", content = "hi")))
    msgs <- llamaR:::.anthropic_to_messages(body)
    expect_identical(msgs[[1]]$role, "system")
    expect_true(grepl("ONLY_SYS", msgs[[1]]$content, fixed = TRUE))
})

test_that(".anthropic_to_messages keeps user/assistant turns in order", {
    body <- list(messages = list(
        list(role = "user",      content = "Q1"),
        list(role = "assistant", content = "A1"),
        list(role = "user",      content = "Q2")))
    msgs <- llamaR:::.anthropic_to_messages(body)
    roles <- vapply(msgs, function(m) m$role, character(1))
    expect_identical(roles, c("user", "assistant", "user"))
})

# --- .anthropic_content_to_openai: typed content blocks -----------------------

test_that(".anthropic_content_to_openai passes plain string content through", {
    out <- llamaR:::.anthropic_content_to_openai("user", "hello")
    expect_length(out, 1L)
    expect_identical(out[[1]]$role, "user")
    expect_identical(out[[1]]$content, "hello")
})

test_that(".anthropic_content_to_openai joins multiple text blocks", {
    blocks <- list(list(type = "text", text = "foo"),
                   list(type = "text", text = "bar"))
    out <- llamaR:::.anthropic_content_to_openai("user", blocks)
    expect_identical(out[[1]]$content, "foobar")
})

test_that(".anthropic_content_to_openai maps tool_use to an OpenAI tool_call", {
    blocks <- list(list(type = "tool_use", id = "tu_1", name = "get_weather",
                        input = list(city = "Paris")))
    out <- llamaR:::.anthropic_content_to_openai("assistant", blocks)
    msg <- out[[1]]
    expect_identical(msg$role, "assistant")
    expect_length(msg$tool_calls, 1L)
    expect_identical(msg$tool_calls[[1]]$`function`$name, "get_weather")
})

test_that(".anthropic_content_to_openai maps tool_result to a tool message", {
    blocks <- list(list(type = "tool_result", tool_use_id = "tu_1",
                        content = "sunny"))
    out <- llamaR:::.anthropic_content_to_openai("user", blocks)
    tm <- out[[1]]
    expect_identical(tm$role, "tool")
    expect_identical(tm$tool_call_id, "tu_1")
    expect_true(grepl("sunny", tm$content, fixed = TRUE))
})

test_that(".anthropic_content_to_openai silently drops thinking blocks", {
    blocks <- list(list(type = "thinking", thinking = "private", signature = "sig"),
                   list(type = "text", text = "visible"))
    out <- llamaR:::.anthropic_content_to_openai("assistant", blocks)
    expect_length(out, 1L)
    expect_identical(out[[1]]$content, "visible")
})

# --- .anthropic_content_blocks: response text + tool_use ----------------------

test_that(".anthropic_content_blocks strips thinking from the text block", {
    parsed <- list(content = "<think>hmm</think>\nFinal answer", tool_calls = NULL)
    blocks <- llamaR:::.anthropic_content_blocks(parsed, strip_thinking = TRUE)
    expect_identical(blocks[[1]]$type, "text")
    expect_identical(blocks[[1]]$text, "Final answer")
})

test_that(".anthropic_content_blocks keeps thinking when strip_thinking = FALSE", {
    parsed <- list(content = "<think>hmm</think>\nAns", tool_calls = NULL)
    blocks <- llamaR:::.anthropic_content_blocks(parsed, strip_thinking = FALSE)
    expect_true(grepl("think", blocks[[1]]$text, fixed = TRUE))
})

test_that(".anthropic_content_blocks omits an empty text block", {
    # whole output was reasoning -> stripped to nothing -> no text block
    parsed <- list(content = "<think>only reasoning</think>", tool_calls = NULL)
    blocks <- llamaR:::.anthropic_content_blocks(parsed, strip_thinking = TRUE)
    expect_length(blocks, 0L)
})

test_that(".anthropic_content_blocks emits a tool_use block", {
    parsed <- list(content = "",
                   tool_calls = data.frame(
                       id = "tu_9", name = "search",
                       arguments = '{"query":"x"}', stringsAsFactors = FALSE))
    blocks <- llamaR:::.anthropic_content_blocks(parsed, strip_thinking = TRUE)
    expect_identical(blocks[[1]]$type, "tool_use")
    expect_identical(blocks[[1]]$name, "search")
    expect_identical(blocks[[1]]$id, "tu_9")
})

# --- .anthropic_stop_reason ---------------------------------------------------

test_that(".anthropic_stop_reason reports tool_use when tool calls exist", {
    parsed <- list(tool_calls = data.frame(id = "t", name = "n",
                                           arguments = "{}", stringsAsFactors = FALSE))
    expect_identical(llamaR:::.anthropic_stop_reason(parsed, FALSE), "tool_use")
})

test_that(".anthropic_stop_reason reports max_tokens on a length cap", {
    parsed <- list(tool_calls = NULL)
    expect_identical(llamaR:::.anthropic_stop_reason(parsed, TRUE), "max_tokens")
})

test_that(".anthropic_stop_reason reports end_turn otherwise", {
    parsed <- list(tool_calls = NULL)
    expect_identical(llamaR:::.anthropic_stop_reason(parsed, FALSE), "end_turn")
})

# --- .anthropic_tools_to_openai -----------------------------------------------

test_that(".anthropic_tools_to_openai converts Anthropic tools to OpenAI shape", {
    tools <- list(list(name = "get_weather", description = "weather",
                       input_schema = list(type = "object")))
    out <- llamaR:::.anthropic_tools_to_openai(tools)
    expect_length(out, 1L)
    expect_identical(out[[1]]$type, "function")
    expect_identical(out[[1]]$`function`$name, "get_weather")
    expect_identical(out[[1]]$`function`$parameters$type, "object")
})

test_that(".anthropic_tools_to_openai returns NULL for no tools", {
    expect_null(llamaR:::.anthropic_tools_to_openai(NULL))
    expect_null(llamaR:::.anthropic_tools_to_openai(list()))
})

# --- vision: image-block parsing (no model needed) ----------------------------

# A tiny but valid 1x1 PNG, base64-encoded. Lets us exercise the base64 decode +
# temp-file write path without a graphics device. (8-byte sig + IHDR + IDAT + IEND.)
.tiny_png_b64 <- paste0(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk",
    "+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==")

test_that(".anthropic_image_to_file decodes an inline base64 image to a file", {
    b <- list(type = "image",
              source = list(type = "base64", media_type = "image/png",
                            data = .tiny_png_b64))
    p <- llamaR:::.anthropic_image_to_file(b)
    on.exit(unlink(p), add = TRUE)
    expect_true(file.exists(p))
    expect_gt(file.info(p)$size, 0)
    expect_match(p, "\\.png$")
    # round-trips: decoded bytes == original payload
    expect_identical(readBin(p, "raw", file.info(p)$size),
                     jsonlite::base64_dec(.tiny_png_b64))
})

test_that(".anthropic_image_to_file returns NULL for non-base64 / empty sources", {
    expect_null(llamaR:::.anthropic_image_to_file(
        list(type = "image", source = list(type = "url", url = "http://x/y.png"))))
    expect_null(llamaR:::.anthropic_image_to_file(
        list(type = "image", source = list(type = "base64", data = ""))))
    expect_null(llamaR:::.anthropic_image_to_file(list(type = "image")))
})

test_that(".anthropic_to_messages splices the marker and collects paths (vision on)", {
    body <- list(messages = list(list(role = "user", content = list(
        list(type = "text", text = "What is this? "),
        list(type = "image", source = list(type = "base64",
             media_type = "image/png", data = .tiny_png_b64))))))
    msgs <- llamaR:::.anthropic_to_messages(body, marker = "<__media__>")
    paths <- attr(msgs, "image_paths")
    on.exit(unlink(paths), add = TRUE)
    # marker spliced into the user text where the image was
    expect_true(grepl("<__media__>", msgs[[length(msgs)]]$content, fixed = TRUE))
    expect_length(paths, 1L)
    expect_true(file.exists(paths))
})

test_that(".anthropic_to_messages drops image blocks when vision is off (no marker)", {
    body <- list(messages = list(list(role = "user", content = list(
        list(type = "text", text = "What is this? "),
        list(type = "image", source = list(type = "base64",
             media_type = "image/png", data = .tiny_png_b64))))))
    msgs <- llamaR:::.anthropic_to_messages(body)   # marker = NULL -> text-only
    expect_false(grepl("__media__", msgs[[length(msgs)]]$content))
    expect_null(attr(msgs, "image_paths"))
})
