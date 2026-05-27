MODEL_PATH <- "/mnt/Data2/DS_projects/llm_models/tiny-mistral-test-Q2_K.gguf"

# ============================================================
# chat_llamar(): ellmer::Chat constructor.
# Fast guard/shape tests need no model and no network. The
# end-to-end test spawns a real server and is gated on model,
# ellmer, callr and drogonR all being present.
# ============================================================

test_that("chat_llamar errors without ellmer", {
    if (requireNamespace("ellmer", quietly = TRUE)) {
        skip("ellmer is installed; cannot exercise the missing-package guard")
    }
    expect_error(chat_llamar(base_url = "http://127.0.0.1:1/v1"), "ellmer")
})

test_that("chat_llamar requires exactly one of model_path / base_url", {
    skip_if_not_installed("ellmer")
    expect_error(chat_llamar(), "exactly one")
    expect_error(
        chat_llamar(model_path = "m.gguf",
                    base_url = "http://127.0.0.1:1/v1"),
        "exactly one")
})

test_that("chat_llamar(base_url=) returns a Chat without spawning", {
    skip_if_not_installed("ellmer")
    chat <- chat_llamar(base_url = "http://127.0.0.1:11434/v1")
    expect_s3_class(chat, "Chat")
    # mode B owns no process
    expect_false(chat_llamar_stop(chat))
})

test_that("chat_llamar(model_path=) errors on a missing model file", {
    skip_if_not_installed("ellmer")
    skip_if_not_installed("callr")
    expect_error(
        chat_llamar(model_path = "/no/such/model.gguf"),
        "model file not found")
})

test_that("chat_llamar(model_path=) spawns a server and answers", {
    skip_if_not(file.exists(MODEL_PATH), "test model not available")
    skip_if_not_installed("ellmer")
    skip_if_not_installed("callr")
    skip_if_not_installed("drogonR")

    # Use a non-default port to avoid clashing with a real local server.
    # n_ctx must leave room for ellmer's default max_tokens (512) plus the
    # prompt, else the server returns 400 context_length_exceeded.
    chat <- chat_llamar(model_path = MODEL_PATH, port = 18434L, n_ctx = 1024L)
    on.exit(chat_llamar_stop(chat), add = TRUE)

    expect_s3_class(chat, "Chat")
    reply <- chat$chat("Say hi.", echo = "none")
    expect_type(reply, "character")
    expect_gt(nchar(reply), 0L)

    # stopping kills the spawned process; a second stop is a no-op
    expect_true(chat_llamar_stop(chat))
    expect_false(chat_llamar_stop(chat))
})
