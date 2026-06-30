# ============================================================
# Anthropic server — end-to-end (HEAVY).
# Spawns llama_serve_anthropic() in a background process and
# drives it over HTTP (GET /v1/models, POST /v1/messages,
# blocking and streaming). Mirrors inst/examples/serve_openai.R's
# self-test. Needs a model + drogonR + callr + curl; listed in
# tests/testthat.R `heavy` so it is skipped on CRAN.
# ============================================================

MODEL_PATH <- "/mnt/Data2/DS_projects/llm_models/tiny-mistral-test-Q2_K.gguf"

skip_if_no_model <- function() {
    if (!file.exists(MODEL_PATH)) skip("test model not available")
}

# Spawn a server, wait for /v1/models to answer 200, return base_url + handle.
.spawn_anthropic <- function(port, n_ctx = 1024L) {
    server <- callr::r_bg(
        function(model_path, port, n_ctx) {
            library(llamaR)
            llama_serve_anthropic(model_path, port = port, n_ctx = n_ctx,
                                  model_id = "test-model")
        },
        args = list(MODEL_PATH, port, n_ctx))

    base_url <- sprintf("http://127.0.0.1:%d", port)
    deadline <- Sys.time() + 60
    ok <- FALSE
    repeat {
        code <- suppressWarnings(system2(
            "curl", c("-s", "-o", "/dev/null", "-w", "%{http_code}",
                      "--max-time", "2", paste0(base_url, "/v1/models")),
            stdout = TRUE, stderr = NULL))
        if (length(code) && identical(code, "200")) { ok <- TRUE; break }
        if (!server$is_alive() || Sys.time() > deadline) break
        Sys.sleep(0.5)
    }
    if (!ok) {
        err <- tryCatch(server$read_all_error(), error = function(e) "")
        server$kill()
        stop("anthropic server did not come up:\n", err)
    }
    list(server = server, base_url = base_url)
}

.post_messages <- function(base_url, body, stream = FALSE) {
    bf <- tempfile(fileext = ".json"); on.exit(unlink(bf))
    writeLines(body, bf)
    system2("curl", c(if (stream) "-N", "-s", "--max-time", "30",
                      "-H", "Content-Type: application/json",
                      "--data", paste0("@", bf),
                      paste0(base_url, "/v1/messages")),
            stdout = TRUE, stderr = NULL)
}

test_that("llama_serve_anthropic serves /v1/models and /v1/messages", {
    skip_if_no_model()
    skip_if_not_installed("drogonR")
    skip_if_not_installed("callr")
    skip_if_not_installed("jsonlite")
    skip_if_not(nzchar(Sys.which("curl")), "curl not available")

    srv <- .spawn_anthropic(port = 18435L)
    on.exit(srv$server$kill(), add = TRUE)

    # --- GET /v1/models ---
    models_raw <- system2("curl", c("-s", paste0(srv$base_url, "/v1/models")),
                          stdout = TRUE)
    models <- jsonlite::fromJSON(paste(models_raw, collapse = ""),
                                 simplifyVector = FALSE)
    # Anthropic /v1/models shape: a `data` array of model objects + has_more.
    expect_gte(length(models$data), 1L)
    expect_identical(models$data[[1]]$id, "test-model")
    expect_identical(models$data[[1]]$type, "model")
    expect_false(models$has_more)

    # --- POST /v1/messages (blocking) ---
    body <- paste0('{"model":"test-model","max_tokens":16,',
                   '"messages":[{"role":"user","content":"Say hi."}]}')
    res <- jsonlite::fromJSON(paste(.post_messages(srv$base_url, body),
                                    collapse = ""), simplifyVector = FALSE)
    expect_identical(res$type, "message")
    expect_identical(res$role, "assistant")
    expect_true(length(res$content) >= 1L)
    # at least one text block with non-empty text (no "empty/malformed response").
    texts <- Filter(function(b) identical(b$type, "text"), res$content)
    expect_gte(length(texts), 1L)
    expect_gt(nchar(texts[[1]]$text), 0L)
    expect_true(res$stop_reason %in% c("end_turn", "max_tokens"))
})

test_that("llama_serve_anthropic streams the Anthropic SSE event sequence", {
    skip_if_no_model()
    skip_if_not_installed("drogonR")
    skip_if_not_installed("callr")
    skip_if_not(nzchar(Sys.which("curl")), "curl not available")

    srv <- .spawn_anthropic(port = 18436L)
    on.exit(srv$server$kill(), add = TRUE)

    body <- paste0('{"model":"test-model","max_tokens":16,"stream":true,',
                   '"messages":[{"role":"user","content":"Say hi."}]}')
    lines <- .post_messages(srv$base_url, body, stream = TRUE)
    blob  <- paste(lines, collapse = "\n")

    # The Anthropic stream must emit the framed event sequence.
    expect_true(grepl("event: message_start",       blob, fixed = TRUE))
    expect_true(grepl("event: content_block_delta", blob, fixed = TRUE))
    expect_true(grepl("event: message_stop",        blob, fixed = TRUE))
})
