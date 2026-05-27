#!/usr/bin/env Rscript
#
# Serve a local GGUF model over an OpenAI-compatible HTTP API, and
# (optionally) smoke-test both endpoints end-to-end.
#
# Two ways to use this file:
#
#   1. Just start the server (blocks until Ctrl-C):
#
#        Rscript inst/examples/serve_openai.R /path/to/model.gguf
#        Rscript inst/examples/serve_openai.R /path/to/model.gguf 11434
#        Rscript inst/examples/serve_openai.R /path/to/model.gguf 11434 16384
#
#      Positional args after the model path are [port] [n_ctx]
#      (defaults: port 11434, n_ctx 16384).
#
#   2. Self-test: start the server in a background process, hit
#      GET /v1/models and POST /v1/chat/completions (stream and
#      non-stream), print the results, then shut it down. Needs the
#      `callr` package and the `curl` command-line tool:
#
#        Rscript inst/examples/serve_openai.R /path/to/model.gguf --selftest
#
# Point any OpenAI client at http://127.0.0.1:<port>/v1 — e.g. OpenCode,
# ellmer's chat_openai(base_url = ...), or the openai Python SDK.

library(llamaR)

args       <- commandArgs(trailingOnly = TRUE)
model_path <- if (length(args) >= 1) args[[1]] else
    stop("usage: serve_openai.R <model.gguf> [port] [--selftest]")
selftest   <- "--selftest" %in% args
# positional numeric args after the model path: [port] [n_ctx]
nums       <- suppressWarnings(as.integer(args[!grepl("^--", args)][-1]))
port       <- if (length(nums) >= 1 && !is.na(nums[1])) nums[1] else 11434L
n_ctx      <- if (length(nums) >= 2 && !is.na(nums[2])) nums[2] else 16384L

# ---------------------------------------------------------------------------
# Mode 1: just serve (blocks)
# ---------------------------------------------------------------------------
if (!selftest) {
    llama_serve_openai(model_path, port = port, n_ctx = n_ctx)
    quit(save = "no")
}

# ---------------------------------------------------------------------------
# Mode 2: self-test against a background server
# ---------------------------------------------------------------------------
stopifnot(requireNamespace("callr", quietly = TRUE),
          requireNamespace("jsonlite", quietly = TRUE),
          nzchar(Sys.which("curl")))

base_url <- sprintf("http://127.0.0.1:%d", port)

server <- callr::r_bg(
    function(model_path, port, n_ctx) {
        library(llamaR)
        llama_serve_openai(model_path, port = port, n_ctx = n_ctx,
                           model_id = "demo-model")
    },
    args = list(model_path = model_path, port = port, n_ctx = n_ctx))

on.exit(server$kill(), add = TRUE)

# wait until GET /v1/models answers 200
message("waiting for server on ", base_url, " ...")
ok <- FALSE
deadline <- Sys.time() + 60
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
    cat(paste(server$read_error_lines(), collapse = "\n"), "\n")
    stop("server did not come up")
}

post <- function(body, stream = FALSE) {
    bf <- tempfile(fileext = ".json"); on.exit(unlink(bf))
    writeLines(body, bf)
    system2("curl", c(if (stream) "-N", "-s", "--max-time", "30",
                      "-H", "Content-Type: application/json",
                      "--data", paste0("@", bf),
                      paste0(base_url, "/v1/chat/completions")),
            stdout = TRUE, stderr = NULL)
}

cat("\n== GET /v1/models ==\n")
models <- system2("curl", c("-s", paste0(base_url, "/v1/models")),
                  stdout = TRUE)
cat(models, "\n")

cat("\n== POST /v1/chat/completions (non-stream) ==\n")
body <- '{"model":"demo-model","messages":[{"role":"user","content":"Say hello."}],"max_tokens":32,"temperature":0}'
res  <- jsonlite::fromJSON(paste(post(body), collapse = ""))
cat("content      :", res$choices$message$content, "\n")
cat("finish_reason:", res$choices$finish_reason, "\n")
cat("usage        :", res$usage$prompt_tokens, "+",
    res$usage$completion_tokens, "=", res$usage$total_tokens, "tokens\n")

cat("\n== POST /v1/chat/completions (stream) ==\n")
body_s <- '{"model":"demo-model","messages":[{"role":"user","content":"Say hello."}],"max_tokens":32,"temperature":0,"stream":true}'
for (line in post(body_s, stream = TRUE)) if (nzchar(line)) cat(line, "\n")

cat("\nself-test complete; shutting down server.\n")
