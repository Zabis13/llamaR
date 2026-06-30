#!/usr/bin/env Rscript
#
# Serve TWO local GGUF models behind one Anthropic-compatible HTTP API:
# a text model for chat/tools and a vision model (+mmproj) for images. The
# server routes each /v1/messages request to the right model — a request that
# carries an image goes to the vision model, everything else to the text model.
#
# This is the "without Claude Code" path: you talk to the server directly with
# curl (or any Anthropic client), no `claude` CLI involved.
#
# Two ways to use this file:
#
#   1. Just start the server (blocks until Ctrl-C):
#
#        Rscript inst/examples/serve_anthropic_vision.R \
#          /models/Qwen3.5-9B-UD-Q6_K_XL.gguf \
#          /models/Qwen2-VL-2B-Instruct-Q8_0.gguf \
#          /models/mmproj-Qwen2-VL-2B-Instruct-Q8_0.gguf
#
#      Positional args: <text_model> <vision_model> <mmproj> [port] [n_ctx]
#      (defaults: port 11435, n_ctx 32768; vision context is fixed small).
#
#   2. Self-test: start the server in the background, draw a red circle, send it
#      as a base64 Anthropic image block to /v1/messages, print the reply, then
#      shut down. Needs the `callr` package and the `curl` CLI:
#
#        Rscript inst/examples/serve_anthropic_vision.R \
#          <text> <vision> <mmproj> --selftest
#
# --- The raw request the server expects (this is what Claude Code sends too) ---
#
# A vision request is a normal Anthropic /v1/messages body whose user content is
# an array of blocks, with an image block carrying inline base64:
#
#   {
#     "model": "qwen2-vl",
#     "max_tokens": 128,
#     "messages": [{
#       "role": "user",
#       "content": [
#         {"type": "text", "text": "What shape and color is this?"},
#         {"type": "image", "source": {
#            "type": "base64", "media_type": "image/png", "data": "<BASE64>"}}
#       ]
#     }]
#   }
#
# Equivalent curl (BASE=http://127.0.0.1:11435), with the image in a file:
#
#   B64=$(base64 -w0 circle.png)
#   curl -s "$BASE/v1/messages" -H 'content-type: application/json' -d "{
#     \"max_tokens\": 128,
#     \"messages\": [{\"role\":\"user\",\"content\":[
#       {\"type\":\"text\",\"text\":\"What is this?\"},
#       {\"type\":\"image\",\"source\":{\"type\":\"base64\",
#        \"media_type\":\"image/png\",\"data\":\"$B64\"}}]}]}"

library(llamaR)

args <- commandArgs(trailingOnly = TRUE)
selftest <- "--selftest" %in% args
args <- args[args != "--selftest"]

text_model   <- if (length(args) >= 1) args[[1]] else
    "/mnt/Data2/DS_projects/llm_models/Qwen3.5-9B-UD-Q6_K_XL.gguf"
vision_model <- if (length(args) >= 2) args[[2]] else
    "/mnt/Data2/DS_projects/llm_models/Qwen2-VL-2B-Instruct-Q8_0.gguf"
mmproj       <- if (length(args) >= 3) args[[3]] else
    "/mnt/Data2/DS_projects/llm_models/mmproj-Qwen2-VL-2B-Instruct-Q8_0.gguf"
port  <- if (length(args) >= 4) as.integer(args[[4]]) else 11435L
n_ctx <- if (length(args) >= 5) as.integer(args[[5]]) else 32768L

serve <- function() {
    llama_serve_anthropic(
        text_model, port = port, n_ctx = n_ctx,
        vision_model_path = vision_model, mmproj_path = mmproj,
        vision_n_ctx = 8192L)
}

if (!selftest) {
    serve()   # blocks until interrupted
    quit(save = "no")
}

# --- self-test: spawn the server, send an image request, print the reply ------
stopifnot(requireNamespace("callr", quietly = TRUE))
if (Sys.which("curl") == "") stop("self-test needs the 'curl' command-line tool")

# Draw a red circle on white -> PNG -> base64.
img <- tempfile(fileext = ".png")
grDevices::png(img, width = 224, height = 224)
op <- graphics::par(mar = c(0, 0, 0, 0)); graphics::plot.new()
graphics::plot.window(xlim = c(0, 1), ylim = c(0, 1))
graphics::symbols(0.5, 0.5, circles = 0.3, inches = FALSE, add = TRUE,
                  bg = "red", fg = "red")
grDevices::dev.off(); graphics::par(op)
b64 <- jsonlite::base64_enc(readBin(img, "raw", file.info(img)$size))

base <- sprintf("http://127.0.0.1:%d", port)
message(">> starting two-model server in the background...")
srv <- callr::r_bg(function(tm, vm, mm, p, nc) {
    llamaR::llama_serve_anthropic(tm, port = p, n_ctx = nc,
        vision_model_path = vm, mmproj_path = mm, vision_n_ctx = 8192L)
}, args = list(text_model, vision_model, mmproj, port, n_ctx))
on.exit({ srv$kill(); unlink(img) }, add = TRUE)

# wait until the server answers (large models load slowly)
ready <- FALSE
for (i in seq_len(300)) {
    if (!srv$is_alive()) stop("server died before becoming ready:\n",
                              paste(srv$read_all_error_lines(), collapse = "\n"))
    code <- suppressWarnings(system2("curl",
        c("-s", "-o", "/dev/null", "-w", "%{http_code}", "--max-time", "2",
          paste0(base, "/v1/models")), stdout = TRUE, stderr = NULL))
    if (identical(code, "200")) { ready <- TRUE; break }
    Sys.sleep(1)
}
if (!ready) stop("timed out waiting for the server")
message(sprintf("   ready after a moment; sending image request to %s", base))

# Build the Anthropic body with the inline image block and POST it.
body <- jsonlite::toJSON(list(
    max_tokens = 128L,
    messages = list(list(role = "user", content = list(
        list(type = "text", text = "What shape and color is in this image?"),
        list(type = "image", source = list(type = "base64",
             media_type = "image/png", data = b64)))))),
    auto_unbox = TRUE)
bf <- tempfile(); writeLines(body, bf); on.exit(unlink(bf), add = TRUE)

resp <- system2("curl", c("-s", paste0(base, "/v1/messages"),
    "-H", "content-type: application/json", "--data", paste0("@", bf)),
    stdout = TRUE)
cat("\n===== /v1/messages response =====\n")
parsed <- tryCatch(jsonlite::fromJSON(paste(resp, collapse = ""),
                                      simplifyVector = FALSE), error = function(e) NULL)
if (!is.null(parsed$content)) {
    txt <- vapply(parsed$content,
                  function(b) if (is.null(b$text)) "" else b$text, character(1))
    cat(paste(txt, collapse = ""), "\n")
} else {
    cat(paste(resp, collapse = "\n"), "\n")
}
cat("=================================\n")
