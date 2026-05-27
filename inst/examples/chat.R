#!/usr/bin/env Rscript
#
# Chat with a local GGUF model through an ellmer::Chat object.
#
# Two ways to use this file:
#
#   1. Spawn a server for the model and chat with it (the background
#      server is stopped automatically on exit):
#
#        Rscript inst/examples/chat.R /path/to/model.gguf
#        Rscript inst/examples/chat.R /path/to/model.gguf 11434
#        Rscript inst/examples/chat.R /path/to/model.gguf 11434 4096
#
#      Positional args after the model path are [port] [n_ctx]
#      (defaults: port 11434, n_ctx 4096). Add "--timeout <seconds>" to wait
#      longer for a big model to load (default 180s), and "--system <prompt>"
#      to set a system prompt (works in both modes).
#
#   2. Connect to a server you already started with llama_serve_openai():
#
#        Rscript inst/examples/chat.R --url http://127.0.0.1:11434/v1
#
# With no extra argument the script starts an interactive prompt (type a
# message, Enter to send, blank line or Ctrl-D to quit). Pass a single
# message as the last argument to send just that and print the reply:
#
#        Rscript inst/examples/chat.R /path/to/model.gguf "Why is the sky blue?"
#
# Needs the optional `ellmer` package (and `callr` when spawning a server).

library(llamaR)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) {
    stop("usage: chat.R <model.gguf> [port] [n_ctx] [message]\n",
         "   or: chat.R --url <base_url> [message]")
}

# Pull optional flag-options out of the args before positional parsing.
timeout <- 180L
ti <- match("--timeout", args)
if (!is.na(ti)) {
    timeout <- suppressWarnings(as.integer(args[ti + 1L]))
    if (is.na(timeout)) stop("--timeout needs a number of seconds")
    args <- args[-c(ti, ti + 1L)]
}

system_prompt <- NULL
si <- match("--system", args)
if (!is.na(si)) {
    system_prompt <- args[si + 1L]
    if (is.na(system_prompt)) stop("--system needs a prompt string")
    args <- args[-c(si, si + 1L)]
}

# ---------------------------------------------------------------------------
# Build the chat: either connect to a URL (--url) or spawn for a model file.
# ---------------------------------------------------------------------------
if (identical(args[[1]], "--url")) {
    base_url <- args[[2]]
    if (is.na(base_url) || !nzchar(base_url)) {
        stop("--url needs a base URL, e.g. --url http://127.0.0.1:11434/v1")
    }
    message("Connecting to ", base_url, " ...")
    chat <- chat_llamar(base_url = base_url, system_prompt = system_prompt)
    rest <- args[-(1:2)]
    on_exit_stop <- NULL                       # we don't own this server
} else {
    model_path <- args[[1]]
    nums <- suppressWarnings(as.integer(args[-1]))
    port  <- if (length(nums) >= 1 && !is.na(nums[1])) nums[1] else 11434L
    n_ctx <- if (length(nums) >= 2 && !is.na(nums[2])) nums[2] else 4096L
    message("Loading ", model_path, " and starting a server on port ", port,
            " (timeout ", timeout, "s) ...")
    chat <- chat_llamar(model_path = model_path, port = port, n_ctx = n_ctx,
                        timeout = timeout, system_prompt = system_prompt)
    on_exit_stop <- chat                       # stop our spawned server on exit
    # remaining args are the positional numbers plus an optional message
    rest <- args[-1][is.na(nums)]
}
if (!is.null(on_exit_stop)) {
    on.exit(chat_llamar_stop(on_exit_stop), add = TRUE)
}

# A trailing non-numeric argument is treated as a one-shot message.
one_shot <- if (length(rest) >= 1) rest[[length(rest)]] else NULL

# ---------------------------------------------------------------------------
# One-shot mode: send a single message, print the reply, done.
# ---------------------------------------------------------------------------
if (!is.null(one_shot)) {
    cat(chat$chat(one_shot, echo = "none"), "\n", sep = "")
    quit(save = "no")
}

# ---------------------------------------------------------------------------
# Interactive REPL: read a line, send it, echo the streamed reply.
# ---------------------------------------------------------------------------
message("Ready. Type a message and press Enter (blank line or Ctrl-D quits).")
con <- file("stdin")
open(con)
repeat {
    cat("\n> ")
    line <- readLines(con, n = 1L)
    if (length(line) == 0L || !nzchar(trimws(line))) break  # EOF or blank
    chat$chat(line, echo = "output")
    cat("\n")
}
close(con)
