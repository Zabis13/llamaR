# OpenAI-compatible HTTP server on top of drogonR + llamaR streaming.
# See llama_serve_openai() below.

`%||%` <- function(a, b) if (is.null(a)) b else a

# Build a non-stream chat.completion JSON body (matches llama-server's
# to_json_oaicompat_chat shape closely enough for OpenAI clients).
.openai_completion <- function(id, model_id, text, created,
                               prompt_tokens, completion_tokens, finish_reason) {
    list(
        id      = id,
        object  = "chat.completion",
        created = created,
        model   = model_id,
        choices = list(list(
            index         = 0L,
            message       = list(role = "assistant", content = text),
            finish_reason = finish_reason
        )),
        usage = list(
            prompt_tokens     = prompt_tokens,
            completion_tokens = completion_tokens,
            total_tokens      = prompt_tokens + completion_tokens
        )
    )
}

# One streaming chunk (object = chat.completion.chunk). `delta` is a named
# list; a finishing chunk passes delta = list() and a finish_reason. A
# non-finishing chunk leaves finish_reason = NA, which serialises to JSON
# null (the OpenAI shape); NULL would be dropped by jsonlite.
.openai_chunk <- function(id, model_id, created, delta, finish_reason = NA) {
    list(
        id      = id,
        object  = "chat.completion.chunk",
        created = created,
        model   = model_id,
        choices = list(list(
            index         = 0L,
            delta         = delta,
            finish_reason = finish_reason
        ))
    )
}

#' Serve an OpenAI-compatible HTTP API for a local model
#'
#' Loads a GGUF model once and exposes it over an OpenAI-compatible HTTP
#' API so any OpenAI client (OpenCode, ellmer, the `openai` Python SDK, …)
#' can talk to it. Implements `GET /v1/models` and
#' `POST /v1/chat/completions` (both blocking and `stream = true`). The
#' HTTP/SSE layer is provided by \pkg{drogonR}; generation runs through
#' llamaR's streaming API (\code{\link{llama_gen_begin}} /
#' \code{\link{llama_gen_next}} / \code{\link{llama_gen_end}}).
#'
#' The server is single-sequence: requests are handled one at a time on the
#' main R thread (each streamed token is one event-loop pump). This is meant
#' for a single local user/agent, not concurrent load.
#'
#' \code{drogonR} is an optional dependency (\code{Suggests}); install it
#' with \code{install.packages("drogonR")} (or from its repository) before
#' calling this function.
#'
#' @param model_path Path to a GGUF model file.
#' @param port Port to listen on. Default \code{11434} (the Ollama port, so
#'   clients pointed at a local Ollama work unchanged).
#' @param n_ctx Context size for the loaded model.
#' @param n_gpu_layers Layers to offload to GPU (\code{-1} = all).
#' @param model_id Identifier reported in \code{/v1/models} and echoed in
#'   responses. Defaults to the model file's base name.
#' @param host Address to bind. Default \code{"127.0.0.1"} (local only).
#' @param template Chat template string, or \code{NULL} to use the model's
#'   built-in template.
#' @param max_tokens Default \code{max_new_tokens} when a request omits it.
#' @param ... Reserved for future options.
#'
#' @return Invisibly \code{NULL}. Blocks serving until \code{drogonR::dr_stop()}
#'   is called (typically from another process or an interrupt).
#' @seealso [llama_gen_begin], [llama_generate]
#' @export
#' @examples
#' \dontrun{
#' llama_serve_openai("model.gguf", port = 11434L)
#' # In another shell, point any OpenAI client at
#' #   http://127.0.0.1:11434/v1
#' # e.g. GET /v1/models and POST /v1/chat/completions
#' }
llama_serve_openai <- function(model_path, port = 11434L,
                               n_ctx = 4096L, n_gpu_layers = -1L,
                               model_id = NULL, host = "127.0.0.1",
                               template = NULL, max_tokens = 512L, ...) {
    if (!requireNamespace("drogonR", quietly = TRUE)) {
        stop("llama_serve_openai() requires the 'drogonR' package: ",
             "install.packages('drogonR')", call. = FALSE)
    }
    if (!file.exists(model_path)) {
        stop("model file not found: ", model_path, call. = FALSE)
    }
    if (is.null(model_id)) {
        model_id <- tools::file_path_sans_ext(basename(model_path))
    }

    # --- load model once; keep model/ctx alive for the server's lifetime ---
    model <- llama_load_model(model_path, n_gpu_layers = as.integer(n_gpu_layers))
    ctx   <- llama_new_context(model, n_ctx = as.integer(n_ctx))
    ctx_size <- llama_n_ctx(ctx)
    chat_template <- template %||% tryCatch(llama_chat_template(model),
                                            error = function(e) NULL)

    # Turn an OpenAI `messages` array into a prompt via the chat template.
    # jsonlite gives messages as a data.frame (simplifyVector); normalise to
    # the list(list(role=, content=)) shape llama_chat_apply_template wants.
    build_prompt <- function(messages) {
        if (is.data.frame(messages)) {
            messages <- lapply(seq_len(nrow(messages)), function(i) {
                list(role = as.character(messages$role[i]),
                     content = as.character(messages$content[i]))
            })
        }
        llama_chat_apply_template(messages, template = chat_template,
                                  add_generation_prompt = TRUE)
    }

    # Extract sampling params from a parsed request body, applying defaults.
    gen_args <- function(body) {
        list(
            max_new_tokens = as.integer(body$max_tokens %||% max_tokens),
            temp           = as.double(body$temperature %||% 0.8),
            top_p          = as.double(body$top_p %||% 0.9),
            seed           = as.integer(body$seed %||% 42L)
        )
    }

    new_id <- function() paste0("chatcmpl-", paste0(
        sample(c(0:9, letters), 24, replace = TRUE), collapse = ""))

    app <- drogonR::dr_app()

    # --- GET /v1/models ---
    app <- drogonR::dr_get(app, "/v1/models", function(req) {
        drogonR::dr_json(list(
            object = "list",
            data = list(list(
                id       = model_id,
                object   = "model",
                created  = as.integer(Sys.time()),
                owned_by = "llamaR"
            ))
        ))
    })

    # --- POST /v1/chat/completions ---
    app <- drogonR::dr_post(app, "/v1/chat/completions", function(req) {
        body <- drogonR::dr_body(req, as = "json")
        if (is.null(body$messages)) {
            return(drogonR::dr_json(
                list(error = list(message = "missing 'messages'", type = "invalid_request_error")),
                status = 400L))
        }
        prompt  <- build_prompt(body$messages)
        args    <- gen_args(body)
        created <- as.integer(Sys.time())
        id      <- new_id()
        # parse_special = TRUE to match the generation path's tokenization
        # (role markers are control tokens), so this count matches reality.
        prompt_tokens <- length(llama_tokenize(ctx, prompt, parse_special = TRUE))
        stream  <- isTRUE(body$stream)

        # Reject prompts that don't leave room to generate, instead of
        # silently returning empty content: once the prompt fills the KV
        # cache the very first sample is EOG. Mirror llama-server's 400.
        if (prompt_tokens + args$max_new_tokens > ctx_size) {
            return(drogonR::dr_json(
                list(error = list(
                    message = sprintf(paste0(
                        "prompt is too long: %d tokens + %d requested ",
                        "exceed the context window of %d. Increase n_ctx ",
                        "when starting the server or shorten the input."),
                        prompt_tokens, args$max_new_tokens, ctx_size),
                    type = "invalid_request_error",
                    code = "context_length_exceeded")),
                status = 400L))
        }

        if (!stream) {
            # blocking: drain the whole generation into one string
            st <- do.call(llama_gen_begin, c(list(ctx, prompt), args))
            chunks <- character(0)
            n_completion <- 0L
            repeat {
                chunk <- llama_gen_next(st)
                if (is.null(chunk)) break
                chunks <- c(chunks, chunk)
                n_completion <- n_completion + 1L
            }
            chunks <- c(chunks, llama_gen_end(st))
            text <- paste0(chunks, collapse = "")
            finish <- if (n_completion >= args$max_new_tokens) "length" else "stop"
            return(drogonR::dr_json(.openai_completion(
                id, model_id, text, created, prompt_tokens, n_completion, finish)))
        }

        # streaming: one SSE event per generated token
        st <- do.call(llama_gen_begin, c(list(ctx, prompt), args))
        state <- new.env(parent = emptyenv())
        state$role_sent <- FALSE
        state$flushed   <- FALSE   # gen_end remainder emitted
        state$n         <- 0L
        state$finishing <- FALSE

        generator <- function(s, cancelled) {
            if (cancelled) {
                llama_gen_end(st)
                return(list(data = "", state = s, done = TRUE))
            }
            # first event carries the assistant role with empty content
            if (!state$role_sent) {
                state$role_sent <- TRUE
                obj <- .openai_chunk(id, model_id, created, list(role = "assistant"))
                return(list(data = as.character(jsonlite::toJSON(obj, auto_unbox = TRUE)),
                            state = s, done = FALSE))
            }
            if (!state$finishing) {
                chunk <- llama_gen_next(st)
                if (!is.null(chunk)) {
                    state$n <- state$n + 1L
                    obj <- .openai_chunk(id, model_id, created, list(content = chunk))
                    return(list(data = as.character(jsonlite::toJSON(obj, auto_unbox = TRUE)),
                                state = s, done = FALSE))
                }
                # generation ended: flush any buffered tail, then emit finish chunk
                tail <- llama_gen_end(st)
                state$finishing <- TRUE
                if (nzchar(tail)) {
                    obj <- .openai_chunk(id, model_id, created, list(content = tail))
                    return(list(data = as.character(jsonlite::toJSON(obj, auto_unbox = TRUE)),
                                state = s, done = FALSE))
                }
            }
            if (!state$flushed) {
                state$flushed <- TRUE
                finish <- if (state$n >= args$max_new_tokens) "length" else "stop"
                obj <- .openai_chunk(id, model_id, created,
                                     structure(list(), names = character()),
                                     finish_reason = finish)
                return(list(data = as.character(jsonlite::toJSON(obj, auto_unbox = TRUE)),
                            state = s, done = FALSE))
            }
            # final SSE frame: the OpenAI sentinel, then close
            list(data = "[DONE]", state = s, done = TRUE)
        }

        drogonR::dr_stream_sse(generator, state = state)
    })

    message(sprintf("llamaR OpenAI server: http://%s:%d  (model '%s')",
                    host, as.integer(port), model_id))
    drogonR::dr_serve(app, port = as.integer(port))

    # dr_serve() returns immediately; Drogon runs in background C++ threads
    # and dispatches to R handlers via the later event loop. Block here,
    # pumping the loop, so the server stays up until interrupted (then stop
    # cleanly — otherwise Drogon's destructor aborts at process exit).
    on.exit(drogonR::dr_stop(), add = TRUE)
    tryCatch(
        while (drogonR::dr_running()) {
            later::run_now(timeoutSecs = 1, all = TRUE)
        },
        interrupt = function(e) invisible(NULL)
    )
    invisible(NULL)
}
