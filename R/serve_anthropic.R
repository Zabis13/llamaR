# Anthropic Messages API-compatible HTTP server on top of drogonR + llamaR.
# Lets Claude Code (and any Anthropic SDK client) talk to a local GGUF model:
# point ANTHROPIC_BASE_URL at this server. See llama_serve_anthropic() below.
#
# Layering mirrors R/serve.R (the OpenAI server): drogonR for HTTP/SSE, llamaR's
# streaming generation, and the tool-aware chat layer (llama_chat_build /
# llama_chat_parse) for prompt construction, grammar, and tool-call parsing.

`%||%` <- function(a, b) if (is.null(a)) b else a

# --- request translation: Anthropic -> internal (OpenAI-shape) messages -------

# Decode one Anthropic image block's base64 payload to a temp file and return
# its path, or NULL if the block has no usable inline data. Anthropic image
# blocks look like {type:"image", source:{type:"base64", media_type:"image/png",
# data:"...."}}. We only handle inline base64 (Claude Code sends screenshots
# this way); URL sources are skipped.
.anthropic_image_to_file <- function(b) {
    src <- b$source
    if (is.null(src) || !identical(src$type %||% "", "base64")) return(NULL)
    data <- src$data
    if (is.null(data) || !nzchar(data)) return(NULL)
    mt  <- src$media_type %||% "image/png"
    ext <- switch(mt, "image/png" = ".png", "image/jpeg" = ".jpg",
                  "image/webp" = ".webp", "image/gif" = ".gif", ".png")
    path <- tempfile("llamar_img_", fileext = ext)
    writeBin(jsonlite::base64_dec(data), path)
    path
}

# Anthropic `content` is either a plain string or a list of typed blocks
# (text / tool_use / tool_result / image). Flatten one message's content into
# the OpenAI shape llama_chat_build expects, returning a list of OpenAI-shape
# messages (a single Anthropic message can expand into several, e.g. a user
# turn carrying tool_result blocks becomes one or more `tool` role messages).
#
# `img_acc` (optional environment with $paths and $marker): when supplied, image
# blocks are decoded to temp files (paths appended to img_acc$paths) and the
# media marker (img_acc$marker) is spliced into the text at the image position,
# so the chat template wraps it correctly (e.g. <|vision_start|>..<|vision_end|>).
# Without img_acc (text-only server), image blocks are dropped as before.
.anthropic_content_to_openai <- function(role, content, img_acc = NULL) {
    # Plain string content: pass through unchanged.
    if (is.character(content) && length(content) == 1) {
        return(list(list(role = role, content = content)))
    }
    # jsonlite may give a data.frame (homogeneous blocks) or a list of lists.
    blocks <- if (is.data.frame(content)) {
        lapply(seq_len(nrow(content)), function(i) as.list(content[i, , drop = FALSE]))
    } else {
        content
    }

    out <- list()
    text_parts <- character(0)
    tool_calls <- list()

    for (b in blocks) {
        type <- b$type %||% "text"
        if (type == "text") {
            text_parts <- c(text_parts, b$text %||% "")
        } else if (type == "tool_use") {
            # assistant calling a tool -> OpenAI tool_call
            tool_calls[[length(tool_calls) + 1]] <- list(
                id   = b$id %||% "",
                type = "function",
                "function" = list(
                    name      = b$name %||% "",
                    arguments = jsonlite::toJSON(b$input %||% stats::setNames(list(), character()),
                                                 auto_unbox = TRUE)
                )
            )
        } else if (type == "tool_result") {
            # user returning a tool result -> OpenAI `tool` role message
            res <- b$content
            res_txt <- if (is.character(res)) paste(res, collapse = "") else
                       jsonlite::toJSON(res, auto_unbox = TRUE)
            out[[length(out) + 1]] <- list(
                role = "tool",
                content = as.character(res_txt),
                tool_call_id = b$tool_use_id %||% ""
            )
        } else if (type == "image" && !is.null(img_acc)) {
            # Vision enabled: decode the image and splice the media marker into
            # the text where the image appeared, so chat_build's template wraps
            # it. Without img_acc this falls through and the image is dropped.
            p <- .anthropic_image_to_file(b)
            if (!is.null(p)) {
                img_acc$paths <- c(img_acc$paths, p)
                text_parts <- c(text_parts, img_acc$marker)
            }
        }
        # Other block types are intentionally dropped: image blocks when vision
        # is off (text-only) and `thinking` blocks (we strip reasoning on output,
        # so we never emit a valid signed thinking block for the client to echo
        # back; silently ignoring any that arrive keeps multi-turn robust).
    }

    # Emit the assistant/user text+tool_calls message (if any) before/after the
    # tool_result messages, preserving Anthropic's ordering: assistant tool_use
    # comes from assistant turns, tool_result from user turns, so they never mix.
    if (length(tool_calls) > 0 || length(text_parts) > 0) {
        msg <- list(role = role, content = paste(text_parts, collapse = ""))
        if (length(tool_calls) > 0) msg$tool_calls <- tool_calls
        out <- c(list(msg), out)
    }
    out
}

# Turn a full Anthropic request body into (system, messages) for llama_chat_build.
# When `marker` is non-NULL (vision server), image blocks are decoded to temp
# files and the marker is spliced into the text; the decoded paths are attached
# to the result as the "image_paths" attribute (NULL/empty when no images).
.anthropic_to_messages <- function(body, marker = NULL) {
    img_acc <- if (!is.null(marker)) {
        e <- new.env(parent = emptyenv()); e$paths <- character(0); e$marker <- marker; e
    } else NULL
    # Collect ALL system text into a single leading system message. Chat templates
    # (DeepSeek, Qwen, ...) expect exactly one system turn at the start; emitting a
    # second one — e.g. Claude Code sends a top-level `system` *and* a system role
    # inside `messages` — makes the template drop one of them (silent content
    # loss). So we merge the top-level `system` and any system-role messages, in
    # order, and keep only non-system turns in the message list.
    sys_parts <- character(0)

    # `system` is a top-level field (string or array of text blocks).
    if (!is.null(body$system)) {
        sys_txt <- if (is.character(body$system)) paste(body$system, collapse = "\n") else {
            blocks <- body$system
            if (is.data.frame(blocks)) paste(blocks$text, collapse = "\n")
            else paste(vapply(blocks, function(b) b$text %||% "", character(1)), collapse = "\n")
        }
        sys_parts <- c(sys_parts, sys_txt)
    }

    raw <- body$messages
    raw <- if (is.data.frame(raw)) {
        lapply(seq_len(nrow(raw)), function(i) list(role = raw$role[i], content = raw$content[[i]]))
    } else raw

    msgs <- list()
    for (m in raw) {
        converted <- .anthropic_content_to_openai(m$role, m$content, img_acc)
        for (cm in converted) {
            if (identical(cm$role, "system")) {
                # fold any system-role turn into the merged system text
                if (nzchar(cm$content %||% "")) sys_parts <- c(sys_parts, cm$content)
            } else {
                msgs <- c(msgs, list(cm))
            }
        }
    }

    if (length(sys_parts) > 0) {
        msgs <- c(list(list(role = "system",
                            content = paste(sys_parts, collapse = "\n\n"))), msgs)
    }
    if (!is.null(img_acc)) attr(msgs, "image_paths") <- img_acc$paths
    msgs
}

# caption-then-reason: run ONE image through the vision model to produce a
# textual observation, which the (stronger) text model then reasons over. The
# vision model is asked to DESCRIBE what's relevant to the user's question, not
# to answer it — the answer is the text model's job. Returns the caption string.
#
#   vl_model/vl_ctx/mctx : the vision pool member (model + its ctx + projector)
#   image_path           : a decoded image file
#   user_question        : the user's text for this turn (focuses the description)
#   max_tokens           : caption length budget
# The vision ctx KV is cleared first (M-RoPE needs strictly increasing positions
# across turns; see the multi-turn fix). marker goes where the image belongs.
.vision_caption <- function(vl_model, vl_ctx, mctx, image_path,
                            user_question, max_tokens = 256L) {
    marker <- llama_mtmd_marker()
    q <- if (nzchar(user_question)) user_question else "the overall content"
    prompt_msgs <- list(list(role = "user", content = paste0(
        marker, "\nDescribe in detail what you see in this image, focusing on: ",
        q)))
    built <- llama_chat_build(vl_model, prompt_msgs, enable_thinking = FALSE)
    if (!grepl(marker, built$prompt, fixed = TRUE)) {
        stop("vision: media marker did not survive the vision model's chat ",
             "template — the vision model likely lacks a vision slot.")
    }
    llama_memory_clear(vl_ctx)
    bitmap <- llama_image_load(mctx, image_path)
    n_past <- llama_image_eval(mctx, vl_ctx, built$prompt, bitmap)
    st <- llama_gen_begin_at(vl_ctx, n_past, max_new_tokens = as.integer(max_tokens),
                             temp = 0.3)
    out <- character(0)
    repeat {
        chunk <- llama_gen_next(st)
        if (is.null(chunk)) break
        out <- c(out, chunk)
    }
    out <- c(out, llama_gen_end(st))
    trimws(paste0(out, collapse = ""))
}

# Anthropic tools -> OpenAI tools shape that llama_chat_build expects.
.anthropic_tools_to_openai <- function(tools) {
    if (is.null(tools) || length(tools) == 0) return(NULL)
    tl <- if (is.data.frame(tools)) {
        lapply(seq_len(nrow(tools)), function(i) as.list(tools[i, , drop = FALSE]))
    } else tools
    lapply(tl, function(t) {
        list(type = "function", "function" = list(
            name        = t$name,
            description = t$description %||% "",
            parameters  = t$input_schema %||% list(type = "object")
        ))
    })
}

# --- response construction: internal -> Anthropic ------------------------------

# Remove <think>...</think> reasoning blocks (reasoning models emit them inline;
# the Anthropic API never carries reasoning in the text). Handles three shapes:
#   * complete <think>...</think> pairs,
#   * a dangling closing "...</think>" (template injected the opening tag),
#   * an unclosed trailing "<think>..." (reasoning still streaming, not yet
#     finished) — everything from the opener on is held back.
# Trims leftover leading whitespace. Returns the cleaned text.
.strip_thinking <- function(txt) {
    if (!nzchar(txt)) return(txt)
    # full <think>...</think> pairs (non-greedy, across newlines)
    cleaned <- gsub("(?s)<think>.*?</think>", "", txt, perl = TRUE)
    # dangling closing tag with no opener (template injected the opening <think>)
    cleaned <- gsub("(?s)^.*?</think>", "", cleaned, perl = TRUE)
    # unclosed opener: drop it and everything after (reasoning not yet complete)
    cleaned <- gsub("(?s)<think>.*$", "", cleaned, perl = TRUE)
    cleaned <- sub("^[[:space:]\r\n]+", "", cleaned)
    cleaned
}

# Build the Anthropic content array from a parsed message: a text block (if any)
# followed by one tool_use block per tool call. With strip_thinking, drop any
# <think> reasoning from the text block.
.anthropic_content_blocks <- function(parsed, strip_thinking = TRUE) {
    blocks <- list()
    text <- parsed$content %||% ""
    if (isTRUE(strip_thinking)) text <- .strip_thinking(text)
    if (nzchar(text)) {
        blocks[[length(blocks) + 1]] <- list(type = "text", text = text)
    }
    tc <- parsed$tool_calls
    if (!is.null(tc) && nrow(tc) > 0) {
        for (i in seq_len(nrow(tc))) {
            input <- tryCatch(jsonlite::fromJSON(tc$arguments[i]),
                              error = function(e) list())
            blocks[[length(blocks) + 1]] <- list(
                type  = "tool_use",
                id    = if (nzchar(tc$id[i])) tc$id[i] else
                        paste0("toolu_", paste0(sample(c(0:9, letters), 20, replace = TRUE), collapse = "")),
                name  = tc$name[i],
                input = input
            )
        }
    }
    blocks
}

# stop_reason: tool_use if any tool calls, max_tokens if length-capped, else end_turn.
.anthropic_stop_reason <- function(parsed, hit_limit) {
    if (!is.null(parsed$tool_calls) && nrow(parsed$tool_calls) > 0) return("tool_use")
    if (hit_limit) return("max_tokens")
    "end_turn"
}

#' Serve an Anthropic Messages API-compatible endpoint for a local model
#'
#' Loads a GGUF model once and exposes it over an Anthropic Messages
#' API-compatible HTTP API, so Claude Code (or any Anthropic SDK client) can run
#' against local inference. Point Claude Code at it with
#' \code{ANTHROPIC_BASE_URL=http://127.0.0.1:<port>} and any non-empty
#' \code{ANTHROPIC_API_KEY}.
#'
#' Implements \code{POST /v1/messages} (blocking and \code{stream = true}) and a
#' minimal \code{GET /v1/models}. Tool use is supported: \code{tools} in the
#' request are passed through the tool-aware chat layer
#' (\code{\link{llama_chat_build}}), generation is grammar-constrained, and the
#' model's output is parsed back into \code{tool_use} blocks
#' (\code{\link{llama_chat_parse}}).
#'
#' Single-sequence: requests are handled one at a time on the main R thread,
#' like \code{\link{llama_serve_openai}}. Meant for one local user/agent.
#'
#' @param model_path Path to a GGUF model file. Use a tool-calling-capable model
#'   (e.g. Qwen, Llama-3.x, Mistral/Mixtral) for Claude Code to work well.
#' @param port Port to listen on. Default \code{11435}.
#' @param n_ctx Context size for the loaded model. Default \code{32768}: Claude
#'   Code sends a large system prompt (tool definitions + rules, often
#'   20k+ tokens), so a small context window rejects every request. Raise it
#'   further for long sessions if the model supports it.
#' @param n_gpu_layers Layers to offload to GPU (\code{-1} = all).
#' @param model_id Identifier echoed in responses and \code{/v1/models}.
#'   Defaults to the model file's base name.
#' @param host Address to bind. Default \code{"127.0.0.1"} (local only).
#' @param max_tokens Default \code{max_tokens} when a request omits it.
#' @param strip_thinking Drop \code{<think>...</think>} reasoning blocks from the
#'   response text before returning (default \code{TRUE}). Reasoning models
#'   (DeepSeek-R1, QwQ) emit these inline; the Anthropic API never puts reasoning
#'   in the text, so Claude Code expects clean content. Non-reasoning models
#'   (Qwen, Mistral, Llama) emit no such blocks, so this is a no-op for them. Set
#'   \code{FALSE} to keep the reasoning visible (e.g. for debugging).
#' @param enable_thinking Ask the chat template to enable the model's reasoning
#'   mode (default \code{FALSE}). Hybrid thinking models (Qwen3.5, etc.) otherwise
#'   spend their whole token budget inside an unclosed \code{<think>} block and
#'   never reach the answer, which \code{strip_thinking} then turns into an empty
#'   reply. Kept \code{FALSE} so Claude Code gets direct answers and fast tool
#'   calls; set \code{TRUE} only if you also raise \code{max_tokens} enough for
#'   the model to finish reasoning. No effect on non-thinking models.
#' @param vision_model_path Optional path to a SECOND, vision-capable GGUF model
#'   (e.g. Qwen2-VL). When given together with \code{mmproj_path}, the server
#'   uses a \emph{caption-then-reason} pipeline: a request carrying an image is
#'   first passed to this vision model, which DESCRIBES the image (focused on the
#'   user's question); that description is then spliced into the conversation as
#'   text and answered by the main \code{model_path} model — so the stronger text
#'   model does the reasoning, tool calls, and streaming, while the vision model
#'   only provides "eyes". \code{NULL} (default) keeps the server text-only and
#'   image blocks are dropped, as before.
#' @param mmproj_path Path to the clip projector (mmproj) GGUF paired with
#'   \code{vision_model_path}. Required to enable vision; must match that model.
#' @param vision_n_ctx Context size for the vision model's own context
#'   (default \code{8192}). Vision turns (screenshot + question) are short, so a
#'   small KV cache keeps both models within VRAM.
#' @param vision_debug If \code{TRUE}, log each vision caption to the server
#'   console (not returned to the client). Default \code{FALSE}: the caption is
#'   internal, the user sees only the text model's final answer.
#' @param ... Reserved for future options.
#'
#' @return Invisibly \code{NULL}. Blocks serving until interrupted.
#' @seealso [llama_serve_openai], [llama_chat_build], [llama_chat_parse]
#' @export
#' @examples
#' \dontrun{
#' llama_serve_anthropic("Qwen3-14B-Q8_0.gguf", port = 11435L)
#' # Then, in another shell:
#' #   ANTHROPIC_BASE_URL=http://127.0.0.1:11435 \
#' #   ANTHROPIC_API_KEY=sk-local claude
#' }
llama_serve_anthropic <- function(model_path, port = 11435L,
                                  n_ctx = 32768L, n_gpu_layers = -1L,
                                  model_id = NULL, host = "127.0.0.1",
                                  max_tokens = 1024L, strip_thinking = TRUE,
                                  enable_thinking = FALSE,
                                  vision_model_path = NULL, mmproj_path = NULL,
                                  vision_n_ctx = 8192L, vision_debug = FALSE, ...) {
    if (!requireNamespace("drogonR", quietly = TRUE)) {
        stop("llama_serve_anthropic() requires the 'drogonR' package: ",
             "install.packages('drogonR')", call. = FALSE)
    }
    if (!file.exists(model_path)) {
        stop("model file not found: ", model_path, call. = FALSE)
    }
    # Vision is enabled only when BOTH a vision model and its projector are given.
    vision_on <- !is.null(vision_model_path) && !is.null(mmproj_path)
    if (!is.null(vision_model_path) && !file.exists(vision_model_path)) {
        stop("vision model file not found: ", vision_model_path, call. = FALSE)
    }
    if (!is.null(mmproj_path) && !file.exists(mmproj_path)) {
        stop("mmproj (vision projector) file not found: ", mmproj_path, call. = FALSE)
    }
    if (xor(is.null(vision_model_path), is.null(mmproj_path))) {
        stop("vision needs BOTH vision_model_path and mmproj_path (got only one).",
             call. = FALSE)
    }
    model_id <- model_id %||% tools::file_path_sans_ext(basename(model_path))

    # Text model (primary): used for all non-image requests.
    model <- llama_load_model(model_path, n_gpu_layers = as.integer(n_gpu_layers))
    ctx   <- llama_new_context(model, n_ctx = as.integer(n_ctx))
    ctx_size <- llama_n_ctx(ctx)

    # Vision model pool member: a SEPARATE model + its own (smaller) context +
    # clip projector, loaded only when vision is enabled. Requests carrying an
    # image are routed here; everything else stays on the text model. Both models
    # live in VRAM at once, so the vision context is kept small (vision_n_ctx):
    # screenshot + question turns are short and don't need a large KV cache.
    # When vision_model_path is NULL the server behaves exactly as before
    # (text-only, image blocks dropped) — backward compatible.
    vl_model <- NULL; vl_ctx <- NULL; vl_ctx_size <- 0L; mctx <- NULL
    if (vision_on) {
        vl_model <- llama_load_model(vision_model_path, n_gpu_layers = as.integer(n_gpu_layers))
        vl_ctx   <- llama_new_context(vl_model, n_ctx = as.integer(vision_n_ctx))
        vl_ctx_size <- llama_n_ctx(vl_ctx)
        mctx     <- llama_mtmd_load(vl_model, mmproj_path)
        message(sprintf("vision enabled: model '%s' + mmproj '%s' (supports_vision=%s, n_ctx=%d)",
                        basename(vision_model_path), basename(mmproj_path),
                        llama_mtmd_support_vision(mctx), vl_ctx_size))
    }

    # Single-ctx serialization queue. One shared ctx / KV cache means only one
    # request may generate at a time: a second concurrent request (Claude Code
    # fires the main turn and a session-title turn at once) calling gen_begin ->
    # llama_memory_clear + prefill on top of an in-flight request corrupts both
    # (garbled, swallowed tokens). Rather than reject overlap with 503 and make
    # the client retry, we SERIALIZE: every request — streaming or blocking — is
    # answered through dr_stream (drogonR's only deferred-response mechanism), so
    # its next_chunk pumps across event-loop ticks. Each request takes a FIFO
    # ticket; a request starts generating only when the ctx is free AND its ticket
    # is at the head of the queue. Until then it emits keep-alives and yields.
    #
    # State lives in the handler closure (single R thread, so no locking needed):
    #   ctx_busy     — TRUE while a request holds the ctx and is generating.
    #   q_next       — monotonic ticket counter (next ticket to hand out).
    #   q_head       — ticket currently allowed to acquire (head of the FIFO).
    # A request with ticket T may acquire when !ctx_busy && T == q_head. On
    # release it sets ctx_busy<-FALSE and bumps q_head so the next waiter runs.
    # Tickets that give up (client disconnect before acquiring) are skipped by
    # advancing q_head past any ticket no longer waiting (tracked in q_waiting).
    ctx_busy  <- FALSE
    q_next    <- 1L            # next ticket to assign
    q_head    <- 1L            # ticket allowed to acquire the ctx now
    q_waiting <- integer(0)    # tickets still waiting (for skip-on-disconnect)

    new_id <- function() paste0("msg_", paste0(
        sample(c(0:9, letters), 24, replace = TRUE), collapse = ""))

    app <- drogonR::dr_app()

    # --- GET /v1/models ---
    app <- drogonR::dr_get(app, "/v1/models", function(req) {
        drogonR::dr_json(list(
            data = list(list(
                id         = model_id,
                type       = "model",
                display_name = model_id,
                created_at = format(Sys.time(), "%Y-%m-%dT%H:%M:%SZ")
            )),
            has_more = FALSE
        ))
    })

    # --- POST /v1/messages ---
    app <- drogonR::dr_post(app, "/v1/messages", function(req) {
        # Parse with simplifyVector = FALSE so JSON arrays stay as lists rather
        # than collapsing (single-element arrays like "required":["city"] would
        # otherwise become scalars, and objects would become data.frames),
        # which keeps tool input_schema valid for the C++ JSON-schema converter.
        body <- jsonlite::fromJSON(drogonR::dr_body(req, as = "text") %||% "null",
                                   simplifyVector = FALSE)
        if (is.null(body$messages)) {
            return(drogonR::dr_json(list(type = "error", error = list(
                type = "invalid_request_error", message = "missing 'messages'")),
                status = 400L))
        }

        # Take a FIFO ticket. The request only touches the ctx (chat_build,
        # tokenize, gen_begin) once it reaches the head of the queue AND the ctx is
        # free — see the `acquire` phase in next_chunk below. Everything up to here
        # is ctx-free, so it's safe to run on the request thread immediately.
        ticket <- q_next
        q_next   <<- q_next + 1L
        q_waiting <<- c(q_waiting, ticket)
        message(sprintf("[queue #%d] ENQUEUE (stream=%s, waiting=%d)",  # DEBUG
                        ticket, isTRUE(body$stream), length(q_waiting)))

        stream <- isTRUE(body$stream)
        id     <- new_id()

        # release_ctx(): hand the ctx to the next waiter. Called once from a
        # terminal/cancel path. Clears ctx_busy and advances q_head past this
        # ticket so the next FIFO waiter acquires on its following pump. Guarded
        # `q_head <= ticket` so a late call can't rewind the head.
        release_ctx <- function() {
            ctx_busy <<- FALSE
            if (q_head <= ticket) q_head <<- ticket + 1L
            message(sprintf("[queue #%d] RELEASE ctx (head now %d)", ticket, q_head))  # DEBUG
        }
        # drop_ticket(): a waiter that never acquired (client vanished while still
        # queued) must step out of the FIFO so it doesn't wedge waiters behind it.
        drop_ticket <- function() {
            q_waiting <<- setdiff(q_waiting, ticket)
            if (q_head == ticket) q_head <<- ticket + 1L
        }

        # Build the Anthropic content array + stop_reason for a finished
        # generation. Shared by the streaming finalize and the blocking JSON body.
        build_message_body <- function(raw, n) {
            parsed <- llama_chat_parse(raw, format = built$format,
                                       parser = built$parser)
            blocks <- .anthropic_content_blocks(parsed, strip_thinking)
            # Never return empty content: Claude Code rejects a message with no
            # content blocks as "empty or malformed response". A thinking model
            # can also spend its whole budget inside an unclosed <think> and
            # never reach the answer; stripping then leaves nothing. Fall back to
            # a single (possibly raw) text block — reasoning beats silence.
            if (length(blocks) == 0) {
                fb <- if (nzchar(parsed$content %||% "")) parsed$content else " "
                blocks <- list(list(type = "text", text = fb))
            }
            list(
                id      = id,
                type    = "message",
                role    = "assistant",
                model   = model_id,
                content = blocks,
                stop_reason   = .anthropic_stop_reason(parsed, n >= req_max),
                stop_sequence = NA,
                usage = list(input_tokens = prompt_tokens, output_tokens = n)
            )
        }

        # Anthropic SSE frames need an `event:` name plus a JSON `data:` line,
        # which dr_stream_sse can't express (it only emits `data:`). So we build
        # complete raw frames ("event: X\ndata: {...}\n\n") and feed them through
        # the lower-level dr_stream, which writes the chunk verbatim.
        sse <- function(event, obj) {
            paste0("event: ", event, "\n",
                   "data: ", as.character(jsonlite::toJSON(obj, auto_unbox = TRUE)),
                   "\n\n")
        }

        # Per-request streaming state. Both streaming and blocking requests are
        # answered through dr_stream (drogonR's only deferred-response mechanism)
        # so they serialize on the single ctx via the FIFO queue. The phase
        # machine: wait -> acquire -> start -> text -> finalize -> done. For
        # blocking requests (`stream == FALSE`) the intermediate phases emit
        # nothing; finalize sends the whole JSON message as one chunk.
        state <- new.env(parent = emptyenv())
        state$phase    <- "wait"
        state$n        <- 0L
        state$text     <- character(0)
        state$acquired <- FALSE

        # These are filled in at acquire time (they touch the ctx, so they must
        # not run until this request owns it).
        st           <- NULL
        prompt_tokens <- 0L
        req_max      <- 0L
        defer_stream <- FALSE   # set at acquire for GENERIC format (see below)
        built        <- NULL

        next_chunk <- function(s, cancelled) {
            if (cancelled) {
                # Client gone. If we held the ctx, end generation and release it;
                # if we were still queued, step out of the FIFO.
                if (isTRUE(state$acquired)) { tryCatch(llama_gen_end(st), error = function(.) NULL); release_ctx() }
                else drop_ticket()
                return(list(chunk = "", state = s, done = TRUE))
            }

          tryCatch({
            # --- wait: not our turn yet, or ctx busy. Yield with a keep-alive. ---
            if (state$phase == "wait") {
                if (ctx_busy || q_head != ticket) {
                    # Streaming clients get an SSE ping keep-alive; blocking
                    # clients can't see body bytes yet (no framing), so send "".
                    ka <- if (stream) sse("ping", list(type = "ping")) else ""
                    return(list(chunk = ka, state = s, done = FALSE))
                }
                state$phase <- "acquire"
                # fall through
            }

            # --- acquire: take the ctx, build the prompt, begin generation. ---
            if (state$phase == "acquire") {
                ctx_busy <<- TRUE
                state$acquired <<- TRUE
                q_waiting <<- setdiff(q_waiting, ticket)
                message(sprintf("[queue #%d] ACQUIRE ctx", ticket))  # DEBUG

                # Vision: pass the media marker so image blocks are decoded and
                # spliced into the message text; collect the decoded file paths.
                vision_marker <- if (!is.null(mctx)) llama_mtmd_marker() else NULL
                messages <- .anthropic_to_messages(body, marker = vision_marker)
                img_paths <- attr(messages, "image_paths") %||% character(0)
                tools    <- .anthropic_tools_to_openai(body$tools)
                req_max  <<- as.integer(body$max_tokens %||% max_tokens)
                tool_choice <- if (!is.null(body$tool_choice)) {
                    tc <- body$tool_choice$type %||% body$tool_choice
                    if (identical(tc, "any")) "required" else if (identical(tc, "tool")) "required" else "auto"
                } else NULL

                # output_config.effort (low|medium|high): Claude Code's reasoning-
                # effort hint. Local GGUF models have no native effort control, so
                # map it onto the generation budget — higher effort allows more
                # tokens (longer chain-of-thought) before truncation.
                effort <- body$output_config$effort %||% NULL
                if (!is.null(effort)) {
                    mult <- switch(effort, low = 0.5, medium = 0.75, high = 1.0, 1.0)
                    req_max <<- as.integer(max(1L, floor(req_max * mult)))
                }

                has_image <- length(img_paths) > 0
                # caption-then-reason: an image request is preprocessed by the
                # VISION model into a textual observation, then answered by the
                # TEXT model like any other turn (full tools/grammar/streaming).
                # So we DON'T route the whole request to the small vision model;
                # we only borrow its eyes. The user's question focuses the caption,
                # then each image marker in the messages is replaced by
                # "[Image description: ...]" and the rest of the pipeline runs
                # purely on the text model.
                if (has_image) {
                    marker <- llama_mtmd_marker()
                    # The user's question for this turn = the text around the
                    # marker in the last user message (drives a focused caption).
                    last_user <- ""
                    for (mm in rev(messages)) if (identical(mm$role, "user")) { last_user <- mm$content %||% ""; break }
                    user_q <- trimws(gsub(marker, " ", last_user, fixed = TRUE))
                    captions <- vapply(img_paths, function(p)
                        .vision_caption(vl_model, vl_ctx, mctx, p, user_q,
                                        max_tokens = 256L), character(1))
                    unlink(img_paths)
                    if (isTRUE(vision_debug)) {
                        for (cap in captions)
                            message(sprintf("[queue #%d] vision caption: %s", ticket, cap))
                    }
                    # Replace each marker (in order) with its caption text.
                    ci <- 0L
                    repl_marker <- function(s) {
                        while (grepl(marker, s, fixed = TRUE) && ci < length(captions)) {
                            ci <<- ci + 1L
                            s <- sub(marker, paste0("[Image description: ", captions[ci], "]"),
                                     s, fixed = TRUE)
                        }
                        s
                    }
                    messages <- lapply(messages, function(mm) {
                        if (!is.null(mm$content)) mm$content <- repl_marker(mm$content)
                        mm
                    })
                }

                # From here the request is text-only (image markers are now
                # "[Image description: ...]" text), answered by the text model.
                built <<- llama_chat_build(model, messages, tools = tools,
                                           tool_choice = tool_choice,
                                           enable_thinking = enable_thinking)
                # Decide whether to defer the stream (generate silently, parse, then
                # emit at finalize) instead of streaming raw token deltas live.
                # Raw deltas are only safe when they ARE the final text. They are NOT
                # when the model emits structured markup that only becomes valid
                # content after llama_chat_parse:
                #   * GENERIC (format id 1) wraps the whole reply in a JSON object
                #     ({"response": "..."} / {"tool_call": ...}).
                #   * Any tool-calling format (Hermes etc.) emits tool calls as
                #     <tool_call>{json}</tool_call> markup; streaming it live leaks
                #     the raw tags to the client and never yields a tool_use block.
                # So defer whenever the request carries tools, or for GENERIC. Plain
                # chat with no tools still streams live.
                defer_stream <<- identical(as.integer(built$format), 1L) ||
                                 (!is.null(tools) && length(tools) > 0)
                # triggers only for lazy grammars (see llama_chat_build)
                tp <- if (isTRUE(built$grammar_lazy)) built$trigger_patterns else NULL
                tt <- if (isTRUE(built$grammar_lazy)) built$trigger_tokens   else NULL

                prompt_tokens <<- length(llama_tokenize(ctx, built$prompt, parse_special = TRUE))
                # Clients (Claude Code especially) request a very large max_tokens
                # as an upper bound, not a demand. Clamp to what the context window
                # leaves after the prompt, with a small margin, instead of erroring.
                avail <- ctx_size - prompt_tokens - 8L
                if (avail < 1L) {
                    release_ctx()
                    emsg <- sprintf(paste0("Prompt too long: %d tokens exceed the ",
                        "%d-token context window. Restart the server with a larger ",
                        "n_ctx (e.g. N_CTX=131072 in the launcher), or compact the ",
                        "conversation."), prompt_tokens, ctx_size)
                    if (stream) {
                        # Emit a COMPLETE Anthropic message (start -> text -> stop),
                        # not a lone `error` event: Claude Code rejects an error that
                        # arrives before message_start as a malformed response. A
                        # well-formed message whose text is the error reads cleanly.
                        obj <- list(type = "message_start", message = list(
                            id = id, type = "message", role = "assistant", model = model_id,
                            content = list(), stop_reason = NA,
                            usage = list(input_tokens = prompt_tokens, output_tokens = 0L)))
                        frames <- paste0(
                            sse("message_start", obj),
                            sse("content_block_start", list(type = "content_block_start",
                                index = 0L, content_block = list(type = "text", text = ""))),
                            sse("content_block_delta", list(type = "content_block_delta",
                                index = 0L, delta = list(type = "text_delta", text = emsg))),
                            sse("content_block_stop", list(type = "content_block_stop", index = 0L)),
                            sse("message_delta", list(type = "message_delta",
                                delta = list(stop_reason = "end_turn", stop_sequence = NA),
                                usage = list(output_tokens = 0L))),
                            sse("message_stop", list(type = "message_stop")))
                        return(list(chunk = frames, state = s, done = TRUE))
                    }
                    body_json <- as.character(jsonlite::toJSON(list(type = "error",
                        error = list(type = "invalid_request_error", message = emsg)),
                        auto_unbox = TRUE))
                    return(list(chunk = body_json, state = s, done = TRUE))
                }
                req_max <<- min(req_max, avail)

                # Always the text model now: any image was turned into an
                # "[Image description: ...]" text block by the VL caption step
                # above, so generation is a normal text turn (tools/grammar incl.).
                gen_args <- list(
                    ctx, built$prompt, max_new_tokens = req_max,
                    temp = as.double(body$temperature %||% 0.7),
                    top_p = as.double(body$top_p %||% 0.9),
                    grammar = if (nzchar(built$grammar)) built$grammar else NULL,
                    trigger_patterns = tp, trigger_tokens = tt
                )
                st <<- do.call(llama_gen_begin, gen_args)
                state$phase <- "start"
                # fall through
            }

            if (state$phase == "start") {
                state$phase <- "text"
                # Blocking requests emit no intermediate frames — skip straight to
                # generating. Only streaming clients get the message_start event.
                if (!stream) return(list(chunk = "", state = s, done = FALSE))
                obj <- list(type = "message_start", message = list(
                    id = id, type = "message", role = "assistant", model = model_id,
                    content = list(), stop_reason = NA,
                    usage = list(input_tokens = prompt_tokens, output_tokens = 0L)))
                # message_start followed by a ping, matching Anthropic's stream
                # (ping is a keep-alive the SDK expects between events).
                ping <- sse("ping", list(type = "ping"))
                return(list(chunk = paste0(sse("message_start", obj), ping),
                            state = s, done = FALSE))
            }

            if (state$phase == "text") {
                chunk <- llama_gen_next(st)
                if (!is.null(chunk)) {
                    state$n <- state$n + 1L
                    state$text <- c(state$text, chunk)

                    # Blocking requests — and streaming GENERIC requests, whose raw
                    # JSON wrapper can't be streamed live — emit nothing here; just
                    # accumulate and pump again. The full reply is parsed and sent
                    # at finalize (one JSON chunk for blocking; SSE events for stream).
                    if (!stream || defer_stream) return(list(chunk = "", state = s, done = FALSE))

                    # Determine the delta to actually stream. With strip_thinking
                    # we can't pass <think> chunks through live (the closing tag
                    # arrives later), so we recompute the cleaned text each step
                    # and only emit the newly-revealed tail. While reasoning is
                    # still open the cleaned text is empty, so nothing is sent
                    # until </think> closes.
                    if (isTRUE(strip_thinking)) {
                        full_clean <- .strip_thinking(paste0(state$text, collapse = ""))
                        already <- state$emitted %||% ""
                        if (startsWith(full_clean, already) &&
                            nchar(full_clean) > nchar(already)) {
                            out_delta <- substr(full_clean, nchar(already) + 1L, nchar(full_clean))
                            state$emitted <- full_clean
                        } else {
                            out_delta <- ""
                        }
                    } else {
                        out_delta <- chunk
                    }

                    if (!nzchar(out_delta)) {
                        return(list(chunk = "", state = s, done = FALSE))
                    }
                    # stream text deltas as a single text content block (index 0)
                    if (!isTRUE(state$block_open)) {
                        state$block_open <- TRUE
                        start <- sse("content_block_start", list(
                            type = "content_block_start", index = 0L,
                            content_block = list(type = "text", text = "")))
                        delta <- sse("content_block_delta", list(
                            type = "content_block_delta", index = 0L,
                            delta = list(type = "text_delta", text = out_delta)))
                        return(list(chunk = paste0(start, delta), state = s, done = FALSE))
                    }
                    delta <- sse("content_block_delta", list(
                        type = "content_block_delta", index = 0L,
                        delta = list(type = "text_delta", text = out_delta)))
                    return(list(chunk = delta, state = s, done = FALSE))
                }
                state$text <- c(state$text, llama_gen_end(st))
                state$phase <- "finalize"
                # fall through to finalize on next pump
            }

            if (state$phase == "finalize") {
                state$phase <- "done"
                raw    <- paste0(state$text, collapse = "")
                # Blocking: emit the whole Anthropic message as one JSON chunk.
                # dr_stream was opened with content_type "application/json", so the
                # client sees an ordinary (chunked) JSON response, not SSE.
                if (!stream) {
                    release_ctx()
                    body_json <- as.character(jsonlite::toJSON(
                        build_message_body(raw, state$n), auto_unbox = TRUE))
                    return(list(chunk = body_json, state = s, done = TRUE))
                }
                parsed <- llama_chat_parse(raw, format = built$format, parser = built$parser)
                hit_limit <- state$n >= req_max
                frames <- character(0)
                # Deferred (GENERIC) text wasn't streamed live — emit the parsed
                # content now as a whole text block (start+delta+stop). This is the
                # path that unwraps {"response": "..."} into clean text.
                txt <- if (isTRUE(strip_thinking)) .strip_thinking(parsed$content %||% "")
                       else parsed$content %||% ""
                if (defer_stream && !isTRUE(state$block_open) && nzchar(txt)) {
                    state$block_open <- TRUE
                    frames <- c(frames,
                        sse("content_block_start", list(
                            type = "content_block_start", index = 0L,
                            content_block = list(type = "text", text = ""))),
                        sse("content_block_delta", list(
                            type = "content_block_delta", index = 0L,
                            delta = list(type = "text_delta", text = txt))))
                }
                # close the streamed text block if one was opened
                if (isTRUE(state$block_open)) {
                    frames <- c(frames, sse("content_block_stop",
                        list(type = "content_block_stop", index = 0L)))
                }
                # emit tool_use blocks whole (indices after the text block)
                tc <- parsed$tool_calls
                idx <- if (isTRUE(state$block_open)) 1L else 0L
                if (!is.null(tc) && nrow(tc) > 0) {
                    for (i in seq_len(nrow(tc))) {
                        tuid <- if (nzchar(tc$id[i])) tc$id[i] else
                                paste0("toolu_", paste0(sample(c(0:9, letters), 20, replace = TRUE), collapse = ""))
                        frames <- c(frames,
                            sse("content_block_start", list(
                                type = "content_block_start", index = idx,
                                content_block = list(type = "tool_use", id = tuid,
                                                     name = tc$name[i], input = stats::setNames(list(), character())))),
                            sse("content_block_delta", list(
                                type = "content_block_delta", index = idx,
                                delta = list(type = "input_json_delta",
                                             partial_json = tc$arguments[i]))),
                            sse("content_block_stop", list(
                                type = "content_block_stop", index = idx)))
                        idx <- idx + 1L
                    }
                }
                # Guarantee at least one content block. If parsing yielded neither
                # text nor tool calls (e.g. a thinking model spent its budget inside
                # <think> and strip_thinking emptied it, or the model returned only
                # whitespace), an Anthropic message with empty `content` is rejected
                # by Claude Code as "empty or malformed response". Emit a minimal
                # text block so the message is always well-formed.
                if (idx == 0L && !isTRUE(state$block_open)) {
                    fallback <- if (nzchar(raw)) raw else " "
                    frames <- c(frames,
                        sse("content_block_start", list(
                            type = "content_block_start", index = 0L,
                            content_block = list(type = "text", text = ""))),
                        sse("content_block_delta", list(
                            type = "content_block_delta", index = 0L,
                            delta = list(type = "text_delta", text = fallback))),
                        sse("content_block_stop", list(
                            type = "content_block_stop", index = 0L)))
                }
                frames <- c(frames,
                    sse("message_delta", list(
                        type = "message_delta",
                        delta = list(stop_reason = .anthropic_stop_reason(parsed, hit_limit),
                                     stop_sequence = NA),
                        usage = list(output_tokens = state$n))),
                    sse("message_stop", list(type = "message_stop")))
                release_ctx()
                return(list(chunk = paste0(frames, collapse = ""), state = s, done = TRUE))
            }

            release_ctx()
            list(chunk = "", state = s, done = TRUE)
          }, error = function(e) {
            # Generation failed mid-stream: emit an error so the client sees a
            # clean failure instead of a truncated stream, then end. Release the
            # ctx if we held it; a failure before acquire leaves the FIFO via
            # drop_ticket so waiters behind us aren't wedged.
            if (isTRUE(state$acquired)) {
                tryCatch(llama_gen_end(st), error = function(.) NULL)
                release_ctx()
            } else drop_ticket()
            if (stream) {
                err <- sse("error", list(type = "error", error = list(
                    type = "api_error", message = conditionMessage(e))))
                return(list(chunk = err, state = s, done = TRUE))
            }
            body_json <- as.character(jsonlite::toJSON(list(type = "error",
                error = list(type = "api_error", message = conditionMessage(e))),
                auto_unbox = TRUE))
            list(chunk = body_json, state = s, done = TRUE)
          })
        }

        # Every request — streaming or blocking — is answered through dr_stream so
        # it serializes on the single ctx via the FIFO queue (next_chunk pumps
        # across event-loop ticks; the ctx is acquired only at the head of the
        # queue). Streaming clients get SSE; blocking clients get a single JSON
        # chunk, so the content type differs. The queue/ctx are released inside
        # next_chunk's terminal/cancel paths, never here.
        ct <- if (stream) "text/event-stream" else "application/json"
        drogonR::dr_stream(next_chunk, state = state, content_type = ct)
    })

    message(sprintf("llamaR Anthropic server: http://%s:%d  (model '%s')\n  Point Claude Code at it: ANTHROPIC_BASE_URL=http://%s:%d ANTHROPIC_API_KEY=sk-local claude",
                    host, as.integer(port), model_id, host, as.integer(port)))
    drogonR::dr_serve(app, port = as.integer(port))

    on.exit(drogonR::dr_stop(), add = TRUE)
    tryCatch(
        while (drogonR::dr_running()) {
            later::run_now(timeoutSecs = 1, all = TRUE)
        },
        interrupt = function(e) invisible(NULL)
    )
    invisible(NULL)
}
