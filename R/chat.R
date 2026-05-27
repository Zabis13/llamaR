# ellmer::Chat constructor for a local llamaR model (DBI-style: either
# connect to a running server, or spin one up and own its lifetime).
# See chat_llamar() below.

# Poll until the server accepts TCP connections on `port`, the spawned
# process dies, or the deadline passes. socketConnection() avoids pulling
# in an HTTP-client dependency just for a readiness check.
.wait_for_server <- function(proc, port, host = "127.0.0.1",
                             timeout = 60) {
    deadline <- Sys.time() + timeout
    repeat {
        if (!proc$is_alive()) {
            err <- tryCatch(proc$read_all_error(), error = function(e) "")
            stop("llama_serve_openai process died before becoming ready",
                 if (nzchar(err)) paste0(":\n", err) else ".", call. = FALSE)
        }
        con <- tryCatch(
            socketConnection(host, port, open = "r", blocking = TRUE,
                             timeout = 1),
            warning = function(w) NULL, error = function(e) NULL)
        if (!is.null(con)) {
            close(con)
            return(invisible(TRUE))
        }
        if (Sys.time() > deadline) {
            proc$kill()
            stop(sprintf("timed out after %gs waiting for the server on port %d",
                         timeout, port), call. = FALSE)
        }
        Sys.sleep(0.2)
    }
}

#' Chat with a local model through an ellmer::Chat object
#'
#' Returns an \pkg{ellmer} \code{Chat} object backed by a local GGUF model,
#' so the whole ellmer / ragnar toolchain (turns, tools, streaming,
#' structured output, \code{ragnar_register_tool_retrieve()}, …) works
#' against local inference. Transport is the OpenAI-compatible HTTP API
#' from \code{\link{llama_serve_openai}}; this function is a thin
#' \code{\link[ellmer]{chat_vllm}} wrapper over it. (We use the vLLM provider
#' because it speaks \code{/v1/chat/completions} — the de-facto standard our
#' server implements — whereas ellmer's \code{chat_openai}/
#' \code{chat_openai_compatible} target OpenAI's newer \code{/v1/responses}.)
#'
#' Two modes, picked by which argument you pass (DBI-style — like
#' \code{DBI::dbConnect()} accepting either connection parameters or a
#' ready connection):
#'
#' \describe{
#'   \item{\code{base_url}}{Connect to a server you already started (e.g.
#'     \code{llama_serve_openai()} in another process, or a worker pool).
#'     No process is spawned.}
#'   \item{\code{model_path}}{Spin up \code{llama_serve_openai()} in a
#'     background R process (via \pkg{callr}), wait for it to come up, and
#'     return a \code{Chat} pointed at it. The server process's lifetime is
#'     tied to the returned object: when it is garbage-collected (or R
#'     exits), the process is killed. Stop it eagerly with
#'     \code{\link{chat_llamar_stop}}.}
#' }
#'
#' Exactly one of \code{base_url} or \code{model_path} must be supplied.
#'
#' @section Concurrency:
#' The server is single-sequence (one request at a time); see
#' \code{\link{llama_serve_openai}}. For parallel sessions, run a pool of
#' servers on different ports and create one \code{chat_llamar(base_url=)}
#' per worker.
#'
#' @section Tool calls:
#' Tool calling and structured output are mediated by the OpenAI protocol,
#' so they work only as far as the server implements them. The current
#' server does not emit \code{tool_calls} yet (see TODO), so ellmer tools
#' registered on the returned chat will not be invoked by the model.
#'
#' @param model_path Path to a GGUF model file. Spawns a server (mode A).
#'   Mutually exclusive with \code{base_url}.
#' @param base_url Base URL of a running OpenAI-compatible server, e.g.
#'   \code{"http://127.0.0.1:11434/v1"}. Connects to it (mode B). Mutually
#'   exclusive with \code{model_path}.
#' @param port Port for the spawned server (mode A only). Default
#'   \code{11434}.
#' @param n_ctx,n_gpu_layers Passed to \code{\link{llama_serve_openai}}
#'   when spawning (mode A only).
#' @param model_id Model identifier reported to ellmer. Defaults to the
#'   model file's base name in mode A; \code{"llamar"} in mode B.
#' @param system_prompt Optional system prompt for the chat.
#' @param timeout Seconds to wait for a spawned server to accept
#'   connections before erroring (mode A only). Default \code{180} — large
#'   models (e.g. a 14B at Q8) can take a couple of minutes to load from disk.
#' @param ... Passed on to \code{\link[ellmer]{chat_vllm}}.
#'
#' @return An \pkg{ellmer} \code{Chat} object. In mode A it additionally
#'   carries the background process handle (see \code{\link{chat_llamar_stop}}).
#' @seealso [llama_serve_openai], [chat_llamar_stop]
#' @export
#' @examples
#' \dontrun{
#' # Mode A: spawn a server for this model and chat with it.
#' chat <- chat_llamar(model_path = "model.gguf")
#' chat$chat("Why is the sky blue?")
#' chat_llamar_stop(chat)            # or let GC do it
#'
#' # Mode B: connect to a server you already run.
#' llama_serve_openai("model.gguf", port = 11434L)   # in another process
#' chat <- chat_llamar(base_url = "http://127.0.0.1:11434/v1")
#' chat$chat("Hello!")
#' }
chat_llamar <- function(model_path = NULL, base_url = NULL, port = 11434L,
                        n_ctx = 4096L, n_gpu_layers = -1L, model_id = NULL,
                        system_prompt = NULL, timeout = 180, ...) {
    if (!requireNamespace("ellmer", quietly = TRUE)) {
        stop("chat_llamar() requires the 'ellmer' package: ",
             "install.packages('ellmer')", call. = FALSE)
    }
    has_path <- !is.null(model_path)
    has_url  <- !is.null(base_url)
    if (has_path == has_url) {
        stop("supply exactly one of `model_path` or `base_url`", call. = FALSE)
    }

    # --- mode B: connect to an already-running server --------------------
    if (has_url) {
        return(ellmer::chat_vllm(
            base_url = base_url,
            model    = model_id %||% "llamar",
            credentials = function() "sk-noauth",
            system_prompt = system_prompt,
            ...))
    }

    # --- mode A: spawn a server in a background process ------------------
    if (!requireNamespace("callr", quietly = TRUE)) {
        stop("spawning a server (model_path=) requires the 'callr' package: ",
             "install.packages('callr')", call. = FALSE)
    }
    if (!file.exists(model_path)) {
        stop("model file not found: ", model_path, call. = FALSE)
    }
    port     <- as.integer(port)
    model_id <- model_id %||% tools::file_path_sans_ext(basename(model_path))

    proc <- callr::r_bg(
        function(model_path, port, n_ctx, n_gpu_layers, model_id) {
            llamaR::llama_serve_openai(model_path, port = port, n_ctx = n_ctx,
                                       n_gpu_layers = n_gpu_layers,
                                       model_id = model_id)
        },
        args = list(model_path, port, as.integer(n_ctx),
                    as.integer(n_gpu_layers), model_id))

    # If the server fails to come up, this errors (and kills the process).
    .wait_for_server(proc, port, timeout = timeout)

    chat <- ellmer::chat_vllm(
        base_url = sprintf("http://127.0.0.1:%d/v1", port),
        model    = model_id,
        credentials = function() "sk-noauth",
        system_prompt = system_prompt,
        ...)

    # Tie the server's lifetime to the chat object. The Chat is a locked R6
    # environment, so we can't stash the handle inside it; instead keep the
    # process in a small environment, hang it off the object as an attribute,
    # and finalize that environment. onexit is best-effort (R does not
    # guarantee finalizers run on exit); callr's own watchdog is the backstop,
    # killing the child when this process ends.
    holder <- new.env(parent = emptyenv())
    holder$proc <- proc
    attr(chat, "llamar_holder") <- holder
    reg.finalizer(holder, function(e) {
        if (!is.null(e$proc) && e$proc$is_alive()) e$proc$kill()
    }, onexit = TRUE)

    chat
}

#' Stop the server spawned by chat_llamar()
#'
#' Kills the background \code{\link{llama_serve_openai}} process that
#' \code{\link{chat_llamar}} started in mode A. A no-op for chats created
#' in mode B (\code{base_url=}), which own no process. Safe to call more
#' than once.
#'
#' @param chat A \code{Chat} object returned by \code{\link{chat_llamar}}.
#' @return Invisibly \code{TRUE} if a process was killed, \code{FALSE}
#'   otherwise.
#' @seealso [chat_llamar]
#' @export
chat_llamar_stop <- function(chat) {
    holder <- attr(chat, "llamar_holder", exact = TRUE)
    p <- if (is.null(holder)) NULL else holder$proc
    if (!is.null(p) && p$is_alive()) {
        p$kill()
        return(invisible(TRUE))
    }
    invisible(FALSE)
}
