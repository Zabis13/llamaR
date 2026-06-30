#!/usr/bin/env Rscript
# A minimal stdio MCP server exposing two tools:
#   * web_search(query) -> top DuckDuckGo results (title, URL, snippet)
#   * web_fetch(url)    -> the readable text of a web page
# Backed by DuckDuckGo's no-JS endpoints and plain HTTP. No API key, no extra
# service: everything runs in *this* process on your machine, so it works behind
# a local llamaR Anthropic server (the model calls the tools; this server does
# the network I/O and feeds results back into the model's context).
#
# Wire it into Claude Code with (run from any directory):
#
#   claude mcp add web-search -- Rscript /abs/path/to/mcp_web_search.R
#
# or add to ~/.claude.json / .mcp.json manually:
#
#   { "mcpServers": { "web-search": {
#       "command": "Rscript",
#       "args": ["/abs/path/to/mcp_web_search.R"] } } }
#
# Then the local model can call the `web_search` tool; Claude Code executes the
# search here and feeds the results back into the model's context.
#
# Protocol: MCP over stdio = newline-delimited JSON-RPC 2.0 on stdin/stdout.
# We implement the three methods Claude Code needs: initialize, tools/list,
# tools/call (plus the notifications/initialized no-op). All logging goes to
# stderr so it never corrupts the stdout JSON-RPC stream.

suppressPackageStartupMessages({
    library(jsonlite)
    library(curl)
    library(rvest)
    library(xml2)
})

PROTOCOL_VERSION <- "2024-11-05"

`%||%` <- function(a, b) if (is.null(a)) b else a
log_err <- function(...) cat(sprintf(...), "\n", file = stderr())

# --- DuckDuckGo HTML search ---------------------------------------------------
# DuckDuckGo has two server-rendered (no-JS) endpoints we can scrape:
#   * lite.duckduckgo.com/lite/  -> a table layout, links as <a.result-link>
#     with clean hrefs and snippets in <td.result-snippet>. Less aggressively
#     rate-limited, so we try this first.
#   * html.duckduckgo.com/html/  -> richer .result__body blocks; used as a
#     fallback when lite yields nothing.
# DDG throttles rapid/automated requests by serving an HTTP 202 anti-bot page
# instead of 200; we retry a couple of times with a short backoff on 202.

.ddg_fetch <- function(url, query, retries = 2L) {
    for (attempt in seq_len(retries + 1L)) {
        h <- new_handle()
        handle_setheaders(h,
            "User-Agent" = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36")
        handle_setform(h, q = query)
        r <- curl_fetch_memory(url, h)
        if (r$status_code == 200L) return(read_html(rawToChar(r$content)))
        if (r$status_code == 202L && attempt <= retries) {
            Sys.sleep(0.7 * attempt)  # backoff on anti-bot throttle
            next
        }
        stop(sprintf("DuckDuckGo (%s) returned HTTP %d", url, r$status_code))
    }
    stop("DuckDuckGo kept returning HTTP 202 (rate limited)")
}

.parse_lite <- function(doc, n) {
    links <- html_elements(doc, "a.result-link")
    snips <- html_elements(doc, "td.result-snippet")
    if (length(links) == 0L) return(NULL)
    k <- min(n, length(links))
    vapply(seq_len(k), function(i) {
        title <- trimws(html_text2(links[[i]]))
        url   <- html_attr(links[[i]], "href"); url <- if (is.na(url)) "" else trimws(url)
        snip  <- if (i <= length(snips)) trimws(html_text2(snips[[i]])) else ""
        sprintf("%d. %s\n%s\n%s", i, title, url, snip)
    }, character(1))
}

.parse_html <- function(doc, n) {
    nodes <- html_elements(doc, ".result__body")
    if (length(nodes) == 0L) return(NULL)
    k <- min(n, length(nodes))
    vapply(seq_len(k), function(i) {
        title   <- html_text2(html_element(nodes[[i]], ".result__title"))
        snippet <- html_text2(html_element(nodes[[i]], ".result__snippet"))
        url     <- html_attr(html_element(nodes[[i]], ".result__a"), "href")
        title   <- if (is.na(title)) "" else trimws(title)
        snippet <- if (is.na(snippet)) "" else trimws(snippet)
        url     <- if (is.na(url)) "" else trimws(url)
        sprintf("%d. %s\n%s\n%s", i, title, url, snippet)
    }, character(1))
}

ddg_search <- function(query, max_results = 5L) {
    # Try the lite endpoint first, fall back to the html one. DuckDuckGo throttles
    # bursts of automated requests with HTTP 202; if BOTH endpoints throttle, tell
    # the model plainly so it waits/retries instead of looping on the same call.
    throttled <- FALSE
    grab <- function(url, parser) {
        tryCatch(parser(.ddg_fetch(url, query), max_results),
                 error = function(e) {
                     if (grepl("HTTP 202|rate limited", conditionMessage(e))) throttled <<- TRUE
                     NULL
                 })
    }
    parts <- grab("https://lite.duckduckgo.com/lite/", .parse_lite)
    if (is.null(parts)) parts <- grab("https://html.duckduckgo.com/html/", .parse_html)
    if (is.null(parts) || length(parts) == 0L) {
        if (throttled)
            return(paste("Web search is temporarily rate-limited by DuckDuckGo.",
                         "Wait a minute before searching again."))
        return("No results found.")
    }
    paste(parts, collapse = "\n\n")
}

# --- web_fetch: download a URL and return readable text -----------------------
# Fetches a URL and, for HTML, strips scripts/styles/nav-only nodes and collapses
# whitespace to return the page's readable text. Non-HTML (JSON, plain text) is
# returned verbatim. Output is capped at `max_chars` so a huge page can't blow up
# the model's context. Only http/https are allowed (no file://, no other schemes).
fetch_url <- function(url, max_chars = 8000L) {
    if (!grepl("^https?://", url, ignore.case = TRUE))
        stop("Only http(s) URLs are allowed")
    h <- new_handle()
    handle_setheaders(h,
        "User-Agent" = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36")
    r <- curl_fetch_memory(url, h)
    if (r$status_code != 200L)
        stop(sprintf("Fetch returned HTTP %d", r$status_code))
    ctype <- tolower(handle_data(h)$type %||% "")
    raw_txt <- tryCatch(rawToChar(r$content), error = function(e) "")

    text <- if (grepl("html", ctype) || grepl("^\\s*<", raw_txt)) {
        doc <- read_html(raw_txt)
        xml_remove(html_elements(doc, "script, style, noscript, svg, head, nav, footer"))
        body <- html_element(doc, "body")
        body <- if (length(body) == 0L || is.na(body)) doc else body
        gsub("\n{3,}", "\n\n", trimws(html_text2(body)))
    } else {
        raw_txt
    }
    if (nchar(text) > max_chars)
        text <- paste0(substr(text, 1L, max_chars), "\n\n[...truncated...]")
    text
}

# --- JSON-RPC plumbing --------------------------------------------------------
send <- function(obj) {
    # One JSON object per line on stdout; auto_unbox so scalars aren't arrays.
    cat(toJSON(obj, auto_unbox = TRUE, null = "null"), "\n", sep = "")
    flush(stdout())
}

result_msg <- function(id, result) list(jsonrpc = "2.0", id = id, result = result)
error_msg  <- function(id, code, message)
    list(jsonrpc = "2.0", id = id, error = list(code = code, message = message))

# The tools we advertise. Each inputSchema is a JSON Schema object.
TOOLS <- list(
    list(
        name = "web_search",
        description = paste(
            "Search the public web via DuckDuckGo and return the top results",
            "(title, URL, snippet) as text. Use for current events, documentation,",
            "facts, or anything outside the model's training data."),
        inputSchema = list(
            type = "object",
            properties = list(
                query = list(type = "string", description = "The search query."),
                max_results = list(type = "integer",
                    description = "How many results to return (default 5, max 10).")
            ),
            required = list("query")
        )
    ),
    list(
        name = "web_fetch",
        description = paste(
            "Download a web page by URL and return its readable text content.",
            "Use after web_search to read a result in full, or to fetch any known",
            "http(s) URL (docs, articles, raw files)."),
        inputSchema = list(
            type = "object",
            properties = list(
                url = list(type = "string", description = "The http(s) URL to fetch."),
                max_chars = list(type = "integer",
                    description = "Truncate output to this many characters (default 8000).")
            ),
            required = list("url")
        )
    )
)

# Run one tool by name; returns the text result (or a "Search/Fetch failed: ..."
# message). Errors are reported to the model as text, not as JSON-RPC errors, so
# it can react (retry, pick another URL) instead of the call hard-failing.
run_tool <- function(name, args) {
    if (name == "web_search") {
        query <- args$query
        if (is.null(query) || !nzchar(query)) stop("Missing required argument: query")
        mr <- max(1L, min(10L, as.integer(args$max_results %||% 5L)))
        tryCatch(ddg_search(query, mr),
                 error = function(e) paste("Search failed:", conditionMessage(e)))
    } else if (name == "web_fetch") {
        url <- args$url
        if (is.null(url) || !nzchar(url)) stop("Missing required argument: url")
        mc <- max(200L, as.integer(args$max_chars %||% 8000L))
        tryCatch(fetch_url(url, mc),
                 error = function(e) paste("Fetch failed:", conditionMessage(e)))
    } else {
        stop(paste("Unknown tool:", name))
    }
}

handle <- function(req) {
    method <- req$method
    id <- req$id  # absent for notifications

    if (method == "initialize") {
        return(result_msg(id, list(
            protocolVersion = PROTOCOL_VERSION,
            capabilities = list(tools = setNames(list(), character(0))),
            serverInfo = list(name = "llamaR-web-search", version = "0.1.0")
        )))
    }
    if (method == "notifications/initialized" || method == "initialized") {
        return(NULL)  # notification: no response
    }
    if (method == "tools/list") {
        return(result_msg(id, list(tools = TOOLS)))
    }
    if (method == "tools/call") {
        params <- req$params
        name <- params$name %||% ""
        args <- params$arguments %||% list()
        text <- tryCatch(run_tool(name, args),
                         error = function(e) conditionMessage(e))
        log_err("[%s] %s -> %d chars", name,
                args$query %||% args$url %||% "", nchar(text))
        return(result_msg(id, list(
            content = list(list(type = "text", text = text)),
            isError = FALSE
        )))
    }
    # Unknown method: only answer requests (those with an id), ignore notifications.
    if (!is.null(id)) return(error_msg(id, -32601, paste("Method not found:", method)))
    NULL
}

# --- main loop: read one JSON-RPC message per line ----------------------------
log_err("llamaR web-search MCP server ready (stdio)")
con <- file("stdin", open = "r")
repeat {
    line <- readLines(con, n = 1L, warn = FALSE)
    if (length(line) == 0L) break          # EOF: client closed the pipe
    line <- trimws(line)
    if (!nzchar(line)) next
    req <- tryCatch(fromJSON(line, simplifyVector = FALSE),
                    error = function(e) NULL)
    if (is.null(req)) { log_err("bad JSON: %s", line); next }
    resp <- tryCatch(handle(req), error = function(e) {
        log_err("handler error: %s", conditionMessage(e))
        if (!is.null(req$id)) error_msg(req$id, -32603, conditionMessage(e)) else NULL
    })
    if (!is.null(resp)) send(resp)
}
