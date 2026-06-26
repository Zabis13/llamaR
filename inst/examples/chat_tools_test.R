#!/usr/bin/env Rscript
# Tool-calling end-to-end test on an arbitrary model.
#   Rscript inst/examples/chat_tools_test.R /path/to/model.gguf
#
# Exercises llama_chat_build (tool-aware prompt + grammar + triggers),
# grammar-constrained generation (lazy-aware), and llama_chat_parse.

args <- commandArgs(trailingOnly = TRUE)
MODEL <- if (length(args) >= 1) args[1] else
  "/mnt/Data2/DS_projects/llm_models/Qwen3-14B-Q8_0.gguf"
if (!file.exists(MODEL)) stop("model not found: ", MODEL)

suppressMessages(library(llamaR))
cat("model:", basename(MODEL), "\n")
model <- llama_load_model(MODEL, n_gpu_layers = -1L)
ctx   <- llama_new_context(model, n_ctx = 4096L)

messages <- list(
  list(role = "system", content = "You are a helpful assistant. Use the available tools when appropriate."),
  list(role = "user",   content = "What is the weather in Paris right now?")
)
tools <- list(
  list(type = "function", "function" = list(
    name = "get_weather",
    description = "Get the current weather for a given city",
    parameters = list(
      type = "object",
      properties = list(city = list(type = "string", description = "City name")),
      required = list("city")
    )
  ))
)

for (tc in c("auto", "required")) {
  b <- llama_chat_build(model, messages, tools = tools, tool_choice = tc)
  cat(sprintf("\n=== tool_choice=%s  format=%d  lazy=%s  n_trig=%d ===\n",
              tc, b$format, b$grammar_lazy, length(b$trigger_patterns)))
  # triggers only for lazy grammars (non-lazy must constrain from token 0)
  tp <- if (isTRUE(b$grammar_lazy)) b$trigger_patterns else NULL
  tt <- if (isTRUE(b$grammar_lazy)) b$trigger_tokens   else NULL
  out <- llama_generate(ctx, b$prompt, max_new_tokens = 256L, temp = 0.2,
                        grammar = b$grammar, trigger_patterns = tp, trigger_tokens = tt)
  cat("RAW:", out, "\n")
  p <- tryCatch(llama_chat_parse(out, format = b$format, parser = b$parser),
                error = function(e) { cat("parse error:", conditionMessage(e), "\n"); NULL })
  if (!is.null(p)) {
    cat("content:", p$content, "\n")
    cat("tool_calls:", nrow(p$tool_calls), "\n")
    if (nrow(p$tool_calls) > 0) print(p$tool_calls)
  }
}
cat("\nDONE\n")
