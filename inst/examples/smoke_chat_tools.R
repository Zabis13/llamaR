#!/usr/bin/env Rscript
# Smoke test for the tool-aware chat layer (llama_chat_build / llama_chat_parse).
# Run after R CMD INSTALL:  Rscript tools/smoke_chat_tools.R
#
# Verifies the R<->C++ binding end to end:
#   1. build a tool-aware prompt + grammar from messages + one tool definition
#   2. generate constrained output through that grammar
#   3. parse the raw output back into content / tool_calls

library(llamaR)

MODEL <- "/mnt/Data2/DS_projects/llm_models/tiny-mistral-test-Q2_K.gguf"
if (!file.exists(MODEL)) stop("test model not found: ", MODEL)

model <- llama_load_model(MODEL, n_gpu_layers = 0L)
ctx   <- llama_new_context(model, n_ctx = 2048L)

messages <- list(
  list(role = "system", content = "You are a helpful assistant. Use tools when needed."),
  list(role = "user",   content = "What is the weather in Paris?")
)

tools <- list(
  list(
    type = "function",
    "function" = list(
      name = "get_weather",
      description = "Get the current weather for a city",
      parameters = list(
        type = "object",
        properties = list(
          city = list(type = "string", description = "City name")
        ),
        required = list("city")
      )
    )
  )
)

cat("== llama_chat_build ==\n")
built <- llama_chat_build(model, messages, tools = tools, tool_choice = "auto")
cat("format id   :", built$format, "\n")
cat("grammar_lazy:", built$grammar_lazy, "\n")
cat("grammar?    :", nzchar(built$grammar), "(", nchar(built$grammar), "chars )\n")
cat("stops       :", paste(built$additional_stops, collapse = ", "), "\n")
cat("--- prompt (first 600 chars) ---\n")
cat(substr(built$prompt, 1, 600), "\n...\n")

cat("\n== generate (grammar-constrained) ==\n")
# Lazy grammars only kick in after a trigger token; passing a non-lazy grammar
# here is the simple path. If grammar_lazy, fall back to unconstrained so the
# smoke test still exercises chat_parse.
g <- if (!built$grammar_lazy && nzchar(built$grammar)) built$grammar else NULL
out <- llama_generate(ctx, built$prompt, max_new_tokens = 200L, temp = 0.0,
                      grammar = g)
cat("--- raw output ---\n")
cat(out, "\n")

cat("\n== llama_chat_parse ==\n")
parsed <- llama_chat_parse(out, format = built$format, parser = built$parser)
cat("content          :", parsed$content, "\n")
cat("reasoning_content:", parsed$reasoning_content, "\n")
cat("tool_calls rows  :", nrow(parsed$tool_calls), "\n")
if (nrow(parsed$tool_calls) > 0) {
  print(parsed$tool_calls)
}

cat("\nOK: chat_build/chat_parse binding works end to end.\n")
