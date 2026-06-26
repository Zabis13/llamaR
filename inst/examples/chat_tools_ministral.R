suppressMessages(library(llamaR))

MODEL <- "/mnt/Data2/DS_projects/llm_models/Ministral-3-3B-Instruct-2512-Q4_K_M.gguf"
model <- llama_load_model(MODEL, n_gpu_layers = -1L)
ctx   <- llama_new_context(model, n_ctx = 4096L)

messages <- list(
  list(role = "system", content = "You are a helpful assistant. Use the available tools to answer."),
  list(role = "user",   content = "What's the weather in Paris and Tokyo right now?")
)

tools <- list(
  list(type = "function", "function" = list(
    name = "get_weather",
    description = "Get the current weather for a given city",
    parameters = list(
      type = "object",
      properties = list(
        city = list(type = "string", description = "City name, e.g. Paris"),
        unit = list(type = "string", enum = list("celsius", "fahrenheit"))
      ),
      required = list("city")
    )
  ))
)

cat("== chat_build ==\n")
b <- llama_chat_build(model, messages, tools = tools, tool_choice = "auto")
cat("format id:", b$format, " grammar_lazy:", b$grammar_lazy,
    " grammar chars:", nchar(b$grammar), "\n")

cat("\n== generate (grammar-constrained, lazy triggers) ==\n")
# Always pass the grammar; for lazy formats also pass the triggers so the
# grammar only kicks in after the trigger token/word (e.g. [TOOL_CALLS]).
out <- llama_generate(ctx, b$prompt, max_new_tokens = 300L, temp = 0.3, top_p = 0.9,
                      grammar = b$grammar,
                      trigger_patterns = b$trigger_patterns,
                      trigger_tokens = b$trigger_tokens)
cat(out, "\n")

cat("\n== chat_parse ==\n")
p <- llama_chat_parse(out, format = b$format, parser = b$parser)
cat("content:", p$content, "\n")
cat("tool_calls:", nrow(p$tool_calls), "\n")
if (nrow(p$tool_calls) > 0) print(p$tool_calls)
cat("\nDONE\n")
