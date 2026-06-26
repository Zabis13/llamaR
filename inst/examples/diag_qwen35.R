#!/usr/bin/env Rscript
# Diagnostic: raw Qwen3.5 generation WITHOUT the serve/chat-parse layer.
# Goal: see what the model actually emits (incl. how it wraps <think>),
# to decide whether the empty-content bug is in the arch or in serve/chat.
suppressMessages(library(llamaR))

MODEL <- "/mnt/Data2/DS_projects/llm_models/Qwen3.5-9B-UD-Q6_K_XL.gguf"

cat("== loading model ==\n")
model <- llama_load_model(MODEL, n_gpu_layers = 99L)
ctx   <- llama_new_context(model, n_ctx = 4096L)

# Apply the model's built-in chat template so the prompt is in Qwen3.5 format.
msgs <- list(list(role = "user", content = "Привет! Расскажи коротко, что ты умеешь."))
prompt <- llama_chat_apply_template(msgs, add_generation_prompt = TRUE)
cat("== templated prompt ==\n")
cat(prompt, "\n")
cat("== [end prompt] ==\n\n")

cat("== generating (greedy, temp=0) ==\n")
out <- llama_generate(ctx, prompt, max_new_tokens = 200L, temp = 0)
cat("== RAW OUTPUT ==\n")
cat(out, "\n")
cat("== [end output] ==\n")
cat("nchar:", nchar(out), "\n\n")

# ---------------------------------------------------------------------------
# Tool-call diagnostic: which chat FORMAT does Qwen3.5 resolve to with tools?
# format 1 == GENERIC (the bad fallback). Anything else (e.g. Hermes 2 Pro)
# means tool calls are <tool_call>{json}</tool_call> and the parser handles them.
# ---------------------------------------------------------------------------
cat("== chat_build WITH a tool ==\n")
tools <- list(list(
  type = "function",
  "function" = list(
    name = "get_weather",
    description = "Get the current weather for a city",
    parameters = list(
      type = "object",
      properties = list(city = list(type = "string", description = "City name")),
      required = list("city")
    )
  )
))
tmsgs <- list(list(role = "user", content = "What's the weather in Paris?"))
built <- llama_chat_build(model, tmsgs, tools = tools, enable_thinking = FALSE)
cat("format id     :", built$format, "  (1 = GENERIC/fallback, else native)\n")
cat("grammar_lazy  :", built$grammar_lazy, "\n")
cat("trigger_pats  :", paste(built$trigger_patterns, collapse = " | "), "\n")
cat("grammar (head):\n"); cat(substr(built$grammar, 1, 400), "\n")

cat("\n== parse a synthetic Hermes-style tool call ==\n")
synthetic <- "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>"
parsed <- llama_chat_parse(synthetic, format = built$format)
cat("content    :", parsed$content, "\n")
cat("tool_calls :\n"); print(parsed$tool_calls)

# ---------------------------------------------------------------------------
# Multi-turn WITH tool_result: this is the SECOND Claude Code request, where
# the history carries the assistant's tool_call AND the tool's result. This is
# the round that fails ("empty or malformed response"). Build the prompt and
# generate to see if (a) the prompt is well-formed and (b) the model answers.
# ---------------------------------------------------------------------------
cat("\n== chat_build multi-turn (user -> assistant tool_call -> tool result) ==\n")
mt <- list(
  list(role = "user", content = "What's the weather in Paris?"),
  list(role = "assistant", content = "",
       tool_calls = list(list(
         id = "call_1", type = "function",
         "function" = list(name = "get_weather",
                           arguments = jsonlite::toJSON(list(city = "Paris"), auto_unbox = TRUE))))),
  list(role = "tool", content = "18°C, sunny", tool_call_id = "call_1")
)
built2 <- tryCatch(
  llama_chat_build(model, mt, tools = tools, enable_thinking = FALSE),
  error = function(e) { cat("!! chat_build ERROR:", conditionMessage(e), "\n"); NULL })
if (!is.null(built2)) {
  cat("format id :", built2$format, "\n")
  cat("-- prompt (last 600 chars) --\n")
  cat(substr(built2$prompt, max(1, nchar(built2$prompt) - 600), nchar(built2$prompt)), "\n")
  cat("-- [end prompt] --\n")
  out2 <- llama_generate(ctx, built2$prompt, max_new_tokens = 120L, temp = 0)
  cat("-- RAW gen --\n"); cat(out2, "\n")
  p2 <- llama_chat_parse(out2, format = built2$format, parser = built2$parser)
  cat("parsed content:", p2$content, "\n")
  cat("parsed tool_calls rows:", nrow(p2$tool_calls), "\n")
}
