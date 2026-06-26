# Диагностический пример: многоходовый диалог с активной грамматикой (tools),
# воспроизводящий серверный путь llama_serve_anthropic:
#   llama_chat_build(messages, tools) -> grammar -> llama_gen_begin/next/end (стрим).
# Каждый ход несёт растущую историю на одном ctx (как Claude Code). Печатает
# validUTF8 и байты каждого ответа — проверка, что многоходовая стрим-генерация
# с грамматикой не теряет и не искажает токены.
#
# Запуск:  Rscript inst/examples/dialog_grammar_test.R
devtools::load_all("/mnt/Data2/DS_projects/llamaR", quiet = TRUE)

model <- llama_load_model("/mnt/Data2/DS_projects/llm_models/deepseek-r1-distill-qwen-14b-q4_k_m.gguf", n_gpu_layers=-1L)
ctx   <- llama_new_context(model, n_ctx=8192L)

# Набор tools (как шлёт Claude Code) — заставляет llama_chat_build выдать грамматику.
tools <- list(
  list(type="function", "function"=list(
    name="get_weather", description="Узнать погоду в городе",
    parameters=list(type="object",
      properties=list(city=list(type="string", description="Город")),
      required=list("city")))),
  list(type="function", "function"=list(
    name="search", description="Поиск информации",
    parameters=list(type="object",
      properties=list(query=list(type="string", description="Запрос")),
      required=list("query"))))
)

# messages — растущая история в OpenAI-форме (как .anthropic_to_messages)
messages <- list(list(role="system",
  content="Ты — полезный ассистент. Отвечай по-русски, целыми словами, развёрнуто. Вызывай инструмент только когда явно нужно."))

turn <- function(user_text, n) {
  messages[[length(messages)+1]] <<- list(role="user", content=user_text)
  built <- llama_chat_build(model, messages, tools=tools, tool_choice="auto")
  tp <- if (isTRUE(built$grammar_lazy)) built$trigger_patterns else NULL
  tt <- if (isTRUE(built$grammar_lazy)) built$trigger_tokens else NULL
  ntok <- length(llama_tokenize(ctx, built$prompt, parse_special=TRUE))

  st <- llama_gen_begin(ctx, built$prompt, max_new_tokens=150L, temp=0.7, top_p=0.9, seed=42L,
                        grammar=if (nzchar(built$grammar)) built$grammar else NULL,
                        trigger_patterns=tp, trigger_tokens=tt)
  ch <- character(0); repeat { x <- llama_gen_next(st); if (is.null(x)) break; ch <- c(ch, x) }
  ch <- c(ch, llama_gen_end(st))
  r <- paste0(ch, collapse="")
  cat(sprintf("\n===== ХОД %d  [промпт=%d ток, grammar=%s lazy=%s]  validUTF8=%s bytes=%d =====\n",
              n, ntok, nzchar(built$grammar), isTRUE(built$grammar_lazy),
              validUTF8(r), nchar(r,type="bytes")))
  cat(r, "\n")
  messages[[length(messages)+1]] <<- list(role="assistant", content=r)
  r
}

turn("Привет! Расскажи коротко, кто ты.", 1)
turn("Расскажи подробно про инструменты, целыми словами, развёрнуто.", 2)
turn("Перечисли пять языков программирования и кратко опиши каждый.", 3)
