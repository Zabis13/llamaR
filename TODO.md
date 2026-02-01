## TODO

### R Interface — Core
- [x] `llama_load_model()` — загрузка GGUF модели
- [x] `llama_free_model()` — освобождение модели (+ автоматический GC-финализер)
- [x] `llama_new_context()` — создание контекста
- [x] `llama_free_context()` — освобождение контекста (+ автоматический GC-финализер)
- [x] `llama_tokenize()` — токенизация текста
- [x] `llama_detokenize()` — детокенизация
- [x] `llama_generate()` — генерация текста (полный цикл: tokenize → encode → decode loop → detokenize)
- [x] `llama_embeddings()` — получение эмбеддингов последнего токена
- [x] `llama_model_info()` — метаданные модели (n_ctx_train, n_embd, n_vocab, n_layer, n_head, desc)
- [x] `llama_supports_gpu()` — проверка доступности GPU
- [x] `llama_set_verbosity()` / `llama_get_verbosity()` — управление логированием

### R Interface — Chat Templates
- [x] `llama_chat_template()` — получить встроенный шаблон модели
- [x] `llama_chat_apply_template()` — применить шаблон к списку сообщений

### R Interface — LoRA Adapters
- [x] `llama_lora_load()` — загрузка LoRA адаптера
- [x] `llama_lora_apply()` — применение LoRA к контексту (с масштабом)
- [x] `llama_lora_remove()` — удаление конкретного LoRA
- [x] `llama_lora_clear()` — удаление всех LoRA

### Sampling
- [x] Temperature, top_k, top_p через параметры `llama_generate()`
- [x] Greedy decoding при `temp = 0`
- [ ] Exposed sampler chain API (`llama_sampler_*`) для Fine-grained control
- [ ] Repetition penalty, DRY, mirostat
- [ ] Grammar-constrained generation (JSON schema, etc.)

### Backend
- [x] CPU inference
- [x] GPU inference через Vulkan (ggmlR)
- [x] `n_gpu_layers` параметр для GPU offloading
- [ ] Явный выбор backend (CPU / Vulkan / auto)
- [ ] Multi-GPU split через ggmlR scheduler

### Documentation
- [x] Roxygen2 @export + @param для всех функций → man/*.Rd
- [x] @examples для всех функций
- [x] benchmark.R — скрипт сравнения CPU vs GPU
- [ ] Vignette: Quick start guide
- [ ] Vignette: Chat with Instruct models
- [ ] Vignette: Using LoRA adapters

### Testing
- [x] Unit tests: model load + info
- [x] Unit tests: context create/free
- [x] Unit tests: tokenize ↔ detokenize round-trip
- [x] Unit tests: generation (non-empty output)
- [x] Unit tests: greedy determinism
- [x] Unit tests: embeddings dimensionality
- [ ] Unit tests: chat template application
- [ ] Unit tests: LoRA loading and application
- [ ] Unit tests: GPU offloading (n_gpu_layers = -1)
- [ ] Edge cases: empty prompt, very long prompt, context overflow
- [ ] Stress test: repeated generate calls (memory leak check)

### Future
- [ ] State save/load (`llama_state_save_file`, `llama_state_load_file`)
- [ ] Streaming generation (token-by-token callback)
- [ ] Performance stats (`llama_perf_context`)
- [ ] Model metadata access (`llama_model_meta_*`)
- [ ] Vocab info (BOS, EOS, special tokens)
- [ ] Quantization API (`llama_model_quantize`)
- [ ] Batch inference

---

## Выбор модели

### Формат
llamaR работает только с моделями в формате **GGUF**. Модели можно скачать с:
- [Hugging Face](https://huggingface.co/models?library=gguf) — поиск по тегу `gguf`
- [TheBloke](https://huggingface.co/TheBloke) — квантованные версии популярных моделей

### Квантизация
Квантизация уменьшает размер модели и требования к памяти:

| Квант | Биты | Качество | Размер (7B) | Рекомендация |
|-------|------|----------|-------------|--------------|
| Q2_K  | 2    | Низкое   | ~2.5 GB     | Только тест  |
| Q4_K_M| 4    | Хорошее  | ~4.0 GB     | Оптимальный баланс |
| Q5_K_M| 5    | Отличное | ~5.0 GB     | Качество важнее |
| Q6_K  | 6    | Высокое  | ~5.5 GB     | Почти без потерь |
| Q8_0  | 8    | Максимум | ~7.0 GB     | Максимальное качество |
| F16   | 16   | Эталон   | ~14 GB      | Требует много RAM |

### Размеры моделей
| Параметры | RAM (Q4) | GPU VRAM (Q4) | Примеры |
|-----------|----------|---------------|---------|
| 1B-3B     | 2-4 GB   | 2-4 GB        | Llama 3.2 1B, Qwen2 1.5B |
| 7B-8B     | 4-6 GB   | 6-8 GB        | Llama 3.1 8B, Mistral 7B |
| 13B       | 8-10 GB  | 10-12 GB      | Llama 2 13B |
| 70B       | 40+ GB   | 48+ GB        | Llama 3.1 70B (нужен split) |

### Пример использования с Instruct моделью

```r
library(llamaR)

# Загрузить модель на GPU
model <- llama_load_model("llama-3.2-1B-Instruct.Q4_K_M.gguf", n_gpu_layers = -1L)
ctx <- llama_new_context(model, n_ctx = 4096L)

# Получить chat template модели
tmpl <- llama_chat_template(model)

# Сформировать диалог
messages <- list(
  list(role = "system", content = "Ты полезный ассистент."),
  list(role = "user", content = "Что такое R?")
)

# Применить шаблон
prompt <- llama_chat_apply_template(messages, template = tmpl)

# Генерировать ответ
response <- llama_generate(ctx, prompt, max_new_tokens = 200L)
cat(response)
```

### Пример с LoRA адаптером

```r
library(llamaR)

# Загрузить базовую модель
model <- llama_load_model("base-model.gguf", n_gpu_layers = -1L)

# Загрузить LoRA адаптер
lora <- llama_lora_load(model, "fine-tuned-adapter.gguf")

# Создать контекст и применить LoRA
ctx <- llama_new_context(model)
llama_lora_apply(ctx, lora, scale = 1.0)  # scale: 0.0-1.0+

# Генерация теперь использует LoRA-модифицированную модель
result <- llama_generate(ctx, "Hello!")

# Можно убрать LoRA и вернуться к базовой модели
llama_lora_clear(ctx)
```

### Рекомендации
1. **Начните с Q4_K_M** — лучший баланс качества и размера
2. **Используйте GPU** (`n_gpu_layers = -1L`) — в 10-40 раз быстрее CPU
3. **Instruct-модели** лучше для диалогов (суффикс `-Instruct`, `-Chat`)
4. **Используйте chat template** — без него Instruct модели работают плохо
5. **Проверьте VRAM** — модель должна помещаться в память GPU
