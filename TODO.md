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

### R Interface — Hugging Face
- [x] `llama_hf_cache_dir()` — путь к кэшу
- [x] `llama_hf_list()` — список GGUF файлов в HF-репозитории
- [x] `llama_hf_download()` — скачивание моделей с HF
- [x] `llama_load_model_hf()` — загрузка модели напрямую из HF
- [x] `llama_hf_cache_info()` — информация о кэше
- [x] `llama_hf_cache_clear()` — очистка кэша

### Sampling
- [x] Temperature, top_k, top_p через параметры `llama_generate()`
- [x] Greedy decoding при `temp = 0`
- [x] min_p sampling (`llama_sampler_init_min_p`)
- [x] Repetition penalty (`llama_sampler_init_penalties`)
- [x] Mirostat v1/v2 (`llama_sampler_init_mirostat`, `_v2`)
- [x] Typical sampling (`llama_sampler_init_typical`)
- [x] Frequency / presence penalty через `llama_generate()`
- [x] Grammar-constrained generation — GBNF (`llama_sampler_init_grammar`)
- [ ] DRY sampler (`llama_sampler_init_dry`)
- [ ] Dynamic temperature (`llama_sampler_init_temp_ext`)
- [ ] XTC sampler (`llama_sampler_init_xtc`)
- [ ] Top-n sigma (`llama_sampler_init_top_n_sigma`)
- [ ] Logit bias (`llama_sampler_init_logit_bias`)
- [ ] Infill / fill-in-the-middle (`llama_sampler_init_infill`)
- [ ] Exposed sampler chain API (`llama_sampler_*`) для fine-grained control

### State Management
- [x] `llama_state_save()` / `llama_state_load()` — сохранение/загрузка состояния контекста
- [ ] State get/set data (`llama_state_get_data`, `llama_state_set_data`)
- [ ] Per-sequence state (`llama_state_seq_*`)

### Memory / KV Cache Control
- [x] `llama_memory_clear()` — очистка KV cache
- [x] `llama_memory_seq_rm()` — удаление токенов по позиции
- [x] `llama_memory_seq_cp()` — копирование последовательности
- [x] `llama_memory_seq_keep()` — оставить только указанную последовательность
- [x] `llama_memory_seq_add()` — сдвиг позиций
- [x] `llama_memory_seq_pos_range()` — границы позиций (min + max)
- [x] `llama_memory_can_shift()` — проверка поддержки сдвига

### Logits & Embeddings Output
- [x] `llama_get_logits()` — сырые логиты после decode
- [ ] `llama_get_embeddings` / `_ith` / `_seq` — эмбеддинги по позициям

### Model Metadata (расширенное)
- [x] `llama_model_info()` расширен: size, n_params, has_encoder, has_decoder, is_recurrent
- [x] `llama_model_meta()` — все метаданные как named character vector
- [x] `llama_model_meta_val()` — чтение метаданных по ключу

### Vocabulary
- [x] `llama_vocab_info()` — все специальные токены (bos/eos/eot/sep/nl/pad/fim_*)
- [x] `llama_chat_builtin_templates()` — список встроенных шаблонов
- [ ] `llama_vocab_get_text` / `_score` / `_attr` — свойства токенов
- [ ] `llama_vocab_is_eog` / `_control` — проверки типа токена

### Context Configuration
- [x] `llama_set_threads()` — изменение числа потоков
- [x] `llama_set_causal_attn()` — каузальное / свободное внимание
- [x] `llama_n_ctx()` — текущий размер контекста

### Backend
- [x] CPU inference
- [x] GPU inference через Vulkan (ggmlR)
- [x] `n_gpu_layers` параметр для GPU offloading
- [ ] Явный выбор backend (CPU / Vulkan / auto)
- [ ] Multi-GPU split через ggmlR scheduler

### Performance & Debug
- [x] `llama_perf()` — счётчики производительности
- [x] `llama_perf_reset()` — сброс счётчиков
- [x] `llama_system_info()` — системная информация
- [ ] `llama_memory_breakdown_print` — разбивка памяти
- [ ] Streaming generation (token-by-token callback)

### Hardware / System
- [x] `llama_supports_mmap()` / `llama_supports_mlock()`
- [x] `llama_max_devices()`
- [ ] `llama_supports_rpc`
- [ ] `llama_parallel_sequences`

### Quantization & Training
- [ ] `llama_model_quantize` — квантизация модели на диск
- [ ] `llama_model_save_to_file` — сохранение модели
- [ ] `llama_opt_init` / `_epoch` — fine-tuning

### Documentation
- [x] Roxygen2 @export + @param для всех функций → man/*.Rd
- [x] @examples для всех функций
- [x] @return / \value с описанием класса и структуры для всех функций
- [x] benchmark.R — скрипт сравнения CPU vs GPU

### Testing
- [x] Unit tests: model load + info
- [x] Unit tests: context create/free
- [x] Unit tests: tokenize ↔ detokenize round-trip
- [x] Unit tests: generation (non-empty output)
- [x] Unit tests: greedy determinism
- [x] Unit tests: embeddings dimensionality
- [x] Unit tests: chat template application
- [x] Unit tests: LoRA loading and application
- [x] Unit tests: extended model info (size, n_params, encoder/decoder/recurrent)
- [x] Unit tests: model metadata (llama_model_meta, llama_model_meta_val)
- [x] Unit tests: vocabulary info (llama_vocab_info)
- [x] Unit tests: context config (n_ctx, set_threads, set_causal_attn)
- [x] Unit tests: KV cache operations (clear, seq_rm, seq_keep, seq_pos_range, can_shift)
- [x] Unit tests: state save/load
- [x] Unit tests: logits
- [x] Unit tests: performance counters
- [x] Unit tests: hardware/system info (no model)
- [x] Unit tests: chat builtin templates (no model)
- [x] Unit tests: advanced sampling parameters (min_p, repeat_penalty, mirostat)
- [ ] Unit tests: GPU offloading (n_gpu_layers = -1)
- [ ] Unit tests: HF download/list
- [ ] Edge cases: empty prompt, very long prompt, context overflow
- [ ] Stress test: repeated generate calls (memory leak check)

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

