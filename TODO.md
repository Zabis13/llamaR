## TODO


### Concurrency / Production serving (roadmap)

**Позиционирование.** llamaR + ggmlR — нативный GPU-инференс прямо в R-процессе:
без Python, без JVM, без HTTP к внешнему серверу. На RX 9070 — ~22–24 tok/sec на
14B Q8 (конкурентный результат, не игрушка). HTTP/SSE отдаёт **drogonR** (C++ async
сервер на Drogon — один из самых быстрых в бенчмарках), не httpuv/plumber.

**Где реальное ограничение.** Не в производительности инференса, а в том, что R —
single-threaded event loop. Пока `llama_gen_next()` считает на GPU, R-поток занят, и
второй запрос ждёт. `llama_serve_openai()` поэтому single-sequence: один
пользователь/агент за раз (см. docstring в `R/serve.R`).

**Вывод.** Параллелизм решается **архитектурой (worker pool из процессов), а не
многопоточностью внутри одного R-процесса.** Это реалистично и архитектурно чисто:
llamaR может быть production inference-сервером для R-экосистемы. Три фазы:

#### Фаза 1 — Worker pool (MVP)
- [ ] N R-процессов (`mirai`-воркеры), каждый держит свою модель/ctx
      на GPU; модель грузится один раз на воркер. `callr` как fallback.
      `mirai` предпочтителен: async-native, интегрируется с later/event loop
      без блокировки, persistent-воркеры со состоянием (модель живёт в воркере);
      `callr` синхронен (wait), `parallel` нативно persistent-воркеров с
      загруженной моделью не держит.
- [ ] drogonR-фронт принимает запрос → диспетчер выбирает свободного воркера →
      отдаёт ответ (стрим прокидывается обратно через SSE).
- [ ] Без внешних зависимостей кроме process-бэкенда (`callr`/`mirai`).
- [ ] Решить: сколько воркеров на один GPU помещается по VRAM (модель × N).
- [ ] Backpressure: что делать, когда все воркеры заняты (очередь in-memory vs 503).

Модель практической работы: каждый воркер держит свою копию модели (на своём GPU,
либо несколько воркеров на одном GPU, если VRAM позволяет); drogonR-фронт раздаёт
входящие по свободным воркерам. Каждый клиент получает свой стрим и не ждёт других:
клиент 1 и клиент 2 отвечаются одновременно, а не по очереди. Пропускная способность
складывается по воркерам — напр. 4 воркера × ~24 tok/sec ≈ ~96 tok/sec суммарно по
всем клиентам.

Ограничение: это **не** continuous batching как в vLLM — один воркер обслуживает один
запрос за раз (нет батчинга нескольких последовательностей в одном forward). Масштаб
здесь экстенсивный: пропускная способность растёт **горизонтально** — больше воркеров /
больше GPU (Фазы 2–3), а не за счёт батчинга в одном процессе. Высокий RPS (вплоть до
1000+) при таком подходе достижим — на коротких запросах и достаточном числе воркеров.
Целевой профиль R-экосистемы (агенты, RAG-пайплайны, IDE-ассистенты) — параллельные
сессии. По сути — локальный аналог Ollama, но нативно в R, с прямым доступом к llamaR
API (без HTTP-хопа внутри процесса).

#### Фаза 2 — Очередь / брокер (масштаб)
- [ ] drogonR принимает → внешняя очередь (Redis/брокер) → worker отвечает async.
- [ ] Переживает рестарт фронта; воркеры можно поднимать/гасить независимо.
- [ ] Тянет внешний брокер — оправдано только при росте нагрузки за пределы in-memory.

#### Фаза 3 — Multi-GPU + балансировщик
- [ ] Каждый GPU = отдельный процесс (надстройка над worker pool, не альтернатива).
- [ ] Load balancer перед процессами; маршрутизация по загрузке/VRAM.
- [ ] Опционально: разные модели на разных GPU за одним фронтом.

Открытые вопросы: разделяемый KV-кэш между воркерами (вряд ли — каждый свой);
прокидывание стрима из воркера во фронт без блокировки event loop фронта;
graceful drain воркера при рестарте.


### OpenAI-совместимый сервер (подключение OpenCode и др. агентов)

Цель: OpenCode (и любой OpenAI-совместимый клиент) подключается к локальной
модели. Архитектура — вариант B: HTTP и SSE отдаёт **drogonR**, генерацию —
**llamaR**, всё через R-API. Формат запроса/ответа копируется 1-в-1 с
эталонного `llama-server` (`llama.cpp/tools/server/server-task.cpp`).
НЕ встраиваем сам llama-server (он тащит cpp-httplib + common/ + mtmd + curl
и обесценивает drogonR).

#### Этап 1 — стрим-генерация в llamaR (token-by-token) ✅ ГОТОВО
- [x] C: `r_llama_gen_begin` — токенизация, sampler-цепочка (1-в-1 с
      `r_llama_generate`), KV clear, prefill. Возвращает `externalptr`
      `llama_gen_state` с GC-финализатором (`gen_state_finalizer`).
- [x] C: `r_llama_gen_next` — sample → EOG/лимит ⇒ `NULL`; иначе accept,
      detokenize в UTF-8-буфер, decode, отдать валидный UTF-8 префикс
      (хвост недописанного символа держится в буфере; `utf8_incomplete_tail`).
- [x] C: `r_llama_gen_end` — дослить остаток буфера, пометить done; sampler
      освобождается финализатором.
- [x] Зарегистрированы в CallEntries.
- [x] R: `llama_gen_begin()` / `llama_gen_next()` / `llama_gen_end()`, `@export`.
- [x] `devtools::document()`.
- [x] Тест на tiny-mistral: склейка кусков + gen_end == `llama_generate()`
      (greedy) — совпадает бит-в-бит, включая многобайтовый UTF-8.

#### Этап 2 — OpenAI-роуты на drogonR ✅ ГОТОВО
Реализовано в `R/serve.R` → `llama_serve_openai(model_path, port=11434, n_ctx, ...)`.
`drogonR` + `later` в `Suggests`, guard `requireNamespace` в начале функции.
- [x] `GET /v1/models` — `{object:"list", data:[{id, object:"model", ...}]}`.
- [x] `POST /v1/chat/completions`:
  - [x] склейка `messages` через `llama_chat_apply_template()`
        (data.frame от jsonlite нормализуется в list(role,content)).
  - [x] non-stream: `chat.completion` + `message` + `usage` + `finish_reason`.
  - [x] stream: роль-чанк → `delta.content`-чанки → финальный `delta:{}` +
        `finish_reason` (`null` в промежуточных) → `data: [DONE]`.
        Через `dr_stream_sse()` + цикл `llama_gen_next()`.
- [x] Модель грузится один раз при старте; держится в замыкании сервера.
- [x] Блокирующий цикл `later::run_now()` + `on.exit(dr_stop())` после
      `dr_serve()` (тот возвращается сразу; без цикла процесс выходит и
      Drogon-деструктор делает abort).
- [x] Тесты: `tests/testthat/test-serve.R` — быстрые юнит (форма JSON,
      finish_reason null/{}, guard'ы). End-to-end вынесен в
      `inst/examples/serve_openai.R --selftest`.
- [x] Пример `inst/examples/serve_openai.R` (аргументы: model, [port], [n_ctx]).
- [x] `opencode.json` в корне проекта (provider @ai-sdk/openai-compatible).
- [x] Подключён OpenCode, работает end-to-end (Ministral-3B Q4 на 11434).
- Решение по открытому вопросу: выбрана функция `llama_serve_openai()`
      (drogonR в Suggests), не отдельный скрипт.

#### Багфиксы по ходу Этапа 2
- [x] Chunked prefill: `llama_decode` бьёт промпт на куски по `llama_n_batch()`
      в `r_llama_gen_begin` и `r_llama_generate`. Раньше длинный системный
      промпт OpenCode ронял процесс через
      `GGML_ASSERT(n_tokens_all <= cparams.n_batch)`.
- [x] Проверка переполнения контекста в `/v1/chat/completions`: если
      `prompt_tokens + max_tokens > n_ctx` — HTTP 400 `context_length_exceeded`.
      Раньше промпт у предела n_ctx переполнял KV → первый sample = EOG →
      молчаливый `content:""` (симптом «пустой ответ после первого запроса
      в OpenCode»). Теперь клиент видит явную ошибку и обрезает историю.

#### Вне объёма (для базового OpenCode не нужно)
- [ ] tool_calls / function calling (потом, через GBNF-грамматику)
- [ ] `/v1/completions` (legacy), `/v1/embeddings`, `/v1/responses`
- [ ] API-key middleware, мультимодальность (mtmd)
- [ ] Фильтрация спец-токенов в выводе (Qwen3 просачивает `<|im_end|>` в текст;
      llama-server их чистит) — заметно на reasoning-моделях.


### R Interface — ragnar / ellmer Integration
- [x] `embed_llamar()` — embedding-провайдер для `ragnar_store_create(embed = ...)`
- [x] `llama_embed_batch()` — батчевый embed списка строк
- [~] `chat_llamar()` — ellmer::Chat-совместимый объект. ЧАСТИЧНО (`R/chat.R`).
      Подход B2 (HTTP через loopback, не S7-провайдер): `chat_llamar(base_url=)`
      подключается к запущенному серверу, `chat_llamar(model_path=)` поднимает
      `llama_serve_openai` фоном (`callr::r_bg`) и привязывает его жизнь к chat
      через finalizer (+ `chat_llamar_stop()`). Внутри — `ellmer::chat_openai_compatible`
      (`/v1/chat/completions`, не `/v1/responses`). `ellmer`/`callr` в Suggests.
      Тесты + e2e на tiny-mistral. ОСТАЁТСЯ для `ragnar_register_tool_retrieve()`:
      tool-calling — сервер пока не emit'ит `tool_calls` (см. «Вне объёма» Этапа 2),
      поэтому retrieve как tool ещё не вызывается моделью.

### R Interface — ellmer Integration (S7 Provider)

> Примечание: для `chat_llamar()` выбран подход B2 (HTTP через loopback +
> `chat_openai_compatible`), а не нативный S7-провайдер. Блок ниже —
> альтернатива на случай, если loopback-хоп окажется узким местом; пока
> отложен, нативный провайдер не требуется.

#### Core Provider
- [ ] `ProviderLlamaR` — S7 класс, наследует от `ellmer::Provider` (отложено, см. B2)
- [x] `chat_llamar()` — конструктор, возвращает `ellmer::Chat` (через B2, см. выше)
- [ ] `ellmer::chat_perform()` generic — не нужен в B2 (транспорт у ellmer/httr2)
- [ ] `ellmer::stream_perform()` generic — не нужен в B2 (стрим у сервера через SSE)

#### Turns / Messages
- [ ] `ellmer::value_turn()` generic — конвертация ответа модели в `Turn`
- [ ] `ellmer::as_turn()` — конвертация `llama_generate()` output в `ellmer::Turn`
- [ ] Маппинг `list(role, content)` → `llama_chat_apply_template()` формат

#### Tools
- [ ] `ellmer::tool_call()` generic — вызов инструментов
- [ ] `ellmer::as_tool()` generic — регистрация R-функции как tool
- [ ] GBNF grammar для structured tool calls (через `llama_sampler_init_grammar()`)

#### Structured Output
- [ ] `ellmer::structured_output()` generic
- [ ] JSON schema → GBNF конвертер (через `llama_sampler_init_grammar()`)

#### System / Config
- [ ] `ellmer::system_prompt()` generic — getter/setter
- [ ] `ellmer::tokens_used()` generic — через `llama_perf()`
- [ ] `ellmer::model_info()` generic — через `llama_model_info()`

#### Зависимости внутри llamaR (должны быть готовы раньше)
- [ ] Streaming generation (`llama_generate_stream()`)
- [x] `llama_sampler_init_grammar()` (уже есть)
- [x] `llama_perf()` (уже есть)

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
- [ ] Adaptive-p sampler (`llama_sampler_init_adaptive_p`)
- [ ] Grammar with lazy pattern triggers (`llama_sampler_init_grammar_lazy_patterns`)
- [ ] Exposed sampler chain API (`llama_sampler_*`) для fine-grained control:
  - [ ] `llama_sampler_init()` / `llama_sampler_free()`
  - [ ] `llama_sampler_name()` / `llama_sampler_reset()` / `llama_sampler_clone()`
  - [ ] `llama_sampler_chain_get()` / `llama_sampler_chain_n()` / `llama_sampler_chain_remove()`
  - [ ] `llama_sampler_get_seed()`

### State Management
- [x] `llama_state_save()` / `llama_state_load()` — сохранение/загрузка состояния контекста
- [x] `llama_state_get_size()` — размер сериализованного состояния в байтах
- [ ] `llama_state_get_data()` / `llama_state_set_data()` — raw state bytes
- [ ] Per-sequence state:
  - [ ] `llama_state_seq_get_size()` / `llama_state_seq_get_data()` / `llama_state_seq_set_data()`
  - [ ] `llama_state_seq_save_file()` / `llama_state_seq_load_file()`
  - [ ] `llama_state_seq_get_size_ext()` / `llama_state_seq_get_data_ext()` / `llama_state_seq_set_data_ext()` (SWA/recurrent)





### Model Metadata (расширенное)
- [x] `llama_model_info()` расширен: size, n_params, has_encoder, has_decoder, is_recurrent
- [x] `llama_model_meta()` — все метаданные как named character vector
- [x] `llama_model_meta_val()` — чтение метаданных по ключу
- [ ] `llama_model_n_embd_inp()` / `llama_model_n_embd_out()` — размеры входных/выходных эмбеддингов
- [x] `llama_model_n_head_kv()` — количество KV головок (добавлен в `llama_model_info()`)
- [ ] `llama_model_n_swa()` — размер sliding window attention
- [ ] `llama_model_rope_type()` / `llama_model_rope_freq_scale_train()` — параметры RoPE
- [ ] `llama_model_n_cls_out()` / `llama_model_cls_label()` — для моделей-классификаторов
- [ ] `llama_model_decoder_start_token()` — для encoder-decoder моделей
- [ ] `llama_model_is_hybrid()` / `llama_model_is_diffusion()` — тип архитектуры
- [ ] `llama_model_meta_key_str()` — ключ метаданных по enum
- [ ] `llama_flash_attn_type_name()` — название типа Flash Attention

### Vocabulary
- [x] `llama_vocab_info()` — все специальные токены (bos/eos/eot/sep/nl/pad/fim_*)
- [x] `llama_chat_builtin_templates()` — список встроенных шаблонов
- [x] `llama_vocab_type()` — тип словаря (BPE, SPM, WPM и т.д.)
- [x] `llama_vocab_get_text()` / `llama_vocab_get_score()` — свойства токена по id
- [ ] `llama_vocab_get_attr()` — атрибуты токена
- [x] `llama_vocab_is_eog()` / `llama_vocab_is_control()` — проверки типа токена
- [ ] `llama_vocab_get_add_bos()` / `llama_vocab_get_add_eos()` / `llama_vocab_get_add_sep()` — флаги добавления спец. токенов
- [ ] `llama_vocab_mask()` / `llama_vocab_fim_pad()` — токены маски и FIM padding
- [x] `llama_token_to_piece()` — преобразование одного токена в строку




### Performance & Debug
- [x] `llama_perf()` — счётчики производительности
- [x] `llama_perf_reset()` — сброс счётчиков
- [x] `llama_system_info()` — системная информация
- [x] `llama_memory_breakdown_print()` — разбивка памяти по устройствам
- [x] `llama_perf_context_print()` — вывод метрик контекста (`llama_perf_print()`)
- [ ] `llama_perf_sampler()` / `llama_perf_sampler_print()` / `llama_perf_sampler_reset()` — метрики сэмплера
- [ ] Streaming generation (token-by-token callback)

### Hardware / System
- [x] `llama_supports_mmap()` / `llama_supports_mlock()`
- [x] `llama_max_devices()`
- [x] `llama_supports_rpc()` — поддержка RPC
- [ ] `llama_max_parallel_sequences()` — максимум параллельных последовательностей
- [ ] `llama_max_tensor_buft_overrides()` — максимум переопределений буферов
- [ ] `llama_params_fit()` — подгонка параметров к доступной памяти

### Model File Operations
- [ ] `llama_model_load_from_splits()` — загрузка модели из нескольких GGUF частей
- [ ] `llama_model_save_to_file()` — сохранение модели в файл
- [ ] `llama_model_quantize()` — квантизация модели на диск
- [ ] `llama_split_path()` / `llama_split_prefix()` — работа с путями разделённых файлов

### LoRA Adapters (расширенное)
- [ ] `llama_adapter_meta_count()` / `llama_adapter_meta_val_str()` / `llama_adapter_meta_key_by_index()` — метаданные адаптера
- [ ] `llama_apply_adapter_cvec()` — применить control vector
- [ ] `llama_adapter_get_alora_n_invocation_tokens()` / `llama_adapter_get_alora_invocation_tokens()` — ALora

### Training / Fine-tuning
- [ ] `llama_opt_init()` / `llama_opt_epoch()` — fine-tuning
- [ ] `llama_opt_param_filter_all()` — фильтр параметров для обучения
