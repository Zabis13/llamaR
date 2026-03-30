## TODO


### R Interface — ragnar / ellmer Integration
- [x] `embed_llamar()` — embedding-провайдер для `ragnar_store_create(embed = ...)`
- [x] `llama_embed_batch()` — батчевый embed списка строк
- [ ] `chat_llamar()` — ellmer::Chat-совместимый объект для `ragnar_register_tool_retrieve()` (см. ниже)

### R Interface — ellmer Integration (S7 Provider)

#### Core Provider
- [ ] `ProviderLlamaR` — S7 класс, наследует от `ellmer::Provider`
- [ ] `chat_llamar()` — конструктор, возвращает `ellmer::Chat`
- [ ] `ellmer::chat_perform()` generic — синхронная генерация
- [ ] `ellmer::stream_perform()` generic — стриминг (требует streaming generation в llamaR)

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
