## TODO

### ✅ VISION ЧЕРЕЗ CLAUDE CODE + ФИКСЫ (сессия 2026-06-29, в 0.2.5)
Двухмодельный vision-сервер живой через Claude Code. Обновлены NEWS/README/TODO.
- [x] **Segfault `fflush((FILE*)1)` в фоновом потоке логгера.** chat_build на
      Qwen3.5+tools ронял serve_anthropic (Claude Code "ConnectionRefused").
      Причина: `r_llama_compat.h` редиректит stdout/stderr→(FILE*)1 и оборачивает
      fprintf/fputs/printf, но НЕ fflush; log.cpp worker (фоновый поток) делал
      fflush(sentinel)→SIGSEGV (асинхронно→ASan не ловил). Фикс: обёртка fflush в
      r_llama_compat.h (no-op для sentinel). Проверены все force-include файлы —
      других необёрнутых stdio-вызовов с sentinel нет.
- [x] **`llama_gen_begin_at()`** (C `r_llama_gen_begin_at` + R + NAMESPACE):
      генерация с УЖЕ заполненного KV (без clear/prefill) — для продолжения после
      `llama_image_eval`. gen_begin чистит KV и стёр бы image-эмбеддинги.
- [x] **Vision напрямую (R-API) РАБОТАЕТ** на Qwen2-VL-2B+mmproj: красный круг →
      "red circle". Пайплайн mtmd_load→image_load→image_eval→gen_begin_at→gen_next.
- [x] **Vision через Claude Code (двухмодельный serve_anthropic), caption-then-reason.**
      `llama_serve_anthropic(model_path, vision_model_path=, mmproj_path=, vision_n_ctx=8192L,
      vision_debug=FALSE)`: грузит ОБЕ модели. Картинка (Anthropic base64 image-блок) →
      Qwen2-VL описывает (фокус на вопросе юзера, `.vision_caption`) → маркер в messages
      заменяется на "[Image description: ...]" → Qwen3.5 отвечает обычным текстовым путём
      (tools/grammar/стрим). vision_model_path=NULL → text-only (обратно совместимо).
      Проверено через Claude Code: "что на картинке?" → связный ответ. KV vision-ctx
      чистится перед image_eval (M-RoPE требует возрастающих позиций — иначе segfault на
      повторном запросе).
- [x] Юнит-тесты парсинга image-блоков (tests/testthat/test-serve-anthropic.R, light):
      base64→файл round-trip, marker-вставка, text-only роняет картинку.
- [x] Примеры: `inst/examples/serve_anthropic_vision.R` (запуск без Claude Code + curl
      с base64 + --selftest), `claude_code_launcher.sh` (env VISION_MODEL/MMPROJ, по
      умолчанию Qwen3.5+Qwen2-VL вместе).
- [ ] ОСТАЛОСЬ (не блокеры 0.2.5): CRAN-редирект stderr/stdout в clip/mtmd; miniaudio.h
      (4 МБ single-header) убрать/опционализировать; несколько картинок в одном запросе
      сейчас даёт caption на каждую (ок), но без проверки порядка вставки на edge-кейсах.

### ✅ РЕЛИЗ 0.2.5 (сессия 2026-06-27)
Движок переведён на классовый каркас master (128 арок), Qwen3.5 работает
end-to-end, vision/mtmd на месте. Обновлены DESCRIPTION/NEWS/README/TODO.
- [x] LoRA-биндинги под master multi-adapter API (llama_set_adapters_lora) —
      закрыло последний undefined-символ; пакет грузится, тесты FAIL 0 | PASS 360.
- [x] Qwen3.5 (qwen35) ЖИВОЙ e2e через llama_serve_anthropic + Claude Code.
- [x] enable_thinking=FALSE по умолчанию в llama_serve_anthropic: thinking-модели
      (Qwen3.5) иначе тратят весь бюджет в незакрытом <think> → пустой ответ.
      + strip_thinking-страховка (не отдавать пустоту, если вырезание съело всё).
      Диагностика: inst/examples/diag_qwen35.R (сырой ген без chat-слоя).
- [x] Tool-вызовы в serve_anthropic: defer_stream теперь и для tool-форматов
      (не только GENERIC). Qwen3.5 даёт Hermes-формат (id 11) — стрим раньше слал
      сырые <tool_call>{...} живьём до парсинга → Claude Code видел мусор. Теперь
      при наличии tools копим молча и отдаём распарсенные tool_use на finalize.
      + гарантия непустого content (text/blocking fallback), иначе пустой
      content-массив = "empty or malformed response" у Claude Code.
- [x] Переполнение контекста после Read: история Claude Code растёт быстро
      (2 файла ≈ 56k токенов > n_ctx). claude_code_launcher.sh: n_ctx=65536 по
      умолчанию (override N_CTX=...). serve_anthropic ветка avail<1 при stream
      теперь шлёт ПОЛНУЮ Anthropic-последовательность (message_start→текст→stop),
      а не одинокий error до message_start.
- [x] **CLAUDE CODE НА QWEN3.5 РАБОТАЕТ END-TO-END**: привет → обзор функций с
      Read 2 файлов → связный ответ, без malformed. Все 3 бага сессии закрыты.

### ПЛАН: КРУПНЫЙ БЛОК ЯДРА — классовый каркас master (сессия 2026-06-26)

Самый большой кусок апгрейда. Решение user: делать целиком (create_tensor-
рефакторинг НЕОТДЕЛИМ от перевода 104 моделей в классы).

**Что меняется (разведка, факты):**
- master: `llama_model` (абстракт.) → `llama_model_base` → per-arch классы
  `llama_model_X : llama_model_base` с виртуальными методами `load_arch_hparams`/
  `load_arch_tensors`/`build_arch_graph`. Фабрика: `case LLM_ARCH_X: return new
  llama_model_X(params)` в llama-model.cpp.
- `create_tensor(tn(...), {ne}, flags)` стал МЕТОДОМ класса модели (знает ml/buft
  через this->ml), а не loader напрямую. 1857 вызовов в наших моделях — НО
  сигнатура ВЫЗОВА почти та же (tn, ne, flags), меняется только КУДА он определён.
- model.cpp 8490→2514 (логика арок ушла в models/*.cpp классы), arch.cpp
  2595→909 (рег. вынесена), model-loader 1261→1695 (buft-распределение + ctx_map).
- models/: у нас 104 (b7898-стиль), master 128 (классы). Наши *-iswa/t5-dec/enc/
  graph-context-mamba — b7898-именование, заменяются master-версиями.

**КЛЮЧЕВОЕ УПРОЩЕНИЕ:** в файлах крупного блока (model/arch/graph/context/loader)
НАШИХ ПАТЧЕЙ НЕТ (0 маркеров). Наши патчи — в grammar/unicode/json/chat-слое,
который НЕ часть этого блока. Значит блок можно брать из master БЛИЖЕ к as-is.

**ЧТО НЕ ТЕРЯТЬ при взятии master-каркаса:**
- наш `n_embd_features` (hparams, posnet) — УЖЕ внесён в master-совместимом виде.
- наш delta-net (models/qwen3next.cpp — СВОИ build_delta_net_chunking/AR, ggmlR
  не имеет fused-op). master qwen3next зовёт llm_build_delta_net_base.build_delta_net
  → при взятии master models/ ПРОВЕРИТЬ что delta-net-base.cpp компилируется на
  ggmlR (solve_tri/cumsum/tri есть; fused — нет, нужен fallback).
- GGML_MAX_DIMS=5 vs master 4 (см reference): structured-binding на ->ne сломается
  в master models/*.cpp — искать как в deepseekocr.
- наши r_*-биндинги НЕ затрагиваются (отдельный слой).

**ПЛАН (порядок, точки проверки):**
- [ ] 0. БЭКАП всего src/ перед началом (большой блок, нужен надёжный откат).
- [x] 0. БЭКАП всего src/ → scratchpad/src_backup_precore.
- [x] 1. КАРКАС взят из master + компилируется пофайлово:
      llama-model.h, llama-graph.h, llama-model-loader.{h,cpp}, models/models.h.
      В наш llama.h добавлены: typedef `llama_model_set_tensor_data_t` +
      ftype enum NVFP4(39)/Q1_0(40) (нужны loader'у). model-loader.cpp использует
      ggml_backend_* (buft-распределение) — ВСЕ есть в ggmlR (проверено в libggml.a).
      Движок пока НЕ собирается целиком (arch/graph/context/model .cpp + models/*.cpp
      ещё b7898 — Шаги 2-5).
- [x] 2. llama-arch.{h,cpp} ВЗЯТЫ ЦЕЛИКОМ из master, компилируется. Сверка
      ДО взятия: ВСЕ наши арки/тензоры/KV-ключи есть в master (master=надмножество,
      comm пусто) → ничего не потеряно. Наших патчей в arch нет. Enum теперь полный
      master-набор (QWEN35/QWEN35MOE/KIMI_LINEAR/GLM_DSA/...).
- [!] РАЗБОР ПОЛЁТОВ: в Шагах 1-3 я брал СУЩЕСТВУЮЩИЕ файлы через cp (нарушил
      правило «существует→править руками»). Проверка 3-сторонним diff (бэкап vs
      ЧИСТЫЙ b7898) показала: почти всё затёртое = мои же правки этой сессии
      (head_k→методы, SWA-ключи — master их имеет). РЕАЛЬНО потеряна 1 наша
      доработка → ВОССТАНОВЛЕНА вручную: профайлер LLAMA_DECODE_PROF (perf-gap
      диагностика) в llama-context.cpp (process_ubatch + decode, ~25 точек замера
      адаптированы под master-тело) + метод n_splits() (context.h+cpp). Компилируется.
      [Старый graph.cpp-обход `ggml_build_forward_select not available` НЕ нужен —
      функция теперь ЕСТЬ в ggmlR.] ВПРЕДЬ: существующие файлы — только руками.
- [x] 3. КЛАСТЕР graph/context из master (развернулся шире — потянул цепочку):
      llama-graph.cpp, llama-context.{h,cpp}, llama-kv-cache.{h,cpp},
      llama-io.{h,cpp} (read(dst,size) API), llama-memory-recurrent.{h,cpp},
      llama-cparams.h (fused_gdn), + НОВЫЙ файл llama-ext.h (master вынес туда
      llama_memory_breakdown_data — убран из context.h). Наших патчей нигде нет.
      В llama.h добавлены enum: LLAMA_STATE_SEQ_FLAGS_ON_DEVICE=2,
      LLAMA_SPLIT_MODE_TENSOR=3. Все + соседи (kv-cache-iswa, memory-hybrid*,
      memory) компилируются. ОСТАЁТСЯ Makevars: добавить llama-io.cpp если не было,
      llama-ext.h header-only.
- [x] 4. llama-model.cpp из master (8490→2514, фабрика new llama_model_X +
      load_arch). 3-сторонний diff (бэкап vs ЧИСТЫЙ b7898) ПОДТВЕРДИЛ: наших
      b7898-доработок НЕТ (только мои правки этой сессии — master их содержит).
      n_embd_features master НЕ использует (грузит FEATURES в n_embd). Компил-ся.
- [x] 5. models/*.cpp: взяты ВСЕ 128 master-классов (вкл. qwen35/qwen35moe — ЦЕЛЬ!).
      Удалены 11 b7898-only файлов (cohere2-iswa/t5-dec/enc/graph-context-mamba/...
      — master их объединил). qwen3next 3-сторонний diff = чистый b7898 (наших
      правок нет). ВСЕ 128 компилируются (fail=0). GGML_MAX_DIMS-проблема НЕ
      всплыла (была только в deepseekocr=mtmd). delta-net-base + qwen35 + qwen3next
      OK — fused gated_delta_net ЕСТЬ в ggmlR (поправка к памяти).
- [x] 6. Makevars: НЕ потребовал правок (MODELS_CPP=$(wildcard models/*.cpp)
      подхватил 128 авто, удалённые 11 b7898-файлов исчезли с диска).
      Доп. потребители каркаса приведены к master (3-сторонний diff = наших
      доработок нет): llama-vocab.{h,cpp} (мои pre-tok в master есть),
      unicode.{h,cpp} (master отказался от wregex→custom; наш CRAN-warning-патч
      неактуален, redirect через Makevars -include r_llama_compat.h сохранён),
      llama-quant.cpp, llama.cpp (главный). ВСЕ 42 корневых + 128 models + 32 mtmd
      компилируются ПОФАЙЛОВО (fail=0). Makevars без stale-ссылок.
      ПРИМЕЧАНИЕ: llama-sampling.cpp оставлен (master переименовал в
      llama-sampler.cpp, но наш sampling работает — не трогаем пока).
- [x] 7. ПОЛНАЯ СБОРКА R CMD INSTALL — 1-я попытка нашла рассинхрон:
      llama-model-saver.{h,cpp} был ещё b7898 (ctor по ссылке vs указатель в
      master llama.cpp). 3-сторонний diff: наших правок нет → взят из master.
      Урок: компиляция с -include r_llama_compat.h маскировала; перепроверил
      ВЕСЬ корень БЕЗ compat (+ R.h для r_*) — все 42 .cpp чисты.
      2-я попытка нашла: master llama-quant.cpp использует новые поля
      llama_model_quantize_params (dry_run, типизир. imatrix/kv_overrides/
      tt_overrides/prune_layers + новые типы llama_model_imatrix_data/
      tensor_override). Обновил struct + добавил типы в наш llama.h ВРУЧНУЮ.
      **СБОРКА КОМПИЛИРУЕТСЯ И ЛИНКУЕТСЯ В .so** (3-я+ попытки). Осталось
      добить undefined-символы при загрузке .so (наши r_*-биндинги зовут
      master-переименованные/удалённые функции):
        - [x] `llama_memory_breakdown_print` — master УБРАЛ, заменил на
          `llama_get_memory_breakdown()` (возвращает std::map). Переписал наш
          биндинг r_llama_memory_breakdown_print: агрегирует map + Rprintf.
          + `#include "llama-ext.h"` в r_llama_interface.cpp.
        - [x] `llama_n_splits` — наша C-обёртка профайлера (была в бэкапе, я её
          НЕ восстановил вместе с методом n_splits()). Восстановил: декларация
          в llama.h + определение `int llama_n_splits(ctx){return ctx->n_splits();}`
          в llama-context.cpp.
        - [x] ПРОВЕРИТЬ ОСТАЛЬНЫЕ undefined: `nm -D -u llamaR.so` (ggml линкуется
          статически через --whole-archive → его символы уже в .so, остаются
          только наши недозванные llama_*). Нашлось РОВНО 3, все про LoRA:
          master заменил `llama_set_adapter_lora`/`llama_rm_adapter_lora`/
          `llama_clear_adapter_lora` одним `llama_set_adapters_lora(ctx,
          adapters**, n, scales*)` (множественные адаптеры, перезапись всего
          набора). Переписал 3 биндинга в r_llama_interface.cpp с per-ctx
          tracking-map (std::map<ctx,{adapter:scale}>): apply добавляет/обновляет,
          remove убирает один (keep others, -1 если не было), clear очищает —
          каждое действие пересобирает массив и зовёт set_adapters_lora.
          Сохранён документированный контракт multiple/remove-specific.
          Декларации в llama.h приведены к master. +#include <map>.
      [x] R CMD INSTALL ПРОШЁЛ ЗАГРУЗКУ → devtools::test() = FAIL 0 | PASS 360
      | SKIP 9. Реальный инференс на tiny-mistral (Vulkan decode) работает.
      КРУПНЫЙ БЛОК ЯДРА ЗАВЕРШЁН — движок на классовом каркасе master.
      [x] ТОЧКА ИСТИНЫ Qwen3.5: `Qwen3.5-9B-UD-Q6_K_XL.gguf` грузится и генерит
      связный текст ЖИВЬЁМ через llama_serve_anthropic — Claude Code подключён
      end-to-end (claude_code_launcher.sh, дефолтная модель → qwen3.5).
      qwen35-арка (master models/qwen35.cpp) работает, delta-net не выдаёт мусор.
      ЗАМЕЧЕНО: <think>-блок модели утекает в ответ — у serve_anthropic есть
      strip_thinking=TRUE, но в лаунчере не прокинут (можно добавить при желании).
      БЭКАП до крупного блока: scratchpad/src_backup_precore (откат если надо).

**РИСК:** это ломает рабочий движок до полной сборки (часы/несколько сессий).
Откат — из бэкапа. Делать аккуратно, слоями, с компиляцией на каждом шаге.

### РАМКА: полный перевод llamaR на новую версию апстрима (master) — сессия 2026-06-26

**Курс задан:** это НЕ точечный бэкпорт отдельных арок, а **обновление всего
движка** llamaR на свежий `/mnt/Data2/DS_projects/llama.cpp-master`. qwen35,
vision/OCR, новые арки (kimi_linear, glm_dsa, mistral4, gemma4, step35 …),
рефакторинг каркаса (модели-классы), новые KV-параметры — всё это части ОДНОГО
обновления. Принцип: приводим наш код к виду master (нейминг файлов/классов/
тензоров), сохраняя наши наработки (delta-net, CRAN-редиректы, serve_*-слой,
chat/tool-слой, grammar-патчи). Цель — минимальный diff с master.
Подсистемы, которых у нас нет (vision/mtmd) — копируем целиком. Vision-фундамент
(STB + clip-движок) нужен и сам по себе, и под vision-арки qwen3vl/OCR.

### КАРТА расхождений основного движка (b7898 → master) — сессия 2026-06-26

Сверка `llamaR/src/*.{cpp,h}` (без mtmd/vendor/наших r_*/chat-слоя) против
`llama.cpp-master/src/`. Из 52 общих файлов: **12 идентичны, 40 расходятся.**

**Переименования / новое в master (структура):**
- `llama-sampling.cpp/.h` → **`llama-sampler.cpp/.h`** (переименован апстримом).
- новый **`llama-ext.h`** (нет у нас).

**КРУПНЫЙ рефакторинг каркаса (master в РАЗЫ меньше — логика вынесена в классы
models/*.cpp):** — нельзя мержить построчно, нужна осознанная миграция:
- `llama-model.cpp`  8479→2514 (Δ8029) — главный; switch-графы → классы.
- `llama-arch.cpp`   2592→909  (Δ1987) — рег. арок/тензоров вынесена.
- `llama.cpp`        1178→578  (Δ1092)
- `llama-quant.cpp`  1069→1407 (Δ1074)
- `llama-context.cpp`3798→3826 (Δ1276)
- `llama-model-loader.cpp` 1261→1695 (Δ692), `llama-graph.cpp` 2525→2929 (Δ642).

**Средние (десятки-сотни строк, мержабельно с вниманием):** llama-vocab.cpp
(266), unicode.cpp (290), llama-kv-cache.cpp (332), llama-model.h (236),
llama-model-saver.cpp (218), llama-hparams.* (42+46), llama-graph.h (86),
llama-arch.h (62).

**Мелкие (≤10 строк — тривиально):** llama-batch.* (4+2), llama-io.* (9+6),
llama-mmap.h (1), unicode.h (2), llama-memory-hybrid* (6+6), llama-vocab.h (6),
llama-kv-cache-iswa.cpp (4), llama-cparams.h (3), llama-impl.cpp (10) и др.

**12 ИДЕНТИЧНЫХ (не трогать):** llama-cparams.cpp, llama-grammar.h,
llama-kv-cache-iswa.h, llama-kv-cells.h, llama-memory.{cpp,h},
llama-memory-hybrid.h, llama-memory-hybrid-iswa.h, llama-memory-recurrent.h,
llama-quant.h, unicode-data.{cpp,h}.

**НАШИ ПАТЧИ (нельзя потерять при миграции):**
- `[llamaR patch]` / CRAN-правки в: `llama-grammar.cpp` (parse_sequence,
  advance_stack, print_rule_binary удалён, end_element), `unicode.cpp`,
  `json-schema-to-grammar.cpp` (maxLength>1024), `llama-sampling.cpp`,
  `common_chat_support.cpp`.
- Force-include `r_llama_compat.h` (CRAN stderr/exit редирект) на: llama-grammar,
  llama-impl, llama-quant, unicode, json-partial, json-schema-to-grammar.
- Весь наш слой (НЕ в master): r_llama_interface, r_chat_interface,
  r_mtmd_interface, r_llama_compat.h, chat*, peg-parser, json-*, regex-*,
  common*, log.*, build-info.

**ПОРТ ИЗМЕНЕНИЙ (вариант а, не файлы целиком) — прогресс:**
- [x] **pre-tokenizer типы** QWEN35(46)/TINY_AYA(47)/JOYAI_LLM(48)/JAIS2(49) →
      `llama-vocab.h` (enum, значения = master) + `llama-vocab.cpp` (regex-case
      verbatim + детекция по tokenizer_pre с флагами). Компилируется. GEMMA4(50)/
      SARVAM_MOE(51) ПРОПУЩЕНЫ — требуют byte_encode=false (механики нет). Все
      правки помечены `[llamaR]`. QWEN35 pre-tok нужен для Qwen3.5.
- [x] `llama-batch.h` (опечатка sequantial→sequential) → ИДЕНТИЧЕН master.
- [x] `llama-impl.cpp` (формат %5→%6 ×4, BOOL через int8_t) → ИДЕНТИЧЕН master.
- [x] `llama-batch.cpp` (M-RoPE: pos.resize(n_tokens*n_pos_per_embd) в
      ubatch_reserve — баг-фикс для mrope qwen2vl/qwen35; n_pos_per_embd —
      поле нашего класса, изолировано) → ИДЕНТИЧЕН master.
- [x] `llama-impl.h` (buffer_view<T> template + FGDN tensor-name defines +
      выравнивание FATTN) → ИДЕНТИЧЕН master. Аддитив, пока не используется.
- [x] **ПРОВЕРЕНО СБОРКОЙ:** R CMD INSTALL + devtools::test() = FAIL 0 | PASS 360
      | SKIP 9 (с vocab/batch/impl-правками). Чистая база.

**Сознательно ПРОПУЩЕНЫ (не изолированы / фича не нужна сейчас):**
- `llama-io.{h,cpp}` — рефакторинг I/O-API (read(size)→read(dst,size),
  read_tensor, write_tensor(non-const)); тянет llama-kv-cache.cpp (10+ вызовов
  read_to), llama-context. Связный кластер, не точечно.
- `llama-hparams.cpp` — `n_embd_head_k/v` СТАЛИ методами с аргументом `il`
  (было поле) + n_embd_head_kda (Kimi). Ломающий рефакторинг, тянет graph/model.
- `llama-cparams.h` fused_gdn_* флаги — верхушка fused-gdn фичи; логика в
  llama-context.cpp (рефакторинг) + требует ggml_gated_delta_net (нет в ggmlR).
  Наш delta-net и так работает через chunking/autoregressive.

- [x] `llama-chat.{h,cpp}` — новые встроенные chat-шаблоны (deepseek-ocr,
      granite-4.0, hunyuan-ocr) + переименование GRANITE→GRANITE_3_X (изолировано,
      нигде вовне не используется). Наших патчей нет → взято целиком из master,
      ИДЕНТИЧНО. Компилируется. (Это встроенные llama-chat шаблоны, НЕ наш
      tool-aware chat-слой chat.cpp/jinja.)
- [x] `llama-adapter.{h,cpp}` — LoRA lifecycle: adapter хранит model*, при free
      удаляет себя из model->loras. Завязка model->loras УЖЕ есть у нас
      (llama-model.h:481). Наших патчей нет → взято целиком, ИДЕНТИЧНО. Компил-ся.

- [x] `llama-mmap.{h,cpp}` — новый `llama_file(FILE*)` ctor + owns_fp механика +
      std::ftell/fseek→llama_mmap_ftell/fseek (off_t, 64-bit) + Windows mmap-фиксы
      (#ifdef). Наших патчей нет → взято целиком, ИДЕНТИЧНО. Компилируется (Linux).
- [~] `unicode.{h,cpp}` — ПРОПУЩЕНО. НЕ аддитив: master выкинул std::codecvt/
      std::wregex (C++17 deprecated) и переписал regex-split через custom-функции
      (Δ290) + у нас есть `[llamaR patch]` (удалён unicode_cpts_to_utf8, CRAN).
      Связный рефакторинг + наш патч. byte_encode-сигнатура нужна только GEMMA4
      (тоже пропущен). Вернуться отдельным кластером.

**КЛАСТЕР hparams (Способ B — полный как master) — СДЕЛАНО (компиляция):**
- [x] `llama-hparams.h`: поля n_embd_head_k/v, n_rot → разбиты на _full/_swa;
      объявлены методы n_embd_head_k(il)/n_embd_head_v(il)/n_rot(il). Новые поля:
      moe_latent_size, rope_scaling_alpha, n_embd_head_kda, f_attn_value_scale,
      indexer_* (DSA), n_embd_per_layer (gemma4), swiglu_clamp_* (Step35).
      Сохранено наше n_embd_features (нет в master, используется posnet).
- [x] `llama-hparams.cpp`: реализации методов (verbatim master) + внутренние
      обращения (n_embd_k/v_gqa, _mla) → методы.
- [x] `llama-arch.{h,cpp}`: новые KV-ключи ATTENTION_KEY/VALUE_LENGTH_SWA,
      ROPE_DIMENSION_COUNT_SWA (enum + строки маппинга).
- [x] `llama-model.cpp` загрузчик: записи k/v/rot → _full + новый SWA-блок
      (default _swa = _full + чтение *_SWA ключей). Per-arch записи → _full.
- [x] ВОЛНА чтений (~135): hparams.n_embd_head_k → n_embd_head_k() во всех
      потребителях — llama-model/graph/kv-cache/context/model-saver + ВСЕ 104
      models/*.cpp (графы). kv-cache: const auto& → const auto (метод по значению).
- [x] **ПРОВЕРЕНО СБОРКОЙ:** R CMD INSTALL + devtools::test() = FAIL 0 | PASS 360
      | SKIP 9. Реальный инференс на tiny-mistral (decode/генерация) работает —
      семантическая эквивалентность (_swa = _full для не-SWA) подтверждена,
      поведение не изменилось. Бэкап в scratchpad/hparams_backup можно удалить.

**ИТОГ «мелкие/средние» (вариант а):** приведены к master —
llama-vocab.{h,cpp}, llama-batch.{h,cpp}, llama-impl.{h,cpp}, llama-chat.{h,cpp},
llama-adapter.{h,cpp}, llama-mmap.{h,cpp}. Осталось проверить сборкой.
ПРОПУЩЕНЫ (связные кластеры рефакторинга): unicode, llama-io, llama-hparams.cpp
(методы il), llama-model-loader (ml.meta→ml.metadata), cparams fused_gdn,
и весь КРУПНЫЙ блок (llama-model/arch/graph/context).

**ВЫВОД:** привести весь движок к master = переписать ядро под классовый каркас
(не построчный мерж). Главный блок — llama-model/arch/graph + переименование
sampling→sampler. РЕШИТЬ порядок: (а) сначала мелкие+средние (низкий риск,
приближает к master), (б) потом крупный рефакторинг каркаса, (в) или оставить
ядро b7898 и точечно добавлять новые арки. См. вопрос в конце сессии.

### Vision / OCR (mtmd-подсистема) — порт из master (план, сессия 2026-06-26)

**Цель:** мультимодальные модели (paddleocr, deepseek-ocr, hunyuan-vl и др. VL/OCR).
Подсистемы vision у нас НЕТ вообще → не бэкпорт, а **копирование целиком из master**
(`tools/mtmd/`), приведённое к нашему build/CRAN-контексту.

#### Результаты разведки (кода НЕ трогали)

- **У нас vision-слоя нет** — ни одного mtmd/clip-файла. Копируем подсистему как есть.
- **Все блокеры сняты:**
  - ✅ Все clip ggml-ops (`conv_2d`, `im2col`, `interpolate`, `pool_2d`, `gelu_quick`,
    `group_norm`, `flash_attn_ext`, `concat`, `pad`, `norm` …) **есть в ggmlR**
    (`ggmlR/inst/include/ggml.h`) → ViT-энкодеры заведутся без правок бэкенда.
  - ✅ Наш `llama_batch` уже принимает `embd` (`[n_embd, n_tokens]`,
    `llama-batch.h:46`) → image-эмбеддинги подаются как inputs-embeddings.
  - ✅ `llama_model_n_embd_inp()` у нас УЖЕ есть (`llama.h:539`,
    `llama-model.cpp:8159`) — точка интеграции clip→llama готова.
- **Объём:** `tools/mtmd/` без audio/cli/legacy ≈ **7700 строк**
  (clip.cpp 4220 + mtmd.cpp 1525 + mtmd-image + mtmd-helper) + энкодеры в
  `tools/mtmd/models/*.cpp` (cogvlm, deepseekocr, dotsocr, gemma4v, glm4v,
  hunyuanocr, internvl, paddleocr …).
- **Внешние зависимости (управляемы):**
  - **STB image** — vendored одиночный header `vendor/stb/stb_image.h` → копируем.
  - **miniaudio** — ТОЛЬКО для audio-моделей → откладываем, для OCR/VL не нужен.
  - `windows.h` — под `#ifdef`, не наша платформа.
  - common/chat/sampling — нужны только `mtmd-cli.cpp` (демо, НЕ копируем).
    Сама библиотека mtmd от них не зависит.
- **Архитектура:** clip — отдельный движок, параллельный llama. VL/OCR-модель =
  текстовый декодер (llama-арка) + vision-энкодер (clip-арка, `PROJECTOR_TYPE_*`).
  clip кодирует картинку → эмбеддинги → `llama_batch.embd` → обычный decode.

#### План (эпик; копируем, потом адаптируем под наш build/CRAN)

- [x] **Шаг 1 — vendored STB + clip-движок + mtmd скопированы в `src/mtmd/`**
      (зеркало master). Все **32 .cpp компилируются** пробной g++-сборкой
      с `-I. -Imtmd -I$(ggmlR include) -DGGML_USE_CPU`. Раскладка: clip*.{cpp,h},
      mtmd*.{cpp,h}, mtmd-audio.*, mtmd-image.*, mtmd-helper.*,
      debug/mtmd-debug.h, models/*.cpp, stb/stb_image.h, miniaudio/miniaudio.h.
- [x] **Шаг 2/3 — clip + mtmd-обёртка** (вошло в Шаг 1; audio оставлен, т.к.
      mtmd.cpp вшит в audio на 110 строк — выдирать дороже, чем держать).
- [x] **Шаг 4 (частично) — Makevars.** `MTMD_CPP` (wildcard) добавлен в OBJECTS;
      `-Imtmd` скоупнут на mtmd-объекты (для `debug/mtmd-debug.h`→`mtmd.h`);
      `-Imtmd/models` НЕ глобально (иначе наш src/models/ схватит чужой models.h);
      `mtmd-debug.cpp` НЕ собираем (CLI-утилита; символы в mtmd.cpp).
      Правка апстрима: `deepseekocr.cpp` structured binding `auto[c,w,h,b]=x->ne`
      → индексация (ggmlR GGML_MAX_DIMS=5 vs master 4).
      ОСТАЁТСЯ: CRAN-редирект stderr/stdout/exit в clip/mtmd через r_llama_compat;
      собрать пакет целиком (`R CMD INSTALL .`), а не пофайлово.
- [ ] **Шаг 4b — miniaudio для CRAN.** `miniaudio.h` = 4 МБ single-header —
      многовато для CRAN. ПЕРЕМЕСТИТЬ/убрать позже (по указанию user): варианты —
      отключить audio в helper через `#ifdef`, или вынести в неустанавливаемый
      путь, или собрать audio опционально.
- [x] **Шаг 5 — R-биндинги.** `src/r_mtmd_interface.cpp` (7 функций): `r_mtmd_init`
      (load projector), `r_mtmd_support_vision/audio`, `r_mtmd_marker`,
      `r_mtmd_set_verbosity`, `r_mtmd_bitmap_from_file`, `r_mtmd_eval`. Используют
      высокоуровневый helper (`mtmd_helper_bitmap_init_from_file` +
      `mtmd_tokenize` + `mtmd_helper_eval_chunks`) — helper сам гоняет text+image
      chunks через llama_context, возвращает new_n_past. Зарегистрированы в
      CallEntries (+forward-decls) в r_llama_interface.cpp; добавлен в Makevars
      (SOURCES_CPP + `-Imtmd`). R-обёртки `R/mtmd.R`: `llama_mtmd_load`,
      `llama_mtmd_support_vision/audio`, `llama_mtmd_marker`,
      `llama_mtmd_set_verbosity`, `llama_image_load`, `llama_image_eval`.
      `devtools::document()` прошёл (exit 0, заодно перекомпилировал весь пакет
      — линковка mtmd OK): 7 export'ов в NAMESPACE + man-страницы созданы.
- [~] **Шаг 6 — интеграция в генерацию.** Базовый путь УЖЕ работает через
      `llama_image_eval` (eval_chunks пишет в KV cache, отдаёт new_n_past →
      дальше обычный `llama_gen_next`). Прямой проброс image-эмбеддингов в
      `r_llama_gen_begin` не нужен для MVP. ОСТАЁТСЯ: проверить, что gen-loop
      корректно стартует с не-нулевого n_past после image_eval.
- [x] **Шаг 7a — юнит-тесты** `tests/testthat/test-mtmd.R`: 14 no-model тестов
      (marker, set_verbosity, валидация аргументов всех 7 функций, guard'ы) +
      2 projector-зависимых (load mmproj, capabilities, image→text) со
      `skip_if_no_mm()`. Полный набор: FAIL 0 | PASS 360 | SKIP 9. Пути для
      e2e-теста в начале файла (MM_MODEL_PATH/MM_MMPROJ_PATH) — выставить, когда
      появится локальная VL/OCR-модель.
- [ ] **Шаг 7b — живой e2e** на реальной OCR/VL-GGUF (нужен mmproj-GGUF):
      `model + mmproj → image_load → image_eval(prompt+marker) → gen_next` →
      связный текст (не мусор). Проверить per-модель препроцессинг (Риск №1).
      Локально mmproj-модели НЕТ — тест скипается, ждёт модель.

**Источники:** `/mnt/Data2/DS_projects/llama.cpp-master/tools/mtmd/` (вся подсистема),
`vendor/stb/stb_image.h`. Точка интеграции: `llama_batch.embd` +
`llama_model_n_embd_inp()` (оба у нас уже есть).
**Риск №1:** препроцессинг изображения (resize/normalize per-модель) — тонкий,
ошибка даёт молчаливый мусор; сверять на реальной модели.
**Риск №2:** clip-проектор грузит СВОЙ GGUF (отдельный от модели) — нужен путь к
mmproj-файлу; проверить нашу загрузку GGUF на projector-tensors.

### Очередь + GENERIC-распаковка в serve_anthropic (сессия 2026-06-26) ✅

Продолжение стабилизации Claude Code. Правки в `R/serve_anthropic.R`, проверено
на живой llama3-8b end-to-end (curl, stream — текст и tool).

- [x] **503 → FIFO-очередь (сериализация вместо отказа).** Раньше параллельный
      запрос Claude Code получал HTTP 503 `overloaded_error` (busy-флаг); при
      исчерпании ретраев — «API error». Подтверждено логом guard: слот честно
      освобождался (ACQUIRE→RELEASE), 503 был именно из-за перекрытия. Решение:
      drogonR async-respond есть ТОЛЬКО в форме стрима (нет общего API — проверено
      в `src/r_dispatcher.cpp`/`stream_session.cpp`), поэтому **ВСЕ** запросы
      (stream и blocking) отвечаются через `dr_stream`. Blocking отдаётся одним
      JSON-чанком с `content_type="application/json"` (chunked, без SSE-обвязки —
      для клиента обычный ответ). Сериализация — FIFO-тикеты (`ctx_busy`, `q_next`,
      `q_head`, `q_waiting`) в замыкании хендлера; ctx берётся в фазе `acquire`
      только в голове очереди. `next_chunk` — фазовая машина:
      wait→acquire→start→text→finalize→done. Отладочные `[queue #N]`-логи оставлены
      на обкатку.
- [x] **GENERIC-формат утекал сырым JSON в стрим.** Симптом: Claude Code на
      llama3-8b показывал `{"response": "..."}` / не вызывал tools. Причина НЕ в
      парсере: `llama_chat_parse` корректно распаковывает `{"response":...}`→content
      и `{"tool_call":...}`→tool_calls (проверено напрямую). Баг — стрим слал сырые
      токен-дельты ЖИВЬЁМ, до парсинга, поэтому `{"response":` утекал. Детект формата
      тоже исправен: llama3-gguf со стоковым шаблоном (без tool-разметки) апстрим
      кладёт в GENERIC (id 1) — это поведение llama.cpp, не баг. Фикс: флаг
      `defer_stream` (format==1) — для GENERIC стрим НЕ эмитит дельты, а копит молча
      (как blocking) и в finalize отдаёт распарсенный content одним text-блоком +
      tool_use-блоки. Прочие форматы стримятся живьём как раньше.

### Qwen3.5 (qwen35 / qwen35moe) — порт новой архитектуры (план, сессия 2026-06-26)

**Симптом:** `unknown model architecture: 'qwen35'` на `Qwen3.5-9B-UD-Q6_K_XL.gguf`.
Chat-форматы ПОЛНЫЕ (сверено ранее) — добавлять формат не нужно, только арку.

#### Результаты разведки (diff наш b7898 ↔ master, кода НЕ трогали)

- **Фундамент уже есть.** b7898 содержит recurrent/hybrid память
  (`llama-memory-recurrent`, `llama-memory-hybrid`), все `ssm_*` поля и
  `recurrent_layer_arr`/`is_recurrent()` в hparams, и — главное — **рабочий
  `qwen3next`** (`src/models/qwen3next.cpp`), прямой предок qwen35: та же
  gated delta-net (linear attention) + hybrid recurrent.
- **qwen35 ≈ qwen3next + дельта** (по master `diff qwen3next↔qwen35`):
  - FFN: qwen3next MoE → **qwen35 dense** (ffn_gate/down/up). qwen35moe — MoE.
  - RoPE: **mrope** (`rope_sections[4]`, key `ROPE_DIMENSION_SECTIONS`) —
    у нас уже грузится в hparams, но в qwen3next-attn НЕ прокинут; для qwen35
    надо передать `sections[4]` в `ggml_rope_multi`/attn.
  - SSM beta/alpha: qwen3next объединённый `ssm_beta_alpha` (`SSM_BETA_ALPHA`,
    `blk.%d.ssm_ba`) → qwen35 **раздельные** `ssm_beta`+`ssm_alpha`
    (`SSM_BETA`/`SSM_ALPHA`) + новые тензоры `SSM_CONV`, `SSM_OUT`,
    `SSM_A_NOSCAN`, `SSM_NORM`, `SSM_DT`.
  - Типы: 0.8B/2B/4B/9B/27B (по n_layer 24/32/64 + n_embd).
- **Каркас разошёлся (рефакторинг master), но локально.** У нас стиль
  `struct llm_build_qwen3next : llm_graph_context_mamba` + hparams/tensors в
  switch-блоках `llama-model.cpp`. В master — `llama_model_qwen3next : graph`
  + `llm_build_delta_net_base`. **Поэтому master `qwen35.cpp` нельзя копировать
  как есть** — переписываем в НАШ стиль, взяв за основу наш qwen3next.
- **Delta-net математика РАЗНАЯ, но у нас рабочая.** master вынес её в общий
  `delta-net-base.cpp` (`build_delta_net` → fused/chunking/autoregressive).
  Наш b7898 qwen3next имеет СВОИ `build_delta_net_chunking` (CHUNK_SIZE=64,
  `ggml_solve_tri`/`ggml_cumsum`/`ggml_tri`) + `build_delta_net_autoregressive`.
  **Эти ops уже есть в ggmlR** (наш qwen3next их компилирует и вызывает) →
  qwen35 строим поверх НАШЕГО delta-net. Fused `ggml_gated_delta_net`
  (master-only, опц. через `cparams.fused_gdn_*`) НЕ нужен, ggmlR не трогаем.
- **Мелкий API-«налог»** (если что-то тянуть из master дословно): у master
  `n_embd_head_v()` стало методом (у нас поле), `build_attn` +1 arg,
  `build_lora_mm(..., _s)` LoRA-scale, новая сигнатура `build_moe_ffn`. При
  порте в наш стиль НЕ копируем master-вызовы — используем наши.

**ПРИНЦИП НЕЙМИНГА (общий для всего порта, реш. 2026-06-26):** наш код
приводим к ВИДУ апстрима — имена файлов/классов/символов/тензоров как в master
(`llm_build_qwen3next` → `llama_model_qwen3next : graph`, `ssm_beta`/`ssm_alpha`,
`src/models/qwen35.cpp` и т.д.), структурируем похоже. НО логику delta-net и
наши фичи (CRAN-редиректы, наши патчи) сохраняем своими. Цель — минимальный diff
с master, чтобы новые арки переносились легко. Не слепое копирование.

#### План (точечно, наши фичи не трогаем — только ДОБАВЛЯЕМ case'ы/файлы)

- [ ] **Шаг 1 — `llama-arch`.** enum `LLM_ARCH_QWEN35` (+`QWEN35MOE`), имена
      `"qwen35"`/`"qwen35moe"`, новые tensor-enum `SSM_BETA`/`SSM_ALPHA`
      (+маппинг `blk.%d.ssm_beta`/`ssm_alpha`), tensor-таблицы для обеих арок.
      Ничего не ломает.
- [ ] **Шаг 2 — `llama-hparams`.** Проверить, что всех полей хватает (ssm_*,
      rope_sections, full_attention_interval) — по разведке всё есть, вероятно
      no-op.
- [ ] **Шаг 3 — `llama-model.cpp` load_hparams.** case QWEN35/QWEN35MOE:
      ssm_* + mrope sections + `recurrent_layer_arr` по `FULL_ATTENTION_INTERVAL`
      (=4) + type switch (24/32/64 → 0.8B/2B/4B/9B/27B; moe → свои).
- [ ] **Шаг 4 — `llama-model.cpp` load_tensors.** case: раздельные
      `ssm_beta`/`ssm_alpha` вместо `ssm_beta_alpha`, dense FFN (qwen35) /
      MoE FFN (qwen35moe, как в qwen3next), attn-тензоры + q/k norm.
- [ ] **Шаг 5 — `src/models/qwen35.cpp`.** Новый `llm_build_qwen35`
      (копия нашего `llm_build_qwen3next` как база) с правками: mrope в attn
      (`sections[4]`), раздельные beta/alpha в `build_layer_attn_linear`,
      dense `build_layer_ffn`. Переиспользуем наш delta-net (chunking/AR).
      Зарегистрировать в `models.h` + Makevars + dispatch в `llama-model.cpp`.
- [ ] **Шаг 6 — qwen35moe.** После dense: вернуть MoE-FFN ветку (есть в
      qwen3next) + объединённый/раздельный beta/alpha по факту GGUF.
- [ ] **Шаг 7 — сборка + тест** на `Qwen3.5-9B-UD-Q6_K_XL.gguf`: грузится,
      генерит связный текст (delta-net не выдаёт мусор), chat/tool-слой ок.

**Источники для сверки:** master `src/models/qwen35.cpp` (473), `qwen35moe.cpp`
(527), `qwen3next.cpp` (633), `delta-net-base.cpp` (445), `models.h` (классы
1663-1789). Наш `src/models/qwen3next.cpp` (873) — основа. Diff'ы собраны в
scratchpad сессии (`diff_qwen3next.txt`, `master_qwen35*.cpp`).
**Риск №1:** раздельные ssm_beta/alpha — проверить, что наша delta-net
математика принимает их в той же семантике, что объединённый ssm_beta_alpha
(иначе модель молча выдаёт мусор). Сверять на реальном GGUF.

### Claude Code на локальной модели — стабилизирован (сессия 2026-06-26) ✅

Серия багов, ронявших Claude Code через `llama_serve_anthropic`. Все исправлены;
для постоянного эффекта через launcher нужна пересборка `R CMD INSTALL .`
(grammar-фиксы — C++, остальное — R-код; всё вместе через INSTALL).

- [x] **Большая tool-грамматика не парсилась → segfault.** Симптом был
      `number of repetitions exceeds sane defaults` (одиночная проверка `{m,n}`
      порога 2000 в `parse_sequence`). НО первопричина оказалась НЕ в
      `parse_sequence`: `json-schema-to-grammar.cpp` генерил `char{0,524288}` для
      строкового поля с большим `maxLength` (инструмент Workflow). Фикс:
      `[llamaR patch]` в `src/json-schema-to-grammar.cpp` — при `maxLength > 1024`
      трактовать как без предела → `char*` вместо `char{0,N}`. (master тоже валит
      эту грамматику: порог 2000 у него такой же.)
- [x] **Перенос parse_sequence + advance_stack из master** (`src/llama-grammar.cpp`).
      `parse_sequence`: кумулятивная проверка `n_prev_rules * total_rules >=
      MAX_REPETITION_THRESHOLD`, сброс `n_prev_rules` в терминалах, `stoul`→`stoull`.
      `advance_stack`: итеративная версия (`std::set seen` + `todo`-очередь,
      `#include <set>`) — защита от stack overflow / экспоненциального взрыва.
      Плюс фикс `grammar_root` (хардкод `"root"` → параметр). Тела идентичны master.
      Сохранены наши патчи: `print_rule_binary` (удалён, CRAN), `end_element`
      (пустые правила), `fprintf`-логирование (CRAN-редирект).
- [x] **Второй+ ответ «глотал буквы» (конкурентность).** Claude Code при старте
      шлёт ПАРАЛЛЕЛЬНЫЕ stream-запросы (основной + генерация заголовка сессии).
      Сервер single-sequence (один ctx/KV) — конкурентные `gen_begin` топтали общий
      контекст → пропуск токенов (текст валидный UTF-8, но выпадали слоги). Фикс:
      busy-флаг в `serve_anthropic.R` → второй запрос получает HTTP 503
      `overloaded_error` (Claude Code ретраит). Тонкость R: освобождение через
      `released <- TRUE` (локальное `<-`, НЕ `<<-`: `<<-` пропускает локальную
      переменную обработчика и пишет во внешнее окружение → `on.exit` гасил busy
      преждевременно). Диагностика: прямые R-вызовы / curl / транспорт SSE — чисты;
      баг только при конкурентном стриме.
- [x] **Graceful shutdown сервера.** launcher слал SIGTERM (R игнорирует) по обёртке
      Rscript (внутренний exec/R сиротел). Фикс в `claude_code_launcher.sh`:
      `setsid` + `kill -INT` по группе (R ловит SIGINT как `interrupt` → чистый
      выход через `on.exit(dr_stop())`), затем `kill -KILL` по таймауту.

### Сверка формата Anthropic Messages API (сессия 2026-06-26) ✅

Сверено по РЕАЛЬНЫМ данным с провода (репо `/mnt/Data2/DS_projects/claude-code-main`
содержит только плагины/примеры, спецификации API в нём НЕТ). Правки в
`R/serve_anthropic.R`.

- [x] **Двойной system — потеря данных.** Claude Code шлёт top-level `system` И
      `system`-роль внутри `messages`. Chat-шаблон ждёт ОДИН system → первый
      (top-level, основной системный промпт) молча терялся. Фикс:
      `.anthropic_to_messages` сливает весь system-контент в один лидирующий блок.
- [x] **`output_config.effort`** (low/medium/high) — маппим на бюджет генерации
      (0.5/0.75/1.0 × max_tokens).
- [x] **`ping`-события** в стрим (после `message_start`) — keep-alive по спеке.
- [x] **`error`-событие** в стриме при сбое генерации (был молчаливый обрыв).
- [x] **`strip_thinking = TRUE`** (параметр `llama_serve_anthropic`). Вырезаем
      `<think>...</think>` (полные пары, висячий `</think>`, незакрытый `<think>`
      в стриме — буферизуем до закрытия). `thinking`-блок Anthropic НЕ эмулируем:
      требует криптографическую `signature`. Входящие `thinking`-блоки игнорируем.
      Не-reasoning модели — no-op.
- Осознанно игнорируем (безвредно): `metadata.user_id`, `cache_control`, `model`.

**Решение.** Перенести из llama.cpp master (`/mnt/Data2/DS_projects/llama.cpp-master/
src/llama-grammar.cpp`, `parse_sequence`, ~строки 458-540) кумулятивную логику:
отслеживать `n_prev_rules` и `total_rules`, проверять `n_prev_rules * total_rules >=
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

### Continuous batching (vLLM-стиль, in-process) — план (сессия 2026-07-04)

**Позиционирование.** ОРТОГОНАЛЬНО worker-pool выше: там пропускная способность
растёт горизонтально (N процессов × 1 seq каждый), здесь — один процесс декодит
N последовательностей в ОДНОМ forward (батчинг seq в общем decode-loop, как
llama-server `-np N` / vLLM). Оба слоя совместимы: воркер-пул можно поставить
поверх continuous-batching-воркеров позже. Это тот самый «не continuous batching»,
про который сказано в roadmap выше — закрываем этот пробел.

**Разведка (факты, код НЕ трогали):**
- **Фундамент decode-loop УЖЕ есть.** `r_llama_generate_batch`
  (`src/r_llama_interface.cpp:1037`) содержит настоящий многосессионный цикл
  `while (n_active > 0)` (:1207): один `llama_decode` на N живых seq за шаг
  (:1228), per-seq state (`active[]`/`seq_pos[]`/`generated[]`/`n_generated[]`/
  `finished[]` + отдельный sampler на seq) — это уже слот-таблица. KV чистится
  на лету per-seq: `llama_memory_seq_rm(mem, s, -1, -1)` при EOS/лимите
  (:1253/:1266). ЕДИНСТВЕННОЕ ограничение: `n_seq` фиксируется ДО входа в
  `while` — новые запросы внутрь не добавляются (static batching, не continuous),
  результат возвращается разом в конце. Т.е. нужна ДОПИСКА событийности в
  существующий цикл, а не новый scheduler.
- **Серверный слой УЖЕ есть.** `R/serve.R` (`llama_serve_openai`) — сервер на
  drogonR, но через `dr_stream_sse` + `dr_post` (HTTP+SSE, R-путь) и
  single-sequence: каждый запрос — свой независимый generator, не общий батч.
- **Транспорт (drogonR) ГОТОВ ПОЛНОСТЬЮ** (drogonR 0.1.8). Всё, что нужно
  continuous batching'у, уже несёт ABI `drogonr_ws_handler_t`
  (`drogonR/inst/include/drogonR.h`):
    * session (= conn_id) переживает вызов хендлера; `send`/`close`/
      `is_connected` thread-safe из detached decode-потока (документированная
      гарантия) — decode-loop раздаёт токены N сессиям асинхронно.
    * O(1)-контракт хендлера зафиксирован в ABI: хендлер только кладёт запрос в
      нашу очередь и возвращается, генерация — на нашем потоке.
    * `dr_ws_cpp(app, path, package, callable, max_conns, idle_timeout,
      max_lifetime)` — грубый транспортный предохранитель: reject сверх лимита
      (WS close 1013), reap зависших сессий. НЕ заменяет наш seq-лимит по VRAM.

**Граница (что где живёт):**
- drogonR — транспорт (готов, не трогаем): HTTP/WS/SSE, async-ABI, backpressure-
  предохранитель. НЕ знает про seq_id / KV / VRAM.
- llamaR — scheduler (эта работа): очередь, слот-таблица seq→session, decode-loop
  событийный, seq-лимит по VRAM. ggmlR НЕ участвует (уровень тензоров, ниже).
- PagedAttention/пейджинг KV — ВНУТРИ llama.cpp (`llama_memory` уже страничный);
  с R-уровня мы только управляем seq_id-слотами, пейджингом занимается llama.cpp.

**План (дописка событийности в существующий батч-декодер, не новый scheduler):**
- [ ] **Шаг 1 — thread-safe очередь запросов** (пункт 2). Новый C++-модуль:
      `std::deque<PendingGen>` (prompt + drogonr_ws_session handle + send_fn) под
      mutex + condvar. cpp-WS-хендлер (drogonr_ws_handler_t) кладёт запрос и
      сразу возвращается (O(1)-контракт ABI). Decode-поток парковается на condvar,
      когда батч пуст.
- [ ] **Шаг 2 — событийный decode-loop** (пункт 2/4). Взять `r_llama_generate_batch`
      цикл (:1207) за основу, вынести в отдельный decode-поток (НЕ R-main), три
      вставки: (а) в начало итерации — дренаж очереди: новым запросам выделить
      свободный seq_id, prefill промпта, добавить в active[], n_active++;
      (б) после sampling токена — вместо накопления в generated[s] сразу
      `send(session[s], token)` в drogonR-сессию этого seq; (в) условие цикла
      `while (n_active > 0 || server_running)` — не выходить на пустом батче,
      парковаться на очереди. n_seq больше НЕ фиксирован.
- [ ] **Шаг 3 — слот-таблица seq→session** (пункт 4). Расширить существующий
      per-seq state полем `drogonr_ws_session* session` (+ n_prompt_tokens,
      позиция). Свободные слоты = пул seq_id [0, n_parallel). При завершении seq
      (EOS/лимит/disconnect) — seq_rm + close(session) + вернуть слот в пул.
      Проверять `is_connected(session)` между decode-шагами (ABI: дёшев, для
      этого и предназначен) — выкидывать disconnected seq из батча, не тратя
      compute.
- [ ] **Шаг 4 — seq-лимит по VRAM** (пункт 5, политика). `n_parallel` слотов =
      функция от размера KV-cache / VRAM (llamaR знает, drogonR — нет). Запросы
      сверх лимита ждут в очереди (не отклонять — очередь FIFO, как в serve_anthropic).
      drogonR `max_conns` выставить >= n_parallel как грубый потолок соединений.
- [ ] **Шаг 5 — `llama_serve_openai_ws()`** (пункт 4b). Новый серверный вход рядом
      с `llama_serve_openai` (HTTP/SSE оставить как есть для single-seq клиентов):
      регистрирует `dr_ws_cpp` на горячий путь (C++↔C++, минуя R-main), поднимает
      decode-поток, отдаёт токены через WS. OpenAI/SSE-совместимость поверх WS-фрейма
      (или отдельный WS-протокол — решить).
- [ ] **Шаг 6 — тесты.** Юнит: очередь (enqueue/dequeue под нагрузкой), слот-пул
      (выделение/возврат, исчерпание). E2e (heavy, нужна модель): N параллельных
      WS-клиентов получают токены ОДНОВРЕМЕННО (не по очереди), disconnect одного
      не роняет батч, seq-лимит держит N активных.

**Риск №1:** decode-поток — не R-main; он НЕ смеет трогать SEXP (то же правило,
что в drogonR-мосте). Всё R-взаимодействие (загрузка модели, конфиг) — до старта
потока. Токены наружу идут через drogonR send() (C-путь), НЕ через R.
**Риск №2:** динамический prefill новых seq в общий батч — проверить, что
`llama_decode` корректно микширует prefill (много токенов одного seq) и decode
(1 токен на seq) в одном вызове, либо разнести prefill/decode по разным шагам.
**Риск №3:** объём работы — Шаги 1-3 это дописка в РАБОЧИЙ `r_llama_generate_batch`;
беречь его синхронный путь (нужен для llama_generate_batch API) — не ломать, а
добавить событийный вариант рядом (флаг/отдельная функция).

**Источники:** `src/r_llama_interface.cpp:1037-1272` (батч-декодер),
`R/serve.R` (серверный слой), `drogonR/inst/include/drogonR.h`
(drogonr_ws_handler_t ABI), `drogonR/src/ws_session.cpp` (образец моста
IO-тред↔очередь). Транспорт готов — работа целиком в llamaR.


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
