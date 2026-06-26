# llamaR 0.2.5

## Engine upgrade (upstream llama.cpp master)

* The bundled llama.cpp engine was migrated to the current upstream `master`
  class-based architecture: per-architecture model classes (`llama_model_X`),
  the refactored `create_tensor`/buffer-type loader, and the split-out `models/*`
  graph builders. This brings 128 model architectures and unlocks the newest
  ones while preserving llamaR's own patches (grammar/CRAN redirects, chat/tool
  layer, delta-net).
* New architecture support including **Qwen3.5** (`qwen35` / `qwen35moe`,
  hybrid gated delta-net + attention) — loads and generates from R and over the
  Anthropic/OpenAI servers.

## Multimodal (vision)

* New `mtmd` subsystem (vendored from upstream) for vision/OCR models:
  `llama_mtmd_load()` loads a multimodal projector (mmproj GGUF),
  `llama_image_load()` reads an image, and `llama_image_eval()` feeds
  text + image chunks through the context for generation. Capability checks via
  `llama_mtmd_support_vision()` / `llama_mtmd_support_audio()`.

## Tool calling

* `llama_chat_build()` — apply a model's chat template (Jinja path) to messages
  plus tool definitions, returning the prompt, the grammar that constrains tool
  calls, the format id, lazy-grammar triggers, and the serialized PEG parser
  arena. Backed by the vendored llama.cpp `common/` chat layer.
* `llama_chat_parse()` — parse raw model output back into content,
  `reasoning_content`, and structured tool calls (name, arguments, id),
  including PEG-based formats (Mistral/Qwen).
* `llama_generate()` / `llama_gen_begin()` gain `trigger_patterns` /
  `trigger_tokens` arguments and use `llama_sampler_init_grammar_lazy_patterns`
  for lazy grammars (constrain only after a trigger such as `[TOOL_CALLS]`).

## Anthropic Messages API server for Claude Code

* `llama_serve_anthropic()` — serve a local GGUF model over an Anthropic
  Messages API-compatible HTTP API (`POST /v1/messages`, `GET /v1/models`,
  streaming and blocking, with tool use). Point Claude Code at it with
  `ANTHROPIC_BASE_URL`. Requires the optional `drogonR` package.
* New `enable_thinking` argument (default `FALSE`) toggles the chat template's
  reasoning mode for hybrid thinking models (Qwen3.5, etc.). Disabled by default
  so Claude Code gets direct answers and fast tool calls; set `TRUE` (and raise
  `max_tokens`) to keep the reasoning trace.

---

# llamaR 0.2.4

## Streaming generation

* `llama_gen_begin()` / `llama_gen_next()` / `llama_gen_end()` — token-by-token generation matching `llama_generate()` output, with valid-UTF-8 chunks.

## OpenAI-compatible server

* `llama_serve_openai()` — serve a local GGUF model over an OpenAI-compatible HTTP API (`/v1/models`, `/v1/chat/completions`, streaming and blocking) via the optional `drogonR` package.

## ellmer integration

* `chat_llamar()` — returns an `ellmer::Chat` backed by a local model, connecting to a running server (`base_url=`) or spawning one (`model_path=`); `chat_llamar_stop()` stops a spawned server.

## Bug fixes

* Long prompts no longer abort: prefill is now split into `llama_n_batch()`-sized chunks (was `GGML_ASSERT(n_tokens_all <= cparams.n_batch)`).

---

# llamaR 0.2.3

## Context getters

* `llama_n_ctx_seq()` — per-sequence context window size.
* `llama_n_batch()` — logical batch size (max tokens per `llama_decode` call).
* `llama_n_ubatch()` — physical micro-batch size.
* `llama_n_seq_max()` — maximum number of concurrent sequences.
* `llama_n_threads()` / `llama_n_threads_batch()` — read back thread counts set via `llama_set_threads()`.
* `llama_pooling_type()` — pooling type of the context as a string (`"none"`, `"mean"`, `"cls"`, `"last"`, `"rank"`).

## Bug fixes

* Fixed macOS compilation error: removed `fflush` macro from `r_llama_compat.h`
  that broke `std::fflush` in `<fstream>` (Apple clang / libc++).

## Logits

* `llama_get_logits_ith()` — logit vector for a specific token position in the last decoded batch. Supports negative indexing (`-1` = last token).

---

# llamaR 0.2.2

## ragnar integration

* `embed_llamar()` — high-level embedding provider compatible with
  `ragnar_store_create(embed = ...)`. Supports partial application (lazy model
  loading), direct call returning a matrix, and data.frame input. L2
  normalization on by default.

## Batch embeddings

* `llama_embed_batch()` — embed multiple texts in one call. Uses true pooled
  batch decode (`llama_get_embeddings_seq`) for embedding models, with automatic
  fallback to sequential last-token decode for generative models.
* `llama_get_embeddings_ith()` — get embedding vector for the i-th token
  (supports negative indexing).
* `llama_get_embeddings_seq()` — get pooled embedding for a sequence ID.

## Context embedding mode

* `llama_new_context()` gains `embedding` parameter. When `TRUE`, sets
  `cparams.embeddings = true` and disables causal attention at creation time.
  `llama_embed_batch()` uses this flag to choose the optimal code path.

## Backend & device selection

* `llama_load_model()` gains `devices` parameter for explicit backend selection.
  Accepts device names from `llama_backend_devices()`, type keywords (`"cpu"`,
  `"gpu"`), or numeric indices. Multiple devices enable multi-GPU split.
* `llama_backend_devices()` — list all available compute devices (CPU, GPU,
  iGPU, accelerator) as a data.frame.

## Hardware & system

* `llama_numa_init()` — NUMA optimization with strategies: disabled, distribute,
  isolate, numactl, mirror.
* `llama_time_us()` — current time in microseconds.

## Tests

* 40+ new test blocks covering all new functions.
* Total: 143 passing, 4 expected skips.

---

# llamaR 0.2.1

## New functions

* `llama_token_to_piece()` — convert a single token ID to its text piece.
* `llama_encode()` — run the encoder pass for encoder-decoder models (e.g. T5, BART).
* `llama_batch_init()` / `llama_batch_free()` — low-level batch allocation and release
  with automatic GC finalizer.

## Bug fixes

* Fixed compilation failure on macOS with Apple clang 17 / Xcode 16.4:
  removed `extern "C"` block wrapping `#include <R.h>` in `r_llama_compat.h`
  (C++ templates cannot appear inside `extern "C"` linkage).
* Fixed macro conflict between `Rinternals.h` `#define length(x)` and
  `std::codecvt::length()` in `r_llama_interface.cpp`:
  C++ standard headers are now included before R headers, followed by
  `#undef length`.

## Tests

* Added 9 new test blocks covering `llama_token_to_piece`, `llama_batch_init`,
  `llama_batch_free`, and `llama_encode`, including GPU context variants.
* Total: 103 passing, 4 expected skips.

---

# llamaR 0.2.0

## Hugging Face integration

### New functions
* `llama_hf_list()` — list GGUF files in a Hugging Face repository.
* `llama_hf_download()` — download a GGUF model with local caching.
  Supports exact filename, glob pattern, or Ollama-style tag selection.
* `llama_load_model_hf()` — download and load a model in one step.
* `llama_hf_cache_dir()` — get the cache directory path.
* `llama_hf_cache_info()` — inspect cached models.
* `llama_hf_cache_clear()` — clear the model cache.

### Dependencies
* Added `jsonlite` and `utils` to `Imports`.

---

# llamaR 0.1.3

## GPU and build system improvements

### Vulkan GPU support on Windows
* Added Vulkan linking support to `configure.win` and `Makevars.win.in`.
* Windows builds now link with Vulkan when `ggmlR` is built with GPU support.

### CRAN compliance
* Added `exit()` / `_Exit()` overrides to `r_llama_compat.h` to prevent
  process termination (redirects to `Rf_error()`).

### Dependencies
* Requires `ggmlR` >= 0.5.4.
* Bumped minimum R version to 4.1.0 (matches `ggmlR`).

### DESCRIPTION
* Updated description to mention Vulkan GPU support via `ggmlR`.

---

# llamaR 0.1.2

## CRAN compliance fixes

### Documentation
* Expanded all acronyms in DESCRIPTION (LLMs, GPU).
* Added detailed `\value` tags to all exported functions describing
  return class, structure, and meaning.
* Replaced `\dontrun{}` with `\donttest{}` in all examples.

### DESCRIPTION
* Added Georgi Gerganov as copyright holder (`cph`) for bundled
  'llama.cpp' code.

### Packaging
* Included `NEWS.md` in the package tarball (removed from `.Rbuildignore`).
* Created `cran-comments.md`.
* Cleaned up duplicate entries in `.Rbuildignore`.

---

# llamaR 0.1.1

## R interface — first working release

Full LLM inference cycle is now available from R:

* `llama_load_model()` / `llama_free_model()` — load and free GGUF models
* `llama_new_context()` / `llama_free_context()` — context management
* `llama_tokenize()` / `llama_detokenize()` — tokenization and detokenization
* `llama_generate()` — text generation with temperature, top_k, top_p, greedy support
* `llama_embeddings()` — embedding extraction
* `llama_model_info()` — model metadata

### Memory management

Model and context are wrapped as ExternalPtr with automatic GC finalizers.
The context holds a reference to the model ExternalPtr, preventing premature
collection.

### Generation internals

`llama_generate()` runs the full pipeline in a single C++ call: prompt
tokenization → encode → autoregressive decode loop with a sampler chain →
detokenization of generated tokens.

### Tests

19 assertions across 7 test blocks, all passing.

---

# llamaR 0.1.0

## Initial Release

* Basic package structure with llama.cpp integration
* Links against `libggml.a` from ggmlR package
* Includes all llama.cpp model implementations (~100 architectures)
* Vulkan GPU support (optional)

### Dependencies

* Requires ggmlR >= 0.5.1 for static library export

### Known Limitations

* `ggml_build_forward_select` replaced with simplified branch selection
