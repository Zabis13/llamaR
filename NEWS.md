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
