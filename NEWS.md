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
