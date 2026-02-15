# llamaR

R interface to [llama.cpp](https://github.com/ggml-org/llama.cpp) for running large language models locally.

## Features

- Load GGUF format models
- Text generation with configurable sampling parameters
- Tokenization and detokenization
- Embedding extraction
- Chat template support (ChatML, Llama, Mistral, etc.)
- LoRA adapters
- Optional GPU acceleration via Vulkan (auto-detected on Linux and Windows)

## Installation

### Dependencies

Requires [ggmlR](https://github.com/Zabis13/ggmlR) >= 0.5.4:

```r
# Install ggmlR first
remotes::install_github("Zabis13/ggmlR")

# Then llamaR
remotes::install_github("Zabis13/llamaR")
```

### System Requirements

- R >= 4.1.0
- C++17 compiler
- GNU make

## Quick Start

```r
library(llamaR)

# Load model
model <- llama_load_model("path/to/model.gguf")

# Create context
ctx <- llama_new_context(model, n_ctx = 2048L, n_threads = 8L)

# Generate text
result <- llama_generate(ctx, "Once upon a time", max_new_tokens = 100L)
cat(result)

# Free resources (optional, GC handles this automatically)
llama_free_context(ctx)
llama_free_model(model)
```

## Downloading Models from Hugging Face

Download GGUF models directly from Hugging Face with automatic caching:

```r
library(llamaR)

# List available GGUF files in a repository
files <- llama_hf_list("TheBloke/Llama-2-7B-GGUF")
print(files)

# Download a specific quantization
path <- llama_hf_download("TheBloke/Llama-2-7B-GGUF", pattern = "*q4_k_m*")

# Or download and load in one step
model <- llama_load_model_hf("TheBloke/Llama-2-7B-GGUF",
                              pattern = "*q4_k_m*",
                              n_gpu_layers = -1L)

# Manage cache
llama_hf_cache_info()
llama_hf_cache_clear()
```

For private repositories, set the `HF_TOKEN` environment variable or pass `token` directly.

## Usage

### Loading Models

```r
# CPU only
model <- llama_load_model("model.gguf")

# With GPU acceleration (all layers)
model <- llama_load_model("model.gguf", n_gpu_layers = -1L)

# Partial GPU offload (first 20 layers)
model <- llama_load_model("model.gguf", n_gpu_layers = 20L)

# Check GPU availability
if (llama_supports_gpu()) {
  message("GPU available")
}
```

### Model Information

```r
info <- llama_model_info(model)
cat("Model:", info$desc, "\n")
cat("Layers:", info$n_layer, "\n")
cat("Context:", info$n_ctx_train, "\n")
cat("Embedding size:", info$n_embd, "\n")
```

### Text Generation

```r
ctx <- llama_new_context(model, n_ctx = 4096L)

# Basic generation
result <- llama_generate(ctx, "The meaning of life is")

# Greedy decoding (deterministic)
result <- llama_generate(ctx, "2 + 2 =", temp = 0)

# Creative output
result <- llama_generate(ctx,
  prompt = "Write a haiku about R:",
  max_new_tokens = 50L,
  temp = 1.0,
  top_p = 0.95,
  top_k = 40L
)
```

### Chat Models

```r
model <- llama_load_model("llama-3.2-instruct.gguf", n_gpu_layers = -1L)
ctx <- llama_new_context(model)

# Get template from model
tmpl <- llama_chat_template(model)

# Build conversation
messages <- list(
  list(role = "system", content = "You are a helpful assistant."),
  list(role = "user", content = "What is R?")
)

# Apply template
prompt <- llama_chat_apply_template(messages, template = tmpl)

# Generate response
response <- llama_generate(ctx, prompt, max_new_tokens = 200L)
cat(response)
```

### Tokenization

```r
# Text -> tokens
tokens <- llama_tokenize(ctx, "Hello, world!")

# Tokens -> text
text <- llama_detokenize(ctx, tokens)
```

### Embeddings

```r
emb1 <- llama_embeddings(ctx, "machine learning")
emb2 <- llama_embeddings(ctx, "artificial intelligence")

# Cosine similarity
similarity <- sum(emb1 * emb2) / (sqrt(sum(emb1^2)) * sqrt(sum(emb2^2)))
cat("Similarity:", similarity, "\n")
```

### LoRA Adapters

```r
model <- llama_load_model("base-model.gguf")
ctx <- llama_new_context(model)

# Load and apply adapter
lora <- llama_lora_load(model, "fine-tuned.gguf")
llama_lora_apply(ctx, lora, scale = 1.0)

# Generate with LoRA
result <- llama_generate(ctx, "prompt")

# Remove all LoRA adapters
llama_lora_clear(ctx)
```

### Verbosity Control

```r
# Levels: 0 = silent, 1 = errors only, 2 = normal, 3 = verbose
llama_set_verbosity(0)  # Suppress all output
llama_set_verbosity(3)  # Debug mode
```

## Supported Models

Supports all llama.cpp compatible architectures (100+), including:

- LLaMA 1/2/3
- Mistral / Mixtral
- Qwen / Qwen2
- Gemma / Gemma 2
- Phi-2 / Phi-3
- DeepSeek
- Command-R
- and many more

Models must be in GGUF format. Convert models using llama.cpp tools.

## License

MIT

## Author

Yuri Baramykov

## Links

- [GitHub](https://github.com/Zabis13/llamaR)
- [Bug Reports](https://github.com/Zabis13/llamaR/issues)
- [ggmlR](https://github.com/Zabis13/ggmlR) - tensor operations dependency
- [llama.cpp](https://github.com/ggml-org/llama.cpp) - inference backend
