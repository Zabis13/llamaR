# llamaR

R interface to [llama.cpp](https://github.com/ggml-org/llama.cpp) for running local inference of large language models (LLMs) directly from R.

The package supports GPU acceleration via Vulkan, and automatically falls back to CPU when no GPU is available.

## Key Features

- Load and unload models in GGUF format (`llama_load_model`, `llama_free_model`)
- Create and free contexts (`llama_new_context`, `llama_free_context`)
- Tokenization, detokenization and text generation (`llama_tokenize`, `llama_detokenize`, `llama_generate`)
- Streaming (token-by-token) generation (`llama_gen_begin`, `llama_gen_next`, `llama_gen_end`)
- OpenAI-compatible HTTP server for local models (`llama_serve_openai`) — connect OpenCode, ellmer, the `openai` SDK, etc.
- Anthropic Messages API server (`llama_serve_anthropic`) with tool use — run Claude Code against a local model via `ANTHROPIC_BASE_URL`.
- ellmer `Chat` objects backed by local models (`chat_llamar`) — use the ellmer/ragnar toolchain against local inference
- Tool-aware chat layer (`llama_chat_build`, `llama_chat_parse`) — apply a model's template with tool definitions and parse tool calls back out, with lazy-grammar constraining
- Multimodal (vision) inference (`llama_mtmd_load`, `llama_image_load`, `llama_image_eval`) — vision/OCR models via an mmproj projector
- Embedding extraction: single (`llama_embeddings`), batch (`llama_embed_batch`), ragnar-compatible (`embed_llamar`)
- Hugging Face integration: download and cache models (`llama_hf_download`, `llama_load_model_hf`, etc.)
- Encoder-decoder model support (T5, BART) via `llama_encode`
- Explicit backend/device selection and multi-GPU split (`llama_load_model(devices = ...)`)
- NUMA optimization (`llama_numa_init`)

## GPU and CPU Support

The package uses [ggmlR](https://github.com/Zabis13/ggmlR) as the low-level backend.
If ggmlR was built with Vulkan support enabled, llamaR automatically uses the GPU for computation.
On systems without a GPU, all code runs on CPU with no additional configuration required.

### How Vulkan linking works

Vulkan support is compiled entirely within ggmlR — llamaR does not compile any Vulkan code itself.
However, since llamaR links against `libggml.a` (from ggmlR) using `--whole-archive`, the Vulkan
symbols (e.g. `vkCmdCopyBuffer`, `vkGetInstanceProcAddr`) need to be resolved at link time.

The llamaR `configure` script handles this automatically:
- **Linux**: checks `pkg-config --exists vulkan` and adds `-lvulkan` to the linker flags
- **Windows**: checks for the `VULKAN_SDK` environment variable and adds `-lvulkan-1`

If Vulkan is not found on the system, the build proceeds without it — the Vulkan backend
in `libggml.a` will simply remain unused, and inference runs on CPU only.

## Performance

Measured on AMD Ryzen 5 5600 + AMD RX 9070, model Ministral-3-3B-Instruct-2512-Q8_0, 50 tokens, avg of 3 runs:

| Backend | Speed (tokens/sec) | Speedup |
|---|---:|---:|
| CPU (8 threads) | 8.5 | 1.0x |
| GPU (Vulkan) | 108.0 | 12.7x |

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

### Full Linux setup (Ubuntu) — install and run Claude Code

End-to-end instructions: R, the Vulkan toolchain, ggmlR/llamaR, Claude Code, a
model, and the Anthropic server. Tested on Ubuntu 22.04 (Jammy) and 24.04
(Noble).

**1. R and the Vulkan runtime:**

```bash
sudo apt install -y r-base
sudo apt install vulkan-tools libvulkan-dev
```

**2. The `glslc` shader compiler** (needed to build ggmlR's Vulkan backend):

```bash
# Ubuntu 24.04 (Noble)
sudo add-apt-repository universe
sudo apt update
sudo apt install glslc
```

```bash
# Ubuntu 22.04 (Jammy) — install the LunarG Vulkan SDK instead
wget -qO- https://packages.lunarg.com/lunarg-signing-key-pub.asc | \
  sudo tee /etc/apt/trusted.gpg.d/lunarg.asc

sudo wget -qO /etc/apt/sources.list.d/lunarg-vulkan-jammy.list \
  https://packages.lunarg.com/vulkan/lunarg-vulkan-jammy.list

sudo apt update
sudo apt install -y vulkan-sdk
```

Verify the GPU is visible to Vulkan:

```bash
vulkaninfo --summary
```

**3. ggmlR** (the tensor backend, built with SIMD):

```bash
sudo Rscript -e 'install.packages("ggmlR", configure.args = "--with-simd")'

Rscript -e 'library(ggmlR)
ggml_vulkan_status()'
```

**4. llamaR** (plus `drogonR` for the HTTP servers):

```bash
sudo Rscript -e 'install.packages("llamaR")'
sudo Rscript -e 'install.packages("drogonR")'
```

Or install the development version from GitHub:

```bash
sudo apt install -y libcurl4-openssl-dev libssl-dev libgit2-dev
sudo Rscript -e 'install.packages("remotes")'
sudo Rscript -e 'remotes::install_github("Zabis13/llamaR")'
sudo Rscript -e 'install.packages("drogonR")'
```

**5. Claude Code:**

```bash
sudo apt install npm
npm install -g @anthropic-ai/claude-code
```

**6. Download a model** from Hugging Face:

```bash
pip install -U "huggingface_hub[cli]"
mkdir -p ~/llm_models

hf download unsloth/Qwen3.5-9B-GGUF \
  Qwen3.5-9B-UD-Q6_K_XL.gguf \
  --local-dir ~/llm_models
```

```
✓ Downloaded
  path: /home/user/llm_models/Qwen3.5-9B-UD-Q6_K_XL.gguf
```

**7. Start the llamaR Anthropic server and run Claude Code.** Start the server:

```bash
Rscript -e "llamaR::llama_serve_anthropic('/home/user/llm_models/Qwen3.5-9B-UD-Q6_K_XL.gguf', port=11435L)"
```

Then, in another shell, point Claude Code at it and launch:

```bash
unset ANTHROPIC_API_KEY
export ANTHROPIC_BASE_URL=http://127.0.0.1:11435
export ANTHROPIC_AUTH_TOKEN=sk-local
export CLAUDE_CODE_SKIP_PREFLIGHT_CHECK=1
export ANTHROPIC_MODEL=Qwen3.5-9B-UD-Q6_K_XL
export ANTHROPIC_SMALL_FAST_MODEL=Qwen3.5-9B-UD-Q6_K_XL
claude
```

Or use the bundled launcher, which starts the server, waits for it, and runs
Claude Code in one step:

```bash
SCRIPT=$(Rscript -e "cat(system.file('examples/claude_code_launcher.sh', package='llamaR'))")

VISION_MODEL= MMPROJ= \
  bash "$SCRIPT" \
  /home/user/llm_models/Qwen3.5-9B-UD-Q6_K_XL.gguf 11435
```

> **Multi-GPU note:** on a host with several GPUs the model is split across all
> of them by default (`split_mode = "layer"`), which can hang on the Vulkan
> backend. If the model fits in a single card's VRAM, pin it to one GPU with
> `SPLIT_MODE=none` (launcher) or `split_mode = "none"`
> (`llama_serve_anthropic()`).

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

## Vignettes

Two guides walk through the package in depth:

- **Getting Started** — loading models, generation, chat templates,
  tokenization, and embeddings.
- **Chat and Agents** — `chat_llamar()`, the OpenAI and Anthropic servers
  (OpenCode / ellmer / Claude Code), tool calling, and retrieval-augmented
  chat with ragnar.

```r
browseVignettes("llamaR")
vignette("getting-started", package = "llamaR")
vignette("chat-and-agents", package = "llamaR")
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

# Explicit device selection (see llama_backend_devices())
model <- llama_load_model("model.gguf", n_gpu_layers = -1L, devices = "Vulkan0")

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

### Streaming Generation

Pull tokens one at a time instead of waiting for the full result — useful for
live output or feeding a stream. Concatenating every chunk reproduces the
`llama_generate()` result for the same seed.

```r
st <- llama_gen_begin(ctx, "Once upon a time", max_new_tokens = 100L)
repeat {
  chunk <- llama_gen_next(st)   # next piece of text, or NULL when done
  if (is.null(chunk)) break
  cat(chunk)
}
cat(llama_gen_end(st))          # flush any held-back trailing bytes
```

### OpenAI-Compatible Server

Serve a local model over an OpenAI-compatible HTTP API so any OpenAI client can
talk to it. Requires the optional `drogonR` package
(`install.packages("drogonR")`).

```r
# Blocks, serving GET /v1/models and POST /v1/chat/completions
# (both blocking and stream = true). Default port 11434.
llama_serve_openai("model.gguf", port = 11434L)
```

Point any OpenAI client at `http://127.0.0.1:11434/v1`, e.g.:

```bash
curl http://127.0.0.1:11434/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"model","messages":[{"role":"user","content":"Hello"}]}'
```

A runnable example lives at `inst/examples/serve_openai.R`:

```bash
# Just serve:  args are <model.gguf> [port] [n_ctx]
Rscript inst/examples/serve_openai.R model.gguf 11434 16384

# Or self-test both endpoints end-to-end (needs callr + curl):
Rscript inst/examples/serve_openai.R model.gguf --selftest
```

To connect [OpenCode](https://opencode.ai), add an OpenAI-compatible provider in
`opencode.json` (see the one in this repo) pointing `baseURL` at
`http://127.0.0.1:11434/v1`, with the model id matching what `/v1/models`
reports.

### Serving an Anthropic API for Claude Code

`llama_serve_anthropic()` exposes an Anthropic Messages API-compatible endpoint
(`POST /v1/messages`, blocking and streaming, with tool use), so
[Claude Code](https://claude.com/claude-code) can run against a local model.
Also requires the optional `drogonR` package.

```r
# Blocks, serving POST /v1/messages and GET /v1/models. Default port 11435.
# Use a tool-calling-capable model (Qwen, Llama-3.x, Mistral/Mixtral, …).
llama_serve_anthropic("Qwen3.5-9B-UD-Q6_K_XL.gguf", port = 11435L)
```

For hybrid *thinking* models (Qwen3.5, etc.) the server keeps `enable_thinking =
FALSE` by default: otherwise the model can spend its whole token budget inside a
`<think>` block and never reach the answer, leaving Claude Code with an empty
reply. Set `enable_thinking = TRUE` (and raise `max_tokens`) if you want the
reasoning trace.

Then point Claude Code at it with environment variables and launch as usual:

```bash
ANTHROPIC_BASE_URL=http://127.0.0.1:11435 \
ANTHROPIC_API_KEY=sk-local \
claude
```

`ANTHROPIC_API_KEY` only needs to be non-empty (the server does not check it).
Tool calling works: tools sent by Claude Code are passed through the chat
template, generation is grammar-constrained, and the model's output is parsed
back into `tool_use` blocks. A curl smoke-test (non-stream, tool, and SSE) lives
at `inst/examples/serve_anthropic_test.sh`:

```bash
bash inst/examples/serve_anthropic_test.sh model.gguf 11435
```

#### Vision (images) with a second model

Give the server a vision model and its projector to handle images that Claude
Code sends (e.g. screenshots). The server then runs a *caption-then-reason*
pipeline: the vision model (e.g. Qwen2-VL) describes each image — focused on the
user's question — and that description is handed to the main text model, which
reasons over it and answers as usual (with tools and streaming). The user sees
only the text model's reply; set `vision_debug = TRUE` to log captions.

```r
llama_serve_anthropic(
  "Qwen3.5-9B-UD-Q6_K_XL.gguf", port = 11435L,
  vision_model_path = "Qwen2-VL-2B-Instruct-Q8_0.gguf",
  mmproj_path       = "mmproj-Qwen2-VL-2B-Instruct-Q8_0.gguf",
  vision_n_ctx      = 8192L)        # small vision context keeps both in VRAM
```

Both models stay loaded; image requests use the vision model for the caption,
everything else stays on the text model. Without `vision_model_path` the server
is text-only and image blocks are dropped (unchanged behaviour). The launcher
`inst/examples/claude_code_launcher.sh` enables this by default via the
`VISION_MODEL` / `MMPROJ` environment variables, and
`inst/examples/serve_anthropic_vision.R` shows the raw request format plus a
`--selftest` that sends a base64 image over curl (no Claude Code needed).

### Chatting via ellmer

`chat_llamar()` returns an [ellmer](https://ellmer.tidyverse.org/) `Chat`
object backed by a local model, so the whole ellmer / ragnar toolchain works
against local inference. Requires the optional `ellmer` package (and `callr`
when spawning a server).

```r
# Spawn a server for this model and chat with it; the background process is
# tied to the returned object (stop it with chat_llamar_stop(), or let GC).
chat <- chat_llamar(model_path = "model.gguf")
chat$chat("Why is the sky blue?")
chat_llamar_stop(chat)

# Or connect to a server you already started with llama_serve_openai():
chat <- chat_llamar(base_url = "http://127.0.0.1:11434/v1")
chat$chat("Hello!")
```

It wraps `ellmer::chat_vllm()`, talking to the server's
`/v1/chat/completions` endpoint.

### Tokenization

```r
# Text -> tokens
tokens <- llama_tokenize(ctx, "Hello, world!")

# Tokens -> text
text <- llama_detokenize(ctx, tokens)
```

### Embeddings

```r
# Single text embedding
emb1 <- llama_embeddings(ctx, "machine learning")
emb2 <- llama_embeddings(ctx, "artificial intelligence")

# Cosine similarity
similarity <- sum(emb1 * emb2) / (sqrt(sum(emb1^2)) * sqrt(sum(emb2^2)))
cat("Similarity:", similarity, "\n")

# Batch embeddings (matrix output)
ctx <- llama_new_context(model, n_ctx = 512L, embedding = TRUE)
mat <- llama_embed_batch(ctx, c("hello world", "foo bar", "test"))
# mat is a 3 x n_embd matrix
```

### ragnar Integration

Use `embed_llamar()` as an embedding provider for [ragnar](https://ragnar.tidyverse.org/):

```r
library(ragnar)

# Create store with local embedding model
store <- ragnar_store_create(
  "my_store",
  embed = embed_llamar(
    model = "nomic-embed-text-v1.5.Q8_0.gguf",
    n_gpu_layers = -1,
    embedding = TRUE
  )
)

# Insert and retrieve documents as usual
ragnar_store_insert(store, documents)
ragnar_retrieve(store, "search query")
```

### Backend and Device Selection

```r
# List available devices
llama_backend_devices()
#>         name           description  type
#> 1 CPU        CPU (threads: 16)      cpu
#> 2 Vulkan0    NVIDIA GeForce RTX 4090 gpu

# Load model on specific device
model <- llama_load_model("model.gguf", n_gpu_layers = -1, devices = "Vulkan0")

# CPU-only (even if GPU is available)
model <- llama_load_model("model.gguf", devices = "cpu")

# Multi-GPU split
model <- llama_load_model("model.gguf", n_gpu_layers = -1,
                          devices = c("Vulkan0", "Vulkan1"))
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

Supports all llama.cpp compatible architectures (128, upstream `master`),
including:

- LLaMA 1/2/3
- Mistral / Mixtral
- Qwen / Qwen2 / Qwen3 / **Qwen3.5** (`qwen35` / `qwen35moe`)
- Gemma / Gemma 2 / Gemma 3
- Phi-2 / Phi-3 / Phi-4
- DeepSeek
- Command-R
- Vision/OCR models (via the `mtmd` projector)
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
