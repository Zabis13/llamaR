#' Set logging verbosity level
#'
#' Controls how much diagnostic output is printed during model loading and inference.
#'
#' @param level Integer verbosity level:
#'   - 0: Silent (no output)
#'   - 1: Errors only (default)
#'   - 2: Normal (warnings and info)
#'   - 3: Verbose (all debug messages)
#' @return Invisible NULL
#' @export
#' @examples
#' # Suppress all output
#' llama_set_verbosity(0)
#'
#' # Show only errors
#' llama_set_verbosity(1)
#'
#' # Verbose output for debugging
#' llama_set_verbosity(3)
llama_set_verbosity <- function(level) {
    stopifnot(is.numeric(level), length(level) == 1)
    .Call("r_llama_set_verbosity", as.integer(level))
    invisible(NULL)
}

#' Get current verbosity level
#'
#' @return Integer verbosity level (0-3)
#' @export
#' @examples
#' llama_get_verbosity()
llama_get_verbosity <- function() {
    .Call("r_llama_get_verbosity")
}

#' Check whether GPU offloading is available
#'
#' Returns `TRUE` if at least one GPU backend (e.g. Vulkan) was detected at
#' runtime. Use the result to decide whether to pass `n_gpu_layers != 0`
#' to [llama_load_model].
#'
#' @return Logical scalar
#' @export
#' @examples
#' if (llama_supports_gpu()) {
#'   message("GPU available, will use Vulkan backend")
#' } else {
#'   message("GPU not available, using CPU only")
#' }
llama_supports_gpu <- function() {
    .Call("r_llama_supports_gpu")
}

#' Load a GGUF model file
#'
#' @param path Path to the .gguf model file
#' @param n_gpu_layers Number of layers to offload to GPU (0 = CPU only, -1 = all)
#' @return An opaque model handle (ExternalPtr). Freed automatically on GC or via [llama_free_model].
#' @export
#' @examples
#' \dontrun{
#' # Load model on CPU only
#' model <- llama_load_model("model.gguf")
#'
#' # Load model with all layers on GPU
#' model <- llama_load_model("model.gguf", n_gpu_layers = -1L)
#'
#' # Load model with first 10 layers on GPU
#' model <- llama_load_model("model.gguf", n_gpu_layers = 10L)
#' }
llama_load_model <- function(path, n_gpu_layers = 0L) {
    stopifnot(is.character(path), length(path) == 1)
    if (!file.exists(path)) stop("llamaR: model file does not exist: ", path)
    .Call("r_llama_load_model", path, as.integer(n_gpu_layers))
}

#' Free a loaded model
#'
#' @param model Model handle returned by [llama_load_model]
#' @return Invisible NULL
#' @export
#' @examples
#' \dontrun{
#' model <- llama_load_model("model.gguf")
#' # ... use model ...
#' llama_free_model(model)
#' }
llama_free_model <- function(model) {
    .Call("r_llama_free_model", model)
    invisible(NULL)
}

#' Get model metadata
#'
#' @param model Model handle returned by [llama_load_model]
#' @return A named list with fields:
#'   - `n_ctx_train`: context size the model was trained with
#'   - `n_embd`: embedding dimension
#'   - `n_vocab`: vocabulary size
#'   - `n_layer`: number of layers
#'   - `n_head`: number of attention heads
#'   - `desc`: human-readable model description string
#' @export
#' @examples
#' \dontrun{
#' model <- llama_load_model("model.gguf")
#' info <- llama_model_info(model)
#' cat("Model:", info$desc, "\n")
#' cat("Layers:", info$n_layer, "\n")
#' cat("Context:", info$n_ctx_train, "\n")
#' }
llama_model_info <- function(model) {
    .Call("r_llama_model_info", model)
}

#' Create an inference context
#'
#' @param model Model handle returned by [llama_load_model]
#' @param n_ctx Context window size (number of tokens). 0 means use the model's trained value.
#' @param n_threads Number of CPU threads to use
#' @return An opaque context handle (ExternalPtr). Freed automatically on GC or via [llama_free_context].
#' @export
#' @examples
#' \dontrun{
#' model <- llama_load_model("model.gguf")
#' ctx <- llama_new_context(model, n_ctx = 4096L, n_threads = 8L)
#' # ... use context for generation ...
#' llama_free_context(ctx)
#' llama_free_model(model)
#' }
llama_new_context <- function(model, n_ctx = 2048L, n_threads = 4L) {
    .Call("r_llama_new_context", model, as.integer(n_ctx), as.integer(n_threads))
}

#' Free an inference context
#'
#' @param ctx Context handle returned by [llama_new_context]
#' @return Invisible NULL
#' @export
#' @examples
#' \dontrun{
#' model <- llama_load_model("model.gguf")
#' ctx <- llama_new_context(model)
#' # ... use context ...
#' llama_free_context(ctx)
#' }
llama_free_context <- function(ctx) {
    .Call("r_llama_free_context", ctx)
    invisible(NULL)
}

#' Tokenize text into token IDs
#'
#' @param ctx Context handle returned by [llama_new_context]
#' @param text Character string to tokenize
#' @param add_special Whether to add special tokens (BOS/EOS) as configured by the model
#' @return Integer vector of token IDs
#' @export
#' @examples
#' \dontrun{
#' model <- llama_load_model("model.gguf")
#' ctx <- llama_new_context(model)
#'
#' tokens <- llama_tokenize(ctx, "Hello, world!")
#' print(tokens)
#' # [1] 1 15043 29892 3186 29991
#'
#' # Without special tokens
#' tokens <- llama_tokenize(ctx, "Hello", add_special = FALSE)
#' }
llama_tokenize <- function(ctx, text, add_special = TRUE) {
    stopifnot(is.character(text), length(text) == 1)
    .Call("r_llama_tokenize", ctx, text, as.logical(add_special))
}

#' Detokenize token IDs back to text
#'
#' @param ctx Context handle returned by [llama_new_context]
#' @param tokens Integer vector of token IDs (as returned by [llama_tokenize])
#' @return Character string
#' @export
#' @examples
#' \dontrun{
#' model <- llama_load_model("model.gguf")
#' ctx <- llama_new_context(model)
#'
#' # Round-trip: text -> tokens -> text
#' original <- "Hello, world!"
#' tokens <- llama_tokenize(ctx, original, add_special = FALSE)
#' restored <- llama_detokenize(ctx, tokens)
#' identical(original, restored)  # TRUE
#' }
llama_detokenize <- function(ctx, tokens) {
    stopifnot(is.integer(tokens))
    .Call("r_llama_detokenize", ctx, tokens)
}

#' Generate text from a prompt
#'
#' Tokenizes the prompt, runs the full autoregressive decode loop with sampling,
#' and returns the generated text (excluding the original prompt).
#'
#' @param ctx Context handle returned by [llama_new_context]
#' @param prompt Character string prompt
#' @param max_new_tokens Maximum number of tokens to generate
#' @param temp Sampling temperature. 0 = greedy decoding.
#' @param top_k Top-K filtering (0 = disabled)
#' @param top_p Top-P (nucleus) filtering (1.0 = disabled)
#' @param seed Random seed for sampling
#' @return Character string with generated text
#' @export
#' @examples
#' \dontrun{
#' model <- llama_load_model("model.gguf", n_gpu_layers = -1L)
#' ctx <- llama_new_context(model, n_ctx = 2048L)
#'
#' # Basic generation
#' result <- llama_generate(ctx, "Once upon a time")
#' cat(result)
#'
#' # Greedy decoding (deterministic)
#' result <- llama_generate(ctx, "The answer is", temp = 0)
#'
#' # More creative output
#' result <- llama_generate(ctx, "Write a poem about R:",
#'                          max_new_tokens = 100L,
#'                          temp = 1.0, top_p = 0.95)
#' }
llama_generate <- function(ctx, prompt, max_new_tokens = 256L,
                           temp = 0.8, top_k = 50L, top_p = 0.9, seed = 42L) {
    stopifnot(is.character(prompt), length(prompt) == 1)
    .Call("r_llama_generate", ctx, prompt,
          as.integer(max_new_tokens), as.double(temp),
          as.integer(top_k), as.double(top_p), as.integer(seed))
}

#' Extract embeddings for a text
#'
#' Runs the model in embeddings mode and returns the hidden-state vector
#' of the last token. Note: meaningful only for models that support embeddings.
#'
#' @param ctx Context handle returned by [llama_new_context]
#' @param text Character string to embed
#' @return Numeric vector of length `n_embd`
#' @export
#' @examples
#' \dontrun{
#' model <- llama_load_model("model.gguf")
#' ctx <- llama_new_context(model)
#'
#' emb1 <- llama_embeddings(ctx, "Hello world")
#' emb2 <- llama_embeddings(ctx, "Hi there")
#'
#' # Cosine similarity
#' similarity <- sum(emb1 * emb2) / (sqrt(sum(emb1^2)) * sqrt(sum(emb2^2)))
#' cat("Similarity:", similarity, "\n")
#' }
llama_embeddings <- function(ctx, text) {
    stopifnot(is.character(text), length(text) == 1)
    .Call("r_llama_embeddings", ctx, text)
}

# ============================================================
# Chat templates
# ============================================================

#' Get model's built-in chat template
#'
#' Returns the chat template string embedded in the model file, if any.
#' Common templates include ChatML, Llama, Mistral, etc.
#'
#' @param model Model handle returned by [llama_load_model]
#' @param name Optional template name (NULL for default)
#' @return Character string with the template, or NULL if not available
#' @export
#' @examples
#' \dontrun{
#' model <- llama_load_model("llama-3.2-instruct.gguf")
#' tmpl <- llama_chat_template(model)
#' cat(tmpl)
#' }
llama_chat_template <- function(model, name = NULL) {
    .Call("r_llama_chat_template", model, name)
}

#' Apply chat template to messages
#'
#' Formats a conversation using the specified chat template.
#' This is essential for instruct/chat models to work correctly.
#'
#' @param messages List of messages, each with `role` and `content` elements.
#'   Roles are typically "system", "user", "assistant".
#' @param template Template string (from [llama_chat_template]) or NULL to use default
#' @param add_generation_prompt Whether to add the assistant prompt prefix at the end
#' @return Formatted prompt string ready for generation
#' @export
#' @examples
#' \dontrun{
#' model <- llama_load_model("llama-3.2-instruct.gguf")
#' tmpl <- llama_chat_template(model)
#'
#' messages <- list(
#'   list(role = "system", content = "You are a helpful assistant."),
#'   list(role = "user", content = "What is R?")
#' )
#'
#' prompt <- llama_chat_apply_template(messages, template = tmpl)
#' cat(prompt)
#'
#' ctx <- llama_new_context(model)
#' response <- llama_generate(ctx, prompt)
#' }
llama_chat_apply_template <- function(messages, template = NULL, add_generation_prompt = TRUE) {
    stopifnot(is.list(messages))
    .Call("r_llama_chat_apply_template", template, messages, as.logical(add_generation_prompt))
}

# ============================================================
# LoRA adapters
# ============================================================

#' Load a LoRA adapter
#'
#' Loads a LoRA (Low-Rank Adaptation) adapter file that can be applied
#' to modify the model's behavior without changing the base weights.
#'
#' @param model Model handle returned by [llama_load_model]
#' @param path Path to the LoRA adapter file (.gguf or .bin)
#' @return LoRA adapter handle (ExternalPtr)
#' @export
#' @examples
#' \dontrun{
#' model <- llama_load_model("base-model.gguf")
#' lora <- llama_lora_load(model, "fine-tuned-adapter.gguf")
#'
#' ctx <- llama_new_context(model)
#' llama_lora_apply(ctx, lora, scale = 1.0)
#'
#' # Now generation uses the LoRA-modified model
#' result <- llama_generate(ctx, "Hello")
#' }
llama_lora_load <- function(model, path) {
    stopifnot(is.character(path), length(path) == 1)
    if (!file.exists(path)) stop("llamaR: LoRA file does not exist: ", path)
    .Call("r_llama_lora_load", model, path)
}

#' Apply a LoRA adapter to context
#'
#' Activates a loaded LoRA adapter for the given context.
#' Multiple LoRA adapters can be applied simultaneously.
#'
#' @param ctx Context handle returned by [llama_new_context]
#' @param lora LoRA adapter handle from [llama_lora_load]
#' @param scale Scaling factor for the adapter (1.0 = full effect, 0.5 = half effect)
#' @return Invisible NULL
#' @export
#' @examples
#' \dontrun{
#' model <- llama_load_model("base-model.gguf")
#' lora <- llama_lora_load(model, "adapter.gguf")
#' ctx <- llama_new_context(model)
#'
#' # Apply with full strength
#' llama_lora_apply(ctx, lora, scale = 1.0)
#'
#' # Or apply with reduced effect
#' llama_lora_apply(ctx, lora, scale = 0.5)
#' }
llama_lora_apply <- function(ctx, lora, scale = 1.0) {
    .Call("r_llama_lora_apply", ctx, lora, as.double(scale))
    invisible(NULL)
}

#' Remove a LoRA adapter from context
#'
#' Deactivates a specific LoRA adapter from the context.
#'
#' @param ctx Context handle returned by [llama_new_context]
#' @param lora LoRA adapter handle to remove
#' @return Integer: 0 on success, -1 if adapter was not applied
#' @export
llama_lora_remove <- function(ctx, lora) {
    .Call("r_llama_lora_remove", ctx, lora)
}

#' Remove all LoRA adapters from context
#'
#' Deactivates all LoRA adapters from the context, returning to base model behavior.
#'
#' @param ctx Context handle returned by [llama_new_context]
#' @return Invisible NULL
#' @export
#' @examples
#' \dontrun{
#' # Apply multiple LoRAs
#' llama_lora_apply(ctx, lora1)
#' llama_lora_apply(ctx, lora2)
#'
#' # Remove all at once
#' llama_lora_clear(ctx)
#' }
llama_lora_clear <- function(ctx) {
    .Call("r_llama_lora_clear", ctx)
    invisible(NULL)
}
