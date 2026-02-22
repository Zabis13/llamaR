#' Set logging verbosity level
#'
#' Controls how much diagnostic output is printed during model loading and inference.
#'
#' @param level Integer verbosity level:
#'   - 0: Silent (no output)
#'   - 1: Errors only (default)
#'   - 2: Normal (warnings and info)
#'   - 3: Verbose (all debug messages)
#' @return No return value, called for side effects. Sets the global
#'   verbosity level used by the underlying 'llama.cpp' library.
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
#' @return An integer scalar indicating the current verbosity level
#'   (0 = silent, 1 = errors only, 2 = normal, 3 = verbose).
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
#' @return A logical scalar: \code{TRUE} if at least one GPU backend
#'   (e.g. Vulkan) is available, \code{FALSE} otherwise.
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
#' @return An external pointer (class \code{externalptr}) wrapping the loaded
#'   model. This handle is required by \code{\link{llama_new_context}},
#'   \code{\link{llama_model_info}}, and other model-level functions.
#'   Freed automatically by the garbage collector or manually via
#'   \code{\link{llama_free_model}}.
#' @export
#' @examples
#' if (FALSE) {
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
#' @return No return value, called for side effects. Releases the memory
#'   associated with the model.
#' @export
#' @examples
#' if (FALSE) {
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
#'   - `size`: model size in bytes
#'   - `n_params`: number of parameters
#'   - `has_encoder`: whether the model has an encoder
#'   - `has_decoder`: whether the model has a decoder
#'   - `is_recurrent`: whether the model is recurrent (e.g. Mamba)
#' @export
#' @examples
#' if (FALSE) {
#' model <- llama_load_model("model.gguf")
#' info <- llama_model_info(model)
#' cat("Model:", info$desc, "\n")
#' cat("Layers:", info$n_layer, "\n")
#' cat("Context:", info$n_ctx_train, "\n")
#' cat("Size:", info$size / 1e9, "GB\n")
#' }
llama_model_info <- function(model) {
    info <- .Call("r_llama_model_info", model)
    info$size         <- .Call("r_llama_model_size", model)
    info$n_params     <- .Call("r_llama_model_n_params", model)
    info$has_encoder  <- .Call("r_llama_model_has_encoder", model)
    info$has_decoder  <- .Call("r_llama_model_has_decoder", model)
    info$is_recurrent <- .Call("r_llama_model_is_recurrent", model)
    info
}

#' Create an inference context
#'
#' @param model Model handle returned by [llama_load_model]
#' @param n_ctx Context window size (number of tokens). 0 means use the model's trained value.
#' @param n_threads Number of CPU threads to use
#' @return An external pointer (class \code{externalptr}) wrapping the inference
#'   context. This handle is required by generation, tokenization, and embedding
#'   functions. Freed automatically by the garbage collector or manually via
#'   \code{\link{llama_free_context}}.
#' @export
#' @examples
#' if (FALSE) {
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
#' @return No return value, called for side effects. Releases the memory
#'   associated with the inference context.
#' @export
#' @examples
#' if (FALSE) {
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
#' @return An integer vector of token IDs as used by the model's vocabulary.
#' @export
#' @examples
#' if (FALSE) {
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
#' @return A character scalar containing the decoded text.
#' @export
#' @examples
#' if (FALSE) {
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
#' @param min_p Min-P filtering threshold (0.0 = disabled)
#' @param typical_p Locally typical sampling threshold (1.0 = disabled)
#' @param repeat_penalty Repetition penalty (1.0 = disabled)
#' @param repeat_last_n Number of last tokens to penalize (0 = disabled, -1 = context size)
#' @param frequency_penalty Frequency penalty (0.0 = disabled)
#' @param presence_penalty Presence penalty (0.0 = disabled)
#' @param mirostat Mirostat sampling mode (0 = disabled, 1 = Mirostat, 2 = Mirostat 2.0)
#' @param mirostat_tau Mirostat target entropy (tau parameter)
#' @param mirostat_eta Mirostat learning rate (eta parameter)
#' @param grammar GBNF grammar string for constrained generation (NULL = disabled)
#' @return A character scalar containing the generated text (excluding the
#'   original prompt).
#' @export
#' @examples
#' if (FALSE) {
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
#'
#' # With repetition penalty
#' result <- llama_generate(ctx, "List items:",
#'                          repeat_penalty = 1.1, repeat_last_n = 64L)
#'
#' # JSON output with grammar
#' result <- llama_generate(ctx, "Output JSON:",
#'                          grammar = 'root ::= "{" "}" ')
#' }
llama_generate <- function(ctx, prompt, max_new_tokens = 256L,
                           temp = 0.8, top_k = 50L, top_p = 0.9, seed = 42L,
                           min_p = 0.0, typical_p = 1.0,
                           repeat_penalty = 1.0, repeat_last_n = 64L,
                           frequency_penalty = 0.0, presence_penalty = 0.0,
                           mirostat = 0L, mirostat_tau = 5.0, mirostat_eta = 0.1,
                           grammar = NULL) {
    stopifnot(is.character(prompt), length(prompt) == 1)
    .Call("r_llama_generate", ctx, prompt,
          as.integer(max_new_tokens), as.double(temp),
          as.integer(top_k), as.double(top_p), as.integer(seed),
          as.double(min_p), as.double(typical_p),
          as.double(repeat_penalty), as.integer(repeat_last_n),
          as.double(frequency_penalty), as.double(presence_penalty),
          as.integer(mirostat), as.double(mirostat_tau), as.double(mirostat_eta),
          grammar)
}

#' Extract embeddings for a text
#'
#' Runs the model in embeddings mode and returns the hidden-state vector
#' of the last token. Note: meaningful only for models that support embeddings.
#'
#' @param ctx Context handle returned by [llama_new_context]
#' @param text Character string to embed
#' @return A numeric vector of length \code{n_embd} (the model's embedding
#'   dimension) containing the hidden-state representation of the input text.
#' @export
#' @examples
#' if (FALSE) {
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
#' @return A character scalar with the chat template string, or \code{NULL} if
#'   the model does not contain a built-in template.
#' @export
#' @examples
#' if (FALSE) {
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
#' @return A character scalar containing the formatted prompt string, ready
#'   to be passed to \code{\link{llama_generate}}.
#' @export
#' @examples
#' if (FALSE) {
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
#' @return An external pointer (class \code{externalptr}) wrapping the loaded
#'   LoRA (Low-Rank Adaptation) adapter. Pass this handle to
#'   \code{\link{llama_lora_apply}} to activate the adapter.
#' @export
#' @examples
#' if (FALSE) {
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
#' @return No return value, called for side effects. Activates the LoRA adapter
#'   for the given context.
#' @export
#' @examples
#' if (FALSE) {
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
#' @return An integer scalar: 0 on success, -1 if the adapter was not applied
#'   to this context.
#' @export
#' @examples
#' if (FALSE) {
#' llama_lora_remove(ctx, lora)
#' }
llama_lora_remove <- function(ctx, lora) {
    .Call("r_llama_lora_remove", ctx, lora)
}

#' Remove all LoRA adapters from context
#'
#' Deactivates all LoRA adapters from the context, returning to base model behavior.
#'
#' @param ctx Context handle returned by [llama_new_context]
#' @return No return value, called for side effects. Removes all active LoRA
#'   adapters from the context.
#' @export
#' @examples
#' if (FALSE) {
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

# ============================================================
# Model metadata (individual access)
# ============================================================

#' Get all model metadata as a named character vector
#'
#' Returns all key-value metadata pairs stored in the GGUF model file.
#'
#' @param model Model handle returned by [llama_load_model]
#' @return A named character vector where names are metadata keys and values
#'   are the corresponding metadata values.
#' @export
#' @examples
#' if (FALSE) {
#' model <- llama_load_model("model.gguf")
#' meta <- llama_model_meta(model)
#' print(meta)
#' }
llama_model_meta <- function(model) {
    .Call("r_llama_model_meta", model)
}

#' Get a single model metadata value by key
#'
#' @param model Model handle returned by [llama_load_model]
#' @param key Character string metadata key (e.g. "general.name", "general.architecture")
#' @return A character scalar with the metadata value, or \code{NULL} if the key
#'   does not exist.
#' @export
#' @examples
#' if (FALSE) {
#' model <- llama_load_model("model.gguf")
#' llama_model_meta_val(model, "general.name")
#' llama_model_meta_val(model, "general.architecture")
#' }
llama_model_meta_val <- function(model, key) {
    stopifnot(is.character(key), length(key) == 1)
    .Call("r_llama_model_meta_val", model, key)
}

# ============================================================
# Vocabulary info
# ============================================================

#' Get vocabulary special token IDs
#'
#' Returns the token IDs for special tokens (BOS, EOS, etc.) and
#' fill-in-middle (FIM) tokens used by the model's vocabulary.
#' A value of -1 indicates the token is not defined.
#'
#' @param model Model handle returned by [llama_load_model]
#' @return A named integer vector with token IDs for: \code{bos}, \code{eos},
#'   \code{eot}, \code{sep}, \code{nl}, \code{pad}, \code{fim_pre},
#'   \code{fim_suf}, \code{fim_mid}, \code{fim_rep}, \code{fim_sep}.
#'   A value of -1 means the token is not defined by the model.
#' @export
#' @examples
#' if (FALSE) {
#' model <- llama_load_model("model.gguf")
#' vocab <- llama_vocab_info(model)
#' cat("BOS token:", vocab["bos"], "\n")
#' cat("EOS token:", vocab["eos"], "\n")
#' }
llama_vocab_info <- function(model) {
    .Call("r_llama_vocab_info", model)
}

# ============================================================
# Context configuration
# ============================================================

#' Set the number of threads for a context
#'
#' @param ctx Context handle returned by [llama_new_context]
#' @param n_threads Number of threads for single-token generation
#' @param n_threads_batch Number of threads for batch processing (prompt encoding).
#'   Defaults to the same value as \code{n_threads}.
#' @return No return value, called for side effects.
#' @export
#' @examples
#' if (FALSE) {
#' model <- llama_load_model("model.gguf")
#' ctx <- llama_new_context(model)
#' llama_set_threads(ctx, n_threads = 8L)
#' }
llama_set_threads <- function(ctx, n_threads, n_threads_batch = n_threads) {
    .Call("r_llama_set_n_threads", ctx, as.integer(n_threads), as.integer(n_threads_batch))
    invisible(NULL)
}

#' Set causal attention mode
#'
#' When disabled, the model uses full (bidirectional) attention.
#' This is useful for embedding models.
#'
#' @param ctx Context handle returned by [llama_new_context]
#' @param causal Logical; \code{TRUE} for causal (autoregressive) attention,
#'   \code{FALSE} for full bidirectional attention
#' @return No return value, called for side effects.
#' @export
#' @examples
#' if (FALSE) {
#' model <- llama_load_model("model.gguf")
#' ctx <- llama_new_context(model)
#' llama_set_causal_attn(ctx, FALSE)  # for embeddings
#' }
llama_set_causal_attn <- function(ctx, causal) {
    .Call("r_llama_set_causal_attn", ctx, as.logical(causal))
    invisible(NULL)
}

#' Get context window size
#'
#' @param ctx Context handle returned by [llama_new_context]
#' @return An integer scalar: the context window size (number of tokens).
#' @export
#' @examples
#' if (FALSE) {
#' model <- llama_load_model("model.gguf")
#' ctx <- llama_new_context(model, n_ctx = 4096L)
#' llama_n_ctx(ctx)  # 4096
#' }
llama_n_ctx <- function(ctx) {
    .Call("r_llama_n_ctx", ctx)
}

# ============================================================
# Memory / KV Cache
# ============================================================

#' Clear the KV cache
#'
#' Removes all tokens from the KV cache. Call this before starting
#' a new generation from scratch.
#'
#' @param ctx Context handle returned by [llama_new_context]
#' @return No return value, called for side effects.
#' @export
#' @examples
#' if (FALSE) {
#' llama_memory_clear(ctx)
#' }
llama_memory_clear <- function(ctx) {
    .Call("r_llama_memory_clear", ctx)
    invisible(NULL)
}

#' Remove tokens from a sequence in the KV cache
#'
#' Removes cached tokens for the given sequence in the position range [p0, p1).
#' Use p0 = -1 and p1 = -1 to remove all tokens for the sequence.
#'
#' @param ctx Context handle returned by [llama_new_context]
#' @param seq_id Sequence ID (integer)
#' @param p0 Start position (inclusive, -1 for beginning)
#' @param p1 End position (exclusive, -1 for end)
#' @return A logical scalar: \code{TRUE} if tokens were successfully removed.
#' @export
#' @examples
#' if (FALSE) {
#' # Remove all tokens from sequence 0
#' llama_memory_seq_rm(ctx, seq_id = 0L, p0 = -1L, p1 = -1L)
#' }
llama_memory_seq_rm <- function(ctx, seq_id, p0 = -1L, p1 = -1L) {
    .Call("r_llama_memory_seq_rm", ctx, as.integer(seq_id),
          as.integer(p0), as.integer(p1))
}

#' Copy a sequence in the KV cache
#'
#' Copies cached tokens from one sequence to another in the position range [p0, p1).
#'
#' @param ctx Context handle returned by [llama_new_context]
#' @param seq_id_src Source sequence ID
#' @param seq_id_dst Destination sequence ID
#' @param p0 Start position (inclusive, -1 for beginning)
#' @param p1 End position (exclusive, -1 for end)
#' @return No return value, called for side effects.
#' @export
#' @examples
#' if (FALSE) {
#' # Copy sequence 0 to sequence 1
#' llama_memory_seq_cp(ctx, seq_id_src = 0L, seq_id_dst = 1L,
#'                     p0 = -1L, p1 = -1L)
#' }
llama_memory_seq_cp <- function(ctx, seq_id_src, seq_id_dst, p0 = -1L, p1 = -1L) {
    .Call("r_llama_memory_seq_cp", ctx, as.integer(seq_id_src),
          as.integer(seq_id_dst), as.integer(p0), as.integer(p1))
    invisible(NULL)
}

#' Keep only one sequence in the KV cache
#'
#' Removes all sequences except the specified one from the KV cache.
#'
#' @param ctx Context handle returned by [llama_new_context]
#' @param seq_id Sequence ID to keep
#' @return No return value, called for side effects.
#' @export
#' @examples
#' if (FALSE) {
#' llama_memory_seq_keep(ctx, seq_id = 0L)
#' }
llama_memory_seq_keep <- function(ctx, seq_id) {
    .Call("r_llama_memory_seq_keep", ctx, as.integer(seq_id))
    invisible(NULL)
}

#' Shift token positions in a sequence
#'
#' Adds a position delta to all tokens in the given sequence within [p0, p1).
#' This is useful for implementing context shifting (sliding window).
#'
#' @param ctx Context handle returned by [llama_new_context]
#' @param seq_id Sequence ID
#' @param p0 Start position (inclusive)
#' @param p1 End position (exclusive)
#' @param delta Position shift amount (can be negative)
#' @return No return value, called for side effects.
#' @export
#' @examples
#' if (FALSE) {
#' # Shift positions left by 100 for context window management
#' llama_memory_seq_add(ctx, seq_id = 0L, p0 = 100L, p1 = -1L, delta = -100L)
#' }
llama_memory_seq_add <- function(ctx, seq_id, p0, p1, delta) {
    .Call("r_llama_memory_seq_add", ctx, as.integer(seq_id),
          as.integer(p0), as.integer(p1), as.integer(delta))
    invisible(NULL)
}

#' Get position range for a sequence
#'
#' Returns the minimum and maximum token positions for a given sequence
#' in the KV cache.
#'
#' @param ctx Context handle returned by [llama_new_context]
#' @param seq_id Sequence ID
#' @return A named integer vector with elements \code{min} and \code{max}.
#' @export
#' @examples
#' if (FALSE) {
#' range <- llama_memory_seq_pos_range(ctx, seq_id = 0L)
#' cat("Positions:", range["min"], "to", range["max"], "\n")
#' }
llama_memory_seq_pos_range <- function(ctx, seq_id) {
    .Call("r_llama_memory_seq_pos_range", ctx, as.integer(seq_id))
}

#' Check if the KV cache supports shifting
#'
#' @param ctx Context handle returned by [llama_new_context]
#' @return A logical scalar: \code{TRUE} if the memory supports position shifting.
#' @export
#' @examples
#' if (FALSE) {
#' if (llama_memory_can_shift(ctx)) {
#'   message("Context shifting is supported")
#' }
#' }
llama_memory_can_shift <- function(ctx) {
    .Call("r_llama_memory_can_shift", ctx)
}

# ============================================================
# State save / load
# ============================================================

#' Save context state to file
#'
#' Saves the full context state (including KV cache) to a binary file.
#' This allows resuming generation later from the exact same state.
#'
#' @param ctx Context handle returned by [llama_new_context]
#' @param path File path to save state to
#' @return A logical scalar: \code{TRUE} on success (errors on failure).
#' @export
#' @examples
#' if (FALSE) {
#' llama_state_save(ctx, "state.bin")
#' }
llama_state_save <- function(ctx, path) {
    stopifnot(is.character(path), length(path) == 1)
    .Call("r_llama_state_save", ctx, path)
}

#' Load context state from file
#'
#' Restores a previously saved context state (including KV cache).
#'
#' @param ctx Context handle returned by [llama_new_context]
#' @param path File path to load state from
#' @return A logical scalar: \code{TRUE} on success (errors on failure).
#' @export
#' @examples
#' if (FALSE) {
#' llama_state_load(ctx, "state.bin")
#' # Continue generation from saved state
#' result <- llama_generate(ctx, "")
#' }
llama_state_load <- function(ctx, path) {
    stopifnot(is.character(path), length(path) == 1)
    if (!file.exists(path)) stop("llamaR: state file does not exist: ", path)
    .Call("r_llama_state_load", ctx, path)
}

# ============================================================
# Logits
# ============================================================

#' Get logits from the last decode step
#'
#' Returns the raw logit vector (unnormalized log-probabilities) from the
#' last token position after a decode operation.
#'
#' @param ctx Context handle returned by [llama_new_context]
#' @return A numeric vector of length \code{n_vocab} containing the logits.
#' @export
#' @examples
#' if (FALSE) {
#' model <- llama_load_model("model.gguf")
#' ctx <- llama_new_context(model)
#' result <- llama_generate(ctx, "The capital of France is", max_new_tokens = 1L)
#' logits <- llama_get_logits(ctx)
#' # Find top token
#' top_id <- which.max(logits)
#' }
llama_get_logits <- function(ctx) {
    .Call("r_llama_get_logits", ctx)
}

# ============================================================
# Performance
# ============================================================

#' Get performance statistics
#'
#' Returns timing and count statistics for the current context,
#' including prompt processing time, token generation time, and counts.
#'
#' @param ctx Context handle returned by [llama_new_context]
#' @return A named list with fields:
#'   - `t_load_ms`: model load time in milliseconds
#'   - `t_p_eval_ms`: prompt processing time in milliseconds
#'   - `t_eval_ms`: token generation time in milliseconds
#'   - `n_p_eval`: number of prompt tokens processed
#'   - `n_eval`: number of tokens generated
#'   - `n_reused`: number of reused compute graphs
#' @export
#' @examples
#' if (FALSE) {
#' result <- llama_generate(ctx, "Hello world")
#' perf <- llama_perf(ctx)
#' cat("Prompt speed:", perf$n_p_eval / (perf$t_p_eval_ms / 1000), "tok/s\n")
#' cat("Generation speed:", perf$n_eval / (perf$t_eval_ms / 1000), "tok/s\n")
#' }
llama_perf <- function(ctx) {
    .Call("r_llama_perf_context", ctx)
}

#' Reset performance counters
#'
#' Resets the timing and token count statistics for the context.
#'
#' @param ctx Context handle returned by [llama_new_context]
#' @return No return value, called for side effects.
#' @export
#' @examples
#' if (FALSE) {
#' llama_perf_reset(ctx)
#' }
llama_perf_reset <- function(ctx) {
    .Call("r_llama_perf_context_reset", ctx)
    invisible(NULL)
}

# ============================================================
# System & Hardware info
# ============================================================

#' Get system information string
#'
#' Returns a string with information about the system capabilities
#' detected by llama.cpp (SIMD support, etc.).
#'
#' @return A character scalar with system capability information.
#' @export
#' @examples
#' cat(llama_system_info(), "\n")
llama_system_info <- function() {
    .Call("r_llama_system_info")
}

#' Check whether memory-mapped file I/O is supported
#'
#' @return A logical scalar: \code{TRUE} if mmap is supported.
#' @export
#' @examples
#' llama_supports_mmap()
llama_supports_mmap <- function() {
    .Call("r_llama_supports_mmap")
}

#' Check whether memory locking is supported
#'
#' @return A logical scalar: \code{TRUE} if mlock is supported.
#' @export
#' @examples
#' llama_supports_mlock()
llama_supports_mlock <- function() {
    .Call("r_llama_supports_mlock")
}

#' Get maximum number of devices
#'
#' @return An integer scalar: the maximum number of compute devices available.
#' @export
#' @examples
#' llama_max_devices()
llama_max_devices <- function() {
    .Call("r_llama_max_devices")
}

# ============================================================
# Chat: builtin templates
# ============================================================

#' List built-in chat templates
#'
#' Returns a character vector of all chat template names supported by llama.cpp.
#'
#' @return A character vector of built-in template names.
#' @export
#' @examples
#' llama_chat_builtin_templates()
llama_chat_builtin_templates <- function() {
    .Call("r_llama_chat_builtin_templates")
}

#' Convert a single token ID to its text piece
#'
#' @param ctx A context pointer (llama_context).
#' @param token Integer token ID.
#' @param special Logical. If TRUE, render special tokens (e.g. \code{<bos>}).
#' @return A character string — the text piece for the token.
#' @export
#' @examples
#' \dontrun{
#' model <- llama_load_model("model.gguf")
#' ctx   <- llama_new_context(model)
#' llama_token_to_piece(ctx, 1L)
#' }
llama_token_to_piece <- function(ctx, token, special = FALSE) {
    .Call("r_llama_token_to_piece", ctx, as.integer(token), as.logical(special))
}

#' Encode tokens using the encoder (encoder-decoder models only)
#'
#' Runs the encoder pass for encoder-decoder architectures (e.g. T5, BART).
#' The encoder output is stored internally and used by subsequent decoder calls.
#'
#' @param ctx A context pointer (llama_context).
#' @param tokens Integer vector of token IDs to encode.
#' @return Integer return code (0 = success, negative = error).
#' @export
#' @examples
#' \dontrun{
#' model  <- llama_load_model("t5-model.gguf")
#' ctx    <- llama_new_context(model)
#' tokens <- llama_tokenize(ctx, "Hello world")
#' llama_encode(ctx, tokens)
#' }
llama_encode <- function(ctx, tokens) {
    .Call("r_llama_encode", ctx, as.integer(tokens))
}

#' Initialise a llama batch
#'
#' Allocates a \code{llama_batch} that can hold up to \code{n_tokens} tokens.
#' Use \code{llama_batch_free()} to release the memory when done.
#'
#' @param n_tokens Maximum number of tokens in the batch.
#' @param embd Embedding size; 0 means token-ID mode (normal inference).
#' @param n_seq_max Maximum number of sequences per token.
#' @return An external pointer to the allocated batch.
#' @export
#' @examples
#' \dontrun{
#' batch <- llama_batch_init(512L)
#' llama_batch_free(batch)
#' }
llama_batch_init <- function(n_tokens, embd = 0L, n_seq_max = 1L) {
    .Call("r_llama_batch_init", as.integer(n_tokens), as.integer(embd), as.integer(n_seq_max))
}

#' Free a llama batch allocated with \code{llama_batch_init()}
#'
#' @param batch An external pointer returned by \code{llama_batch_init()}.
#' @return \code{NULL} invisibly.
#' @export
#' @examples
#' \dontrun{
#' batch <- llama_batch_init(512L)
#' llama_batch_free(batch)
#' }
llama_batch_free <- function(batch) {
    invisible(.Call("r_llama_batch_free", batch))
}
