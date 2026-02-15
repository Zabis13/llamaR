#include <R.h>
#include <Rinternals.h>
#include <R_ext/Rdynload.h>
#include <vector>
#include <string>
#include <cstring>

#include "llama.h"

// ============================================================
// Logging control
// ============================================================

static int log_verbosity = 1;  // 0 = silent, 1 = errors only, 2 = normal, 3 = verbose

static void llama_log_callback(ggml_log_level level, const char * text, void * user_data) {
    (void) user_data;
    if (log_verbosity == 0) return;
    if (log_verbosity == 1 && level != GGML_LOG_LEVEL_ERROR) return;
    if (log_verbosity == 2 && level == GGML_LOG_LEVEL_DEBUG) return;
    // verbosity 3 = show everything
    Rprintf("%s", text);
}

// ============================================================
// Backend initialization (lazy, once)
// ============================================================

static bool backend_initialized = false;

static void ensure_backend_init(void) {
    if (!backend_initialized) {
        llama_log_set(llama_log_callback, NULL);
        llama_backend_init();
        backend_initialized = true;
    }
}

// ============================================================
// Finalizers — auto-free on GC
// ============================================================

static void model_finalizer(SEXP x) {
    llama_model * model = (llama_model *) R_ExternalPtrAddr(x);
    if (model) {
        llama_model_free(model);
        R_SetExternalPtrAddr(x, NULL);
    }
}

static void context_finalizer(SEXP x) {
    llama_context * ctx = (llama_context *) R_ExternalPtrAddr(x);
    if (ctx) {
        llama_free(ctx);
        R_SetExternalPtrAddr(x, NULL);
    }
}

// ============================================================
// Version
// ============================================================

extern "C" SEXP r_llama_version(void) {
    return Rf_mkString("0.1.1");
}

extern "C" SEXP r_llama_supports_gpu(void) {
    ensure_backend_init();
    return Rf_ScalarLogical(llama_supports_gpu_offload() ? TRUE : FALSE);
}

extern "C" SEXP r_llama_set_verbosity(SEXP r_level) {
    int level = INTEGER(r_level)[0];
    if (level < 0) level = 0;
    if (level > 3) level = 3;
    log_verbosity = level;
    return R_NilValue;
}

extern "C" SEXP r_llama_get_verbosity(void) {
    return Rf_ScalarInteger(log_verbosity);
}

// ============================================================
// Model: load / free / info
// ============================================================

extern "C" SEXP r_llama_load_model(SEXP r_path, SEXP r_n_gpu_layers) {
    ensure_backend_init();

    const char * path = CHAR(STRING_ELT(r_path, 0));
    int n_gpu_layers  = INTEGER(r_n_gpu_layers)[0];

    struct llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = n_gpu_layers;

    llama_model * model = llama_model_load_from_file(path, mparams);
    if (!model) {
        Rf_error("llamaR: failed to load model from '%s'", path);
    }

    SEXP result = PROTECT(R_MakeExternalPtr(model, R_NilValue, R_NilValue));
    R_RegisterCFinalizer(result, model_finalizer);
    UNPROTECT(1);
    return result;
}

extern "C" SEXP r_llama_free_model(SEXP r_model) {
    llama_model * model = (llama_model *) R_ExternalPtrAddr(r_model);
    if (model) {
        llama_model_free(model);
        R_SetExternalPtrAddr(r_model, NULL);
    }
    return R_NilValue;
}

extern "C" SEXP r_llama_model_info(SEXP r_model) {
    llama_model * model = (llama_model *) R_ExternalPtrAddr(r_model);
    if (!model) Rf_error("llamaR: invalid model pointer");

    const llama_vocab * vocab = llama_model_get_vocab(model);

    char desc[256];
    llama_model_desc(model, desc, sizeof(desc));

    SEXP result = PROTECT(Rf_allocVector(VECSXP, 6));
    SET_VECTOR_ELT(result, 0, Rf_ScalarInteger(llama_model_n_ctx_train(model)));
    SET_VECTOR_ELT(result, 1, Rf_ScalarInteger(llama_model_n_embd(model)));
    SET_VECTOR_ELT(result, 2, Rf_ScalarInteger(llama_vocab_n_tokens(vocab)));
    SET_VECTOR_ELT(result, 3, Rf_ScalarInteger(llama_model_n_layer(model)));
    SET_VECTOR_ELT(result, 4, Rf_ScalarInteger(llama_model_n_head(model)));
    SET_VECTOR_ELT(result, 5, Rf_mkString(desc));

    SEXP names = PROTECT(Rf_allocVector(STRSXP, 6));
    SET_STRING_ELT(names, 0, Rf_mkChar("n_ctx_train"));
    SET_STRING_ELT(names, 1, Rf_mkChar("n_embd"));
    SET_STRING_ELT(names, 2, Rf_mkChar("n_vocab"));
    SET_STRING_ELT(names, 3, Rf_mkChar("n_layer"));
    SET_STRING_ELT(names, 4, Rf_mkChar("n_head"));
    SET_STRING_ELT(names, 5, Rf_mkChar("desc"));
    Rf_setAttrib(result, R_NamesSymbol, names);

    UNPROTECT(2);
    return result;
}

// ============================================================
// Context: new / free
// ============================================================

extern "C" SEXP r_llama_new_context(SEXP r_model, SEXP r_n_ctx, SEXP r_n_threads) {
    llama_model * model = (llama_model *) R_ExternalPtrAddr(r_model);
    if (!model) Rf_error("llamaR: invalid model pointer");

    struct llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx            = (uint32_t) INTEGER(r_n_ctx)[0];
    cparams.n_threads        = INTEGER(r_n_threads)[0];
    cparams.n_threads_batch  = INTEGER(r_n_threads)[0];

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        Rf_error("llamaR: failed to create context");
    }

    // prot = r_model keeps the model ExternalPtr alive as long as ctx exists
    SEXP result = PROTECT(R_MakeExternalPtr(ctx, R_NilValue, r_model));
    R_RegisterCFinalizer(result, context_finalizer);
    UNPROTECT(1);
    return result;
}

extern "C" SEXP r_llama_free_context(SEXP r_ctx) {
    llama_context * ctx = (llama_context *) R_ExternalPtrAddr(r_ctx);
    if (ctx) {
        llama_free(ctx);
        R_SetExternalPtrAddr(r_ctx, NULL);
    }
    return R_NilValue;
}

// ============================================================
// Tokenize / Detokenize
// ============================================================

extern "C" SEXP r_llama_tokenize(SEXP r_ctx, SEXP r_text, SEXP r_add_special) {
    llama_context * ctx = (llama_context *) R_ExternalPtrAddr(r_ctx);
    if (!ctx) Rf_error("llamaR: invalid context pointer");

    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);

    const char * text       = CHAR(STRING_ELT(r_text, 0));
    bool         add_special = LOGICAL(r_add_special)[0] != 0;
    int          text_len   = (int) strlen(text);

    // first pass: get required buffer size (returns negative on "need more space")
    int n_tokens = llama_tokenize(vocab, text, text_len, NULL, 0, add_special, false);
    if (n_tokens < 0) n_tokens = -n_tokens;

    std::vector<llama_token> tokens(n_tokens);
    int actual = llama_tokenize(vocab, text, text_len, tokens.data(), n_tokens, add_special, false);
    if (actual < 0) {
        Rf_error("llamaR: tokenization failed");
    }

    SEXP r_result = PROTECT(Rf_allocVector(INTSXP, actual));
    for (int i = 0; i < actual; i++) {
        INTEGER(r_result)[i] = tokens[i];
    }
    UNPROTECT(1);
    return r_result;
}

extern "C" SEXP r_llama_detokenize(SEXP r_ctx, SEXP r_tokens) {
    llama_context * ctx = (llama_context *) R_ExternalPtrAddr(r_ctx);
    if (!ctx) Rf_error("llamaR: invalid context pointer");

    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);

    int n_tokens = LENGTH(r_tokens);
    std::vector<llama_token> tokens(n_tokens);
    for (int i = 0; i < n_tokens; i++) {
        tokens[i] = INTEGER(r_tokens)[i];
    }

    // first pass: get required buffer size
    int text_len = llama_detokenize(vocab, tokens.data(), n_tokens, NULL, 0, true, false);
    if (text_len < 0) text_len = -text_len;

    std::vector<char> text(text_len + 1);
    int actual = llama_detokenize(vocab, tokens.data(), n_tokens, text.data(), text_len, true, false);
    if (actual < 0) actual = 0;
    text[actual] = '\0';

    return Rf_mkString(text.data());
}

// ============================================================
// Generate: prompt → encode → decode loop → text
// ============================================================

extern "C" SEXP r_llama_generate(SEXP r_ctx, SEXP r_prompt,
                                  SEXP r_max_new_tokens, SEXP r_temp,
                                  SEXP r_top_k, SEXP r_top_p, SEXP r_seed,
                                  SEXP r_min_p, SEXP r_typical_p,
                                  SEXP r_repeat_penalty, SEXP r_repeat_last_n,
                                  SEXP r_frequency_penalty, SEXP r_presence_penalty,
                                  SEXP r_mirostat, SEXP r_mirostat_tau,
                                  SEXP r_mirostat_eta, SEXP r_grammar) {
    llama_context * ctx = (llama_context *) R_ExternalPtrAddr(r_ctx);
    if (!ctx) Rf_error("llamaR: invalid context pointer");

    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);

    const char * prompt         = CHAR(STRING_ELT(r_prompt, 0));
    int          max_new_tokens = INTEGER(r_max_new_tokens)[0];
    float        temp           = (float) REAL(r_temp)[0];
    int          top_k          = INTEGER(r_top_k)[0];
    float        top_p          = (float) REAL(r_top_p)[0];
    uint32_t     seed           = (uint32_t) INTEGER(r_seed)[0];
    float        min_p          = (float) REAL(r_min_p)[0];
    float        typical_p      = (float) REAL(r_typical_p)[0];
    float        repeat_penalty = (float) REAL(r_repeat_penalty)[0];
    int          repeat_last_n  = INTEGER(r_repeat_last_n)[0];
    float        freq_penalty   = (float) REAL(r_frequency_penalty)[0];
    float        pres_penalty   = (float) REAL(r_presence_penalty)[0];
    int          mirostat       = INTEGER(r_mirostat)[0];
    float        mirostat_tau   = (float) REAL(r_mirostat_tau)[0];
    float        mirostat_eta   = (float) REAL(r_mirostat_eta)[0];
    const char * grammar        = Rf_isNull(r_grammar) ? NULL : CHAR(STRING_ELT(r_grammar, 0));

    int prompt_len = (int) strlen(prompt);

    // --- tokenize prompt ---
    int n_tokens = llama_tokenize(vocab, prompt, prompt_len, NULL, 0, true, false);
    if (n_tokens < 0) n_tokens = -n_tokens;
    if (n_tokens == 0) {
        Rf_error("llamaR: prompt produced zero tokens");
    }

    std::vector<llama_token> prompt_tokens(n_tokens);
    int actual = llama_tokenize(vocab, prompt, prompt_len,
                                prompt_tokens.data(), n_tokens, true, false);
    if (actual < 0) Rf_error("llamaR: tokenization failed");
    n_tokens = actual;

    // --- build sampler chain ---
    auto sparams = llama_sampler_chain_default_params();
    llama_sampler * smpl = llama_sampler_chain_init(sparams);

    // Grammar (must be added first to constrain logits before other samplers)
    if (grammar && strlen(grammar) > 0) {
        llama_sampler_chain_add(smpl, llama_sampler_init_grammar(vocab, grammar, "root"));
    }

    // Penalties (applied before sampling)
    if (repeat_penalty != 1.0f || freq_penalty != 0.0f || pres_penalty != 0.0f) {
        llama_sampler_chain_add(smpl,
            llama_sampler_init_penalties(repeat_last_n, repeat_penalty, freq_penalty, pres_penalty));
    }

    if (mirostat == 0) {
        // Standard sampling chain
        if (top_k > 0)
            llama_sampler_chain_add(smpl, llama_sampler_init_top_k(top_k));
        if (min_p > 0.0f)
            llama_sampler_chain_add(smpl, llama_sampler_init_min_p(min_p, 1));
        if (typical_p < 1.0f)
            llama_sampler_chain_add(smpl, llama_sampler_init_typical(typical_p, 1));
        if (top_p < 1.0f)
            llama_sampler_chain_add(smpl, llama_sampler_init_top_p(top_p, 1));
        if (temp > 0.0f)
            llama_sampler_chain_add(smpl, llama_sampler_init_temp(temp));

        if (temp > 0.0f)
            llama_sampler_chain_add(smpl, llama_sampler_init_dist(seed));
        else
            llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
    } else if (mirostat == 1) {
        int n_vocab_size = llama_vocab_n_tokens(vocab);
        llama_sampler_chain_add(smpl, llama_sampler_init_temp(temp > 0.0f ? temp : 0.8f));
        llama_sampler_chain_add(smpl, llama_sampler_init_mirostat(n_vocab_size, seed, mirostat_tau, mirostat_eta, 100));
    } else if (mirostat == 2) {
        llama_sampler_chain_add(smpl, llama_sampler_init_temp(temp > 0.0f ? temp : 0.8f));
        llama_sampler_chain_add(smpl, llama_sampler_init_mirostat_v2(seed, mirostat_tau, mirostat_eta));
    }

    // --- clear KV cache and encode prompt ---
    llama_memory_clear(llama_get_memory(ctx), true);

    struct llama_batch batch = llama_batch_get_one(prompt_tokens.data(), n_tokens);
    if (llama_decode(ctx, batch) != 0) {
        llama_sampler_free(smpl);
        Rf_error("llamaR: failed to process prompt");
    }

    // --- autoregressive decode loop ---
    std::vector<llama_token> generated;
    llama_token current_token;

    for (int i = 0; i < max_new_tokens; i++) {
        current_token = llama_sampler_sample(smpl, ctx, -1);

        if (llama_vocab_is_eog(vocab, current_token)) break;

        generated.push_back(current_token);
        llama_sampler_accept(smpl, current_token);

        batch = llama_batch_get_one(&current_token, 1);
        if (llama_decode(ctx, batch) != 0) {
            llama_sampler_free(smpl);
            Rf_error("llamaR: failed during token generation");
        }
    }

    llama_sampler_free(smpl);

    // --- detokenize generated tokens ---
    if (generated.empty()) {
        return Rf_mkString("");
    }

    int text_len = llama_detokenize(vocab, generated.data(), (int) generated.size(),
                                    NULL, 0, false, false);
    if (text_len < 0) text_len = -text_len;

    std::vector<char> text(text_len + 1);
    int result = llama_detokenize(vocab, generated.data(), (int) generated.size(),
                                  text.data(), text_len, false, false);
    if (result < 0) result = 0;
    text[result] = '\0';

    return Rf_mkString(text.data());
}

// ============================================================
// Embeddings
// ============================================================

extern "C" SEXP r_llama_embeddings(SEXP r_ctx, SEXP r_text) {
    llama_context * ctx = (llama_context *) R_ExternalPtrAddr(r_ctx);
    if (!ctx) Rf_error("llamaR: invalid context pointer");

    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);

    const char * text     = CHAR(STRING_ELT(r_text, 0));
    int          text_len = (int) strlen(text);

    // tokenize
    int n_tokens = llama_tokenize(vocab, text, text_len, NULL, 0, true, false);
    if (n_tokens < 0) n_tokens = -n_tokens;

    std::vector<llama_token> tokens(n_tokens);
    int actual = llama_tokenize(vocab, text, text_len, tokens.data(), n_tokens, true, false);
    if (actual < 0) Rf_error("llamaR: tokenization failed");
    n_tokens = actual;

    // switch to embeddings mode, clear cache, run model
    llama_set_embeddings(ctx, true);
    llama_memory_clear(llama_get_memory(ctx), true);

    struct llama_batch batch = llama_batch_get_one(tokens.data(), n_tokens);

    int ret = llama_decode(ctx, batch);
    if (ret != 0) {
        llama_set_embeddings(ctx, false);
        Rf_error("llamaR: failed to compute embeddings (decode returned %d)", ret);
    }

    llama_synchronize(ctx);

    float * emb = llama_get_embeddings_ith(ctx, -1);
    if (!emb) {
        llama_set_embeddings(ctx, false);
        Rf_error("llamaR: embeddings output is NULL — model may not support embeddings");
    }

    int n_embd = llama_model_n_embd(model);
    SEXP r_result = PROTECT(Rf_allocVector(REALSXP, n_embd));
    for (int i = 0; i < n_embd; i++) {
        REAL(r_result)[i] = (double) emb[i];
    }
    UNPROTECT(1);

    llama_set_embeddings(ctx, false);
    return r_result;
}

// ============================================================
// Chat templates
// ============================================================

extern "C" SEXP r_llama_chat_template(SEXP r_model, SEXP r_name) {
    llama_model * model = (llama_model *) R_ExternalPtrAddr(r_model);
    if (!model) Rf_error("llamaR: invalid model pointer");

    const char * name = Rf_isNull(r_name) ? NULL : CHAR(STRING_ELT(r_name, 0));
    const char * tmpl = llama_model_chat_template(model, name);

    if (!tmpl) {
        return R_NilValue;
    }
    return Rf_mkString(tmpl);
}

extern "C" SEXP r_llama_chat_apply_template(SEXP r_tmpl, SEXP r_messages, SEXP r_add_ass) {
    const char * tmpl = Rf_isNull(r_tmpl) ? NULL : CHAR(STRING_ELT(r_tmpl, 0));
    bool add_ass = LOGICAL(r_add_ass)[0];

    // r_messages is a list of lists with $role and $content
    int n_msg = Rf_length(r_messages);
    std::vector<llama_chat_message> messages(n_msg);
    std::vector<std::string> roles(n_msg);
    std::vector<std::string> contents(n_msg);

    for (int i = 0; i < n_msg; i++) {
        SEXP msg = VECTOR_ELT(r_messages, i);
        SEXP r_role = Rf_getAttrib(msg, Rf_install("role"));
        SEXP r_content = Rf_getAttrib(msg, Rf_install("content"));

        // Try list element access if attributes don't work
        if (Rf_isNull(r_role)) {
            r_role = VECTOR_ELT(msg, 0);
            r_content = VECTOR_ELT(msg, 1);
        }

        roles[i] = CHAR(STRING_ELT(r_role, 0));
        contents[i] = CHAR(STRING_ELT(r_content, 0));
        messages[i].role = roles[i].c_str();
        messages[i].content = contents[i].c_str();
    }

    // First call to get required size
    int size = llama_chat_apply_template(tmpl, messages.data(), n_msg, add_ass, NULL, 0);
    if (size < 0) {
        Rf_error("llamaR: failed to apply chat template");
    }

    std::vector<char> buf(size + 1);
    int actual = llama_chat_apply_template(tmpl, messages.data(), n_msg, add_ass, buf.data(), buf.size());
    if (actual < 0) {
        Rf_error("llamaR: failed to apply chat template");
    }
    buf[actual] = '\0';

    return Rf_mkString(buf.data());
}

// ============================================================
// LoRA adapters
// ============================================================

static void lora_finalizer(SEXP x) {
    // LoRA adapters are freed with the model, so we don't free here
    // Just clear the pointer
    R_SetExternalPtrAddr(x, NULL);
}

extern "C" SEXP r_llama_lora_load(SEXP r_model, SEXP r_path) {
    llama_model * model = (llama_model *) R_ExternalPtrAddr(r_model);
    if (!model) Rf_error("llamaR: invalid model pointer");

    const char * path = CHAR(STRING_ELT(r_path, 0));

    llama_adapter_lora * adapter = llama_adapter_lora_init(model, path);
    if (!adapter) {
        Rf_error("llamaR: failed to load LoRA adapter from '%s'", path);
    }

    SEXP result = PROTECT(R_MakeExternalPtr(adapter, R_NilValue, R_NilValue));
    R_RegisterCFinalizer(result, lora_finalizer);
    UNPROTECT(1);
    return result;
}

extern "C" SEXP r_llama_lora_apply(SEXP r_ctx, SEXP r_adapter, SEXP r_scale) {
    llama_context * ctx = (llama_context *) R_ExternalPtrAddr(r_ctx);
    if (!ctx) Rf_error("llamaR: invalid context pointer");

    llama_adapter_lora * adapter = (llama_adapter_lora *) R_ExternalPtrAddr(r_adapter);
    if (!adapter) Rf_error("llamaR: invalid LoRA adapter pointer");

    float scale = (float) REAL(r_scale)[0];

    int ret = llama_set_adapter_lora(ctx, adapter, scale);
    if (ret != 0) {
        Rf_error("llamaR: failed to apply LoRA adapter (error %d)", ret);
    }

    return R_NilValue;
}

extern "C" SEXP r_llama_lora_remove(SEXP r_ctx, SEXP r_adapter) {
    llama_context * ctx = (llama_context *) R_ExternalPtrAddr(r_ctx);
    if (!ctx) Rf_error("llamaR: invalid context pointer");

    llama_adapter_lora * adapter = (llama_adapter_lora *) R_ExternalPtrAddr(r_adapter);
    if (!adapter) Rf_error("llamaR: invalid LoRA adapter pointer");

    int ret = llama_rm_adapter_lora(ctx, adapter);
    return Rf_ScalarInteger(ret);
}

extern "C" SEXP r_llama_lora_clear(SEXP r_ctx) {
    llama_context * ctx = (llama_context *) R_ExternalPtrAddr(r_ctx);
    if (!ctx) Rf_error("llamaR: invalid context pointer");

    llama_clear_adapter_lora(ctx);
    return R_NilValue;
}

// ============================================================
// Extended Model Info
// ============================================================

extern "C" SEXP r_llama_model_size(SEXP r_model) {
    llama_model * model = (llama_model *) R_ExternalPtrAddr(r_model);
    if (!model) Rf_error("llamaR: invalid model pointer");
    return Rf_ScalarReal((double) llama_model_size(model));
}

extern "C" SEXP r_llama_model_n_params(SEXP r_model) {
    llama_model * model = (llama_model *) R_ExternalPtrAddr(r_model);
    if (!model) Rf_error("llamaR: invalid model pointer");
    return Rf_ScalarReal((double) llama_model_n_params(model));
}

extern "C" SEXP r_llama_model_has_encoder(SEXP r_model) {
    llama_model * model = (llama_model *) R_ExternalPtrAddr(r_model);
    if (!model) Rf_error("llamaR: invalid model pointer");
    return Rf_ScalarLogical(llama_model_has_encoder(model) ? TRUE : FALSE);
}

extern "C" SEXP r_llama_model_has_decoder(SEXP r_model) {
    llama_model * model = (llama_model *) R_ExternalPtrAddr(r_model);
    if (!model) Rf_error("llamaR: invalid model pointer");
    return Rf_ScalarLogical(llama_model_has_decoder(model) ? TRUE : FALSE);
}

extern "C" SEXP r_llama_model_is_recurrent(SEXP r_model) {
    llama_model * model = (llama_model *) R_ExternalPtrAddr(r_model);
    if (!model) Rf_error("llamaR: invalid model pointer");
    return Rf_ScalarLogical(llama_model_is_recurrent(model) ? TRUE : FALSE);
}

extern "C" SEXP r_llama_model_meta(SEXP r_model) {
    llama_model * model = (llama_model *) R_ExternalPtrAddr(r_model);
    if (!model) Rf_error("llamaR: invalid model pointer");

    int32_t count = llama_model_meta_count(model);

    SEXP names  = PROTECT(Rf_allocVector(STRSXP, count));
    SEXP values = PROTECT(Rf_allocVector(STRSXP, count));

    char buf[512];
    for (int32_t i = 0; i < count; i++) {
        int32_t klen = llama_model_meta_key_by_index(model, i, buf, sizeof(buf));
        if (klen > 0) {
            buf[klen] = '\0';
            SET_STRING_ELT(names, i, Rf_mkChar(buf));
        } else {
            SET_STRING_ELT(names, i, Rf_mkChar(""));
        }

        int32_t vlen = llama_model_meta_val_str_by_index(model, i, buf, sizeof(buf));
        if (vlen > 0) {
            buf[(vlen < (int32_t)sizeof(buf) - 1) ? vlen : (int32_t)sizeof(buf) - 1] = '\0';
            SET_STRING_ELT(values, i, Rf_mkChar(buf));
        } else {
            SET_STRING_ELT(values, i, Rf_mkChar(""));
        }
    }

    Rf_setAttrib(values, R_NamesSymbol, names);
    UNPROTECT(2);
    return values;
}

extern "C" SEXP r_llama_model_meta_val(SEXP r_model, SEXP r_key) {
    llama_model * model = (llama_model *) R_ExternalPtrAddr(r_model);
    if (!model) Rf_error("llamaR: invalid model pointer");

    const char * key = CHAR(STRING_ELT(r_key, 0));
    char buf[512];
    int32_t len = llama_model_meta_val_str(model, key, buf, sizeof(buf));
    if (len < 0) return R_NilValue;
    buf[(len < (int32_t)sizeof(buf) - 1) ? len : (int32_t)sizeof(buf) - 1] = '\0';
    return Rf_mkString(buf);
}

// ============================================================
// Vocabulary Info
// ============================================================

extern "C" SEXP r_llama_vocab_info(SEXP r_model) {
    llama_model * model = (llama_model *) R_ExternalPtrAddr(r_model);
    if (!model) Rf_error("llamaR: invalid model pointer");

    const llama_vocab * vocab = llama_model_get_vocab(model);

    SEXP result = PROTECT(Rf_allocVector(INTSXP, 11));
    INTEGER(result)[0]  = llama_vocab_bos(vocab);
    INTEGER(result)[1]  = llama_vocab_eos(vocab);
    INTEGER(result)[2]  = llama_vocab_eot(vocab);
    INTEGER(result)[3]  = llama_vocab_sep(vocab);
    INTEGER(result)[4]  = llama_vocab_nl(vocab);
    INTEGER(result)[5]  = llama_vocab_pad(vocab);
    INTEGER(result)[6]  = llama_vocab_fim_pre(vocab);
    INTEGER(result)[7]  = llama_vocab_fim_suf(vocab);
    INTEGER(result)[8]  = llama_vocab_fim_mid(vocab);
    INTEGER(result)[9]  = llama_vocab_fim_rep(vocab);
    INTEGER(result)[10] = llama_vocab_fim_sep(vocab);

    SEXP names = PROTECT(Rf_allocVector(STRSXP, 11));
    SET_STRING_ELT(names, 0,  Rf_mkChar("bos"));
    SET_STRING_ELT(names, 1,  Rf_mkChar("eos"));
    SET_STRING_ELT(names, 2,  Rf_mkChar("eot"));
    SET_STRING_ELT(names, 3,  Rf_mkChar("sep"));
    SET_STRING_ELT(names, 4,  Rf_mkChar("nl"));
    SET_STRING_ELT(names, 5,  Rf_mkChar("pad"));
    SET_STRING_ELT(names, 6,  Rf_mkChar("fim_pre"));
    SET_STRING_ELT(names, 7,  Rf_mkChar("fim_suf"));
    SET_STRING_ELT(names, 8,  Rf_mkChar("fim_mid"));
    SET_STRING_ELT(names, 9,  Rf_mkChar("fim_rep"));
    SET_STRING_ELT(names, 10, Rf_mkChar("fim_sep"));
    Rf_setAttrib(result, R_NamesSymbol, names);

    UNPROTECT(2);
    return result;
}

// ============================================================
// Context Config
// ============================================================

extern "C" SEXP r_llama_set_n_threads(SEXP r_ctx, SEXP r_n_threads, SEXP r_n_threads_batch) {
    llama_context * ctx = (llama_context *) R_ExternalPtrAddr(r_ctx);
    if (!ctx) Rf_error("llamaR: invalid context pointer");

    int32_t n_threads       = INTEGER(r_n_threads)[0];
    int32_t n_threads_batch = INTEGER(r_n_threads_batch)[0];
    llama_set_n_threads(ctx, n_threads, n_threads_batch);
    return R_NilValue;
}

extern "C" SEXP r_llama_set_causal_attn(SEXP r_ctx, SEXP r_causal) {
    llama_context * ctx = (llama_context *) R_ExternalPtrAddr(r_ctx);
    if (!ctx) Rf_error("llamaR: invalid context pointer");

    bool causal = LOGICAL(r_causal)[0] != 0;
    llama_set_causal_attn(ctx, causal);
    return R_NilValue;
}

extern "C" SEXP r_llama_n_ctx(SEXP r_ctx) {
    llama_context * ctx = (llama_context *) R_ExternalPtrAddr(r_ctx);
    if (!ctx) Rf_error("llamaR: invalid context pointer");
    return Rf_ScalarInteger((int) llama_n_ctx(ctx));
}

// ============================================================
// Memory / KV Cache
// ============================================================

extern "C" SEXP r_llama_memory_clear(SEXP r_ctx) {
    llama_context * ctx = (llama_context *) R_ExternalPtrAddr(r_ctx);
    if (!ctx) Rf_error("llamaR: invalid context pointer");
    llama_memory_clear(llama_get_memory(ctx), true);
    return R_NilValue;
}

extern "C" SEXP r_llama_memory_seq_rm(SEXP r_ctx, SEXP r_seq_id, SEXP r_p0, SEXP r_p1) {
    llama_context * ctx = (llama_context *) R_ExternalPtrAddr(r_ctx);
    if (!ctx) Rf_error("llamaR: invalid context pointer");

    llama_seq_id seq_id = INTEGER(r_seq_id)[0];
    llama_pos p0 = INTEGER(r_p0)[0];
    llama_pos p1 = INTEGER(r_p1)[0];

    bool ok = llama_memory_seq_rm(llama_get_memory(ctx), seq_id, p0, p1);
    return Rf_ScalarLogical(ok ? TRUE : FALSE);
}

extern "C" SEXP r_llama_memory_seq_cp(SEXP r_ctx, SEXP r_seq_src, SEXP r_seq_dst, SEXP r_p0, SEXP r_p1) {
    llama_context * ctx = (llama_context *) R_ExternalPtrAddr(r_ctx);
    if (!ctx) Rf_error("llamaR: invalid context pointer");

    llama_memory_seq_cp(llama_get_memory(ctx),
                        INTEGER(r_seq_src)[0], INTEGER(r_seq_dst)[0],
                        INTEGER(r_p0)[0], INTEGER(r_p1)[0]);
    return R_NilValue;
}

extern "C" SEXP r_llama_memory_seq_keep(SEXP r_ctx, SEXP r_seq_id) {
    llama_context * ctx = (llama_context *) R_ExternalPtrAddr(r_ctx);
    if (!ctx) Rf_error("llamaR: invalid context pointer");

    llama_memory_seq_keep(llama_get_memory(ctx), INTEGER(r_seq_id)[0]);
    return R_NilValue;
}

extern "C" SEXP r_llama_memory_seq_add(SEXP r_ctx, SEXP r_seq_id, SEXP r_p0, SEXP r_p1, SEXP r_delta) {
    llama_context * ctx = (llama_context *) R_ExternalPtrAddr(r_ctx);
    if (!ctx) Rf_error("llamaR: invalid context pointer");

    llama_memory_seq_add(llama_get_memory(ctx),
                         INTEGER(r_seq_id)[0],
                         INTEGER(r_p0)[0], INTEGER(r_p1)[0],
                         INTEGER(r_delta)[0]);
    return R_NilValue;
}

extern "C" SEXP r_llama_memory_seq_pos_range(SEXP r_ctx, SEXP r_seq_id) {
    llama_context * ctx = (llama_context *) R_ExternalPtrAddr(r_ctx);
    if (!ctx) Rf_error("llamaR: invalid context pointer");

    llama_seq_id seq_id = INTEGER(r_seq_id)[0];
    llama_memory_t mem = llama_get_memory(ctx);

    SEXP result = PROTECT(Rf_allocVector(INTSXP, 2));
    INTEGER(result)[0] = llama_memory_seq_pos_min(mem, seq_id);
    INTEGER(result)[1] = llama_memory_seq_pos_max(mem, seq_id);

    SEXP names = PROTECT(Rf_allocVector(STRSXP, 2));
    SET_STRING_ELT(names, 0, Rf_mkChar("min"));
    SET_STRING_ELT(names, 1, Rf_mkChar("max"));
    Rf_setAttrib(result, R_NamesSymbol, names);

    UNPROTECT(2);
    return result;
}

extern "C" SEXP r_llama_memory_can_shift(SEXP r_ctx) {
    llama_context * ctx = (llama_context *) R_ExternalPtrAddr(r_ctx);
    if (!ctx) Rf_error("llamaR: invalid context pointer");
    return Rf_ScalarLogical(llama_memory_can_shift(llama_get_memory(ctx)) ? TRUE : FALSE);
}

// ============================================================
// State Save / Load
// ============================================================

extern "C" SEXP r_llama_state_save(SEXP r_ctx, SEXP r_path) {
    llama_context * ctx = (llama_context *) R_ExternalPtrAddr(r_ctx);
    if (!ctx) Rf_error("llamaR: invalid context pointer");

    const char * path = CHAR(STRING_ELT(r_path, 0));
    bool ok = llama_state_save_file(ctx, path, NULL, 0);
    if (!ok) Rf_error("llamaR: failed to save state to '%s'", path);
    return Rf_ScalarLogical(TRUE);
}

extern "C" SEXP r_llama_state_load(SEXP r_ctx, SEXP r_path) {
    llama_context * ctx = (llama_context *) R_ExternalPtrAddr(r_ctx);
    if (!ctx) Rf_error("llamaR: invalid context pointer");

    const char * path = CHAR(STRING_ELT(r_path, 0));
    size_t n_token_count = 0;
    bool ok = llama_state_load_file(ctx, path, NULL, 0, &n_token_count);
    if (!ok) Rf_error("llamaR: failed to load state from '%s'", path);
    return Rf_ScalarLogical(TRUE);
}

// ============================================================
// Logits
// ============================================================

extern "C" SEXP r_llama_get_logits(SEXP r_ctx) {
    llama_context * ctx = (llama_context *) R_ExternalPtrAddr(r_ctx);
    if (!ctx) Rf_error("llamaR: invalid context pointer");

    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    int n_vocab = llama_vocab_n_tokens(vocab);

    float * logits = llama_get_logits(ctx);
    if (!logits) Rf_error("llamaR: logits are NULL (no decode has been performed)");

    SEXP result = PROTECT(Rf_allocVector(REALSXP, n_vocab));
    for (int i = 0; i < n_vocab; i++) {
        REAL(result)[i] = (double) logits[i];
    }
    UNPROTECT(1);
    return result;
}

// ============================================================
// Performance
// ============================================================

extern "C" SEXP r_llama_perf_context(SEXP r_ctx) {
    llama_context * ctx = (llama_context *) R_ExternalPtrAddr(r_ctx);
    if (!ctx) Rf_error("llamaR: invalid context pointer");

    struct llama_perf_context_data perf = llama_perf_context(ctx);

    SEXP result = PROTECT(Rf_allocVector(VECSXP, 6));
    SET_VECTOR_ELT(result, 0, Rf_ScalarReal(perf.t_load_ms));
    SET_VECTOR_ELT(result, 1, Rf_ScalarReal(perf.t_p_eval_ms));
    SET_VECTOR_ELT(result, 2, Rf_ScalarReal(perf.t_eval_ms));
    SET_VECTOR_ELT(result, 3, Rf_ScalarInteger(perf.n_p_eval));
    SET_VECTOR_ELT(result, 4, Rf_ScalarInteger(perf.n_eval));
    SET_VECTOR_ELT(result, 5, Rf_ScalarInteger(perf.n_reused));

    SEXP names = PROTECT(Rf_allocVector(STRSXP, 6));
    SET_STRING_ELT(names, 0, Rf_mkChar("t_load_ms"));
    SET_STRING_ELT(names, 1, Rf_mkChar("t_p_eval_ms"));
    SET_STRING_ELT(names, 2, Rf_mkChar("t_eval_ms"));
    SET_STRING_ELT(names, 3, Rf_mkChar("n_p_eval"));
    SET_STRING_ELT(names, 4, Rf_mkChar("n_eval"));
    SET_STRING_ELT(names, 5, Rf_mkChar("n_reused"));
    Rf_setAttrib(result, R_NamesSymbol, names);

    UNPROTECT(2);
    return result;
}

extern "C" SEXP r_llama_perf_context_reset(SEXP r_ctx) {
    llama_context * ctx = (llama_context *) R_ExternalPtrAddr(r_ctx);
    if (!ctx) Rf_error("llamaR: invalid context pointer");
    llama_perf_context_reset(ctx);
    return R_NilValue;
}

extern "C" SEXP r_llama_system_info(void) {
    ensure_backend_init();
    return Rf_mkString(llama_print_system_info());
}

// ============================================================
// Hardware Support
// ============================================================

extern "C" SEXP r_llama_supports_mmap(void) {
    return Rf_ScalarLogical(llama_supports_mmap() ? TRUE : FALSE);
}

extern "C" SEXP r_llama_supports_mlock(void) {
    return Rf_ScalarLogical(llama_supports_mlock() ? TRUE : FALSE);
}

extern "C" SEXP r_llama_max_devices(void) {
    return Rf_ScalarInteger((int) llama_max_devices());
}

// ============================================================
// Chat: builtin templates
// ============================================================

extern "C" SEXP r_llama_chat_builtin_templates(void) {
    // First call to get count
    int32_t count = llama_chat_builtin_templates(NULL, 0);
    if (count <= 0) {
        return Rf_allocVector(STRSXP, 0);
    }

    std::vector<const char *> names(count);
    llama_chat_builtin_templates(names.data(), count);

    SEXP result = PROTECT(Rf_allocVector(STRSXP, count));
    for (int32_t i = 0; i < count; i++) {
        SET_STRING_ELT(result, i, Rf_mkChar(names[i] ? names[i] : ""));
    }
    UNPROTECT(1);
    return result;
}

// ============================================================
// Registration
// ============================================================

static const R_CallMethodDef CallEntries[] = {
    // Version & hardware
    {"r_llama_version",               (DL_FUNC) &r_llama_version,               0},
    {"r_llama_supports_gpu",          (DL_FUNC) &r_llama_supports_gpu,          0},
    {"r_llama_supports_mmap",         (DL_FUNC) &r_llama_supports_mmap,         0},
    {"r_llama_supports_mlock",        (DL_FUNC) &r_llama_supports_mlock,        0},
    {"r_llama_max_devices",           (DL_FUNC) &r_llama_max_devices,           0},
    {"r_llama_system_info",           (DL_FUNC) &r_llama_system_info,           0},
    // Verbosity
    {"r_llama_set_verbosity",         (DL_FUNC) &r_llama_set_verbosity,         1},
    {"r_llama_get_verbosity",         (DL_FUNC) &r_llama_get_verbosity,         0},
    // Model
    {"r_llama_load_model",            (DL_FUNC) &r_llama_load_model,            2},
    {"r_llama_free_model",            (DL_FUNC) &r_llama_free_model,            1},
    {"r_llama_model_info",            (DL_FUNC) &r_llama_model_info,            1},
    {"r_llama_model_size",            (DL_FUNC) &r_llama_model_size,            1},
    {"r_llama_model_n_params",        (DL_FUNC) &r_llama_model_n_params,        1},
    {"r_llama_model_has_encoder",     (DL_FUNC) &r_llama_model_has_encoder,     1},
    {"r_llama_model_has_decoder",     (DL_FUNC) &r_llama_model_has_decoder,     1},
    {"r_llama_model_is_recurrent",    (DL_FUNC) &r_llama_model_is_recurrent,    1},
    {"r_llama_model_meta",            (DL_FUNC) &r_llama_model_meta,            1},
    {"r_llama_model_meta_val",        (DL_FUNC) &r_llama_model_meta_val,        2},
    // Vocabulary
    {"r_llama_vocab_info",            (DL_FUNC) &r_llama_vocab_info,            1},
    // Context
    {"r_llama_new_context",           (DL_FUNC) &r_llama_new_context,           3},
    {"r_llama_free_context",          (DL_FUNC) &r_llama_free_context,          1},
    {"r_llama_n_ctx",                 (DL_FUNC) &r_llama_n_ctx,                 1},
    {"r_llama_set_n_threads",         (DL_FUNC) &r_llama_set_n_threads,         3},
    {"r_llama_set_causal_attn",       (DL_FUNC) &r_llama_set_causal_attn,       2},
    // Tokenize / Detokenize
    {"r_llama_tokenize",              (DL_FUNC) &r_llama_tokenize,              3},
    {"r_llama_detokenize",            (DL_FUNC) &r_llama_detokenize,            2},
    // Generate
    {"r_llama_generate",              (DL_FUNC) &r_llama_generate,              17},
    // Embeddings & Logits
    {"r_llama_embeddings",            (DL_FUNC) &r_llama_embeddings,            2},
    {"r_llama_get_logits",            (DL_FUNC) &r_llama_get_logits,            1},
    // Memory / KV Cache
    {"r_llama_memory_clear",          (DL_FUNC) &r_llama_memory_clear,          1},
    {"r_llama_memory_seq_rm",         (DL_FUNC) &r_llama_memory_seq_rm,         4},
    {"r_llama_memory_seq_cp",         (DL_FUNC) &r_llama_memory_seq_cp,         5},
    {"r_llama_memory_seq_keep",       (DL_FUNC) &r_llama_memory_seq_keep,       2},
    {"r_llama_memory_seq_add",        (DL_FUNC) &r_llama_memory_seq_add,        5},
    {"r_llama_memory_seq_pos_range",  (DL_FUNC) &r_llama_memory_seq_pos_range,  2},
    {"r_llama_memory_can_shift",      (DL_FUNC) &r_llama_memory_can_shift,      1},
    // State
    {"r_llama_state_save",            (DL_FUNC) &r_llama_state_save,            2},
    {"r_llama_state_load",            (DL_FUNC) &r_llama_state_load,            2},
    // Chat templates
    {"r_llama_chat_template",         (DL_FUNC) &r_llama_chat_template,         2},
    {"r_llama_chat_apply_template",   (DL_FUNC) &r_llama_chat_apply_template,   3},
    {"r_llama_chat_builtin_templates",(DL_FUNC) &r_llama_chat_builtin_templates,0},
    // LoRA
    {"r_llama_lora_load",             (DL_FUNC) &r_llama_lora_load,             2},
    {"r_llama_lora_apply",            (DL_FUNC) &r_llama_lora_apply,            3},
    {"r_llama_lora_remove",           (DL_FUNC) &r_llama_lora_remove,           2},
    {"r_llama_lora_clear",            (DL_FUNC) &r_llama_lora_clear,            1},
    // Performance
    {"r_llama_perf_context",          (DL_FUNC) &r_llama_perf_context,          1},
    {"r_llama_perf_context_reset",    (DL_FUNC) &r_llama_perf_context_reset,    1},
    {NULL, NULL, 0}
};

extern "C" void R_init_llamaR(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
