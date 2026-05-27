#include <vector>
#include <string>
#include <cstring>
#include <chrono>

#include <R.h>
#include <Rinternals.h>
#include <R_ext/Rdynload.h>

// Rinternals.h defines length() as a macro which conflicts with C++ methods
#ifdef length
#undef length
#endif

#include "llama.h"
#include <ggml-backend.h>
#include <ggml-cpu.h>

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

// Streaming generation state (token-by-token), see r_llama_gen_* below.
struct llama_gen_state {
    llama_context *      ctx   = NULL;   // borrowed, not owned
    const llama_vocab *  vocab = NULL;   // borrowed
    llama_sampler *      smpl  = NULL;   // owned, freed by finalizer
    int                  n_remaining = 0;
    bool                 done  = false;
    std::string          utf8_buf;       // bytes of an incomplete trailing UTF-8 char
};

static void gen_state_finalizer(SEXP x) {
    llama_gen_state * st = (llama_gen_state *) R_ExternalPtrAddr(x);
    if (st) {
        if (st->smpl) llama_sampler_free(st->smpl);
        delete st;
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
// Time / NUMA / Backend devices
// ============================================================

extern "C" SEXP r_llama_time_us(void) {
    return Rf_ScalarReal((double) llama_time_us());
}

extern "C" SEXP r_llama_numa_init(SEXP r_strategy) {
    ensure_backend_init();
    int strategy = INTEGER(r_strategy)[0];
    if (strategy < 0 || strategy >= GGML_NUMA_STRATEGY_COUNT)
        Rf_error("llamaR: invalid NUMA strategy %d (valid: 0..%d)", strategy,
                 GGML_NUMA_STRATEGY_COUNT - 1);
    llama_numa_init((enum ggml_numa_strategy) strategy);
    return R_NilValue;
}

extern "C" SEXP r_llama_backend_devices(void) {
    ensure_backend_init();
    size_t n = ggml_backend_dev_count();

    SEXP names_vec = PROTECT(Rf_allocVector(STRSXP, (R_xlen_t) n));
    SEXP descs_vec = PROTECT(Rf_allocVector(STRSXP, (R_xlen_t) n));
    SEXP types_vec = PROTECT(Rf_allocVector(STRSXP, (R_xlen_t) n));

    for (size_t i = 0; i < n; i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        SET_STRING_ELT(names_vec, (R_xlen_t) i, Rf_mkChar(ggml_backend_dev_name(dev)));
        SET_STRING_ELT(descs_vec, (R_xlen_t) i, Rf_mkChar(ggml_backend_dev_description(dev)));

        enum ggml_backend_dev_type t = ggml_backend_dev_type(dev);
        const char * type_str = "unknown";
        if (t == GGML_BACKEND_DEVICE_TYPE_CPU)   type_str = "cpu";
        else if (t == GGML_BACKEND_DEVICE_TYPE_GPU)   type_str = "gpu";
        else if (t == GGML_BACKEND_DEVICE_TYPE_IGPU)  type_str = "igpu";
        else if (t == GGML_BACKEND_DEVICE_TYPE_ACCEL) type_str = "accel";
        SET_STRING_ELT(types_vec, (R_xlen_t) i, Rf_mkChar(type_str));
    }

    // Build data.frame
    SEXP df = PROTECT(Rf_allocVector(VECSXP, 3));
    SET_VECTOR_ELT(df, 0, names_vec);
    SET_VECTOR_ELT(df, 1, descs_vec);
    SET_VECTOR_ELT(df, 2, types_vec);

    SEXP col_names = PROTECT(Rf_allocVector(STRSXP, 3));
    SET_STRING_ELT(col_names, 0, Rf_mkChar("name"));
    SET_STRING_ELT(col_names, 1, Rf_mkChar("description"));
    SET_STRING_ELT(col_names, 2, Rf_mkChar("type"));
    Rf_setAttrib(df, R_NamesSymbol, col_names);

    SEXP row_names = PROTECT(Rf_allocVector(INTSXP, 2));
    INTEGER(row_names)[0] = NA_INTEGER;
    INTEGER(row_names)[1] = -(int) n;
    Rf_setAttrib(df, R_RowNamesSymbol, row_names);
    Rf_setAttrib(df, R_ClassSymbol, Rf_mkString("data.frame"));

    UNPROTECT(6);
    return df;
}

// ============================================================
// Model: load / free / info
// ============================================================

extern "C" SEXP r_llama_load_model(SEXP r_path, SEXP r_n_gpu_layers, SEXP r_devices,
                                   SEXP r_split_mode, SEXP r_use_mmap, SEXP r_use_mlock) {
    ensure_backend_init();

    const char * path = CHAR(STRING_ELT(r_path, 0));
    int n_gpu_layers  = INTEGER(r_n_gpu_layers)[0];

    struct llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = n_gpu_layers;
    mparams.split_mode   = (enum llama_split_mode) INTEGER(r_split_mode)[0];
    mparams.use_mmap     = LOGICAL(r_use_mmap)[0];
    mparams.use_mlock    = LOGICAL(r_use_mlock)[0];

    // device selection
    std::vector<ggml_backend_dev_t> devs;
    if (!Rf_isNull(r_devices)) {
        int n_devs = Rf_length(r_devices);
        size_t n_available = ggml_backend_dev_count();
        for (int i = 0; i < n_devs; i++) {
            const char * dev_name = CHAR(STRING_ELT(r_devices, i));
            // try by name first
            ggml_backend_dev_t dev = ggml_backend_dev_by_name(dev_name);
            if (!dev) {
                // try by type keyword: "cpu", "gpu"
                if (strcmp(dev_name, "cpu") == 0) {
                    dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
                } else if (strcmp(dev_name, "gpu") == 0) {
                    dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
                    if (!dev)
                        dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU);
                } else {
                    // try as numeric index (0-based)
                    char * endptr;
                    long idx = strtol(dev_name, &endptr, 10);
                    if (*endptr == '\0' && idx >= 0 && (size_t) idx < n_available)
                        dev = ggml_backend_dev_get((size_t) idx);
                }
            }
            if (!dev) Rf_error("llamaR: device not found: '%s'", dev_name);
            devs.push_back(dev);
        }
        devs.push_back(nullptr);  // NULL-terminated
        mparams.devices = devs.data();
    }

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

    SEXP result = PROTECT(Rf_allocVector(VECSXP, 7));
    SET_VECTOR_ELT(result, 0, Rf_ScalarInteger(llama_model_n_ctx_train(model)));
    SET_VECTOR_ELT(result, 1, Rf_ScalarInteger(llama_model_n_embd(model)));
    SET_VECTOR_ELT(result, 2, Rf_ScalarInteger(llama_vocab_n_tokens(vocab)));
    SET_VECTOR_ELT(result, 3, Rf_ScalarInteger(llama_model_n_layer(model)));
    SET_VECTOR_ELT(result, 4, Rf_ScalarInteger(llama_model_n_head(model)));
    SET_VECTOR_ELT(result, 5, Rf_ScalarInteger(llama_model_n_head_kv(model)));
    SET_VECTOR_ELT(result, 6, Rf_mkString(desc));

    SEXP names = PROTECT(Rf_allocVector(STRSXP, 7));
    SET_STRING_ELT(names, 0, Rf_mkChar("n_ctx_train"));
    SET_STRING_ELT(names, 1, Rf_mkChar("n_embd"));
    SET_STRING_ELT(names, 2, Rf_mkChar("n_vocab"));
    SET_STRING_ELT(names, 3, Rf_mkChar("n_layer"));
    SET_STRING_ELT(names, 4, Rf_mkChar("n_head"));
    SET_STRING_ELT(names, 5, Rf_mkChar("n_head_kv"));
    SET_STRING_ELT(names, 6, Rf_mkChar("desc"));
    Rf_setAttrib(result, R_NamesSymbol, names);

    UNPROTECT(2);
    return result;
}

// ============================================================
// Context: new / free
// ============================================================

extern "C" SEXP r_llama_new_context(SEXP r_model, SEXP r_n_ctx,
                                    SEXP r_n_threads, SEXP r_n_threads_batch,
                                    SEXP r_n_batch, SEXP r_n_ubatch,
                                    SEXP r_n_seq_max,
                                    SEXP r_flash_attn, SEXP r_embedding) {
    llama_model * model = (llama_model *) R_ExternalPtrAddr(r_model);
    if (!model) Rf_error("llamaR: invalid model pointer");

    bool embedding = LOGICAL(r_embedding)[0];

    struct llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx            = (uint32_t) INTEGER(r_n_ctx)[0];
    cparams.n_threads        = INTEGER(r_n_threads)[0];
    cparams.n_threads_batch  = INTEGER(r_n_threads_batch)[0];
    cparams.n_batch          = (uint32_t) INTEGER(r_n_batch)[0];
    cparams.n_ubatch         = (uint32_t) INTEGER(r_n_ubatch)[0];
    cparams.n_seq_max        = (uint32_t) INTEGER(r_n_seq_max)[0];
    cparams.flash_attn_type  = (enum llama_flash_attn_type) INTEGER(r_flash_attn)[0];
    cparams.embeddings       = embedding;
    cparams.kv_unified       = false; // match llama-bench / upstream default
    cparams.swa_full         = false; // match llama-bench
    cparams.no_perf          = false; // enable llama_perf_context() counters

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        Rf_error("llamaR: failed to create context");
    }

    if (embedding) {
        llama_set_causal_attn(ctx, false);
    }

    // tag = embedding flag, prot = r_model (keeps model alive)
    SEXP tag = PROTECT(Rf_ScalarLogical(embedding));
    SEXP result = PROTECT(R_MakeExternalPtr(ctx, tag, r_model));
    R_RegisterCFinalizer(result, context_finalizer);
    UNPROTECT(2);
    return result;
}

// Helper: check if context was created in embedding mode
static bool ctx_is_embedding(SEXP r_ctx) {
    SEXP tag = R_ExternalPtrTag(r_ctx);
    if (tag != R_NilValue && TYPEOF(tag) == LGLSXP) {
        return LOGICAL(tag)[0];
    }
    return false;
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

extern "C" SEXP r_llama_tokenize(SEXP r_ctx, SEXP r_text, SEXP r_add_special,
                                 SEXP r_parse_special) {
    llama_context * ctx = (llama_context *) R_ExternalPtrAddr(r_ctx);
    if (!ctx) Rf_error("llamaR: invalid context pointer");

    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);

    const char * text          = CHAR(STRING_ELT(r_text, 0));
    bool         add_special   = LOGICAL(r_add_special)[0] != 0;
    bool         parse_special = LOGICAL(r_parse_special)[0] != 0;
    int          text_len      = (int) strlen(text);

    // first pass: get required buffer size (returns negative on "need more space")
    int n_tokens = llama_tokenize(vocab, text, text_len, NULL, 0, add_special, parse_special);
    if (n_tokens < 0) n_tokens = -n_tokens;

    std::vector<llama_token> tokens(n_tokens);
    int actual = llama_tokenize(vocab, text, text_len, tokens.data(), n_tokens, add_special, parse_special);
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
                                  SEXP r_mirostat_eta, SEXP r_grammar,
                                  SEXP r_with_timings) {
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
    bool         with_timings   = (Rf_asLogical(r_with_timings) == TRUE);

    int prompt_len = (int) strlen(prompt);

    using clk = std::chrono::steady_clock;
    auto t_total_start = clk::now();
    double t_tokenize_ms = 0, t_build_sampler_ms = 0, t_kv_clear_ms = 0;
    double t_prefill_dispatch_ms = 0, t_prefill_sync_ms = 0;
    double t_gpu_sync_ms = 0, t_sample_ms = 0, t_decode_dispatch_ms = 0;
    double t_post_decode_sync_ms = 0;
    double t_detokenize_ms = 0;
    int    n_iterations = 0;
    int    n_splits_prefill = 0;
    int    n_splits_decode  = 0;

    auto tic = clk::now();

    // --- tokenize prompt ---
    // parse_special = true: the prompt has already been through the chat
    // template, so role markers like [INST]/<|im_start|> are control tokens
    // and must be parsed as such, not split into literal characters.
    int n_tokens = llama_tokenize(vocab, prompt, prompt_len, NULL, 0, true, true);
    if (n_tokens < 0) n_tokens = -n_tokens;
    if (n_tokens == 0) {
        Rf_error("llamaR: prompt produced zero tokens");
    }

    std::vector<llama_token> prompt_tokens(n_tokens);
    int actual = llama_tokenize(vocab, prompt, prompt_len,
                                prompt_tokens.data(), n_tokens, true, true);
    if (actual < 0) Rf_error("llamaR: tokenization failed");
    n_tokens = actual;

    if (with_timings) {
        t_tokenize_ms = std::chrono::duration<double, std::milli>(clk::now() - tic).count();
        tic = clk::now();
    }

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

    if (with_timings) {
        t_build_sampler_ms = std::chrono::duration<double, std::milli>(clk::now() - tic).count();
        tic = clk::now();
    }

    // --- clear KV cache ---
    llama_memory_clear(llama_get_memory(ctx), true);

    if (with_timings) {
        t_kv_clear_ms = std::chrono::duration<double, std::milli>(clk::now() - tic).count();
        tic = clk::now();
    }

    // --- prefill: dispatch (chunked by n_batch) ---
    // A single llama_decode carries at most n_batch tokens; split long
    // prompts so we never trip llama.cpp's n_tokens <= n_batch assert.
    struct llama_batch batch;
    {
        int n_batch = (int) llama_n_batch(ctx);
        if (n_batch <= 0) n_batch = n_tokens;
        for (int off = 0; off < n_tokens; off += n_batch) {
            int n_chunk = (n_tokens - off < n_batch) ? (n_tokens - off) : n_batch;
            batch = llama_batch_get_one(prompt_tokens.data() + off, n_chunk);
            if (llama_decode(ctx, batch) != 0) {
                llama_sampler_free(smpl);
                Rf_error("llamaR: failed to process prompt (chunk at offset %d of %d)",
                         off, n_tokens);
            }
        }
    }

    if (with_timings) {
        t_prefill_dispatch_ms = std::chrono::duration<double, std::milli>(clk::now() - tic).count();
        tic = clk::now();
        // honest GPU wait for prefill, isolated from first sample
        llama_synchronize(ctx);
        t_prefill_sync_ms = std::chrono::duration<double, std::milli>(clk::now() - tic).count();
        n_splits_prefill = llama_n_splits(ctx);
    }

    // --- autoregressive decode loop ---
    std::vector<llama_token> generated;
    llama_token current_token;

    for (int i = 0; i < max_new_tokens; i++) {
        if (with_timings) {
            // before sampling, ensure GPU is done with the previous decode dispatch
            // (skipped on i=0 because prefill_sync already drained it)
            if (i > 0) {
                auto t0 = clk::now();
                llama_synchronize(ctx);
                t_gpu_sync_ms += std::chrono::duration<double, std::milli>(clk::now() - t0).count();
            }
        }

        auto t_s0 = with_timings ? clk::now() : clk::time_point{};
        current_token = llama_sampler_sample(smpl, ctx, -1);

        if (llama_vocab_is_eog(vocab, current_token)) {
            if (with_timings) {
                t_sample_ms += std::chrono::duration<double, std::milli>(clk::now() - t_s0).count();
                n_iterations = i + 1;
            }
            break;
        }

        generated.push_back(current_token);
        llama_sampler_accept(smpl, current_token);

        if (with_timings) {
            t_sample_ms += std::chrono::duration<double, std::milli>(clk::now() - t_s0).count();
        }

        auto t_d0 = with_timings ? clk::now() : clk::time_point{};
        batch = llama_batch_get_one(&current_token, 1);
        if (llama_decode(ctx, batch) != 0) {
            llama_sampler_free(smpl);
            Rf_error("llamaR: failed during token generation");
        }
        if (with_timings) {
            t_decode_dispatch_ms += std::chrono::duration<double, std::milli>(clk::now() - t_d0).count();
            // immediate post-dispatch drain: how much GPU work remains right
            // after llama_decode returns? compare to t_gpu_sync_ms which is
            // measured at the *start* of the next iteration (after batch_get_one etc.)
            auto t_p0 = clk::now();
            llama_synchronize(ctx);
            t_post_decode_sync_ms += std::chrono::duration<double, std::milli>(clk::now() - t_p0).count();
            if (i == 0) {
                n_splits_decode = llama_n_splits(ctx);
            }
            n_iterations = i + 1;
        }
    }

    llama_sampler_free(smpl);

    // --- detokenize generated tokens ---
    if (generated.empty()) {
        SEXP empty = PROTECT(Rf_mkString(""));
        if (with_timings) {
            double t_total_ms = std::chrono::duration<double, std::milli>(clk::now() - t_total_start).count();
            const char * names[] = {
                "t_tokenize_ms", "t_build_sampler_ms", "t_kv_clear_ms",
                "t_prefill_dispatch_ms", "t_prefill_sync_ms",
                "t_gpu_sync_ms", "t_sample_ms", "t_decode_dispatch_ms",
                "t_post_decode_sync_ms", "t_detokenize_ms",
                "n_iterations", "n_splits_prefill", "n_splits_decode",
                "t_total_ms", NULL
            };
            SEXP timings = PROTECT(Rf_allocVector(REALSXP, 14));
            REAL(timings)[0]  = t_tokenize_ms;
            REAL(timings)[1]  = t_build_sampler_ms;
            REAL(timings)[2]  = t_kv_clear_ms;
            REAL(timings)[3]  = t_prefill_dispatch_ms;
            REAL(timings)[4]  = t_prefill_sync_ms;
            REAL(timings)[5]  = t_gpu_sync_ms;
            REAL(timings)[6]  = t_sample_ms;
            REAL(timings)[7]  = t_decode_dispatch_ms;
            REAL(timings)[8]  = t_post_decode_sync_ms;
            REAL(timings)[9]  = 0.0;  // detokenize (empty path)
            REAL(timings)[10] = (double) n_iterations;
            REAL(timings)[11] = (double) n_splits_prefill;
            REAL(timings)[12] = (double) n_splits_decode;
            REAL(timings)[13] = t_total_ms;
            SEXP nm = PROTECT(Rf_allocVector(STRSXP, 14));
            for (int k = 0; k < 14; k++) SET_STRING_ELT(nm, k, Rf_mkChar(names[k]));
            Rf_setAttrib(timings, R_NamesSymbol, nm);
            Rf_setAttrib(empty, Rf_install("timings"), timings);
            UNPROTECT(2);
        }
        UNPROTECT(1);
        return empty;
    }

    if (with_timings) tic = clk::now();

    int text_len = llama_detokenize(vocab, generated.data(), (int) generated.size(),
                                    NULL, 0, false, false);
    if (text_len < 0) text_len = -text_len;

    std::vector<char> text(text_len + 1);
    int result = llama_detokenize(vocab, generated.data(), (int) generated.size(),
                                  text.data(), text_len, false, false);
    if (result < 0) result = 0;
    text[result] = '\0';

    if (with_timings) {
        t_detokenize_ms = std::chrono::duration<double, std::milli>(clk::now() - tic).count();
    }

    SEXP r_text = PROTECT(Rf_mkString(text.data()));
    if (with_timings) {
        double t_total_ms = std::chrono::duration<double, std::milli>(clk::now() - t_total_start).count();
        const char * names[] = {
            "t_tokenize_ms", "t_build_sampler_ms", "t_kv_clear_ms",
            "t_prefill_dispatch_ms", "t_prefill_sync_ms",
            "t_gpu_sync_ms", "t_sample_ms", "t_decode_dispatch_ms",
            "t_post_decode_sync_ms", "t_detokenize_ms",
            "n_iterations", "n_splits_prefill", "n_splits_decode",
            "t_total_ms", NULL
        };
        SEXP timings = PROTECT(Rf_allocVector(REALSXP, 14));
        REAL(timings)[0]  = t_tokenize_ms;
        REAL(timings)[1]  = t_build_sampler_ms;
        REAL(timings)[2]  = t_kv_clear_ms;
        REAL(timings)[3]  = t_prefill_dispatch_ms;
        REAL(timings)[4]  = t_prefill_sync_ms;
        REAL(timings)[5]  = t_gpu_sync_ms;
        REAL(timings)[6]  = t_sample_ms;
        REAL(timings)[7]  = t_decode_dispatch_ms;
        REAL(timings)[8]  = t_post_decode_sync_ms;
        REAL(timings)[9]  = t_detokenize_ms;
        REAL(timings)[10] = (double) n_iterations;
        REAL(timings)[11] = (double) n_splits_prefill;
        REAL(timings)[12] = (double) n_splits_decode;
        REAL(timings)[13] = t_total_ms;
        SEXP nm = PROTECT(Rf_allocVector(STRSXP, 14));
        for (int k = 0; k < 14; k++) SET_STRING_ELT(nm, k, Rf_mkChar(names[k]));
        Rf_setAttrib(timings, R_NamesSymbol, nm);
        Rf_setAttrib(r_text, Rf_install("timings"), timings);
        UNPROTECT(2);
    }
    UNPROTECT(1);
    return r_text;
}

// ============================================================
// Streaming generation: begin / next / end (token-by-token)
// ============================================================
//
// Splits r_llama_generate into three calls so callers can pull one chunk of
// text at a time (e.g. to push into an SSE stream). State lives in an
// externalptr with a GC finalizer that frees the sampler chain.

// Length in bytes of the trailing run of bytes in `s` that forms an
// incomplete (truncated) UTF-8 sequence. Returns 0 when `s` ends on a
// complete character. Only the final, still-growing code point is held back.
static size_t utf8_incomplete_tail(const std::string & s) {
    size_t n = s.size();
    // Scan back over continuation bytes (10xxxxxx) to find the last lead byte.
    size_t i = n;
    while (i > 0) {
        unsigned char c = (unsigned char) s[i - 1];
        if ((c & 0xC0) != 0x80) {  // not a continuation byte: this is the lead
            i--;
            break;
        }
        i--;
    }
    if (i >= n) return 0;  // last byte was itself a lead with no body, handled below
    unsigned char lead = (unsigned char) s[i];
    size_t need;
    if      ((lead & 0x80) == 0x00) need = 1;  // 0xxxxxxx
    else if ((lead & 0xE0) == 0xC0) need = 2;  // 110xxxxx
    else if ((lead & 0xF0) == 0xE0) need = 3;  // 1110xxxx
    else if ((lead & 0xF8) == 0xF0) need = 4;  // 11110xxx
    else return 0;                             // stray continuation byte: emit as-is
    size_t have = n - i;
    return have < need ? have : 0;  // hold back only if the char is unfinished
}

extern "C" SEXP r_llama_gen_begin(SEXP r_ctx, SEXP r_prompt,
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
    // parse_special = true: prompt has been through the chat template, so its
    // role markers are control tokens (see r_llama_generate for rationale).
    int n_tokens = llama_tokenize(vocab, prompt, prompt_len, NULL, 0, true, true);
    if (n_tokens < 0) n_tokens = -n_tokens;
    if (n_tokens == 0) Rf_error("llamaR: prompt produced zero tokens");

    std::vector<llama_token> prompt_tokens(n_tokens);
    int actual = llama_tokenize(vocab, prompt, prompt_len,
                                prompt_tokens.data(), n_tokens, true, true);
    if (actual < 0) Rf_error("llamaR: tokenization failed");
    n_tokens = actual;

    // --- build sampler chain (identical to r_llama_generate) ---
    auto sparams = llama_sampler_chain_default_params();
    llama_sampler * smpl = llama_sampler_chain_init(sparams);

    if (grammar && strlen(grammar) > 0) {
        llama_sampler_chain_add(smpl, llama_sampler_init_grammar(vocab, grammar, "root"));
    }
    if (repeat_penalty != 1.0f || freq_penalty != 0.0f || pres_penalty != 0.0f) {
        llama_sampler_chain_add(smpl,
            llama_sampler_init_penalties(repeat_last_n, repeat_penalty, freq_penalty, pres_penalty));
    }
    if (mirostat == 0) {
        if (top_k > 0)     llama_sampler_chain_add(smpl, llama_sampler_init_top_k(top_k));
        if (min_p > 0.0f)  llama_sampler_chain_add(smpl, llama_sampler_init_min_p(min_p, 1));
        if (typical_p < 1.0f) llama_sampler_chain_add(smpl, llama_sampler_init_typical(typical_p, 1));
        if (top_p < 1.0f)  llama_sampler_chain_add(smpl, llama_sampler_init_top_p(top_p, 1));
        if (temp > 0.0f)   llama_sampler_chain_add(smpl, llama_sampler_init_temp(temp));
        if (temp > 0.0f)   llama_sampler_chain_add(smpl, llama_sampler_init_dist(seed));
        else               llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
    } else if (mirostat == 1) {
        int n_vocab_size = llama_vocab_n_tokens(vocab);
        llama_sampler_chain_add(smpl, llama_sampler_init_temp(temp > 0.0f ? temp : 0.8f));
        llama_sampler_chain_add(smpl, llama_sampler_init_mirostat(n_vocab_size, seed, mirostat_tau, mirostat_eta, 100));
    } else if (mirostat == 2) {
        llama_sampler_chain_add(smpl, llama_sampler_init_temp(temp > 0.0f ? temp : 0.8f));
        llama_sampler_chain_add(smpl, llama_sampler_init_mirostat_v2(seed, mirostat_tau, mirostat_eta));
    }

    // --- clear KV cache and prefill the prompt ---
    // Split the prefill into chunks of at most n_batch tokens: a single
    // llama_decode call may carry no more than n_batch tokens (llama.cpp
    // asserts otherwise). Positions continue automatically across calls
    // since the KV cache was just cleared and grows with each decode.
    llama_memory_clear(llama_get_memory(ctx), true);
    int n_batch = (int) llama_n_batch(ctx);
    if (n_batch <= 0) n_batch = n_tokens;
    for (int off = 0; off < n_tokens; off += n_batch) {
        int n_chunk = (n_tokens - off < n_batch) ? (n_tokens - off) : n_batch;
        struct llama_batch batch = llama_batch_get_one(prompt_tokens.data() + off, n_chunk);
        if (llama_decode(ctx, batch) != 0) {
            llama_sampler_free(smpl);
            Rf_error("llamaR: failed to process prompt (chunk at offset %d of %d)",
                     off, n_tokens);
        }
    }

    llama_gen_state * st = new llama_gen_state();
    st->ctx         = ctx;
    st->vocab       = vocab;
    st->smpl        = smpl;
    st->n_remaining = max_new_tokens;
    st->done        = false;

    SEXP ptr = PROTECT(R_MakeExternalPtr(st, Rf_install("llama_gen_state"), R_NilValue));
    R_RegisterCFinalizerEx(ptr, gen_state_finalizer, TRUE);
    UNPROTECT(1);
    return ptr;
}

// One generation step. Returns a length-1 character vector with the next
// chunk of text (possibly ""), or NULL when generation is finished (EOG
// reached or token budget exhausted). Holds back an incomplete trailing
// UTF-8 char until the next call; r_llama_gen_end() flushes any remainder.
extern "C" SEXP r_llama_gen_next(SEXP r_state) {
    llama_gen_state * st = (llama_gen_state *) R_ExternalPtrAddr(r_state);
    if (!st) Rf_error("llamaR: invalid generation state pointer");
    if (st->done || st->n_remaining <= 0) {
        st->done = true;
        return R_NilValue;
    }

    llama_token tok = llama_sampler_sample(st->smpl, st->ctx, -1);
    if (llama_vocab_is_eog(st->vocab, tok)) {
        st->done = true;
        return R_NilValue;
    }

    llama_sampler_accept(st->smpl, tok);
    st->n_remaining--;

    // detokenize this single token, appending to the UTF-8 carry buffer
    char piece[256];
    int np = llama_token_to_piece(st->vocab, tok, piece, sizeof(piece), 0, false);
    if (np < 0) {
        std::vector<char> big(-np);
        np = llama_token_to_piece(st->vocab, tok, big.data(), (int) big.size(), 0, false);
        if (np > 0) st->utf8_buf.append(big.data(), np);
    } else if (np > 0) {
        st->utf8_buf.append(piece, np);
    }

    // decode the accepted token to advance the context for the next step
    struct llama_batch batch = llama_batch_get_one(&tok, 1);
    if (llama_decode(st->ctx, batch) != 0) {
        st->done = true;
        Rf_error("llamaR: failed during token generation");
    }

    // emit everything except a possibly-incomplete trailing UTF-8 char
    size_t tail = utf8_incomplete_tail(st->utf8_buf);
    size_t emit = st->utf8_buf.size() - tail;
    SEXP r_text = PROTECT(Rf_ScalarString(
        Rf_mkCharLenCE(st->utf8_buf.data(), (int) emit, CE_UTF8)));
    st->utf8_buf.erase(0, emit);
    UNPROTECT(1);
    return r_text;
}

// Flush any bytes still held in the carry buffer and mark the state done.
// Safe to call multiple times. The sampler is freed by the GC finalizer.
extern "C" SEXP r_llama_gen_end(SEXP r_state) {
    llama_gen_state * st = (llama_gen_state *) R_ExternalPtrAddr(r_state);
    if (!st) Rf_error("llamaR: invalid generation state pointer");
    st->done = true;
    SEXP r_text = PROTECT(Rf_ScalarString(
        Rf_mkCharLenCE(st->utf8_buf.data(), (int) st->utf8_buf.size(), CE_UTF8)));
    st->utf8_buf.clear();
    UNPROTECT(1);
    return r_text;
}

// ============================================================
// Generate batch: continuous batching for N independent prompts
// ============================================================

// Forward declaration; tokenize_text is defined later in the Embeddings section.
static int tokenize_text(const llama_vocab * vocab, const char * text,
                         std::vector<llama_token> & out);

extern "C" SEXP r_llama_generate_batch(SEXP r_ctx, SEXP r_prompts,
                                       SEXP r_max_new_tokens, SEXP r_temp,
                                       SEXP r_top_k, SEXP r_top_p, SEXP r_seed,
                                       SEXP r_min_p, SEXP r_typical_p,
                                       SEXP r_repeat_penalty, SEXP r_repeat_last_n,
                                       SEXP r_frequency_penalty, SEXP r_presence_penalty,
                                       SEXP r_grammar) {
    llama_context * ctx = (llama_context *) R_ExternalPtrAddr(r_ctx);
    if (!ctx) Rf_error("llamaR: invalid context pointer");

    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);

    int n_seq = Rf_length(r_prompts);
    if (n_seq <= 0) Rf_error("llamaR: prompts must be non-empty character vector");

    int max_new_tokens = INTEGER(r_max_new_tokens)[0];
    float    temp           = (float) REAL(r_temp)[0];
    int      top_k          = INTEGER(r_top_k)[0];
    float    top_p          = (float) REAL(r_top_p)[0];
    uint32_t seed           = (uint32_t) INTEGER(r_seed)[0];
    float    min_p          = (float) REAL(r_min_p)[0];
    float    typical_p      = (float) REAL(r_typical_p)[0];
    float    repeat_penalty = (float) REAL(r_repeat_penalty)[0];
    int      repeat_last_n  = INTEGER(r_repeat_last_n)[0];
    float    freq_penalty   = (float) REAL(r_frequency_penalty)[0];
    float    pres_penalty   = (float) REAL(r_presence_penalty)[0];
    const char * grammar    = Rf_isNull(r_grammar) ? NULL : CHAR(STRING_ELT(r_grammar, 0));

    // Capacity check: all prompts + max_new_tokens for all seqs must fit in n_ctx
    {
        uint32_t n_ctx = llama_n_ctx(ctx);
        uint32_t n_seq_max = llama_n_seq_max(ctx);
        if ((uint32_t) n_seq > n_seq_max) {
            Rf_error("llamaR: %d prompts exceeds context n_seq_max=%u "
                     "(rebuild context with larger n_seq_max)", n_seq, n_seq_max);
        }
        // Capacity check itself happens after tokenization below.
        (void) n_ctx;
    }

    int64_t t_batch_build = 0, t_decode = 0, t_sample = 0;
    int64_t t_prefill = 0, t_tokenize = 0, t_detokenize = 0;
    int64_t t_gpu_sync = 0;

    // --- tokenize all prompts ---
    int64_t t0_tok = llama_time_us();
    std::vector<std::vector<llama_token>> prompt_tokens(n_seq);
    int total_prompt_tokens = 0;
    for (int s = 0; s < n_seq; s++) {
        const char * p = CHAR(STRING_ELT(r_prompts, s));
        if (tokenize_text(vocab, p, prompt_tokens[s]) < 0) {
            Rf_error("llamaR: tokenization failed for prompt %d", s + 1);
        }
        if (prompt_tokens[s].empty()) {
            Rf_error("llamaR: prompt %d produced zero tokens", s + 1);
        }
        total_prompt_tokens += (int) prompt_tokens[s].size();
    }
    t_tokenize += llama_time_us() - t0_tok;

    {
        uint32_t n_ctx = llama_n_ctx(ctx);
        uint32_t need  = (uint32_t) total_prompt_tokens + (uint32_t) (n_seq * max_new_tokens);
        if (need > n_ctx) {
            Rf_error("llamaR: required tokens (%u) exceed context n_ctx=%u "
                     "(prompts=%d, max_new=%d)", need, n_ctx, total_prompt_tokens, max_new_tokens);
        }
    }

    // --- build per-seq sampler chains (seed = base_seed + seq_id) ---
    auto build_sampler = [&](uint32_t s_seed) -> llama_sampler * {
        auto sparams = llama_sampler_chain_default_params();
        llama_sampler * smpl = llama_sampler_chain_init(sparams);
        if (grammar && strlen(grammar) > 0) {
            llama_sampler_chain_add(smpl, llama_sampler_init_grammar(vocab, grammar, "root"));
        }
        if (repeat_penalty != 1.0f || freq_penalty != 0.0f || pres_penalty != 0.0f) {
            llama_sampler_chain_add(smpl,
                llama_sampler_init_penalties(repeat_last_n, repeat_penalty, freq_penalty, pres_penalty));
        }
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
            llama_sampler_chain_add(smpl, llama_sampler_init_dist(s_seed));
        else
            llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
        return smpl;
    };

    std::vector<llama_sampler *> smpls(n_seq, nullptr);
    for (int s = 0; s < n_seq; s++) {
        smpls[s] = build_sampler(seed + (uint32_t) s);
    }

    auto cleanup_samplers = [&]() {
        for (int s = 0; s < n_seq; s++) {
            if (smpls[s]) { llama_sampler_free(smpls[s]); smpls[s] = nullptr; }
        }
    };

    // --- prefill: pack all prompts into one batch with distinct seq_ids ---
    llama_memory_clear(llama_get_memory(ctx), true);

    struct llama_batch batch = llama_batch_init(total_prompt_tokens, 0, n_seq);
    std::vector<int> last_logits_idx(n_seq, -1);  // position in batch of last prompt token per seq
    std::vector<int> seq_pos(n_seq, 0);            // next position to write for each seq
    {
        int pos = 0;
        for (int s = 0; s < n_seq; s++) {
            int n_p = (int) prompt_tokens[s].size();
            for (int t = 0; t < n_p; t++) {
                batch.token[pos]     = prompt_tokens[s][t];
                batch.pos[pos]       = (llama_pos) t;
                batch.n_seq_id[pos]  = 1;
                batch.seq_id[pos][0] = (llama_seq_id) s;
                batch.logits[pos]    = (t == n_p - 1) ? 1 : 0;
                pos++;
            }
            last_logits_idx[s] = pos - 1;
            seq_pos[s]         = n_p;
        }
        batch.n_tokens = total_prompt_tokens;
    }

    int64_t t0_prefill = llama_time_us();
    if (llama_decode(ctx, batch) != 0) {
        llama_batch_free(batch);
        cleanup_samplers();
        Rf_error("llamaR: prefill decode failed");
    }
    t_prefill += llama_time_us() - t0_prefill;
    llama_batch_free(batch);

    // --- per-seq state ---
    std::vector<bool>                       active(n_seq, true);
    std::vector<int>                        n_generated(n_seq, 0);
    std::vector<std::vector<llama_token>>   generated(n_seq);
    std::vector<int>                        finished(n_seq, 0);  // 0=running, 1=eos, 2=max_tokens

    // Sample first token per seq from prefill logits
    llama_memory_t mem = llama_get_memory(ctx);
    int n_active = n_seq;
    for (int s = 0; s < n_seq; s++) {
        llama_token tok = llama_sampler_sample(smpls[s], ctx, last_logits_idx[s]);
        if (llama_vocab_is_eog(vocab, tok) || max_new_tokens <= 0) {
            active[s]   = false;
            finished[s] = (max_new_tokens <= 0) ? 2 : 1;
            n_active--;
            llama_memory_seq_rm(mem, (llama_seq_id) s, -1, -1);
            continue;
        }
        generated[s].push_back(tok);
        llama_sampler_accept(smpls[s], tok);
        n_generated[s] = 1;
    }

    // --- decode loop: one token per active seq per iteration ---
    struct llama_batch dbatch = llama_batch_init(n_seq, 0, n_seq);
    int n_decode_steps = 0;
    int n_splits_first_decode = -1;
    while (n_active > 0) {
        // Build batch from last-generated tokens of each active seq
        int64_t t0_bb = llama_time_us();
        int pos = 0;
        std::vector<int> idx_in_batch(n_seq, -1);
        for (int s = 0; s < n_seq; s++) {
            if (!active[s]) continue;
            llama_token tok = generated[s].back();
            dbatch.token[pos]     = tok;
            dbatch.pos[pos]       = (llama_pos) seq_pos[s];
            dbatch.n_seq_id[pos]  = 1;
            dbatch.seq_id[pos][0] = (llama_seq_id) s;
            dbatch.logits[pos]    = 1;
            idx_in_batch[s]       = pos;
            seq_pos[s]++;
            pos++;
        }
        dbatch.n_tokens = pos;
        t_batch_build += llama_time_us() - t0_bb;

        int64_t t0_dec = llama_time_us();
        if (llama_decode(ctx, dbatch) != 0) {
            llama_batch_free(dbatch);
            cleanup_samplers();
            Rf_error("llamaR: decode failed during batch generation");
        }
        t_decode += llama_time_us() - t0_dec;
        if (n_splits_first_decode < 0) {
            n_splits_first_decode = llama_n_splits(ctx);
        }

        // Explicit GPU sync — separates "wait for GPU" from "CPU sample work"
        int64_t t0_sync = llama_time_us();
        llama_synchronize(ctx);
        t_gpu_sync += llama_time_us() - t0_sync;

        // Sample one token per active seq
        int64_t t0_smp = llama_time_us();
        for (int s = 0; s < n_seq; s++) {
            if (!active[s]) continue;
            llama_token tok = llama_sampler_sample(smpls[s], ctx, idx_in_batch[s]);

            if (llama_vocab_is_eog(vocab, tok)) {
                active[s]   = false;
                finished[s] = 1;
                n_active--;
                llama_memory_seq_rm(mem, (llama_seq_id) s, -1, -1);
                continue;
            }

            generated[s].push_back(tok);
            llama_sampler_accept(smpls[s], tok);
            n_generated[s]++;

            if (n_generated[s] >= max_new_tokens) {
                active[s]   = false;
                finished[s] = 2;
                n_active--;
                llama_memory_seq_rm(mem, (llama_seq_id) s, -1, -1);
            }
        }
        t_sample += llama_time_us() - t0_smp;
        n_decode_steps++;
    }
    llama_batch_free(dbatch);
    cleanup_samplers();

    // --- detokenize per seq, build result list ---
    int64_t t0_detok = llama_time_us();
    SEXP result = PROTECT(Rf_allocVector(VECSXP, n_seq));
    for (int s = 0; s < n_seq; s++) {
        SEXP item = PROTECT(Rf_allocVector(VECSXP, 3));

        // text
        SEXP r_text;
        if (generated[s].empty()) {
            r_text = PROTECT(Rf_mkString(""));
        } else {
            int text_len = llama_detokenize(vocab, generated[s].data(), (int) generated[s].size(),
                                            NULL, 0, false, false);
            if (text_len < 0) text_len = -text_len;
            std::vector<char> buf(text_len + 1);
            int written = llama_detokenize(vocab, generated[s].data(), (int) generated[s].size(),
                                           buf.data(), text_len, false, false);
            if (written < 0) written = 0;
            buf[written] = '\0';
            r_text = PROTECT(Rf_mkString(buf.data()));
        }
        SET_VECTOR_ELT(item, 0, r_text);
        UNPROTECT(1);

        SET_VECTOR_ELT(item, 1, Rf_ScalarInteger(n_generated[s]));

        const char * reason = (finished[s] == 1) ? "eos"
                            : (finished[s] == 2) ? "max_tokens"
                            : "running";
        SET_VECTOR_ELT(item, 2, Rf_mkString(reason));

        SEXP item_names = PROTECT(Rf_allocVector(STRSXP, 3));
        SET_STRING_ELT(item_names, 0, Rf_mkChar("text"));
        SET_STRING_ELT(item_names, 1, Rf_mkChar("n_tokens"));
        SET_STRING_ELT(item_names, 2, Rf_mkChar("finished_reason"));
        Rf_setAttrib(item, R_NamesSymbol, item_names);
        UNPROTECT(1);

        SET_VECTOR_ELT(result, s, item);
        UNPROTECT(1);
    }
    t_detokenize += llama_time_us() - t0_detok;

    int64_t t_accounted = t_tokenize + t_prefill + t_batch_build + t_decode + t_gpu_sync + t_sample + t_detokenize;
    REprintf("=== Decode loop timing ===\n");
    REprintf("  t_tokenize:    %.1f ms\n", t_tokenize    / 1000.0);
    REprintf("  t_prefill:     %.1f ms\n", t_prefill     / 1000.0);
    REprintf("  t_batch_build: %.1f ms (%d steps)\n", t_batch_build / 1000.0, n_decode_steps);
    REprintf("  t_decode:      %.1f ms (%d steps, dispatch only)\n", t_decode    / 1000.0, n_decode_steps);
    REprintf("  t_gpu_sync:    %.1f ms (%d steps, GPU wait)\n",      t_gpu_sync  / 1000.0, n_decode_steps);
    REprintf("  t_sample:      %.1f ms (%d steps, pure CPU)\n",      t_sample    / 1000.0, n_decode_steps);
    REprintf("  t_detokenize:  %.1f ms\n", t_detokenize  / 1000.0);
    REprintf("  ACCOUNTED:     %.1f ms\n", t_accounted   / 1000.0);
    REprintf("  n_splits (1st decode call): %d\n", n_splits_first_decode);

    UNPROTECT(1);
    return result;
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

    // Copy out of the context's embedding buffer, then reset the flag, before
    // building the R result — keeps r_result from being live across the
    // llama_set_embeddings call (which rchk flags as allocating).
    std::vector<float> emb_copy(emb, emb + n_embd);
    llama_set_embeddings(ctx, false);

    SEXP r_result = PROTECT(Rf_allocVector(REALSXP, n_embd));
    for (int i = 0; i < n_embd; i++) {
        REAL(r_result)[i] = (double) emb_copy[i];
    }
    UNPROTECT(1);
    return r_result;
}

extern "C" SEXP r_llama_get_embeddings_ith(SEXP r_ctx, SEXP r_i) {
    llama_context * ctx = (llama_context *) R_ExternalPtrAddr(r_ctx);
    if (!ctx) Rf_error("llamaR: invalid context pointer");

    int32_t i = INTEGER(r_i)[0];
    const llama_model * model = llama_get_model(ctx);
    int n_embd = llama_model_n_embd(model);

    float * emb = llama_get_embeddings_ith(ctx, i);
    if (!emb) Rf_error("llamaR: embeddings NULL for index %d", i);

    SEXP r_result = PROTECT(Rf_allocVector(REALSXP, n_embd));
    for (int j = 0; j < n_embd; j++)
        REAL(r_result)[j] = (double) emb[j];
    UNPROTECT(1);
    return r_result;
}

extern "C" SEXP r_llama_get_embeddings_seq(SEXP r_ctx, SEXP r_seq_id) {
    llama_context * ctx = (llama_context *) R_ExternalPtrAddr(r_ctx);
    if (!ctx) Rf_error("llamaR: invalid context pointer");

    llama_seq_id seq_id = (llama_seq_id) INTEGER(r_seq_id)[0];
    const llama_model * model = llama_get_model(ctx);
    int n_embd = llama_model_n_embd(model);

    float * emb = llama_get_embeddings_seq(ctx, seq_id);
    if (!emb) Rf_error("llamaR: pooled embeddings NULL for seq_id %d (model may not support pooling)", seq_id);

    SEXP r_result = PROTECT(Rf_allocVector(REALSXP, n_embd));
    for (int j = 0; j < n_embd; j++)
        REAL(r_result)[j] = (double) emb[j];
    UNPROTECT(1);
    return r_result;
}

extern "C" SEXP r_llama_get_embeddings(SEXP r_ctx, SEXP r_n_outputs) {
    llama_context * ctx = (llama_context *) R_ExternalPtrAddr(r_ctx);
    if (!ctx) Rf_error("llamaR: invalid context pointer");

    const llama_model * model = llama_get_model(ctx);
    int n_embd    = llama_model_n_embd(model);
    int n_outputs = INTEGER(r_n_outputs)[0];
    if (n_outputs < 1) Rf_error("llamaR: n_outputs must be >= 1");

    float * emb = llama_get_embeddings(ctx);
    if (!emb) Rf_error("llamaR: embeddings NULL (no decode performed, or pooling_type != none)");

    // Return a matrix: n_outputs rows × n_embd cols (R is column-major)
    SEXP r_result = PROTECT(Rf_allocMatrix(REALSXP, n_outputs, n_embd));
    for (int i = 0; i < n_outputs * n_embd; i++)
        REAL(r_result)[i] = (double) emb[i];
    UNPROTECT(1);
    return r_result;
}

// Helper: tokenize a single C-string, returns token count
static int tokenize_text(const llama_vocab * vocab, const char * text,
                         std::vector<llama_token> & out) {
    int text_len = (int) strlen(text);
    int n_tok = llama_tokenize(vocab, text, text_len, NULL, 0, true, false);
    if (n_tok < 0) n_tok = -n_tok;
    out.resize(n_tok);
    int actual = llama_tokenize(vocab, text, text_len, out.data(), n_tok, true, false);
    if (actual < 0) return -1;
    out.resize(actual);
    return actual;
}

// Helper: embed a single text using decode + embeddings_ith(-1)
static void embed_single(llama_context * ctx, const llama_vocab * vocab,
                          const char * text, float * out, int n_embd, int idx) {
    std::vector<llama_token> tokens;
    if (tokenize_text(vocab, text, tokens) < 0)
        Rf_error("llamaR: tokenization failed for text %d", idx + 1);

    llama_memory_clear(llama_get_memory(ctx), true);
    struct llama_batch batch = llama_batch_get_one(tokens.data(), (int) tokens.size());
    int ret = llama_decode(ctx, batch);
    if (ret != 0)
        Rf_error("llamaR: embed decode failed for text %d (code %d)", idx + 1, ret);
    llama_synchronize(ctx);

    float * emb = llama_get_embeddings_ith(ctx, -1);
    if (!emb)
        Rf_error("llamaR: embeddings NULL for text %d", idx + 1);
    memcpy(out, emb, n_embd * sizeof(float));
}

// Batch embeddings: pooled path (embedding=TRUE) or sequential (embedding=FALSE)
extern "C" SEXP r_llama_embed_batch(SEXP r_ctx, SEXP r_texts) {
    llama_context * ctx = (llama_context *) R_ExternalPtrAddr(r_ctx);
    if (!ctx) Rf_error("llamaR: invalid context pointer");

    bool embedding = ctx_is_embedding(r_ctx);

    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    int n_embd = llama_model_n_embd(model);
    int n_texts = Rf_length(r_texts);

    if (n_texts == 0) {
        SEXP r_mat = PROTECT(Rf_allocMatrix(REALSXP, 0, n_embd));
        UNPROTECT(1);
        return r_mat;
    }

    // tokenize all texts
    std::vector<std::vector<llama_token>> all_tokens(n_texts);
    int total_tokens = 0;
    for (int s = 0; s < n_texts; s++) {
        const char * text = CHAR(STRING_ELT(r_texts, s));
        if (tokenize_text(vocab, text, all_tokens[s]) < 0)
            Rf_error("llamaR: tokenization failed for text %d", s + 1);
        total_tokens += (int) all_tokens[s].size();
    }

    SEXP r_mat = PROTECT(Rf_allocMatrix(REALSXP, n_texts, n_embd));
    double * mat_ptr = REAL(r_mat);

    if (embedding) {
        // --- pooled batch: one decode for all texts ---
        struct llama_batch batch = llama_batch_init(total_tokens, 0, n_texts);
        int pos = 0;
        for (int s = 0; s < n_texts; s++) {
            for (int t = 0; t < (int) all_tokens[s].size(); t++) {
                batch.token[pos]      = all_tokens[s][t];
                batch.pos[pos]        = (llama_pos) t;
                batch.n_seq_id[pos]   = 1;
                batch.seq_id[pos][0]  = (llama_seq_id) s;
                batch.logits[pos]     = (t == (int) all_tokens[s].size() - 1) ? 1 : 0;
                pos++;
            }
        }
        batch.n_tokens = total_tokens;

        llama_memory_clear(llama_get_memory(ctx), true);
        int ret = llama_decode(ctx, batch);
        llama_batch_free(batch);
        if (ret != 0) {
            UNPROTECT(1);
            Rf_error("llamaR: batch embedding decode failed (code %d)", ret);
        }
        llama_synchronize(ctx);

        for (int s = 0; s < n_texts; s++) {
            float * emb = llama_get_embeddings_seq(ctx, (llama_seq_id) s);
            if (!emb) {
                UNPROTECT(1);
                Rf_error("llamaR: pooled embeddings NULL for seq %d", s);
            }
            for (int j = 0; j < n_embd; j++)
                mat_ptr[s + j * n_texts] = (double) emb[j];
        }
    } else {
        // --- sequential: one decode per text, last-token embedding ---
        llama_set_embeddings(ctx, true);
        std::vector<float> tmp(n_embd);
        for (int s = 0; s < n_texts; s++) {
            const char * text = CHAR(STRING_ELT(r_texts, s));
            embed_single(ctx, vocab, text, tmp.data(), n_embd, s);
            for (int j = 0; j < n_embd; j++)
                mat_ptr[s + j * n_texts] = (double) tmp[j];
        }
        llama_set_embeddings(ctx, false);
    }

    UNPROTECT(1);
    return r_mat;
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

    // Cache the symbols so Rf_install (an allocating call) runs once up front,
    // not inside the loop where r_role/r_content would be live across it.
    SEXP sym_role = Rf_install("role");
    SEXP sym_content = Rf_install("content");

    for (int i = 0; i < n_msg; i++) {
        SEXP msg = VECTOR_ELT(r_messages, i);

        // Resolve role and content independently and extract each string
        // immediately, so no SEXP is held live across another allocating call
        // (PROTECT-wise rchk-clean). r_role/r_content are scoped per branch.
        {
            SEXP r_role = Rf_getAttrib(msg, sym_role);
            if (Rf_isNull(r_role)) r_role = VECTOR_ELT(msg, 0);
            roles[i] = CHAR(STRING_ELT(r_role, 0));
        }
        {
            SEXP r_content = Rf_getAttrib(msg, sym_content);
            if (Rf_isNull(r_content)) r_content = VECTOR_ELT(msg, 1);
            contents[i] = CHAR(STRING_ELT(r_content, 0));
        }

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

extern "C" SEXP r_llama_vocab_type(SEXP r_model) {
    llama_model * model = (llama_model *) R_ExternalPtrAddr(r_model);
    if (!model) Rf_error("llamaR: invalid model pointer");

    const llama_vocab * vocab = llama_model_get_vocab(model);
    int vt = (int) llama_vocab_type(vocab);
    const char * name;
    switch (vt) {
        case 0: name = "none";   break;
        case 1: name = "spm";    break;
        case 2: name = "bpe";    break;
        case 3: name = "wpm";    break;
        case 4: name = "ugm";    break;
        case 5: name = "rwkv";   break;
        case 6: name = "plamo2"; break;
        default: name = "unknown"; break;
    }
    return Rf_mkString(name);
}

extern "C" SEXP r_llama_vocab_is_eog(SEXP r_model, SEXP r_token) {
    llama_model * model = (llama_model *) R_ExternalPtrAddr(r_model);
    if (!model) Rf_error("llamaR: invalid model pointer");
    const llama_vocab * vocab = llama_model_get_vocab(model);
    llama_token token = INTEGER(r_token)[0];
    return Rf_ScalarLogical(llama_vocab_is_eog(vocab, token) ? TRUE : FALSE);
}

extern "C" SEXP r_llama_vocab_is_control(SEXP r_model, SEXP r_token) {
    llama_model * model = (llama_model *) R_ExternalPtrAddr(r_model);
    if (!model) Rf_error("llamaR: invalid model pointer");
    const llama_vocab * vocab = llama_model_get_vocab(model);
    llama_token token = INTEGER(r_token)[0];
    return Rf_ScalarLogical(llama_vocab_is_control(vocab, token) ? TRUE : FALSE);
}

extern "C" SEXP r_llama_vocab_get_text(SEXP r_model, SEXP r_token) {
    llama_model * model = (llama_model *) R_ExternalPtrAddr(r_model);
    if (!model) Rf_error("llamaR: invalid model pointer");
    const llama_vocab * vocab = llama_model_get_vocab(model);
    llama_token token = INTEGER(r_token)[0];
    const char * text = llama_vocab_get_text(vocab, token);
    if (!text) return R_NilValue;
    return Rf_mkString(text);
}

extern "C" SEXP r_llama_vocab_get_score(SEXP r_model, SEXP r_token) {
    llama_model * model = (llama_model *) R_ExternalPtrAddr(r_model);
    if (!model) Rf_error("llamaR: invalid model pointer");
    const llama_vocab * vocab = llama_model_get_vocab(model);
    llama_token token = INTEGER(r_token)[0];
    return Rf_ScalarReal((double) llama_vocab_get_score(vocab, token));
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

extern "C" SEXP r_llama_get_model(SEXP r_ctx) {
    // The model R object is stored as the "prot" of the context external pointer.
    // Return it directly — same R externalptr that was passed to llama_new_context().
    if (!R_ExternalPtrAddr(r_ctx)) Rf_error("llamaR: invalid context pointer");
    return R_ExternalPtrProtected(r_ctx);
}

extern "C" SEXP r_llama_set_warmup(SEXP r_ctx, SEXP r_warmup) {
    llama_context * ctx = (llama_context *) R_ExternalPtrAddr(r_ctx);
    if (!ctx) Rf_error("llamaR: invalid context pointer");
    llama_set_warmup(ctx, LOGICAL(r_warmup)[0] != 0);
    return R_NilValue;
}

// Global abort callback state (one slot — sufficient for single-context use)
static SEXP s_abort_callback = R_NilValue;

static bool r_abort_callback(void * data) {
    (void) data;
    if (s_abort_callback == R_NilValue) return false;
    SEXP call   = PROTECT(Rf_lang1(s_abort_callback));
    int  error  = 0;
    SEXP result = R_tryEval(call, R_GlobalEnv, &error);
    UNPROTECT(1);
    if (error) return true;  // abort on R error
    if (TYPEOF(result) == LGLSXP && LENGTH(result) >= 1)
        return LOGICAL(result)[0] != 0;
    return false;
}

extern "C" SEXP r_llama_set_abort_callback(SEXP r_ctx, SEXP r_fn) {
    llama_context * ctx = (llama_context *) R_ExternalPtrAddr(r_ctx);
    if (!ctx) Rf_error("llamaR: invalid context pointer");

    // Validate before mutating any state, so an error leaves the old callback
    // intact rather than half-replaced.
    if (r_fn != R_NilValue && !Rf_isFunction(r_fn)) {
        Rf_error("llamaR: abort_callback must be a function or NULL");
    }

    // Release the previously preserved callback (if any) before replacing it,
    // otherwise it leaks on the precious list and stays alive forever.
    if (s_abort_callback != R_NilValue) {
        R_ReleaseObject(s_abort_callback);
        s_abort_callback = R_NilValue;
    }

    if (r_fn == R_NilValue) {
        llama_set_abort_callback(ctx, NULL, NULL);
    } else {
        s_abort_callback = r_fn;
        R_PreserveObject(s_abort_callback);
        llama_set_abort_callback(ctx, r_abort_callback, NULL);
    }
    return R_NilValue;
}

extern "C" SEXP r_llama_n_ctx(SEXP r_ctx) {
    llama_context * ctx = (llama_context *) R_ExternalPtrAddr(r_ctx);
    if (!ctx) Rf_error("llamaR: invalid context pointer");
    return Rf_ScalarInteger((int) llama_n_ctx(ctx));
}

extern "C" SEXP r_llama_n_ctx_seq(SEXP r_ctx) {
    llama_context * ctx = (llama_context *) R_ExternalPtrAddr(r_ctx);
    if (!ctx) Rf_error("llamaR: invalid context pointer");
    return Rf_ScalarInteger((int) llama_n_ctx_seq(ctx));
}

extern "C" SEXP r_llama_n_batch(SEXP r_ctx) {
    llama_context * ctx = (llama_context *) R_ExternalPtrAddr(r_ctx);
    if (!ctx) Rf_error("llamaR: invalid context pointer");
    return Rf_ScalarInteger((int) llama_n_batch(ctx));
}

extern "C" SEXP r_llama_n_ubatch(SEXP r_ctx) {
    llama_context * ctx = (llama_context *) R_ExternalPtrAddr(r_ctx);
    if (!ctx) Rf_error("llamaR: invalid context pointer");
    return Rf_ScalarInteger((int) llama_n_ubatch(ctx));
}

extern "C" SEXP r_llama_n_seq_max(SEXP r_ctx) {
    llama_context * ctx = (llama_context *) R_ExternalPtrAddr(r_ctx);
    if (!ctx) Rf_error("llamaR: invalid context pointer");
    return Rf_ScalarInteger((int) llama_n_seq_max(ctx));
}

extern "C" SEXP r_llama_n_threads(SEXP r_ctx) {
    llama_context * ctx = (llama_context *) R_ExternalPtrAddr(r_ctx);
    if (!ctx) Rf_error("llamaR: invalid context pointer");
    return Rf_ScalarInteger(llama_n_threads(ctx));
}

extern "C" SEXP r_llama_n_threads_batch(SEXP r_ctx) {
    llama_context * ctx = (llama_context *) R_ExternalPtrAddr(r_ctx);
    if (!ctx) Rf_error("llamaR: invalid context pointer");
    return Rf_ScalarInteger(llama_n_threads_batch(ctx));
}

extern "C" SEXP r_llama_pooling_type(SEXP r_ctx) {
    llama_context * ctx = (llama_context *) R_ExternalPtrAddr(r_ctx);
    if (!ctx) Rf_error("llamaR: invalid context pointer");
    int pt = (int) llama_pooling_type(ctx);
    const char * name;
    switch (pt) {
        case -1: name = "unspecified"; break;
        case  0: name = "none";        break;
        case  1: name = "mean";        break;
        case  2: name = "cls";         break;
        case  3: name = "last";        break;
        case  4: name = "rank";        break;
        default: name = "unknown";     break;
    }
    return Rf_mkString(name);
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

extern "C" SEXP r_llama_memory_seq_div(SEXP r_ctx, SEXP r_seq_id, SEXP r_p0, SEXP r_p1, SEXP r_d) {
    llama_context * ctx = (llama_context *) R_ExternalPtrAddr(r_ctx);
    if (!ctx) Rf_error("llamaR: invalid context pointer");

    llama_memory_seq_div(llama_get_memory(ctx),
                         INTEGER(r_seq_id)[0],
                         INTEGER(r_p0)[0], INTEGER(r_p1)[0],
                         INTEGER(r_d)[0]);
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

extern "C" SEXP r_llama_state_get_size(SEXP r_ctx) {
    llama_context * ctx = (llama_context *) R_ExternalPtrAddr(r_ctx);
    if (!ctx) Rf_error("llamaR: invalid context pointer");
    return Rf_ScalarReal((double) llama_state_get_size(ctx));
}

extern "C" SEXP r_llama_synchronize(SEXP r_ctx) {
    llama_context * ctx = (llama_context *) R_ExternalPtrAddr(r_ctx);
    if (!ctx) Rf_error("llamaR: invalid context pointer");
    llama_synchronize(ctx);
    return R_NilValue;
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

extern "C" SEXP r_llama_get_logits_ith(SEXP r_ctx, SEXP r_i) {
    llama_context * ctx = (llama_context *) R_ExternalPtrAddr(r_ctx);
    if (!ctx) Rf_error("llamaR: invalid context pointer");

    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    int n_vocab = llama_vocab_n_tokens(vocab);

    int32_t i = INTEGER(r_i)[0];
    float * logits = llama_get_logits_ith(ctx, i);
    if (!logits) Rf_error("llamaR: logits_ith is NULL for i=%d", i);

    SEXP result = PROTECT(Rf_allocVector(REALSXP, n_vocab));
    for (int k = 0; k < n_vocab; k++) {
        REAL(result)[k] = (double) logits[k];
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

extern "C" SEXP r_llama_perf_context_print(SEXP r_ctx) {
    llama_context * ctx = (llama_context *) R_ExternalPtrAddr(r_ctx);
    if (!ctx) Rf_error("llamaR: invalid context pointer");
    llama_perf_context_print(ctx);
    return R_NilValue;
}

extern "C" SEXP r_llama_memory_breakdown_print(SEXP r_ctx) {
    llama_context * ctx = (llama_context *) R_ExternalPtrAddr(r_ctx);
    if (!ctx) Rf_error("llamaR: invalid context pointer");
    llama_memory_breakdown_print(ctx);
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

extern "C" SEXP r_llama_supports_rpc(void) {
    return Rf_ScalarLogical(llama_supports_rpc() ? TRUE : FALSE);
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
// Batch: init / free
// ============================================================

extern "C" SEXP r_llama_batch_init(SEXP r_n_tokens, SEXP r_embd, SEXP r_n_seq_max) {
    int32_t n_tokens  = INTEGER(r_n_tokens)[0];
    int32_t embd      = INTEGER(r_embd)[0];
    int32_t n_seq_max = INTEGER(r_n_seq_max)[0];

    struct llama_batch * batch = new llama_batch;
    *batch = llama_batch_init(n_tokens, embd, n_seq_max);

    SEXP tag = PROTECT(Rf_mkString("llama_batch"));
    SEXP result = PROTECT(R_MakeExternalPtr(batch, tag, R_NilValue));
    R_RegisterCFinalizer(result, [](SEXP x) {
        llama_batch * b = (llama_batch *) R_ExternalPtrAddr(x);
        if (b) {
            llama_batch_free(*b);
            delete b;
            R_SetExternalPtrAddr(x, NULL);
        }
    });
    UNPROTECT(2);
    return result;
}

extern "C" SEXP r_llama_batch_free(SEXP r_batch) {
    llama_batch * b = (llama_batch *) R_ExternalPtrAddr(r_batch);
    if (b) {
        llama_batch_free(*b);
        delete b;
        R_SetExternalPtrAddr(r_batch, NULL);
    }
    return R_NilValue;
}

// ============================================================
// Encode (encoder-decoder models)
// ============================================================

extern "C" SEXP r_llama_encode(SEXP r_ctx, SEXP r_tokens) {
    llama_context * ctx = (llama_context *) R_ExternalPtrAddr(r_ctx);
    if (!ctx) Rf_error("llamaR: invalid context pointer");

    int n_tokens = LENGTH(r_tokens);
    std::vector<llama_token> tokens(n_tokens);
    for (int i = 0; i < n_tokens; i++) {
        tokens[i] = INTEGER(r_tokens)[i];
    }

    struct llama_batch batch = llama_batch_get_one(tokens.data(), n_tokens);
    int32_t ret = llama_encode(ctx, batch);
    if (ret < 0) Rf_error("llamaR: llama_encode failed (code %d)", ret);

    return Rf_ScalarInteger(ret);
}

// ============================================================
// Token to piece
// ============================================================

extern "C" SEXP r_llama_token_to_piece(SEXP r_ctx, SEXP r_token, SEXP r_special) {
    llama_context * ctx = (llama_context *) R_ExternalPtrAddr(r_ctx);
    if (!ctx) Rf_error("llamaR: invalid context pointer");

    const llama_vocab * vocab = llama_model_get_vocab(llama_get_model(ctx));
    llama_token token = INTEGER(r_token)[0];
    bool special = LOGICAL(r_special)[0] != 0;

    char buf[256];
    int32_t n = llama_token_to_piece(vocab, token, buf, sizeof(buf) - 1, 0, special);
    if (n < 0) Rf_error("llamaR: llama_token_to_piece failed (buffer too small)");
    buf[n] = '\0';

    return Rf_mkString(buf);
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
    {"r_llama_time_us",               (DL_FUNC) &r_llama_time_us,               0},
    {"r_llama_numa_init",             (DL_FUNC) &r_llama_numa_init,             1},
    {"r_llama_backend_devices",       (DL_FUNC) &r_llama_backend_devices,       0},
    {"r_llama_load_model",            (DL_FUNC) &r_llama_load_model,            6},
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
    {"r_llama_vocab_type",            (DL_FUNC) &r_llama_vocab_type,            1},
    {"r_llama_vocab_is_eog",          (DL_FUNC) &r_llama_vocab_is_eog,          2},
    {"r_llama_vocab_is_control",      (DL_FUNC) &r_llama_vocab_is_control,      2},
    {"r_llama_vocab_get_text",        (DL_FUNC) &r_llama_vocab_get_text,        2},
    {"r_llama_vocab_get_score",       (DL_FUNC) &r_llama_vocab_get_score,       2},
    // Context
    {"r_llama_new_context",           (DL_FUNC) &r_llama_new_context,           9},
    {"r_llama_free_context",          (DL_FUNC) &r_llama_free_context,          1},
    {"r_llama_get_model",             (DL_FUNC) &r_llama_get_model,             1},
    {"r_llama_set_warmup",            (DL_FUNC) &r_llama_set_warmup,            2},
    {"r_llama_set_abort_callback",    (DL_FUNC) &r_llama_set_abort_callback,    2},
    {"r_llama_n_ctx",                 (DL_FUNC) &r_llama_n_ctx,                 1},
    {"r_llama_n_ctx_seq",             (DL_FUNC) &r_llama_n_ctx_seq,             1},
    {"r_llama_n_batch",               (DL_FUNC) &r_llama_n_batch,               1},
    {"r_llama_n_ubatch",              (DL_FUNC) &r_llama_n_ubatch,              1},
    {"r_llama_n_seq_max",             (DL_FUNC) &r_llama_n_seq_max,             1},
    {"r_llama_n_threads",             (DL_FUNC) &r_llama_n_threads,             1},
    {"r_llama_n_threads_batch",       (DL_FUNC) &r_llama_n_threads_batch,       1},
    {"r_llama_pooling_type",          (DL_FUNC) &r_llama_pooling_type,          1},
    {"r_llama_set_n_threads",         (DL_FUNC) &r_llama_set_n_threads,         3},
    {"r_llama_set_causal_attn",       (DL_FUNC) &r_llama_set_causal_attn,       2},
    // Tokenize / Detokenize / Token piece
    {"r_llama_tokenize",              (DL_FUNC) &r_llama_tokenize,              4},
    {"r_llama_detokenize",            (DL_FUNC) &r_llama_detokenize,            2},
    {"r_llama_token_to_piece",        (DL_FUNC) &r_llama_token_to_piece,        3},
    // Batch
    {"r_llama_batch_init",            (DL_FUNC) &r_llama_batch_init,            3},
    {"r_llama_batch_free",            (DL_FUNC) &r_llama_batch_free,            1},
    // Encode
    {"r_llama_encode",                (DL_FUNC) &r_llama_encode,                2},
    // Generate
    {"r_llama_generate",              (DL_FUNC) &r_llama_generate,              18},
    {"r_llama_gen_begin",             (DL_FUNC) &r_llama_gen_begin,             17},
    {"r_llama_gen_next",              (DL_FUNC) &r_llama_gen_next,              1},
    {"r_llama_gen_end",               (DL_FUNC) &r_llama_gen_end,               1},
    {"r_llama_generate_batch",        (DL_FUNC) &r_llama_generate_batch,        14},
    // Embeddings & Logits
    {"r_llama_embeddings",            (DL_FUNC) &r_llama_embeddings,            2},
    {"r_llama_embed_batch",           (DL_FUNC) &r_llama_embed_batch,           2},
    {"r_llama_get_embeddings",        (DL_FUNC) &r_llama_get_embeddings,        2},
    {"r_llama_get_embeddings_ith",    (DL_FUNC) &r_llama_get_embeddings_ith,    2},
    {"r_llama_get_embeddings_seq",    (DL_FUNC) &r_llama_get_embeddings_seq,    2},
    {"r_llama_get_logits",            (DL_FUNC) &r_llama_get_logits,            1},
    {"r_llama_get_logits_ith",        (DL_FUNC) &r_llama_get_logits_ith,        2},
    // Memory / KV Cache
    {"r_llama_memory_clear",          (DL_FUNC) &r_llama_memory_clear,          1},
    {"r_llama_memory_seq_rm",         (DL_FUNC) &r_llama_memory_seq_rm,         4},
    {"r_llama_memory_seq_cp",         (DL_FUNC) &r_llama_memory_seq_cp,         5},
    {"r_llama_memory_seq_keep",       (DL_FUNC) &r_llama_memory_seq_keep,       2},
    {"r_llama_memory_seq_add",        (DL_FUNC) &r_llama_memory_seq_add,        5},
    {"r_llama_memory_seq_div",        (DL_FUNC) &r_llama_memory_seq_div,        5},
    {"r_llama_memory_seq_pos_range",  (DL_FUNC) &r_llama_memory_seq_pos_range,  2},
    {"r_llama_memory_can_shift",      (DL_FUNC) &r_llama_memory_can_shift,      1},
    // State
    {"r_llama_state_save",            (DL_FUNC) &r_llama_state_save,            2},
    {"r_llama_state_load",            (DL_FUNC) &r_llama_state_load,            2},
    {"r_llama_state_get_size",        (DL_FUNC) &r_llama_state_get_size,        1},
    {"r_llama_synchronize",           (DL_FUNC) &r_llama_synchronize,           1},
    // Chat templates
    {"r_llama_chat_template",         (DL_FUNC) &r_llama_chat_template,         2},
    {"r_llama_chat_apply_template",   (DL_FUNC) &r_llama_chat_apply_template,   3},
    {"r_llama_chat_builtin_templates",(DL_FUNC) &r_llama_chat_builtin_templates,0},
    // LoRA
    {"r_llama_lora_load",             (DL_FUNC) &r_llama_lora_load,             2},
    {"r_llama_lora_apply",            (DL_FUNC) &r_llama_lora_apply,            3},
    {"r_llama_lora_remove",           (DL_FUNC) &r_llama_lora_remove,           2},
    {"r_llama_lora_clear",            (DL_FUNC) &r_llama_lora_clear,            1},
    // Performance & Debug
    {"r_llama_perf_context",          (DL_FUNC) &r_llama_perf_context,          1},
    {"r_llama_perf_context_reset",    (DL_FUNC) &r_llama_perf_context_reset,    1},
    {"r_llama_perf_context_print",    (DL_FUNC) &r_llama_perf_context_print,    1},
    {"r_llama_memory_breakdown_print",(DL_FUNC) &r_llama_memory_breakdown_print,1},
    // Hardware
    {"r_llama_supports_rpc",          (DL_FUNC) &r_llama_supports_rpc,          0},
    {NULL, NULL, 0}
};

extern "C" void R_init_llamaR(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
