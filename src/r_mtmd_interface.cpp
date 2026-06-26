// R bindings for the vendored multimodal (mtmd / clip) subsystem.
//
// Minimal image -> text pipeline, mirroring how mtmd-cli drives the library:
//
//   r_mtmd_init(model, mmproj_path, n_threads, use_gpu)
//       Loads the multimodal projector (mmproj GGUF) on top of an already
//       loaded text model, returning an mtmd_context externalptr (GC-finalized).
//
//   r_mtmd_support_vision(mctx) / r_mtmd_support_audio(mctx)
//       Capability probes.
//
//   r_mtmd_bitmap_from_file(mctx, path)
//       Decodes an image file (via the vendored stb_image) into an
//       mtmd_bitmap externalptr (GC-finalized).
//
//   r_mtmd_eval(mctx, lctx, prompt, bitmap, n_past)
//       Tokenizes prompt (which must contain the media marker) + bitmap into
//       chunks, then runs mtmd_helper_eval_chunks() to decode text and image
//       chunks into the llama_context KV cache. Returns the new n_past so the
//       caller can continue token generation with the existing llama_gen_* loop.
//
// The marker string (r_mtmd_marker) is what must appear in the prompt where the
// image should be injected (e.g. "<__media__>").

#include <string>
#include <vector>

// IMPORTANT: include the heavy C++ headers BEFORE R headers. Rinternals.h
// defines a `length(x)` macro that otherwise mangles STL headers.
#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include <R.h>
#include <Rinternals.h>

#ifdef length
#undef length
#endif

// Logging verbosity shared with r_llama_interface.cpp's scheme (0 silent ..
// 3 verbose). We keep an independent copy here; mtmd routes through its own
// log callback set in r_mtmd_init.
static int mtmd_log_verbosity = 1;

static void r_mtmd_log_callback(ggml_log_level level, const char * text, void * user_data) {
    (void) user_data;
    if (mtmd_log_verbosity == 0) return;
    if (mtmd_log_verbosity == 1 && level != GGML_LOG_LEVEL_ERROR) return;
    if (mtmd_log_verbosity == 2 && level == GGML_LOG_LEVEL_DEBUG) return;
    Rprintf("%s", text);
}

// ============================================================
// Finalizers
// ============================================================

static void mtmd_ctx_finalizer(SEXP x) {
    mtmd_context * mctx = (mtmd_context *) R_ExternalPtrAddr(x);
    if (mctx) {
        mtmd_free(mctx);
        R_SetExternalPtrAddr(x, NULL);
    }
}

static void mtmd_bitmap_finalizer(SEXP x) {
    mtmd_bitmap * bmp = (mtmd_bitmap *) R_ExternalPtrAddr(x);
    if (bmp) {
        mtmd_bitmap_free(bmp);
        R_SetExternalPtrAddr(x, NULL);
    }
}

// ============================================================
// Init / capabilities
// ============================================================

extern "C" SEXP r_mtmd_init(SEXP r_model, SEXP r_mmproj_path,
                            SEXP r_n_threads, SEXP r_use_gpu) {
    llama_model * model = (llama_model *) R_ExternalPtrAddr(r_model);
    if (!model) Rf_error("llamaR: invalid model pointer");

    const char * mmproj = CHAR(STRING_ELT(r_mmproj_path, 0));
    const int n_threads = Rf_asInteger(r_n_threads);
    const bool use_gpu  = (bool) Rf_asLogical(r_use_gpu);

    mtmd_log_set(r_mtmd_log_callback, NULL);

    mtmd_context_params params = mtmd_context_params_default();
    params.use_gpu       = use_gpu;
    params.print_timings = false;
    params.n_threads     = n_threads;
    params.media_marker  = mtmd_default_marker();

    mtmd_context * mctx = mtmd_init_from_file(mmproj, model, params);
    if (!mctx) {
        Rf_error("llamaR: failed to load multimodal projector from '%s'", mmproj);
    }

    // prot = r_model keeps the text model alive while mctx references it
    SEXP result = PROTECT(R_MakeExternalPtr(mctx, R_NilValue, r_model));
    R_RegisterCFinalizer(result, mtmd_ctx_finalizer);
    UNPROTECT(1);
    return result;
}

extern "C" SEXP r_mtmd_support_vision(SEXP r_mctx) {
    mtmd_context * mctx = (mtmd_context *) R_ExternalPtrAddr(r_mctx);
    if (!mctx) Rf_error("llamaR: invalid mtmd context pointer");
    return Rf_ScalarLogical(mtmd_support_vision(mctx));
}

extern "C" SEXP r_mtmd_support_audio(SEXP r_mctx) {
    mtmd_context * mctx = (mtmd_context *) R_ExternalPtrAddr(r_mctx);
    if (!mctx) Rf_error("llamaR: invalid mtmd context pointer");
    return Rf_ScalarLogical(mtmd_support_audio(mctx));
}

extern "C" SEXP r_mtmd_marker(void) {
    return Rf_mkString(mtmd_default_marker());
}

extern "C" SEXP r_mtmd_set_verbosity(SEXP r_level) {
    mtmd_log_verbosity = Rf_asInteger(r_level);
    return R_NilValue;
}

// ============================================================
// Bitmap loading
// ============================================================

extern "C" SEXP r_mtmd_bitmap_from_file(SEXP r_mctx, SEXP r_path) {
    mtmd_context * mctx = (mtmd_context *) R_ExternalPtrAddr(r_mctx);
    if (!mctx) Rf_error("llamaR: invalid mtmd context pointer");

    const char * path = CHAR(STRING_ELT(r_path, 0));
    mtmd_bitmap * bmp = mtmd_helper_bitmap_init_from_file(mctx, path);
    if (!bmp) {
        Rf_error("llamaR: failed to load media file '%s'", path);
    }

    SEXP result = PROTECT(R_MakeExternalPtr(bmp, R_NilValue, R_NilValue));
    R_RegisterCFinalizer(result, mtmd_bitmap_finalizer);
    UNPROTECT(1);
    return result;
}

// ============================================================
// Tokenize + eval (text + image) into the llama context
// ============================================================

extern "C" SEXP r_mtmd_eval(SEXP r_mctx, SEXP r_lctx, SEXP r_prompt,
                            SEXP r_bitmap, SEXP r_n_past) {
    mtmd_context  * mctx = (mtmd_context *)  R_ExternalPtrAddr(r_mctx);
    llama_context * lctx = (llama_context *) R_ExternalPtrAddr(r_lctx);
    mtmd_bitmap   * bmp  = (mtmd_bitmap *)   R_ExternalPtrAddr(r_bitmap);
    if (!mctx) Rf_error("llamaR: invalid mtmd context pointer");
    if (!lctx) Rf_error("llamaR: invalid llama context pointer");
    if (!bmp)  Rf_error("llamaR: invalid bitmap pointer");

    const char * prompt = CHAR(STRING_ELT(r_prompt, 0));
    const llama_pos n_past_in = (llama_pos) Rf_asInteger(r_n_past);

    mtmd_input_text text;
    text.text          = prompt;
    text.add_special   = true;   // add BOS etc. on first segment
    text.parse_special = true;   // parse the media marker + special tokens

    const mtmd_bitmap * bitmaps[1] = { bmp };

    mtmd_input_chunks * chunks = mtmd_input_chunks_init();
    if (!chunks) Rf_error("llamaR: failed to allocate mtmd input chunks");

    int32_t tok = mtmd_tokenize(mctx, chunks, &text, bitmaps, 1);
    if (tok != 0) {
        mtmd_input_chunks_free(chunks);
        if (tok == 1) {
            Rf_error("llamaR: number of media markers in prompt does not match "
                     "number of images (expected exactly one '%s')",
                     mtmd_default_marker());
        }
        Rf_error("llamaR: mtmd_tokenize failed (image preprocessing error, code %d)", (int) tok);
    }

    const int32_t n_batch = (int32_t) llama_n_batch(lctx);
    llama_pos new_n_past = n_past_in;

    int32_t ev = mtmd_helper_eval_chunks(mctx, lctx, chunks,
                                         /*n_past*/ n_past_in,
                                         /*seq_id*/ 0,
                                         n_batch,
                                         /*logits_last*/ true,
                                         &new_n_past);
    mtmd_input_chunks_free(chunks);
    if (ev != 0) {
        Rf_error("llamaR: mtmd_helper_eval_chunks failed (code %d)", (int) ev);
    }

    return Rf_ScalarInteger((int) new_n_past);
}
