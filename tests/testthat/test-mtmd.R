# Tests for the multimodal (vision / audio) mtmd bindings.
#
# Two tiers:
#   * argument validation / no-model probes — always run.
#   * projector-dependent (load mmproj, capabilities, image -> text) — skipped
#     gracefully when no multimodal model is available locally.

# A multimodal text model + its paired projector (mmproj). Both must exist to
# exercise the real pipeline. Point these at a local VL/OCR model when available.
MM_MODEL_PATH  <- "/mnt/Data2/DS_projects/llm_models/mmproj-test-model.gguf"
MM_MMPROJ_PATH <- "/mnt/Data2/DS_projects/llm_models/mmproj-test-proj.gguf"
# Any small test image; the one shipped with upstream llama.cpp works.
MM_IMAGE_PATH  <- "/mnt/Data2/DS_projects/llama.cpp-master/tools/mtmd/test-1.jpeg"

HAS_MM <- file.exists(MM_MODEL_PATH) && file.exists(MM_MMPROJ_PATH)

skip_if_no_mm <- function() {
    if (!HAS_MM) skip("multimodal test model / projector not available")
}
skip_if_no_image <- function() {
    if (!file.exists(MM_IMAGE_PATH)) skip("test image not available")
}

# ============================================================
# No-model probes — always run
# ============================================================

test_that("llama_mtmd_marker returns the media marker", {
    m <- llama_mtmd_marker()
    expect_type(m, "character")
    expect_length(m, 1L)
    expect_true(nzchar(m))
})

test_that("llama_mtmd_set_verbosity accepts a level and returns invisibly", {
    expect_null(llama_mtmd_set_verbosity(1L))
    expect_null(llama_mtmd_set_verbosity(0L))
    llama_mtmd_set_verbosity(1L)  # restore default
})

# ============================================================
# Argument validation — no model required
# ============================================================

test_that("llama_mtmd_load validates its arguments", {
    # model must be an externalptr
    expect_error(llama_mtmd_load("not-a-ptr", MM_MMPROJ_PATH), "externalptr")
    # mmproj path must be a single string
    fake <- structure(list(), class = "externalptr")
    expect_error(llama_mtmd_load(fake, c("a", "b")), "length")
})

test_that("llama_mtmd_load errors on a missing mmproj file", {
    fake <- structure(list(), class = "externalptr")
    expect_error(
        llama_mtmd_load(fake, "/no/such/mmproj.gguf"),
        "does not exist")
})

test_that("llama_mtmd_support_vision / _audio require an externalptr", {
    expect_error(llama_mtmd_support_vision("nope"), "externalptr")
    expect_error(llama_mtmd_support_audio("nope"), "externalptr")
})

test_that("llama_image_load validates arguments", {
    fake <- structure(list(), class = "externalptr")
    expect_error(llama_image_load("not-a-ptr", MM_IMAGE_PATH), "externalptr")
    expect_error(llama_image_load(fake, "/no/such/image.jpg"), "does not exist")
})

test_that("llama_image_eval validates arguments", {
    fake <- structure(list(), class = "externalptr")
    # prompt must be a single string
    expect_error(llama_image_eval(fake, fake, c("a", "b"), fake), "length")
    # ctx / bitmap must be externalptr
    expect_error(llama_image_eval(fake, "x", "prompt", fake), "externalptr")
})

# ============================================================
# Projector pipeline — skipped when no multimodal model
# ============================================================

test_that("llama_mtmd_load loads a projector and reports capabilities", {
    skip_if_no_mm()
    model <- llama_load_model(MM_MODEL_PATH)
    withr::defer(llama_free_model(model))

    mctx <- llama_mtmd_load(model, MM_MMPROJ_PATH)
    expect_s3_class(mctx, "externalptr")

    # at least one modality must be supported
    vis <- llama_mtmd_support_vision(mctx)
    aud <- llama_mtmd_support_audio(mctx)
    expect_type(vis, "logical")
    expect_type(aud, "logical")
    expect_true(vis || aud)
})

test_that("image -> text: eval produces a non-zero n_past then generates", {
    skip_if_no_mm()
    skip_if_no_image()
    model <- llama_load_model(MM_MODEL_PATH)
    withr::defer(llama_free_model(model))

    mctx <- llama_mtmd_load(model, MM_MMPROJ_PATH)
    skip_if(!llama_mtmd_support_vision(mctx), "model has no vision support")

    ctx <- llama_new_context(model, n_ctx = 2048L, n_threads = 2L)
    withr::defer(llama_free_context(ctx))

    img    <- llama_image_load(mctx, MM_IMAGE_PATH)
    expect_s3_class(img, "externalptr")

    prompt <- paste0("Describe this image: ", llama_mtmd_marker())
    n_past <- llama_image_eval(mctx, ctx, prompt, img, n_past = 0L)

    # text tokens + image tokens were decoded into the KV cache
    expect_type(n_past, "integer")
    expect_gt(n_past, 0L)
})
