# Multimodal (vision / audio) support via the vendored mtmd / clip subsystem.
# Loads a multimodal projector (mmproj GGUF) on top of a text model and lets you
# evaluate an image together with a text prompt into a llama context, after which
# generation proceeds with the usual llama_gen_* loop.

#' Load a multimodal projector (mmproj)
#'
#' Loads the vision/audio projector that pairs with a multimodal model. The
#' projector is a separate GGUF file (commonly named \code{mmproj-*.gguf}),
#' loaded on top of an already-loaded text model from
#' \code{\link{llama_load_model}}.
#'
#' @param model A model handle from \code{\link{llama_load_model}} (the text
#'   model the projector pairs with).
#' @param mmproj_path Path to the multimodal projector GGUF file.
#' @param n_threads Number of CPU threads for the vision/audio encoder
#'   (default \code{4L}).
#' @param use_gpu Logical; run the encoder on the GPU when available
#'   (default \code{TRUE}).
#' @return An external pointer wrapping the mtmd context. Freed automatically by
#'   the garbage collector. Required by \code{\link{llama_image_eval}} and the
#'   capability probes.
#' @seealso \code{\link{llama_mtmd_support_vision}}, \code{\link{llama_image_eval}}
#' @export
llama_mtmd_load <- function(model, mmproj_path, n_threads = 4L, use_gpu = TRUE) {
    stopifnot(inherits(model, "externalptr"))
    stopifnot(is.character(mmproj_path), length(mmproj_path) == 1)
    if (!file.exists(mmproj_path)) {
        stop("llamaR: mmproj file does not exist: ", mmproj_path)
    }
    .Call(r_mtmd_init, model, mmproj_path, as.integer(n_threads),
          as.logical(use_gpu))
}

#' Does this multimodal context support vision (images)?
#'
#' @param mctx An mtmd context from \code{\link{llama_mtmd_load}}.
#' @return Logical scalar.
#' @export
llama_mtmd_support_vision <- function(mctx) {
    stopifnot(inherits(mctx, "externalptr"))
    .Call(r_mtmd_support_vision, mctx)
}

#' Does this multimodal context support audio?
#'
#' @param mctx An mtmd context from \code{\link{llama_mtmd_load}}.
#' @return Logical scalar.
#' @export
llama_mtmd_support_audio <- function(mctx) {
    stopifnot(inherits(mctx, "externalptr"))
    .Call(r_mtmd_support_audio, mctx)
}

#' Media marker string for multimodal prompts
#'
#' Returns the placeholder token that must appear in a prompt where the image
#' (or audio) should be injected. Pass a prompt containing exactly one marker
#' per media item to \code{\link{llama_image_eval}}.
#'
#' @return Character scalar, e.g. \code{"<__media__>"}.
#' @export
llama_mtmd_marker <- function() {
    .Call(r_mtmd_marker)
}

#' Set verbosity of the multimodal subsystem
#'
#' @param level Integer 0 (silent) .. 3 (verbose). Default 1 (errors only).
#' @return Invisibly \code{NULL}.
#' @export
llama_mtmd_set_verbosity <- function(level = 1L) {
    invisible(.Call(r_mtmd_set_verbosity, as.integer(level)))
}

#' Load an image file into an mtmd bitmap
#'
#' Decodes an image (jpg, png, bmp, gif, ...) into the bitmap representation the
#' encoder expects. Uses the vendored stb_image decoder.
#'
#' @param mctx An mtmd context from \code{\link{llama_mtmd_load}}.
#' @param path Path to the image file.
#' @return An external pointer wrapping the bitmap. Freed automatically by the
#'   garbage collector.
#' @export
llama_image_load <- function(mctx, path) {
    stopifnot(inherits(mctx, "externalptr"))
    stopifnot(is.character(path), length(path) == 1)
    if (!file.exists(path)) stop("llamaR: image file does not exist: ", path)
    .Call(r_mtmd_bitmap_from_file, mctx, path)
}

#' Evaluate an image + prompt into a llama context
#'
#' Tokenizes \code{prompt} (which must contain the media marker, see
#' \code{\link{llama_mtmd_marker}}) together with \code{bitmap}, encodes the
#' image, and decodes both the text and image chunks into the llama context's
#' KV cache. After this returns, continue generation with the usual
#' \code{\link{llama_gen_next}} loop, passing the returned position as the
#' starting \code{n_past}.
#'
#' @param mctx An mtmd context from \code{\link{llama_mtmd_load}}.
#' @param ctx A llama context from \code{\link{llama_new_context}} (created on
#'   the same text model the projector was loaded against).
#' @param prompt Prompt string containing exactly one media marker where the
#'   image should be injected.
#' @param bitmap A bitmap from \code{\link{llama_image_load}}.
#' @param n_past Starting position in the KV cache (default \code{0L} for a
#'   fresh context).
#' @return Integer: the new \code{n_past} after evaluation, to continue
#'   generation from.
#' @seealso \code{\link{llama_mtmd_marker}}, \code{\link{llama_image_load}}
#' @export
#' @examples
#' \dontrun{
#' model  <- llama_load_model("model.gguf")
#' mctx   <- llama_mtmd_load(model, "mmproj-model.gguf")
#' ctx    <- llama_new_context(model, n_ctx = 4096)
#' img    <- llama_image_load(mctx, "photo.jpg")
#' prompt <- paste0("Describe this image: ", llama_mtmd_marker())
#' n_past <- llama_image_eval(mctx, ctx, prompt, img)
#' # ... continue with llama_gen_next() from n_past
#' }
llama_image_eval <- function(mctx, ctx, prompt, bitmap, n_past = 0L) {
    stopifnot(inherits(mctx, "externalptr"))
    stopifnot(inherits(ctx, "externalptr"))
    stopifnot(is.character(prompt), length(prompt) == 1)
    stopifnot(inherits(bitmap, "externalptr"))
    .Call(r_mtmd_eval, mctx, ctx, prompt, bitmap, as.integer(n_past))
}
