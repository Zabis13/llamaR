library(testthat)
library(llamaR)

# Heavy tests load a real GGUF model (and some spawn HTTP servers): too slow and
# too resource-hungry for CRAN, which has no model file anyway. They are listed
# explicitly below and run only locally, where devtools::test() / R CMD check
# with NOT_CRAN=true is in effect. On CRAN we filter them out so the files are
# never sourced. (Light test files guard their own model-dependent blocks with
# skip_if_no_model(); the heavy files hold the parts that *require* a model.)
heavy <- c(
  "chat-build-roundtrip",   # llama_chat_build/parse build->generate->parse
  "serve-anthropic-e2e",    # llama_serve_anthropic spawned + driven over HTTP
  "lora-multi",             # LoRA multi-adapter apply/remove/clear contract
  "core-extra"              # generate_batch / memory_seq_add / embeddings_seq
)

on_cran <- !identical(Sys.getenv("NOT_CRAN"), "true")

if (on_cran) {
  test_dir <- if (dir.exists("testthat")) "testthat" else "tests/testthat"
  all_tests <- list.files(test_dir, pattern = "^test-.*\\.R$")
  all_names <- sub("^test-(.*)\\.R$", "\\1", all_tests)
  light <- setdiff(all_names, heavy)
  # testthat applies `filter` as grepl(filter, <name>) with no anchors; anchor
  # each name with ^...$ so e.g. "chat" can't also match "chat-build".
  filter_regex <- paste0("^(", paste(light, collapse = "|"), ")$")
  test_check("llamaR", filter = filter_regex)
} else {
  test_check("llamaR")
}
