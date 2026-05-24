# diag_offload_profile.R — GPU offload + scheduler profiling probe
#
# Назначение: подтвердить, что веса реально выгружены на Vulkan, и снять
# профиль decode-шага (где уходит время: copyin / compute / cmdbuf).
#
# Запуск (env-флаги профилировки реализованы в ggmlR, по умолчанию OFF):
#   GGML_SCHED_PROF=1 Rscript inst/scripts/diag_offload_profile.R 2>prof.log
#     -> [CS_PHASE] copyin/compute_async/post мс/шаг; [SCHED_PROF] n_splits;
#        [CS_DUMP] полный список входов split с max n_inputs
#   GGML_VKG_PROF=1  ... -> [VKG_PROF] total/loop/build_graph внутри vk_graph_compute
#   LLAMA_VERBOSITY=3 ... -> печатать load_tensors лог (offloaded N/M layers,
#                            Vulkan0/CPU_Mapped buffer size, sched copies)
#   GGML_LLAMA_NO_PIPELINE=1 ... -> откат к multi-GPU-only pipeline (A/B замер)
#
# Ключевые маркеры в логе:
#   "offloaded 27/27 layers to GPU" + "Vulkan0 model buffer size" -> offload OK
#   "sched copies = N"                                            -> n_copies
#   [CS_PHASE] copyin=...ms                                        -> узкое место
#
# Контекст диагноза — см. memory project_gpu_speedup_diagnosis.

suppressMessages(library(llamaR))

MODEL <- "/mnt/Data2/DS_projects/llm_models/Ministral-3-3B-Instruct-2512-Q8_0.gguf"
if (!file.exists(MODEL)) {
  cand <- list.files("/mnt/Data2/DS_projects",
                      pattern = "Ministral.*Q8_0\\.gguf$",
                      recursive = TRUE, full.names = TRUE)
  if (length(cand)) MODEL <- cand[[1]]
}
cat("MODEL:", MODEL, "exists =", file.exists(MODEL), "\n")
stopifnot(file.exists(MODEL))

verb <- as.integer(Sys.getenv("LLAMA_VERBOSITY", "2"))
llama_set_verbosity(verb)

cat("--- backend devices ---\n");           print(llama_backend_devices())
cat("--- supports_gpu:", llama_supports_gpu(), "---\n")

cat("=== loading GPU model (n_gpu_layers = -1) ===\n")
m   <- llama_load_model(MODEL, n_gpu_layers = -1L)
ctx <- llama_new_context(m, n_ctx = 512L, n_threads = 2L)

cat("=== generate (60 tok, steady-state decode) ===\n")
invisible(llama_generate(ctx, "Hi", max_new_tokens = 60L, temp = 0))

cat("=== done ===\n")
