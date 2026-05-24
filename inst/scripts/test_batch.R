suppressPackageStartupMessages(library(llamaR))

MODEL_PATH <- "/mnt/Data2/DS_projects/llm_models/tiny-mistral-test-Q2_K.gguf"
stopifnot(file.exists(MODEL_PATH))

llama_set_verbosity(0L)

cat("=== loading model ===\n")
model <- llama_load_model(MODEL_PATH, n_gpu_layers = -1L)

cat("\n=== TEST 1: same prompt twice, temp > 0 -> different texts ===\n")
ctx <- llama_new_context(model, n_ctx = 512L, n_seq_max = 2L,
                         flash_attn = "on")
res1 <- llama_generate_batch(ctx, rep("Hello", 2L),
                             max_new_tokens = 20L, temp = 0.7, seed = 42L)
print(res1)
t1_ok <- res1[[1]]$text != res1[[2]]$text
cat("DIFFERENT TEXTS:", t1_ok, "\n")
llama_free_context(ctx)

cat("\n=== TEST 2: prompts of different lengths ===\n")
ctx <- llama_new_context(model, n_ctx = 512L, n_seq_max = 2L,
                         flash_attn = "on")
res2 <- llama_generate_batch(
    ctx,
    c("Hi", "Explain quantum computing in detail:"),
    max_new_tokens = 20L, temp = 0.7, seed = 42L)
print(res2)
t2_ok <- nchar(res2[[1]]$text) > 0 && nchar(res2[[2]]$text) > 0
cat("BOTH NON-EMPTY:", t2_ok, "\n")
llama_free_context(ctx)

cat("\n=== TEST 3: N=1 batch vs single generate ===\n")
ctx <- llama_new_context(model, n_ctx = 512L, n_seq_max = 1L)
r_single <- llama_generate(ctx, "Hello",
                           max_new_tokens = 20L, temp = 0, seed = 42L,
                           mirostat = 0L)
cat("single :", r_single, "\n")
llama_free_context(ctx)

ctx <- llama_new_context(model, n_ctx = 512L, n_seq_max = 1L)
r_batch <- llama_generate_batch(ctx, "Hello",
                                max_new_tokens = 20L, temp = 0, seed = 42L)
cat("batch  :", r_batch[[1]]$text, "\n")
t3_ok <- identical(r_single, r_batch[[1]]$text)
cat("IDENTICAL:", t3_ok, "\n")
llama_free_context(ctx)

llama_free_model(model)

cat("\n=== summary ===\n")
cat("test 1 (different seeds):", t1_ok, "\n")
cat("test 2 (mixed lengths)  :", t2_ok, "\n")
cat("test 3 (N=1 vs single)  :", t3_ok, "\n")
if (!all(c(t1_ok, t2_ok, t3_ok))) quit(status = 1)
