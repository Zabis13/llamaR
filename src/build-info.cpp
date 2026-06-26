// Build-info constants for the vendored llama.cpp common/ layer.
//
// Upstream generates this file from common/build-info.cpp.in at CMake
// configure time. llamaR builds with R's Makevars (no CMake), so we provide a
// static definition for the build tag b7898. common.h declares these as
// `extern` and references LLAMA_BUILD_NUMBER/LLAMA_COMMIT during static
// initialisation of `build_info`, so they must be defined in exactly one TU.

int LLAMA_BUILD_NUMBER = 7898;
char const *LLAMA_COMMIT = "b7898";
char const *LLAMA_COMPILER = "";
char const *LLAMA_BUILD_TARGET = "";
