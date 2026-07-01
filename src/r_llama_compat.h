// R compatibility header for llamaR
// Redirects C stdio functions to R-safe equivalents
// This header is force-included via -include in Makevars

#ifndef R_LLAMA_COMPAT_H
#define R_LLAMA_COMPAT_H

#include <stdint.h>
#include <R.h>
#include <R_ext/Print.h>

// Pull in the C++ standard headers that call std::fflush / std::fprintf
// internally (libc++'s <fstream> flushes via std::fflush in basic_filebuf::sync)
// BEFORE our function-like macros below are defined. Once included here, their
// include guards make the .cpp's own #include a no-op, so the macros never
// rewrite the std:: qualified calls inside them (which would produce
// "std::((stream == ...) ? 0 : fflush(...))" -> a syntax error on macOS/libc++).
#ifdef __cplusplus
#include <cstdio>
#include <fstream>
#include <ostream>
#include <iostream>
#endif

// Override stderr/stdout to prevent direct usage
// R CMD check flags these as non-portable
#ifdef stderr
#undef stderr
#endif
#ifdef stdout
#undef stdout
#endif

// Sentinel file pointers for our fprintf/fputs wrappers below.
// Using a non-zero dummy address avoids -Wnonnull warnings from gcc
// when fprintf(stderr, ...) is expanded.
static FILE *const r_llama_dummy_stream_ = (FILE*)(void*)(intptr_t)1;
#define stderr r_llama_dummy_stream_
#define stdout r_llama_dummy_stream_

// Wrapper for fprintf to stderr -> REprintf
#define fprintf(stream, ...) \
    ((stream == r_llama_dummy_stream_) ? (REprintf(__VA_ARGS__), 0) : fprintf(stream, __VA_ARGS__))

// Wrapper for fputs to stderr -> REprintf
#define fputs(str, stream) \
    ((stream == r_llama_dummy_stream_) ? (REprintf("%s", str), 0) : fputs(str, stream))

// Redirect printf -> Rprintf. R CMD check flags __printf_chk (from plain
// printf) as non-portable; route it through the R console instead.
#undef printf
#define printf(...) Rprintf(__VA_ARGS__)

// Wrapper for fflush: flushing our sentinel stream (stdout/stderr redirected to
// (FILE*)1) would call libc fflush on a bogus pointer -> SIGSEGV. The R console
// is flushed by REprintf/Rprintf already, so make it a no-op for the sentinel.
// NB: this is required for code that flushes stdout/stderr from a background
// thread (e.g. common_log's worker in log.cpp), where the crash is asynchronous.
#define fflush(stream) \
    ((stream == r_llama_dummy_stream_) ? 0 : fflush(stream))

// Override exit/_Exit to prevent process termination (CRAN requirement)
static inline void r_llama_exit(int status) {
    Rf_error("llama: exit called with status %d", status);
    while(1) {} // Rf_error never returns, but silence compiler warnings
}

#undef exit
#define exit(status) r_llama_exit(status)

#undef _Exit
#define _Exit(status) r_llama_exit(status)

#endif // R_LLAMA_COMPAT_H
