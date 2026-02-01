// R compatibility header for llamaR
// Redirects C stdio functions to R-safe equivalents
// This header is force-included via -include in Makevars

#ifndef R_LLAMA_COMPAT_H
#define R_LLAMA_COMPAT_H

#ifdef __cplusplus
extern "C" {
#endif

#include <R.h>
#include <R_ext/Print.h>

// Override stderr/stdout to prevent direct usage
// R CMD check flags these as non-portable
#ifdef stderr
#undef stderr
#endif
#ifdef stdout
#undef stdout
#endif

// Dummy file pointers that will cause compile errors if used directly
// This forces code to use our wrapper functions instead
#define stderr ((FILE*)0)
#define stdout ((FILE*)0)

// Wrapper for fprintf to stderr -> REprintf
#define fprintf(stream, ...) \
    ((stream == (FILE*)0) ? (REprintf(__VA_ARGS__), 0) : fprintf(stream, __VA_ARGS__))

// Wrapper for fputs to stderr -> REprintf
#define fputs(str, stream) \
    ((stream == (FILE*)0) ? (REprintf("%s", str), 0) : fputs(str, stream))

// fflush is a no-op for our dummy streams
#define fflush(stream) \
    ((stream == (FILE*)0) ? 0 : fflush(stream))

#ifdef __cplusplus
}
#endif

#endif // R_LLAMA_COMPAT_H
