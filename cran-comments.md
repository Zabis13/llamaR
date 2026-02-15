## R CMD check results

0 errors | 0 warnings | 0 notes

## Notes

* This package includes C++ source code from the 'llama.cpp' library
  (MIT license) by Georgi Gerganov, bundled in `src/`.
  The copyright holder is listed in `Authors@R` with the `cph` role.

* Examples that require a pre-trained model file are wrapped in
  `\donttest{}` since model files are not available on CRAN.

* The `Remotes` field in DESCRIPTION is used for development and CI only.
  The dependency 'ggmlR' will be submitted to CRAN prior to this package.

## Test environments

* local: Ubuntu 24.04, R 4.4.x, GCC 14
