## R CMD check results

0 errors | 0 warnings | 4 notes

## Notes

* **installed package size is 91.0Mb**
  — The package bundles C++ source code from the llama.cpp library
  which compiles to a large shared object. This is unavoidable
  for a local LLM inference engine.

* **unable to verify current time**
  — Transient network issue during check, not a package problem.

* **GNU make is a SystemRequirements**
  — GNU make is declared in DESCRIPTION SystemRequirements and is
  required for the build system.

* **Compilation used the following non-portable flag(s): '-mno-omit-leaf-frame-pointer'**
  — This flag is not set by the package. It comes from the system R
  configuration (Makeconf) and is outside the package's control.

## General notes

* This package includes C++ source code from the 'llama.cpp' library
  (MIT license) by Georgi Gerganov, bundled in `src/`.
  The copyright holder is listed in `Authors@R` with the `cph` role.

* Examples that require a pre-trained model file are wrapped in
  `\donttest{}` since model files are not available on CRAN.

## Test environments

* local: Ubuntu 24.04, R 4.4.x, GCC 14
* win-builder: r-devel-windows-x86_64
* Debian: r-devel-linux-x86_64-debian-gcc
