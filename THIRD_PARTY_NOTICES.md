# Third-Party Notices

This repository contains AdaptiveIMI code plus selected third-party code and
runtime dependencies. The top-level `LICENSE` applies to AdaptiveIMI original
code only. Third-party components remain under their own licenses.

## Vendored or Adapted Code

### NVIDIA RULER

- Paths: `benchmark/ruler/**`
- License: Apache License 2.0
- License text: `licenses/Apache-2.0.txt`
- Notes: files retain NVIDIA copyright and Apache-2.0 headers. Local changes
  adapt benchmark configuration, data paths, prediction, and evaluation flow
  for AdaptiveIMI experiments.

## Runtime or Build Dependencies

### FlashInfer

- Used by: sparse decoding/runtime support as installed in `README.md`
- License: Apache License 2.0
- License text: `licenses/Apache-2.0.txt`
- Notes: FlashInfer is installed as an external dependency and is not vendored
  as source in this repository.

### NVIDIA CUTLASS

- Used by: CUDA extension builds and FlashAttention build flow
- License: BSD 3-Clause
- License text: `licenses/BSD-3-Clause-CUTLASS.txt`
- Notes: CUTLASS is fetched as an external checkout in `README.md` and is not
  vendored in this repository.
