# Response to Reviewer R2D1

We performed a functional attribution analysis using five approximately 128K-token inputs under the same single-GPU setting. The results indicate that AdaptiveIMI's advantage comes from three architectural factors:

1. **Reduced CPU participation and Host-Device round trips.** AdaptiveIMI performs cache planning on the GPU and updates cache state asynchronously. This reduces the CPU-side lookup and synchronization path required by RetroInfer.

2. **Lower gather data movement.** AdaptiveIMI has a lower gather cost because its cache-aware execution path reduces the amount of KV data that must be assembled and transferred for attention.

3. **Higher steady-state cache hit ratio.** AdaptiveIMI reaches 83.19% block hit ratio and 82.55% token hit ratio in steady state, compared with 75.18% and 75.11% in the RetroInfer profile. The higher hit ratio reduces miss-page movement and directly supports the lower gather cost.

The detailed values are provided in `tpot_attribution_comparison.csv`. The CPU/Host-Device values are reported as functional attribution evidence rather than a strict additive decomposition because the two implementations expose different stage boundaries.
