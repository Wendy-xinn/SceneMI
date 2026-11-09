"""Body-aware scene refinement after a frozen EGO diffusion sampler.

The sampler, shape predictor and motion representation retain their 22k
weights. The trainable component operates on complete sampled motions.
"""
