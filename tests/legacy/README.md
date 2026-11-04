# Native30 regression tests

These 22 tests were moved from the experiment root. Configuration fixtures are
exact copies from `archive/retired_runs_oct16/root_records`; historical model
and source-bundle paths remain rooted at the original experiment directory.
The still-sealed `test_native30_refinement.py` remains at its original path.

Four old pipeline integration tests require the original interpolated/5 Hz
archive, which was unavailable before this cleanup. They skip explicitly when
`SOURCE/READY.json` identifies the recovered native protocol. The two independent
rotation/control tests still run. Two CUDA equivalence checks skip in CPU-only
runs. No recovered data is substituted into those historical tests.

The RICH split test now verifies official TEST provenance and disjoint validation
identities. Its old assertion that RICH TEST must be empty predated the official
TEST loader and was stale.

Run from the repository root with the documented resource wrapper:

```bash
env CUDA_VISIBLE_DEVICES= python -m unittest discover -s tests -t .
```

The current local suite includes licensed data integration checks and needs the
documented sibling repository and local dataset assets. Optional new SMPL-H and
EgoAllo checks skip when those assets are absent; this is not a standalone CI
dataset release.
