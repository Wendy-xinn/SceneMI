# SceneMI on RTX 5090

The upstream requirements are an environment export, not a consistent portable
installation specification. This setup covers SceneMI training, preprocessing
dependencies, and motion visualization; unrelated web/NLP applications from the
original export are not installed.

## Versions

- Conda environment: `scenemi`, Python 3.9.
- PyTorch 2.8.0 / torchvision 0.23.0, CUDA 12.8 (required for this GPU).
- PyTorch3D 0.7.8, commit `75ebeeaea0908c5527e7b1e305fbc7681382db47`,
  compiled with CUDA and architecture 12.0.
- NVIDIA Kaolin 0.18.0, official PyTorch 2.8.0 / CUDA 12.8 wheel.
- Conda `libglu` for pyrender; environment activation sets
  `LD_LIBRARY_PATH` to this environment's `lib` directory and
  `PYOPENGL_PLATFORM=egl` for offscreen rendering.
- Most application dependencies retain the upstream versions; see
  `requirements-rtx5090.txt`.

Do not reinstall the original requirements over this environment: its torch,
torchvision, CUDA libraries, triton and typing_extensions pins conflict with
the GPU-compatible stack. Its `kaolin==0.1` is not NVIDIA's functional library.

## Use

```bash
conda activate scenemi
cd /home/wenxin/projects/SceneMI
python check_environment.py
python -m train.train_diffusion_scenemib --help
```

Training still requires the dataset and SMPL body model assets. Import and
synthetic CUDA checks are not a full training run or a scientific reproduction
of the original old software stack.

Verified: pip check, training CLI help, imports, CUDA ViT forward/backward,
PyTorch3D KNN forward/backward, ChamferDistance, and Kaolin mesh check_sign.
Interactive viewers and full offscreen image rendering were not tested.
Preprocessing stops at body model loading because `body_models/` is absent;
the repository also has no local `dataset/` directory yet.

Two upstream source imports were corrected: preprocessing now imports
`utils.utils_transform`; the diffusion module uses the installed standard
`smplx.create` instead of the absent `smplx_model.create`. The original custom
module was not distributed, so equivalence beyond the visible standard API
cannot be guaranteed. Run preprocessing with:

```bash
python preprocess_dataset.py --help
```

## Rebuild CUDA extensions

Install PyTorch and the ordinary dependencies before building PyTorch3D.
`--no-build-isolation` requires setuptools, wheel, pybind11 and numpy to be
installed in the environment first. `setuptools==80.9.0` retains pkg_resources
used by old setup scripts.

The build performed here borrowed only the CUDA 12.8 toolkit from the existing
`crisp` environment; it did not change that environment. Cached extension wheels
are in `.wheels/`. A fresh installation can instead use its own CUDA 12.8 toolkit.

```bash
CUDA_HOME=/home/wenxin/miniconda3/envs/crisp \
PATH=/home/wenxin/miniconda3/envs/crisp/bin:$PATH \
CPLUS_INCLUDE_PATH=/home/wenxin/miniconda3/envs/crisp/targets/x86_64-linux/include \
LIBRARY_PATH=/home/wenxin/miniconda3/envs/crisp/targets/x86_64-linux/lib \
FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST=12.0 MAX_JOBS=4 \
python -m pip install --no-build-isolation -r requirements-rtx5090.txt
```
