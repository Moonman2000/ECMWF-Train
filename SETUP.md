# Setting up the course environment

This guide installs the environment for the training course on your machine using
[uv](https://docs.astral.sh/uv/). Your machine requires an NVIDIA GPU to fully work with the course material.

## What you need

- A Linux machine, x86_64 (Intel/AMD), with an **NVIDIA GPU**.
- An NVIDIA **driver** that supports CUDA 12.4 or newer. The driver is the
  software that lets programs talk to the GPU. This means driver version 550 or higher.
  Check with:
  ```bash
  nvidia-smi
  ```
  Look for `Driver Version: 5xx.xx` at the top.
- About 15 GB of free disk space (the environment is ~7 GB, plus a few GB for
  uv's download cache).
- Internet access.
- `git` and `wget` installed (`git --version`, `wget --version`).

## 1. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Open a new terminal (or run `source ~/.bashrc`), then check:

```bash
uv --version
```

## 2. Get the course material

```bash
git clone https://github.com/ecmwf-training/2026-ml-training.git
cd 2026-ml-training
```

## 3. Create the environment

A *virtual environment* (venv) is a private folder holding the Python and
packages for this course, so it does not disturb anything else on your machine.

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv -r requirements.txt
source .venv/bin/activate
```

- uv downloads Python 3.12 for you if it is missing.
- The install resolves 279 packages. With an empty cache it took about
  7.5 minutes. Later installs are much faster.
- Run `source .venv/bin/activate` again in every new terminal.

## 4. Make a Jupyter kernel

A *kernel* is the Python that a notebook runs on. Register the venv as one:

```bash
python -m ipykernel install --user --name ml-course --display-name "ML course (uv)"
jupyter lab
```

In each notebook, choose the kernel **ML course (uv)**
(Kernel menu, then "Change kernel").

## 5. Check the GPU

Save this as `check_gpu.py`, or paste it into `python`:

```python
import torch

print("torch:", torch.__version__)
print("CUDA (built with):", torch.version.cuda)
print("GPU available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    x = torch.ones(3, device="cuda")
    print("tensor op:", (x * 2).sum().item())  # expect 6.0
```

You should see torch `2.6.0+cu124`, CUDA `12.4`, `True`, your GPU name and `6.0`.

## 6. Course data

The notebooks download some files the first time you run them (internet needed):

- ECMWF open data (via `ecmwf-opendata`).
- The AIFS model checkpoint (notebook `3-run-AIFS`, uses `wget`).
- `earthkit-regrid` interpolation weight files.
- Cartopy map shapefiles (Natural Earth).

**Toy training dataset.** The training notebook (`5-Anemoi-train`) needs a small
zarr dataset. Get it from the course organisers, then set its location in
`5-Anemoi-train/configs/system/course.yaml` (around line 9, the dataset path;
the default is `/datasets/aifs-ea-an-oper-0001-mars-o48-2010-2022-6h-v1-toy.zarr`).
Training outputs are written to `$HOME/anemoi-output`.

## Optional: flash-attn

Only needed for the `transformer*` model configs. The notebooks use
`graphtransformer` with the triton backend, so you can skip this.

```bash
uv pip install --python .venv "flash-attn @ https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.1.post2/flash_attn-2.7.1.post2+cu12torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl"
```

## Troubleshooting

**`torch.cuda.is_available()` is `False`.** Usually the driver is too old
(below 550) or you are on a machine without a GPU (for example an HPC login
node). Check `nvidia-smi`. If it fails, ask your system administrator for a
driver update or use a GPU node.

**Home directory quota is small.** uv caches downloads in `~/.cache/uv`.
Point it elsewhere before installing:

```bash
export UV_CACHE_DIR=/path/with/space/uv-cache
```

You can also clone the course repo (and so `.venv`) on that disk.

**Wrong Python version.** Always create the venv with
`uv venv --python 3.12 .venv`. Check with `python --version` after activating.
If you see another version, you are probably not in the activated venv.