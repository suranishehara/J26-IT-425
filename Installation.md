# Component 1 — Installation on a separate Windows laptop

This guide creates a **new destination-laptop environment**. Do not run it over the original working installation. Dependencies, model assets and hardware are separate checks; successful pip installation alone does not prove image generation works.

## 1. Prerequisites and transfer

- Install **64-bit Python 3.12**, then open a new PowerShell window.
- Current CatVTON generation requires a compatible **NVIDIA CUDA GPU/driver**; CPU generation is not implemented in the current worker.
- Keep sufficient disk space for PyTorch, model weights and temporary/output images. Runtime memory requirements differ from download sizes.
- Extract the ZIP completely, preferably to `C:\FashionProject\component-1-virtual-tryon`. Change the path below if needed.
- Leave the original `.venv` out of the transfer ZIP. Preserve backend/frontend, scripts, configuration, `external/CatVTON`, `external/VITON-HD`, required runtime models/checkpoints and relevant data/licences.
- Installing Python packages does not download/recreate missing upstream source, dataset files or every checkpoint. Preserve full model-cache structure, not only isolated weight files.

Check:

```powershell
py -3.12 --version
nvidia-smi
```

If Python 3.12 is missing, install it first. If there is no compatible NVIDIA GPU/driver, stop before expecting the generation workflow to run. Do not install an arbitrary CUDA Toolkit as a substitute for the correct driver/PyTorch setup.

## 2. Create the environment and install packages

Run this block in PowerShell **on the destination laptop**. It refuses to reuse an existing `.venv` and checks native command exit codes.

```powershell
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath 'C:\FashionProject\component-1-virtual-tryon'

if (Test-Path -LiteralPath '.venv') {
    throw 'A .venv already exists. Inspect it; do not overwrite a working environment.'
}

py -3.12 -m venv .venv
if ($LASTEXITCODE -ne 0) { throw 'Environment creation failed.' }

function Run-Venv {
    & .\.venv\Scripts\python.exe @args
    if ($LASTEXITCODE -ne 0) { throw 'Python command failed. Stop and inspect the error.' }
}

Run-Venv -m pip install --upgrade pip setuptools wheel

Run-Venv -m pip install `
    'torch==2.14.1+cu126' `
    'torchvision==0.29.1+cu126' `
    --index-url https://download.pytorch.org/whl/cu126

Run-Venv -m pip install -r requirements-catvton.txt

Run-Venv -m pip install `
    'numpy==2.5.2' `
    'Pillow==12.3.0' `
    'opencv-python==5.0.0.93' `
    'torchgeometry==0.1.2' `
    'scipy==1.18.1' `
    'safetensors==0.8.0'

Run-Venv -m pip install -r requirements-viton.txt
Run-Venv -m pip check

Run-Venv -c "import torch, torchvision, diffusers, transformers, cv2, torchgeometry; print('Imports passed'); print('Torch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); assert torch.cuda.is_available(), 'CUDA required for current try-on worker'"
```

The extra image/VITON package versions match the recorded original environment. The CatVTON file pins Diffusers/Transformers and related core libraries. Some transitive requirements remain resolver-selected; retain the original records and test the destination rather than claiming guaranteed byte-identical environment reproduction.

Do not replace this narrower inference environment with the entire upstream CatVTON application dependency stack blindly. Activation is not required; explicit interpreter commands avoid PowerShell activation-policy changes.

## 3. Verify assets and software

```powershell
.\.venv\Scripts\python.exe scripts\verify_viton_assets.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -q
```

The VITON verifier checks that retained pipeline's assets, not every CatVTON generation requirement. Read failures/skips; do not continue as though an error were a pass.

Transferred configurations/manifests may refer to original paths. Inspect such errors deliberately; do not perform a blanket text replacement across frozen manifests or rename trained checkpoints to bypass validation. Model-host access permissions and upstream licence conditions still apply. First use may require internet if an asset/cache was not transferred.

## 4. Start and test

```powershell
.\.venv\Scripts\python.exe backend\app.py
```

Open **http://localhost:8000/**. Keep the terminal open. Use permitted sample images and complete an actual try-on generation to verify model execution, not just page loading. Read [README.md](README.md) for workflow/model details. Ctrl+C in that terminal stops this server.

## 5. Troubleshooting

- **Activation blocked:** use the explicit interpreter above; do not change machine-wide execution policy.
- **`.venv` already exists:** inspect it first. A copied old-laptop environment is not portable; preserve/rename it manually if needed before making a fresh one. Never delete an environment blindly.
- **CUDA false/out of memory:** check driver/GPU compatibility and other GPU workloads. The current generation path has no promised CPU fallback.
- **Missing model/source/data:** restore the correct assets with their configuration and provenance. Installing requirements does not supply them all.
- **Port 8000 occupied:** check whether the correct server already runs. Inspect its owner instead of terminating all Python processes.
- **No matching distribution/dependency error:** stop and record the full error, Python version, Windows architecture and GPU information. Do not randomly substitute versions and claim equivalence.

Installation commands need internet unless an appropriate offline wheel/model distribution has been prepared. Hardware and destination operation still require testing; this document is not a claim that installation has been executed on the other laptop.
