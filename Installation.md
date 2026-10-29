# Component 2 — Installation on a separate Windows laptop

Create a fresh environment after extracting the complete transfer archive. Do not recreate the original working environment just to follow this guide. Choose **one** GPU or CPU package branch below.

## 1. Prerequisites and files

- **64-bit Python 3.12**, Windows and PowerShell.
- Compatible NVIDIA GPU/driver for the recorded CUDA setup; CPU inference can be used with the alternate branch, with different performance.
- Example extraction path: `C:\FashionProject\component-2-fashion-recommendation`.
- Transfer source, configuration, catalogue/data, model caches, trained adaptation checkpoints, reports and licences as required. Do not exclude the entire `runtime` folder.
- Exclude the old `.venv`; each laptop gets its own environment. Retain complete pretrained-cache/checkpoint structures, not only individual files.

```powershell
py -3.12 --version
```

For NVIDIA installation also check:

```powershell
nvidia-smi
```

## 2. Create the new environment — both hardware branches

Run in a new PowerShell window on the destination. Change the folder path if necessary.

```powershell
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath 'C:\FashionProject\component-2-fashion-recommendation'

if (Test-Path -LiteralPath '.venv') {
    throw 'A .venv already exists. Inspect it rather than overwriting it.'
}

py -3.12 -m venv .venv
if ($LASTEXITCODE -ne 0) { throw 'Environment creation failed.' }

function Run-Venv {
    & .\.venv\Scripts\python.exe @args
    if ($LASTEXITCODE -ne 0) { throw 'Python command failed. Stop and inspect the error.' }
}

Run-Venv -m pip install --upgrade pip setuptools wheel
```

Activation is unnecessary; the explicit environment interpreter is used. Keep this PowerShell window for the following blocks so `Run-Venv` remains defined.

## 3A. NVIDIA/CUDA package installation

Use this branch for a compatible NVIDIA machine, not a CPU-only laptop:

```powershell
Run-Venv -m pip install -r requirements-gpu.txt
Run-Venv -m pip install -r requirements-lock.txt
```

The recorded GPU files pin CUDA 12.6 PyTorch `2.14.1+cu126` and torchvision `0.29.1+cu126`, with the recorded application dependency snapshot. Do not additionally run the CPU branch.

## 3B. CPU-only alternative

Use this **instead of 3A**, after Section 2:

```powershell
Run-Venv -m pip install `
    'torch==2.14.1' `
    'torchvision==0.29.1' `
    --index-url https://download.pytorch.org/whl/cpu

Run-Venv -m pip install -r requirements.txt
```

Do not subsequently install the GPU file or CUDA-pinned lock file into that CPU environment. The base requirements retain core application pins but are not a complete frozen CPU environment. This is an alternate dependency setup that must be verified, not a claim of identical GPU performance. GPU training scripts do not automatically become CPU-compatible.

## 4. Verify

```powershell
Run-Venv -m pip check
Run-Venv scripts\verify_environment.py
Run-Venv -m unittest discover -s tests -q
```

The environment verifier checks imports and, when available, a small actual CUDA operation without downloading model weights. CPU false is expected for the CPU branch. Read model availability and test failures/skips; these checks are not proof that all transferred caches/checkpoints are present or that every destination feature works.

## 5. Start and perform a real application check

```powershell
.\.venv\Scripts\python.exe backend\app.py
```

Open **http://localhost:8001/**. Keep the terminal open; Ctrl+C stops that server.

Test both catalogue/profile recommendation and the configured image-analysis paths using permitted photos. A page that loads successfully is not sufficient verification of CLIP/DINOv2/local-head/person-analysis assets. Required pretrained downloads may need internet if caches are absent; upstream access/licence requirements still apply.

Check any stale absolute dataset/model paths individually. Preserve original paths inside versioned manifests until a controlled migration is defined; do not blindly replace all occurrences of the old user directory or rename files to bypass hashes.

## 6. Transfer note for Component 3's laptop

Component 3 currently imports these existing Component 2 shared source files from a sibling folder:

```text
C:\FashionProject\component-2-fashion-recommendation\backend\user_profile.py
C:\FashionProject\component-2-fashion-recommendation\backend\taxonomy.py
```

When preparing the Component 3 laptop, make those authoritative files available in that layout as well (or extract the Component 2 folder alongside it). That shared use does **not** require a running Component 2 server, its model files or a second Component 2 environment. Do not rewrite the measurement/taxonomy implementation to hide missing shared files. See Component 3's Installation.md.

## 7. Troubleshooting and safety

- Use the actual destination directory in Set-Location; commands above are not intended for a ZIP preview window.
- Stop on pip/import errors; record the exact failing command rather than substituting packages randomly.
- If an old transferred `.venv` is present, inspect/preserve it manually rather than overwriting a working environment.
- For occupied port 8001, inspect/reuse the correct server; do not terminate every Python process.
- CPU inference can be slower. No universal performance or hardware compatibility is guaranteed by pip success.
- Preserve private photo/profile data appropriately; only transfer data/models you are permitted to distribute.

Full feature/model/dataset details: [README.md](README.md). This installation guide does not modify trained models, data or the saved checkpoint and does not claim that another laptop has already been tested.
