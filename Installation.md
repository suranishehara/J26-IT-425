# Component 3 — Installation on a separate Windows laptop

This component needs a fresh local environment **and two existing shared Component 2 Python files**. A ZIP containing only Component 3 cannot satisfy that current sibling-source dependency by itself.

## 1. Required destination layout

Prefer short extraction paths and preserve component folder names:

```text
C:\FashionProject\
├── component-3-fabric-design\
│   ├── backend\
│   ├── frontend\
│   ├── config\
│   ├── data\
│   └── runtime\
└── component-2-fashion-recommendation\
    └── backend\
        ├── user_profile.py
        └── taxonomy.py
```

On the Component 3 laptop, extract Component 2 alongside it, or copy those **exact existing authoritative shared files** into the shown sibling layout. At the current checkpoint they use only standard-library dependencies; they do not require Component 2's `.venv`, models or a running Component 2 server. `backend/shared_contract.py` loads them by file location. Do not independently reimplement the shared measurement/taxonomy logic.

Preserve Component 3 model checkpoints, configurations, catalogue/specimen/profile data, prepared datasets/manifests, source licences and required runtime assets. Exclude the old `.venv`, not the whole `runtime` directory. Install **64-bit Python 3.12** first.

## 2. Preflight and new environment

Run on the destination laptop; replace the example directory if needed:

```powershell
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath 'C:\FashionProject\component-3-fabric-design'

py -3.12 --version
if ($LASTEXITCODE -ne 0) { throw 'Install 64-bit Python 3.12 first.' }

if (-not (Test-Path -LiteralPath '..\component-2-fashion-recommendation\backend\user_profile.py')) {
    throw 'Missing sibling Component 2 shared user_profile.py. Restore the required layout first.'
}
if (-not (Test-Path -LiteralPath '..\component-2-fashion-recommendation\backend\taxonomy.py')) {
    throw 'Missing sibling Component 2 shared taxonomy.py. Restore the required layout first.'
}
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

Keep this PowerShell window so the helper stays available. Activation is not required. Choose **one** following package branch.

## 3A. Recorded NVIDIA/CUDA environment

For a compatible NVIDIA machine, first confirm `nvidia-smi` works, then:

```powershell
Run-Venv -m pip install `
    'torch==2.14.1+cu126' `
    'torchvision==0.29.1+cu126' `
    --index-url https://download.pytorch.org/whl/cu126

Run-Venv -m pip install -r requirements-ml-lock.txt
```

The lock file restores the recorded ML dependency snapshot, including timm. Do not also install the CPU branch.

## 3B. CPU-only alternative

For a laptop without NVIDIA CUDA, use this **instead of 3A**:

```powershell
Run-Venv -m pip install `
    'torch==2.14.1' `
    'torchvision==0.29.1' `
    --index-url https://download.pytorch.org/whl/cpu

Run-Venv -m pip install -r requirements.txt
Run-Venv -m pip install -r requirements-ml.txt
```

Do not install the CUDA-pinned ML lock file afterward. The current application's classifier inference uses CPU; the recorded training workflows require CUDA. This alternate dependency resolution is not a frozen CPU snapshot or a destination performance guarantee. Installing only minimal `requirements.txt` is not sufficient to restore all three classifier implementations.

## 4. Verify dependencies, shared modules and rules

```powershell
Run-Venv -m pip check
Run-Venv -c "import torch, torchvision, timm, PIL, safetensors; from backend.shared_contract import contract; print('ML imports passed'); print('Torch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('Shared taxonomy:', contract()['taxonomy_version'])"
Run-Venv scripts\check_design_rules.py
Run-Venv -m unittest discover -s tests -q
```

Some optional checks may require additional runtime assets or Node. Record failures/skips rather than asserting that every feature passed because imports succeeded.

## 5. Start and test

```powershell
.\.venv\Scripts\python.exe backend\app.py
```

Open **http://localhost:8002/**. Keep the terminal open; Ctrl+C stops this server.

Test with a permitted fabric photo, check that the trained model is available, analyse it, apply the category, enter/reuse appropriate fabric-property information, and request design suggestions with known measurements. Use the examples and limitations in [README.md](README.md).

The default trained classifier is ConvNeXt V2-Tiny with nine categories. Restoring libraries does not recreate a missing checkpoint. Preserve model/report/configuration identities and full runtime paths. Do not move linen/leather/nylon image folders just because their historical parent directory says `unsupported_fabrics`; frozen manifests reference those paths.

## 6. Troubleshooting and transfer boundaries

- **Component 2 shared module missing:** fix the sibling-source layout in Section 1; creating `.venv` cannot fix absent files.
- **Model unavailable:** inspect the configured checkpoint/report and required caches; do not rename weights or change label order to bypass validation.
- **Old absolute path error:** inspect the exact path in configuration/manifests. Migrate deliberately, preserving provenance, not a blanket replacement across frozen data.
- **Port 8002 occupied:** first check whether the correct server is already running; do not terminate all Python processes.
- **Activation-policy error:** use the explicit environment interpreter; no machine-wide policy change is necessary.
- **Dependency/distribution error:** stop at the failure and record Python/Windows architecture, hardware and complete error.
- Preserve non-commercial model/dataset terms and private specimen records. The installation does not validate physical fabric properties or model accuracy on a new laptop.

These instructions rebuild dependencies without changing the paused component's implementation or training outputs. They do not claim destination installation has already been executed.
