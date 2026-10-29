# Component 4 — Installation on a separate Windows laptop

There are two different setups:

1. **Run the chatbot:** Python standard-library server plus separately installed Ollama for generated replies/semantic retrieval.
2. **Reproduce the standalone NLP experiment:** optional Python ML packages and an independent CUDA PyTorch installation.

The live chatbot's intent parser remains rule-based. DeBERTa is a saved native SIMMC dialogue-act experiment, not an Ollama chat model or automatically activated live parser.

## 1. Prerequisites and transfer

- Install **64-bit Python 3.12**, then open PowerShell.
- Extract to a short path such as `C:\FashionProject\component-4-conversational-ai`.
- Exclude the old `.venv`. Preserve source, configuration, `data/knowledge.json`, datasets/checkpoints/reports/licences as needed, and documentation/checkpoint files.
- Do not exclude the entire `runtime` folder if you need to retain trained research artifacts.
- Decide deliberately whether private chat history should be transferred. `runtime/chat_history.sqlite3` is unencrypted local data, not required to reproduce the classifier benchmark. Stop the original server before a consistent file-copy database backup and preserve companion files if present.
- Ollama's installation and normal model storage are outside this component folder. Transferring the ZIP does not automatically install them.

## 2. Create the chatbot's new environment

Run on the destination laptop. Replace the sample directory if required:

```powershell
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath 'C:\FashionProject\component-4-conversational-ai'

if (Test-Path -LiteralPath '.venv') {
    throw 'A .venv already exists. Inspect it rather than overwriting it.'
}

py -3.12 -m venv .venv
if ($LASTEXITCODE -ne 0) { throw 'Install 64-bit Python 3.12 and retry environment creation.' }

function Run-Venv {
    & .\.venv\Scripts\python.exe @args
    if ($LASTEXITCODE -ne 0) { throw 'Python command failed. Stop and inspect the error.' }
}

Run-Venv -m pip install --upgrade pip setuptools wheel
Run-Venv -m pip check
```

The baseline web server needs no extra pip dependencies. Activation is not required. Keep this PowerShell window for the helper, or use `.\.venv\Scripts\python.exe` explicitly in a new one.

## 3. Install and prepare Ollama

For natural generated answers, install [official Ollama for Windows](https://ollama.com/download/windows), open the installed application, and check in a new PowerShell window if the command was newly added to PATH:

```powershell
ollama --version
ollama list
```

If the following models are missing, download them:

```powershell
ollama pull qwen3:4b
ollama pull all-minilm:22m
```

`qwen3:4b` generates replies; `all-minilm:22m` produces embeddings. Downloads need internet/disk space. An installed model may be unloaded from memory normally; that does not mean it must be downloaded again. A newer runner/model tag is not guaranteed to be identical to the original recorded digest; preserve identities when comparing research results.

Ollama normally serves `127.0.0.1:11434`. Open the installed application, or use `ollama serve` in a separate terminal only when it is not already running. Do not start duplicate services. CPU inference may be slow; compatible GPU acceleration depends on hardware. The Python chatbot itself does not require CUDA.

## 4. Start and verify the chatbot

From the component directory:

```powershell
.\.venv\Scripts\python.exe backend\app.py
```

Open **http://localhost:8003/**. Keep that terminal open; Ctrl+C stops only this app's server. Click **Refresh models**, choose **Local language model — Ollama**, select **qwen3:4b**, and inspect the retrieval dropdown.

In another PowerShell window, these readiness checks do not create chat records:

```powershell
Invoke-RestMethod http://localhost:8003/api/health
Invoke-RestMethod http://localhost:8003/api/models
Invoke-RestMethod http://localhost:8003/api/retrieval
```

Try `Is satin always silk?`, then inspect the generated answer, source cards and actual method metadata. Test a saved chat's rename/reopen using a harmless demonstration conversation. For a baseline-only test choose templates + BM25; that path works without Ollama and intentionally replies quickly.

Follow the [complete user manual](USER_MANUAL.md) for 36 manual test specifications, supplied C2/C3 fixture JSON and troubleshooting. C2/C3 explanations in Component 4 are pasted, read-only JSON; their servers or Python environments are not needed for that feature.

## 5. Optional independent NLP training environment

Skip this section for ordinary chat use. The completed original experiment reused Component 3's PyTorch **read-only**. On a separate laptop install PyTorch in Component 4 itself, so Component 3 is not needed.

Check a compatible NVIDIA driver/GPU with `nvidia-smi`, then, in the Section 2 helper's PowerShell window:

```powershell
Run-Venv -m pip install `
    'torch==2.14.1+cu126' `
    --index-url https://download.pytorch.org/whl/cu126

Run-Venv -m pip install -r requirements-training.txt
Run-Venv -m pip check
Run-Venv -c "import torch, transformers, sklearn, sentencepiece; print('Training imports passed'); print('CUDA:', torch.cuda.is_available()); assert torch.cuda.is_available(), 'Recorded full fine-tuning requires CUDA'"
```

Optional training requirements pin Transformers 4.57.1, scikit-learn 1.7.2 and SentencePiece 0.2.1, with safetensors/protobuf dependencies. This is separate from the server's minimal requirements.

If deliberately reproducing training, use a **new** output directory and omit the original `--torch-site` argument:

```powershell
Run-Venv scripts\train_simmc_acts.py --output runtime\training\simmc_acts_destination_01
```

This is resource-intensive and is not an installation test. Preserve the original run `runtime/training/simmc_acts_03` and its scores rather than overwriting it. Source data/checkpoints must exist; setup code may download pinned pretrained weights if absent. The script requires CUDA for its configured full fine-tuning run; it does not silently switch that training job to CPU.

Inspecting saved JSON reports needs neither CUDA nor retraining. Do not activate the trained native-act model in the app merely because its packages are installed. Exact scope, configuration, recorded versions and limitations: [SIMMC_NLP_EXPERIMENT.md](docs/SIMMC_NLP_EXPERIMENT.md).

## 6. Software verification

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -q
.\.venv\Scripts\python.exe -m pip check
```

On a fresh minimal setup, optional ML/Node checks can skip. Read the test summary; the source machine's 85 passing tests are not a guarantee of identical coverage on a laptop missing optional dependencies. If Node is installed, also run:

```powershell
node --check frontend\app.js
```

A successful page load or import test is not an independent answer-quality or cultural-accuracy measurement.

## 7. Troubleshooting and preservation

- **Ollama unavailable:** open the installed runner, check `ollama list`, then Refresh models. Do not assume the ZIP carried its system installation/model storage.
- **Semantic/hybrid disabled:** check all-minilm availability; baseline BM25 remains usable.
- **No DeBERTa in dropdown:** expected; it is an encoder experiment, not an Ollama answer generator.
- **Old `.venv` exists:** inspect/preserve it; do not overwrite an original working environment. Explicit interpreter commands avoid activation-policy changes.
- **Port 8003 occupied:** inspect/reuse the correct running application; do not terminate every Python process.
- **Shared PyTorch path missing:** with Section 5 completed, omit `--torch-site`; do not point at another laptop's absolute path.
- **Slow or failed generation:** inspect readiness and GPU/RAM pressure; avoid training and chatting simultaneously. Do not present an undisclosed template fallback as Qwen3 output.
- **Missing dataset/model path:** inspect the exact error and source manifests/configuration before making a controlled migration; do not blindly rewrite frozen hashes/paths.

Runtime is excluded from Git. Keep datasets, checkpoints, provenance/licences, reports and any required private database backups separately. Ordinary chats are not automatically research/training data, and Delete only moves them to Recently deleted. This guide adds documentation only; it does not recreate environments on the source computer or claim that another laptop has already passed installation.
