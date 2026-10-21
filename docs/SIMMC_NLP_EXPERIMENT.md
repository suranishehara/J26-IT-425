# Native SIMMC dialogue-act model comparison

## Research task and scope

Predict the **current user's native SIMMC dialogue act**, using current text and the immediately preceding completed user/assistant exchange. This is supervised NLP classification, not an LLM prompt demonstration. It is a text-only adaptation, **not the full official multimodal SIMMC benchmark**, and does not measure Sri Lankan cultural knowledge, token-level NER, complete dialogue-state tracking or explanation quality.

Native classes (stable sorted output order):

```text
ASK:GET
INFORM:DISAMBIGUATE
INFORM:GET
INFORM:REFINE
REQUEST:ADD_TO_CART
REQUEST:COMPARE
REQUEST:GET
```

These are not equivalent to the chatbot's seven existing intent labels. No classifier from this experiment is automatically activated in the chatbot, and predicting `ADD_TO_CART` does not execute an action.

## Fixed input and leakage policy

Both learned methods receive the same formatted text:

```text
Current user: <first 64 whitespace-delimited words>
Previous user: <last 24 words of preceding user turn, if present>
Previous assistant: <last 32 words of preceding assistant turn, if present>
```

No current assistant response, act name, slot value, object annotation, ambiguity label or future turn is included in the input. Native current user acts are targets only. Previous assistant text is legitimate past conversation context but is the dataset's reference response; this is not an autonomous end-to-end generated-dialogue evaluation. Only one preceding exchange is supplied. TF-IDF consumes this entire budgeted string; DeBERTa additionally truncates to 128 tokens, with the current user first so old context is truncated first.

## Data and selection protocol

Prepared source: `runtime/datasets/simmc21_fashion_01`; input SHA-256 values are verified against its audit before loading each partition.

| Partition | Conversations | Labelled user turns | Use |
|---|---:|---:|---|
| train | 4,667 | 27,567 | All fitting |
| dev | 563 | 3,494 | Hyperparameter/checkpoint selection |
| devtest | 1,172 | 6,549 | Assessment after selection is frozen |

No random turn-wise repartitioning. No fitting vocabulary, model weights or class mapping on devtest. The training script opens that file only after saving a selection-freeze record; intake had already inspected counts/schema. The official `teststd` is not downloaded or evaluated. Do not call this an untouched official final-test result.

Dataset licence: **CC-BY-NC-SA-4.0**, including attribution and restrictions. See [SIMMC_DATASET.md](SIMMC_DATASET.md) for the source citation, upstream commit, mixed/synthetic data caveats and repeated-text audit.

## Compared methods

1. **Majority class:** choose the most frequent training label; no fitted language model.
2. **TF-IDF + multinomial logistic regression:** word 1–2 grams, character-within-word 3–5 grams, up to 50,000 features per branch; minimum document frequencies 2 and 3 respectively. Fit on train only. Candidate regularization `C={0.5,2.0}`, at most 500 iterations, seed 42. Choose validation macro-F1; exact ties retain the first candidate. Natural class distribution; no resampling.
3. **DeBERTa-v3-small fine-tuning:** official [Microsoft checkpoint](https://huggingface.co/microsoft/deberta-v3-small), revision `a36c739020e01763fe789b4b85e2df55d6180012`; newly initialized seven-label head and all encoder parameters trained. Three epochs, batch 16, max length 128, AdamW learning rate `2e-5`, weight decay `0.01`, linear schedule with 10% warm-up, gradient clipping 1.0, CUDA mixed FP16. Select epoch by validation macro-F1, first on ties. No test-based retry or tuning.

The provider's 44M figure is backbone-only; the embedding layer contributes approximately 98M more parameters. Do not describe the whole model as having only 44M parameters. Encoder checkpoint is MIT; the dataset's separate restrictions still apply. Pretrained weight SHA-256 from the official LFS metadata: `d4b0ebcc7799a2c2dde1a39199cb0ed709f2c6b63a48b8c05133b77955f271a9` (286,059,269 bytes).

This is a practical comparison, not a matched-compute architecture study or proof that neural models are always superior. One random seed does not estimate training-seed variability. The existing app rules cannot be fairly assigned native-act accuracy without a reviewed mapping; they remain unchanged rather than being falsely compared under incompatible labels.

## Artifacts and reproducibility

`scripts/train_simmc_acts.py` refuses existing run directories. Each run saves configuration, package/hardware identities, data hashes, label order, baseline candidates, progress and selected models. Completed runs additionally save per-class precision/recall/F1, confusion matrices, assessment predictions, checkpoint hashes and a completed report. The selected neural checkpoint is reloaded and checked against validation predictions before assessment.

Optional dependencies are in `requirements-training.txt`; serving the chatbot does not require them. This machine has limited disk space, so the existing Component 3 CUDA/PyTorch installation is used **read-only via an explicit dependency path**. Nothing is installed into or modified within that environment. New training libraries, pretrained source files and checkpoints live in Component 4. For a portable independent setup install an appropriate CUDA PyTorch build into Component 4 and omit `--torch-site`.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-training.txt
.\.venv\Scripts\python.exe scripts\train_simmc_acts.py --torch-site '..\component-3-fabric-design\.venv\Lib\site-packages' --output runtime\training\simmc_acts_REPRO
```

SentencePiece on this Windows installation cannot open the workspace's en-dash path directly. The small vocabulary is temporarily copied into a managed ASCII-path directory; the workspace and datasets are not renamed. Official model files remain in Component 4. A stalled library weight download was replaced by an explicit pinned HTTPS downloader with upstream SHA-256 verification and progress output.

Interrupted setup runs `simmc_acts_01` and `simmc_acts_02` are retained, not overwritten or presented as completed experiments. Run 01 completed baseline fitting but failed initial neural tokenizer setup; run 02 reused its baseline and was stopped during a stalled pretrained download. Run 03 uses the same fitted baseline after verifying task/data/seed compatibility. These were pre-assessment setup repairs, not performance-driven model selection.

`scripts/compare_simmc_results.py` adds paired whole-conversation bootstrap intervals after a completed run, with 1,000 replicates and seed 42. It never changes model selection. The intervals describe this fixed public sample/checkpoint, not seed variance, causal improvement, new speakers or Sri Lankan cultural performance.

## Current results

Completed run: **`runtime/training/simmc_acts_03`**, 19 October 2026. Selected linear `C=2.0`; selected neural **epoch 2**, using validation only. Checkpoint reload reproduced selected validation predictions. No production activation.

| Method | Validation accuracy | Validation macro-F1 | Assessment accuracy | Assessment macro-F1 |
|---|---:|---:|---:|---:|
| Majority class | See report | See report | 33.36% | 0.0715 |
| TF-IDF + logistic regression | 87.84% | 0.8621 | **87.46%** | **0.8598** |
| Fine-tuned DeBERTa-v3-small | 96.62% | 0.9612 | **97.53%** | **0.9724** |

Assessment contains **6,549 user turns in 1,172 conversations**. DeBERTa's observed accuracy difference versus TF-IDF is **+10.06 percentage points**, not a relative percentage and not a cultural-quality score. Validation epoch macro-F1 values were 0.951787, 0.961187 and 0.960497; the last epoch was not automatically chosen.

The 1,000-replicate paired whole-conversation bootstrap yields:

- TF-IDF assessment accuracy 95% interval: **86.74–88.25%**.
- DeBERTa assessment accuracy 95% interval: **97.15–97.89%**.
- DeBERTa assessment macro-F1 interval: **0.9681–0.9765**.
- Accuracy difference (DeBERTa minus TF-IDF) interval: **+9.26 to +10.78 percentage points**.

These are percentile sampling intervals for fixed models and the public assessment split, not uncertainty across training seeds or independently recruited speakers. No significance or causal claim beyond that design is implied.

### Assessment class-level results

| Native act | Support | DeBERTa F1 |
|---|---:|---:|
| ASK:GET | 736 | 0.9565 |
| INFORM:DISAMBIGUATE | 679 | 0.9978 |
| INFORM:GET | 668 | 0.9910 |
| INFORM:REFINE | 692 | 0.9357 |
| REQUEST:ADD_TO_CART | 956 | 0.9995 |
| REQUEST:COMPARE | 633 | 0.9463 |
| REQUEST:GET | 2,185 | 0.9802 |

The selected neural model makes 162 errors. Major confusions are REQUEST:GET → INFORM:REFINE (49), INFORM:REFINE → REQUEST:GET (34), REQUEST:COMPARE → ASK:GET (31), and ASK:GET → REQUEST:COMPARE (29). They are retained as reported weaknesses; no further tuning was done after seeing them.

### Repeated-input diagnostic

Of the 6,549 assessment turns, **131** have normalized formatted inputs already found in training; these are all REQUEST:GET. All three methods classify that easy slice correctly. This is a descriptive overlap observation, not a new independent benchmark.

On the remaining **6,418 novel formatted inputs**:

| Method | Accuracy | Macro-F1 |
|---|---:|---:|
| TF-IDF + logistic regression | 87.21% | 0.8590 |
| DeBERTa-v3-small | 97.48% | 0.9723 |

Novel means not identical under NFKC/case/whitespace normalization of the entire budgeted input. It does **not** prove semantic/template/source/speaker independence. This slice was computed after selection without changing weights or thresholds.

### Saved evidence

- `configuration.json`: initial run protocol, source-audit hash and dependency/hardware identities.
- `progress.json`: validation candidates and all neural epoch results.
- `tfidf_logistic.joblib`: selected fitted vocabulary/vectorizers/classifier (load only this trusted local artifact, never untrusted pickle/joblib files).
- `deberta/`: selected safetensors checkpoint, configuration, native label mapping and tokenizer vocabulary/configuration.
- `selection_frozen.json`: chosen candidates/checkpoint hash before the assessment file was opened by the trainer.
- `assessment_predictions.jsonl`: truth and predictions indexed by conversation/turn; no current reply or label was used as input.
- `report.json`: completed exact metrics, per-class reports, confusion matrices and hashes.
- `comparison_uncertainty.json`: paired confidence intervals and repeated-input diagnostic.

Selected model SHA-256: `c56cc44c9abd812e43e83b98babd95cbf756bf26bf0c7fbb140e74886867ad73`.

Baseline artifact SHA-256: `00c01fb3e15b57249811f1e0c333f74fc0a956c19e9bce75dc3f9ec74f9ac421`.

Assessment prediction SHA-256: `9bf67e7f7ce9df698ef5a37a93b343a5a0285a165f0830dd00e19d396db4bf38`.

Recorded software: PyTorch **2.14.1+cu126**, Transformers **4.57.1**, scikit-learn **1.7.2**, NumPy **2.5.4**; NVIDIA RTX 4050 Laptop GPU. Run-03 elapsed time was 1,204.375 seconds including setup/download but excluding the reused baseline's original fitting; it is not a model-compute comparison. Peak PyTorch allocated GPU memory was 3,016,863,232 bytes, not total system/GPU memory. **85 automated tests pass**, including input leakage, source hashes, metric arithmetic and validation-only selection checks.

To regenerate uncertainty in a completed fresh run (existing output is refused):

```powershell
.\.venv\Scripts\python.exe scripts\compare_simmc_results.py --run runtime\training\simmc_acts_REPRO --dataset runtime\datasets\simmc21_fashion_01
```

### Suitable panel statement

> We fine-tuned DeBERTa-v3-small for seven native fashion-shopping dialogue acts using SIMMC 2.1. Under a defined text-plus-previous-turn protocol, the selected model achieved 97.53% accuracy and 0.9724 macro-F1 on the provider devtest fashion subset, compared with 87.46% and 0.8598 for a TF-IDF logistic-regression baseline. This is a text-only component benchmark, not proof of Sri Lankan cultural reasoning or full multimodal performance.

## Required next evidence

Before making broader research claims: review held-out per-class confusions, preserve this run as a fixed first experiment, run a predeclared multi-seed comparison if resources permit, define compatibility with app intents separately, and evaluate local cultural/occasion queries using independent reviewed data. Do not add public benchmark conversations to the factual RAG corpus as if their catalogue statements were verified cultural facts.
