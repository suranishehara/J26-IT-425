# Component 4 — Saved stop point

**Saved:** 19 October 2026, approximately 22:36 Asia/Colombo.  
**Status:** implementation paused at the user's request; resume only on a later user instruction.  
**Component root:** `C:\Users\Ravindu\Videos\Component 1 – Virtual Try-On and Draping\component-4-conversational-ai`.

This is the resume record for completed work, exact saved artifacts, integration boundaries and subsequent tasks. **It is not a separate backup or a claim that the complete research project is finished.** Existing source/data/models were preserved; no cleanup, retraining, model activation or process termination was performed to make this checkpoint. Components 1–3 retain their existing stop points.

## 1. Read first when resuming

1. This checkpoint.
2. [README.md](README.md): rewritten end-to-end completed implementation, models, datasets and measured results, without the old contradictory milestone narrative.
3. [USER_MANUAL.md](USER_MANUAL.md): complete operating instructions, sample messages, validated fictional C2/C3 JSON and **36 manual test specifications**.
4. [SIMMC_NLP_EXPERIMENT.md](docs/SIMMC_NLP_EXPERIMENT.md): trained NLP protocol, results, scope and panel-ready wording.
5. [RESEARCH_PLAN.md](docs/RESEARCH_PLAN.md): wider research questions and remaining evaluation scope.

Most important boundary: **the live chatbot still uses rule-based app intent/entities. The trained DeBERTa predicts native SIMMC acts and is not integrated into that request path.** Do not describe it as already powering the chatbot's intent panel.

## 2. Completed live application

- Python loopback service at `http://localhost:8003/`; responsive HTML/CSS/JavaScript frontend.
- English app intents: recommend, explain, compare, update_preferences, knowledge_question, greeting, clarify.
- Rule-based terminology aliases, LKR budget, occasion/weather/setting/style/colour extraction, included/excluded terms and retained state.
- Recognized follow-up topic reuse and supplied-option numbering/ordinals; two-option “other one”; deterministic missing/ambiguous-option clarifications.
- BM25, local semantic cosine search and BM25+semantic RRF (constant 60), with scoped source cards and corpus/model-identity caching.
- Local Qwen3 4B generation through Ollama and separately selectable template baseline. Real NDJSON streaming, Stop response, native thinking/content separation, bounded history and no hidden generation fallback.
- Read-only supplied Component 2 recommendation / Component 3 design JSON explanations; no automatic upstream service execution or preference changes.
- Left chat history rail: New chat, reopen/continue, rename, recoverable delete, Recently deleted and restore. Only one New chat button.
- SQLite completed-turn persistence and bounded resume snapshots; latest-output scrolling; revision and concurrency safeguards.
- Source-ID diagnostics, not a factual-entailment verifier. Unsupported inputs can still retrieve unrelated evidence; retain this known limitation.

## 3. Models, runtime and scope

| Asset | Saved/current role |
|---|---|
| Ollama 0.40.5 | Installed local runner |
| `qwen3:4b` | Installed pretrained answer generator; not fine-tuned here |
| `all-minilm:22m` | Installed pretrained local embedding model; not fine-tuned here |
| TF-IDF + logistic regression | Trained native-act benchmark baseline |
| DeBERTa-v3-small | Fine-tuned native-act benchmark; saved selected checkpoint, not activated in app |

Runner/model setup identities are in `runtime/setup/ollama_qwen3_4b_01/report.json` and `runtime/setup/all_minilm_22m_01/report.json`.

Component 4 has its own Python environment and optional training libraries. The trained run reused Component 3's installed CUDA/PyTorch packages **read-only**, via `--torch-site`; nothing was installed into or changed in Component 3. Training package/hardware versions are recorded in the completed report. A portable standalone environment needs an independently installed compatible CUDA PyTorch runtime.

## 4. Saved datasets and knowledge

### Curated knowledge

- `data/knowledge.json`, `curated-0.3`: **23 active passages**, seven Sri Lankan, fourteen global/contextual and two project-implementation passages.
- Eleven external source URLs; scoped publisher summaries and source-and-scope review recorded.
- External passages are source-checked, **not cultural-expert-approved**.
- See [KNOWLEDGE_REVIEW.md](docs/KNOWLEDGE_REVIEW.md).

### SIMMC 2.1

- Official upstream commit: `6e3de4e3d11e2914edc3ab9123e6b0eb88933237`.
- Raw: `runtime/datasets/raw/simmc21_official_01/` (download manifest, official dialogue files, fashion metadata and licence).
- Prepared: `runtime/datasets/simmc21_fashion_01/` (train/dev/devtest fashion JSON and audit).
- **6,402 fashion conversations, 37,610 user/assistant pairs, 75,220 utterances**.
- Training: 4,667 conversations / 27,567 user turns.
- Validation: 563 conversations / 3,494 user turns.
- Assessment: 1,172 conversations / 6,549 user turns.
- Provider partitions, native annotations and source files preserved. No cross-split dialogue-ID overlap; repeated messages were reported, not hidden.
- No official `teststd` final-test file or scene/image archives downloaded/evaluated.
- Licence: CC-BY-NC-SA-4.0. Details: [SIMMC_DATASET.md](docs/SIMMC_DATASET.md).

### Volunteer data and development tools

- `runtime/evaluations/volunteer_intake_01/`: preserved source workbook, records, duplicate/split audit and pending review queue.
- 630 source rows, 30 participant codes, 90 conversations, **229 distinct texts**. Shared wording connects participants into one strict split group.
- Offline review pack: `runtime/evaluations/volunteer_review_01/`; collection form in `review_tool/`.
- Manual app-intent/entity review/training was explicitly postponed by the user. **Do not automatically resume it or label messages as reviewed.** These records did not train the SIMMC classifiers.
- 28 author-written starter annotation messages; original 13-case and separate 16-case retrieval development suites remain distinct from independent research data.

## 5. Completed trained-model evidence

Authoritative run: **`runtime/training/simmc_acts_03/`**. Runs 01 and 02 are retained setup interruptions, not completed neural experiments.

Task: seven native SIMMC user acts, **not the app's seven intents**. Both learned methods use current-user text plus a bounded preceding completed exchange; no current assistant answer, current act/slot/object labels or future text enters the features.

- TF-IDF word/character features + logistic regression; train-only fit, `C=2.0` selected on dev.
- DeBERTa-v3-small revision `a36c739020e01763fe789b4b85e2df55d6180012`, new classification head and full encoder fine-tuning.
- Three epochs, seed 42, batch 16, length 128, AdamW `2e-5`, CUDA AMP FP16. **Epoch 2** selected by validation macro-F1.
- Selection frozen and saved-checkpoint validation predictions reproduced before devtest scoring.

| Method | Held-out accuracy | Macro-F1 |
|---|---:|---:|
| Majority | 33.36% | 0.0715 |
| TF-IDF + logistic regression | 87.46% | 0.8598 |
| DeBERTa-v3-small | 97.53% | 0.9724 |

Assessment: 6,549 user turns / 1,172 conversations. Observed accuracy difference: +10.06 percentage points. Paired conversation-bootstrap 95% accuracy-difference interval: +9.26 to +10.78 points. DeBERTa accuracy interval: 97.15–97.89%.

Post-selection overlap diagnostic: 131 seen formatted inputs; on 6,418 novel formatted inputs, DeBERTa accuracy was 97.48%, macro-F1 0.9723. This does not certify source/speaker/template independence.

Saved artifacts: `configuration.json`, `progress.json`, `tfidf_logistic.joblib`, `deberta/`, `selection_frozen.json`, `assessment_predictions.jsonl`, `report.json`, `comparison_uncertainty.json`.

These are single-seed **text-only native-act** scores, not full official multimodal benchmark results, live-chat accuracy or Sri Lankan cultural validation. The public assessment is now used: do not tune against it and continue calling it an untouched final test.

## 6. Other recorded checks

- **85 automated tests passed** in the last verification on 19 October 2026 at approximately 22:34 local time; no skips reported in that run. `pip check` previously reported no broken installed requirements.
- README local links and both JSON examples validated; selected model file hash rechecked against its completed report.
- User-manual C2/C3 examples validated against the real adapter; all its local documentation links resolved. Its 36 manual cases are not 36 claimed executed passes.
- Expanded 16-case author-written retrieval comparison: Recall@3 1.0 for all three methods on fourteen answerable cases. Insufficient-evidence no-hit fractions BM25 0.5, dense/hybrid 0.0. These are development checks, not independent accuracy.
- Original-suite BM25 regression on expanded corpus: mean Recall@3 0.9394 over eleven answerable cases; ranking change retained rather than hidden.
- Real local streaming/topic/hybrid transport smokes saved in `runtime/evaluations/`; some citation omissions were correctly flagged. Successful streaming is not measured factuality.

## 7. Artifact fingerprints at this stop point

Paths below are relative to Component 4. Re-check if these files change; do not apply old results to a different checkpoint.

| File | SHA-256 |
|---|---|
| `data/knowledge.json` | `0fb507b7634ec2e1e4c61345f8797d91d0a6d607eefd8b58674e82bd50b26b76` |
| `runtime/datasets/simmc21_fashion_01/audit.json` | `459e2c8476c690d2868665f26e88bfc0bc6fd36b25e7cb7e520d8da4cce15de4` |
| `runtime/training/simmc_acts_03/report.json` | `92629eba41cecf730bb735cdc40a8cc06c57da1cf742984fed27d927e8a62c91` |
| `runtime/training/simmc_acts_03/comparison_uncertainty.json` | `f50e9556f30ff125c1c79f3e3e1fc99d0f2368dcfa935f232f536d91948e4fb7` |
| `scripts/train_simmc_acts.py` | `c97b17b4414dde642b68e0150aff557850bb85fe981bcd0cbcca663965f9e1ba` |
| `runtime/training/simmc_acts_03/deberta/model.safetensors` | `c56cc44c9abd812e43e83b98babd95cbf756bf26bf0c7fbb140e74886867ad73` |
| `runtime/training/simmc_acts_03/tfidf_logistic.joblib` | `00c01fb3e15b57249811f1e0c333f74fc0a956c19e9bce75dc3f9ec74f9ac421` |
| `runtime/training/simmc_acts_03/assessment_predictions.jsonl` | `9bf67e7f7ce9df698ef5a37a93b343a5a0285a165f0830dd00e19d396db4bf38` |

## 8. How to resume safely

Read this checkpoint before implementing anything. Do not retrain, re-download, relabel, rearrange frozen data, expose services publicly or overwrite existing runs just to resume.

```powershell
Set-Location -LiteralPath 'C:\Users\Ravindu\Videos\Component 1 – Virtual Try-On and Draping\component-4-conversational-ai'
```

First check whether the existing app still serves:

```powershell
Invoke-RestMethod http://localhost:8003/api/health
```

If not running, start in a terminal and keep it open:

```powershell
.\.venv\Scripts\python.exe backend\app.py
```

Inspect `ollama list`, open http://localhost:8003/, Refresh models, and select the intended engine/retrieval. Use New chat for unrelated tests. Basic verification:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -q
```

**Process observation, not a permanent identity:** at checkpoint creation the app was listening on `127.0.0.1:8003`, PID 24984. It was left running. Do not reuse that PID blindly later or terminate it based only on this historical record; inspect the current owner.

## 9. Next major work on a later authorized resume

Recommended first implementation decision: **define the trained native-act model's role in the actual chatbot**. Establish a reviewed task/mapping or separate auxiliary native-act output rather than renaming SIMMC acts into incompatible app intents. Define local inference loading, confidence/unknown handling, comparison/fallback visibility and regression checks before activation. Do not claim the existing benchmark establishes performance on the app's broader cultural questions.

Subsequent research tasks, keeping development and evaluation separate:

1. Evaluate independent global/Sri Lankan/mixed conversations, with reviewed relevance, factual consistency, explanation quality and cultural scope ratings. Add knowledgeable Sri Lankan review with actual recorded consent/attribution; never fabricate approval.
2. Improve missing-evidence detection and claim/citation support beyond source-ID presence. Current semantic retrieval can return unrelated evidence.
3. Extend/review factual knowledge through suitable primary sources and precise community/venue/date scope, not automatic cultural generalization from benchmark dialogue text.
4. Any reranking or model changes need controlled comparisons and genuinely reserved evaluation, not repeated tuning on already scored devtest.
5. Any live C2/C3 connection requires separately scoped integration; current pasted-JSON explanation is read-only and must stay truthfully labelled.
6. Resume volunteer app-intent/entity annotation only on explicit user direction; preserve its current pending labels and split audit.

These are resume notes, not implemented features or authority to start work while paused. The saved README intentionally describes completed work only.

## 10. Preserve before moving or cleaning

The `runtime/` folder is excluded from Git. Keep datasets, licence/provenance manifests, trained models, reports, volunteer originals and chat history. A source-code copy alone does not preserve them. This checkpoint is a document, not an external backup.

For a consistent ordinary database file backup, stop the app through its own terminal, then copy—not move—the required Component 4 files/folders to a secure destination, preserving SQLite companion files if present. Do not remove `.venv`, model caches or frozen dataset paths without an explicit migration/backup plan. No backup destination or cleanup was requested/performed at this stop point.

### Suggested later resume message

> Resume Component 4 from CHECKPOINT.md. Read the saved scope and artifacts first. Continue by defining how the trained SIMMC native-act model can be integrated without replacing the app's different intent labels blindly. Preserve existing results, source data and Components 1–3.
