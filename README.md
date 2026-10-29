<<<<<<< HEAD
# Component 4 — Explainable Conversational Fashion AI

**New laptop setup:** [Installation.md](Installation.md).

**Completed implementation and recorded results — 19 October 2026**

**Saved stop point:** [CHECKPOINT.md](CHECKPOINT.md) — current artifacts, integration boundaries, restart instructions and next steps for a later resume.

Component 4 is an English-language, locally operated conversational fashion advisor combining inspectable NLP, source-aware retrieval, local language-model generation and explanations of supplied garment/design recommendations. It includes a separately trained and evaluated fashion dialogue-act classifier.

This README documents **implemented functionality, prepared data, executed experiments and saved results**. It is not a proposal or a list of planned features. “Completed” refers to those specific implementations and experiments, not universal answer accuracy or cultural certification.

**Application:** [http://localhost:8003/](http://localhost:8003/)  
**User guide:** [Complete user manual with 36 manual test cases](USER_MANUAL.md)  
**Trained NLP evidence:** [Full SIMMC experiment report](docs/SIMMC_NLP_EXPERIMENT.md)

## 1. Implemented functionality

| Completed feature | Actual implementation |
|---|---|
| English intent understanding | Seven inspectable rule-based app intents |
| Preference extraction | LKR budget, occasion, setting, weather, style, colours and included/excluded fashion terms |
| Terminology normalization | Aliases such as Osari/Osariya/Kandyan saree; Unicode normalization and simple nearby negation |
| Multi-turn state | Retained constraints, explicit updates/removal, topic reuse for recognized follow-ups |
| Evidence retrieval | BM25, local semantic cosine search and hybrid reciprocal-rank fusion |
| Sourced explanations | Retrieved IDs, passage text, scope, publisher reference and review status displayed |
| Answer generation | Template/extractive baseline and genuine streamed local Ollama replies |
| Local models | Installed Qwen3 4B generator and all-minilm:22m embedding model |
| Upstream explanation | Read-only JSON adapters for supplied Component 2 recommendations and Component 3 designs |
| Option references | Number/ordinal selection, two-option “other one,” missing/out-of-range clarification |
| Chat management | New chat, reopening/continuation, rename, recoverable deletion and restoration |
| Persistence | Local SQLite completed-turn storage and bounded resume snapshots |
| Published dataset intake | Official pinned SIMMC 2.1 download, hashes, fashion filter and original split preservation |
| Supervised NLP experiment | Trained TF-IDF/logistic regression and fine-tuned DeBERTa-v3-small; held-out comparison |
| Research reporting | Per-class metrics, confusion matrices, frozen selection record, predictions, hashes and bootstrap intervals |
| Volunteer-data tooling | Workbook intake/duplicate audit, annotation validation, offline review and collection pages |
| Verification | 85 passing automated software tests on the configured machine |
| User documentation | Complete manual, sample messages, fictional JSON fixtures and 36 manual test specifications |

**Live application versus trained experiment:** the live app still uses `backend/nlp.py` for its seven application intents. The trained DeBERTa model predicts seven **different native SIMMC dialogue acts** and is a saved standalone experiment. Its 97.53% accuracy is not the accuracy of the live chatbot or its Sri Lankan cultural answers. No model was silently substituted into the interface.

## 2. System architecture and actual processing

### 2.1 Live conversational pipeline

```text
User message + optional supplied Component 2/3 JSON
    → input validation and conversation/revision checks
    → rule-based intent, terminology and preference extraction
    → update retained constraints; resolve recognized topic/option references
    → selected evidence retrieval: BM25 / semantic / hybrid
    → scoped source passages + bounded history + supplied-result summary
    → template baseline OR local Qwen3 4B generation
    → streamed final-answer content + citation-ID diagnostics
    → completed-turn/state transaction in local SQLite history
    → conversation, NLP state, evidence and methods displayed
```

Actual request stages:

1. The browser sends a message, expected conversation revision, selected generation/retrieval modes and optional component result.
2. The backend rejects invalid messages, stale revisions, malformed context and concurrent requests for the same conversation.
3. English rules identify the app intent and supported values. Explicit new values replace corresponding old preferences; recognized exclusions and removal requests update state.
4. Conservative reference rules identify requested supplied options and recognized follow-up topics. Missing or ambiguous references can return an immediate deterministic clarification **without retrieval or an LLM call**.
5. Normal retrieval queries combine the current/retained user topic with supported state values. Budget and excluded-term state fields are omitted from query augmentation. Previous generated claims are not inserted as new evidence.
6. The selected retriever ranks active source passages; the default evidence count is three. Source scope and review status accompany the selected text.
7. The baseline formats source/context information directly. Ollama receives selected evidence, current constraints, a bounded recent conversation and a filtered component summary.
8. The browser receives NDJSON `start`, `delta`, `done` or `error` events. Only final-answer content is shown; Qwen3's separate thinking channel is discarded.
9. Completed turns are saved transactionally. Interrupted/failed partial responses do not become completed saved turns. The interface shows actual engine/retrieval metadata and applicable citation-ID warnings.

### 2.2 Separate supervised NLP pipeline

```text
Official SIMMC 2.1 files at pinned upstream commit
    → integrity/provenance manifest
    → fashion-only filter, keeping original provider partitions
    → current user + previous completed exchange, native act targets
    → train-only majority / TF-IDF / DeBERTa fitting
    → validation-only candidate and checkpoint selection
    → frozen selection and saved-checkpoint reload verification
    → held-out devtest assessment
    → per-class reports, confusion matrices, predictions and uncertainty
```

This pipeline produces research artifacts; it does not execute inside the live chat request path.

## 3. Techniques and models

### 3.1 Application intent/entity and state processing

Source: `backend/nlp.py`, `backend/conversation.py`.

Application intent labels:

```text
recommend
explain
compare
update_preferences
knowledge_question
greeting
clarify
```

Implemented techniques:

- Unicode NFKC normalization, case folding and regular-expression matching.
- Fashion alias matching with normalized-text mention offsets.
- Basic nearby negation for included/excluded terms.
- Supported scalar preference extraction, LKR budget parsing and explicit removal.
- Conversation-state merging, retained user topics and explicit option-reference rules.
- No fabricated classifier-confidence value for this rule parser.

These inspectable rules have limited vocabulary and negation/reference coverage. They are not a trained NER model or unrestricted language understanding.

### 3.2 Retrieval

Source: `backend/retrieval.py`, `backend/semantic.py`.

| Method | Implemented technique | Role |
|---|---|---|
| BM25 | Term-frequency/document-frequency scoring; `k1=1.5`, `b=0.75`; stable ID tie-breaking | Default lexical baseline |
| Semantic | Local all-minilm:22m embeddings, L2 normalization, cosine similarity | Paraphrase-oriented ranking |
| Hybrid | Equal-weight reciprocal-rank fusion, constant 60 | Combines lexical and semantic ranks |

```text
RRF(document) = Σ 1 / (60 + rank_in_each_available_list)
```

BM25 indexes title, text and tags. Semantic embeddings use title, text and scope. Draft entries are excluded. The semantic corpus cache is keyed by raw knowledge SHA-256 and installed embedding-model digest; user-query vectors are not cached to disk. Changing the corpus/model invalidates that cache.

Retrieval ranks are not calibrated confidence. Semantic/hybrid methods can return passages for unsupported questions. A selected neural retrieval failure is visible; it is not silently replaced after the request by another method.

### 3.3 Local generation

Source: `backend/llm.py`.

| Asset | Actual role | Project training status |
|---|---|---|
| **Qwen3 4B**, Ollama tag `qwen3:4b` | Natural answer generation | Pretrained model used locally; not fine-tuned here |
| **all-minilm:22m** | Sentence embeddings | Pretrained model used locally; not fine-tuned here |
| Template/extractive baseline | Fast source/context formatting | No neural generation |

Recorded local runner: **Ollama 0.40.5**.

Recorded model identities:

```text
qwen3:4b
359d7dd4bcdab3d86b87d73ac27966f4dbb9f5efdfcc75d34a8764a09474fae7

all-minilm:22m
1b226e2802dbb772b5fc32a58f103ca1804ef7501331012de126ab22f67475ef
```

Qwen3 settings used by the adapter: temperature 0.2, `num_predict=1536`, `num_ctx=8192`, five-minute keep-alive, and native thinking separation for Qwen3 names. Only final content is streamed/stored. Recent successful messages and prompt/context text are bounded; this is not unlimited memory.

Requests target the fixed local Ollama endpoint, bypass environment proxies and refuse redirects. Cloud-tagged model names are filtered, and an embedding-only model is excluded from the chat-model list. The app does not automatically download a model or silently replace failed generation with baseline prose.

### 3.4 Trained NLP methods

| Method | Implemented training |
|---|---|
| Majority baseline | Most frequent training act: REQUEST:GET |
| TF-IDF + logistic regression | Word/character features fitted on train; regularization selected on dev |
| **DeBERTa-v3-small** | New seven-act head and full encoder fine-tuning on fashion dialogue acts |

Pretrained encoder source: [Microsoft DeBERTa-v3-small](https://huggingface.co/microsoft/deberta-v3-small), revision `a36c739020e01763fe789b4b85e2df55d6180012`, MIT. The provider's approximately 44M backbone parameters exclude approximately 98M embedding parameters; the whole model must not be described as only 44M parameters.

## 4. Datasets and knowledge sources

### 4.1 Curated factual corpus

File: `data/knowledge.json`, version **curated-0.3**.

| Region/type | Active passages |
|---|---:|
| Sri Lankan | 7 |
| Global/contextual | 14 |
| Local project implementation | 2 |
| **Total** | **23** |

Eleven distinct external source URLs are recorded across the external passages. Publisher sources include:

- Sri Lanka Export Development Board: batik process/material examples, handloom materials, product forms and craft context.
- Sri Lanka Tourism: Temple of the Tooth visitor clothing and local craft/shopping context.
- Emily Post: wedding-context guidance and named Western dress conventions.
- UNESCO: Indonesian batik technique, explicitly scoped to that context.
- Victoria and Albert Museum: Indian textile context, plain weave, satin construction and velvet pile.
- GINETEX: article-specific garment-care labelling context.

Two project passages describe the implemented Component 2/3 ranking/explanation behaviour.

Each entry records an ID, title, region, review state, scope, original short summary, tags and source reference. The corpus has a source-and-scope review record; new passages also record relevant publisher sections and checking method. Source-checked external summaries are explicitly not cultural expert approvals. Indian/Indonesian context is not relabelled as Sri Lankan etiquette, and product listings are not treated as universal wedding rules.

Full source register and review record: [KNOWLEDGE_REVIEW.md](docs/KNOWLEDGE_REVIEW.md). The corpus is retrieval evidence, **not the model's supervised training dataset**.

### 4.2 Published SIMMC 2.1 data

Source: [official facebookresearch/simmc2 repository](https://github.com/facebookresearch/simmc2), commit:

```text
6e3de4e3d11e2914edc3ab9123e6b0eb88933237
```

Downloaded dialogue train/dev/devtest JSON, fashion metadata, repository/data READMEs and licence. Total downloaded source bytes: **79,439,985**, approximately 79.44 decimal MB. The fashion metadata has **288 entries**; its Git-LFS content was checked against upstream size/SHA-256.

Raw: `runtime/datasets/raw/simmc21_official_01/`  
Prepared: `runtime/datasets/simmc21_fashion_01/`

| Provider split | Raw dialogues | Prepared fashion dialogues | Fashion user turns / paired exchanges | Fashion utterances, both speakers |
|---|---:|---:|---:|---:|
| train | 7,307 | 4,667 | 27,567 | 55,134 |
| dev | 563 | 563 | 3,494 | 6,988 |
| devtest | 1,687 | 1,172 | 6,549 | 13,098 |
| **Total** | **9,557** | **6,402** | **37,610** | **75,220** |

Completed preparation:

1. Pin official source files; record URLs, sizes and SHA-256 values.
2. Resolve and verify the fashion metadata LFS object.
3. Preserve raw files and upstream headers unchanged.
4. Filter only dialogues whose per-dialogue domain is `fashion`.
5. Preserve dialogue/turn IDs, text, native acts, slot values, requested slots, ambiguity candidates and scene references.
6. Keep provider train/dev/devtest boundaries intact; never randomly split individual turns.
7. Record per-class/slot counts and cross-split repeated-text diagnostics.

No duplicate dialogue IDs were found across downloaded partitions. Shared normalized fashion user texts numbered 257 (train/dev), 423 (train/devtest) and 126 (dev/devtest); these are not automatically duplicate conversations or proof of speaker independence. SIMMC uses simulator-generated flows and manual paraphrasing, and some catalogue attributes are synthetic.

The official `teststd`, scene archives and images were not downloaded/evaluated. This is a **text-only fashion subset**. The fetched filenames are 2.1 release files although upstream top-level version headers retain `simmc_2_dstc10`; provenance records preserve that distinction.

Dataset licence: **CC-BY-NC-SA-4.0**. Full audit and citation: [SIMMC_DATASET.md](docs/SIMMC_DATASET.md).

### 4.3 Volunteer collection intake and completed review tooling

User-supplied source: `Fashion_Chatbot_30_Volunteers_630_Messages.xlsx`; the user confirmed independently written messages and research permission.

Completed intake found:

- **630 rows**, **30 participant codes**, **90 conversations**.
- **229 distinct message texts**, with 401 repeated occurrences beyond those distinct texts.
- Shared text links all participants into one connected grouping for the strict split audit.

The original workbook was preserved as a hashed source snapshot with source records, duplicate manifest, near-duplicate candidates, audit report and a pending annotation-review queue. These records were **not used to fit the SIMMC models**.

Implemented review tooling includes an offline page for 229 distinct texts, original-context inspection, hidden rule suggestions, draft/review validation, progress JSON export/resume, and a separate consent-aware collection form. It does not fabricate completed human reviews or independently approved labels. This is a completed data/tooling implementation, not a completed volunteer-trained classifier.

Artifacts: `runtime/evaluations/volunteer_intake_01/`, `runtime/evaluations/volunteer_review_01/`. Details: [VOLUNTEER_INTAKE.md](docs/VOLUNTEER_INTAKE.md), [ANNOTATION_REVIEW.md](docs/ANNOTATION_REVIEW.md).

### 4.4 Author-written development/annotation fixtures

- **28 starter messages**: implemented annotation-validation/span-export/rule-baseline workflow; artificial author-written examples, not participant generalization evidence.
- **13 original retrieval cases**: global, Sri Lankan, mixed, project and insufficient-evidence groups.
- **16 expansion retrieval cases**: fourteen answerable cases and two insufficient-evidence cases.
- **36 manual test specifications** in [USER_MANUAL.md](USER_MANUAL.md), including sample inputs and acceptance checks; a specification is not an automatically executed pass.

The original 13-case retrieval suite remains unchanged, separate from the expansion suite.

## 5. Completed NLP training process

Script: `scripts/train_simmc_acts.py`. Completed run: **`runtime/training/simmc_acts_03/`**.

### 5.1 Task and labels

Predict the current user's native fashion-shopping dialogue act:

```text
ASK:GET
INFORM:DISAMBIGUATE
INFORM:GET
INFORM:REFINE
REQUEST:ADD_TO_CART
REQUEST:COMPARE
REQUEST:GET
```

These seven classes are not a renamed version of the app intents. In particular, an ADD_TO_CART prediction does not execute an action. Native slot-value labels were preserved but not misreported as token-level NER span evaluation.

### 5.2 Identical input construction

Both learned methods receive:

```text
Current user: first 64 words of the current message
Previous user: last 24 words of the preceding user turn, if present
Previous assistant: last 32 words of the preceding assistant turn, if present
```

Current assistant response, current act/slot/object annotations and future turns are excluded from features. The preceding reference assistant reply is past conversation context; this is not autonomous generated-dialogue evaluation. DeBERTa additionally truncates to 128 tokens; TF-IDF consumes the full budgeted string.

### 5.3 Fitting and validation selection

| Setting | TF-IDF + logistic regression | DeBERTa-v3-small |
|---|---|---|
| Fit partition | train only | train only |
| Representation | Word 1–2 grams and character-within-word 3–5 grams | Pretrained transformer encoder + new classification head |
| Feature budget | Up to 50,000 features per branch | Maximum 128 tokens |
| Main configuration | `C={0.5,2.0}`, maximum 500 iterations | Three epochs, batch 16, learning rate `2e-5` |
| Optimizer/regularization | Logistic-regression regularization | AdamW, weight decay 0.01, 10% linear warm-up, gradient clipping 1.0 |
| Training precision | CPU numerical library implementation | CUDA AMP FP16 |
| Seed | 42 | 42 |
| Selection criterion | Validation macro-F1 | Validation macro-F1 |

Natural class proportions were retained; no evaluation rebalance or turn-wise repartitioning. Best linear setting: **C=2.0**. Best neural checkpoint: **epoch 2**.

### 5.4 Frozen assessment and uncertainty

After all validation-based choices, the script saved the selected candidates/checkpoint identity, reloaded the neural checkpoint, verified matching validation predictions, and only then opened devtest for scoring. The prior intake had inspected schema/counts, so this is not a wholly unopened official final test.

The report records accuracy, macro/weighted F1, per-class precision/recall/F1, confusion matrices, per-turn predictions and model hashes. A separate post-selection script computes 1,000 paired bootstrap replicates resampling entire assessment conversations and reports repeated-input slices without changing weights or thresholds.

Setup interruptions in runs 01/02 are retained but are not reported as completed neural experiments. Run 03 reused the already completed baseline from run 01 after task/data/seed checks; setup repairs occurred before assessment. Model source bytes were verified against the pinned publisher LFS hash.

## 6. Recorded results

### 6.1 Held-out native dialogue-act comparison

Assessment: **6,549 user turns in 1,172 conversations**.

| Method | Assessment accuracy | Macro-F1 | Weighted F1 |
|---|---:|---:|---:|
| Majority class | 33.36% | 0.0715 | 0.1669 |
| TF-IDF + logistic regression | **87.46%** | **0.8598** | 0.8739 |
| Fine-tuned DeBERTa-v3-small | **97.53%** | **0.9724** | 0.9753 |

Observed DeBERTa accuracy difference versus TF-IDF: **+10.06 percentage points**. This is a one-seed text-only task comparison, not a full multimodal result, a state-of-the-art claim, or overall chatbot/cultural accuracy.

Validation accuracy/macro-F1: TF-IDF **87.84% / 0.8621**; selected DeBERTa **96.62% / 0.9612**. DeBERTa epoch macro-F1 values were 0.951787, 0.961187 and 0.960497; the final epoch was not automatically selected.

### 6.2 Class-level assessment

| Native act | Support | DeBERTa F1 |
|---|---:|---:|
| ASK:GET | 736 | 0.9565 |
| INFORM:DISAMBIGUATE | 679 | 0.9978 |
| INFORM:GET | 668 | 0.9910 |
| INFORM:REFINE | 692 | 0.9357 |
| REQUEST:ADD_TO_CART | 956 | 0.9995 |
| REQUEST:COMPARE | 633 | 0.9463 |
| REQUEST:GET | 2,185 | 0.9802 |

The selected neural model made 162 errors. Largest off-diagonal counts were REQUEST:GET → INFORM:REFINE (49), INFORM:REFINE → REQUEST:GET (34), REQUEST:COMPARE → ASK:GET (31), ASK:GET → REQUEST:COMPARE (29). No post-assessment retuning is included in this result.

### 6.3 Bootstrap and overlap diagnostics

Paired whole-conversation percentile bootstrap, 1,000 replicates, seed 42:

| Quantity | 95% interval |
|---|---|
| TF-IDF accuracy | 86.74–88.25% |
| DeBERTa accuracy | 97.15–97.89% |
| DeBERTa macro-F1 | 0.9681–0.9765 |
| DeBERTa minus TF-IDF accuracy | +9.26 to +10.78 percentage points |

There were 131 assessment turns whose normalized formatted inputs also appeared in training, all REQUEST:GET. On the **6,418 novel formatted inputs**, TF-IDF accuracy/macro-F1 was **87.21% / 0.8590**, and DeBERTa **97.48% / 0.9723**.

“Novel” here means no exact normalized formatted-input match, not proof of semantic/template or speaker independence. Bootstrap intervals describe sampling of these fixed checkpoints/public conversations, not training-seed variation or cultural validity.

### 6.4 Retrieval development comparison on the expanded corpus

Suite: `data/evaluation/knowledge_expansion.json`; fourteen answerable and two insufficient-evidence author-written cases; K=3.

| Method | Mean Recall@3, answerable cases | MRR@3 | NDCG@3 | No-hit fraction, two insufficient-evidence cases |
|---|---:|---:|---:|---:|
| BM25 | 1.0000 | 1.0000 | 0.9943 | 0.5000 |
| Dense | 1.0000 | 0.9643 | 0.9781 | 0.0000 |
| Hybrid | 1.0000 | 0.9643 | 0.9781 | 0.0000 |

Questions were written alongside the corpus; these figures are **development coverage checks**, not independent retrieval/answer accuracy or proof of hybrid superiority. Dense/hybrid returned passages for both insufficient-evidence questions, an observed limitation retained in the report.

The unchanged original suite, rerun on the expanded corpus using BM25, had mean Recall@3 **0.9394**, MRR@3 **1.0000**, NDCG@3 **0.9445** over eleven answerable cases, with no-hit fraction **0.5** over two insufficient-evidence cases. New passages displaced old labelled passages on the wedding-comparison case; original labels were preserved rather than edited to hide the ranking change.

Reports: `runtime/evaluations/knowledge_expansion_01/`, `runtime/evaluations/knowledge_regression_01/`. Human-rating fields in exported packets remain blank; they are not measured quality scores.

### 6.5 Local generation and software verification

Real Qwen3 streaming smokes completed with native final-content/thinking separation. An accepted two-turn transport smoke produced 41/83 visible chunks, with first visible text approximately 15.13/12.86 seconds on this machine. These are individual transport observations, not representative latency or answer-quality benchmarks.

Later real follow-up/hybrid smokes also recorded actual streamed generation. Some omitted inline source IDs and were correctly flagged. Transport completion and topic continuity do not imply factual/citation-quality approval. The initial thinking-channel diagnostic is preserved but is not an accepted quality result. Exact records: [LOCAL_LLM.md](docs/LOCAL_LLM.md), [CORE_CHAT.md](docs/CORE_CHAT.md), [SEMANTIC_RETRIEVAL.md](docs/SEMANTIC_RETRIEVAL.md).

**85 automated tests passed** at this checkpoint. Python dependency checks reported no broken installed requirements. User-manual Component 2/3 fixtures validated against the real adapter, and all local manual links resolved. The manual's 36 cases are specifications, not 36 fabricated executed passes.

## 7. Component 2 and 3 explanation adapters

Implementation: `backend/context.py` and reference planning in `backend/conversation.py`.

Users select a source component, paste actual response JSON, then ask why/compare. The adapter reads:

| Source | Root list | Item score | Additional context |
|---|---|---|---|
| Component 2 | `recommendations` | `score` | Recorded reasons/conflicts |
| Component 3 | `designs` | `rule_score` | Reasons/conflicts and `property_basis` |

The adapter validates types and accepts 1–20 records; retains the first five item summaries, up to five reasons/conflicts per item; strips unrelated profiles/images; and marks provenance as supplied, not live-verified. Ordinary model context is further bounded; explicit requested retained options preserve their original numbering.

Implemented references include “second option,” “option 4,” a comparison of numbered options, and “the other one” when exactly two options and one prior selection make it unambiguous. Missing/out-of-range or ambiguous references produce focused deterministic clarification. Clearing the JSON box clears supplied context on the next message.

This is **read-only explanation of supplied outputs**. It does not call upstream recommendation engines, update their preferences or certify user-supplied JSON. Rule points remain points; a Component 3 typical profile remains an illustrative assumption, not a measured physical property or fit guarantee. Copy-and-paste, explicitly fictional fixtures are provided in [USER_MANUAL.md](USER_MANUAL.md).

## 8. Conversation interface and persistence

Implemented frontend: plain HTML/CSS/JavaScript, responsive history rail plus conversation, settings, NLP-understanding and evidence sections. Includes live incremental answers, Stop response, latest-output scrolling, visible engine/retrieval selection and error/status feedback.

Normal startup enables **`runtime/chat_history.sqlite3`** using SQLite:

- Create, reopen/continue, rename, recoverably delete and restore chats.
- Automatic first-message title or validated 1–80-character manual title.
- Save completed user/assistant turns, evidence, methods, model name and bounded resume state.
- Restore completed transcripts/state after browser/server restart or in-memory cache expiry.
- Atomic completed-turn storage and revision protection against stale overwrites.
- Limits: 200 active chats and 100 turns per conversation.

Deleted chats remain in Recently deleted and can be restored. **Delete is not permanent erasure.** Unsent drafts and incomplete responses are not committed as completed history. Ordinary chats are not automatically reused as training/research data. The browser does not store conversation transcripts in localStorage.

## 9. Installation and execution

### 9.1 Start the installed application

```powershell
Set-Location -LiteralPath 'C:\Users\Ravindu\Videos\Component 1 – Virtual Try-On and Draping\component-4-conversational-ai'
.\.venv\Scripts\python.exe backend\app.py
```

Open **http://localhost:8003/**. Keep the terminal running; Ctrl+C stops this server. Reuse an existing correct server rather than starting another on port 8003. Activation is optional with the explicit interpreter.

Fresh Python environment only if missing:

```powershell
py -3.12 -m venv .venv
```

The application server/baseline uses the standard library. Optional training packages are isolated in this component's environment; they are not imported by the live rule/RAG server.

### 9.2 Installed local models

```powershell
ollama list
```

Start installed Ollama, click Refresh models, then choose **Local language model — Ollama**, **qwen3:4b**, and a retrieval method. The interface chooses Ollama on initialization if installed models are available; otherwise baseline remains usable. BM25 is the default retrieval selection.

For a fresh setup with a missing model only:

```powershell
ollama pull qwen3:4b
ollama pull all-minilm:22m
```

Ollama installation/downloads are separate setup steps; they require network/disk space. Normal configured inference calls use the fixed local API `127.0.0.1:11434`. No paid API key is required. The deterministic baseline with BM25 does not need Ollama or internet.

### 9.3 Example use

Start fresh chats for unrelated scenarios:

```text
What clothing guidance is available for visiting the Temple of the Tooth?
```

Same-chat follow-up:

```text
Can you explain that more simply?
```

Preference extraction, then update in the same separate chat:

```text
Suggest something traditional for an outdoor wedding under LKR 15,000
Actually office instead, not batik, handloom for hot weather
Remove the budget
```

Inspect retained/excluded terms and budget changes rather than assuming unrestricted parsing. Other sourced prompts include `Is satin always silk?` and `Does black tie optional allow a dark suit?`. Complete procedures and acceptance checks: [USER_MANUAL.md](USER_MANUAL.md).

## 10. API

| Method | Endpoint | Implemented purpose |
|---|---|---|
| GET | `/api/health` | Service/default flags, corpus hash/count and capability flags |
| GET | `/api/knowledge` | Source passages and provenance/scope |
| GET | `/api/models` | Local generator inventory/readiness |
| GET | `/api/retrieval` | Local embedding readiness/methods |
| GET | `/api/chats` | Active/deleted saved-chat summaries |
| GET | `/api/chats/<id>` | Reopen completed transcript/state |
| POST | `/api/session` | Create a conversation; normal server persists its initial state |
| POST | `/api/chat` | Completed JSON answer |
| POST | `/api/chat/stream` | NDJSON incremental answer/events |
| POST | `/api/chats/rename` | Rename using `session_id`, `title` |
| POST | `/api/chats/delete` | Recoverable deletion using `session_id` |
| POST | `/api/chats/restore` | Restore using `session_id` |

Example chat body (replace the ID with the session response):

```json
{
  "session_id": "ID_FROM_SESSION",
  "expected_revision": 0,
  "message": "Is satin always silk?",
  "generation": "ollama",
  "model": "qwen3:4b",
  "retrieval": "bm25",
  "component_context": null
}
```

Read-only supplied context format:

```json
{
  "component": 3,
  "result": {
    "property_basis": "typical_profile",
    "designs": [
      {
        "name": "FICTIONAL SOFTWARE FIXTURE — shirt",
        "rule_score": 7,
        "reasons": ["Fictional fixture: recorded rule matched"],
        "conflicts": []
      }
    ]
  }
}
```

The context object belongs under `component_context`; the fixture is not an actual recommendation. All POST bodies are bounded to 50,000 bytes and chat text to 2,000 characters. Origins, when supplied, must match localhost/127.0.0.1 at the service port. Non-finite JSON is rejected. This is loopback protection, not authenticated multi-user hosting.

Health defaults such as `llm_enabled:false` describe compatibility defaults, not the engine used for every request. Read actual readiness and per-answer `methods` to distinguish Ollama, template and clarification paths.

## 11. Verification and reproducibility

### 11.1 Software tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -q
.\.venv\Scripts\python.exe -m pip check
```

With Node available:

```powershell
node --check frontend\app.js
```

Coverage includes state/negation, source/retrieval behaviour, semantic cache and malformed vectors, stream consumption/failure rollback, supplied-option references, history transactions/delete/restore, annotation/export validation, dataset filtering/hashes, training input leakage, selection and metric arithmetic. Some optional checks skip on environments lacking Node or training libraries; disclose skips. These software checks do not replace independent human answer evaluation.

### 11.2 Re-run implemented data/evaluation workflows

Use fresh output names; existing recorded runs are protected against overwrites.

```powershell
# Re-prepare the downloaded SIMMC sources without a network download
.\.venv\Scripts\python.exe scripts\prepare_simmc.py --output runtime\datasets\simmc21_fashion_REPRO

# Compare retrieval on the separate expansion development suite
.\.venv\Scripts\python.exe scripts\evaluate_development.py --suite data\evaluation\knowledge_expansion.json --method all --output runtime\evaluations\retrieval_REPRO

# Preview the implemented annotation review/collection pages separately
.\.venv\Scripts\python.exe scripts\serve_nlu_review.py
```

The review preview uses localhost:8004; it does not save reviewed labels automatically or alter the chatbot. Semantic evaluation needs installed local all-minilm. No human rating is auto-filled.

### 11.3 Reproduce the completed trained benchmark

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-training.txt
.\.venv\Scripts\python.exe scripts\train_simmc_acts.py --torch-site '..\component-3-fabric-design\.venv\Lib\site-packages' --output runtime\training\simmc_acts_REPRO
.\.venv\Scripts\python.exe scripts\compare_simmc_results.py --run runtime\training\simmc_acts_REPRO --dataset runtime\datasets\simmc21_fashion_01
```

This machine reused Component 3's CUDA/PyTorch packages **read-only**, avoiding a second large runtime installation. New packages/models/artifacts are in Component 4. For an independent portable environment install compatible CUDA PyTorch in Component 4 and omit `--torch-site`. The script requires CUDA for this recorded full fine-tuning configuration.

Recorded training software: Python 3.12, PyTorch **2.14.1+cu126**, Transformers **4.57.1**, scikit-learn **1.7.2**, NumPy **2.5.4**. Hardware: **NVIDIA GeForce RTX 4050 Laptop GPU**. Optional dependency list: `requirements-training.txt`.

Recorded run-03 elapsed time: **1,204.375 seconds**, including setup/download but excluding the reused baseline's original fit. Peak PyTorch allocated CUDA memory: **3,016,863,232 bytes**, not total VRAM. These figures are not a matched-compute model-speed comparison.

The Windows SentencePiece Unicode-path issue was handled by temporarily copying the small vocabulary to a managed ASCII path; source folders were not renamed. Explicit download/hash validation replaced a stalled library weight transfer. These were setup repairs, not changes selected after assessment scores.

## 12. Project structure and saved artifacts

```text
component-4-conversational-ai/
├── README.md                       # Completed implementation/results
├── USER_MANUAL.md                  # Operating instructions and 36 manual cases
├── requirements.txt                # Baseline server requirements note
├── requirements-training.txt       # Optional supervised experiment dependencies
├── backend/
│   ├── app.py                      # Loopback HTTP service and routes
│   ├── nlp.py                      # Live rule-based app NLP
│   ├── conversation.py             # Topic/option-reference planning
│   ├── retrieval.py                # Corpus validation, BM25 and RRF
│   ├── semantic.py                 # Ollama embeddings/cosine/cache
│   ├── llm.py                      # Local generator transport/grounding
│   ├── context.py                  # Read-only supplied C2/C3 summaries
│   ├── dialogue.py                 # State, revisions, generation/commit logic
│   └── history.py                  # SQLite persistence
├── frontend/                       # Responsive HTML/CSS/JavaScript UI
├── data/
│   ├── knowledge.json              # 23 scoped active passages
│   └── evaluation/                 # Original and expansion development suites
├── scripts/                        # Setup, data, training and evaluation tools
├── review_tool/                    # Offline annotation/collection UI assets
├── tests/                          # Software tests and Node handler harnesses
├── docs/                           # Detailed implementation/experiment records
└── runtime/                        # Local generated data; excluded from Git
    ├── chat_history.sqlite3
    ├── setup/                      # Runner/model identity records
    ├── models/                     # Pinned encoder source and HF cache
    ├── datasets/
    │   ├── raw/simmc21_official_01/
    │   └── simmc21_fashion_01/
    ├── evaluations/                # Intake, review, retrieval and streaming reports
    └── training/simmc_acts_03/
        ├── configuration.json
        ├── progress.json
        ├── tfidf_logistic.joblib
        ├── deberta/                # Selected model/tokenizer/configuration
        ├── selection_frozen.json
        ├── assessment_predictions.jsonl
        ├── report.json
        └── comparison_uncertainty.json
```

Key recorded identities:

| Artifact | SHA-256 |
|---|---|
| Official DeBERTa source weights | `d4b0ebcc7799a2c2dde1a39199cb0ed709f2c6b63a48b8c05133b77955f271a9` |
| Selected fine-tuned checkpoint | `c56cc44c9abd812e43e83b98babd95cbf756bf26bf0c7fbb140e74886867ad73` |
| TF-IDF baseline artifact | `00c01fb3e15b57249811f1e0c333f74fc0a956c19e9bce75dc3f9ec74f9ac421` |
| Assessment predictions | `9bf67e7f7ce9df698ef5a37a93b343a5a0285a165f0830dd00e19d396db4bf38` |

These identities connect the reported results to the local files; an unrelated re-trained checkpoint should not inherit these scores. Load joblib artifacts only from trusted sources.

## 13. Privacy, provenance and licences

- Normal configured generation/embedding calls are local. External source links and explicit setup downloads use the internet. Arbitrary custom model implementations are not certified as local-only.
- SQLite history is unencrypted local application data, not cloud storage or authenticated multi-user access. Do not expose the service publicly or paste unnecessary personal details.
- Completed user chats are not automatically added to datasets or model training. Research reuse has separate consent/anonymization requirements.
- Recoverable deletion leaves data in SQLite. It is not secure/permanent erasure.
- Runtime data is excluded from Git; source control alone is not a backup of datasets, histories, reports or checkpoints. Stop the app before an ordinary file-copy database backup and preserve companion files if present.
- Publisher content is stored as short scoped summaries with attribution, not copied whole articles. Source review does not imply expert endorsement or blanket reuse rights.
- SIMMC 2.1 is declared **CC-BY-NC-SA-4.0**; preserve its downloaded licence/attribution and restrictions. Microsoft DeBERTa-v3-small is **MIT**. Qwen3's provider listing records **Apache-2.0**; retain model/provider terms for each installed artifact. The embedding package includes its own licence metadata. Model, dataset and knowledge-source rights are separate.
- No Component 1–3 source, model, catalogue or dataset was changed by the Component 4 work; the CUDA dependency reuse is read-only.

Dataset citation: Satwik Kottur, Seungwhan Moon, Alborz Geramifard and Babak Damavandi, **SIMMC 2.0: A Task-oriented Dialog Dataset for Immersive Multimodal Conversations**, EMNLP 2021, pp. 4903–4912, [DOI 10.18653/v1/2021.emnlp-main.401](https://aclanthology.org/2021.emnlp-main.401/). The prepared files use the publisher's 2.1 release; retain that version attribution alongside the paper citation.

Encoder reference: Pengcheng He, Jianfeng Gao and Weizhu Chen, **DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing**, [arXiv:2111.09543](https://arxiv.org/abs/2111.09543), with the pinned Microsoft checkpoint identified above.

## 14. Documentation index

| Document | Completed work described |
|---|---|
| [USER_MANUAL.md](USER_MANUAL.md) | Full usage, examples, manual test specifications and troubleshooting |
| [SIMMC_NLP_EXPERIMENT.md](docs/SIMMC_NLP_EXPERIMENT.md) | Training protocol, exact results, error analysis and panel-ready wording |
| [SIMMC_DATASET.md](docs/SIMMC_DATASET.md) | Download, provenance, licence and fashion-subset audit |
| [KNOWLEDGE_REVIEW.md](docs/KNOWLEDGE_REVIEW.md) | Current source register and retrieval checks |
| [CHAT_HISTORY.md](docs/CHAT_HISTORY.md) | Persistence and recoverable deletion implementation |
| [CORE_CHAT.md](docs/CORE_CHAT.md) | Implemented follow-up/option-reference logic and smoke record |
| [LOCAL_LLM.md](docs/LOCAL_LLM.md) | Local setup, streaming transport and recorded smokes |
| [SEMANTIC_RETRIEVAL.md](docs/SEMANTIC_RETRIEVAL.md) | Implemented embeddings, cache, RRF and earlier comparison |
| [EVALUATION.md](docs/EVALUATION.md) | Implemented evaluation/export machinery and metric definitions |
| [VOLUNTEER_INTAKE.md](docs/VOLUNTEER_INTAKE.md) | Completed workbook intake and split/duplicate audit |
| [ANNOTATION_REVIEW.md](docs/ANNOTATION_REVIEW.md) | Implemented offline review and collection tools |
| [NLU_DATA_COLLECTION.md](docs/NLU_DATA_COLLECTION.md) | Implemented starter schema/validation workflow |

Some linked development records retain historical milestone statements. This README describes the current completed state: **live local rule/RAG/LLM chat plus a separately completed supervised native-act experiment**, with scope-specific evidence rather than a single claimed “whole-system accuracy.”
=======
# RP_Planner_Test_Repo
>>>>>>> 8273d8976fe5f864e18803b60d901180dc7642ea
