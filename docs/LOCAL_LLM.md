# Local grounded LLM and genuine streaming

Implemented 19 October 2026. User selected local Ollama rather than a paid API. This adds optional neural generation; English NLU remains rule-based, and retrieval remains BM25. It does not implement dense/hybrid retrieval or a reranker.

## Setup and runtime

Ollama serves its local API at `127.0.0.1:11434`. Component 4 remains at `127.0.0.1:8003`. No paid key is needed. Install [official Ollama for Windows](https://ollama.com/download/windows), then download a local model, for example:

```powershell
ollama pull qwen3:4b
ollama list
```

The [Qwen3 4B listing](https://ollama.com/library/qwen3:4b) identifies a roughly 2.5 GB Q4_K_M model and Apache-2.0 terms. Installer/GPU libraries and runtime memory need additional space. Disk size is not total inference RAM/VRAM. Preserve provider terms; model quality, latency and cultural accuracy need separate evaluation. A GPU may accelerate it, but performance depends on hardware/context and cannot be promised from file size.

The model download is a setup step, not an automatic chat request. The app never calls a pull/download API. Runtime prompts go only to the fixed local endpoint, bypass environment HTTP proxies, and refuse redirects. Cloud-tagged model names are filtered. Do not configure custom Ollama models that forward to cloud services if local privacy is required; the app cannot certify arbitrary custom model implementations.

The chat application never initiates a model pull. An explicitly invoked, separate researcher setup utility is also available: `scripts/setup_local_model.py --output runtime/setup/NEW_SETUP_VERSION`. It pulls only the allowlisted Qwen3 4B tag, prints bounded progress and records runner version/model digest after success. It is not imported by the server or called by chat routes. Standard Ollama model storage is under the user's `.ollama/models`, outside this component's Python venv. Ollama is separately installed per-user software; its own logs/settings are distinct from this app's in-memory conversation storage.

Refresh Component 4 with Ctrl+F5, then **Refresh models**. Choose **Local language model — Ollama** and an installed model. On initial load, the UI selects LLM mode if installed local models are available; otherwise it clearly retains baseline mode. Choosing baseline deliberately uses templates, with one immediate answer, not fake LLM typing. Availability does not guarantee the selected model is a compatible text/chat model; incompatible models fail visibly.

## What streams

`POST /api/chat/stream` uses newline-delimited JSON:

- `start`: NLP state/evidence and actual generation-method metadata.
- `delta`: newly generated text as Ollama supplies it.
- `done`: completed answer, committed revision and citation-ID check.
- `error`: incomplete generation; never silently replaced by a baseline answer.

The UI displays the user's message immediately, a waiting indicator, then incremental output. No artificial delays or character-by-character replay. Baseline responses remain fast and are labelled baseline. **Stop response** aborts the browser request; server/provider cleanup can wait until the next write or model timeout. Partial text is not a completed answer. In the narrow case where the server completed but the browser missed the final event, the next revision check can fail; start a new conversation rather than silently duplicate/replay a turn.

`POST /api/chat` remains compatible and returns the complete JSON response. Both APIs accept `generation: "baseline" | "ollama"` and an installed `model` name for Ollama. `/api/models` reports runner availability, model names/digests/sizes without credentials. Health's `llm_enabled: false` describes the server's compatibility default, not a ban on optional per-request LLM generation; `llm_supported`, `llm_selection_per_request` and per-answer methods distinguish the modes.

## Grounding and history

The model sees a constrained system instruction, up to eight recent successful user/assistant messages, current parsed constraints, scoped retrieved passages/source IDs and a sanitized supplied Component 2/3 summary. Assistant history entries are bounded; complete histories are not persisted to disk. Inputs/context remain user data, not instructions. No tools or upstream writes are exposed.

Prompt assembly uses a character budget, keeps recent complete turn pairs, shortens verbose reasons and limits supplied context to three result items. The prompt records that limited scope; it must not claim the full upstream catalogue was inspected. Character budgeting is a resource safeguard, not an exact tokenizer-based guarantee for every language.

Prompts require evidence citations, contextual limits, no invented garment prices/rankings, no universal cultural superiority and preserved Typical-profile/relative-score/provisional-ease distinctions. These instructions reduce risk; they do not prove hallucination resistance. Unknown/missing citation IDs are checked after generation, but ID matching is **not entailment or factual verification**. Streamed words are provisional until completion and still need human/benchmark assessment.

Sessions reserve one active turn each without holding the global session lock during model I/O. One local generation runs at a time to limit resource contention. Provider failures and abandoned generators release reservations and do not commit partial turns. State/history are committed only on complete generation. The local inventory may change between requests; retain model digests and configuration when creating formal research reports. Current generation options: temperature 0.2, up to 1536 predicted tokens including the model's internal reasoning, context setting 8192; device/runtime may impose additional constraints. Token-limit truncation is treated as incomplete, not silently committed.

Qwen3's thinking build uses Ollama's native thinking parser: `think: true` for Qwen3 model names, provider default for other models. Only `message.content` is streamed and stored; `message.thinking` is discarded. The first real transport diagnostic with thinking disabled exposed reasoning inside content and cut off the final answer. That diagnostic (`local_llm_stream_smoke_01`) is not an accepted response-quality result. A direct native-channel check confirmed separated thinking and final content; subsequent accepted smokes must use the corrected parser path. Users may see a genuine waiting period before the first final-answer token; no internal reasoning is shown as a progress substitute.

## Verification and boundaries

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -q
node --check frontend\app.js
```

Transport tests use a fake local Ollama server to prove a delta is received before completion, same-session reservations, separate-session responsiveness, history/grounding construction, no partial commit, missing-model rejection, cloud-tag filtering and truthful baseline labeling. These are mechanics tests, not a real LLM quality benchmark. Record actual installed-model smoke results separately once the runner/model are available.

References: [Ollama chat API](https://docs.ollama.com/api/chat), [streaming](https://docs.ollama.com/capabilities/streaming), [local model inventory](https://docs.ollama.com/api/tags), [Windows runtime](https://docs.ollama.com/windows).

## Installed and verified locally

Ollama **0.40.5** was installed per-user from the hash-verified official release using WinGet. Qwen3 4B downloaded and passed Ollama's SHA-256 verification. Model manifest digest: `359d7dd4bcdab3d86b87d73ac27966f4dbb9f5efdfcc75d34a8764a09474fae7`; local inventory size: 2,497,293,931 bytes. Setup record: `runtime/setup/ollama_qwen3_4b_01/report.json`.

Corrected real streaming smoke: `runtime/evaluations/local_llm_stream_smoke_02/report.json`. Two actual local turns produced 41/83 visible-answer chunks, with first answer text at approximately 15.13/12.86 seconds and completed responses at 15.84/14.36 seconds on this machine. Reasoning was discarded via the native channel; no thinking markers appeared in the answers. Citation-ID checks passed. The follow-up referred to the previous answer. These are development mechanics observations, not a representative latency benchmark, independent quality evaluation or factual-entailment proof. The initial reasoning-leak diagnostic remains preserved and explicitly not accepted. Later wording refinements can have separate smoke outputs; do not overwrite reports.

The browser was checked with the computer-use workflow: Local language model — Ollama and qwen3:4b are available/selected after refresh. Source-grounded answers, broader corpus quality, cultural correctness and human evaluation remain research work. Local inference occupied approximately 3.8 GB GPU memory during this check; idle models may unload after the configured five-minute keep-alive.

Final wording-refinement transport check: `runtime/evaluations/local_llm_stream_smoke_03/report.json`, with 42/25 visible chunks and first visible text at approximately 15.19/14.66 seconds. Both were concise, final-only answers. The simplified follow-up omitted an inline citation, which the checker flagged; source cards remained visible. This is a demonstrated limitation, not a passing factuality/citation-quality benchmark. Preserve both reports and evaluate citation completeness independently. Current regression suite: **25 tests passed**.
