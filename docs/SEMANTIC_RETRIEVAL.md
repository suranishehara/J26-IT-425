# Local semantic and hybrid retrieval

Implemented 19 October 2026. This is an additional neural NLP stage, not a trained intent/NER model and not a claim of improved answer accuracy.

## Pipeline and model

English rule-based intent/slots → retained conversation constraints → retrieval query → selected retrieval method → scoped evidence → baseline or Qwen3 answer.

The embedding model is **all-minilm:22m**, served locally by Ollama, separate from the Qwen3 4B answer generator. [Ollama's model listing](https://ollama.com/library/all-minilm) describes this as an embedding-only model, approximately 46 MB with a 512-token context. We use the native [embedding endpoint](https://docs.ollama.com/api/embed), not a cloud API. Exact installed identity is in `runtime/setup/all_minilm_22m_01/report.json`. The package includes its licence; publisher/source attribution is retained. No domain fine-tuning has been performed.

| Selection | Algorithm | Important limit |
|---|---|---|
| BM25 | Existing lexical scoring | Can miss paraphrases or retrieve merely overlapping words |
| Semantic | Normalized sentence vectors, cosine similarity | Returns nearest passages even for unsupported questions |
| Hybrid | Equal-weight reciprocal-rank fusion of BM25 and semantic rankings; constant 60 | Rank aggregation is not a factual verifier or calibrated confidence |

Corpus embeddings use title, passage text and scope. BM25 retains its original title/text/tags representation for baseline compatibility. Draft entries remain excluded. Candidate pool is the complete current corpus (10 entries); evidence defaults to top three. RRF adds `1/(60+rank)` for each available list. Stable ID breaks score ties. Dense cosine, RRF and BM25 values have different scales; never interpret them as probabilities or compare them as accuracy.

The app checks embedding availability on page load. BM25 remains the UI/API default: this development comparison did not establish hybrid superiority. Select semantic/hybrid explicitly for comparison. Model absence selects BM25 visibly during readiness checks. A failure during an explicitly selected dense/hybrid request produces an error, never an undisclosed fallback. The embedding model is excluded from the chat-model dropdown.

## Setup and use

Setup was completed on this machine. Fresh installation only:

```powershell
.\.venv\Scripts\python.exe scripts\setup_local_model.py --model all-minilm:22m --output runtime\setup\all_minilm_22m_new
```

No chat route downloads a model. No new pip dependency is required. Start/restart the Component 4 server, refresh the browser and choose **Evidence retrieval**. Use Refresh models to refresh both chat and embedding availability. `GET /api/retrieval` reports current embedding readiness. Add `"retrieval":"bm25"`, `"dense"` or `"hybrid"` to chat API payloads; successful responses identify `methods.retrieval`. Evidence from dense/hybrid also records embedding model/digest; hybrid records component ranks. Server health retains `retrieval:bm25` and `semantic_retrieval_enabled:false` as compatibility/default indicators, with explicit per-request support flags; these do not mean all requests used BM25.

## Cache, failure and privacy

One bounded in-process public-corpus cache is keyed by raw knowledge-file SHA-256 plus model digest. Either changing invalidates it. User query vectors are not cached or written to disk. Each query is embedded locally. Cache builds serialize under a dedicated embedding lock; network calls do not hold the global conversation lock. Retrieval failure releases the session reservation and does not commit a partial turn. Corpus vectors disappear on restart. The chat session still retains bounded recent messages in process memory as documented separately.

The adapter rejects zero/nonfinite vectors, count/dimension mismatches and oversized responses. Input truncation is disabled to avoid silently dropping constraints. Very long messages plus retained constraints may exceed the model's 512-token input; shorten them or select BM25 if Ollama rejects the input. This first version does not implement long-query chunking, learned rejection thresholds, reranking, multilingual retrieval or an out-of-domain detector. Proximity to a passage is not proof that it answers the question.

## Comparison

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_development.py --method all --output runtime\evaluations\retrieval_comparison_new
.\.venv\Scripts\python.exe -m unittest discover -s tests -q
node --check frontend\app.js
```

The comparison uses exactly the same 13 author-written questions, corpus and state-query construction for BM25, dense and hybrid. It records model digest, corpus/suite hashes and per-group retrieval metrics; answer packets remain unrated baseline-template outputs, not LLM-quality measurements. See [EVALUATION.md](EVALUATION.md). This development suite was written with the corpus and cannot establish independent gains. Preserve negative results; do not tune against a reserved final test.

Next research milestones: independently labelled paraphrases and cultural questions; freeze development/final-test dialogue families; compare retrieval and actual model answers under controlled settings; collect human reviews. Train intent/entity models only after reliable annotations exist, then compare against the current explicit rules. The neural embedding stage is implemented now; learned intent/NER and cross-encoder reranking are not.

### Recorded first comparison

All 37 automated tests pass, including malformed embeddings, corpus/model cache invalidation, RRF, failure rollback, unlocked session handling and frontend request/stream consumption. The real HTTP two-turn Qwen3+hybrid smoke also passed: `runtime/evaluations/hybrid_llm_smoke_01/report.json`, 33/23 visible content chunks, approximately 19/9 seconds to first text in this run. Both turns record the selected hybrid method and actual embedding digest. The second answer omitted an inline citation and was correctly flagged; a transport smoke pass is not an answer-quality or factual-verification pass.

`runtime/evaluations/retrieval_comparison_01/report.json` contains the completed real local run. Across 11 positive development cases, BM25 and hybrid each achieved mean Recall@3 0.9697, MRR@3 1.0000 and NDCG@3 0.9731. Dense-only achieved 0.9242, 0.9545 and 0.9222 respectively. On the two unsupported cases, no-hit rates were 0.5 for BM25 and 0.0 for both dense and hybrid. **There was no measured improvement on this tiny author-written suite**, and nearest-neighbour methods retrieved evidence for both unsupported cases. Preserve this result; no score threshold or fusion weight was tuned to improve it. Independent evaluation and stronger answer-level abstention remain necessary.
