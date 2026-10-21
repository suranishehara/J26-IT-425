# Component 4 — NLP research scope and evaluation plan

Started 19 October 2026. English-first task-oriented fashion conversations with Sri Lankan terminology, scoped cultural evidence and read-only explanations of upstream outputs. This is a substantial planned research component; a working baseline does not complete that plan.

## Research questions

1. Does domain-aware intent/constraint extraction improve constraint retention and task success over simply forwarding conversation text to a generator?
2. Does hybrid retrieval plus reranking improve relevant evidence retrieval over BM25 and dense-only baselines?
3. Does grounding explanations in both retrieved passages and actual Component 2/3 outputs improve citation support/factual consistency compared with ungrounded generation?
4. How do relevance, explanation quality and cultural appropriateness differ for global, Sri Lankan/traditional and mixed-context queries?

Do not claim no related system exists without a systematic literature review. NLP contribution should be evaluated through actual annotated tasks and controlled comparisons, not inferred from use of an LLM API.

## Milestones

| Milestone | Current status |
|---|---|
| English intent/entity baseline, terminology normalization and session-state updates | Implemented, rule-based |
| Source-aware knowledge schema and lexical retrieve/explain baseline | Implemented with 23 scoped entries; source-and-scope review documented in KNOWLEDGE_REVIEW.md; not a complete or expert-reviewed corpus |
| Read-only adapters for Components 2/3 | Explicit pasted response JSON implemented; no live execution/automatic profile transfer |
| Annotated real intent/entity/dialogue dataset | User-confirmed permitted volunteer collection ingested: 630 rows/229 texts; provided intents need review, entity/slot labels pending; one connected participant/text group blocks three-way splitting. See VOLUNTEER_INTAKE.md. Starter fixtures remain separate. |
| Train/evaluate learned NLU against rules | Volunteer app-intent/entity annotation/training remains deferred. A separate native SIMMC dialogue-act experiment is completed against TF-IDF/majority, not falsely mapped to incompatible app rules. See SIMMC_NLP_EXPERIMENT.md. |
| Published fashion-dialogue dataset | SIMMC 2.1 downloaded, hashed and filtered: 6,402 fashion conversations / 75,220 utterances; provider splits and native labels preserved. DeBERTa native-act held-out accuracy 97.53%, macro-F1 0.9724; TF-IDF 87.46% / 0.8598. Text-only, one seed, not full multimodal or local cultural evaluation; no app activation. |
| Dense embeddings and hybrid retrieval | Implemented: local all-minilm:22m, cosine and RRF; independent gains not established |
| Cross-encoder reranking | Pending |
| Evidence-conditioned language-model generation | Local Ollama adapter/streaming implemented; Qwen3 4B chosen as initial local model; setup/smoke and independent quality evaluation are separate |
| Citation/claim checking and robust abstention | Baseline exact evidence display only; broader implementation pending |
| Live approved component integrations | Pending, with explicit authority before mutations or recommendation execution |
| Independent human evaluation and held-out reporting | Pending |
| Local chat-history interface | Implemented: SQLite persistence, resume/rename/recoverable delete/restore; supporting usability feature, not automatic research/training data |

Development evaluation tooling is now implemented: 13 author-written retrieval cases, group-wise Recall@K/MRR@K/NDCG@K, separate unsupported-query no-hit diagnostics, reproducibility hashes and unrated baseline answer packets. See [EVALUATION.md](EVALUATION.md). These fixtures are not independent annotations or final-test evidence. Human relevance, factual consistency and explanation/cultural-scope ratings remain uncollected.

Knowledge expansion adds a separate sixteen-case author-written suite and an actual three-method retrieval comparison. The original suite remains unchanged and its regression result is retained. See [KNOWLEDGE_REVIEW.md](KNOWLEDGE_REVIEW.md), including unrelated-query hits and the wedding-query ranking change. No independent expert approval or human answer-quality scores are claimed.

## Annotation schema

Intents: recommend, explain, compare, update_preferences, knowledge_question, greeting, clarify. Confirm definitions, potentially multi-intent handling and disagreements before freezing a trained model's labels.

Slots: occasion, budget_lkr, weather, setting, style, colours, included/excluded fashion terms. Future extensions can include explicit garment/fabric entity types, body measurements, fit and source constraints. Do not assume current generic fashion-term entities are a trained NER system. Their offsets refer to normalized English text.

Conversation annotations should mark carried-over values, changed values, exclusions, removed constraints, unresolved ambiguity and reference resolution. Splits must keep whole conversations and paraphrase/template families separate. Do not split turns from one dialogue across train/test. Separate knowledge authoring/development from independently labelled evaluation queries; never report fixture tests as benchmark accuracy.

Knowledge entries require stable IDs, source reference, text, topic tags, region, scope and status. Draft entries are excluded from retrieval. Source-checked means the passage was checked, not culturally expert-reviewed or universally applicable. Preserve dates, attribution and source licences. In particular, Indonesian batik symbolism must not be transferred to Sri Lankan event rules; Component 2's proposed cultural bonuses must not be promoted into cultural facts.

## Controlled evaluation

- Intent: macro-F1 and per-class confusion.
- Entities/slots: span and normalized-value precision/recall/F1; budget/currency parsing accuracy.
- State: slot-value accuracy, joint state accuracy, update/exclusion correctness and task success across follow-ups.
- Retrieval: independently judged relevance, Recall@K, MRR, NDCG; BM25, dense, hybrid, hybrid+reranker on the same frozen questions/corpus.
- Answer: relevance, supported-claim fraction, citation correctness/completeness, factual consistency with upstream outputs, appropriate abstention.
- Explanation: clarity, usefulness and contextual/cultural appropriateness using a documented human rubric. Reviewers with local expertise for Sri Lankan claims; anonymize systems and randomize ordering where practical. Report agreement and disagreements.
- Group reports: global, Sri Lankan/traditional, mixed-context, insufficient-evidence and multi-turn cases. Do not guess a query's provenance from user appearance.
- Comparisons: no-retrieval LLM, basic RAG, proposed NLU/state-aware grounded RAG. Control generation model/settings when isolating retrieval/NLU effects. Keep all systems evaluated on the same held-out set.

No human ratings, test benchmark scores or multilingual capability have been measured yet. Tests are software mechanics only. Sinhala/Tamil and code-switching are optional separately evaluated extensions, not automatic features of cultural terminology normalization.

## Starting sources / methods

- Lewis et al., [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401).
- [Sentence Transformers retrieve and rerank](https://sbert.net/examples/sentence_transformer/applications/retrieve_rerank/README.html).
- [Dialogue State Tracking with Incremental Reasoning](https://aclanthology.org/2021.tacl-1.34/).

These inform the plan, not a claim that their neural methods are already implemented. Scope needs meaningful evidence and evaluation, not just a larger chat UI.
