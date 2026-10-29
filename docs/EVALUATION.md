# Knowledge expansion and evaluation checkpoint — 19 October 2026

## Implemented

The corpus now has 10 entries, up from six. Four additions cover Sri Lankan batik process/base fabrics, handloom products, wedding-context questions and black-tie conventions. These are short original summaries, not copied articles or a training dataset. All four are source-checked, **not culturally expert-reviewed**. Existing source scopes are preserved. No Components 1–3 files were changed.

New source register:

| Evidence IDs | Primary source / checked passage | Deliberate boundary |
|---|---|---|
| sl_batik_materials | [EDB, Sri Lanka's Batik Industry](https://www.srilankabusiness.com/blog/sri_lankas_batik_industry.html), published 21 August 2015; wax decoration, local artists and cotton/silk paragraphs | Historical industry account, not current prices or wedding etiquette |
| sl_handloom_products | [EDB Handloom overview](https://www.srilankabusiness.com/handloom/overview.html), introduction and product/design range | Sarees, toys and upholstery; no universal GSM, fibre or sustainability certification claim |
| global_wedding_context, global_black_tie | [Emily Post Wedding Guest Attire](https://emilypost.com/advice/wedding-guest-attire), First Clues and Formal/Evening | Etiquette framework, not universal Sri Lankan community rules |

Checked 19 October 2026. Source rights remain with publishers. Source summaries do not establish that batik always beats handloom for any occasion. There is still no local-expert-reviewed Osariya, wedding-colour or community-specific ceremony corpus. Such gaps must prompt clarification, not invented cultural certainty.

## Run the development check

From the Component 4 folder:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_development.py --output runtime\evaluations\knowledge_dev_01
.\.venv\Scripts\python.exe -m unittest discover -s tests -q
```

Use a new output name on every run. Existing report directories are never overwritten. The development command does not call Ollama, download anything, or change upstream projects. It tests the real baseline dialogue/state-augmented BM25 query and records:

- Recall@K, truncated reciprocal rank (MRR@K) and binary-relevance NDCG@K.
- Separate global, Sri Lankan, mixed, project and insufficient-evidence groups, with denominators.
- No-hit rate on no-relevance cases separately; recall is undefined there, not 100%.
- Corpus/suite SHA-256, timestamp, ranked evidence IDs and baseline answers.
- An unrated `human_review.json` packet. Null means uncollected, not zero or a passing score.

The 13 questions and relevance labels were written while expanding this corpus. They are **development fixtures**, not an independent benchmark or final test. Relevant-ID lists are provisional and incomplete until judged independently. A lexical match on an insufficient-evidence question can still happen; no-hit is a retrieval diagnostic, not proof of correct abstention or factual consistency. Human-review answers always use the deployed top-three baseline evidence, even if a different retrieval K is requested for diagnostics.

## Human assessment protocol

### Recorded development run

`runtime/evaluations/knowledge_dev_01/report.json` records the first run. On the 11 positive author-written cases, mean Recall@3 was 0.9697, MRR@3 1.0000 and NDCG@3 0.9731. On two insufficient-evidence cases the no-hit rate was 0.5: a private family ceremony query retrieved general wedding passages despite lacking the required personal rules. This is a documented limitation, not evidence of successful unknown-question detection. Do not present these tiny, author-written results as model accuracy, human answer quality or independent research performance. No tuning was performed to hide that failure.

Collect new permitted questions from people who did not author these fixtures. Have two reviewers independently label relevant evidence and unsupported questions before system runs; involve Sri Lankan cultural/domain expertise for traditional cases. Keep whole dialogues and paraphrase families in one split. Freeze IDs, corpus hash, questions, labels and model settings before final testing. Do not tune on final outcomes.

For each frozen question run the same model/settings with no retrieval, basic RAG, and state-aware grounded RAG. Randomize answer order and replace system names with opaque IDs before review. The current generated packet is a transparent baseline development packet, **not blinded**; do not represent it as a completed controlled comparison. Local LLM answers must be collected separately with model digest/settings and successful full response recorded. Never save hidden model reasoning.

Rate each response 1–5, retaining each reviewer's independent record:

| Dimension | 1 | 3 | 5 |
|---|---|---|---|
| Relevance | Misses the question | Partly addresses it | Directly answers or asks the necessary clarification |
| Factual consistency | Major contradiction/invention | Some unsupported details | All checkable claims supported by supplied evidence/context |
| Explanation quality | Confusing or no rationale | Understandable but incomplete | Clear, useful reasoning and appropriate uncertainty |
| Cultural scope | Stereotypes or universalizes a local rule | Scope incompletely stated | Correct source/community scope; no unsupported preference hierarchy |

Mark supported claims and total checkable claims; count unsupported details explicitly. For no-checkable-claim responses the fraction is undefined, not automatically perfect. Assess citation support and completeness by reading passages, not merely matching IDs. An abstention can be appropriate; blanket refusal despite adequate evidence is not.

Report per-group means/distributions and sample counts, supported-claim fraction, citation support/completeness, appropriate abstention and reviewer agreement (plus disagreements/adjudication). No human ratings, agreement, cultural validation or final-test model accuracy are available yet.

## Try in the app

1. New chat: `What base fabrics are used for Sri Lankan batik?`
2. `Does that mean every batik fabric has the same weight?`
3. New chat: `Is batik always better than handloom for a Sri Lankan wedding?`
4. New chat: `Can traditional attire replace a tuxedo on a black tie invitation?`

Inspect evidence IDs/source scopes, not just fluency. The model should distinguish technique from material, avoid inventing weight and ask about host/venue requirements rather than impose a universal ranking. These manual checks are not completed human research ratings.
