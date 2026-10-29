# Knowledge expansion and source review — 19 October 2026

## What changed

`data/knowledge.json` is now **curated-0.3: 23 active passages** (previously 10): seven Sri Lankan, fourteen global/contextual, and two local implementation passages. There are eleven distinct external source URLs. Thirteen new focused passages were added; existing IDs and the original 13-case development suite were preserved.

This is an assistant source-and-scope review, **not cultural expert approval, independent factual verification, or a complete cultural knowledge base**. Every external passage remains `source_checked_not_expert_reviewed`. The corpus-level review record explicitly leaves cultural expert review and independent answer evaluation pending. The two `project_verified` passages describe implementation, not validated fashion advice.

## Newly checked source register

These four publisher pages were opened and their relevant body sections read. New passages record `verification_method: publisher_page_body`, a section locator, checked date and reuse note. Text is a short original summary, not a copied article. Checking a page on this date does not imply that every statement on that page was verified independently.

| Publisher / source | New passage IDs | Supported subject | Important boundary |
|---|---|---|---|
| [Sri Lanka Export Development Board — Handloom Industry](https://www.srilankabusiness.com/apparel/handloom/) | `sl_handloom_materials`, `sl_handloom_garments`, `sl_handloom_craft` | Industry materials, sarees/shawls/sarongs, cottage production and weaving heritage | Product examples are not wedding permission, fibre tests or ethical certification of an uploaded item. Brand-specific promotional claims were not generalized. |
| [Emily Post — Attire Guide](https://emilypost.com/advice/attire-guide-dress-codes-from-casual-to-white-tie) | `global_business_casual`, `global_business_formal`, `global_dressy_casual`, `global_semi_formal`, `global_black_tie_optional`, `global_white_tie` | Six named dress codes and concise clothing examples | Western etiquette framework, not universal Sri Lankan rules. Employer/host expectations still need clarification. Examples are not mandatory gender prescriptions. |
| [V&A — An A–Z of Opus Anglicanum](https://www.vam.ac.uk/articles/a-z-opus-anglicanum) | `global_plain_weave`, `global_satin_weave`, `global_velvet_pile` | Plain weave, satin construction and velvet pile | Historical textile glossary, not photo recognition or measured weight/drape. Satin-versus-fibre distinction is explanatory terminology reasoning. |
| [GINETEX — Our Mission and Objectives](https://www.ginetex.net/article/FR/our-mission-and-objectives) | `global_care_labels` | Treatment categories and article-specific professional care selection | No guessed washing temperature, care-symbol recognition, copied symbol artwork or item-specific care guarantee. |

The earlier sources remain in the corpus with their own provenance: EDB batik and handloom overview, Sri Lanka Tourism visitor/shopping pages, Emily Post wedding guidance, UNESCO Indonesian batik, and V&A Indian textiles. Their scope restrictions were retained. Indonesian batik symbolism and Indian sari context must not become Sri Lankan ceremony rules.

## How the expanded knowledge reaches a reply

1. The current question and retained conversation constraints form the retrieval query.
2. BM25, local all-minilm:22m dense search, or hybrid RRF ranks the active passages. Draft entries are excluded.
3. Selected IDs, text, source, review status and scope go to the baseline answer or selected local Ollama model.
4. The evidence column displays the retrieved sources and scope. A citation-ID check is not claim-entailment verification.

The server reads the corpus for each request; it does not need restarting for this JSON change. Dense corpus embeddings are rebuilt when the knowledge SHA-256 changes. The live health endpoint was checked and reported 23 entries. No Components 1–3, volunteer data, trained NLU models, generation prompts or user history were modified by this milestone.

## Software and development checks

**76 automated tests pass.** Four new tests cover corpus provenance/review state, retrieval of explicit new terms, valid expansion fixtures/unrated review packets, and preservation of cultural/measurement boundaries. These checks are not model accuracy measurements.

New suite: `data/evaluation/knowledge_expansion.json`, sixteen author-written cases: three Sri Lankan, ten global, one mixed, and two insufficient-evidence questions. Fourteen cases have provisional positive relevance labels. The original thirteen cases are unchanged.

Saved outputs:

- `runtime/evaluations/knowledge_expansion_01/report.json`: actual BM25/dense/hybrid comparison, using local all-minilm:22m for dense retrieval.
- `runtime/evaluations/knowledge_expansion_01/human_review.json`: baseline answers with all human ratings blank.
- `runtime/evaluations/knowledge_regression_01/report.json`: unchanged original suite, BM25 on the expanded corpus.

| New suite, K=3 | Recall@3, 14 answerable cases | MRR@3 | NDCG@3 | No-hit fraction, 2 insufficient-evidence cases |
|---|---:|---:|---:|---:|
| BM25 | 1.0000 | 1.0000 | 0.9943 | 0.5000 |
| Dense | 1.0000 | 0.9643 | 0.9781 | 0.0000 |
| Hybrid | 1.0000 | 0.9643 | 0.9781 | 0.0000 |

The questions were written alongside the passages, so high retrieval coverage is expected and **not evidence of generalization, cultural accuracy or superiority of a method**. Dense/hybrid return passages even for gibberish: retrieval does not establish that evidence answers a question. Independent abstention and answer-support evaluation remain necessary.

On the original suite, BM25 mean Recall@3 over eleven answerable cases is **0.9394**, MRR@3 **1.0000**, NDCG@3 **0.9445**; insufficient-evidence no-hit fraction is **0.5** over two cases. For `sl04` (batik versus handloom at a wedding), new handloom/craft passages displaced two old labelled passages from the top three. Its original labels were not silently rewritten. Some newly returned passages are relevant too, so independent relevance adjudication is needed; this is not claimed as an improvement.

Reproduce with fresh output directory names (the scripts refuse overwrites):

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -q
.\.venv\Scripts\python.exe scripts\evaluate_development.py --suite data\evaluation\knowledge_expansion.json --method all --output runtime\evaluations\knowledge_expansion_02
.\.venv\Scripts\python.exe scripts\evaluate_development.py --output runtime\evaluations\knowledge_regression_02
```

## Try these in a new chat

- Which fibres are used in Sri Lankan handloom textiles?
- Are sarongs and shawls included in Sri Lankan handloom products?
- Does handloom automatically mean the product is ethically certified?
- Does black tie optional allow a dark suit instead of a tuxedo?
- Is satin always silk?
- What gives velvet its pile texture?
- Can you determine the washing temperature from a fabric category alone?
- Does a handloom sarong automatically meet a black tie optional invitation?

Choose a new chat for unrelated questions to avoid unintended retained constraints. Check the evidence column as well as the answer. With the Ollama engine selected, new source passages ground the generated answer; baseline mode remains a fast template/extractive demonstration. Neither mode has independent human-quality scores yet.

## Remaining review and expansion

- Obtain Sri Lankan garment/cultural expert review, with named consenting reviewer codes, dates, corrections and disagreements. No expert signatures or ratings have been invented.
- Source community-, venue- and occasion-specific guidance for Osariya/Kandyan dress, weddings and religious events. Do not infer mandatory colours, gender restrictions or ceremony approval from product pages.
- Do not activate incomplete research leads (Dumbara craft, historical university notices or convocation dress rules) without adequate source checking and clear craft/event/date scope.
- Review common Sinhala/Tamil terminology with speakers before adding multilingual claims; English-first remains the current system scope.
- Collect independent global/Sri Lankan/mixed questions and relevance judgements; reserve a final test before further tuning. Human answer relevance, factual consistency, explanation quality and cultural scope ratings are still pending.
- Improve missing-evidence detection and citation/claim support checks as a separate implementation milestone. Do not treat every retrieved passage as an answer.

For subsequent edits, keep IDs stable, attach a primary source and locator, distinguish source facts from inference, and keep incomplete entries `draft` (not retrievable). Never convert a generic material/category label into measured GSM, stretch or drapability.
