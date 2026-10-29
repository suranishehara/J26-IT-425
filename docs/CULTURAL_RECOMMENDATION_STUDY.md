# Optional cultural guidance and recommendation evaluation

The Recommendations tab now offers baseline and Sri Lankan guidance modes. Both use the same catalogue, budget constraint, preferences and available size charts. Guidance adds at most two points using versioned graph relationships in `data/cultural_guidance.json`. It never excludes Western clothing or mandates a cultural choice.

The four hypotheses are Osariya for weddings, batik for casual/office contexts, and handloom for festivals. These come from the project proposal, not an expert-validated cultural standard. Occasion, workplace and individual preferences vary. The baseline already contains occasion tags, so this comparison measures the **additional explicit guidance bonus**, not all cultural information versus none. Textile-technique metadata needs a reliable source; it cannot be verified from an image alone.

## Prepared pilot

Open `runtime/recommendation_studies/pilot_02/rate_garments.html` in a browser. Enter an anonymous reviewer ID, rate every candidate from 0 to 3, then export the JSON. Rankings are hidden and candidates shuffled; this is interface blinding, not protection against someone inspecting the embedded packet. Ratings remain in memory until exported: do not refresh before downloading.

The four constructed scenarios and catalogue snapshot are frozen in `packet.json`, including images, rankings and the exact cultural rules. Prices, synthetic demonstration images and catalogue tags remain illustrative. This is a workflow pilot, not evidence of real-world effectiveness. Independent local reviewers should validate rules and catalogue metadata before a formal study. Use multiple reviewers and more representative scenarios for research conclusions.

Copy the exported JSON into the pilot directory as `relevance_ratings.json`. From the Component 2 project folder run:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_recommendations.py --packet runtime\recommendation_studies\pilot_02\packet.json --ratings runtime\recommendation_studies\pilot_02\relevance_ratings.json --output runtime\recommendation_studies\pilot_02\results.json --k 5
```

Each reviewer exports a separate file; choose a new output filename for each. The evaluator reports per-reviewer scenario averages, not significance or inter-rater agreement. Do not tune cultural rules on final evaluation judgments; use separate development scenarios/reviewers, then freeze the rules for final testing.

## Metrics

- Precision@K: relevant items in the top K divided by K (even when fewer items exist).
- Recall@K: relevant items retrieved divided by all relevant items in the eligible, within-budget catalogue pool. Every eligible item must be judged.
- NDCG@K: ranking quality using graded relevance, gain `2^grade - 1` and logarithmic discount.
- Grades 2 and 3 count as relevant for precision/recall. No-relevant pools yield undefined recall/NDCG, not invented zeros.
- Global and traditional recall are also reported within the combined top K; this is coverage analysis, not demographic fairness.

Packet integrity, reviewer identity and complete judgments are checked. No research ratings or performance scores have been fabricated. Automated tests use clearly artificial fixtures only.

To create a new frozen packet after improving the catalogue:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_recommendation_study.py --output runtime\recommendation_studies\pilot_03
```

The incomplete `pilot_01` packet was created during a Windows encoding check; use the complete `pilot_02` instead.
