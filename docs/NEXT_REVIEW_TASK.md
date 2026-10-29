# Next human review task

The optional cultural-guidance comparison and rating workflow are implemented. All 44 software tests pass; these are correctness tests, not recommendation-quality evidence.

1. Refresh the Recommendations tab at http://localhost:8001. Choose baseline or proposed Sri Lankan guidance under Ranking mode.
2. Open `runtime/recommendation_studies/pilot_02/rate_garments.html` in a browser. It contains 24 candidate/scenario ratings; choose a reviewer ID and score each from 0 to 3.
3. Export the completed ratings. Follow `docs/CULTURAL_RECOMMENDATION_STUDY.md` to calculate Precision@5, Recall@5 and NDCG@5.

Do not report recommendation performance until real relevance judgments exist. Local cultural validation, representative evaluation, skin-tone calibration/fairness, and local pattern/drape/texture training remain separate research requirements. Component 1 remains paused and unchanged.
