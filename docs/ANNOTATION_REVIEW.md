# Human annotation review and new-cohort collection

## Ready now

An offline review page was built for all 229 distinct messages at `runtime/evaluations/volunteer_review_01/review.html`. It contains the original source contexts, provided intent labels and **separately marked rule suggestions**. Every human annotation starts pending/null. No label has been automatically approved and no model trained.

Open that file in a browser, or run the read-only loopback preview:

```powershell
.\.venv\Scripts\python.exe scripts\serve_nlu_review.py
```

Visit `http://localhost:8004/review.html`. Port 8003's chatbot/parser is unchanged. The preview serves only the review and collection HTML pages; no write/save API or directory listing exists. Messages stay local, but loopback is not multi-user authentication. Stop the preview when finished. This does not send a review invitation to anyone.

## Reviewer steps

1. Enter a pseudonymous reviewer code such as R01. Do not put names or contact details in the annotation file.
2. Read the message and use the occurrence/conversation selector to inspect the relevant original contexts. One distinct wording may have many occurrences; if intent differs by context, defer it for discussion rather than impose one shared label.
3. Choose the intent, assign a paraphrase-family ID and fill entity, slot and removal JSON. Use `[]`, `{}`, `[]` only when you checked and found none. Blank means unannotated. Follow [NLU_DATA_COLLECTION.md](NLU_DATA_COLLECTION.md).
4. Prefer independent annotation without revealing source labels or rule suggestions. If you click **Show source intent and rule suggestions**, the record is marked assisted. Copying predictions only fills a draft; correct wrong/missing entities, negation and slots before approval. Assisted annotations are not independent rule-baseline ground truth.
5. Tick the review confirmation and choose **Save reviewed annotation**. Incomplete fields, invalid label types, missing family/reviewer codes, bad spans and malformed JSON are rejected. UI checks are an aid, not proof of correct semantics or consent. The Python annotation validator remains authoritative for future normalized exports.
6. Use **Save draft** for unfinished work or **Defer for discussion** with a note when uncertain. Navigation preserves modified fields as an in-tab draft, not an approved label.
7. Click **Download progress JSON** regularly and before closing/reloading. There is no automatic disk save, network upload or localStorage persistence. Confirm the browser saved the file. Resume it later with the file chooser; pack/source identities must match. The exported progress is a review record, **not a split-approved training manifest**.

Give a fresh copy of the review page to a second reviewer. Each should return a separate file without reading the other's labels. Independent annotation requires leaving suggestions hidden. Human adjudication and agreement reporting are subsequent steps; do not merge reviewed labels automatically or call a single review expert validation. Source wording such as “fabric term lapels” may need ontology notes instead of invented supported entities.

## New data collection form

The standalone form is `review_tool/collect.html`, also previewed at `http://localhost:8004/collect.html`. Send the file itself through your approved study process if volunteers are on other computers; **localhost links point to the recipient's own machine and cannot be used remotely**. No public hosting or file sharing was set up.

The form asks for three naturally worded messages per intent (up to 21 per participant), plus optional context. It contains task descriptions, not example answers to copy. It records an assigned participant code/cohort and explicit participant permission confirmation, then downloads JSON. It neither uploads nor silently stores responses. The institutional study information/consent process must still be supplied separately. Requested purpose is retained separately from reviewed intent; entity/slot annotations remain null.

Suggested next collection:

- Training expansion: prioritize genuinely varied greeting, explain and clarify wording; these each have only three distinct messages now. Collect variation across situations rather than copy-and-replace fabric names.
- Development: a new participant cohort, assigned before collection.
- Reserved final test: a different new cohort, kept separate from development and not examined for tuning. A practical initial plan is 10 development and 10 final-cohort volunteers, three messages per intent each (30 per intent per cohort), subject to actual usable/unique counts and your supervisor's study design.

Keep participant/conversation/paraphrase families together. Audit new responses against existing text and one another. Common greetings may naturally repeat: preserve original responses but mark cross-cohort exact/near-duplicate scored items as excluded with explicit reasons, or redesign collection. Do not claim participant-independent, novel-wording evaluation by silently ignoring overlap. Preserving context is separate from scoring an excluded duplicate. Final cohort allocation does not itself guarantee independence; checks are still required.

## Current limits and verification

53 automated tests pass, including review-pack provenance, source-link integrity, malicious text escaping and browser-side annotation-type/span validation. No genuine human-review completion, inter-reviewer agreement, new-cohort collection, trained intent/entity model or independent test accuracy is claimed. Reviews and consent are human tasks. The source workbook, intake audit, and existing production NLP rules remain unchanged.

To rebuild a fresh page from the preserved intake (never overwrite an old run):

```powershell
.\.venv\Scripts\python.exe scripts\build_nlu_review.py --output runtime\evaluations\volunteer_review_02
```

For a different page, pass `--review` to the preview command. Close/restart the preview only after downloading in-tab progress. See [VOLUNTEER_INTAKE.md](VOLUNTEER_INTAKE.md) for why training/testing remains gated.
