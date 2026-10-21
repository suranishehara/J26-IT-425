# Volunteer collection intake checkpoint

The user confirmed in chat that the messages were independently written by volunteers with permission for university research. This is recorded as **user-confirmed provenance**, not independently verified consent documentation. The current workbook no longer includes the earlier synthetic-data notes. Repetition is a data-quality/split issue; it is not evidence that the user's provenance confirmation is false.

## Saved intake

`runtime/evaluations/volunteer_intake_01/` contains:

- `source_workbook.xlsx`: byte-identical snapshot; original Downloads workbook was not changed.
- `source_records.json`: all 630 rows with source sheet/row, message, participant and conversation IDs, and provided intended-purpose labels.
- `duplicate_manifest.json`: all 229 normalized text groups and their source-row IDs. No source rows were deleted.
- `annotation_review_queue.json`: one queue item per distinct text, linked to all its source contexts. Reviewed intent, entity/slot/removal labels, family and split remain null. Empty annotations must not be assumed correct.
- `near_duplicate_candidates.json`: character-similarity candidates for human paraphrase-family review, not automatically accepted semantic duplicate labels.
- `audit_report.json`: counts, source SHA-256, provenance, conflicts and connected-group diagnostics.

Source SHA-256: `5f0a30ea30fe65ae69766fcfa404a6a587033c4af5d07c112ec2c05ec36a9525`.

| Provided intent | Collected rows | Distinct normalized texts |
|---|---:|---:|
| recommend | 90 | 30 |
| explain | 90 | 3 |
| compare | 90 | 90 |
| update_preferences | 90 | 30 |
| knowledge_question | 90 | 70 |
| greeting | 90 | 3 |
| clarify | 90 | 3 |
| Total | 630 | 229 |

No conflicting intended-purpose labels were found among exact-text groups. That does not establish label correctness. For example a greeting containing a request for fashion advice may need a primary-intent decision. Reviewers should inspect full conversations rather than copy the provided purpose blindly. Message turns lack actual assistant replies, so do not claim this collection evaluates complete assistant/user dialogue task success.

## Why training/test splitting stopped

The audit connects rows when they share a participant, a conversation or an identical normalized message, then takes transitive connected groups. **All 630 rows form one connected group.** Repeated messages shared across volunteers bridge their participant groups. Under these constraints there is no valid three-way train/dev/final-test split. Dropping duplicate occurrences while ignoring their other owners would hide, not resolve, that overlap.

No random row split, relaxed grouping, model training, production activation or reserved-test evaluation was performed. Class balance in row counts masks the very low distinct wording count for three intents. A perfect-looking random-split result would not demonstrate generalization to new messages or volunteers.

## Recommended way forward

Use this collection as candidate **training/development material only**, after independent label review. Collect a separate cohort of new, naturally worded messages for development and final testing, with no copied/shared prompts or message texts across cohorts. Reserve the final cohort before model selection and do not inspect it to tune rules. If using the current collection for training, do not later describe its already-inspected rows as an untouched final test.

Prioritize greeting, explanation and clarification: each currently has only three distinct texts. Aim initially for 30–50 genuinely different reviewed messages per intent across the expanded collection; counts alone do not guarantee sufficient coverage. Preserve conversational context and group closely related wording, even if punctuation or fabric names differ. Also review whether some classes are inherently context-dependent and need a context-aware classifier rather than isolated-text labels.

An alternative is to ask for additional independently written wording in all seven intents and rebuild whole participant/family-separated cohorts. Do not discard ambiguous/conflicting rows silently: record exclusions and reasons in a manifest. Consent records remain separately governed by university research requirements.

## Reproduce the intake

The importer uses read-only `openpyxl`; this library is already in the bundled document runtime but not added to the production app venv. On this machine, from Component 4:

```powershell
& 'C:\Users\Ravindu\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' scripts\audit_volunteer_workbook.py --input 'C:\Users\Ravindu\Downloads\Fashion_Chatbot_30_Volunteers_630_Messages.xlsx' --output runtime\evaluations\volunteer_intake_02 --user-confirmed-volunteer-permission
.\.venv\Scripts\python.exe -m unittest discover -s tests -q
```

Use a new output directory. The flag records the already-given human confirmation; it does not verify a consent document. Source rows and the snapshot are hashed. Grouping and review exports are deterministic; timestamps identify the run. No live services, LLMs or external APIs are contacted by intake. See [NLU_DATA_COLLECTION.md](NLU_DATA_COLLECTION.md) for the annotation definitions and training gate.
