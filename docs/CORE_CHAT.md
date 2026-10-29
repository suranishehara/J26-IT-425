# Core conversation improvements — 19 October 2026

The user explicitly postponed manual annotation and trained intent/entity work. Intake data, review tools and all originals are preserved; nothing was approved or trained. This milestone improves the working chatbot without relying on volunteer annotation completion.

## Implemented

- Clear follow-ups such as “Can you explain that more simply?” retain the prior successful **user topic** for evidence retrieval. Retrieval does not use the previous generated answer as a new source. Repeated follow-ups do not recursively append full prior queries.
- Explicit new fabric/occasion/constraint terms, and “about/regarding” a new topic, prevent this narrow follow-up inheritance. Ordinary constraint retention still works as before. These are explicit English rules, not a trained reference-resolution system.
- Supplied Component 2/3 options can be addressed by ordinal or number: “Explain the second option”, “Explain option 4”, “Compare the second and fourth options”. Original option numbers accompany the selected records in model grounding. The model receives those selected records rather than silently substituting the first result.
- “The other one” resolves only with exactly two supplied options and one previously selected option. Missing results, unavailable numbers, one-item comparisons and ambiguous alternatives prompt a focused clarification.
- Those clarification replies are deterministic and explicitly reported as `deterministic_clarification_no_llm`, with retrieval `not_run_clarification`. They do not call an embedding model or Qwen3, invent a recommendation, or masquerade as generated LLM text. Normal grounded replies continue to stream from the selected local model.
- Topic and selected-item state commit only after successful completion. Model/retrieval failure or abandoned streaming does not commit partial state. Session restart/reset clears them. Supplying different component context clears the prior selected-item reference.
- Frontend method details show retained-topic follow-ups and selected option numbers. Missing-lookup clarification is distinguished from a search that returned no evidence.

## Try it

Refresh `http://localhost:8003/` and start a new conversation after the server restart.

1. Ask `What does black tie mean for a wedding invitation?`
2. Follow with `Can you explain that more simply?`
3. Confirm the evidence still concerns black tie and the method details say the previous user topic was retained.
4. Ask `Tell me more about batik`. Confirm this is treated as a new explicit topic, not a black-tie continuation (existing user constraints may still be retained).
5. In a new conversation, ask `Explain the second option` without supplying results. Expect a request for the actual Component 2/3 JSON, not a guessed item.

For the result-selection path, paste this **fictional software fixture**, selecting Component 3. These are not actual generated recommendations or measured properties:

```json
{"property_basis":"typical_profile","designs":[
  {"name":"Demo shirt","rule_score":7,"reasons":["Fixture: recorded structured drapability matched"],"conflicts":[]},
  {"name":"Demo kurta","rule_score":3,"reasons":["Fixture: selected style matched"],"conflicts":["Fixture: recorded drapability mismatch"]}
]}
```

Ask `Why the second option?`, then `What about the other one?`, then `Explain option 6`. Expected: Demo kurta, then Demo shirt, then an unavailable-option clarification. Typical profiles and rule-score limits must remain explicit; no upstream recommendations or saved profiles change. For real use, supply actual outputs instead of the fixture. The adapter retains up to five item summaries; numbers refer to that retained order. Name-based matching, arbitrary pronouns and references to suggestions appearing only in generated prose are not implemented.

## Verification and limits

63 automated tests pass. New checks cover retained user topics, new-topic switching, second/fourth selection, selected-only model grounding, ambiguity, missing/out-of-range results, paired “other” references, context reset and failure rollback. These are software checks, not measured conversational accuracy. The review/model-training work remains deferred at the user's request.

Actual local two-turn Qwen3/BM25 smoke is recorded at `runtime/evaluations/core_followup_smoke_01/report.json`: 50/42 content chunks; the second turn reports `topic_reused:true` and retains the black-tie evidence. First visible text took approximately 36/17 seconds in this run, not a general latency guarantee. Both responses omitted inline evidence IDs and were flagged by the citation checker. Transport/topic success is not a citation-completeness or factual-quality pass.

The core still uses rule-based intent/entities, selectable BM25/dense/hybrid retrieval and optional Qwen3 generation. Citation-ID checks do not verify factual entailment. Unsupported questions can still retrieve loosely related passages; this change does not add a general unknown-question detector. Cultural/explanation quality requires separate assessment. No Component 1–3 files or volunteer labels were changed.
