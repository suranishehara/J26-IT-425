# Component 4 — Complete User Manual and Test Guide

**Application:** Explainable Conversational Fashion Advisor  
**Application address:** http://localhost:8003/  

## 1. Purpose and current capabilities

Component 4 provides English fashion conversations grounded in a small, sourced knowledge base. It explains global dress conventions, Sri Lankan textile context and explicitly supplied Component 2/3 results. It supports follow-up questions, inspectable preferences, source cards and local conversation history.

There are **two different NLP uses** in this component:

| Part | Current implementation | Where to use it |
|---|---|---|
| Chatbot intent/preferences | Inspectable English rules and terminology aliases | The web interface |
| Evidence retrieval | BM25; optional local sentence embeddings or hybrid fusion | Evidence retrieval dropdown |
| Natural answer generation | Local Qwen3 4B through Ollama, or a template baseline | Reply engine dropdown |
| Trained dialogue-act research model | Fine-tuned DeBERTa-v3-small compared with TF-IDF logistic regression | Saved standalone experiment, **not connected to the chat UI** |

What the application does not currently do:

## 2. Requirements and first-time setup

### 2.1 Existing project setup

On this computer, the Python environment, Ollama, `qwen3:4b` and `all-minilm:22m` were previously installed. **Do not download models again unless they are missing.** A model may be installed on disk but unloaded from memory; unloading is normal.

You need:

- Windows and PowerShell.
- Python 3.12 for the project's environment.
- A current browser with streaming response support, such as Chrome or Edge.
- Ollama running locally for generated replies or semantic retrieval.
- Sufficient free RAM/VRAM and disk space. Actual requirements vary; the local model's download size is not its total runtime memory requirement.

The web baseline uses Python's standard library. The standalone training experiment additionally uses optional packages; they are not needed just to chat.

### 2.2 Open the correct project directory

```powershell
Set-Location -LiteralPath 'C:\Users\component-4-conversational-ai'
```

Use the exact directory spelling, including the en dash in the parent folder. The remaining commands assume this directory is current.

### 2.3 Create an environment only if it is missing

Skip this step when `.venv` already exists and works.

```powershell
py -3.12 -m venv .venv
```

Activation is optional. The explicit interpreter command avoids PowerShell activation-policy issues:

```powershell
.\.venv\Scripts\python.exe --version
```

### 2.4 Check Ollama

```powershell
ollama --version
ollama list
```

Look for `qwen3:4b` and `all-minilm:22m`. If Ollama is missing, install it from the [official Windows download page](https://ollama.com/download/windows). If a required model is genuinely missing, download only that model:

```powershell
ollama pull qwen3:4b
ollama pull all-minilm:22m
```

These downloads need internet access and additional disk space. `qwen3:4b` is the answer generator; `all-minilm:22m` produces embeddings. The chatbot does not download models automatically.

If the installed Ollama application is not running, open it from the Windows Start menu. `ollama serve` is an alternative foreground terminal command, but do not start a second server when port 11434 is already served.

## 3. Start and stop the application

### 3.1 Start

```powershell
.\.venv\Scripts\python.exe backend\app.py
```

Keep the terminal open, then open **http://localhost:8003/** in the browser. If this page is already serving the correct application, reuse the existing server rather than starting a duplicate.

Normal startup enables the local SQLite chat-history database. No paid API key is required.

### 3.2 Readiness checklist

1. The title “Ask, understand, explain.” appears.
2. Chat history and New chat are visible.
3. Send message becomes available after the conversation initializes.
4. Click **Refresh models** in Chat settings.
5. Confirm `qwen3:4b` is selectable if you want generated answers.
6. Confirm semantic/hybrid retrieval is selectable if you want embedding-based search.

On initialization, Ollama mode is selected if compatible local models are available; otherwise the interface uses baseline mode. Evidence retrieval defaults to BM25. Inspect the dropdowns instead of assuming a particular mode.

### 3.3 Stop or restart

To stop the web server, press **Ctrl+C in the terminal that started it**. To restart, run the same start command again and refresh the browser. Saved completed chats remain in the local database.

Stopping the web server does not uninstall Ollama or delete any models. Do not terminate an unrelated process just because a port is occupied; see troubleshooting.

## 4. Interface guide

The desktop interface contains a history rail and four functional sections. Narrower windows may stack sections; this does not remove functionality.

### 4.1 Chat history

- **New chat:** starts a separate conversation with cleared topic/preferences/component context.
- **Chat title:** reopens that saved conversation.
- **Rename:** changes its display name.
- **Delete:** moves it to Recently deleted after confirmation.
- **Recently deleted → Restore:** returns the saved chat to active history.

There is only one new-chat action; a second “New conversation” button is not required. A newly created chat may initially have the title “New conversation”—that is a title, not a duplicate button.

### 4.2 Conversation

- Read user and ADVISOR messages here.
- Enter a message in **Your message**, then click **Send message**.
- Messages can contain up to **2,000 characters**. Empty messages are not sent.
- Use the Send message button for predictable submission. Enter in the textarea can insert a new line; no special Enter-to-send shortcut is promised.
- During generation, the user message appears immediately, followed by a waiting indicator and streamed text.
- The transcript follows the latest output; scroll upward to inspect older turns. Incoming output can move the view back to the latest message.
- **Stop response** cancels the browser request. Incomplete output is not a completed answer.
- Status below the composer identifies the turn, model/baseline, supplied-context use and citation-ID warnings where relevant.

### 4.3 Chat settings

| Setting | Meaning | Recommended use |
|---|---|---|
| Local baseline — templates | Fast deterministic/template response using selected evidence | Mechanics tests and comparison |
| Local language model — Ollama | Genuine local generated response | Natural conversation demonstration |
| Installed model | Available local chat model, initially Qwen3 4B on this setup | Select `qwen3:4b` |
| Refresh models | Rechecks generator and embedding availability | After starting Ollama or installing a model |
| BM25 — keyword baseline | Lexical evidence ranking | Reproducible first test; no embedding service needed |
| Semantic — local sentence embeddings | Meaning-oriented vector retrieval with all-minilm | Paraphrase comparison |
| Hybrid — BM25 + semantic (RRF) | Combines keyword and semantic ranks | Controlled retrieval comparison |

An instant baseline response is expected. Do not add artificial delays or present a baseline response as neural generation. Qwen3 can spend time preparing before visible text appears; hidden reasoning is not displayed.

Retrieval scores are ranking values, not answer confidence or accuracy. Semantic/hybrid search can return unrelated passages even for unsupported inputs. A selected method that fails during a request must produce an error rather than an undisclosed replacement answer.

### 4.4 What the NLP understood

The panel displays:

- **Intent:** one of the current app's rule labels: recommend, explain, compare, update preferences, knowledge question, greeting or clarify.
- **Conversation constraints:** retained occasion, budget, setting, weather, style, colours and included/excluded fashion terms where recognized.
- **Recognized terms:** the matched text and normalized term, including exclusions.
- **Explain a Component 2 or 3 result:** optional read-only JSON context, described in Section 7.

Inspect this panel: a natural sentence may not match the rules as intended. Correct it explicitly or start a new chat. Arbitrary English phrasing, complex negation and every currency/unit are not supported. Budget extraction is LKR-oriented; do not assume automatic currency conversion.

### 4.5 Evidence and scope

Each retrieved card shows its ID, title, summary, scope, source reference, region and review state. Click an external source link to open its publisher page. Local component references may display a file reference instead.

- Sri Lankan sources must not be treated as universal rules for every community or event.
- Western etiquette examples are not automatic Sri Lankan dress restrictions.
- Indian/Indonesian textile descriptions are not proof of Sri Lankan ceremony rules.
- A citation such as `[sl_temple]` identifies a passage; it is not automatic verification that every generated sentence is supported.
- “Citation IDs need review” means missing or unknown source IDs were flagged. Even when no warning appears, factual consistency still needs review.
- Footer methods show the actual retrieval/generation path and knowledge checksum. Missing-option clarification can deliberately skip retrieval and the LLM.

## 5. Complete first conversation

### 5.1 Basic setup

1. Click **New chat**.
2. Select **Local language model — Ollama**.
3. Select **qwen3:4b**.
4. Select **BM25 — keyword baseline** initially.
5. Keep the optional Component 2/3 JSON box empty.

### 5.2 Send a sourced question

```text
What clothing guidance is available for visiting the Temple of the Tooth?
```

Wait for completion. Check that the evidence concerns that site. A supported answer can explain the documented coverage guidance while keeping it site-specific; exact wording is not fixed.

### 5.3 Test follow-up continuity

In the same chat:

```text
Can you explain that more simply?
```

The retained topic should remain the temple guidance. The methods area can indicate that the previous user topic was retained. This is a narrow English follow-up rule, not unrestricted conversational memory.

### 5.4 Test preference extraction in a separate chat

Click New chat, then send:

```text
Suggest something traditional for an outdoor wedding under LKR 15,000
```

Check the NLP panel:

| Field | Expected extracted value |
|---|---|
| Intent | recommend |
| Occasion | wedding |
| Setting | outdoor |
| Style | traditional |
| Budget LKR | 15000 |

The app can discuss considerations but must not invent current catalogue stock or claim it ran Component 2. A budget constraint does not prove a recommended item is available at a particular price.

Then send:

```text
Actually office instead, not batik, handloom for hot weather
```

Expected state changes: occasion becomes office, weather becomes hot, handloom is included and batik excluded. Budget remains 15000. Existing outdoor/traditional preferences may remain because they were not explicitly removed. Then send:

```text
Remove the budget
```

The budget constraint should disappear. To clear all preferences and old topic/context, use New chat rather than relying on an arbitrary “reset” message.

## 6. Chat history, rename, delete and restore

### 6.1 Rename and reopen

1. Complete at least one harmless chat turn.
2. In the left rail, click Rename beside that chat.
3. Enter a meaningful name such as `Temple visit demo` and confirm.
4. Click New chat, then click `Temple visit demo` to reopen it.
5. Confirm its completed messages and relevant retained state appear.
6. Refresh the browser and check that saved chats remain available.

Names are 1–80 characters. A manual title is retained; otherwise the first completed user message supplies the automatic title. On page load, the latest active saved chat is reopened.

### 6.2 Delete and restore

Use a disposable demonstration chat, not an important research conversation.

1. Click Delete and read the confirmation.
2. Cancel first to test that nothing changes.
3. Click Delete again and confirm.
4. Confirm it disappears from active history and appears under Recently deleted.
5. Click Restore, then click the restored title to reopen it.

**Delete is recoverable, not permanent erasure.** There is no permanent-purge interface or automatic deleted-chat expiry in the current version.

### 6.3 Limits and drafts

- Up to 200 active chats; a conversation has a 100-turn limit.
- Start a new conversation when the turn limit is reached. Older completed messages remain readable.
- History controls are disabled while a response is being generated.
- Unsent drafts can trigger a confirmation before switching/new chats, but are not crash/reload backups.
- Stopped/failed partial responses are not saved as completed turns. If the browser missed a final server event, reopen the saved chat to inspect its actual state before retrying.
- All completed turns may be displayed, but only bounded recent context is sent to the generator; the model does not reread unlimited history.

## 7. Explain Component 2 and Component 3 results

### 7.1 Real-use procedure

1. Obtain an actual JSON recommendation response from Component 2 or design response from Component 3 using its existing workflow.
2. Start a separate Component 4 chat.
3. Expand **Explain a Component 2 or 3 result** under What the NLP understood.
4. Select the matching **Source component**.
5. Paste the response JSON object into **Response JSON**.
6. Ask an explanation or comparison question and send it.
7. Check the status indicates supplied component context and no upstream changes.

The integration is **manual, read-only supplied context**. Starting another component or pointing the dropdown at it does not establish a live API connection. Do not invent a menu/export button if that component's current UI does not provide one; its actual API result or developer inspection can supply the JSON.

### 7.2 Copy-and-paste Component 2 software fixture

**This JSON is fictional test data, not actual recommendations, prices, measurements or validated cultural advice.** Select **Component 2 — recommendations**, then paste only the JSON:

```json
{
  "recommendations": [
    {
      "name": "DEMO ONLY — handloom shirt",
      "score": 7,
      "reasons": ["Fictional fixture: recorded office occasion matched", "Fictional fixture: selected blue colour matched"],
      "conflicts": []
    },
    {
      "name": "DEMO ONLY — batik shirt",
      "score": 3,
      "reasons": ["Fictional fixture: selected garment category matched"],
      "conflicts": ["Fictional fixture: selected colour did not match"]
    }
  ]
}
```

Ask, in order:

```text
Explain the first option.
```

```text
Compare the first and second options.
```

Expected: explanations refer to the named supplied items, scores and recorded reasons/conflicts. They must not reinterpret 7 rule points as 70% accuracy or invent missing prices. A record saying “office matched” is not expert cultural approval.

### 7.3 Copy-and-paste Component 3 software fixture

Start a new chat. Select **Component 3 — designs**, then paste this **fictional test fixture**:

```json
{
  "property_basis": "typical_profile",
  "designs": [
    {
      "name": "DEMO ONLY — structured shirt",
      "rule_score": 7,
      "reasons": ["Fictional fixture: illustrative structured drapability matched"],
      "conflicts": []
    },
    {
      "name": "DEMO ONLY — relaxed kurta",
      "rule_score": 3,
      "reasons": ["Fictional fixture: selected style matched"],
      "conflicts": ["Fictional fixture: illustrative drapability mismatch"]
    }
  ]
}
```

Ask, in the same chat:

1. `Why the second option?` → should address the relaxed kurta.
2. `What about the other one?` → with exactly two supplied options and the previous selection, should address the structured shirt.
3. `Explain option 6.` → should ask for a valid option; do not invent item 6.

The explanation should retain that `typical_profile` is an illustrative assumption, not a measured property of an uploaded fabric. Rule scores are not probabilities; provisional ease is not a cutting pattern or fit guarantee.

### 7.4 Accepted format and retained scope

| Field | Component 2 | Component 3 |
|---|---|---|
| Root list | `recommendations` | `designs` |
| Each item | Nonempty `name`, numeric `score` | Nonempty `name`, numeric `rule_score` |
| Explanation lists | Optional string arrays `reasons`, `conflicts` | Same |
| Property basis | Internal recorded-tag basis | Optional `property_basis`: `unspecified`, `typical_profile`, `user_record` (adapter also accepts `recorded_garment_tags`) |

The adapter accepts 1–20 supplied records but retains only the first **five** item summaries and up to five reasons/conflicts per item. Ordinary model grounding includes at most three items unless specific retained options are selected. Original retained option numbers are preserved. Do not request option 6 merely because the pasted source had six records; it is outside retained scope.

Input must be valid JSON, not Python dictionary syntax, Markdown fences, `NaN`, booleans used as scores, or a copied screenshot. Names and explanation lengths are bounded. Large bodies can exceed the server's 50,000-byte request limit, including the rest of the request.

For ordinary follow-ups leave the JSON populated. **To remove supplied context, empty the JSON box and send the next message**, or start New chat. This does not delete or change the real upstream result. Unsent edits have no effect until a message is sent.

## 8. Sample test cases and acceptance checks

### 8.1 How to execute these tests

These are **manual test specifications, not a claim that every case has passed**. Exact LLM wording can vary. Evaluate factual support, state changes, selected item and failure behaviour—not identical phrasing. Record defects rather than treating an expected behaviour as a guaranteed outcome.

Use a fresh chat for each test unless it explicitly says “same chat.” Keep component JSON empty unless required. For fast, deterministic mechanics checks choose baseline + BM25; repeat selected natural-answer tests with Ollama. Do not create dozens of unnecessary history records just to demonstrate every prompt.

### 8.2 Startup, settings and generation

| ID | Steps / sample input | Acceptance check |
|---|---|---|
| TC01 | Start server and open the page | Interface loads, initialized chat can send, no duplicate-server error. |
| TC02 | Baseline + BM25; send `Hello` | Fast labelled baseline greeting, not claimed to be generated by Qwen3. |
| TC03 | Ollama + qwen3:4b; ask `Is satin always silk?` | Real generated response completes; grounded distinction between weave and fibre, not a composition test. Waiting time alone is not failure. |
| TC04 | During a sufficiently long generated reply, click Stop response | Stop is acknowledged, incomplete content is marked, and partial reply is not treated as a saved completed turn. A very fast completed answer may make this case not applicable; retry only with a reasonable longer prompt. |
| TC05 | Inspect Installed model / retrieval dropdowns | Embedding-only all-minilm is not offered as the answer generator; semantic/hybrid availability reflects its installation. |
| TC06 | Send an empty message; inspect 2,000-character textarea limit | Empty input is not sent; oversized messages are prevented/rejected, not silently processed. |

Suggested longer prompt for TC04:

```text
Explain business casual, business formal and black tie optional in several short paragraphs. Distinguish the source examples from a Sri Lankan employer's actual policy, and say what I should ask the host or employer.
```

### 8.3 NLP state and conversation continuity

| ID | Input / steps | Acceptance check |
|---|---|---|
| TC07 | `Suggest something traditional for an outdoor wedding under LKR 15,000` | Intent recommend; wedding/outdoor/traditional; budget 15000. No fabricated product availability. |
| TC08 | Same chat: `Actually office instead, not batik, handloom for hot weather` | Occasion office, weather hot, handloom included, batik excluded; prior budget retained. |
| TC09 | Same chat: `Remove the budget` | Budget removed from state. |
| TC10 | Fresh chat: `Compare Kandyan saree and osari; not batik but handloom` | Aliases resolve to osariya; batik excluded; handloom included. This does not prove complete synonym/negation coverage. |
| TC11 | Fresh chat: ask black tie question, then `Can you explain that more simply?` | Follow-up retains black-tie topic and relevant evidence; inspect method details. |
| TC12 | After TC11: `Tell me more about batik` | Explicit new fabric topic is not treated simply as a black-tie follow-up. Old constraints may still remain; New chat clears them. |
| TC13 | New chat after several preference updates | Previous chat's budget, terms and supplied component context do not leak into the new chat. |

### 8.4 Source grounding and difficult questions

| ID | Fresh-chat prompt | Acceptance check |
|---|---|---|
| TC14 | `What should I consider when visiting the Temple of the Tooth?` | Relevant site-specific evidence, not a rule for every Sri Lankan ceremony. |
| TC15 | `Which fibres are used in Sri Lankan handloom textiles?` | Documented cotton/silk industry examples; no guarantee of an individual sample's composition. |
| TC16 | `Is batik always better than handloom for a Sri Lankan wedding?` | No universal superiority claim; clarify ceremony/host/garment context. |
| TC17 | `Does black tie optional allow a dark suit instead of a tuxedo?` | Distinguish optional from stricter black tie and preserve the etiquette source's context. |
| TC18 | `What gives velvet its raised pile texture?` | Textile-construction evidence; no invented GSM or drape measurement. |
| TC19 | `Can you determine a safe washing temperature from the fabric category alone?` | Refer to item-specific care instructions; do not invent a guaranteed temperature. |
| TC20 | Baseline + BM25, fresh chat: `zxqv987` | No matching evidence and a missing-evidence response. Use exactly this string; different strings/retained constraints can produce lexical matches. |
| TC21 | `Which exact colours are forbidden at my family's private ceremony tomorrow?` | Ask for actual family/host requirements; do not manufacture private or current rules. A retrieved card is not proof it answers the question. |
| TC22 | Repeat `Is satin always silk?` in separate fresh chats using BM25, semantic and hybrid | Correct method is recorded each time; record top source IDs and support quality. Similar results are acceptable; do not assume hybrid must win. |

Citation omission, unsupported claims, overly broad cultural advice or wrong source scope should be recorded as answer-quality defects even if the request technically completed.

### 8.5 Component-result explanations

| ID | Steps | Acceptance check |
|---|---|---|
| TC23 | Paste Section 7.2 Component 2 fixture; ask `Explain the first option.` | Correct first item, recorded reasons and 7 rule points; no invented stock/price/upstream execution. |
| TC24 | Same fixture: `Compare the first and second options.` | Both supplied records, reasons and conflicts addressed; no score-as-probability interpretation. |
| TC25 | Paste Section 7.3 Component 3 fixture; ask `Why the second option?`, then `What about the other one?` | Kurta then shirt; retain typical-profile assumption and original item numbering. |
| TC26 | Same fixture: `Explain option 6.` | Focused unavailable-option clarification, not invented item. This clarification can intentionally be immediate and skip the LLM/retrieval. |
| TC27 | Fresh chat, empty JSON: `Explain the second option.` | Request actual supplied results; do not infer an option from nothing. |
| TC28 | Paste `{` as Response JSON, then send a question | Clear parse/error feedback; no completed answer, no upstream mutation. Correct the JSON before retrying. |
| TC29 | Clear valid supplied JSON, then send a normal knowledge question | Supplied context is removed on that request; inspect status for knowledge-only use. |

### 8.6 History and recovery

| ID | Steps | Acceptance check |
|---|---|---|
| TC30 | Complete a harmless chat, rename it `Manual test — temple`, switch away and reopen | Name, completed turns and retained state restore. |
| TC31 | Refresh browser, then restart only this app's server and reopen the chat | Completed saved chat survives; unsent drafts need not survive. |
| TC32 | Delete disposable chat: first cancel, then confirm | Cancel changes nothing; confirm moves it to Recently deleted, not permanent erasure. |
| TC33 | Restore that deleted chat and click its title | Original completed transcript returns. |
| TC34 | Type an unsent draft, then switch/new chat | If prompted, cancel keeps the draft/current chat. No claim that drafts are crash-persistent. |

### 8.7 Availability failure checks — optional, controlled environment only

Do not stop a shared Ollama server, delete models or interfere with another user's work just to test a failure. Run these only on a disposable/approved local setup.

| ID | Steps | Acceptance check |
|---|---|---|
| TC35 | With Ollama unavailable, refresh model readiness | Baseline/BM25 remain available; unavailable neural options are explained. |
| TC36 | If a selected local generator or embedding service fails during a request | Visible failure; no hidden baseline substitution and no completed partial-turn commit. |

For cases requiring real generated text, an installed but incompatible model is a setup failure, not a pass. For partial/cancelled requests, reopen history before retrying if completion status is uncertain.

## 9. Suggested panel demonstration

Prepare this before the presentation; do not start a large download, model training or destructive failure test in front of the panel.

### Before the panel

1. Confirm server and Ollama are running.
2. Confirm qwen3:4b and all-minilm:22m are installed.
3. Refresh the page and inspect selected engine/retrieval.
4. Use clearly named demonstration chats and remove unrelated personal information from view.
5. Open the saved research report and this manual separately.
6. Explain that the live chat parser and standalone trained classifier are different stages with different current integration status.

### Demonstration sequence (allow for variable local-model latency)

1. **Grounded global explanation:** ask `Does black tie optional allow a dark suit?`; show the evidence card and source scope.
2. **Follow-up:** ask `Can you explain that more simply?`; show retained topic.
3. **Sri Lankan context:** in a new chat ask `Which fibres are used in Sri Lankan handloom?`; distinguish industry examples from specimen measurement.
4. **NLP preferences:** run TC07/TC08 and point out extracted/updated constraints.
5. **Upstream explanation:** use actual permitted C2/C3 output, or explicitly announce the Section 7 fixture as fictional. Show second-option explanation.
6. **Uncertainty:** ask the private-family ceremony question; discuss why clarification is necessary.
7. **History:** rename and reopen a harmless demo chat.
8. **Research evidence:** show the trained-model comparison separately, including protocol, held-out size, per-class errors and limitations.

Do not describe a prewritten fixture as a real recommendation, software tests as human evaluation, or the trained native-act score as overall chatbot accuracy.

## 10. Research models, datasets and results

### 10.1 Models and data

| Asset | Current role |
|---|---|
| Qwen3 4B through Ollama | Pretrained answer generator; not fine-tuned by this project |
| all-minilm:22m through Ollama | Pretrained local embedding model; not fine-tuned here |
| BM25 | Keyword retrieval baseline |
| TF-IDF + logistic regression | Trained native SIMMC dialogue-act baseline |
| DeBERTa-v3-small | Fine-tuned native SIMMC dialogue-act model; standalone experiment |
| 23 sourced knowledge passages | Seven Sri Lankan, fourteen global/contextual, two project implementation passages; cultural expert review pending |
| SIMMC 2.1 fashion subset | 6,402 conversations / 75,220 utterances, original provider partitions preserved |
| Volunteer workbook intake | 630 rows from 30 participant codes, 229 distinct message texts; app-intent/entity review/training postponed |
| Author-written retrieval fixtures | Original 13 questions plus 16 expansion questions; development checks, not independent test evidence |

SIMMC's annotations support a native-act task; they do not directly supply the app's intent mapping or Sri Lankan etiquette truths. Some catalogue attributes and conversation flows are synthetic/simulator-based. Model, dataset and publisher-content licences are separate obligations; see linked research documentation.

### 10.2 Completed trained-model comparison

Assessment: **6,549 user turns / 1,172 conversations**, using current text plus a bounded previous exchange. Train/dev selection and assessment were separated. No current assistant answer or current annotation values were supplied as input.

| Method | Assessment accuracy | Macro-F1 |
|---|---:|---:|
| Majority class | 33.36% | 0.0715 |
| TF-IDF + logistic regression | 87.46% | 0.8598 |
| Fine-tuned DeBERTa-v3-small | 97.53% | 0.9724 |

This is one-seed, text-only **native-act classification**, not the full official multimodal benchmark or a cultural-answer accuracy score. The model was selected using validation only; epoch 2 won. See [the complete experiment report](docs/SIMMC_NLP_EXPERIMENT.md) for confidence intervals, confusion matrices, repeated-input diagnostics, source hashes and exact settings.

The web UI still uses the rule parser. Integrating a trained model requires a reviewed task mapping, compatibility checks and a separate implementation step. Training a good public-benchmark classifier does not automatically complete the conversational cultural-reasoning research.

## 11. Researcher commands

Ordinary users can skip this section. Commands run from the Component 4 directory. Use a **fresh output directory** for scripts that refuse overwrites. Existing recorded results should not be silently replaced.

### 11.1 Software regression checks

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -q
```

At the current checkpoint **85 tests pass** on the configured machine. Some optional checks can skip on an environment missing Node/training dependencies; record skips rather than claiming the same test coverage everywhere. These are software checks, not independent answer-quality ratings.

If Node is available:

```powershell
node --check frontend\app.js
```

### 11.2 Inspect readiness without creating a chat

```powershell
Invoke-RestMethod http://localhost:8003/api/health
Invoke-RestMethod http://localhost:8003/api/models
Invoke-RestMethod http://localhost:8003/api/retrieval
```

The health endpoint's default/compatibility flags are not the method used for every request. Inspect per-answer methods and dropdown readiness to determine actual generation/retrieval use.

### 11.3 Run development retrieval comparisons

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_development.py --suite data\evaluation\knowledge_expansion.json --method all --output runtime\evaluations\manual_retrieval_01
```

`--method all` needs the local embedding model. For BM25-only checks use `--method bm25`. Outputs include metric summaries and blank human-review fields. Do not fill those fields with invented ratings or interpret author-written development results as independent research accuracy.

### 11.4 Inspect trained results without retraining

```powershell
Get-Content runtime\training\simmc_acts_03\report.json
Get-Content runtime\training\simmc_acts_03\comparison_uncertainty.json
```

The complete trained run is `simmc_acts_03`. Runs 01/02 are preserved setup interruptions, not completed neural comparison results. Model checkpoints are not downloadable chat models and are not selectable in Ollama's UI dropdown.

### 11.5 Reproduce training only when deliberately required

Training is resource-intensive and can contend with Ollama for the GPU. Do not launch it as part of ordinary chat testing. Use [SIMMC_NLP_EXPERIMENT.md](docs/SIMMC_NLP_EXPERIMENT.md) for the full protocol and environment requirements.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-training.txt
.\.venv\Scripts\python.exe scripts\train_simmc_acts.py --torch-site '..\component-3-fabric-design\.venv\Lib\site-packages' --output runtime\training\simmc_acts_manual_REPRO
```

This machine reuses Component 3's installed CUDA/PyTorch packages **read-only**; no packages are installed into Component 3 by that command. A portable independent environment needs a compatible CUDA PyTorch installation in Component 4, in which case omit `--torch-site`. SentencePiece uses a small temporary ASCII-path vocabulary workaround for the workspace's Unicode path.

Do not tune repeatedly on devtest and keep calling it an untouched final assessment. Volunteer annotation tools are a separate postponed workflow; see [ANNOTATION_REVIEW.md](docs/ANNOTATION_REVIEW.md) if that work is resumed.

## 12. Troubleshooting

| Problem | Likely cause / safe action |
|---|---|
| Browser cannot reach localhost:8003 | Confirm the correct terminal is running `backend\app.py`; check the exact port and terminal error. |
| WinError 10048 / address already in use | An existing server may already be running. Open the page first; inspect the port owner before stopping anything. |
| PowerShell refuses Activate.ps1 | Use the explicit `.\.venv\Scripts\python.exe` commands; activation is not required. Do not weaken system policy unnecessarily. |
| Ollama option disabled / model missing | Start installed Ollama, run `ollama list`, then Refresh models. Download only a genuinely missing model. |
| Semantic/hybrid unavailable | Check all-minilm:22m is installed and Ollama reachable; use BM25 meanwhile. |
| Reply appears instantly | Verify Reply engine. Baseline templates and deterministic missing-option clarifications are intentionally fast. |
| Generated answer starts slowly | Genuine local-model preparation/loading can take time, especially on first use. Inspect terminal/model readiness and avoid concurrent GPU training. No fixed response time is guaranteed. |
| Another local generation is running | Wait for the existing generation/cleanup; current implementation permits one local generation at a time. |
| Stopped reply / stale revision error | Inspect the reopened saved chat to see whether a final turn committed. Start a new chat when uncertain rather than blindly replaying. |
| JSON parse error | Remove Markdown fences; use double-quoted JSON; check braces and the matching component dropdown. Try a Section 7 fixture. |
| Each result needs numeric rule score | C2 uses `score`; C3 uses `rule_score`. Strings, booleans, NaN and missing values are invalid scores. |
| Explain option 6 gives clarification | Only five supplied item summaries are retained; unavailable options are not invented. |
| Wrong occasion/budget carries over | Constraints persist within a conversation. Correct explicitly or start New chat. |
| Source card exists but answer is unsupported | Retrieval overlap is not answer support. Record an answer-quality defect; inspect scope and original source. |
| Citation-ID warning | Missing/unknown inline IDs need review; the checker does not prove or disprove factual entailment. |
| Long semantic query fails | Embedding context is limited; shorten the message/retained context or choose BM25. |
| Old UI after an update | Refresh with Ctrl+F5; restart only this component's server if backend code changed. |
| No persistent history | Start through normal `backend\app.py`; an imported/test service can deliberately use memory-only sessions. Check runtime directory write access. |
| Active-chat or turn limit reached | Move an unneeded active chat to Recently deleted to free an active slot, or start a new conversation for the turn limit. Deletion remains recoverable. |
| DeBERTa missing from chat-model list | Expected: it is a standalone trained encoder, not the live Ollama generator. |

Read-only port inspection:

```powershell
Get-NetTCPConnection -LocalPort 8003 -State Listen | Select-Object LocalAddress,LocalPort,OwningProcess
```

Use the returned exact PID with `Get-Process -Id <PID>` to identify the owner; replace `<PID>` with the number and do not type the angle brackets. Stop an owned application through its original terminal when possible. Do not copy a blanket process-kill command or kill every Python process.

If the error persists, record the prompt, engine, model, retrieval method, source IDs, whether JSON was supplied, visible error and terminal traceback. Avoid including private chat/database contents unnecessarily.

## 13. Privacy, backup and safe operation

### 13.1 Local storage

- Completed chat history: `runtime/chat_history.sqlite3`.
- Raw and prepared SIMMC: `runtime/datasets/`.
- Trained native-act artifacts: `runtime/training/simmc_acts_03/`.
- Curated source corpus: `data/knowledge.json`.
- Local pretrained encoder/cache files: `runtime/models/`.
- Ollama models normally live in the user's Ollama storage, outside Component 4.

The database is **local and unencrypted**, not cloud-synced or account-authenticated. It can include messages, completed answers, evidence and filtered supplied-component reasons. Do not paste unnecessary names, contact details, body measurements or private records.

Ordinary completed chats are not automatically used for training, annotation or research evaluation. Research reuse requires a separate consent/anonymization protocol. The user's volunteer permission does not automatically cover unrelated chats or publisher content.

### 13.2 Backup

To make a reliable ordinary file backup, first stop this app's server through Ctrl+C, then use File Explorer to **copy, not move**, the Component 4 folder or the required data/database/artifact folders to a secure backup location. Preserve any SQLite companion files present. Restart afterward. Do not copy only the database from a busy live SQLite session and assume it is a consistent backup.

The runtime directory is excluded from Git by project configuration. Having source code is not a backup of downloaded datasets, checkpoints or chat history. Preserve licences, provenance manifests, exact reports and model hashes with research backups.

### 13.3 Safe usage boundaries

- Keep the service on localhost; do not expose it to a public network without a separate security design.
- Treat external publisher links as external browser navigation, not local-only activity.
- Standard configured model calls use local Ollama. Arbitrary custom models that forward to cloud services are not certified by this application.
- Do not load untrusted pickle/joblib files. The saved baseline is for trusted local use.
- Do not claim Delete securely erased data or that a citation checker certified cultural correctness.
- Do not delete or reorganize frozen dataset files/manifests as part of manual testing.

## 14. Record and report test results

Use a separate test log. Suggested statuses: **Not run, Pass, Fail, Blocked, Not applicable**. This manual does not pre-fill results for tests you have not performed.

| Field | What to record |
|---|---|
| Test ID / date / tester | For example TC17, date, approved tester code |
| Environment | Browser, server revision/checksum if available, machine |
| Engine / model | Baseline or Ollama; actual model name |
| Retrieval / corpus | BM25/dense/hybrid; knowledge SHA-256 |
| Setup | New chat or retained conversation; component fixture/actual output reference |
| Input and expected behaviour | Exact prompt and relevant acceptance check |
| Actual output | Short permitted excerpt, screenshot or error; avoid private data |
| Outcome | Not run / Pass / Fail / Blocked / Not applicable |
| Evidence review | Which claims are supported, unsupported or require clarification |
| Notes | Defect reference, follow-up action, reviewer disagreement |

For human answer-quality work, independently rate relevance, factual consistency, explanation quality and cultural scope using an agreed rubric. Separate global, Sri Lankan and mixed-context questions. A small successful demonstration is not a research accuracy estimate, and manual fixtures are not an independent final test.

## 15 Short glossary

- **NLP:** methods for processing language, including rules, retrieval, embeddings and trained classifiers.
- **LLM:** language model used here to generate natural responses.
- **RAG:** retrieve relevant source passages and supply them to generation; retrieval does not guarantee truth.
- **Intent / dialogue act:** a task label for a message; label definitions depend on the dataset/application.
- **Slot / constraint:** a value such as occasion, budget or colour.
- **Embedding:** numerical text representation used for similarity search.
- **Macro-F1:** mean class-level F1, giving each defined class equal weight.
- **Checkpoint:** saved model parameters/configuration, not a live service connection.
- **Scope:** what a source actually supports and where generalization is not justified.
- **Typical profile:** illustrative category-level assumptions, not measurements of an individual fabric.