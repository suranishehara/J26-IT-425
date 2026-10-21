# Local chat history

Implemented 19 October 2026 at the user's request. A left-hand history panel supports creating, reopening, renaming, deleting and restoring conversations. The narrow rail keeps the conversation central; on intermediate desktop widths settings/NLP stack while the chat keeps its full height. On phones history becomes a compact block above the chat.

## Storage and continuation

The normal app entry point enables `runtime/chat_history.sqlite3` using Python's standard-library SQLite support. The browser does not put chats in localStorage. The store is local to this project/computer; this is not cloud synchronization or an account-based service.

Each completed turn is saved with the user's message, final answer, existing evidence/source metadata, methods, model name and conversation revision. A bounded resume snapshot preserves constraints, filtered Component 2/3 context, recent successful messages, retrieval topic and selected option references. The transcript can display all completed turns; only the existing bounded recent-history window is sent to the LLM. Model names/metadata alone are not a fully reproducible research run with every model digest and generation setting.

On page load the latest saved active chat is reopened. Click another title to switch. A new chat initially has the title “New conversation”; the first completed user message supplies its automatic title. Manual names are 1–80 characters and are preserved across later turns and restarts. Titles are rendered as literal text, never HTML.

Server restart/idle-cache expiry no longer deletes saved transcripts. On resume the state is loaded from the database. A failed or stopped incomplete response is not committed to persistent history. Unsent drafts, hidden reasoning, raw images and unrelated personal-profile fields in supplied JSON are not saved. A filtered component summary may contain the reasons/names supplied by the user; avoid unnecessary personal information in them. Unsent browser drafts prompt before switching/new chats, but are not recoverable after a browser crash or reload.

History begins when this database-backed version is activated. Earlier in-memory-only conversations are not automatically migrated/recovered. Sessions made by tests/importing the modules remain nonpersistent unless an isolated store is explicitly enabled.

## Delete and restore

Delete asks for confirmation and moves the chat to **Recently deleted**. It disappears from active history and cannot be continued until restored. Restore returns the same transcript and state. Deletion is **recoverable, not permanent erasure**: records remain in the local database. There is no permanent-purge UI or automatic expiry in this milestone, and deleted records can grow the database. Do not describe Delete as securely wiping sensitive information.

The app permits 200 active saved chats, including empty named/new chats, and retains the existing 100-turn conversation limit. Delete an old active chat to free an active slot. Reaching the turn limit requires a new conversation; history remains readable. No automatic pruning removes the user's older chats.

## Privacy and research scope

SQLite history is unencrypted application data on disk, subject to the computer's filesystem access/backups. The localhost service is not authenticated multi-user storage. Other users/processes with access to this computer or service may access chats; do not expose the service publicly. No chat is automatically uploaded, annotated, used as training data or included in research evaluation. Research use requires separately approved consent, anonymization and a defined protocol; ordinary history storage does not establish those conditions.

History helps a demonstration, usability work, and manually revisiting multi-turn behavior. It is a supporting feature, not evidence of NLP accuracy, cultural correctness or a novel research method. Annotation/model training remains postponed at the user's request. Existing source corpora and Components 1–3 are unchanged.

## API and safeguards

| Method | Route | Purpose |
|---|---|---|
| GET | `/api/chats` | Active/deleted titles and revision/timestamp summaries |
| GET | `/api/chats/<id>` | Saved transcript, state and filtered context; reopen for continuation |
| POST | `/api/chats/rename` | `{session_id,title}` |
| POST | `/api/chats/delete` | `{session_id}`; recoverable deletion |
| POST | `/api/chats/restore` | `{session_id}` |

Mutations use the existing loopback-origin and bounded-body guards. Chat IDs/titles are validated; SQL parameters are bound. A database revision check prevents stale turn overwrites. Completed turn and resume snapshot save atomically before memory state commits; database failure yields no completed turn. Rename/delete/restore/open are blocked while that chat is generating. Controls are disabled in the current UI during streaming. This is concurrency protection, not authentication.

## Verification

72 automated tests pass. Added checks cover transcript/state/topic persistence across a fresh store/cache, manual/automatic titles, filtered context restoration, delete/restore, stale revision protection, stream/database rollback, title validation, origin guards and actual frontend handlers for open/rename/cancel/delete/restore/busy/draft protection. Database tests use temporary stores; they do not mutate user chats. Browser layout verification is separate from the automated tests. No answer-quality improvement is inferred from history availability.

Browser check completed: the left history rail displayed a new chat, a harmless baseline “Hello” turn was saved with its automatic title, and the completed question/answer reopened after a page reload without creating another blank chat. Qwen3 remained selectable for future replies. The demonstration chat is retained in local history; no existing user chat was deleted during verification.

Run normally:

```powershell
.\.venv\Scripts\python.exe backend\app.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -q
```

If an older server is already running on 8003, restart it once before refreshing the page. This initializes the local database; do not start duplicate servers on the same port.
