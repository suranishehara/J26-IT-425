"""Local SQLite chat history. No upstream writes or automatic research-data reuse."""
from datetime import datetime, timezone
from contextlib import contextmanager
import json
from pathlib import Path
import re
import sqlite3

ID = re.compile(r'^[a-f0-9]{32}$')


class HistoryStore:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS chats (
                    id TEXT PRIMARY KEY, title TEXT NOT NULL, manual_title INTEGER NOT NULL DEFAULT 0,
                    created TEXT NOT NULL, updated TEXT NOT NULL, revision INTEGER NOT NULL,
                    snapshot TEXT NOT NULL, deleted INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS turns (
                    chat_id TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
                    revision INTEGER NOT NULL, payload TEXT NOT NULL,
                    PRIMARY KEY(chat_id, revision)
                );
            ''')

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.path, timeout=5)
        db.row_factory = sqlite3.Row
        db.execute('PRAGMA foreign_keys=ON')
        try:
            with db:
                yield db
        finally:
            db.close()

    @staticmethod
    def identifier(value):
        if not isinstance(value, str) or not ID.fullmatch(value):
            raise ValueError('Invalid chat ID.')
        return value

    @staticmethod
    def now():
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def serialize(value):
        text = json.dumps(value, ensure_ascii=False, allow_nan=False)
        if len(text.encode('utf-8')) > 1_000_000:
            raise ValueError('Saved chat record exceeds its limit.')
        return text

    def create(self, identifier, snapshot):
        self.identifier(identifier)
        now = self.now()
        with self.connection() as db:
            if db.execute('SELECT COUNT(*) FROM chats WHERE deleted=0').fetchone()[0] >= 200:
                raise ValueError('Saved chat limit reached (200). Delete an old chat before creating another.')
            db.execute('INSERT INTO chats(id,title,created,updated,revision,snapshot) VALUES(?,?,?,?,?,?)',
                       (identifier, 'New conversation', now, now, 0, self.serialize(snapshot)))

    def list(self):
        with self.connection() as db:
            rows = db.execute('SELECT id,title,created,updated,revision,deleted FROM chats ORDER BY updated DESC,id').fetchall()
        return {'enabled': True, 'chats': [dict(row) for row in rows if not row['deleted']],
                'deleted': [dict(row) for row in rows if row['deleted']]}

    def get(self, identifier):
        self.identifier(identifier)
        with self.connection() as db:
            row = db.execute('SELECT * FROM chats WHERE id=? AND deleted=0', (identifier,)).fetchone()
            if row is None: raise ValueError('Saved chat is missing or deleted.')
            turns = db.execute('SELECT payload FROM turns WHERE chat_id=? ORDER BY revision', (identifier,)).fetchall()
        return {'id': identifier, 'title': row['title'], 'snapshot': json.loads(row['snapshot']),
                'turns': [json.loads(turn['payload']) for turn in turns]}

    def append(self, identifier, expected_revision, snapshot, turn):
        self.identifier(identifier)
        encoded = self.serialize(snapshot)
        payload = self.serialize(turn)
        revision = expected_revision + 1
        if snapshot['revision'] != revision or turn['result']['revision'] != revision:
            raise ValueError('Saved turn revision mismatch.')
        title = ' '.join(turn['user'].split())[:70] or 'New conversation'
        with self.connection() as db:
            changed = db.execute('''UPDATE chats SET revision=?,snapshot=?,updated=?,
                title=CASE WHEN revision=0 AND manual_title=0 THEN ? ELSE title END
                WHERE id=? AND revision=? AND deleted=0''',
                (revision, encoded, self.now(), title, identifier, expected_revision)).rowcount
            if changed != 1: raise ValueError('Saved chat changed or was deleted. Reopen it before retrying.')
            db.execute('INSERT INTO turns(chat_id,revision,payload) VALUES(?,?,?)', (identifier, revision, payload))

    def rename(self, identifier, title):
        self.identifier(identifier)
        if not isinstance(title, str) or not 1 <= len(title.strip()) <= 80 or any(ord(char) < 32 for char in title):
            raise ValueError('Enter a chat title of 1–80 characters without control characters.')
        with self.connection() as db:
            if db.execute('UPDATE chats SET title=?,manual_title=1,updated=? WHERE id=? AND deleted=0',
                          (title.strip(), self.now(), identifier)).rowcount != 1:
                raise ValueError('Saved chat is missing or deleted.')

    def set_deleted(self, identifier, deleted):
        self.identifier(identifier)
        with self.connection() as db:
            if not deleted and db.execute('SELECT COUNT(*) FROM chats WHERE deleted=0').fetchone()[0] >= 200:
                raise ValueError('Saved chat limit reached. Delete another active chat before restoring this one.')
            if db.execute('UPDATE chats SET deleted=?,updated=? WHERE id=? AND deleted=?',
                          (int(deleted), self.now(), identifier, int(not deleted))).rowcount != 1:
                raise ValueError('Saved chat was not found in the expected state.')

