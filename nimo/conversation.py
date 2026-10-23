"""Persistent local conversation history for Nimo."""
from __future__ import annotations
import json
from .store import now, uid


class ConversationStore:
    def __init__(self, store):
        self.store=store
        with store.db() as db:
            db.executescript("""
            CREATE TABLE IF NOT EXISTS chats(
              id TEXT PRIMARY KEY,title TEXT NOT NULL,created TEXT NOT NULL,modified TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS chat_messages(
              id TEXT PRIMARY KEY,chat_id TEXT NOT NULL,role TEXT NOT NULL,content TEXT NOT NULL,
              metadata TEXT NOT NULL,created TEXT NOT NULL,
              FOREIGN KEY(chat_id) REFERENCES chats(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_chat_messages_chat ON chat_messages(chat_id,created);
            """)

    def create(self,title="New chat"):
        ident=uid()
        with self.store.db() as db:db.execute("INSERT INTO chats VALUES(?,?,?,?)",(ident,title[:120] or "New chat",now(),now()))
        return {"id":ident,"title":title[:120] or "New chat"}

    def list(self,limit=100):
        with self.store.db() as db:rows=db.execute("SELECT * FROM chats ORDER BY modified DESC LIMIT ?",(min(int(limit),200),)).fetchall()
        return [dict(r) for r in rows]

    def rename(self,ident,title):
        title=" ".join(str(title).split())[:120]
        if not title:raise ValueError("Chat title is empty")
        with self.store.db() as db:
            if not db.execute("UPDATE chats SET title=?,modified=? WHERE id=?",(title,now(),ident)).rowcount:raise ValueError("Chat not found")
        return {"id":ident,"title":title}

    def delete(self,ident):
        with self.store.db() as db:
            db.execute("DELETE FROM chat_messages WHERE chat_id=?",(ident,))
            if not db.execute("DELETE FROM chats WHERE id=?",(ident,)).rowcount:raise ValueError("Chat not found")
        return ident

    def messages(self,chat_id,limit=500):
        with self.store.db() as db:
            rows=db.execute("SELECT * FROM chat_messages WHERE chat_id=? ORDER BY rowid LIMIT ?",(chat_id,min(int(limit),2000))).fetchall()
        return [dict(r,metadata=json.loads(r["metadata"])) for r in rows]

    def add(self,chat_id,role,content,metadata=None):
        if role not in {"system","user","assistant","tool"}:raise ValueError("Invalid message role")
        if not isinstance(content,str) or not content:raise ValueError("Message is empty")
        if len(content)>32768:raise ValueError("Message exceeds 32 KiB")
        with self.store.db() as db:
            if not db.execute("SELECT 1 FROM chats WHERE id=?",(chat_id,)).fetchone():raise ValueError("Chat not found")
            ident=uid();db.execute("INSERT INTO chat_messages VALUES(?,?,?,?,?,?)",
                (ident,chat_id,role,content,json.dumps(metadata or {},ensure_ascii=False),now()))
            db.execute("UPDATE chats SET modified=? WHERE id=?",(now(),chat_id))
            if role=="user":
                count=db.execute("SELECT COUNT(*) FROM chat_messages WHERE chat_id=? AND role='user'",(chat_id,)).fetchone()[0]
                if count==1:
                    title=" ".join(content.split())[:60] or "New chat"
                    db.execute("UPDATE chats SET title=? WHERE id=?",(title,chat_id))
        return {"id":ident}

    def edit_message(self,chat_id,message_id,content,truncate_after=True):
        """Edit a user turn and, by default, discard the obsolete branch after it."""
        content=str(content)
        if not content or len(content)>32768:raise ValueError("Message is empty or too large")
        with self.store.db() as db:
            row=db.execute("SELECT rowid,role FROM chat_messages WHERE id=? AND chat_id=?",(message_id,chat_id)).fetchone()
            if not row:raise ValueError("Message not found")
            if row["role"]!="user":raise ValueError("Only user messages can be edited")
            db.execute("UPDATE chat_messages SET content=? WHERE id=? AND chat_id=?",(content,message_id,chat_id))
            if truncate_after:db.execute("DELETE FROM chat_messages WHERE chat_id=? AND rowid>?",(chat_id,row["rowid"]))
            db.execute("UPDATE chats SET modified=? WHERE id=?",(now(),chat_id))
        return {"id":message_id,"content":content,"truncated":bool(truncate_after)}

    def prepare_regenerate(self,chat_id,expected_user_text):
        """Remove only the latest assistant answer and return the unchanged user branch."""
        with self.store.db() as db:
            rows=db.execute("SELECT rowid,id,role,content FROM chat_messages WHERE chat_id=? ORDER BY rowid DESC LIMIT 2",(chat_id,)).fetchall()
            if not rows:raise ValueError("Chat has no message to regenerate")
            latest=rows[0]
            if latest["role"]=="assistant":
                if len(rows)<2 or rows[1]["role"]!="user":raise ValueError("Latest answer has no user turn")
                if str(rows[1]["content"])!=str(expected_user_text):raise ValueError("Regenerate text does not match the latest user turn")
                db.execute("DELETE FROM chat_messages WHERE id=?",(latest["id"],))
                db.execute("UPDATE chats SET modified=? WHERE id=?",(now(),chat_id))
            elif latest["role"]=="user":
                if str(latest["content"])!=str(expected_user_text):raise ValueError("Regenerate text does not match the latest user turn")
            else:raise ValueError("Latest chat state cannot be regenerated")
        return self.messages(chat_id)

    def delete_message(self,chat_id,message_id):
        with self.store.db() as db:
            changed=db.execute("DELETE FROM chat_messages WHERE id=? AND chat_id=?",(message_id,chat_id)).rowcount
            if not changed:raise ValueError("Message not found")
            db.execute("UPDATE chats SET modified=? WHERE id=?",(now(),chat_id))
        return message_id
