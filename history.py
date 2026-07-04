# history.py — Chat history using JSON file store

from datetime import datetime
from database import _load, _save

class _Conv:
    def __init__(self, id, user_id, title, updated_at):
        self.id = id
        self.user_id = user_id
        self.title = title
        self.updated_at = datetime.fromisoformat(updated_at) if isinstance(updated_at, str) else updated_at

def get_user_conversations(session, user_id: str) -> list[_Conv]:
    store = _load()
    convs = [
        _Conv(cid, c["user_id"], c["title"], c["updated_at"])
        for cid, c in store["conversations"].items()
        if c["user_id"] == str(user_id)
    ]
    return sorted(convs, key=lambda c: c.updated_at, reverse=True)

def create_conversation(session, user_id: str, title: str = "New Chat") -> _Conv:
    store = _load()
    cid = str(len(store["conversations"]) + 1)
    now = datetime.now().isoformat()
    store["conversations"][cid] = {"user_id": str(user_id), "title": title, "updated_at": now}
    store["messages"][cid] = []
    _save(store)
    return _Conv(cid, user_id, title, now)

def update_conversation_title(session, conversation_id: str, title: str):
    store = _load()
    if str(conversation_id) in store["conversations"]:
        store["conversations"][str(conversation_id)]["title"] = title
        _save(store)

def load_conversation_messages(session, conversation_id: str) -> list[dict]:
    store = _load()
    msgs = store["messages"].get(str(conversation_id), [])
    result = []
    for m in msgs:
        d = {"role": m["role"], "content": m["content"], "timestamp": m.get("timestamp", "")}
        if m.get("chunks"):
            d["chunks"] = m["chunks"]
        if m.get("diag"):
            d["diag"] = m["diag"]
        result.append(d)
    return result

def save_message(session, conversation_id: str, role: str, content: str,
                 retrieved_sources=None, confidence_score=None, diagnostics=None):
    store = _load()
    cid = str(conversation_id)
    if cid not in store["messages"]:
        store["messages"][cid] = []
    now = datetime.now().strftime("%I:%M %p")
    entry = {"role": role, "content": content, "timestamp": now}
    if retrieved_sources:
        entry["chunks"] = retrieved_sources
    if diagnostics:
        entry["diag"] = diagnostics
    store["messages"][cid].append(entry)
    if cid in store["conversations"]:
        store["conversations"][cid]["updated_at"] = datetime.now().isoformat()
    _save(store)
