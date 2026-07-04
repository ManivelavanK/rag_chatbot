# auth.py — Authentication using JSON file store

import bcrypt
from database import _load, _save

class _User:
    def __init__(self, id, username, password_hash):
        self.id = id
        self.username = username
        self.password_hash = password_hash

def _hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

def _verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode(), hashed.encode())

def register_user(session, username: str, password: str) -> _User | None:
    store = _load()
    if username in store["users"]:
        return None
    uid = str(len(store["users"]) + 1)
    store["users"][username] = {"id": uid, "password_hash": _hash_password(password)}
    _save(store)
    return _User(uid, username, store["users"][username]["password_hash"])

def authenticate_user(session, username: str, password: str) -> _User | None:
    store = _load()
    user = store["users"].get(username)
    if user and _verify_password(password, user["password_hash"]):
        return _User(user["id"], username, user["password_hash"])
    return None
