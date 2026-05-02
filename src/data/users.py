import json
from pathlib import Path

from src.models import UserProfile


def fixtures_root() -> Path:
    return Path(__file__).resolve().parents[2] / "fixtures" / "users"


def load_user(user_id: str) -> UserProfile:
    for path in fixtures_root().glob("*.json"):
        with open(path, encoding="utf-8") as f:
            candidate = json.load(f)
        if candidate.get("user_id") == user_id:
            return UserProfile(**candidate)
    raise FileNotFoundError(f"User not found: {user_id}")


def list_user_ids() -> list[str]:
    ids: list[str] = []
    for path in fixtures_root().glob("*.json"):
        with open(path, encoding="utf-8") as f:
            ids.append(json.load(f).get("user_id", ""))
    return sorted([x for x in ids if x])
