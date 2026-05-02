from collections import defaultdict
from typing import Dict, List

from src.models import SessionTurn


_store: Dict[str, List[SessionTurn]] = defaultdict(list)


def get_history(session_id: str) -> List[SessionTurn]:
    return _store[session_id]


def add_turn(session_id: str, turn: SessionTurn) -> None:
    _store[session_id].append(turn)
    _store[session_id] = _store[session_id][-10:]


def clear_session(session_id: str) -> None:
    _store[session_id] = []
