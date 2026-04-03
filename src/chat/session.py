from typing import List, Dict, Any
from src.config import get_config


class ChatSession:
    """Manage in-memory conversation history"""

    def __init__(self, max_history: int = 10):
        self.max_history = max_history
        self.history: List[Dict[str, str]] = []

    @classmethod
    def from_config(cls) -> "ChatSession":
        """Create session from configuration"""
        config = get_config()
        return cls(max_history=config.conversation.max_history)

    def add_user_message(self, content: str) -> None:
        """Add a user message to history"""
        self._add_message("user", content)

    def add_ai_message(self, content: str) -> None:
        """Add an AI message to history"""
        self._add_message("ai", content)

    def _add_message(self, role: str, content: str) -> None:
        """Add message and trim history if needed"""
        self.history.append({"role": role, "content": content})

        # Keep only last max_history messages
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

    def get_history_string(self) -> str:
        """Get formatted conversation history"""
        lines = []
        for msg in self.history:
            role = "Human" if msg["role"] == "user" else "Assistant"
            lines.append(f"{role}: {msg['content']}")
        return "\n".join(lines)

    def get_messages(self) -> List[Dict[str, str]]:
        """Get raw messages list"""
        return self.history.copy()

    def clear(self) -> None:
        """Clear conversation history"""
        self.history.clear()
