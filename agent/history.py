"""Conversation history management with rolling window."""


class ConversationHistory:
    """Maintains a rolling window of conversation turns for the LLM."""

    def __init__(self, max_turns=20):
        self._messages = []
        self._max_turns = max_turns

    def add_user(self, content):
        """Append a user message."""
        self._messages.append({"role": "user", "content": content})
        self._trim()

    def add_assistant(self, content):
        """Append an assistant message."""
        self._messages.append({"role": "assistant", "content": content})
        self._trim()

    def to_messages(self):
        """Return the message list for the Anthropic API."""
        return list(self._messages)

    def clear(self):
        """Reset the conversation history."""
        self._messages.clear()

    def _trim(self):
        """Keep only the last max_turns * 2 messages (user + assistant pairs)."""
        max_msgs = self._max_turns * 2
        if len(self._messages) > max_msgs:
            self._messages = self._messages[-max_msgs:]
