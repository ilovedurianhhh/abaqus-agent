"""Kimi (Moonshot AI) API client wrapper — OpenAI-compatible interface."""

import os
import logging

from dotenv import load_dotenv
from openai import OpenAI

# Load .env from project root
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

logger = logging.getLogger(__name__)

# Kimi API endpoint
KIMI_BASE_URL = "https://api.moonshot.cn/v1"
KIMI_DEFAULT_MODEL = os.environ.get("KIMI_MODEL", "kimi-k2.5")


class LLMClient:
    """Thin wrapper around Kimi's OpenAI-compatible API."""

    def __init__(self, api_key=None, model=KIMI_DEFAULT_MODEL):
        self.model = model
        key = api_key or os.environ.get("KIMI_API_KEY")
        if not key:
            raise ValueError(
                "KIMI_API_KEY not found. Set it in .env file or pass api_key argument."
            )
        self._client = OpenAI(api_key=key, base_url=KIMI_BASE_URL)

    def generate(self, system, messages, max_tokens=4096):
        """Send a chat completion request and return the text response.

        Args:
            system: System prompt string.
            messages: List of {"role": ..., "content": ...} dicts.
            max_tokens: Maximum tokens in the response.

        Returns:
            The assistant's text reply as a string.
        """
        # Prepend system message (OpenAI-compatible format)
        full_messages = [{"role": "system", "content": system}] + messages

        logger.info("LLM request: model=%s, messages=%d, max_tokens=%d",
                     self.model, len(full_messages), max_tokens)

        response = self._client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=full_messages,
        )

        reply = response.choices[0].message.content
        usage = response.usage
        if usage:
            logger.info("LLM response: prompt_tokens=%d, completion_tokens=%d, total=%d",
                         usage.prompt_tokens, usage.completion_tokens, usage.total_tokens)
        return reply

    def generate_with_tools(self, system, messages, tools, max_tokens=4096):
        """Chat completion with tool calling support.

        Args:
            system: System prompt string.
            messages: List of message dicts (may include tool results).
            tools: List of tool definitions in OpenAI format.
            max_tokens: Maximum tokens in the response.

        Returns:
            The full response Message object (may contain tool_calls).
        """
        full_messages = [{"role": "system", "content": system}] + messages

        logger.info("LLM tool request: model=%s, messages=%d, tools=%d",
                     self.model, len(full_messages), len(tools))

        kwargs = dict(
            model=self.model,
            max_tokens=max_tokens,
            messages=full_messages,
            tools=tools,
        )

        response = self._client.chat.completions.create(**kwargs)

        msg = response.choices[0].message
        usage = response.usage
        if usage:
            logger.info("LLM tool response: prompt=%d, completion=%d, tool_calls=%s",
                         usage.prompt_tokens, usage.completion_tokens,
                         len(msg.tool_calls) if msg.tool_calls else 0)
        return msg
