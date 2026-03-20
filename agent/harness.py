"""Agent Harness — tool-use loop for complex Abaqus analysis tasks.

The harness gives the LLM agency: it can search docs, check API signatures,
validate code, and submit analyses through tool calls, deciding autonomously
what to do at each step.
"""

import json
import logging
import time

from .tools import TOOL_DEFINITIONS, execute_tool

logger = logging.getLogger(__name__)

MAX_ROUNDS = 10


class AgentHarness:
    """Tool-use agent loop for complex analysis tasks."""

    def __init__(self, llm, rag=None):
        self.llm = llm
        self.rag = rag

    def run(self, user_message, system_prompt):
        """Run the agent loop until the LLM produces a final text response.

        Args:
            user_message: The user's natural language input.
            system_prompt: System prompt for the agent mode.

        Returns:
            The LLM's final text response (should contain <plan> + <code>).
        """
        messages = [{"role": "user", "content": user_message}]

        for round_num in range(1, MAX_ROUNDS + 1):
            logger.info("Agent round %d/%d", round_num, MAX_ROUNDS)

            try:
                t0 = time.time()
                response_msg = self.llm.generate_with_tools(
                    system=system_prompt,
                    messages=messages,
                    tools=TOOL_DEFINITIONS,
                    max_tokens=4096,
                )
                elapsed = time.time() - t0
                logger.info("LLM responded in %.1fs", elapsed)
            except Exception as e:
                logger.error("LLM call failed in agent loop: %s", e)
                return f"LLM 调用失败: {e}"

            # Check if LLM wants to call tools
            if response_msg.tool_calls:
                # Append the assistant message with tool calls
                messages.append(_message_from_response(response_msg))

                # Execute each tool call
                for tc in response_msg.tool_calls:
                    tool_name = tc.function.name
                    tool_args = tc.function.arguments
                    logger.info("Tool call: %s(%s)", tool_name, tool_args[:100])

                    result = execute_tool(tool_name, tool_args, rag=self.rag)

                    # Append tool result
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result[:4000],  # cap tool result size
                    })
                    logger.info("Tool result: %s... (%d chars)",
                                result[:80], len(result))

                # Continue the loop — LLM will see tool results and decide next step
                continue

            # No tool calls — LLM produced a final text response
            final_text = response_msg.content or ""
            if final_text:
                logger.info("Agent finished in %d rounds, response: %d chars",
                            round_num, len(final_text))
                return final_text

            # Edge case: no tool calls and no text
            logger.warning("Agent round %d: empty response", round_num)
            messages.append({"role": "assistant", "content": ""})
            messages.append({
                "role": "user",
                "content": "请继续完成分析。如果需要查询文档，请使用 search_abaqus_docs 工具。",
            })

        # Max rounds exceeded
        logger.error("Agent loop exhausted after %d rounds", MAX_ROUNDS)
        return (
            f"Agent 经过 {MAX_ROUNDS} 轮工具调用仍未完成。"
            f"请尝试简化您的需求描述。"
        )


def _message_from_response(msg):
    """Convert an OpenAI response message to a dict for the messages list."""
    d = {"role": "assistant", "content": msg.content or ""}
    if msg.tool_calls:
        d["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in msg.tool_calls
        ]
    return d
