"""
LLM client for AI-powered ML interpretation.

Calls the Anthropic API with a structured prompt built from the project state
(dataset shape, steps taken, model metrics, domain context) and streams the
response back to the caller token by token.

The Export and AI Interpretation pages call interpret() and render the stream
with st.write_stream().
"""
from __future__ import annotations

from typing import Iterator


def interpret(project: object, api_key: str, domain_context: str) -> Iterator[str]:
    # TODO:
    # 1. Build a system prompt describing the ML workflow assistant role
    # 2. Build a user message summarising project state (steps, metrics, target, task)
    # 3. Instantiate anthropic.Anthropic(api_key=api_key)
    # 4. Call client.messages.stream(...) with claude-sonnet-4-6
    # 5. yield text deltas from the stream
    raise NotImplementedError("LLM interpretation not yet implemented.")
