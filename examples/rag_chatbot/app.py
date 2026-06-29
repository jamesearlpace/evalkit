"""Synthetic RAG chatbot app used by the evalkit example."""

from __future__ import annotations


def answer(input: str, context: str | None = None, prompt: str = "") -> str:
    """Return a small deterministic answer.

    The behavior intentionally improves when the prompt says to ground answers
    in context, which lets the prompt optimizer demonstrate before/after lift
    without calling a live model.
    """

    question = input.lower()
    prompt_text = prompt.lower()
    context_text = (context or "").lower()
    grounded = "context" in prompt_text or "ground" in prompt_text

    if not grounded:
        if "password" in question:
            return "Use the account settings page."
        if "support" in question:
            return "Support is usually available during business hours."
        if "refund" in question:
            return "Refund timing depends on the order."
        return "I need more information."

    if "password" in question and "reset link" in context_text:
        return "Use the account portal reset link."
    if "support" in question and "6 a.m. to 6 p.m. pacific" in context_text:
        return "Support is available from 6 a.m. to 6 p.m. Pacific, Monday through Friday."
    if "refund" in question and "five business days" in context_text:
        return "Refunds are processed within five business days after approval."
    if "delete" in question and "privacy dashboard" in context_text:
        return "Customers can request account deletion from the privacy dashboard."
    return "I do not know based on the provided context."
