"""Underwriting agent with a human-approval checkpoint (Phase 5).

A LangGraph graph (graph.py) reviews one SYNTHETIC loan application or insurance submission: it screens free text
for injected instructions, calls allow-listed tools in parallel, runs the deterministic reference policy, asks an
LLM for a cited memo, checks that memo, and always stops for a human decision. See docs/agent_design.md.
"""
