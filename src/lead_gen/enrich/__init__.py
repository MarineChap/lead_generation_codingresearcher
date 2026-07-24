"""Deterministic enrichment steps that run around the LLM crews.

The crews keep the fuzzy reasoning (relevance, prose); the code in this
package owns the facts: contact extraction, evidence verification, score
enforcement, and reason-for-need synthesis.
"""
