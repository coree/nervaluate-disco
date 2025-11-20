"""
nervaluate-disco: Evaluation for discontinuous and overlapping NER entities.

A clean-break fork of nervaluate designed specifically for evaluating
named entity recognition systems that produce discontinuous entities
and/or overlapping entities.
"""

from .evaluator import Evaluator
from .entities import Entity, EvaluationResult, EvaluationIndices
from .loaders import SpanDictLoader, SimpleSpanLoader
from .strategies import (
    StrictEvaluation,
    PartialEvaluation,
    EntityTypeEvaluation,
    ExactEvaluation
)
from .matching import calculate_overlap_percentage, optimal_match, greedy_match

__version__ = "2.0.0"

__all__ = [
    "Evaluator",
    "Entity",
    "EvaluationResult",
    "EvaluationIndices",
    "SpanDictLoader",
    "SimpleSpanLoader",
    "StrictEvaluation",
    "PartialEvaluation",
    "EntityTypeEvaluation",
    "ExactEvaluation",
    "calculate_overlap_percentage",
    "optimal_match",
    "greedy_match",
]