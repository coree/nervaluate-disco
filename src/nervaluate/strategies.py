from abc import ABC, abstractmethod
from typing import List, Tuple, Set
import numpy as np
from scipy.optimize import linear_sum_assignment

from .entities import Entity, EvaluationResult, EvaluationIndices


class EvaluationStrategy(ABC):
    """Abstract base class for evaluation strategies."""

    def __init__(self, min_overlap_percentage: float = 1.0):
        """
        Initialize strategy with minimum overlap threshold.

        Args:
            min_overlap_percentage: Minimum overlap percentage required (1-100)
        """
        if not 1.0 <= min_overlap_percentage <= 100.0:
            raise ValueError("min_overlap_percentage must be between 1.0 and 100.0")
        self.min_overlap_percentage = min_overlap_percentage

    @staticmethod
    def calculate_overlap_percentage(pred: Entity, true: Entity) -> float:
        """
        Calculate the percentage overlap between predicted and true entities.
        Based on true entity's character coverage.

        Args:
            pred: Predicted entity
            true: True entity

        Returns:
            Overlap percentage (0-100)
        """
        return true.calculate_overlap_percentage(pred)

    def has_sufficient_overlap(self, pred: Entity, true: Entity) -> bool:
        """Check if entities have sufficient overlap based on threshold."""
        overlap_percentage = self.calculate_overlap_percentage(pred, true)
        return overlap_percentage >= self.min_overlap_percentage

    def optimal_match(
        self, 
        true_entities: List[Entity], 
        pred_entities: List[Entity]
    ) -> Tuple[List[Tuple[int, int, float]], Set[int], Set[int]]:
        """
        Find optimal matching between true and predicted entities using Hungarian algorithm.
        
        Args:
            true_entities: List of true entities
            pred_entities: List of predicted entities
            
        Returns:
            Tuple of:
                - List of matches: [(true_idx, pred_idx, overlap_percentage), ...]
                - Set of matched true indices
                - Set of matched pred indices
        """
        if not true_entities or not pred_entities:
            return [], set(), set()
        
        n_true = len(true_entities)
        n_pred = len(pred_entities)
        
        # Build cost matrix (we negate overlap for minimization)
        # High cost (1e9) means no match
        cost_matrix = np.full((n_true, n_pred), 1e9)
        
        for i, true in enumerate(true_entities):
            for j, pred in enumerate(pred_entities):
                overlap = self.calculate_overlap_percentage(pred, true)
                if overlap >= self.min_overlap_percentage:
                    # Negate overlap for minimization (Hungarian finds minimum cost)
                    cost_matrix[i, j] = -overlap
        
        # Find optimal assignment
        true_indices, pred_indices = linear_sum_assignment(cost_matrix)
        
        # Extract valid matches (where cost is not 1e9)
        matches = []
        matched_true = set()
        matched_pred = set()
        
        for t_idx, p_idx in zip(true_indices, pred_indices):
            if cost_matrix[t_idx, p_idx] < 1e9:
                overlap = -cost_matrix[t_idx, p_idx]
                matches.append((t_idx, p_idx, overlap))
                matched_true.add(t_idx)
                matched_pred.add(p_idx)
        
        return matches, matched_true, matched_pred

    @abstractmethod
    def evaluate(
        self, true_entities: List[Entity], pred_entities: List[Entity], tags: List[str], instance_index: int = 0
    ) -> Tuple[EvaluationResult, EvaluationIndices]:
        """Evaluate the predicted entities against the true entities."""


class StrictEvaluation(EvaluationStrategy):
    """
    Strict evaluation strategy - entities must match exactly.

    Correct: Same label AND same spans (character-level exact match)
    Incorrect: Sufficient overlap but not exact match
    Partial: N/A (not used in strict)
    Spurious: Predicted entity with no sufficient match
    Missed: True entity with no sufficient match
    """

    def evaluate(
        self, true_entities: List[Entity], pred_entities: List[Entity], tags: List[str], instance_index: int = 0
    ) -> Tuple[EvaluationResult, EvaluationIndices]:
        """
        Evaluate the predicted entities against the true entities using strict matching.
        """
        result = EvaluationResult()
        indices = EvaluationIndices()
        
        # Get optimal matches
        matches, matched_true, matched_pred = self.optimal_match(true_entities, pred_entities)
        
        # Evaluate each match
        for t_idx, p_idx, overlap in matches:
            true_ent = true_entities[t_idx]
            pred_ent = pred_entities[p_idx]
            
            # Check for exact match (same label AND same spans)
            if pred_ent.label == true_ent.label and pred_ent.spans == true_ent.spans:
                result.correct += 1
                indices.correct_indices.append((instance_index, p_idx))
            else:
                # Has overlap but not exact
                result.incorrect += 1
                indices.incorrect_indices.append((instance_index, p_idx))
        
        # Unmatched predictions = spurious
        for p_idx in range(len(pred_entities)):
            if p_idx not in matched_pred:
                result.spurious += 1
                indices.spurious_indices.append((instance_index, p_idx))
        
        # Unmatched true entities = missed
        for t_idx in range(len(true_entities)):
            if t_idx not in matched_true:
                result.missed += 1
                indices.missed_indices.append((instance_index, t_idx))
        
        result.compute_metrics()
        return result, indices


class PartialEvaluation(EvaluationStrategy):
    """
    Partial evaluation strategy - allows for partial matches.

    Correct: Exact match (same spans)
    Partial: Sufficient overlap but not exact
    Incorrect: N/A (not used in partial - label doesn't matter)
    Spurious: Predicted entity with no sufficient match
    Missed: True entity with no sufficient match
    """

    def evaluate(
        self, true_entities: List[Entity], pred_entities: List[Entity], tags: List[str], instance_index: int = 0
    ) -> Tuple[EvaluationResult, EvaluationIndices]:
        result = EvaluationResult()
        indices = EvaluationIndices()
        
        # Get optimal matches
        matches, matched_true, matched_pred = self.optimal_match(true_entities, pred_entities)
        
        # Evaluate each match
        for t_idx, p_idx, overlap in matches:
            true_ent = true_entities[t_idx]
            pred_ent = pred_entities[p_idx]
            
            # Check for exact span match (label doesn't matter)
            if pred_ent.spans == true_ent.spans:
                result.correct += 1
                indices.correct_indices.append((instance_index, p_idx))
            else:
                # Has overlap but not exact spans
                result.partial += 1
                indices.partial_indices.append((instance_index, p_idx))
        
        # Unmatched predictions = spurious
        for p_idx in range(len(pred_entities)):
            if p_idx not in matched_pred:
                result.spurious += 1
                indices.spurious_indices.append((instance_index, p_idx))
        
        # Unmatched true entities = missed
        for t_idx in range(len(true_entities)):
            if t_idx not in matched_true:
                result.missed += 1
                indices.missed_indices.append((instance_index, t_idx))
        
        result.compute_metrics(partial_or_type=True)
        return result, indices


class EntityTypeEvaluation(EvaluationStrategy):
    """
    Entity type evaluation strategy - checks entity types with overlap.

    Correct: Sufficient overlap with same label
    Incorrect: Sufficient overlap with different label
    Partial: N/A (not used in entity type)
    Spurious: Predicted entity with no sufficient match
    Missed: True entity with no sufficient match
    """

    def evaluate(
        self, true_entities: List[Entity], pred_entities: List[Entity], tags: List[str], instance_index: int = 0
    ) -> Tuple[EvaluationResult, EvaluationIndices]:
        result = EvaluationResult()
        indices = EvaluationIndices()
        
        # Get optimal matches
        matches, matched_true, matched_pred = self.optimal_match(true_entities, pred_entities)
        
        # Evaluate each match
        for t_idx, p_idx, overlap in matches:
            true_ent = true_entities[t_idx]
            pred_ent = pred_entities[p_idx]
            
            # Check label match
            if pred_ent.label == true_ent.label:
                result.correct += 1
                indices.correct_indices.append((instance_index, p_idx))
            else:
                result.incorrect += 1
                indices.incorrect_indices.append((instance_index, p_idx))
        
        # Unmatched predictions = spurious
        for p_idx in range(len(pred_entities)):
            if p_idx not in matched_pred:
                result.spurious += 1
                indices.spurious_indices.append((instance_index, p_idx))
        
        # Unmatched true entities = missed
        for t_idx in range(len(true_entities)):
            if t_idx not in matched_true:
                result.missed += 1
                indices.missed_indices.append((instance_index, t_idx))
        
        result.compute_metrics(partial_or_type=True)
        return result, indices


class ExactEvaluation(EvaluationStrategy):
    """
    Exact evaluation strategy - exact span match regardless of type.

    Correct: Exact span match (label doesn't matter)
    Incorrect: Sufficient overlap but not exact spans
    Partial: N/A (not used in exact)
    Spurious: Predicted entity with no sufficient match
    Missed: True entity with no sufficient match
    """

    def evaluate(
        self, true_entities: List[Entity], pred_entities: List[Entity], tags: List[str], instance_index: int = 0
    ) -> Tuple[EvaluationResult, EvaluationIndices]:
        """
        Evaluate the predicted entities against the true entities using exact span matching.
        Entity type is not considered in the matching.
        """
        result = EvaluationResult()
        indices = EvaluationIndices()
        
        # Get optimal matches
        matches, matched_true, matched_pred = self.optimal_match(true_entities, pred_entities)
        
        # Evaluate each match
        for t_idx, p_idx, overlap in matches:
            true_ent = true_entities[t_idx]
            pred_ent = pred_entities[p_idx]
            
            # Check for exact span match (regardless of label)
            if pred_ent.spans == true_ent.spans:
                result.correct += 1
                indices.correct_indices.append((instance_index, p_idx))
            else:
                # Has overlap but not exact spans
                result.incorrect += 1
                indices.incorrect_indices.append((instance_index, p_idx))
        
        # Unmatched predictions = spurious
        for p_idx in range(len(pred_entities)):
            if p_idx not in matched_pred:
                result.spurious += 1
                indices.spurious_indices.append((instance_index, p_idx))
        
        # Unmatched true entities = missed
        for t_idx in range(len(true_entities)):
            if t_idx not in matched_true:
                result.missed += 1
                indices.missed_indices.append((instance_index, t_idx))
        
        result.compute_metrics()
        return result, indices