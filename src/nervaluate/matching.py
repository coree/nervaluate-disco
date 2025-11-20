"""
Matching algorithms for discontinuous and overlapping entities.

Provides optimal bipartite matching using the Hungarian algorithm for
entity alignment between predictions and ground truth.
"""

from typing import List, Tuple, Set
import numpy as np
from scipy.optimize import linear_sum_assignment

from .entities import Entity


def calculate_overlap_percentage(pred: Entity, true: Entity) -> float:
    """
    Calculate overlap as percentage of true entity's character coverage.
    
    Works for both continuous and discontinuous entities by comparing
    character-level overlap.
    
    Args:
        pred: Predicted entity
        true: True/ground truth entity
        
    Returns:
        Overlap percentage based on true entity span (0-100)
        
    Example:
        >>> true = Entity("GENE", spans=[(0, 5), (20, 25)])  # 10 chars total
        >>> pred = Entity("GENE", spans=[(0, 22)])           # Covers both + gap
        >>> calculate_overlap_percentage(pred, true)
        70.0  # 7 chars overlap out of 10
    """
    pred_chars = pred.get_all_char_positions()
    true_chars = true.get_all_char_positions()
    
    if not true_chars:
        return 0.0
    
    overlap_chars = pred_chars.intersection(true_chars)
    return (len(overlap_chars) / len(true_chars)) * 100.0


def optimal_match(
    true_entities: List[Entity],
    pred_entities: List[Entity],
    threshold: float = 1.0
) -> Tuple[List[Tuple[int, int, float, Entity, Entity]], Set[int], Set[int]]:
    """
    Find optimal bipartite matching between true and predicted entities.
    
    Uses the Hungarian algorithm to find the matching that maximizes total
    overlap while respecting the minimum overlap threshold.
    
    Args:
        true_entities: List of ground truth entities
        pred_entities: List of predicted entities
        threshold: Minimum overlap percentage (1-100) for a valid match
        
    Returns:
        Tuple of:
        - matches: List of (true_idx, pred_idx, overlap, true_entity, pred_entity)
        - matched_true: Set of true entity indices that were matched
        - matched_pred: Set of predicted entity indices that were matched
        
    Example:
        >>> true = [Entity("PER", [(0, 5)])]
        >>> pred = [Entity("PER", [(0, 4)])]
        >>> matches, matched_t, matched_p = optimal_match(true, pred, threshold=50.0)
        >>> len(matches)
        1
    """
    if not true_entities or not pred_entities:
        return [], set(), set()
    
    n_true = len(true_entities)
    n_pred = len(pred_entities)
    
    # Build cost matrix (negate overlap for minimization)
    cost_matrix = np.zeros((n_true, n_pred))
    for i, true in enumerate(true_entities):
        for j, pred in enumerate(pred_entities):
            overlap = calculate_overlap_percentage(pred, true)
            if overlap >= threshold:
                # Use negative overlap since we want to minimize cost
                cost_matrix[i, j] = -overlap
            else:
                # Large cost = no valid match
                cost_matrix[i, j] = 1e9
    
    # Find optimal assignment using Hungarian algorithm
    true_indices, pred_indices = linear_sum_assignment(cost_matrix)
    
    # Extract valid matches (those meeting threshold)
    matches = []
    matched_true = set()
    matched_pred = set()
    
    for t_idx, p_idx in zip(true_indices, pred_indices):
        if cost_matrix[t_idx, p_idx] < 1e9:
            overlap = -cost_matrix[t_idx, p_idx]
            matches.append((
                t_idx,
                p_idx,
                overlap,
                true_entities[t_idx],
                pred_entities[p_idx]
            ))
            matched_true.add(t_idx)
            matched_pred.add(p_idx)
    
    return matches, matched_true, matched_pred


def greedy_match(
    true_entities: List[Entity],
    pred_entities: List[Entity],
    threshold: float = 1.0
) -> Tuple[List[Tuple[int, int, float, Entity, Entity]], Set[int], Set[int]]:
    """
    Find greedy matching between true and predicted entities.
    
    Matches entities in order of decreasing overlap score. Simpler and faster
    than optimal matching, but may not find the globally optimal solution.
    
    Args:
        true_entities: List of ground truth entities
        pred_entities: List of predicted entities
        threshold: Minimum overlap percentage (1-100) for a valid match
        
    Returns:
        Tuple of:
        - matches: List of (true_idx, pred_idx, overlap, true_entity, pred_entity)
        - matched_true: Set of true entity indices that were matched
        - matched_pred: Set of predicted entity indices that were matched
    """
    if not true_entities or not pred_entities:
        return [], set(), set()
    
    # Calculate all overlap scores
    scores = []
    for t_idx, true in enumerate(true_entities):
        for p_idx, pred in enumerate(pred_entities):
            overlap = calculate_overlap_percentage(pred, true)
            if overlap >= threshold:
                scores.append((overlap, t_idx, p_idx, true, pred))
    
    # Sort by overlap (highest first)
    scores.sort(reverse=True)
    
    # Greedy assignment
    matched_true = set()
    matched_pred = set()
    matches = []
    
    for overlap, t_idx, p_idx, true, pred in scores:
        if t_idx not in matched_true and p_idx not in matched_pred:
            matches.append((t_idx, p_idx, overlap, true, pred))
            matched_true.add(t_idx)
            matched_pred.add(p_idx)
    
    return matches, matched_true, matched_pred