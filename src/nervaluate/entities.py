from dataclasses import dataclass, field
from typing import List, Tuple, Set, Optional


@dataclass
class Entity:
    """
    Represents a named entity with potentially discontinuous spans.
    
    Attributes:
        label: Entity type/label (e.g., 'PERSON', 'GENE', 'PROTEIN')
        spans: List of (start, end) character offset tuples (end is exclusive)
        text: Optional text content of the entity
    """
    label: str
    spans: List[Tuple[int, int]]
    text: Optional[str] = None

    def __post_init__(self):
        """Validate and sort spans."""
        if not self.spans:
            raise ValueError("Entity must have at least one span")
        
        # Sort spans by start position
        self.spans = sorted(self.spans, key=lambda x: x[0])
        
        # Validate spans
        for start, end in self.spans:
            if start >= end:
                raise ValueError(f"Invalid span: start ({start}) must be < end ({end})")
        
        # Check for overlapping spans within the same entity
        for i in range(len(self.spans) - 1):
            if self.spans[i][1] > self.spans[i + 1][0]:
                raise ValueError(f"Overlapping spans within entity: {self.spans[i]} and {self.spans[i + 1]}")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Entity):
            return NotImplemented
        return self.label == other.label and self.spans == other.spans

    def __hash__(self) -> int:
        return hash((self.label, tuple(self.spans)))

    @property
    def is_continuous(self) -> bool:
        """Check if entity consists of a single continuous span."""
        return len(self.spans) == 1

    @property
    def total_chars(self) -> int:
        """Get total number of characters covered by all spans."""
        return sum(end - start for start, end in self.spans)

    def get_all_char_positions(self) -> Set[int]:
        """
        Get set of all character positions covered by this entity.
        
        Returns:
            Set of character positions (integers)
        """
        chars = set()
        for start, end in self.spans:
            chars.update(range(start, end))
        return chars

    def overlaps_with(self, other: 'Entity') -> bool:
        """
        Check if this entity has any character overlap with another entity.
        
        Args:
            other: Another Entity instance
            
        Returns:
            True if entities share at least one character position
        """
        for s1_start, s1_end in self.spans:
            for s2_start, s2_end in other.spans:
                # Check for overlap: ranges overlap if max(starts) < min(ends)
                if max(s1_start, s2_start) < min(s1_end, s2_end):
                    return True
        return False

    def calculate_overlap_chars(self, other: 'Entity') -> int:
        """
        Calculate the number of overlapping characters with another entity.
        
        Args:
            other: Another Entity instance
            
        Returns:
            Number of overlapping character positions
        """
        self_chars = self.get_all_char_positions()
        other_chars = other.get_all_char_positions()
        return len(self_chars.intersection(other_chars))

    def calculate_overlap_percentage(self, other: 'Entity') -> float:
        """
        Calculate overlap percentage based on this entity's character coverage.
        
        Args:
            other: Another Entity instance
            
        Returns:
            Percentage of this entity's characters that overlap with other (0-100)
        """
        if self.total_chars == 0:
            return 0.0
        
        overlap_chars = self.calculate_overlap_chars(other)
        return (overlap_chars / self.total_chars) * 100.0


@dataclass
class EvaluationResult:
    """Represents the evaluation metrics for a single entity type or overall."""

    correct: int = 0
    incorrect: int = 0
    partial: int = 0
    missed: int = 0
    spurious: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    actual: int = 0
    possible: int = 0

    def compute_metrics(self, partial_or_type: bool = False) -> None:
        """Compute precision, recall and F1 score."""
        self.actual = self.correct + self.incorrect + self.partial + self.spurious
        self.possible = self.correct + self.incorrect + self.partial + self.missed

        if partial_or_type:
            precision = (self.correct + 0.5 * self.partial) / self.actual if self.actual > 0 else 0
            recall = (self.correct + 0.5 * self.partial) / self.possible if self.possible > 0 else 0
        else:
            precision = self.correct / self.actual if self.actual > 0 else 0
            recall = self.correct / self.possible if self.possible > 0 else 0

        self.precision = precision
        self.recall = recall
        self.f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0


@dataclass
class EvaluationIndices:
    """Represents the indices of entities in different evaluation categories."""

    correct_indices: List[Tuple[int, int]] = field(default_factory=list)
    incorrect_indices: List[Tuple[int, int]] = field(default_factory=list)
    partial_indices: List[Tuple[int, int]] = field(default_factory=list)
    missed_indices: List[Tuple[int, int]] = field(default_factory=list)
    spurious_indices: List[Tuple[int, int]] = field(default_factory=list)