"""Tests for the Entity model and related classes."""

import pytest
import re
from nervaluate.entities import Entity, EvaluationResult, EvaluationIndices


class TestEntity:
    """Tests for the Entity class."""

    def test_continuous_entity(self):
        """Test creation of continuous entity."""
        entity = Entity(label="PERSON", spans=[(10, 15)])
        assert entity.label == "PERSON"
        assert entity.spans == [(10, 15)]
        assert entity.is_continuous is True
        assert entity.total_chars == 5

    def test_discontinuous_entity(self):
        """Test creation of discontinuous entity."""
        entity = Entity(label="PROTEIN", spans=[(10, 15), (45, 52)])
        assert entity.label == "PROTEIN"
        assert entity.spans == [(10, 15), (45, 52)]
        assert entity.is_continuous is False
        assert entity.total_chars == 12  # 5 + 7

    def test_entity_with_text(self):
        """Test entity with text field."""
        entity = Entity(label="GENE", spans=[(0, 5)], text="BRCA1")
        assert entity.text == "BRCA1"

    def test_spans_sorted(self):
        """Test that spans are automatically sorted."""
        entity = Entity(label="LOC", spans=[(50, 55), (10, 15), (30, 35)])
        assert entity.spans == [(10, 15), (30, 35), (50, 55)]

    def test_invalid_span_start_greater_than_end(self):
        """Test that invalid spans raise ValueError."""
        with pytest.raises(ValueError, match=re.escape("Invalid span: start (15) must be < end (10)")):
            Entity(label="PER", spans=[(15, 10)])

    def test_invalid_span_start_equals_end(self):
        """Test that zero-length spans raise ValueError."""
        with pytest.raises(ValueError, match=re.escape("Invalid span: start (10) must be < end (10)")):
            Entity(label="PER", spans=[(10, 10)])

    def test_empty_spans(self):
        """Test that empty spans list raises ValueError."""
        with pytest.raises(ValueError, match="must have at least one span"):
            Entity(label="PER", spans=[])

    def test_overlapping_spans_within_entity(self):
        """Test that overlapping spans within same entity raise ValueError."""
        with pytest.raises(ValueError, match="Overlapping spans within entity"):
            Entity(label="PER", spans=[(10, 20), (15, 25)])

    def test_get_all_char_positions(self):
        """Test getting all character positions."""
        entity = Entity(label="PROTEIN", spans=[(10, 15), (20, 23)])
        chars = entity.get_all_char_positions()
        assert chars == {10, 11, 12, 13, 14, 20, 21, 22}

    def test_overlaps_with_continuous(self):
        """Test overlap detection for continuous entities."""
        e1 = Entity(label="PER", spans=[(10, 20)])
        e2 = Entity(label="ORG", spans=[(15, 25)])
        e3 = Entity(label="LOC", spans=[(24, 30)])

        assert e1.overlaps_with(e2) is True
        assert e2.overlaps_with(e1) is True
        assert e1.overlaps_with(e3) is False
        assert e2.overlaps_with(e3) is True

    def test_overlaps_with_discontinuous(self):
        """Test overlap detection for discontinuous entities."""
        e1 = Entity(label="PROTEIN", spans=[(10, 15), (30, 35)])
        e2 = Entity(label="GENE", spans=[(12, 18)])
        e3 = Entity(label="CELL", spans=[(32, 38)])
        e4 = Entity(label="DISEASE", spans=[(50, 55)])

        assert e1.overlaps_with(e2) is True  # Overlaps with first span
        assert e1.overlaps_with(e3) is True  # Overlaps with second span
        assert e1.overlaps_with(e4) is False  # No overlap

    def test_entity_equality(self):
        """Test entity equality comparison."""
        e1 = Entity(label="PER", spans=[(10, 15)])
        e2 = Entity(label="PER", spans=[(10, 15)])
        e3 = Entity(label="ORG", spans=[(10, 15)])
        e4 = Entity(label="PER", spans=[(10, 16)])

        assert e1 == e2
        assert e1 != e3  # Different label
        assert e1 != e4  # Different spans
        assert e1 != "not an entity"

    def test_entity_hash(self):
        """Test entity hashing."""
        e1 = Entity(label="PER", spans=[(10, 15)])
        e2 = Entity(label="PER", spans=[(10, 15)])
        e3 = Entity(label="ORG", spans=[(10, 15)])

        assert hash(e1) == hash(e2)
        assert hash(e1) != hash(e3)

        # Test that entities can be added to sets
        entities = {e1, e2, e3}
        assert len(entities) == 2  # e1 and e2 are the same

    def test_entity_repr(self):
        """Test entity string representation."""
        e1 = Entity(label="PER", spans=[(10, 15)])
        repr_str = repr(e1)
        assert "PER" in repr_str
        assert "(10, 15)" in repr_str

        e2 = Entity(label="PROTEIN", spans=[(10, 15), (30, 35)], text="alpha")
        repr_str = repr(e2)
        assert "PROTEIN" in repr_str
        assert "(10, 15)" in repr_str
        assert "(30, 35)" in repr_str
        assert "alpha" in repr_str


class TestEvaluationResult:
    """Tests for the EvaluationResult class."""

    def test_compute_metrics_strict(self):
        """Test computation of strict evaluation metrics."""
        result = EvaluationResult(correct=5, incorrect=2, partial=1, missed=1, spurious=1)
        result.compute_metrics(partial_or_type=False)

        assert result.actual == 9  # 5+2+1+1
        assert result.possible == 9  # 5+2+1+1
        assert result.precision == 5 / 9
        assert result.recall == 5 / 9
        assert abs(result.f1 - (2 * (5/9) * (5/9) / (2 * 5/9))) < 0.001

    def test_compute_metrics_partial(self):
        """Test computation of partial evaluation metrics."""
        result = EvaluationResult(correct=5, incorrect=0, partial=2, missed=1, spurious=1)
        result.compute_metrics(partial_or_type=True)

        assert result.actual == 8  # 5+0+2+1
        assert result.possible == 8  # 5+0+2+1
        assert result.precision == 6 / 8  # (5 + 0.5*2) / 8
        assert result.recall == 6 / 8  # (5 + 0.5*2) / 8

    def test_compute_metrics_zero_cases(self):
        """Test evaluation metrics with zero values."""
        result = EvaluationResult()
        result.compute_metrics()

        assert result.precision == 0
        assert result.recall == 0
        assert result.f1 == 0

    def test_compute_metrics_perfect_score(self):
        """Test evaluation metrics with perfect score."""
        result = EvaluationResult(correct=10, incorrect=0, partial=0, missed=0, spurious=0)
        result.compute_metrics()

        assert result.precision == 1.0
        assert result.recall == 1.0
        assert result.f1 == 1.0


class TestEvaluationIndices:
    """Tests for the EvaluationIndices class."""

    def test_default_initialization(self):
        """Test that indices are initialized as empty lists."""
        indices = EvaluationIndices()

        assert indices.correct_indices == []
        assert indices.incorrect_indices == []
        assert indices.partial_indices == []
        assert indices.missed_indices == []
        assert indices.spurious_indices == []

    def test_add_indices(self):
        """Test adding indices."""
        indices = EvaluationIndices()

        indices.correct_indices.append((0, 1))
        indices.incorrect_indices.append((0, 2))
        indices.partial_indices.append((1, 0))

        assert len(indices.correct_indices) == 1
        assert len(indices.incorrect_indices) == 1
        assert len(indices.partial_indices) == 1
        assert indices.correct_indices[0] == (0, 1)



if __name__ == "__main__":
    def test_all():
        # Init class to trigger all tests
        test_entity = TestEntity()
        test_entity.test_continuous_entity()
        test_entity.test_discontinuous_entity()
        test_entity.test_entity_with_text()
        test_entity.test_spans_sorted() 
        test_entity.test_invalid_span_start_greater_than_end()
        test_entity.test_invalid_span_start_equals_end()
        test_entity.test_empty_spans()
        test_entity.test_overlapping_spans_within_entity()
        test_entity.test_get_all_char_positions()
        test_entity.test_overlaps_with_continuous()
        test_entity.test_overlaps_with_discontinuous()
        test_entity.test_entity_equality()  
        test_entity.test_entity_hash()
        test_entity.test_entity_repr()  
        test_evaluation_result = TestEvaluationResult()
        test_evaluation_result.test_compute_metrics_strict()
        test_evaluation_result.test_compute_metrics_partial()
        test_evaluation_result.test_compute_metrics_zero_cases()
        test_evaluation_result.test_compute_metrics_perfect_score()
        test_evaluation_indices = TestEvaluationIndices()
        test_evaluation_indices.test_default_initialization()
        test_evaluation_indices.test_add_indices()  
    test_all()    
    print("✅ All entity tests would pass!")
