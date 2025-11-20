"""Tests for evaluation strategies."""

import pytest
from nervaluate.entities import Entity
from nervaluate.strategies import (
    StrictEvaluation,
    PartialEvaluation,
    EntityTypeEvaluation,
    ExactEvaluation
)


class TestStrictEvaluation:
    """Tests for StrictEvaluation strategy."""

    def test_perfect_match(self):
        """Test perfect match - same label and spans."""
        true = [Entity("PER", spans=[(10, 15)])]
        pred = [Entity("PER", spans=[(10, 15)])]

        strategy = StrictEvaluation()
        result, indices = strategy.evaluate(true, pred, ["PER"])

        assert result.correct == 1
        assert result.incorrect == 0
        assert result.missed == 0
        assert result.spurious == 0
        assert len(indices.correct_indices) == 1

    def test_wrong_label(self):
        """Test wrong label - same spans but different label."""
        true = [Entity("PER", spans=[(10, 15)])]
        pred = [Entity("ORG", spans=[(10, 15)])]

        strategy = StrictEvaluation()
        result, indices = strategy.evaluate(true, pred, ["PER", "ORG"])

        assert result.correct == 0
        assert result.incorrect == 1
        assert result.missed == 0
        assert result.spurious == 0

    def test_wrong_boundary(self):
        """Test wrong boundary - same label but different spans."""
        true = [Entity("PER", spans=[(10, 15)])]
        pred = [Entity("PER", spans=[(10, 16)])]

        strategy = StrictEvaluation()
        result, indices = strategy.evaluate(true, pred, ["PER"])

        assert result.correct == 0
        assert result.incorrect == 1
        assert result.missed == 0
        assert result.spurious == 0

    def test_missed_entity(self):
        """Test missed entity - true entity not in predictions."""
        true = [Entity("PER", spans=[(10, 15)])]
        pred = []

        strategy = StrictEvaluation()
        result, indices = strategy.evaluate(true, pred, ["PER"])

        assert result.correct == 0
        assert result.missed == 1
        assert result.spurious == 0

    def test_spurious_entity(self):
        """Test spurious entity - predicted entity not in true."""
        true = []
        pred = [Entity("PER", spans=[(10, 15)])]

        strategy = StrictEvaluation()
        result, indices = strategy.evaluate(true, pred, ["PER"])

        assert result.correct == 0
        assert result.missed == 0
        assert result.spurious == 1

    def test_discontinuous_perfect_match(self):
        """Test perfect match for discontinuous entity."""
        true = [Entity("PROTEIN", spans=[(10, 15), (30, 35)])]
        pred = [Entity("PROTEIN", spans=[(10, 15), (30, 35)])]

        strategy = StrictEvaluation()
        result, indices = strategy.evaluate(true, pred, ["PROTEIN"])

        assert result.correct == 1
        assert result.incorrect == 0

    def test_discontinuous_wrong_spans(self):
        """Test discontinuous entity with wrong spans."""
        true = [Entity("PROTEIN", spans=[(10, 15), (30, 35)])]
        pred = [Entity("PROTEIN", spans=[(10, 15), (30, 36)])]  # Second span differs

        strategy = StrictEvaluation()
        result, indices = strategy.evaluate(true, pred, ["PROTEIN"])

        assert result.correct == 0
        assert result.incorrect == 1

    def test_overlapping_entities(self):
        """Test evaluation with overlapping entities."""
        true = [
            Entity("PROTEIN", spans=[(10, 20)]),
            Entity("GENE", spans=[(15, 25)])  # Overlaps with PROTEIN
        ]
        pred = [
            Entity("PROTEIN", spans=[(10, 20)]),
            Entity("GENE", spans=[(15, 25)])
        ]

        strategy = StrictEvaluation()
        result, indices = strategy.evaluate(true, pred, ["PROTEIN", "GENE"])

        assert result.correct == 2
        assert result.incorrect == 0

    def test_threshold_below_minimum(self):
        """Test that entities below threshold are not matched."""
        true = [Entity("PER", spans=[(10, 20)])]  # 10 chars
        pred = [Entity("PER", spans=[(10, 12)])]  # 2 chars = 20% overlap

        strategy = StrictEvaluation(min_overlap_percentage=50.0)
        result, indices = strategy.evaluate(true, pred, ["PER"])

        assert result.correct == 0
        assert result.incorrect == 0
        assert result.missed == 1
        assert result.spurious == 1  # Below threshold = spurious


class TestPartialEvaluation:
    """Tests for PartialEvaluation strategy."""

    def test_exact_match(self):
        """Test exact span match (label ignored)."""
        true = [Entity("PER", spans=[(10, 15)])]
        pred = [Entity("ORG", spans=[(10, 15)])]  # Different label but same spans

        strategy = PartialEvaluation()
        result, indices = strategy.evaluate(true, pred, ["PER", "ORG"])

        assert result.correct == 1
        assert result.partial == 0
        assert result.incorrect == 0  # No incorrect in partial evaluation

    def test_partial_overlap(self):
        """Test partial boundary overlap."""
        true = [Entity("PER", spans=[(10, 20)])]
        pred = [Entity("PER", spans=[(10, 15)])]  # Partial overlap

        strategy = PartialEvaluation()
        result, indices = strategy.evaluate(true, pred, ["PER"])

        assert result.correct == 0
        assert result.partial == 1
        assert result.incorrect == 0

    def test_no_incorrect_category(self):
        """Test that partial evaluation never has incorrect."""
        true = [Entity("PER", spans=[(10, 15)])]
        pred = [Entity("PER", spans=[(10, 16)])]

        strategy = PartialEvaluation()
        result, indices = strategy.evaluate(true, pred, ["PER"])

        assert result.incorrect == 0
        assert result.partial == 1

    def test_discontinuous_partial(self):
        """Test partial match for discontinuous entities."""
        true = [Entity("PROTEIN", spans=[(10, 15), (30, 35)])]  # 10 chars
        pred = [Entity("PROTEIN", spans=[(10, 14), (30, 35)])]  # 9 chars overlap

        strategy = PartialEvaluation()
        result, indices = strategy.evaluate(true, pred, ["PROTEIN"])

        assert result.correct == 0
        assert result.partial == 1

    def test_partial_uses_half_credit(self):
        """Test that partial matches get 0.5 credit in metrics."""
        true = [Entity("PER", spans=[(10, 20)])]
        pred = [Entity("PER", spans=[(10, 15)])]

        strategy = PartialEvaluation()
        result, indices = strategy.evaluate(true, pred, ["PER"])

        # Precision = (0 + 0.5*1) / 1 = 0.5
        # Recall = (0 + 0.5*1) / 1 = 0.5
        assert result.precision == 0.5
        assert result.recall == 0.5


class TestEntityTypeEvaluation:
    """Tests for EntityTypeEvaluation strategy."""

    def test_correct_type_with_overlap(self):
        """Test correct when same label with sufficient overlap."""
        true = [Entity("PER", spans=[(10, 20)])]
        pred = [Entity("PER", spans=[(10, 15)])]  # Partial overlap but same label

        strategy = EntityTypeEvaluation()
        result, indices = strategy.evaluate(true, pred, ["PER"])

        assert result.correct == 1
        assert result.incorrect == 0

    def test_wrong_type_with_overlap(self):
        """Test incorrect when different label with overlap."""
        true = [Entity("PER", spans=[(10, 20)])]
        pred = [Entity("ORG", spans=[(10, 15)])]  # Overlap but wrong label

        strategy = EntityTypeEvaluation()
        result, indices = strategy.evaluate(true, pred, ["PER", "ORG"])

        assert result.correct == 0
        assert result.incorrect == 1

    def test_exact_boundary_wrong_type(self):
        """Test incorrect even with exact boundary if wrong type."""
        true = [Entity("PER", spans=[(10, 15)])]
        pred = [Entity("ORG", spans=[(10, 15)])]  # Exact boundary but wrong label

        strategy = EntityTypeEvaluation()
        result, indices = strategy.evaluate(true, pred, ["PER", "ORG"])

        assert result.correct == 0
        assert result.incorrect == 1

    def test_discontinuous_correct_type(self):
        """Test correct type for discontinuous with overlap."""
        true = [Entity("PROTEIN", spans=[(10, 15), (30, 35)])]
        pred = [Entity("PROTEIN", spans=[(10, 14), (30, 35)])]

        strategy = EntityTypeEvaluation()
        result, indices = strategy.evaluate(true, pred, ["PROTEIN"])

        assert result.correct == 1
        assert result.incorrect == 0


class TestExactEvaluation:
    """Tests for ExactEvaluation strategy."""

    def test_exact_boundary_different_label(self):
        """Test correct when exact boundary even with different label."""
        true = [Entity("PER", spans=[(10, 15)])]
        pred = [Entity("ORG", spans=[(10, 15)])]

        strategy = ExactEvaluation()
        result, indices = strategy.evaluate(true, pred, ["PER", "ORG"])

        assert result.correct == 1
        assert result.incorrect == 0

    def test_wrong_boundary_same_label(self):
        """Test incorrect when wrong boundary even with same label."""
        true = [Entity("PER", spans=[(10, 15)])]
        pred = [Entity("PER", spans=[(10, 16)])]

        strategy = ExactEvaluation()
        result, indices = strategy.evaluate(true, pred, ["PER"])

        assert result.correct == 0
        assert result.incorrect == 1

    def test_partial_overlap_incorrect(self):
        """Test that partial overlap is incorrect."""
        true = [Entity("PER", spans=[(10, 20)])]
        pred = [Entity("PER", spans=[(10, 15)])]

        strategy = ExactEvaluation()
        result, indices = strategy.evaluate(true, pred, ["PER"])

        assert result.correct == 0
        assert result.incorrect == 1

    def test_discontinuous_exact_match(self):
        """Test exact match for discontinuous entities."""
        true = [Entity("PROTEIN", spans=[(10, 15), (30, 35)])]
        pred = [Entity("GENE", spans=[(10, 15), (30, 35)])]  # Different label but exact spans

        strategy = ExactEvaluation()
        result, indices = strategy.evaluate(true, pred, ["PROTEIN", "GENE"])

        assert result.correct == 1
        assert result.incorrect == 0


class TestComplexScenarios:
    """Tests for complex evaluation scenarios."""

    def test_multiple_overlapping_entities(self):
        """Test complex scenario with multiple overlapping entities."""
        true = [
            Entity("PROTEIN", spans=[(10, 20)]),
            Entity("GENE", spans=[(15, 25)]),
            Entity("CELL", spans=[(30, 40)])
        ]
        pred = [
            Entity("PROTEIN", spans=[(10, 18)]),
            Entity("GENE", spans=[(16, 25)]),
            Entity("CELL", spans=[(30, 40)])
        ]

        strategy = StrictEvaluation(min_overlap_percentage=50.0)
        result, indices = strategy.evaluate(true, pred, ["PROTEIN", "GENE", "CELL"])

        # PROTEIN: 8/10 = 80% overlap, but not exact -> incorrect
        # GENE: 9/10 = 90% overlap, but not exact -> incorrect
        # CELL: exact match -> correct
        assert result.correct == 1
        assert result.incorrect == 2
        assert result.missed == 0
        assert result.spurious == 0

    def test_discontinuous_vs_continuous(self):
        """Test matching discontinuous true against continuous pred."""
        true = [Entity("PROTEIN", spans=[(10, 15), (30, 35)])]  # 10 chars
        pred = [Entity("PROTEIN", spans=[(10, 35)])]  # Continuous, covers both

        strategy = PartialEvaluation()
        result, indices = strategy.evaluate(true, pred, ["PROTEIN"])

        # Pred covers all true chars but spans are different
        assert result.correct == 0
        assert result.partial == 1

    def test_many_to_one_matching(self):
        """Test that multiple predictions don't match same true entity."""
        true = [Entity("PER", spans=[(10, 20)])]
        pred = [
            Entity("PER", spans=[(10, 15)]),
            Entity("PER", spans=[(15, 20)])
        ]

        strategy = StrictEvaluation()
        result, indices = strategy.evaluate(true, pred, ["PER"])

        # Only one can match, other is spurious
        assert result.correct == 0
        assert result.incorrect == 1  # Best match
        assert result.spurious == 1  # Other prediction
        assert result.missed == 0

    def test_nested_entities(self):
        """Test nested/contained entities."""
        true = [
            Entity("PROTEIN", spans=[(10, 30)]),
            Entity("DOMAIN", spans=[(15, 25)])  # Nested inside PROTEIN
        ]
        pred = [
            Entity("PROTEIN", spans=[(10, 30)]),
            Entity("DOMAIN", spans=[(15, 25)])
        ]

        strategy = StrictEvaluation()
        result, indices = strategy.evaluate(true, pred, ["PROTEIN", "DOMAIN"])

        assert result.correct == 2

    def test_threshold_filtering(self):
        """Test that threshold properly filters matches."""
        true = [
            Entity("PER", spans=[(0, 100)]),  # 100 chars
            Entity("ORG", spans=[(200, 300)])  # 100 chars
        ]
        pred = [
            Entity("PER", spans=[(0, 30)]),  # 30% overlap
            Entity("ORG", spans=[(200, 280)])  # 80% overlap
        ]

        strategy = PartialEvaluation(min_overlap_percentage=50.0)
        result, indices = strategy.evaluate(true, pred, ["PER", "ORG"])

        # PER: 30% < 50% -> not matched -> missed and spurious
        # ORG: 80% > 50% -> matched -> partial
        assert result.partial == 1
        assert result.missed == 1
        assert result.spurious == 1

    def test_empty_predictions(self):
        """Test with no predictions."""
        true = [Entity("PER", spans=[(10, 15)])]
        pred = []

        strategy = StrictEvaluation()
        result, indices = strategy.evaluate(true, pred, ["PER"])

        assert result.correct == 0
        assert result.missed == 1
        assert result.spurious == 0

    def test_empty_true(self):
        """Test with no true entities."""
        true = []
        pred = [Entity("PER", spans=[(10, 15)])]

        strategy = StrictEvaluation()
        result, indices = strategy.evaluate(true, pred, ["PER"])

        assert result.correct == 0
        assert result.missed == 0
        assert result.spurious == 1

    def test_both_empty(self):
        """Test with both empty."""
        true = []
        pred = []

        strategy = StrictEvaluation()
        result, indices = strategy.evaluate(true, pred, ["PER"])

        assert result.correct == 0
        assert result.missed == 0
        assert result.spurious == 0
        assert result.precision == 0
        assert result.recall == 0


class TestInvalidInputs:
    """Tests for invalid inputs and edge cases."""

    def test_invalid_threshold_too_low(self):
        """Test that threshold below 1.0 raises error."""
        with pytest.raises(ValueError, match="must be between 1.0 and 100.0"):
            StrictEvaluation(min_overlap_percentage=0.5)

    def test_invalid_threshold_too_high(self):
        """Test that threshold above 100.0 raises error."""
        with pytest.raises(ValueError, match="must be between 1.0 and 100.0"):
            StrictEvaluation(min_overlap_percentage=101.0)

    def test_valid_threshold_boundaries(self):
        """Test that boundary values are accepted."""
        StrictEvaluation(min_overlap_percentage=1.0)
        StrictEvaluation(min_overlap_percentage=100.0)
        assert True  # If we get here, no exceptions were raised



if __name__ == "__main__":
    
    def test_all():
        # Init class to trigger all tests
        test_strict = TestStrictEvaluation()
        test_strict.test_perfect_match()
        test_strict.test_wrong_label()
        test_strict.test_wrong_boundary()
        test_strict.test_missed_entity()
        test_strict.test_spurious_entity()
        test_strict.test_discontinuous_perfect_match()
        test_strict.test_discontinuous_wrong_spans()
        test_strict.test_overlapping_entities()
        test_strict.test_threshold_below_minimum()
        test_partial = TestPartialEvaluation()
        test_partial.test_exact_match()
        test_partial.test_partial_overlap()
        test_partial.test_no_incorrect_category()
        test_partial.test_discontinuous_partial()
        test_partial.test_partial_uses_half_credit()
        test_entity_type = TestEntityTypeEvaluation()
        test_entity_type.test_correct_type_with_overlap()
        test_entity_type.test_wrong_type_with_overlap()
        test_entity_type.test_exact_boundary_wrong_type()
        test_entity_type.test_discontinuous_correct_type()
        test_exact = TestExactEvaluation()
        test_exact.test_exact_boundary_different_label()
        test_exact.test_wrong_boundary_same_label()
        test_exact.test_partial_overlap_incorrect()
        test_exact.test_discontinuous_exact_match()
        test_complex = TestComplexScenarios()
        test_complex.test_multiple_overlapping_entities()
        test_complex.test_discontinuous_vs_continuous()
        test_complex.test_many_to_one_matching()
        test_complex.test_nested_entities()
        test_complex.test_threshold_filtering()
        test_complex.test_empty_predictions()
        test_complex.test_empty_true()
        test_complex.test_both_empty()
        test_invalid = TestInvalidInputs()
        test_invalid.test_invalid_threshold_too_low()
        test_invalid.test_invalid_threshold_too_high()  
        test_invalid.test_valid_threshold_boundaries()
    
    
    # test_all()  
    # print("All tests in test_strategies.py passed.")

    import pytest
    raise SystemExit(pytest.main(["-vv", __file__]))
        

