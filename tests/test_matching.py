"""Tests for matching algorithms."""

import pytest
from nervaluate.entities import Entity
from nervaluate.matching import calculate_overlap_percentage, optimal_match, greedy_match


class TestCalculateOverlapPercentage:
    """Tests for overlap percentage calculation."""

    def test_complete_overlap_continuous(self):
        """Test 100% overlap for identical continuous entities."""
        true = Entity("PER", spans=[(10, 20)])
        pred = Entity("PER", spans=[(10, 20)])
        assert calculate_overlap_percentage(pred, true) == 100.0

    def test_no_overlap(self):
        """Test 0% overlap for non-overlapping entities."""
        true = Entity("PER", spans=[(10, 20)])
        pred = Entity("PER", spans=[(30, 40)])
        assert calculate_overlap_percentage(pred, true) == 0.0

    def test_partial_overlap(self):
        """Test partial overlap percentage."""
        true = Entity("PER", spans=[(10, 20)])  # 10 chars
        pred = Entity("PER", spans=[(15, 25)])  # Overlaps 15-20 = 5 chars
        overlap = calculate_overlap_percentage(pred, true)
        assert overlap == 50.0  # 5/10 = 50%

    def test_discontinuous_complete_overlap(self):
        """Test 100% overlap for discontinuous entities."""
        true = Entity("PROTEIN", spans=[(10, 15), (30, 35)])  # 10 chars total
        pred = Entity("PROTEIN", spans=[(10, 15), (30, 35)])
        assert calculate_overlap_percentage(pred, true) == 100.0

    def test_discontinuous_partial_overlap(self):
        """Test partial overlap for discontinuous entities."""
        true = Entity("PROTEIN", spans=[(10, 15), (30, 35)])  # 10 chars
        pred = Entity("PROTEIN", spans=[(10, 12), (33, 35)])  # 4 chars overlap
        overlap = calculate_overlap_percentage(pred, true)
        assert overlap == 40.0  # 4/10 = 40%

    def test_continuous_vs_discontinuous(self):
        """Test overlap between continuous and discontinuous entities."""
        true = Entity("GENE", spans=[(10, 15), (30, 35)])  # 10 chars
        pred = Entity("GENE", spans=[(0, 40)])  # Covers both spans plus gap
        overlap = calculate_overlap_percentage(pred, true)
        assert overlap == 100.0  # Covers all true entity chars

    def test_discontinuous_one_span_overlap(self):
        """Test overlap with only one span of discontinuous entity."""
        true = Entity("PROTEIN", spans=[(10, 15), (30, 35)])  # 10 chars
        pred = Entity("PROTEIN", spans=[(10, 15)])  # Only first span
        overlap = calculate_overlap_percentage(pred, true)
        assert overlap == 50.0  # 5/10 = 50%

    def test_pred_larger_than_true(self):
        """Test when prediction is larger than true entity."""
        true = Entity("PER", spans=[(10, 15)])  # 5 chars
        pred = Entity("PER", spans=[(5, 20)])  # 15 chars, but only 5 overlap
        overlap = calculate_overlap_percentage(pred, true)
        assert overlap == 100.0  # All of true is covered

    def test_adjacent_no_overlap(self):
        """Test adjacent entities with no overlap."""
        true = Entity("PER", spans=[(10, 15)])
        pred = Entity("PER", spans=[(15, 20)])
        assert calculate_overlap_percentage(pred, true) == 0.0


class TestOptimalMatch:
    """Tests for optimal (Hungarian) matching algorithm."""

    def test_perfect_matches(self):
        """Test optimal matching with perfect matches."""
        true = [
            Entity("PER", spans=[(10, 15)]),
            Entity("ORG", spans=[(20, 25)])
        ]
        pred = [
            Entity("PER", spans=[(10, 15)]),
            Entity("ORG", spans=[(20, 25)])
        ]

        matches, matched_true, matched_pred = optimal_match(true, pred, threshold=100.0)

        assert len(matches) == 2
        assert len(matched_true) == 2
        assert len(matched_pred) == 2

    def test_no_matches_below_threshold(self):
        """Test that matches below threshold are not included."""
        true = [Entity("PER", spans=[(10, 20)])]  # 10 chars
        pred = [Entity("PER", spans=[(10, 12)])]  # 2 chars overlap = 20%

        matches, matched_true, matched_pred = optimal_match(true, pred, threshold=50.0)

        assert len(matches) == 0
        assert len(matched_true) == 0
        assert len(matched_pred) == 0

    def test_matches_above_threshold(self):
        """Test that matches above threshold are included."""
        true = [Entity("PER", spans=[(10, 20)])]  # 10 chars
        pred = [Entity("PER", spans=[(10, 16)])]  # 6 chars overlap = 60%

        matches, matched_true, matched_pred = optimal_match(true, pred, threshold=50.0)

        assert len(matches) == 1
        assert 0 in matched_true
        assert 0 in matched_pred

    def test_optimal_vs_greedy_difference(self):
        """Test case where optimal differs from greedy."""
        # Set up scenario where greedy would pick suboptimal match
        true = [
            Entity("PER", spans=[(0, 10)]),   # 10 chars
            Entity("ORG", spans=[(20, 30)])   # 10 chars
        ]
        pred = [
            Entity("PER", spans=[(0, 5)]),    # 50% overlap with true[0]
            Entity("ORG", spans=[(20, 30)])   # 100% overlap with true[1]
        ]

        matches, matched_true, matched_pred = optimal_match(true, pred, threshold=1.0)

        # Optimal should match pred[0]->true[0] and pred[1]->true[1]
        assert len(matches) == 2
        assert (0, 0, 50.0, true[0], pred[0]) in matches
        assert (1, 1, 100.0, true[1], pred[1]) in matches

    def test_overlapping_entities(self):
        """Test matching with overlapping entities."""
        true = [
            Entity("PROTEIN", spans=[(10, 20)]),
            Entity("GENE", spans=[(15, 25)])  # Overlaps with PROTEIN
        ]
        pred = [
            Entity("PROTEIN", spans=[(10, 18)]),
            Entity("GENE", spans=[(16, 25)])
        ]

        matches, matched_true, matched_pred = optimal_match(true, pred, threshold=50.0)

        # Should optimally match based on overlap scores
        assert len(matches) == 2

    def test_empty_inputs(self):
        """Test matching with empty inputs."""
        matches, matched_true, matched_pred = optimal_match([], [], threshold=50.0)
        assert len(matches) == 0
        assert len(matched_true) == 0
        assert len(matched_pred) == 0

        true = [Entity("PER", spans=[(10, 15)])]
        matches, matched_true, matched_pred = optimal_match(true, [], threshold=50.0)
        assert len(matches) == 0

        pred = [Entity("PER", spans=[(10, 15)])]
        matches, matched_true, matched_pred = optimal_match([], pred, threshold=50.0)
        assert len(matches) == 0

    def test_more_predictions_than_true(self):
        """Test when there are more predictions than true entities."""
        true = [Entity("PER", spans=[(10, 15)])]
        pred = [
            Entity("PER", spans=[(10, 15)]),
            Entity("ORG", spans=[(20, 25)]),
            Entity("LOC", spans=[(30, 35)])
        ]

        matches, matched_true, matched_pred = optimal_match(true, pred, threshold=100.0)

        assert len(matches) == 1  # Only one can match
        assert len(matched_true) == 1
        assert len(matched_pred) == 1

    def test_more_true_than_predictions(self):
        """Test when there are more true entities than predictions."""
        true = [
            Entity("PER", spans=[(10, 15)]),
            Entity("ORG", spans=[(20, 25)]),
            Entity("LOC", spans=[(30, 35)])
        ]
        pred = [Entity("PER", spans=[(10, 15)])]

        matches, matched_true, matched_pred = optimal_match(true, pred, threshold=100.0)

        assert len(matches) == 1
        assert len(matched_true) == 1
        assert len(matched_pred) == 1

    def test_discontinuous_optimal_matching(self):
        """Test optimal matching with discontinuous entities."""
        true = [
            Entity("PROTEIN", spans=[(10, 15), (30, 35)]),  # 10 chars
            Entity("GENE", spans=[(50, 60)])  # 10 chars
        ]
        pred = [
            Entity("PROTEIN", spans=[(10, 14), (30, 35)]),  # 9 chars overlap = 90%
            Entity("GENE", spans=[(50, 55)])  # 5 chars overlap = 50%
        ]

        matches, matched_true, matched_pred = optimal_match(true, pred, threshold=50.0)

        assert len(matches) == 2
        # Check that each prediction is optimally matched
        match_dict = {m[1]: m for m in matches}  # pred_idx -> match
        assert match_dict[0][0] == 0  # pred[0] -> true[0]
        assert match_dict[1][0] == 1  # pred[1] -> true[1]


class TestGreedyMatch:
    """Tests for greedy matching algorithm."""

    def test_greedy_perfect_matches(self):
        """Test greedy matching with perfect matches."""
        true = [
            Entity("PER", spans=[(10, 15)]),
            Entity("ORG", spans=[(20, 25)])
        ]
        pred = [
            Entity("PER", spans=[(10, 15)]),
            Entity("ORG", spans=[(20, 25)])
        ]

        matches, matched_true, matched_pred = greedy_match(true, pred, threshold=100.0)

        assert len(matches) == 2
        assert len(matched_true) == 2
        assert len(matched_pred) == 2

    def test_greedy_selects_highest_first(self):
        """Test that greedy selects highest overlap first."""
        true = [Entity("PER", spans=[(10, 20)])]  # 10 chars
        pred = [
            Entity("PER", spans=[(10, 15)]),  # 50% overlap
            Entity("PER", spans=[(10, 18)])   # 80% overlap
        ]

        matches, matched_true, matched_pred = greedy_match(true, pred, threshold=1.0)

        # Should select the 80% overlap first
        assert len(matches) == 1
        assert matches[0][1] == 1  # pred index 1 (80% overlap)
        assert matches[0][2] == 80.0  # overlap percentage

    def test_greedy_empty_inputs(self):
        """Test greedy matching with empty inputs."""
        matches, matched_true, matched_pred = greedy_match([], [], threshold=50.0)
        assert len(matches) == 0

    def test_greedy_vs_optimal_comparison(self):
        """Compare greedy and optimal results."""
        true = [
            Entity("PER", spans=[(0, 10)]),
            Entity("ORG", spans=[(20, 30)])
        ]
        pred = [
            Entity("PER", spans=[(0, 8)]),
            Entity("ORG", spans=[(20, 30)])
        ]

        greedy_matches, _, _ = greedy_match(true, pred, threshold=50.0)
        optimal_matches, _, _ = optimal_match(true, pred, threshold=50.0)

        # In this simple case, greedy and optimal should agree
        assert len(greedy_matches) == len(optimal_matches) == 2


if __name__ == "__main__":
    def test_all():
        
        # Init all classses
        test_calc = TestCalculateOverlapPercentage()
        test_optimal = TestOptimalMatch()
        test_greedy = TestGreedyMatch()

        # Run all tests
        for method in dir(test_calc):
            if method.startswith("test_"):
                getattr(test_calc, method)()
        for method in dir(test_optimal):
            if method.startswith("test_"):
                getattr(test_optimal, method)()
        for method in dir(test_greedy):
            if method.startswith("test_"):
                getattr(test_greedy, method)()  
    test_all()  
    print("All tests passed.")