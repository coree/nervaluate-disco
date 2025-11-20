"""
Comprehensive test suite for discontinuous and overlapping entity evaluation.

This test file demonstrates and validates all the key features of nervaluate-disco:
1. Discontinuous entities (entities with multiple non-contiguous spans)
2. Overlapping entities (multiple entities covering the same characters)
3. Optimal matching using Hungarian algorithm
4. Character-based evaluation
5. All evaluation strategies (strict, partial, ent_type, exact)
"""

import pytest
from nervaluate import (
    Evaluator,
    Entity,
    SpanDictLoader,
    SimpleSpanLoader,
)


class TestDiscontinuousEntities:
    """Test evaluation of discontinuous entities."""

    def test_discontinuous_entity_perfect_match(self):
        """Test that discontinuous entities match when spans are identical."""
        # Document: "The CEO resigned and later returned as chairman."
        # Entity: "CEO ... returned" spans characters 4-7 and 26-34
        
        true = [
            {
                'text': 'The CEO resigned and later returned as chairman.',
                'entities': [
                    {
                        'spans': [(4, 7), (27, 35)],  # "CEO" + "returned"
                        'label': 'PERSON',
                        'text': 'CEO returned'
                    }
                ]
            }
        ]
        
        pred = [
            {
                'text': 'The CEO resigned and later returned as chairman.',
                'entities': [
                    {
                        'spans': [(4, 7), (27, 35)],  # Same spans
                        'label': 'PERSON',
                        'text': 'CEO returned'
                    }
                ]
            }
        ]
        
        evaluator = Evaluator(true, pred, tags=['PERSON'], loader='span_dict')
        results = evaluator.evaluate()
        
        # Strict evaluation: should be correct (exact match)
        assert results['overall']['strict'].correct == 1
        assert results['overall']['strict'].incorrect == 0
        assert results['overall']['strict'].missed == 0
        assert results['overall']['strict'].spurious == 0
        
        # All strategies should recognize this as correct
        assert results['overall']['partial'].correct == 1
        assert results['overall']['ent_type'].correct == 1
        assert results['overall']['exact'].correct == 1

    def test_discontinuous_partial_overlap(self):
        """Test partial overlap with discontinuous entities."""
        true = [
            {
                'text': 'The CEO resigned and later returned as chairman.',
                'entities': [
                    {
                        'spans': [(4, 7), (27, 35)],  # "CEO" + "returned"
                        'label': 'PERSON',
                    }
                ]
            }
        ]
        
        # Prediction captures only first span
        pred = [
            {
                'text': 'The CEO resigned and later returned as chairman.',
                'entities': [
                    {
                        'spans': [(4, 7)],  # Only "CEO"
                        'label': 'PERSON',
                    }
                ]
            }
        ]
        
        evaluator = Evaluator(true, pred, tags=['PERSON'], loader='span_dict', min_overlap_percentage=25.0)
        results = evaluator.evaluate()
        
        # Should have partial match (33% overlap: 3 chars out of 11)
        assert results['overall']['partial'].correct == 0
        assert results['overall']['partial'].partial == 1
        assert results['overall']['strict'].incorrect == 1

    def test_three_span_discontinuous_entity(self):
        """Test entity with three discontinuous spans."""
        true = [
            {
                'text': 'The protein p53 regulates cell cycle and prevents tumor growth via apoptosis.',
                'entities': [
                    {
                        'spans': [(4, 11), (12, 15), (69, 78)],  # "protein p53 apoptosis"
                        'label': 'PROTEIN',
                    }
                ]
            }
        ]
        
        pred = [
            {
                'text': 'The protein p53 regulates cell cycle and prevents tumor growth via apoptosis.',
                'entities': [
                    {
                        'spans': [(4, 11), (12, 15), (69, 78)],  # Same
                        'label': 'PROTEIN',
                    }
                ]
            }
        ]
        
        evaluator = Evaluator(true, pred, tags=['PROTEIN'], loader='span_dict')
        results = evaluator.evaluate()
        
        assert results['overall']['strict'].correct == 1
        assert results['overall']['strict'].f1 == 1.0


class TestOverlappingEntities:
    """Test evaluation with overlapping entities."""

    def test_two_overlapping_entities_optimal_matching(self):
        """Test optimal matching with overlapping entities."""
        # Text: "Bank of America Corporation"
        # True entities:
        #   - "Bank of America" (ORG)
        #   - "America" (LOC)
        # Predictions match correctly
        
        true = [
            {
                'text': 'Bank of America Corporation',
                'entities': [
                    {'spans': [(0, 15)], 'label': 'ORG'},  # "Bank of America"
                    {'spans': [(8, 15)], 'label': 'LOC'},  # "America"
                ]
            }
        ]
        
        pred = [
            {
                'text': 'Bank of America Corporation',
                'entities': [
                    {'spans': [(0, 15)], 'label': 'ORG'},
                    {'spans': [(8, 15)], 'label': 'LOC'},
                ]
            }
        ]
        
        evaluator = Evaluator(true, pred, tags=['ORG', 'LOC'], loader='span_dict')
        results = evaluator.evaluate()
        
        # Both should match correctly despite overlap
        assert results['overall']['strict'].correct == 2
        assert results['overall']['strict'].precision == 1.0
        assert results['overall']['strict'].recall == 1.0

    def test_overlapping_entities_wrong_assignment(self):
        """Test that optimal matching prevents wrong entity pairing."""
        # This tests the key benefit of optimal matching over greedy
        
        true = [
            {
                'text': 'John Smith works at Google Inc.',
                'entities': [
                    {'spans': [(0, 10)], 'label': 'PERSON'},   # "John Smith"
                    {'spans': [(20, 30)], 'label': 'ORG'},     # "Google Inc"
                ]
            }
        ]
        
        # Predictions slightly offset but one overlaps both
        pred = [
            {
                'text': 'John Smith works at Google Inc.',
                'entities': [
                    {'spans': [(0, 4)], 'label': 'PERSON'},    # "John" - overlaps true[0]
                    {'spans': [(5, 25)], 'label': 'ORG'},      # "Smith works at Google" - overlaps BOTH!
                ]
            }
        ]
        
        evaluator = Evaluator(true, pred, tags=['PERSON', 'ORG'], loader='span_dict', min_overlap_percentage=30.0)
        results = evaluator.evaluate()
        
        # Optimal matching should pair entities to maximize total overlap
        # pred[0] (40% overlap with PERSON) should match true[0]
        # pred[1] (50% overlap with ORG) should match true[1]
        assert results['overall']['partial'].partial == 2  # Both are partial matches
        assert results['overall']['partial'].spurious == 0
        assert results['overall']['partial'].missed == 0

    def test_heavily_overlapping_entities(self):
        """Test multiple entities all overlapping the same region."""
        true = [
            {
                'text': 'The New York Times building',
                'entities': [
                    {'spans': [(4, 12)], 'label': 'LOC'},      # "New York"
                    {'spans': [(4, 18)], 'label': 'ORG'},      # "New York Times"
                    {'spans': [(19, 27)], 'label': 'BUILDING'}, # "building"
                ]
            }
        ]
        
        pred = [
            {
                'text': 'The New York Times building',
                'entities': [
                    {'spans': [(4, 12)], 'label': 'LOC'},
                    {'spans': [(4, 18)], 'label': 'ORG'},
                    {'spans': [(19, 27)], 'label': 'BUILDING'},
                ]
            }
        ]
        
        evaluator = Evaluator(true, pred, tags=['LOC', 'ORG', 'BUILDING'], loader='span_dict')
        results = evaluator.evaluate()
        
        # All three should match correctly
        assert results['overall']['strict'].correct == 3
        assert results['overall']['strict'].f1 == 1.0


class TestEvaluationStrategies:
    """Test all evaluation strategies."""

    def test_strict_evaluation_requires_label_and_spans(self):
        """Strict: requires both label and spans to match."""
        true = [{'text': 'Test', 'entities': [{'spans': [(0, 4)], 'label': 'A'}]}]
        
        # Same spans, different label
        pred1 = [{'text': 'Test', 'entities': [{'spans': [(0, 4)], 'label': 'B'}]}]
        evaluator1 = Evaluator(true, pred1, tags=['A', 'B'], loader='span_dict')
        results1 = evaluator1.evaluate()
        assert results1['overall']['strict'].correct == 0
        assert results1['overall']['strict'].incorrect == 1
        
        # Different spans, same label
        pred2 = [{'text': 'Test', 'entities': [{'spans': [(0, 3)], 'label': 'A'}]}]
        evaluator2 = Evaluator(true, pred2, tags=['A'], loader='span_dict')
        results2 = evaluator2.evaluate()
        assert results2['overall']['strict'].correct == 0
        assert results2['overall']['strict'].incorrect == 1

    def test_partial_evaluation_ignores_label(self):
        """Partial: only cares about span overlap, not label."""
        true = [{'text': 'Test', 'entities': [{'spans': [(0, 4)], 'label': 'A'}]}]
        pred = [{'text': 'Test', 'entities': [{'spans': [(0, 4)], 'label': 'B'}]}]
        
        evaluator = Evaluator(true, pred, tags=['A', 'B'], loader='span_dict')
        results = evaluator.evaluate()
        
        # Should be correct because spans match (label doesn't matter)
        assert results['overall']['partial'].correct == 1
        assert results['overall']['partial'].incorrect == 0

    def test_ent_type_evaluation_requires_label(self):
        """Ent_type: requires label match, accepts partial span overlap."""
        true = [{'text': 'Test text', 'entities': [{'spans': [(0, 4)], 'label': 'A'}]}]
        
        # Partial overlap with correct label
        pred1 = [{'text': 'Test text', 'entities': [{'spans': [(0, 8)], 'label': 'A'}]}]
        evaluator1 = Evaluator(true, pred1, tags=['A'], loader='span_dict', min_overlap_percentage=50.0)
        results1 = evaluator1.evaluate()
        assert results1['overall']['ent_type'].correct == 1
        
        # Partial overlap with wrong label
        pred2 = [{'text': 'Test text', 'entities': [{'spans': [(0, 8)], 'label': 'B'}]}]
        evaluator2 = Evaluator(true, pred2, tags=['A', 'B'], loader='span_dict', min_overlap_percentage=50.0)
        results2 = evaluator2.evaluate()
        assert results2['overall']['ent_type'].correct == 0
        assert results2['overall']['ent_type'].incorrect == 1

    def test_exact_evaluation_requires_exact_spans(self):
        """Exact: requires exact spans, ignores label."""
        true = [{'text': 'Test', 'entities': [{'spans': [(0, 4)], 'label': 'A'}]}]
        
        # Exact spans, different label
        pred1 = [{'text': 'Test', 'entities': [{'spans': [(0, 4)], 'label': 'B'}]}]
        evaluator1 = Evaluator(true, pred1, tags=['A', 'B'], loader='span_dict')
        results1 = evaluator1.evaluate()
        assert results1['overall']['exact'].correct == 1
        
        # Partial overlap, same label
        pred2 = [{'text': 'Test', 'entities': [{'spans': [(0, 3)], 'label': 'A'}]}]
        evaluator2 = Evaluator(true, pred2, tags=['A'], loader='span_dict')
        results2 = evaluator2.evaluate()
        assert results2['overall']['exact'].correct == 0
        assert results2['overall']['exact'].incorrect == 1


class TestOptimalMatching:
    """Test optimal matching algorithm."""

    def test_optimal_vs_greedy_difference(self):
        """
        Demonstrate case where optimal matching gives better result than greedy.
        
        Setup:
        - True: [Entity A at 0-10, Entity B at 20-30]
        - Pred: [Entity X at 0-15, Entity Y at 18-30]
        
        Greedy might match X->A (60% overlap) and Y->B (40% overlap)
        Optimal should recognize that X->B is invalid and match correctly
        """
        true = [
            {
                'text': '0123456789 0123456789 0123456789',
                'entities': [
                    {'spans': [(0, 10)], 'label': 'A'},   # First 10 chars
                    {'spans': [(21, 31)], 'label': 'B'},  # Last 10 chars
                ]
            }
        ]
        
        pred = [
            {
                'text': '0123456789 0123456789 0123456789',
                'entities': [
                    {'spans': [(0, 15)], 'label': 'A'},   # Overlaps A (100%) and B (0%)
                    {'spans': [(19, 31)], 'label': 'B'},  # Overlaps A (0%) and B (100%)
                ]
            }
        ]
        
        evaluator = Evaluator(true, pred, tags=['A', 'B'], loader='span_dict', min_overlap_percentage=50.0)
        results = evaluator.evaluate()
        
        # Optimal matching should correctly pair them
        assert results['overall']['partial'].partial == 2  # Both are partial (not exact)
        assert results['overall']['partial'].spurious == 0
        assert results['overall']['partial'].missed == 0

    def test_optimal_matching_with_threshold(self):
        """Test that threshold correctly filters matches."""
        true = [
            {
                'text': '0123456789',
                'entities': [
                    {'spans': [(0, 10)], 'label': 'A'},
                ]
            }
        ]
        
        # Prediction with only 30% overlap
        pred = [
            {
                'text': '0123456789',
                'entities': [
                    {'spans': [(0, 3)], 'label': 'A'},  # 30% overlap
                ]
            }
        ]
        
        # With threshold at 50%, should be spurious
        evaluator_high = Evaluator(true, pred, tags=['A'], loader='span_dict', min_overlap_percentage=50.0)
        results_high = evaluator_high.evaluate()
        assert results_high['overall']['partial'].spurious == 1
        assert results_high['overall']['partial'].missed == 1
        
        # With threshold at 25%, should match
        evaluator_low = Evaluator(true, pred, tags=['A'], loader='span_dict', min_overlap_percentage=25.0)
        results_low = evaluator_low.evaluate()
        assert results_low['overall']['partial'].partial == 1
        assert results_low['overall']['partial'].spurious == 0


class TestLoaders:
    """Test data loaders."""

    def test_span_dict_loader(self):
        """Test SpanDictLoader with your format."""
        data = [
            {
                'text': 'The CEO resigned.',
                'entities': [
                    {
                        'spans': [(4, 7)],
                        'label': 'PERSON',
                        'text': 'CEO'
                    }
                ]
            }
        ]
        
        loader = SpanDictLoader()
        loaded = loader.load(data)
        
        assert len(loaded) == 1
        assert len(loaded[0]) == 1
        assert loaded[0][0].label == 'PERSON'
        assert loaded[0][0].spans == [(4, 7)]
        assert loaded[0][0].text == 'CEO'

    def test_simple_span_loader(self):
        """Test SimpleSpanLoader."""
        data = [
            [
                {'label': 'PER', 'spans': [(0, 5)]},
                {'label': 'ORG', 'spans': [(10, 15), (20, 25)]},
            ]
        ]
        
        loader = SimpleSpanLoader()
        loaded = loader.load(data)
        
        assert len(loaded) == 1
        assert len(loaded[0]) == 2
        assert loaded[0][0].label == 'PER'
        assert loaded[0][1].spans == [(10, 15), (20, 25)]

    def test_loader_validation(self):
        """Test that loaders properly validate input."""
        loader = SpanDictLoader()
        
        # Missing 'entities' key
        with pytest.raises(ValueError, match="missing 'entities' key"):
            loader.load([{'text': 'test'}])
        
        # Invalid spans (start >= end)
        with pytest.raises(ValueError, match="Invalid entity"):
            loader.load([{
                'text': 'test',
                'entities': [{'label': 'A', 'spans': [(5, 2)]}]
            }])


class TestCharacterLevelEvaluation:
    """Test character-level precision of evaluation."""

    def test_character_exact_boundaries(self):
        """Test that character boundaries are respected exactly."""
        # "Hello World"
        # Indices: H=0, e=1, l=2, l=3, o=4, space=5, W=6, o=7, r=8, l=9, d=10
        
        true = [{'text': 'Hello World', 'entities': [{'spans': [(0, 5)], 'label': 'A'}]}]  # "Hello"
        pred = [{'text': 'Hello World', 'entities': [{'spans': [(0, 6)], 'label': 'A'}]}]  # "Hello "
        
        evaluator = Evaluator(true, pred, tags=['A'], loader='span_dict')
        results = evaluator.evaluate()
        
        # Should be incorrect (not exact match)
        assert results['overall']['strict'].correct == 0
        assert results['overall']['strict'].incorrect == 1

    def test_overlap_percentage_calculation(self):
        """Test precise overlap percentage calculation."""
        # True entity: 10 characters
        true = [{'text': '0123456789', 'entities': [{'spans': [(0, 10)], 'label': 'A'}]}]
        
        # Pred: 5 characters overlapping (50%)
        pred = [{'text': '0123456789', 'entities': [{'spans': [(0, 5)], 'label': 'A'}]}]
        
        evaluator = Evaluator(true, pred, tags=['A'], loader='span_dict', min_overlap_percentage=50.0)
        results = evaluator.evaluate()
        
        # Should match with exactly 50% threshold
        assert results['overall']['partial'].partial == 1
        
        # Should NOT match with 51% threshold
        evaluator2 = Evaluator(true, pred, tags=['A'], loader='span_dict', min_overlap_percentage=51.0)
        results2 = evaluator2.evaluate()
        assert results2['overall']['partial'].spurious == 1


class TestComplexScenarios:
    """Test complex real-world scenarios."""

    def test_biomedical_text_with_nested_entities(self):
        """Test biomedical NER scenario with nested/overlapping entities."""
        true = [
            {
                'text': 'The p53 protein regulates cell cycle.',
                'entities': [
                    {'spans': [(4, 7)], 'label': 'GENE'},         # "p53"
                    {'spans': [(4, 15)], 'label': 'PROTEIN'},     # "p53 protein" (contains GENE)
                    {'spans': [(26, 36)], 'label': 'PROCESS'},    # "cell cycle"
                ]
            }
        ]
        
        pred = [
            {
                'text': 'The p53 protein regulates cell cycle.',
                'entities': [
                    {'spans': [(4, 7)], 'label': 'GENE'},
                    {'spans': [(4, 15)], 'label': 'PROTEIN'},
                    {'spans': [(26, 36)], 'label': 'PROCESS'},
                ]
            }
        ]
        
        evaluator = Evaluator(true, pred, tags=['GENE', 'PROTEIN', 'PROCESS'], loader='span_dict')
        results = evaluator.evaluate()
        
        # All should match perfectly
        assert results['overall']['strict'].correct == 3
        assert results['overall']['strict'].f1 == 1.0

    def test_multi_document_evaluation(self):
        """Test evaluation across multiple documents."""
        true = [
            {'text': 'Doc 1', 'entities': [{'spans': [(0, 5)], 'label': 'A'}]},
            {'text': 'Doc 2', 'entities': [{'spans': [(0, 5)], 'label': 'B'}]},
            {'text': 'Doc 3', 'entities': [{'spans': [(0, 5)], 'label': 'C'}]},
        ]
        
        pred = [
            {'text': 'Doc 1', 'entities': [{'spans': [(0, 5)], 'label': 'A'}]},  # Correct
            {'text': 'Doc 2', 'entities': [{'spans': [(0, 3)], 'label': 'B'}]},  # Partial
            {'text': 'Doc 3', 'entities': []},                                     # Missed
        ]
        
        evaluator = Evaluator(true, pred, tags=['A', 'B', 'C'], loader='span_dict', min_overlap_percentage=50.0)
        results = evaluator.evaluate()
        
        assert results['overall']['partial'].correct == 1   # Doc 1
        assert results['overall']['partial'].partial == 1   # Doc 2
        assert results['overall']['partial'].missed == 1    # Doc 3

    def test_empty_predictions(self):
        """Test handling of documents with no predictions."""
        true = [
            {'text': 'Test', 'entities': [{'spans': [(0, 4)], 'label': 'A'}]}
        ]
        
        pred = [
            {'text': 'Test', 'entities': []}  # No predictions
        ]
        
        evaluator = Evaluator(true, pred, tags=['A'], loader='span_dict')
        results = evaluator.evaluate()
        
        assert results['overall']['strict'].missed == 1
        assert results['overall']['strict'].spurious == 0
        assert results['overall']['strict'].correct == 0

    def test_empty_ground_truth(self):
        """Test handling of documents with no ground truth."""
        true = [
            {'text': 'Test', 'entities': []}  # No entities
        ]
        
        pred = [
            {'text': 'Test', 'entities': [{'spans': [(0, 4)], 'label': 'A'}]}
        ]
        
        evaluator = Evaluator(true, pred, tags=['A'], loader='span_dict')
        results = evaluator.evaluate()
        
        assert results['overall']['strict'].spurious == 1
        assert results['overall']['strict'].missed == 0
        assert results['overall']['strict'].correct == 0


def test_full_pipeline():
    """
    End-to-end test demonstrating complete evaluation pipeline.
    
    This test shows:
    1. Loading data in your format
    2. Evaluation with all strategies
    3. Report generation
    4. CSV export
    """
    # Complex document with discontinuous and overlapping entities
    true = [
        {
            'text': 'The CEO of Apple Inc resigned but later returned to the company.',
            'entities': [
                {'spans': [(4, 7), (42, 50)], 'label': 'PERSON'},  # "CEO" + "returned" (discontinuous)
                {'spans': [(11, 21)], 'label': 'ORG'},              # "Apple Inc"
                {'spans': [(57, 64)], 'label': 'ORG'},              # "company"
            ]
        }
    ]
    
    pred = [
        {
            'text': 'The CEO of Apple Inc resigned but later returned to the company.',
            'entities': [
                {'spans': [(4, 7), (42, 50)], 'label': 'PERSON'},  # Perfect match
                {'spans': [(11, 21)], 'label': 'COMPANY'},          # Wrong label
                {'spans': [(57, 64)], 'label': 'ORG'},              # Correct
            ]
        }
    ]
    
    # Create evaluator
    evaluator = Evaluator(true, pred, tags=['PERSON', 'ORG', 'COMPANY'], loader='span_dict')
    
    # Run evaluation
    results = evaluator.evaluate()
    
    # Check results
    assert results['overall']['strict'].correct == 2  # PERSON and second ORG
    assert results['overall']['strict'].incorrect == 1  # COMPANY (wrong label)
    
    # Generate report
    report = evaluator.summary_report(mode='overall')
    assert 'strict' in report
    assert 'partial' in report
    
    # Generate CSV
    csv_output = evaluator.results_to_csv(mode='overall')
    assert 'Strategy' in csv_output
    assert 'Precision' in csv_output
    
    print("\n" + "="*80)
    print("FULL PIPELINE TEST - SUMMARY REPORT")
    print("="*80)
    print(report)
    print("\n" + "="*80)
    print("CSV OUTPUT")
    print("="*80)
    print(csv_output)


if __name__ == '__main__':
    # Run all tests
    pytest.main([__file__, '-v', '--tb=short'])
    
    # Run the full pipeline demo
    print("\n\n" + "="*80)
    print("RUNNING FULL PIPELINE DEMONSTRATION")
    print("="*80 + "\n")
    test_full_pipeline()