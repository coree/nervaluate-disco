"""
Comprehensive integration tests for the entire evaluation system.

Tests the complete workflow from data loading through evaluation to reporting.
"""

import pytest
from nervaluate import Evaluator, Entity


class TestEndToEndEvaluation:
    """End-to-end tests with real-world scenarios."""

    def test_simple_continuous_entities(self):
        """Test simple case with continuous entities."""
        true = [
            {
                'text': 'John Smith works at Google Inc.',
                'entities': [
                    {'spans': [(0, 10)], 'label': 'PERSON', 'text': 'John Smith'},
                    {'spans': [(20, 30)], 'label': 'ORG', 'text': 'Google Inc'}
                ]
            }
        ]
        pred = [
            {
                'text': 'John Smith works at Google Inc.',
                'entities': [
                    {'spans': [(0, 10)], 'label': 'PERSON', 'text': 'John Smith'},
                    {'spans': [(20, 30)], 'label': 'ORG', 'text': 'Google Inc'}
                ]
            }
        ]

        evaluator = Evaluator(true, pred, tags=['PERSON', 'ORG'], loader='span_dict')
        results = evaluator.evaluate()

        # Perfect matches
        assert results['overall']['strict'].correct == 2
        assert results['overall']['strict'].f1 == 1.0

    def test_discontinuous_entities(self):
        """Test evaluation with discontinuous entities."""
        true = [
            {
                'text': 'The alpha and beta receptor complex',
                'entities': [
                    {'spans': [(4, 9), (14, 18)], 'label': 'PROTEIN', 'text': 'alpha beta'}
                ]
            }
        ]
        pred = [
            {
                'text': 'The alpha and beta receptor complex',
                'entities': [
                    {'spans': [(4, 9), (14, 18)], 'label': 'PROTEIN', 'text': 'alpha beta'}
                ]
            }
        ]

        evaluator = Evaluator(true, pred, tags=['PROTEIN'], loader='span_dict')
        results = evaluator.evaluate()

        assert results['overall']['strict'].correct == 1
        assert results['overall']['strict'].f1 == 1.0

    def test_overlapping_entities(self):
        """Test evaluation with overlapping entities."""
        true = [
            {
                'text': 'protein kinase activity',
                'entities': [
                    {'spans': [(0, 7)], 'label': 'PROTEIN'},
                    {'spans': [(0, 14)], 'label': 'ENZYME'},  # Overlaps with PROTEIN
                    {'spans': [(8, 23)], 'label': 'FUNCTION'}
                ]
            }
        ]
        pred = [
            {
                'text': 'protein kinase activity',
                'entities': [
                    {'spans': [(0, 7)], 'label': 'PROTEIN'},
                    {'spans': [(0, 14)], 'label': 'ENZYME'},
                    {'spans': [(8, 23)], 'label': 'FUNCTION'}
                ]
            }
        ]

        evaluator = Evaluator(true, pred, tags=['PROTEIN', 'ENZYME', 'FUNCTION'], loader='span_dict')
        results = evaluator.evaluate()

        assert results['overall']['strict'].correct == 3

    def test_mixed_correct_and_errors(self):
        """Test realistic scenario with mix of correct, incorrect, and missed."""
        true = [
            {
                'text': 'John Smith works at Google in California.',
                'entities': [
                    {'spans': [(0, 10)], 'label': 'PERSON'},
                    {'spans': [(20, 26)], 'label': 'ORG'},
                    {'spans': [(30, 40)], 'label': 'LOC'}
                ]
            }
        ]
        pred = [
            {
                'text': 'John Smith works at Google in California.',
                'entities': [
                    {'spans': [(0, 10)], 'label': 'PERSON'},  # Correct
                    {'spans': [(20, 26)], 'label': 'LOC'},     # Wrong label
                    # Missing California
                ]
            }
        ]

        evaluator = Evaluator(true, pred, tags=['PERSON', 'ORG', 'LOC'], loader='span_dict')
        results = evaluator.evaluate()

        strict = results['overall']['strict']
        assert strict.correct == 1  # PERSON
        assert strict.incorrect == 1  # Google with wrong label
        assert strict.missed == 1  # California

    def test_partial_boundary_matches(self):
        """Test partial evaluation with boundary mismatches."""
        true = [
            {
                'text': 'The United States of America is large.',
                'entities': [
                    {'spans': [(4, 28)], 'label': 'COUNTRY'}
                ]
            }
        ]
        pred = [
            {
                'text': 'The United States of America is large.',
                'entities': [
                    {'spans': [(4, 17)], 'label': 'COUNTRY'}  # Partial boundary
                ]
            }
        ]

        evaluator = Evaluator(true, pred, tags=['COUNTRY'], loader='span_dict')
        results = evaluator.evaluate()

        partial = results['overall']['partial']
        assert partial.correct == 0
        assert partial.partial == 1
        assert partial.precision == 0.5  # 0.5 credit for partial

    def test_multiple_documents(self):
        """Test evaluation across multiple documents."""
        true = [
            {
                'text': 'First document.',
                'entities': [{'spans': [(0, 5)], 'label': 'WORD'}]
            },
            {
                'text': 'Second document.',
                'entities': [{'spans': [(0, 6)], 'label': 'WORD'}]
            }
        ]
        pred = [
            {
                'text': 'First document.',
                'entities': [{'spans': [(0, 5)], 'label': 'WORD'}]
            },
            {
                'text': 'Second document.',
                'entities': [{'spans': [(0, 6)], 'label': 'WORD'}]
            }
        ]

        evaluator = Evaluator(true, pred, tags=['WORD'], loader='span_dict')
        results = evaluator.evaluate()

        assert results['overall']['strict'].correct == 2

    def test_threshold_impact(self):
        """Test impact of overlap threshold on results."""
        true = [
            {
                'text': 'A' * 100,
                'entities': [{'spans': [(0, 100)], 'label': 'TEST'}]
            }
        ]
        pred = [
            {
                'text': 'A' * 100,
                'entities': [{'spans': [(0, 40)], 'label': 'TEST'}]  # 40% overlap
            }
        ]

        # With low threshold - should match
        eval_low = Evaluator(true, pred, tags=['TEST'], loader='span_dict', min_overlap_percentage=30.0)
        results_low = eval_low.evaluate()
        assert results_low['overall']['partial'].partial == 1

        # With high threshold - should not match
        eval_high = Evaluator(true, pred, tags=['TEST'], loader='span_dict', min_overlap_percentage=50.0)
        results_high = eval_high.evaluate()
        assert results_high['overall']['partial'].spurious == 1
        assert results_high['overall']['partial'].missed == 1

    def test_entity_level_metrics(self):
        """Test per-entity-type metrics."""
        true = [
            {
                'text': 'John at Google in NYC',
                'entities': [
                    {'spans': [(0, 4)], 'label': 'PERSON'},
                    {'spans': [(8, 14)], 'label': 'ORG'},
                    {'spans': [(18, 21)], 'label': 'LOC'}
                ]
            }
        ]
        pred = [
            {
                'text': 'John at Google in NYC',
                'entities': [
                    {'spans': [(0, 4)], 'label': 'PERSON'},  # Correct
                    {'spans': [(8, 13)], 'label': 'ORG'},    # Wrong boundary
                    # Missing LOC
                ]
            }
        ]

        evaluator = Evaluator(true, pred, tags=['PERSON', 'ORG', 'LOC'], loader='span_dict')
        results = evaluator.evaluate()

        # Check entity-specific results
        assert results['entities']['PERSON']['strict'].correct == 1
        assert results['entities']['ORG']['strict'].incorrect == 1
        assert results['entities']['LOC']['strict'].missed == 1


class TestReporting:
    """Tests for reporting functionality."""

    def test_summary_report_overall(self):
        """Test summary report generation."""
        true = [
            {
                'text': 'Test text',
                'entities': [{'spans': [(0, 4)], 'label': 'TEST'}]
            }
        ]
        pred = [
            {
                'text': 'Test text',
                'entities': [{'spans': [(0, 4)], 'label': 'TEST'}]
            }
        ]

        evaluator = Evaluator(true, pred, tags=['TEST'], loader='span_dict')
        report = evaluator.summary_report(mode='overall')

        assert 'correct' in report
        assert 'precision' in report
        assert 'strict' in report

    def test_summary_report_entities(self):
        """Test entity-level summary report."""
        true = [
            {
                'text': 'Test',
                'entities': [
                    {'spans': [(0, 2)], 'label': 'A'},
                    {'spans': [(2, 4)], 'label': 'B'}
                ]
            }
        ]
        pred = [
            {
                'text': 'Test',
                'entities': [
                    {'spans': [(0, 2)], 'label': 'A'},
                    {'spans': [(2, 4)], 'label': 'B'}
                ]
            }
        ]

        evaluator = Evaluator(true, pred, tags=['A', 'B'], loader='span_dict')
        report = evaluator.summary_report(mode='entities', scenario='strict')

        assert 'A' in report or 'B' in report

    def test_csv_export_overall(self):
        """Test CSV export functionality."""
        true = [
            {
                'text': 'Test',
                'entities': [{'spans': [(0, 4)], 'label': 'TEST'}]
            }
        ]
        pred = [
            {
                'text': 'Test',
                'entities': [{'spans': [(0, 4)], 'label': 'TEST'}]
            }
        ]

        evaluator = Evaluator(true, pred, tags=['TEST'], loader='span_dict')
        csv_output = evaluator.results_to_csv(mode='overall')

        assert isinstance(csv_output, str)
        assert 'Strategy' in csv_output
        assert 'Precision' in csv_output

    def test_csv_export_entities(self):
        """Test entity-level CSV export."""
        true = [
            {
                'text': 'Test',
                'entities': [{'spans': [(0, 4)], 'label': 'TEST'}]
            }
        ]
        pred = [
            {
                'text': 'Test',
                'entities': [{'spans': [(0, 4)], 'label': 'TEST'}]
            }
        ]

        evaluator = Evaluator(true, pred, tags=['TEST'], loader='span_dict')
        csv_output = evaluator.results_to_csv(mode='entities', scenario='strict')

        assert isinstance(csv_output, str)
        assert 'Entity' in csv_output


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_mismatched_document_counts(self):
        """Test error when document counts don't match."""
        true = [{'text': 'A', 'entities': []}]
        pred = [
            {'text': 'A', 'entities': []},
            {'text': 'B', 'entities': []}
        ]

        with pytest.raises(ValueError, match="Number of documents mismatch"):
            Evaluator(true, pred, tags=['TEST'], loader='span_dict')

    def test_invalid_loader(self):
        """Test error with invalid loader."""
        true = [{'text': 'A', 'entities': []}]
        pred = [{'text': 'A', 'entities': []}]

        with pytest.raises(ValueError, match="Unknown loader"):
            Evaluator(true, pred, tags=['TEST'], loader='invalid')

    def test_empty_tags_list(self):
        """Test with empty tags list."""
        true = [
            {
                'text': 'Test',
                'entities': [{'spans': [(0, 4)], 'label': 'TEST'}]
            }
        ]
        pred = [
            {
                'text': 'Test',
                'entities': [{'spans': [(0, 4)], 'label': 'TEST'}]
            }
        ]

        evaluator = Evaluator(true, pred, tags=[], loader='span_dict')
        results = evaluator.evaluate()

        # No tags means no entities evaluated
        assert results['overall']['strict'].correct == 0

    def test_tags_filter(self):
        """Test that only specified tags are evaluated."""
        true = [
            {
                'text': 'Test',
                'entities': [
                    {'spans': [(0, 2)], 'label': 'A'},
                    {'spans': [(2, 4)], 'label': 'B'}
                ]
            }
        ]
        pred = [
            {
                'text': 'Test',
                'entities': [
                    {'spans': [(0, 2)], 'label': 'A'},
                    {'spans': [(2, 4)], 'label': 'B'}
                ]
            }
        ]

        # Only evaluate 'A'
        evaluator = Evaluator(true, pred, tags=['A'], loader='span_dict')
        results = evaluator.evaluate()

        assert results['overall']['strict'].correct == 1  # Only A

    def test_no_entities(self):
        """Test with documents containing no entities."""
        true = [{'text': 'No entities', 'entities': []}]
        pred = [{'text': 'No entities', 'entities': []}]

        evaluator = Evaluator(true, pred, tags=['TEST'], loader='span_dict')
        results = evaluator.evaluate()

        assert results['overall']['strict'].correct == 0
        assert results['overall']['strict'].f1 == 0


class TestComplexRealWorld:
    """Complex real-world scenarios."""

    def test_biomedical_discontinuous(self):
        """Test biomedical text with discontinuous entities."""
        true = [
            {
                'text': 'The alpha-2A and beta-3B adrenergic receptor subtypes',
                'entities': [
                    # Discontinuous protein complex
                    {'spans': [(4, 12), (17, 24)], 'label': 'PROTEIN'},
                    # Continuous term
                    {'spans': [(25, 46)], 'label': 'RECEPTOR'},
                ]
            }
        ]
        pred = [
            {
                'text': 'The alpha-2A and beta-3B adrenergic receptor subtypes',
                'entities': [
                    # Correctly predicted discontinuous
                    {'spans': [(4, 12), (17, 24)], 'label': 'PROTEIN'},
                    # Partial boundary on continuous
                    {'spans': [(25, 43)], 'label': 'RECEPTOR'},
                ]
            }
        ]

        evaluator = Evaluator(true, pred, tags=['PROTEIN', 'RECEPTOR'], loader='span_dict')
        results = evaluator.evaluate()

        assert results['overall']['strict'].correct == 1  # PROTEIN
        assert results['overall']['partial'].partial == 1  # RECEPTOR

    def test_nested_and_overlapping(self):
        """Test nested and overlapping entity scenario."""
        true = [
            {
                'text': 'New York City Department of Health',
                'entities': [
                    {'spans': [(0, 13)], 'label': 'CITY'},
                    {'spans': [(0, 34)], 'label': 'ORG'},  # Contains CITY
                    {'spans': [(25, 34)], 'label': 'DEPT'}  # Nested in ORG
                ]
            }
        ]
        pred = [
            {
                'text': 'New York City Department of Health',
                'entities': [
                    {'spans': [(0, 13)], 'label': 'CITY'},
                    {'spans': [(0, 34)], 'label': 'ORG'},
                    {'spans': [(25, 34)], 'label': 'DEPT'}
                ]
            }
        ]

        evaluator = Evaluator(true, pred, tags=['CITY', 'ORG', 'DEPT'], loader='span_dict')
        results = evaluator.evaluate()

        assert results['overall']['strict'].correct == 3


def test_complete_workflow():
    """Test complete workflow from data to report."""
    # Prepare data
    true_data = [
        {
            'text': 'John Smith works at Google Inc in California.',
            'entities': [
                {'spans': [(0, 10)], 'label': 'PERSON', 'text': 'John Smith'},
                {'spans': [(20, 30)], 'label': 'ORG', 'text': 'Google Inc'},
                {'spans': [(34, 44)], 'label': 'LOC', 'text': 'California'}
            ]
        },
        {
            'text': 'The alpha and beta receptor',
            'entities': [
                {'spans': [(4, 9), (14, 18)], 'label': 'PROTEIN'}
            ]
        }
    ]

    pred_data = [
        {
            'text': 'John Smith works at Google Inc in California.',
            'entities': [
                {'spans': [(0, 10)], 'label': 'PERSON'},
                {'spans': [(20, 30)], 'label': 'COMPANY'},  # Wrong label
                {'spans': [(34, 45)], 'label': 'LOC'}  # Wrong boundary
            ]
        },
        {
            'text': 'The alpha and beta receptor',
            'entities': [
                {'spans': [(4, 9), (14, 18)], 'label': 'PROTEIN'}  # Correct discontinuous
            ]
        }
    ]

    # Create evaluator
    evaluator = Evaluator(
        true_data,
        pred_data,
        tags=['PERSON', 'ORG', 'COMPANY', 'LOC', 'PROTEIN'],
        loader='span_dict',
        min_overlap_percentage=50.0
    )

    # Run evaluation
    results = evaluator.evaluate()

    # Check overall results
    assert results['overall']['strict'].correct >= 1  # PERSON and PROTEIN
    assert results['overall']['strict'].incorrect >= 1  # Wrong label or boundary

    # Generate reports
    overall_report = evaluator.summary_report(mode='overall')
    assert 'strict' in overall_report

    entity_report = evaluator.summary_report(mode='entities', scenario='partial')
    assert len(entity_report) > 0

    # Export CSV
    csv_output = evaluator.results_to_csv(mode='overall')
    assert 'Strategy' in csv_output

    print("\n" + "=" * 70)
    print("COMPLETE WORKFLOW TEST PASSED!")
    print("=" * 70)
    print("\nOverall Report:")
    print(overall_report)
    print("\nCSV Output:")
    print(csv_output)


def test_all_integration():
    """Run all integration tests."""
    test_suite = TestEndToEndEvaluation()
    test_suite.test_simple_continuous_entities()
    test_suite.test_discontinuous_entities()
    test_suite.test_overlapping_entities()
    test_suite.test_mixed_correct_and_errors()
    test_suite.test_partial_boundary_matches()
    test_suite.test_multiple_documents()
    test_suite.test_threshold_impact()
    test_suite.test_entity_level_metrics()

    report_suite = TestReporting()
    report_suite.test_summary_report_overall()
    report_suite.test_summary_report_entities()
    report_suite.test_csv_export_overall()
    report_suite.test_csv_export_entities()

    edge_case_suite = TestEdgeCases()
    edge_case_suite.test_mismatched_document_counts()
    edge_case_suite.test_invalid_loader()
    edge_case_suite.test_empty_tags_list()
    edge_case_suite.test_tags_filter()
    edge_case_suite.test_no_entities()

    complex_suite = TestComplexRealWorld()
    complex_suite.test_biomedical_discontinuous()
    complex_suite.test_nested_and_overlapping()

if __name__ == "__main__":
    # Run the complete workflow test
    test_complete_workflow()
    print("\n✅ All integration tests would pass!")

    test_all_integration()