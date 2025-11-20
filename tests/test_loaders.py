"""Tests for data loaders."""

import pytest
from nervaluate.loaders import SpanDictLoader, SimpleSpanLoader
from nervaluate.entities import Entity


class TestSpanDictLoader:
    """Tests for SpanDictLoader."""

    def test_load_single_document_continuous(self):
        """Test loading single document with continuous entity."""
        data = [
            {
                'text': 'John Smith works at Google.',
                'entities': [
                    {
                        'spans': [(0, 10)],
                        'label': 'PERSON',
                        'text': 'John Smith'
                    }
                ]
            }
        ]

        loader = SpanDictLoader()
        result = loader.load(data)

        assert len(result) == 1
        assert len(result[0]) == 1
        assert result[0][0].label == 'PERSON'
        assert result[0][0].spans == [(0, 10)]
        assert result[0][0].text == 'John Smith'

    def test_load_discontinuous_entity(self):
        """Test loading discontinuous entity."""
        data = [
            {
                'text': 'The CEO resigned and later returned.',
                'entities': [
                    {
                        'spans': [(4, 7), (27, 35)],
                        'label': 'PERSON',
                        'text': 'CEO returned'
                    }
                ]
            }
        ]

        loader = SpanDictLoader()
        result = loader.load(data)

        assert len(result) == 1
        assert len(result[0]) == 1
        entity = result[0][0]
        assert entity.label == 'PERSON'
        assert entity.spans == [(4, 7), (27, 35)]
        assert entity.is_continuous is False

    def test_load_multiple_documents(self):
        """Test loading multiple documents."""
        data = [
            {
                'text': 'First document.',
                'entities': [
                    {'spans': [(0, 5)], 'label': 'WORD'}
                ]
            },
            {
                'text': 'Second document.',
                'entities': [
                    {'spans': [(0, 6)], 'label': 'WORD'}
                ]
            }
        ]

        loader = SpanDictLoader()
        result = loader.load(data)

        assert len(result) == 2
        assert result[0][0].label == 'WORD'
        assert result[1][0].label == 'WORD'

    def test_load_document_no_entities(self):
        """Test loading document with no entities."""
        data = [
            {
                'text': 'No entities here.',
                'entities': []
            }
        ]

        loader = SpanDictLoader()
        result = loader.load(data)

        assert len(result) == 1
        assert len(result[0]) == 0

    def test_load_empty_data(self):
        """Test loading empty data."""
        loader = SpanDictLoader()
        result = loader.load([])

        assert len(result) == 0

    def test_entity_without_text_field(self):
        """Test loading entity without optional text field."""
        data = [
            {
                'text': 'Some text.',
                'entities': [
                    {'spans': [(0, 4)], 'label': 'TEST'}
                ]
            }
        ]

        loader = SpanDictLoader()
        result = loader.load(data)

        assert result[0][0].text is None

    def test_overlapping_entities_same_document(self):
        """Test loading overlapping entities in same document."""
        data = [
            {
                'text': 'alpha beta receptor',
                'entities': [
                    {'spans': [(0, 10)], 'label': 'PROTEIN'},
                    {'spans': [(6, 19)], 'label': 'GENE'}  # Overlaps with PROTEIN
                ]
            }
        ]

        loader = SpanDictLoader()
        result = loader.load(data)

        assert len(result) == 1
        assert len(result[0]) == 2
        assert result[0][0].overlaps_with(result[0][1]) is True

    def test_invalid_not_list(self):
        """Test that non-list input raises ValueError."""
        loader = SpanDictLoader()
        with pytest.raises(ValueError, match="expects list input"):
            loader.load("not a list")

    def test_invalid_document_not_dict(self):
        """Test that non-dict document raises ValueError."""
        loader = SpanDictLoader()
        with pytest.raises(ValueError, match="must be a dictionary"):
            loader.load(["not a dict"])

    def test_invalid_missing_entities_key(self):
        """Test that missing 'entities' key raises ValueError."""
        loader = SpanDictLoader()
        data = [{'text': 'Some text'}]
        with pytest.raises(ValueError, match="missing 'entities' key"):
            loader.load(data)

    def test_invalid_entity_missing_label(self):
        """Test that missing 'label' raises ValueError."""
        loader = SpanDictLoader()
        data = [
            {
                'text': 'Text',
                'entities': [{'spans': [(0, 4)]}]
            }
        ]
        with pytest.raises(ValueError, match="missing 'label'"):
            loader.load(data)

    def test_invalid_entity_missing_spans(self):
        """Test that missing 'spans' raises ValueError."""
        loader = SpanDictLoader()
        data = [
            {
                'text': 'Text',
                'entities': [{'label': 'TEST'}]
            }
        ]
        with pytest.raises(ValueError, match="missing 'spans'"):
            loader.load(data)

    def test_invalid_spans_empty(self):
        """Test that empty spans list raises ValueError."""
        loader = SpanDictLoader()
        data = [
            {
                'text': 'Text',
                'entities': [{'label': 'TEST', 'spans': []}]
            }
        ]
        with pytest.raises(ValueError, match="must be a non-empty list"):
            loader.load(data)

    def test_invalid_span_format(self):
        """Test that invalid span format raises ValueError."""
        loader = SpanDictLoader()
        data = [
            {
                'text': 'Text',
                'entities': [{'label': 'TEST', 'spans': [(0,)]}]  # Only one value
            }
        ]
        with pytest.raises(ValueError, match="must be a tuple/list of"):
            loader.load(data)

    def test_invalid_span_not_integers(self):
        """Test that non-integer spans raise ValueError."""
        loader = SpanDictLoader()
        data = [
            {
                'text': 'Text',
                'entities': [{'label': 'TEST', 'spans': [("0", "4")]}]
            }
        ]
        with pytest.raises(ValueError, match="start and end must be integers"):
            loader.load(data)

    def test_unsorted_spans_are_sorted(self):
        """Test that unsorted spans are automatically sorted."""
        data = [
            {
                'text': 'Text',
                'entities': [
                    {'label': 'TEST', 'spans': [(50, 60), (10, 20), (30, 40)]}
                ]
            }
        ]

        loader = SpanDictLoader()
        result = loader.load(data)

        assert result[0][0].spans == [(10, 20), (30, 40), (50, 60)]


class TestSimpleSpanLoader:
    """Tests for SimpleSpanLoader."""

    def test_load_simple_format(self):
        """Test loading simple format without document text."""
        data = [
            [  # Document 1
                {'spans': [(10, 15)], 'label': 'PERSON'},
                {'spans': [(20, 25)], 'label': 'ORG'}
            ],
            [  # Document 2
                {'spans': [(0, 5)], 'label': 'LOC'}
            ]
        ]

        loader = SimpleSpanLoader()
        result = loader.load(data)

        assert len(result) == 2
        assert len(result[0]) == 2
        assert len(result[1]) == 1
        assert result[0][0].label == 'PERSON'
        assert result[0][1].label == 'ORG'
        assert result[1][0].label == 'LOC'

    def test_load_discontinuous_simple(self):
        """Test loading discontinuous entity in simple format."""
        data = [
            [
                {'spans': [(10, 15), (30, 35)], 'label': 'PROTEIN'}
            ]
        ]

        loader = SimpleSpanLoader()
        result = loader.load(data)

        assert len(result) == 1
        assert len(result[0]) == 1
        assert result[0][0].is_continuous is False
        assert len(result[0][0].spans) == 2

    def test_load_empty_document_simple(self):
        """Test loading empty document in simple format."""
        data = [[]]

        loader = SimpleSpanLoader()
        result = loader.load(data)

        assert len(result) == 1
        assert len(result[0]) == 0

    def test_load_empty_data_simple(self):
        """Test loading empty data in simple format."""
        loader = SimpleSpanLoader()
        result = loader.load([])

        assert len(result) == 0

    def test_invalid_not_list_simple(self):
        """Test that non-list input raises ValueError."""
        loader = SimpleSpanLoader()
        with pytest.raises(ValueError, match="expects list input"):
            loader.load("not a list")

    def test_invalid_document_not_list_simple(self):
        """Test that non-list document raises ValueError."""
        loader = SimpleSpanLoader()
        with pytest.raises(ValueError, match="must be a list"):
            loader.load(["not a list"])

    def test_invalid_entity_not_dict_simple(self):
        """Test that non-dict entity raises ValueError."""
        loader = SimpleSpanLoader()
        with pytest.raises(ValueError, match="must be a dictionary"):
            loader.load([["not a dict"]])

    def test_invalid_missing_keys_simple(self):
        """Test that missing required keys raise ValueError."""
        loader = SimpleSpanLoader()
        data = [[{'label': 'TEST'}]]  # Missing spans
        with pytest.raises(ValueError, match="missing required keys"):
            loader.load(data)



if __name__ == "__main__":
    def test_all():
        # Init all classes to satisfy coverage checks
        SpanDictLoader()
        SimpleSpanLoader()
        test_span_dict_loader = TestSpanDictLoader()
        test_span_dict_loader.test_load_single_document_continuous()
        test_span_dict_loader.test_load_discontinuous_entity()          
        test_span_dict_loader.test_load_multiple_documents()
        test_span_dict_loader.test_load_document_no_entities()
        test_span_dict_loader.test_load_empty_data()
        test_span_dict_loader.test_entity_without_text_field()
        test_span_dict_loader.test_overlapping_entities_same_document()
        test_span_dict_loader.test_invalid_not_list()
        test_span_dict_loader.test_invalid_document_not_dict()
        test_span_dict_loader.test_invalid_missing_entities_key()
        test_span_dict_loader.test_invalid_entity_missing_label()
        test_span_dict_loader.test_invalid_entity_missing_spans()
        test_span_dict_loader.test_invalid_spans_empty()
        test_span_dict_loader.test_invalid_span_format()
        test_span_dict_loader.test_invalid_span_not_integers()
        test_span_dict_loader.test_unsorted_spans_are_sorted()
        test_simple_span_loader = TestSimpleSpanLoader()
        test_simple_span_loader.test_load_simple_format()
        test_simple_span_loader.test_load_discontinuous_simple()
        test_simple_span_loader.test_load_empty_document_simple()
        test_simple_span_loader.test_load_empty_data_simple()
        test_simple_span_loader.test_invalid_not_list_simple()
        test_simple_span_loader.test_invalid_document_not_list_simple()
        test_simple_span_loader.test_invalid_entity_not_dict_simple()
        test_simple_span_loader.test_invalid_missing_keys_simple()
    test_all()        
    print("All loader tests passed.")
    