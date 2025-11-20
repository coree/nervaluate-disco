from abc import ABC, abstractmethod
from typing import List, Dict, Any

from .entities import Entity


class DataLoader(ABC):
    """Abstract base class for data loaders."""

    @abstractmethod
    def load(self, data: Any) -> List[List[Entity]]:
        """Load data into a list of entity lists."""


class SpanDictLoader(DataLoader):
    """
    Loader for dictionary format with discontinuous entities.
    
    Expected format:
    [
        {
            'text': 'The full document text',
            'entities': [
                {
                    'spans': [(start_char, end_char), ...],  # character offsets
                    'label': 'LABEL_NAME',
                    'text': 'optional entity text'
                },
                ...
            ]
        },
        ...
    ]
    """

    def load(self, data: List[Dict[str, Any]]) -> List[List[Entity]]:
        """Load span dictionary format data into a list of Entity lists."""
        if not isinstance(data, list):
            raise ValueError("SpanDictLoader expects list input")

        if not data:
            return []

        result = []

        for doc_idx, doc in enumerate(data):
            if not isinstance(doc, dict):
                raise ValueError(f"Document {doc_idx} must be a dictionary")

            if 'entities' not in doc:
                raise ValueError(f"Document {doc_idx} missing 'entities' key")

            entities = []
            for ent_idx, ent in enumerate(doc['entities']):
                if not isinstance(ent, dict):
                    raise ValueError(f"Document {doc_idx}, entity {ent_idx}: must be a dictionary")

                # Validate required keys
                if 'label' not in ent:
                    raise ValueError(f"Document {doc_idx}, entity {ent_idx}: missing 'label' key")
                if 'spans' not in ent:
                    raise ValueError(f"Document {doc_idx}, entity {ent_idx}: missing 'spans' key")

                # Validate spans format
                if not isinstance(ent['spans'], list) or not ent['spans']:
                    raise ValueError(f"Document {doc_idx}, entity {ent_idx}: 'spans' must be a non-empty list")

                for span_idx, span in enumerate(ent['spans']):
                    if not isinstance(span, (list, tuple)) or len(span) != 2:
                        raise ValueError(
                            f"Document {doc_idx}, entity {ent_idx}, span {span_idx}: "
                            f"must be a tuple/list of (start, end)"
                        )
                    if not isinstance(span[0], int) or not isinstance(span[1], int):
                        raise ValueError(
                            f"Document {doc_idx}, entity {ent_idx}, span {span_idx}: "
                            f"start and end must be integers"
                        )

                # Create entity
                try:
                    entity = Entity(
                        label=ent['label'],
                        spans=[(s[0], s[1]) for s in ent['spans']],
                        text=ent.get('text')
                    )
                    entities.append(entity)
                except ValueError as e:
                    raise ValueError(
                        f"Document {doc_idx}, entity {ent_idx}: Invalid entity - {str(e)}"
                    )

            result.append(entities)

        return result


class SimpleSpanLoader(DataLoader):
    """
    Loader for simplified span format (list of entity lists).
    
    Expected format:
    [
        [  # Document 1
            {'label': 'PER', 'spans': [(0, 5)]},
            {'label': 'ORG', 'spans': [(10, 15), (20, 25)]},
            ...
        ],
        [  # Document 2
            ...
        ]
    ]
    """

    def load(self, data: List[List[Dict[str, Any]]]) -> List[List[Entity]]:
        """Load simple span format data into a list of Entity lists."""
        if not isinstance(data, list):
            raise ValueError("SimpleSpanLoader expects list input")

        if not data:
            return []

        result = []

        for doc_idx, doc in enumerate(data):
            if not isinstance(doc, list):
                raise ValueError(f"Document {doc_idx} must be a list")

            entities = []
            for ent_idx, ent in enumerate(doc):
                if not isinstance(ent, dict):
                    raise ValueError(f"Document {doc_idx}, entity {ent_idx}: must be a dictionary")

                required_keys = {'label', 'spans'}
                if not all(key in ent for key in required_keys):
                    raise ValueError(
                        f"Document {doc_idx}, entity {ent_idx}: missing required keys {required_keys}"
                    )

                try:
                    entity = Entity(
                        label=ent['label'],
                        spans=[(s[0], s[1]) for s in ent['spans']],
                        text=ent.get('text')
                    )
                    entities.append(entity)
                except (ValueError, TypeError, IndexError) as e:
                    raise ValueError(
                        f"Document {doc_idx}, entity {ent_idx}: Invalid entity - {str(e)}"
                    )

            result.append(entities)

        return result