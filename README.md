# nervaluate-disco

**Evaluation for Discontinuous and Overlapping NER Entities**

A clean-break fork of [nervaluate](https://github.com/MantisAI/nervaluate) specifically designed for evaluating Named Entity Recognition (NER) systems that produce **discontinuous entities** and/or **overlapping entities**.

## Key Features

- **Discontinuous entities**: Entities with multiple non-contiguous spans
- **Overlapping entities**: Multiple entities covering the same text
- **Optimal matching**: Uses Hungarian algorithm for globally optimal entity alignment
- **Character-based offsets**: Works with character positions (not token-based)
- **Four evaluation strategies**: Strict, Partial, Entity Type, and Exact
- **Comprehensive metrics**: Precision, recall, F1 with detailed breakdowns

## What Changed from Original nervaluate

This is a **clean-break fork** with fundamental changes:

| Feature | Original nervaluate | nervaluate-disco |
|---------|-------------------|------------------|
| Entity spans | Single continuous span | Multiple discontinuous spans |
| Offsets | Token-based | Character-based |
| Overlapping | Not supported | Fully supported |
| Matching | Greedy 1-to-1 | Optimal bipartite (Hungarian) |
| Format | IOB/CoNLL | Custom span format |

## Installation

```bash
pip install git+https://github.com/coree/nervaluate-disco.git
```

or

```bash
git clone https://github.com/coree/nervaluate-disco.git
cd nervaluate-disco
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

```python
from nervaluate import Evaluator

# Your data format
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
            {'spans': [(0, 10)], 'label': 'PERSON'},
            {'spans': [(20, 30)], 'label': 'ORG'}
        ]
    }
]

# Evaluate
evaluator = Evaluator(true, pred, tags=['PERSON', 'ORG'], loader='span_dict')
results = evaluator.evaluate()

# Print results
print(evaluator.summary_report(mode='overall'))
```

### Discontinuous Entities

```python
true = [
    {
        'text': 'The alpha and beta receptor complex',
        'entities': [
            {
                'spans': [(4, 9), (14, 18)],  # "alpha" and "beta"
                'label': 'PROTEIN',
                'text': 'alpha beta'
            }
        ]
    }
]

pred = [
    {
        'text': 'The alpha and beta receptor complex',
        'entities': [
            {'spans': [(4, 9), (14, 18)], 'label': 'PROTEIN'}
        ]
    }
]

evaluator = Evaluator(true, pred, tags=['PROTEIN'], loader='span_dict')
results = evaluator.evaluate()
print(f"F1 Score: {results['overall']['strict'].f1:.2f}")
```

### Overlapping Entities

```python
true = [
    {
        'text': 'protein kinase activity',
        'entities': [
            {'spans': [(0, 7)], 'label': 'PROTEIN'},
            {'spans': [(0, 14)], 'label': 'ENZYME'},  # Overlaps!
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

evaluator = Evaluator(
    true, pred, 
    tags=['PROTEIN', 'ENZYME', 'FUNCTION'], 
    loader='span_dict'
)
results = evaluator.evaluate()
```

## Evaluation Strategies

### 1. Strict Evaluation
- **Correct**: Same label AND exact same spans
- **Incorrect**: Sufficient overlap but wrong label OR wrong spans
- **Example**: Perfect match required

### 2. Partial Evaluation
- **Correct**: Exact span match (label ignored)
- **Partial**: Sufficient overlap but boundaries differ
- **Note**: No "incorrect" category; uses 0.5 credit for partial matches

### 3. Entity Type Evaluation
- **Correct**: Sufficient overlap AND same label
- **Incorrect**: Sufficient overlap BUT wrong label

### 4. Exact Evaluation
- **Correct**: Exact boundary match (label ignored)
- **Incorrect**: Sufficient overlap but boundaries differ

## Input Format

### Primary Format: `span_dict`

```python
[
    {
        'text': 'Full sentence text',  # Optional but recommended
        'entities': [
            {
                'spans': [(start, end), ...],  # Character offsets (required)
                'label': 'ENTITY_TYPE',        # Required
                'text': 'entity text'           # Optional
            }
        ]
    }
]
```

**Important**: Character offsets are:
- Zero-indexed
- End-exclusive: `text[start:end]`
- Multiple spans for discontinuous entities

### Alternative Format: `simple_span`

```python
[
    [  # Document 1
        {'spans': [(10, 15)], 'label': 'PERSON'},
        {'spans': [(20, 25)], 'label': 'ORG'}
    ],
    [  # Document 2
        {'spans': [(0, 5)], 'label': 'LOC'}
    ]
]
```

## Advanced Options

### Overlap Threshold

Control the minimum overlap percentage required for matching:

```python
evaluator = Evaluator(
    true, pred,
    tags=['PERSON', 'ORG'],
    loader='span_dict',
    min_overlap_percentage=50.0  # 50% minimum overlap
)
```

### Per-Entity Metrics

```python
results = evaluator.evaluate()

# Overall metrics
print(results['overall']['strict'].f1)

# Per-entity metrics
print(results['entities']['PERSON']['strict'].precision)
print(results['entities']['ORG']['partial'].recall)
```

### Report Generation

```python
# Summary report
print(evaluator.summary_report(mode='overall'))
print(evaluator.summary_report(mode='entities', scenario='strict'))

# CSV export
csv_string = evaluator.results_to_csv(mode='overall')
evaluator.results_to_csv(mode='entities', scenario='strict', file_path='results.csv')
```

## How It Works

### 1. Entity Representation

```python
from nervaluate.entities import Entity

# Continuous
entity1 = Entity(label="PERSON", spans=[(10, 15)])

# Discontinuous
entity2 = Entity(label="PROTEIN", spans=[(10, 15), (30, 35)])

# Check properties
print(entity1.is_continuous)  # True
print(entity2.total_chars)    # 10
print(entity2.overlaps_with(entity1))  # False
```

### 2. Overlap Calculation

Character-level overlap percentage based on true entity coverage:

```python
from nervaluate.matching import calculate_overlap_percentage

true = Entity("PER", spans=[(10, 20)])  # 10 characters
pred = Entity("PER", spans=[(10, 15)])  # 5 characters overlap

overlap = calculate_overlap_percentage(pred, true)
print(overlap)  # 50.0
```

### 3. Optimal Matching

Uses the Hungarian algorithm for globally optimal entity alignment:

```python
from nervaluate.matching import optimal_match

true_entities = [Entity("PER", spans=[(0, 10)]), Entity("ORG", spans=[(20, 30)])]
pred_entities = [Entity("PER", spans=[(0, 8)]), Entity("ORG", spans=[(20, 30)])]

matches, matched_true, matched_pred = optimal_match(
    true_entities, 
    pred_entities, 
    threshold=50.0
)

for t_idx, p_idx, overlap, true_ent, pred_ent in matches:
    print(f"True[{t_idx}] ↔ Pred[{p_idx}]: {overlap:.1f}% overlap")
```

## Example Output

```
Scenario: all

              correct   incorrect     partial      missed    spurious   precision      recall    f1-score

ent_type            5           0           0           0           0        1.00        1.00        1.00
   exact            3           2           0           0           0        0.60        0.60        0.60
 partial            3           0           2           0           0        0.80        0.80        0.80
  strict            3           2           0           0           0        0.60        0.60        0.60
```

## Use Cases

### Biomedical NER

```python
# Discontinuous protein names
true = [{
    'text': 'The alpha-2A and beta-3B adrenergic receptors',
    'entities': [
        {'spans': [(4, 12), (17, 24)], 'label': 'PROTEIN'}
    ]
}]
```

### Nested Entities

```python
# Organization contains location
true = [{
    'text': 'New York City Department of Health',
    'entities': [
        {'spans': [(0, 13)], 'label': 'CITY'},
        {'spans': [(0, 34)], 'label': 'ORG'}  # Contains CITY
    ]
}]
```

### Co-reference Resolution

```python
# Same entity mentioned multiple times
true = [{
    'text': 'The CEO resigned. The CEO later returned.',
    'entities': [
        {'spans': [(4, 7), (22, 25)], 'label': 'PERSON'}
    ]
}]
```

## Development

### Running Tests

```bash
# Install dependencies
pip install -r requirements.txt

# Run demo tests
python test_demo.py

# Run full test suite (requires pytest)
pytest tests/
```

### Project Structure

```
nervaluate-disco/
├── src/nervaluate/
│   ├── __init__.py
│   ├── entities.py      # Entity model
│   ├── matching.py      # Optimal matching algorithms
│   ├── loaders.py       # Data loaders
│   ├── strategies.py    # Evaluation strategies
│   └── evaluator.py     # Main evaluator
├── tests/
│   ├── test_entities.py
│   ├── test_matching.py
│   ├── test_loaders.py
│   ├── test_strategies.py
│   └── test_integration.py
├── test_demo.py         # Standalone demo
├── requirements.txt
└── README.md
```

## API Reference

### Evaluator

```python
Evaluator(
    true: List[Dict],
    pred: List[Dict],
    tags: List[str],
    loader: str = 'span_dict',
    min_overlap_percentage: float = 1.0
)
```

**Methods**:
- `evaluate()` → Dict: Run evaluation
- `summary_report(mode='overall', scenario='strict', digits=2)` → str
- `results_to_csv(mode='overall', scenario='strict', file_path=None)` → str

### Entity

```python
Entity(
    label: str,
    spans: List[Tuple[int, int]],
    text: Optional[str] = None
)
```

**Properties**:
- `is_continuous`: bool
- `total_chars`: int
- `get_all_char_positions()`: Set[int]
- `overlaps_with(other)`: bool

## Important Notes

1. **Character offsets**: All spans use character-based offsets (not tokens)
2. **End-exclusive**: Spans are `[start, end)` like Python slicing
3. **Sorted spans**: Spans are automatically sorted by start position
4. **No overlap within entity**: Spans within same entity cannot overlap
5. **Optimal matching**: Uses Hungarian algorithm (may be slower than greedy for large datasets)

## Contributing

This is a specialized fork. For the original nervaluate, see: https://github.com/MantisAI/nervaluate

## License

MIT License (same as original nervaluate)

## Acknowledgments

Based on [nervaluate](https://github.com/MantisAI/nervaluate) by David S. Batista and Matthew A. Upson.

## Support

For issues specific to discontinuous/overlapping entities, please open an issue in this repository.

---

**Version**: 2.0.0  
**Status**: Ready