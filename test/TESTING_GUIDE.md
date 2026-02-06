# Spell Checker Testing Guide

## Overview

This package includes comprehensive testing tools for evaluating your trained spell checkers on custom test datasets.

## Files Included

1. **test_spell_checker.py** - Evaluate a single spell checker
2. **compare_spell_checkers.py** - Compare regular vs weighted spell checkers
3. **sample_test_data.json** - Example test data format

## Test Data Format

Test data should be a JSON file with the following structure:

```json
{
  "test_cases": [
    {
      "id": 1,
      "correct": "Sizinlə tanış olduğuma çox şadam.",
      "typo": "Sizinle tanis olduguma cox sadam."
    },
    {
      "id": 2,
      "correct": "Azərbaycan Respublikası Cənubi Qafqazda yerləşir.",
      "typo": "Azarbaycan Respublikasi Canubi Qafqazda yerlasir."
    }
  ]
}
```

**Fields:**
- `id` - Unique identifier for the test case
- `correct` - The correctly spelled sentence
- `typo` - The sentence with typos/errors

## Usage

### Option 1: Test Single Spell Checker

Test the regular (Levenshtein distance) spell checker:

```bash
python test_spell_checker.py \
    --test-data your_test_data.json \
    --model ./outputs/spell_checker.pkl \
    --output ./test_results
```

Test the weighted (confusion matrix) spell checker:

```bash
python test_spell_checker.py \
    --test-data your_test_data.json \
    --model ./outputs/weighted_spell_checker.pkl \
    --weighted \
    --output ./test_results
```

**Parameters:**
- `--test-data` - Path to your test JSON file (required)
- `--model` - Path to spell checker model (default: ./outputs/spell_checker.pkl)
- `--weighted` - Use weighted spell checker instead of regular
- `--max-distance` - Maximum edit distance for corrections (default: 2.0)
- `--output` - Output directory (default: ./test_results)
- `--examples` - Number of examples to display (default: 10)

### Option 2: Compare Both Spell Checkers

Compare regular and weighted spell checkers side-by-side:

```bash
python compare_spell_checkers.py \
    --test-data your_test_data.json \
    --output ./comparison_results
```

**Parameters:**
- `--test-data` - Path to your test JSON file (required)
- `--regular-model` - Path to regular spell checker (default: ./outputs/spell_checker.pkl)
- `--weighted-model` - Path to weighted spell checker (default: ./outputs/weighted_spell_checker.pkl)
- `--max-distance` - Maximum edit distance (default: 2.0)
- `--output` - Output directory (default: ./comparison_results)

## Output Files

### Single Spell Checker Testing

After running `test_spell_checker.py`, you'll get:

```
test_results/
├── evaluation_results.json    # Detailed results in JSON format
└── evaluation_report.txt       # Human-readable text report
```

**evaluation_results.json** contains:
- Overall statistics (accuracy, precision, recall, F1)
- Individual test case results
- Word-level corrections
- Detailed metrics

**evaluation_report.txt** contains:
- Summary statistics
- Performance metrics
- All test cases with corrections

### Comparison Testing

After running `compare_spell_checkers.py`, you'll get:

```
comparison_results/
├── regular_checker_results.json      # Regular checker results
├── regular_checker_report.txt        # Regular checker report
├── weighted_checker_results.json     # Weighted checker results
├── weighted_checker_report.txt       # Weighted checker report
└── comparison_results.json           # Side-by-side comparison
```

## Metrics Explained

### Accuracy
Percentage of words that were corrected correctly out of all corrected words.

Formula: `Correct Corrections / Total Corrected Words`

### Precision
How many of the spell checker's corrections are actually correct.

Formula: `Correct Corrections / Total Corrected Words`

### Recall
How many of the actual errors were successfully corrected.

Formula: `Correct Corrections / Total Errors in Test Data`

### F1 Score
Harmonic mean of precision and recall.

Formula: `2 × (Precision × Recall) / (Precision + Recall)`

## Understanding Results

### Example Output

```
Test Case #1:
  Typo:      Sizinle tanis olduguma cox sadam.
  Corrected: sizinle tanis olduguma çox sadam
  Expected:  Sizinlə tanış olduğuma çox şadam.
  Accuracy:  100.00%
  Word corrections:
    ✓ 'cox' → 'çox' (expected: 'çox')
    ✗ 'tanis' → 'tanis' (expected: 'tanış')
```

**Symbols:**
- ✓ = Correction is correct
- ✗ = Correction is incorrect or word wasn't corrected

### Performance Interpretation

**High Accuracy (>80%)**: Spell checker works well on your data
**Medium Accuracy (50-80%)**: Spell checker needs improvement or larger vocabulary
**Low Accuracy (<50%)**: Spell checker vocabulary may be too small or data is very different

## Tips for Better Results

### 1. Ensure Vocabulary Coverage

The spell checker can only correct to words in its vocabulary. Make sure you train on a corpus similar to your test data.

```bash
# Train on more data
python main.py --data /path/to/larger/corpus --output ./outputs
```

### 2. Adjust Edit Distance

If words aren't being corrected, try increasing max distance:

```bash
python test_spell_checker.py \
    --test-data your_data.json \
    --max-distance 3.0
```

### 3. Use Weighted Checker for Character Confusions

If your errors involve common character substitutions (ə↔a, ı↔i), use the weighted checker:

```bash
python test_spell_checker.py \
    --test-data your_data.json \
    --model ./outputs/weighted_spell_checker.pkl \
    --weighted
```

### 4. Analyze Failed Cases

Look at the detailed results to understand what corrections failed:

```python
import json

with open('test_results/evaluation_results.json', 'r') as f:
    results = json.load(f)

# Find cases with low accuracy
for case in results['test_results']:
    if case['accuracy'] is not None and case['accuracy'] < 0.5:
        print(f"Low accuracy case #{case['id']}")
        print(f"  Expected: {case['expected']}")
        print(f"  Got: {case['corrected']}")
```

## Advanced Usage

### Test on Subset of Data

```python
import json

# Load full test data
with open('full_test_data.json', 'r') as f:
    data = json.load(f)

# Take first 50 cases
subset = {
    'test_cases': data['test_cases'][:50]
}

# Save subset
with open('subset_test_data.json', 'w') as f:
    json.dump(subset, f, ensure_ascii=False, indent=2)

# Test on subset
# python test_spell_checker.py --test-data subset_test_data.json
```

### Batch Testing on Multiple Files

```bash
#!/bin/bash
# test_all.sh

for file in test_data/*.json; do
    echo "Testing $file..."
    python test_spell_checker.py \
        --test-data "$file" \
        --output "./results/$(basename $file .json)"
done
```

### Generate HTML Report

```python
# After running evaluation
import json

with open('test_results/evaluation_results.json', 'r') as f:
    results = json.load(f)

stats = results['overall_statistics']

html = f"""
<html>
<head><title>Spell Checker Results</title></head>
<body>
    <h1>Spell Checker Evaluation</h1>
    <h2>Overall Statistics</h2>
    <table border="1">
        <tr><td>Accuracy</td><td>{stats['accuracy']:.2%}</td></tr>
        <tr><td>Precision</td><td>{stats['precision']:.2%}</td></tr>
        <tr><td>Recall</td><td>{stats['recall']:.2%}</td></tr>
        <tr><td>F1 Score</td><td>{stats['f1_score']:.4f}</td></tr>
    </table>
</body>
</html>
"""

with open('results.html', 'w') as f:
    f.write(html)
```

## Sample Test Data

The package includes `sample_test_data.json` with 5 test cases demonstrating common Azerbaijani typos:

- Character substitutions (ə→a, ı→i)
- Missing diacritics (ş→s, ç→c)
- Incorrect vowels (o→u, e→a)

Use this to verify your setup works:

```bash
python test_spell_checker.py \
    --test-data sample_test_data.json \
    --output ./sample_results
```

## Troubleshooting

### "Model not found"
```bash
# Make sure you've trained the model first
python main.py --sample --output ./outputs
```

### "No test cases found"
Check your JSON format. Make sure you have:
```json
{
  "test_cases": [...]
}
```

### "Vocabulary too small"
Train on more data:
```bash
python main.py --data /path/to/more/data --output ./outputs
```

### Low accuracy results
- Check if test data words are in training vocabulary
- Increase max-distance parameter
- Use weighted checker for character confusions
- Train on domain-specific data

## Example Workflow

1. **Prepare test data**
   ```bash
   # Format your test cases as JSON
   vim my_test_data.json
   ```

2. **Test regular spell checker**
   ```bash
   python test_spell_checker.py \
       --test-data my_test_data.json \
       --output ./results_regular
   ```

3. **Test weighted spell checker**
   ```bash
   python test_spell_checker.py \
       --test-data my_test_data.json \
       --model ./outputs/weighted_spell_checker.pkl \
       --weighted \
       --output ./results_weighted
   ```

4. **Compare both**
   ```bash
   python compare_spell_checkers.py \
       --test-data my_test_data.json \
       --output ./comparison
   ```

5. **Review results**
   ```bash
   cat ./comparison/comparison_results.json
   ```

## Expected Performance

Based on the sample data:

**Small vocabulary (trained on sample data):**
- Accuracy: 50-70%
- Works well for common words
- May struggle with rare words

**Large vocabulary (trained on full corpus):**
- Accuracy: 70-90%
- Better coverage
- More reliable corrections

**Weighted vs Regular:**
- Weighted typically 5-15% better for character confusion errors
- Regular better for general typos
- Use weighted for Azerbaijani-specific errors

## Quick Commands

```bash
# Basic test
python test_spell_checker.py --test-data your_data.json

# Test with weighted checker
python test_spell_checker.py --test-data your_data.json --weighted --model ./outputs/weighted_spell_checker.pkl

# Compare both checkers
python compare_spell_checkers.py --test-data your_data.json

# Test with higher edit distance
python test_spell_checker.py --test-data your_data.json --max-distance 3

# Show more examples
python test_spell_checker.py --test-data your_data.json --examples 20
```

