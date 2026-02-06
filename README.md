# Azerbaijani NLP Processing Tool 

A comprehensive NLP toolkit for processing Azerbaijani text, implementing tokenization, Heaps' Law analysis, Byte Pair Encoding, sentence segmentation, and spell checking with both Levenshtein distance and weighted edit distance using confusion matrices.

## Project Structure

```
.
├── data                                                                     
│   └── all_contents
├── setup.py
├── src
│   ├── __init__.py
│   ├── data_loader.py              # Data loading and preprocessing  
│   ├── tokenization.py             # Task 1: Tokenization and frequency analysis
│   ├── heaps_law.py                # Task 2: Heaps' Law analysis
│   ├── bpe.py                      # Task 3: Byte Pair Encoding
│   ├── sentence_segmentation.py    # Task 4: Sentence segmentation
│   ├── spell_checking.py           # Task 5: Spell checking (Levenshtein)
│   └── extra_weighted.py           # Extra: Weighted edit distance
├── test
│   ├── compare_spell_checkers.py
│   ├── CONFUSION_MATRIX_GUIDE.md
│   ├── confusion_matrix.png
│   ├── generate_confusion_matrix.py
│   ├── REPORT.md
│   ├── test_data.json
│   ├── TESTING_GUIDE.md
│   ├── TESTING_QUICK_REF.md
│   ├── test_results
│   │   ├── evaluation_report.txt
│   │   └── evaluation_results.json
│   └── test_spell_checker.py
├── main.py                         # Main execution script
├── streamlit_app.py                # Streamlit UI application
├── requirements.txt                # Python dependencies
├── outputs
│   ├── bpe_model.json
│   ├── bpe_model.pkl
│   ├── extra_task_report.txt
│   ├── heaps_law_params.json
│   ├── heaps_law_plot.png
│   ├── sentence_segmenter.pkl
│   ├── sentences.json
│   ├── spell_checker.json
│   ├── spell_checker.pkl
│   ├── spell_test_pairs.json
│   ├── summary.json
│   ├── task2_report.txt
│   ├── task3_report.txt
│   ├── task4_report.txt
│   ├── task5_report.txt
│   ├── vocabulary.json
│   ├── vocabulary.pkl
│   ├── weighted_spell_checker_confusion.json
│   ├── weighted_spell_checker_confusion.pkl
│   └── weighted_spell_checker.pkl
└── README.md                       # This file
```

## Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Your Data

Place your Azerbaijani JSON files in a folder. Each JSON file should contain a list of dictionaries with `page_number` and `content` keys:

```json
[
  {
    "page_number": 1,
    "content": "<text>"
  },
  {
    "page_number": 2,
    "content": "<text>"
  }
]
```

### 3. Run All Tasks

**With your own data:**
```bash
python main.py --data /path/to/your/data --output ./outputs
```

**With sample data (for testing):**
```bash
python main.py --sample --output ./outputs
```

### 4. Launch the UI

```bash
streamlit run streamlit_app.py
```

The UI will open in your browser at `http://localhost:8501`

## Tasks Overview

### Task 1: Tokenization & Frequency Analysis (15%)
- Tokenizes Azerbaijani text
- Calculates token and type frequencies
- Computes type-token ratio
- Outputs:
  - `vocabulary.pkl` - Saved vocabulary
  - `vocabulary.json` - Top 100 tokens with frequencies
  - `task1_report.txt` - Detailed statistics

### Task 2: Heaps' Law Analysis (15%)
- Tests Heaps' Law: V(n) = k × n^β
- Finds optimal k and β parameters
- Generates vocabulary growth visualization
- Outputs:
  - `heaps_law_params.json` - k, β, R² values
  - `heaps_law_plot.png` - Visualization
  - `task2_report.txt` - Analysis report

### Task 3: Byte Pair Encoding (15%)
- Implements BPE algorithm for subword tokenization
- Learns merge operations from corpus
- Supports encoding/decoding
- Outputs:
  - `bpe_model.pkl` - Trained BPE model
  - `bpe_model.json` - Model statistics
  - `task3_report.txt` - Sample encodings

### Task 4: Sentence Segmentation (15%)
- Azerbaijani-aware sentence boundary detection
- Handles abbreviations and titles
- Supports complex punctuation
- Outputs:
  - `sentence_segmenter.pkl` - Segmenter model
  - `sentences.json` - All segmented sentences
  - `task4_report.txt` - Statistics

### Task 5: Spell Checker - Levenshtein Distance (15%)
- Classic spell checking using edit distance
- Provides top-N correction suggestions
- Ranks by distance and frequency
- Outputs:
  - `spell_checker.pkl` - Spell checker model
  - `spell_checker.json` - Vocabulary statistics
  - `spell_test_pairs.json` - Test cases
  - `task5_report.txt` - Evaluation results

### Extra Task: Weighted Edit Distance (20%)
- Advanced spell checking with character confusion matrix
- Learns common character substitution patterns
- Lower weights for common confusions (ə↔a, ı↔i, etc.)
- Outputs:
  - `weighted_spell_checker.pkl` - Weighted model
  - `weighted_spell_checker_confusion.pkl` - Confusion matrix
  - `extra_task_report.txt` - Confusion patterns

## Streamlit UI Features

The interactive UI provides:

### Overview Page
- Dataset statistics
- Summary of all task results
- Quick metrics dashboard

### Task 1: Tokenization
- Frequency distribution visualization
- Top tokens chart
- Interactive tokenization testing

### Task 2: Heaps' Law
- Formula display: V(n) = k × n^β
- Parameter values (k, β, R²)
- Vocabulary growth plots
- Interpretation of β value

### Task 3: BPE
- Sample merge operations
- Vocabulary statistics
- Interactive encoding/decoding
- Token count comparison

### Task 4: Segmentation
- Sample segmented sentences
- Interactive sentence boundary detection
- Handles abbreviations correctly

### Task 5: Spell Checker
- Interactive spell checking
- Top-N suggestions with distances
- Batch text correction
- Edit distance configuration

### Extra: Weighted Edit Distance
- Character confusion matrix display
- Weighted vs. regular comparison
- Interactive weighted spell checking
- Confusion pattern analysis

## Output Files

All outputs are saved to `./outputs/` (or your specified directory):

```
outputs/
├── vocabulary.pkl                      # Task 1
├── vocabulary.json
├── task1_report.txt
├── heaps_law_params.json              # Task 2
├── heaps_law_plot.png
├── task2_report.txt
├── bpe_model.pkl                      # Task 3
├── bpe_model.json
├── task3_report.txt
├── sentence_segmenter.pkl             # Task 4
├── sentences.json
├── task4_report.txt
├── spell_checker.pkl                  # Task 5
├── spell_checker.json
├── spell_test_pairs.json
├── task5_report.txt
├── weighted_spell_checker.pkl         # Extra Task
├── weighted_spell_checker_confusion.pkl
├── weighted_spell_checker_confusion.json
├── extra_task_report.txt
└── summary.json                       # Overall summary
```

## Module Usage Examples

### Using Tokenizer
```python
from task1_tokenization import Tokenizer

tokenizer = Tokenizer()
stats = tokenizer.process_corpus("Azərbaycan dili")
print(f"Tokens: {stats['total_tokens']}")
print(f"Types: {stats['total_types']}")
```

### Using BPE
```python
from task3_bpe import BPETokenizer

bpe = BPETokenizer(num_merges=500)
bpe.train(corpus)

encoded = bpe.encode("Azərbaycan")
decoded = bpe.decode(encoded)
print(f"Encoded: {encoded}")
```

### Using Spell Checker
```python
from task5_spell_checker import SpellChecker

checker = SpellChecker()
checker.build_vocabulary(corpus)

suggestions = checker.correct_word("azrbaycan", max_distance=2)
for word, dist, freq in suggestions:
    print(f"{word} (distance: {dist}, frequency: {freq})")
```

### Using Weighted Spell Checker
```python
from task_extra_weighted import WeightedSpellChecker, ConfusionMatrix

confusion = ConfusionMatrix()
checker = WeightedSpellChecker(confusion_matrix=confusion)
checker.build_vocabulary(corpus)

suggestions = checker.correct_word("azarbaycan", max_distance=2.0)
```

## Command Line Options

### Main Pipeline
```bash
python main.py --help
```

Options:
- `--data PATH`: Path to data folder containing JSON files
- `--output PATH`: Path to output folder (default: ./outputs)
- `--sample`: Create and use sample Azerbaijani data for testing

### Streamlit UI
```bash
streamlit run streamlit_app.py
```

## Report Generation

The tool automatically generates a comprehensive report (Task 6 - 20%) containing:

1. **Dataset Statistics**: Files, pages, characters, lines
2. **Task 1 Results**: Tokenization statistics, frequency distributions
3. **Task 2 Results**: Heaps' Law parameters, vocabulary growth analysis
4. **Task 3 Results**: BPE statistics, merge operations
5. **Task 4 Results**: Sentence segmentation statistics
6. **Task 5 Results**: Spell checker accuracy, test cases
7. **Extra Task Results**: Confusion matrix patterns, weighted accuracy

All reports are saved as:
- Individual task reports: `task{N}_report.txt`
- Overall summary: `summary.json`

## Key Features

- **Azerbaijani Language Support**: Handles special characters (ə, ı, ö, ü, ğ, ş, ç)
- **Modular Design**: Each task is a separate, reusable module
- **Persistent Models**: All models can be saved and loaded
- **Interactive UI**: Beautiful Streamlit interface for testing
- **Comprehensive Testing**: Automatic generation of test cases
- **Detailed Reports**: Text and JSON format reports
- **Visualizations**: Matplotlib plots for Heaps' Law

## Technical Details

### Tokenization
- Pattern-based tokenization using regex
- Preserves Azerbaijani characters
- Case-insensitive processing

### Heaps' Law
- Non-linear curve fitting using scipy
- Logarithmic sampling for efficiency
- R² goodness of fit calculation

### BPE
- Classic BPE algorithm implementation
- Character-level initialization
- End-of-word markers (</w>)

### Sentence Segmentation
- Rule-based segmentation
- Azerbaijani abbreviation handling
- Advanced punctuation support

### Spell Checking
- Dynamic programming Levenshtein distance
- Frequency-based ranking
- Configurable edit distance threshold

### Weighted Edit Distance
- Character confusion matrix learning
- Weighted dynamic programming
- Azerbaijani-specific confusion patterns

## Troubleshooting

**No models found in UI:**
```bash
# Run the main pipeline first
python main.py --sample --output ./outputs
```

**Module import errors:**
```bash
# Make sure all files are in the same directory
# Install dependencies
pip install -r requirements.txt
```

**Encoding errors with Azerbaijani characters:**
```bash
# Ensure your JSON files are UTF-8 encoded
# Python 3.6+ should handle this automatically
```

## Dependencies

- Python 3.7+
- numpy >= 1.21.0
- scipy >= 1.7.0
- matplotlib >= 3.4.0
- streamlit >= 1.28.0
- pandas >= 1.3.0

## Acknowledgments

- Azerbaijani language resources, mainly Presidential Library
- BPE algorithm by Sennrich et al.
- Heaps' Law by Harold Stanley Heaps
