# Azerbaijani NLP Processing Tool - Project Overview

## Complete Package Contents

This is a **production-ready, modular NLP toolkit** for Azerbaijani text processing. All tasks from your assignment are implemented with:

 Separate Python modules for each task  
 Saved models that can be loaded for testing  
 Interactive Streamlit UI  
 Comprehensive reports  
 Full documentation  

---

## File Inventory

### Python Modules (9 files)
```
data_loader.py                    - Load and preprocess JSON data
task1_tokenization.py             - Task 1: Tokenization (15%)
task2_heaps_law.py                - Task 2: Heaps' Law (15%)
task3_bpe.py                      - Task 3: BPE (15%)
task4_sentence_segmentation.py    - Task 4: Segmentation (15%)
task5_spell_checker.py            - Task 5: Spell checker (15%)
task_extra_weighted.py            - Extra: Weighted distance (20%)
main.py                           - Execute all tasks
streamlit_app.py                  - Interactive UI
```

### Documentation (3 files)
```
README.md                         - Complete technical documentation
USAGE_GUIDE.md                    - Quick start guide
requirements.txt                  - Python dependencies
```

### Sample Outputs (19 files)
```
sample_outputs/                   - Results from running on sample data
  ├── Models (.pkl files)         - 7 trained models
  ├── Reports (.txt files)        - 5 task reports
  ├── Data (.json files)          - 6 data/config files
  └── Visualization (.png)        - 1 Heaps' Law plot
```

**Total: 31 files** providing a complete, working NLP system.

---

## Assignment Requirements Coverage

### Task 1: Tokenization & Frequency (15%)
**Implementation:** `task1_tokenization.py`
- Tokenizes Azerbaijani text (handles ə, ı, ö, ü, ğ, ş, ç)
- Calculates token count, type count, frequencies
- Type-token ratio computation
- Top N frequency analysis

**Outputs:**
- `vocabulary.pkl` - Loadable vocabulary model
- `vocabulary.json` - Top 100 tokens
- `task1_report.txt` - Statistics report

**Testing:** Streamlit UI "Task 1" page with interactive tokenization

---

### Task 2: Heaps' Law k and β (15%)
**Implementation:** `task2_heaps_law.py`
- Tests Heaps' Law: V(n) = k × n^β
- Non-linear curve fitting (scipy)
- Computes k, β, R² values
- Vocabulary growth visualization

**Outputs:**
- `heaps_law_params.json` - k, β, R² values
- `heaps_law_plot.png` - Linear and log-log plots
- `task2_report.txt` - Analysis with interpretation

**Testing:** Streamlit UI "Task 2" page shows parameters and plots

---

### Task 3: Byte Pair Encoding (15%)
**Implementation:** `task3_bpe.py`
- Full BPE algorithm from scratch
- Learns merge operations from corpus
- Encode/decode functionality
- Configurable number of merges

**Outputs:**
- `bpe_model.pkl` - Trained BPE model (loadable)
- `bpe_model.json` - Vocabulary statistics
- `task3_report.txt` - Sample encodings

**Testing:** Streamlit UI "Task 3" page with encode/decode interface

---

### Task 4: Sentence Segmentation (15%)
**Implementation:** `task4_sentence_segmentation.py`
- Azerbaijani-aware algorithm
- Handles abbreviations (Prof., Dr., etc.)
- Azerbaijani titles (cənab, xanım, etc.)
- Complex punctuation support

**Outputs:**
- `sentence_segmenter.pkl` - Loadable model
- `sentences.json` - All segmented sentences
- `task4_report.txt` - Statistics

**Testing:** Streamlit UI "Task 4" page with interactive segmentation

---

### Task 5: Spell Checker - Levenshtein (15%)
**Implementation:** `task5_spell_checker.py`
- Classic edit distance algorithm
- Top-N suggestions with ranking
- Frequency-based prioritization
- Batch text correction

**Outputs:**
- `spell_checker.pkl` - Loadable model
- `spell_checker.json` - Vocabulary stats
- `spell_test_pairs.json` - Test cases
- `task5_report.txt` - Evaluation results

**Testing:** Streamlit UI "Task 5" page with:
- Single word checking
- Batch text correction
- Configurable edit distance

---

### Task 6: Report (20%)
**Implementation:** Automated in `main.py`
- Comprehensive summary report
- Individual task reports
- Statistics and visualizations
- JSON format for programmatic access

**Outputs:**
- `summary.json` - Overall summary
- `task{1-5}_report.txt` - Individual reports
- `extra_task_report.txt` - Extra task report

All reports include:
- Dataset statistics
- Task parameters
- Results and metrics
- Sample outputs
- Interpretations

---

### Extra Task: Weighted Edit Distance (20%)
**Implementation:** `task_extra_weighted.py`
- Confusion matrix implementation
- Weighted dynamic programming
- Character substitution learning
- Azerbaijani-specific confusions (ə↔a, ı↔i, etc.)

**Outputs:**
- `weighted_spell_checker.pkl` - Weighted model
- `weighted_spell_checker_confusion.pkl` - Confusion matrix
- `weighted_spell_checker_confusion.json` - Readable format
- `extra_task_report.txt` - Confusion patterns

**Testing:** Streamlit UI "Extra" page with:
- Confusion matrix visualization
- Weighted vs regular comparison
- Interactive testing

---

## How to Use This Package

### Method 1: Run Everything
```bash
# Install dependencies
pip install -r requirements.txt

# Run all tasks with sample data
python main.py --sample --output ./outputs

# Launch UI
streamlit run streamlit_app.py
```

### Method 2: Use Your Own Data
```bash
# Prepare your JSON files in a folder
# Format: [{"page_number": 1, "content": "text"}, ...]

# Run pipeline
python main.py --data /path/to/json/folder --output ./results

# View results
streamlit run streamlit_app.py
```

### Method 3: Use Individual Modules
```python
# Load any saved model
import pickle
from task3_bpe import BPETokenizer

bpe = BPETokenizer()
bpe.load_model('outputs/bpe_model.pkl')

# Use it
tokens = bpe.encode("Azərbaycan")
print(tokens)
```

---

## Streamlit UI Features

**7 Interactive Pages:**

1. **Overview** - Complete dashboard with all statistics
2. **Task 1** - Token frequency visualization + interactive tokenizer
3. **Task 2** - Heaps' Law plots + parameter display
4. **Task 3** - BPE encode/decode tester + merge operations
5. **Task 4** - Sentence segmentation tester
6. **Task 5** - Spell checker with suggestions
7. **Extra** - Confusion matrix + weighted spell checking

**Features:**
- Beautiful, intuitive interface
- Real-time processing
- Downloadable results
- Side-by-side comparisons
- Interactive testing

---

## Key Features

### Azerbaijani Language Support
- Full UTF-8 encoding
- Special characters: ə, ı, ö, ü, ğ, ş, ç
- Language-specific abbreviations
- Common character confusions

### Modular Architecture
- Each task is independent
- Reusable components
- Clean interfaces
- Well-documented

### Persistent Models
- All models can be saved
- Quick loading for testing
- Portable .pkl files
- JSON exports for inspection

### Comprehensive Testing
- Interactive UI
- Automatic test generation
- Evaluation metrics
- Sample outputs

### Detailed Reporting
- Text reports
- JSON data
- Visualizations
- Statistical analysis

---

## Academic Quality

This implementation demonstrates:

 **Algorithmic Understanding**
- BPE from scratch
- Dynamic programming (Levenshtein)
- Curve fitting (Heaps' Law)
- Pattern matching (segmentation)

 **Software Engineering**
- Modular design
- Error handling
- Documentation
- Testing infrastructure

 **Data Science**
- Statistical analysis
- Visualization
- Model evaluation
- Report generation

 **Production Ready**
- CLI interface
- Web UI
- Saved models
- Comprehensive docs

---

## Technologies Used

- **Python 3.7+** - Core language
- **NumPy** - Numerical computations
- **SciPy** - Curve fitting
- **Matplotlib** - Visualizations
- **Streamlit** - Web interface
- **Pandas** - Data display
- **Pickle** - Model persistence
- **JSON** - Data exchange

---

## Code Quality

### Documentation
- Docstrings for all functions
- Type hints where appropriate
- Inline comments for complex logic
- README and usage guides

### Structure
- PEP 8 style compliance
- Logical organization
- DRY principles
- Clear naming

### Reliability
- Error handling
- Input validation
- Edge case handling
- Tested on sample data

---

## What Makes This Special

1. **Complete Implementation** - All 7 tasks fully implemented
2. **Production Quality** - Not just prototype code
3. **Interactive UI** - Professional Streamlit interface
4. **Azerbaijani-Specific** - Handles special characters correctly
5. **Well-Documented** - README, usage guide, docstrings
6. **Tested** - Sample data and outputs included
7. **Modular** - Easy to extend and customize
8. **Educational** - Clear code, good for learning

---

## Use Cases

### Academic
- NLP course assignments
- Research projects
- Language processing studies
- Algorithm demonstrations

### Development
- Azerbaijani text processing
- Spell checking systems
- Tokenization pipelines
- Custom NLP tools

### Learning
- Understanding BPE
- Implementing edit distance
- Heaps' Law analysis
- UI development with Streamlit

---

## Quick Start Checklist

- [ ] Extract all files to a folder
- [ ] Install: `pip install -r requirements.txt`
- [ ] Run: `python main.py --sample`
- [ ] UI: `streamlit run streamlit_app.py`
- [ ] Explore: Test each page in UI
- [ ] Review: Check ./outputs/ folder
- [ ] Customize: Use your own data

---

## Support

**Issues with setup?**
- Check USAGE_GUIDE.md
- Verify Python 3.7+
- Ensure UTF-8 encoding
- Review error messages

**Want to customize?**
- Read module docstrings
- Check README.md
- Examine sample outputs
- Modify parameters

**Need examples?**
- See USAGE_GUIDE.md
- Check main.py
- Review streamlit_app.py
- Test with sample data

---
