# Quick Start Guide - Azerbaijani NLP Tool

## 📦 What You Have

You now have a complete Azerbaijani NLP processing toolkit with:

1. **9 Python modules** - Each implementing specific NLP tasks
2. **Streamlit UI** - Interactive web interface
3. **Sample outputs** - Example results from running the pipeline
4. **Full documentation** - README with detailed instructions

## 🚀 Getting Started (3 Steps)

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

This installs: numpy, scipy, matplotlib, streamlit, pandas

### Step 2: Run the Pipeline

**Option A - Use your own data:**
```bash
python main.py --data /path/to/your/json/folder --output ./outputs
```

**Option B - Test with sample data:**
```bash
python main.py --sample --output ./outputs
```

This will:
- Create/load Azerbaijani text data
- Run all 6 tasks (+ extra task)
- Save models and reports to ./outputs/

### Step 3: Launch the UI
```bash
streamlit run streamlit_app.py
```

Opens in browser at http://localhost:8501

## 📊 What Each File Does

### Core Modules
- `data_loader.py` - Loads JSON files with Azerbaijani text
- `task1_tokenization.py` - Word tokenization & frequency analysis
- `task2_heaps_law.py` - Vocabulary growth analysis
- `task3_bpe.py` - Byte Pair Encoding subword tokenization
- `task4_sentence_segmentation.py` - Sentence boundary detection
- `task5_spell_checker.py` - Spell checking with Levenshtein distance
- `task_extra_weighted.py` - Advanced spell checking with confusion matrix

### Execution & UI
- `main.py` - Runs all tasks in sequence
- `streamlit_app.py` - Interactive web interface

### Documentation
- `README.md` - Complete documentation
- `USAGE_GUIDE.md` - This file
- `requirements.txt` - Python dependencies

## 🎯 Using Individual Modules

### Example 1: Just Tokenization
```python
from task1_tokenization import Tokenizer

text = "Azərbaycan Respublikası Cənubi Qafqazda yerləşir."
tokenizer = Tokenizer()
stats = tokenizer.process_corpus(text)

print(f"Tokens: {stats['total_tokens']}")
print(f"Vocabulary: {stats['total_types']}")
```

### Example 2: BPE Encoding
```python
from task3_bpe import BPETokenizer

bpe = BPETokenizer(num_merges=500)
bpe.train(your_corpus)
bpe.save_model('my_bpe.pkl')

# Later...
bpe.load_model('my_bpe.pkl')
tokens = bpe.encode("Azərbaycan")
print(tokens)  # ['Az', 'ər', 'bay', 'can', '</w>']
```

### Example 3: Spell Checking
```python
from task5_spell_checker import SpellChecker

checker = SpellChecker()
checker.build_vocabulary(your_corpus)

suggestions = checker.correct_word("azrbaycan", max_distance=2, top_n=5)
for word, distance, freq in suggestions:
    print(f"{word} (distance: {distance}, frequency: {freq})")
```

### Example 4: Weighted Spell Checking
```python
from task_extra_weighted import WeightedSpellChecker, ConfusionMatrix

# Create with Azerbaijani-specific confusions
confusion = ConfusionMatrix()
checker = WeightedSpellChecker(confusion_matrix=confusion)
checker.build_vocabulary(your_corpus)

# ə↔a has lower weight, so "azərbaycan" is closer to "azarbaycan"
suggestions = checker.correct_word("azarbaycan", max_distance=2.0)
```

## 📁 Your Data Format

Your JSON files should look like this:

```json
[
  {
    "page_number": 1,
    "content": "Azərbaycan Respublikası Cənubi Qafqazda yerləşir."
  },
  {
    "page_number": 2,
    "content": "Ölkənin paytaxtı Bakı şəhəridir."
  }
]
```

- Each file contains a list of dictionaries
- Each dictionary has `page_number` and `content` keys
- Content is Azerbaijani text (UTF-8 encoded)

## 🖥️ Streamlit UI Pages

1. **Overview** - Dataset statistics and task summaries
2. **Task 1** - Interactive tokenization with charts
3. **Task 2** - Heaps' Law visualization and parameters
4. **Task 3** - BPE encoding/decoding tester
5. **Task 4** - Sentence segmentation tester
6. **Task 5** - Spell checker with suggestions
7. **Extra** - Weighted spell checker with confusion matrix

## 📊 Output Files Explained

After running the pipeline, you'll have:

### Models (loadable with pickle)
- `vocabulary.pkl` - Word vocabulary
- `bpe_model.pkl` - BPE subword tokenizer
- `sentence_segmenter.pkl` - Sentence boundary detector
- `spell_checker.pkl` - Standard spell checker
- `weighted_spell_checker.pkl` - Weighted spell checker
- `weighted_spell_checker_confusion.pkl` - Character confusion matrix

### Reports (human-readable)
- `task1_report.txt` - Tokenization statistics
- `task2_report.txt` - Heaps' Law analysis
- `task3_report.txt` - BPE sample encodings
- `task4_report.txt` - Sentence statistics
- `task5_report.txt` - Spell checker evaluation
- `extra_task_report.txt` - Confusion patterns

### Data Files
- `vocabulary.json` - Top tokens with frequencies
- `sentences.json` - All segmented sentences
- `heaps_law_params.json` - k, β, R² values
- `summary.json` - Complete pipeline summary

### Visualizations
- `heaps_law_plot.png` - Vocabulary growth curves

## 🔧 Customization

### Change BPE Merges
```python
# In main.py or directly
from task3_bpe import run_task3
bpe = run_task3(corpus, num_merges=2000)  # More merges = smaller vocab
```

### Adjust Spell Checker Threshold
```python
# Lower min_frequency = larger vocabulary
checker.build_vocabulary(corpus, min_frequency=1)
```

### Custom Abbreviations
```python
from task4_sentence_segmentation import SentenceSegmenter

segmenter = SentenceSegmenter()
segmenter.abbreviations.update({'cümhur', 'resp', 'xanım'})
```

## ❓ Troubleshooting

**Problem: "No module named 'streamlit'"**
```bash
pip install streamlit
```

**Problem: "No such file or directory: ./outputs"**
```bash
python main.py --sample --output ./outputs
```

**Problem: Encoding errors with Azerbaijani characters**
```bash
# Ensure your JSON files are UTF-8
# Save them with: encoding='utf-8'
```

**Problem: UI shows "Model not found"**
```bash
# Run the pipeline first
python main.py --sample
# Then launch UI
streamlit run streamlit_app.py
```

## 📚 Learning Resources

- **BPE Algorithm**: Sennrich et al., "Neural Machine Translation of Rare Words with Subword Units"
- **Heaps' Law**: Study of vocabulary growth in corpora
- **Levenshtein Distance**: Edit distance for spell checking
- **Azerbaijani Language**: Note special characters: ə, ı, ö, ü, ğ, ş, ç

## 💡 Tips

1. **For small datasets**: Reduce num_merges in BPE (< 500)
2. **For spell checking**: Larger corpus = better suggestions
3. **For Heaps' Law**: Need at least 1000+ tokens for meaningful β
4. **For UI testing**: Use --sample flag for quick demos

## 🎓 Assignment Checklist

- ✅ Task 1: Tokenization & frequency (15%)
- ✅ Task 2: Heaps' Law k and β (15%)
- ✅ Task 3: BPE implementation (15%)
- ✅ Task 4: Sentence segmentation (15%)
- ✅ Task 5: Spell checker (15%)
- ✅ Task 6: Report generation (20%)
- ✅ Extra: Weighted edit distance + confusion matrix (20%)

All tasks generate:
- Code modules ✓
- Saved models ✓
- Test results ✓
- Detailed reports ✓
- Interactive UI ✓

## 🚀 Next Steps

1. Replace sample data with your actual Azerbaijani JSON files
2. Run the full pipeline on your data
3. Explore the Streamlit UI
4. Review generated reports in ./outputs/
5. Use individual modules in your own code
6. Customize parameters for your specific needs

## 📞 Need Help?

Check:
- README.md - Full documentation
- Individual .py files - Detailed docstrings
- sample_outputs/ - Example results
- Task reports - Generated statistics

Happy processing! 🇦🇿
