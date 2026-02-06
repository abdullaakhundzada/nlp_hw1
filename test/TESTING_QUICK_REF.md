# Quick Testing Reference Card

## Most Common Commands

### Test Your Data on Regular Spell Checker
```bash
python test_spell_checker.py --test-data YOUR_TEST_FILE.json
```

### Test Your Data on Weighted Spell Checker
```bash
python test_spell_checker.py --test-data YOUR_TEST_FILE.json --weighted --model ./outputs/weighted_spell_checker.pkl
```

### Compare Both Spell Checkers
```bash
python compare_spell_checkers.py --test-data YOUR_TEST_FILE.json
```

---

## Your Test Data Format

```json
{
  "test_cases": [
    {
      "id": 1,
      "correct": "Correct Azerbaijani sentence.",
      "typo": "Same sentence with typos."
    }
  ]
}
```

---

## Outputs

### From `test_spell_checker.py`:
- **evaluation_results.json** - Detailed JSON results
- **evaluation_report.txt** - Human-readable report
- **Metrics**: Accuracy, Precision, Recall, F1 Score

### From `compare_spell_checkers.py`:
- All of the above for BOTH checkers
- **comparison_results.json** - Side-by-side comparison
- Shows which checker performs better

---

## Customize Your Test

```bash
# Change max edit distance (default: 2.0)
python test_spell_checker.py --test-data data.json --max-distance 3.0

# Change output directory
python test_spell_checker.py --test-data data.json --output ./my_results

# Show more examples in console
python test_spell_checker.py --test-data data.json --examples 20
```

---

## Understanding Results

```
Accuracy: 75.50%    ← How many corrections were right
Precision: 75.50%   ← Quality of corrections
Recall: 45.20%      ← How many errors were caught
F1 Score: 0.5661    ← Overall performance (0-1)
```

**Good Performance**: Accuracy > 70%, F1 > 0.6
**Needs Improvement**: Accuracy < 50%, F1 < 0.4

---

## Quick Troubleshooting

**Error: "Model not found"**
→ Run: `python main.py --sample` first

**Low accuracy (<30%)**
→ Your vocabulary might be too small
→ Train on more data or domain-specific corpus

**No suggestions found**
→ Increase: `--max-distance 3.0`

---

## Complete Workflow

1. **Prepare**: Create your test JSON file
2. **Test Regular**: `python test_spell_checker.py --test-data YOUR_FILE.json`
3. **Test Weighted**: `python test_spell_checker.py --test-data YOUR_FILE.json --weighted --model ./outputs/weighted_spell_checker.pkl`
4. **Compare**: `python compare_spell_checkers.py --test-data YOUR_FILE.json`
5. **Review**: Check `comparison_results/comparison_results.json`

---


See **TESTING_GUIDE.md** for complete details!
