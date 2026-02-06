"""
Task 5: Spell Checker using Levenshtein Distance
Implement spell checking system for Azerbaijani text
"""
import re, json, pickle, os, random
from typing import List, Dict, Tuple, Set, Optional
from collections import Counter


class SpellChecker:
    def __init__(self, vocabulary: Optional[Set[str]] = None):
        """
        Initialize spell checker
        
        :param vocabulary: Set of correct words
        :type vocabulary: Optional[Set[str]] = None

        :rtype: None
        """
        self.vocabulary = vocabulary or set()
        self.word_freq = Counter()
        
    def levenshtein_distance(self, s1: str, s2: str) -> int:
        """
        Calculate Levenshtein distance between two strings
        
        :param s1: First string
        :param s2: Second string

        :type s1: str
        :type s2: str
            
        :returns: Levenshtein distance
        :rtype: int
        """
        if len(s1) < len(s2):
            return self.levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                # Cost of insertions, deletions, or substitutions
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def build_vocabulary(self, corpus: str, min_frequency: int = 2):
        """
        Build vocabulary from corpus
        
        :param corpus: Text corpus
        :param min_frequency: Minimum word frequency to include

        :type corpus: str
        :type min_frequency: int = 2

        :rtype: None
        """
        # Tokenize
        words = re.findall(r'\b[\wəıöüğşç]+\b', corpus.lower())
        
        # Count frequencies
        self.word_freq = Counter(words)
        
        # Build vocabulary (words that appear at least min_frequency times)
        self.vocabulary = {word for word, freq in self.word_freq.items() 
                          if freq >= min_frequency}
        
        print(f"Vocabulary built: {len(self.vocabulary)} words")
    
    def get_candidates(self, word: str, max_distance: int = 2) -> List[Tuple[str, int]]:
        """
        Get candidate corrections for a word
        
        :param word: Misspelled word
        :param max_distance: Maximum edit distance

        :type word: str
        :type max_distance: int = 2
            
        :returns: List of (candidate, distance) tuples sorted by distance
        :rtype: List[Tuple[str, int]]
        """
        candidates = []
        
        for vocab_word in self.vocabulary:
            # Skip if length difference is too large
            if abs(len(vocab_word) - len(word)) > max_distance:
                continue
            
            distance = self.levenshtein_distance(word, vocab_word)
            
            if distance <= max_distance:
                candidates.append((vocab_word, distance))
        
        # Sort by distance, then by frequency
        candidates.sort(key=lambda x: (x[1], -self.word_freq.get(x[0], 0)))
        
        return candidates
    
    def correct_word(self, word: str, max_distance: int = 2, top_n: int = 5) -> List[Tuple[str, int, int]]:
        """
        Get top correction suggestions for a word
        
        :param word: Word to correct
        :param max_distance: Maximum edit distance
        :param top_n: Number of suggestions to return

        :type word: str
        :type max_distance: int = 2
        :type top_n: int = 5
            
        :returns: List of (suggestion, distance, frequency) tuples
        :rtype: List[Tuple[str, int, int]]
        """
        word_lower = word.lower()
        
        # If word is in vocabulary, return it
        if word_lower in self.vocabulary:
            return [(word_lower, 0, self.word_freq.get(word_lower, 0))]
        
        # Get candidates
        candidates = self.get_candidates(word_lower, max_distance)
        
        # Add frequency information
        suggestions = [
            (cand, dist, self.word_freq.get(cand, 0))
            for cand, dist in candidates[:top_n]
        ]
        
        return suggestions
    
    def correct_text(self, text: str, max_distance: int = 2) -> str:
        """
        Correct all words in text
        
        :param text: Input text
        :param max_distance: Maximum edit distance

        :type text: str
        :type max_distance: int = 2
            
        :returns: Corrected text
        :rtype: str
        """
        words = re.findall(r'\b[\wəıöüğşç]+\b', text.lower())
        corrected_words = []
        
        for word in words:
            suggestions = self.correct_word(word, max_distance, top_n=1)
            if suggestions:
                corrected_words.append(suggestions[0][0])
            else:
                corrected_words.append(word)
        
        return ' '.join(corrected_words)
    
    def evaluate(self, test_pairs: List[Tuple[str, str]], max_distance: int = 2) -> Dict:
        """
        Evaluate spell checker on test data
        
        :param test_pairs: List of (misspelled, correct) pairs
        :param max_distance: Maximum edit distance
            
        :type test_pairs: List[Tuple[str, str]]
        :type max_distance: int = 2

        :returns: Evaluation metrics
        :rtype: Dict
        """
        correct_top1 = 0
        correct_top5 = 0
        total = len(test_pairs)
        
        for misspelled, correct in test_pairs:
            suggestions = self.correct_word(misspelled, max_distance, top_n=5)
            
            if suggestions:
                # Check if correct word is in top 1
                if suggestions[0][0] == correct.lower():
                    correct_top1 += 1
                    correct_top5 += 1
                # Check if correct word is in top 5
                elif any(sugg[0] == correct.lower() for sugg in suggestions):
                    correct_top5 += 1
        
        return {
            'total': total,
            'correct_top1': correct_top1,
            'correct_top5': correct_top5,
            'accuracy_top1': correct_top1 / total if total > 0 else 0,
            'accuracy_top5': correct_top5 / total if total > 0 else 0
        }
    
    def save_model(self, filepath: str):
        """
        Save spell checker model.
        
        :param filepath: Target path to save the file to
        :type filepath: str

        :rtype: None
        """
        model_data = {
            'vocabulary': list(self.vocabulary),
            'word_freq': dict(self.word_freq)
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        # Save summary as JSON
        json_path = filepath.replace('.pkl', '.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump({
                'vocabulary_size': len(self.vocabulary),
                'total_words': sum(self.word_freq.values()),
                'top_20_words': dict(self.word_freq.most_common(20))
            }, f, ensure_ascii=False, indent=2)
        
        print(f"Spell checker model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """
        Load spell checker model.
        
        :param filepath: Path to the saved model to load from
        :type filepath: str

        :rtype: None
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.vocabulary = set(model_data['vocabulary'])
        self.word_freq = Counter(model_data['word_freq'])
        
        print(f"Spell checker model loaded from {filepath}")


def generate_test_pairs(vocabulary: Set[str], num_pairs: int = 50) -> List[Tuple[str, str]]:
    """
    Generate test pairs by introducing errors

    :param vocabulary: Set of correct words
    :param num_pairs: Number of test pairs to generate
        
    :type vocabulary: Set[str]
    :type num_pairs: int = 50

    :returns: List of (misspelled, correct) pairs
    :rtype: List[Tuple[str, str]]
    """
    test_pairs = []
    vocab_list = list(vocabulary)
    
    # Ensure we have enough words
    if len(vocab_list) < num_pairs:
        num_pairs = len(vocab_list)
    
    selected_words = random.sample(vocab_list, num_pairs)
    
    for word in selected_words:
        if len(word) < 3:
            continue
        
        # Randomly introduce an error
        error_type = random.choice(['substitute', 'delete', 'insert', 'swap'])
        
        word_list = list(word)
        
        if error_type == 'substitute' and len(word_list) > 0:
            # Substitute a random character
            pos = random.randint(0, len(word_list) - 1)
            chars = 'abcdefghijklmnopqrstuvwxyzəıöüğşç'
            word_list[pos] = random.choice(chars)
        
        elif error_type == 'delete' and len(word_list) > 1:
            # Delete a random character
            pos = random.randint(0, len(word_list) - 1)
            word_list.pop(pos)
        
        elif error_type == 'insert':
            # Insert a random character
            pos = random.randint(0, len(word_list))
            chars = 'abcdefghijklmnopqrstuvwxyzəıöüğşç'
            word_list.insert(pos, random.choice(chars))
        
        elif error_type == 'swap' and len(word_list) > 1:
            # Swap two adjacent characters
            pos = random.randint(0, len(word_list) - 2)
            word_list[pos], word_list[pos + 1] = word_list[pos + 1], word_list[pos]
        
        misspelled = ''.join(word_list)
        if misspelled != word:
            test_pairs.append((misspelled, word))
    
    return test_pairs


def run(corpus: str, output_dir: str = './outputs') -> SpellChecker:
    """
    Run Task 5: Spell checking with Levenshtein distance
    
    :param corpus: Text corpus
    :param output_dir: Output directory

    :type corpus: str
    :type output_dir: str = './outputs'
    
    :returns: Trained spell checker
    :rtype: SpellChecker
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Build spell checker
    spell_checker = SpellChecker()
    spell_checker.build_vocabulary(corpus, min_frequency=2)
    
    # Save model
    model_path = os.path.join(output_dir, 'spell_checker.pkl')
    spell_checker.save_model(model_path)
    
    # Generate test pairs
    test_pairs = generate_test_pairs(spell_checker.vocabulary, num_pairs=50)
    
    # Evaluate
    if test_pairs:
        evaluation = spell_checker.evaluate(test_pairs, max_distance=2)
    else:
        evaluation = {'total': 0, 'accuracy_top1': 0, 'accuracy_top5': 0}
    
    # Save report
    report_path = os.path.join(output_dir, 'task5_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"""
=== SPELL CHECKER REPORT ===

Vocabulary Statistics:
  Vocabulary size: {len(spell_checker.vocabulary):,}
  Total words in corpus: {sum(spell_checker.word_freq.values()):,}

Evaluation Results:
  Test pairs: {evaluation['total']}
  Top-1 accuracy: {evaluation['accuracy_top1']:.2%}
  Top-5 accuracy: {evaluation['accuracy_top5']:.2%}

Sample Test Cases:
""")
        
        for i, (misspelled, correct) in enumerate(test_pairs[:10], 1):
            suggestions = spell_checker.correct_word(misspelled, max_distance=2, top_n=3)
            f.write(f"\n{i}. Misspelled: '{misspelled}' | Correct: '{correct}'\n")
            f.write(f"   Suggestions: {[s[0] for s in suggestions]}\n")
    
    # Save test pairs
    test_path = os.path.join(output_dir, 'spell_test_pairs.json')
    with open(test_path, 'w', encoding='utf-8') as f:
        json.dump({
            'test_pairs': test_pairs,
            'evaluation': evaluation
        }, f, ensure_ascii=False, indent=2)
    
    print(f"Task 5 completed. Spell checker saved to {model_path}")
    print(f"Vocabulary size: {len(spell_checker.vocabulary)}")
    print(f"Evaluation accuracy (Top-1): {evaluation['accuracy_top1']:.2%}")
    
    return spell_checker


if __name__ == "__main__":
    # Test with sample corpus
    sample_corpus = """
    Azərbaycan Respublikası Cənubi Qafqazda yerləşir.
    Azərbaycan dili türk dillər ailəsinə aiddir.
    Bakı Azərbaycanın paytaxtıdır.
    """ * 10
    
    spell_checker = run(sample_corpus)
    
    # Test correction
    test_word = "azrbaycan"  # misspelled
    suggestions = spell_checker.correct_word(test_word)
    print(f"\nCorrections for '{test_word}':")
    for word, dist, freq in suggestions:
        print(f"  {word} (distance: {dist}, frequency: {freq})")
