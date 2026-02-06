"""
Extra Task: Weighted Edit Distance with Confusion Matrix
Advanced spell checking using character confusion probabilities
"""
import re, json, pickle, os
from typing import List, Dict, Tuple, Set
from collections import Counter, defaultdict
from .spell_checking import generate_test_pairs


class ConfusionMatrix:
    def __init__(self):
        """Initialize confusion matrix for character pairs"""
        self.confusion_counts = defaultdict(lambda: defaultdict(int))
        self.total_errors = 0
        
        # Common Azerbaijani character confusions
        self.initialize_azerbaijani_confusions()
    
    def initialize_azerbaijani_confusions(self):
        """Initialize with common Azerbaijani character confusions"""
        # Similar looking characters
        confusions = [
            ('a', 'ə', 3), ('ə', 'a', 3),
            ('i', 'ı', 5), ('ı', 'i', 5),
            ('o', 'ö', 4), ('ö', 'o', 4),
            ('u', 'ü', 4), ('ü', 'u', 4),
            ('g', 'ğ', 3), ('ğ', 'g', 3),
            ('s', 'ş', 4), ('ş', 's', 4),
            ('c', 'ç', 4), ('ç', 'c', 4),
            # Keyboard proximity
            ('a', 's', 2), ('s', 'd', 2),
            ('e', 'r', 2), ('r', 't', 2),
            ('y', 'u', 2), ('u', 'i', 2),
        ]
        
        for char1, char2, weight in confusions:
            self.confusion_counts[char1][char2] = weight
    
    def learn_from_pairs(self, error_pairs: List[Tuple[str, str]]):
        """
        Learn confusion probabilities from error pairs
        
        Args:
            error_pairs: List of (incorrect, correct) word pairs
        """
        for incorrect, correct in error_pairs:
            if len(incorrect) == 0 or len(correct) == 0:
                continue
            
            # Simple alignment: find differing characters
            min_len = min(len(incorrect), len(correct))
            
            for i in range(min_len):
                if incorrect[i] != correct[i]:
                    self.confusion_counts[incorrect[i]][correct[i]] += 1
                    self.total_errors += 1
            
            # Handle length differences
            if len(incorrect) > len(correct):
                for i in range(len(correct), len(incorrect)):
                    self.confusion_counts[incorrect[i]][''] += 1  # deletion
                    self.total_errors += 1
            elif len(correct) > len(incorrect):
                for i in range(len(incorrect), len(correct)):
                    self.confusion_counts[''][correct[i]] += 1  # insertion
                    self.total_errors += 1
    
    def get_confusion_weight(self, char1: str, char2: str) -> float:
        """
        Get confusion weight for character pair
        
        Args:
            char1: First character
            char2: Second character
            
        Returns:
            Confusion weight (lower = more likely to be confused)
        """
        if char1 == char2:
            return 0.0
        
        # Check if we have data on this confusion
        if char2 in self.confusion_counts[char1]:
            # More confusions = lower weight (more likely)
            count = self.confusion_counts[char1][char2]
            return 1.0 / (1.0 + count)
        
        # Default weight
        return 1.0
    
    def save(self, filepath: str):
        """Save confusion matrix"""
        data = {
            'confusion_counts': {k: dict(v) for k, v in self.confusion_counts.items()},
            'total_errors': self.total_errors
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        # Save as JSON for inspection
        json_path = filepath.replace('.pkl', '.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def load(self, filepath: str):
        """Load confusion matrix"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.confusion_counts = defaultdict(lambda: defaultdict(int), 
                                           {k: defaultdict(int, v) for k, v in data['confusion_counts'].items()})
        self.total_errors = data['total_errors']


class WeightedSpellChecker:
    def __init__(self, vocabulary: Set[str] = None, confusion_matrix: ConfusionMatrix = None):
        """
        Initialize weighted spell checker
        
        Args:
            vocabulary: Set of correct words
            confusion_matrix: Character confusion matrix
        """
        self.vocabulary = vocabulary or set()
        self.word_freq = Counter()
        self.confusion_matrix = confusion_matrix or ConfusionMatrix()
    
    def weighted_edit_distance(self, s1: str, s2: str) -> float:
        """
        Calculate weighted edit distance using confusion matrix
        
        Args:
            s1: First string
            s2: Second string
            
        Returns:
            Weighted edit distance
        """
        len1, len2 = len(s1), len(s2)
        
        # Create DP table
        dp = [[0.0] * (len2 + 1) for _ in range(len1 + 1)]
        
        # Initialize base cases
        for i in range(len1 + 1):
            dp[i][0] = float(i)
        for j in range(len2 + 1):
            dp[0][j] = float(j)
        
        # Fill DP table
        for i in range(1, len1 + 1):
            for j in range(1, len2 + 1):
                if s1[i-1] == s2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    # Get confusion weights
                    subst_weight = self.confusion_matrix.get_confusion_weight(s1[i-1], s2[j-1])
                    
                    # Weighted operations
                    substitution = dp[i-1][j-1] + subst_weight
                    insertion = dp[i][j-1] + 1.0
                    deletion = dp[i-1][j] + 1.0
                    
                    dp[i][j] = min(substitution, insertion, deletion)
        
        return dp[len1][len2]
    
    def build_vocabulary(self, corpus: str, min_frequency: int = 2):
        """Build vocabulary from corpus"""
        words = re.findall(r'\b[\wəıöüğşç]+\b', corpus.lower())
        self.word_freq = Counter(words)
        self.vocabulary = {word for word, freq in self.word_freq.items() 
                          if freq >= min_frequency}
    
    def get_candidates(self, word: str, max_distance: float = 2.0) -> List[Tuple[str, float]]:
        """
        Get candidate corrections using weighted distance
        
        Args:
            word: Misspelled word
            max_distance: Maximum weighted edit distance
            
        Returns:
            List of (candidate, distance) tuples
        """
        candidates = []
        
        for vocab_word in self.vocabulary:
            # Skip if length difference is too large
            if abs(len(vocab_word) - len(word)) > max_distance:
                continue
            
            distance = self.weighted_edit_distance(word, vocab_word)
            
            if distance <= max_distance:
                candidates.append((vocab_word, distance))
        
        # Sort by weighted distance, then by frequency
        candidates.sort(key=lambda x: (x[1], -self.word_freq.get(x[0], 0)))
        
        return candidates
    
    def correct_word(self, word: str, max_distance: float = 2.0, top_n: int = 5) -> List[Tuple[str, float, int]]:
        """
        Get top correction suggestions
        
        Args:
            word: Word to correct
            max_distance: Maximum weighted distance
            top_n: Number of suggestions
            
        Returns:
            List of (suggestion, distance, frequency) tuples
        """
        word_lower = word.lower()
        
        if word_lower in self.vocabulary:
            return [(word_lower, 0.0, self.word_freq.get(word_lower, 0))]
        
        candidates = self.get_candidates(word_lower, max_distance)
        
        suggestions = [
            (cand, dist, self.word_freq.get(cand, 0))
            for cand, dist in candidates[:top_n]
        ]
        
        return suggestions
    
    def evaluate(self, test_pairs: List[Tuple[str, str]], max_distance: float = 2.0) -> Dict:
        """Evaluate spell checker"""
        correct_top1 = 0
        correct_top5 = 0
        total = len(test_pairs)
        
        for misspelled, correct in test_pairs:
            suggestions = self.correct_word(misspelled, max_distance, top_n=5)
            
            if suggestions:
                if suggestions[0][0] == correct.lower():
                    correct_top1 += 1
                    correct_top5 += 1
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
        """Save weighted spell checker"""
        model_data = {
            'vocabulary': list(self.vocabulary),
            'word_freq': dict(self.word_freq)
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        # Save confusion matrix separately
        confusion_path = filepath.replace('.pkl', '_confusion.pkl')
        self.confusion_matrix.save(confusion_path)
    
    def load_model(self, filepath: str):
        """Load weighted spell checker"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.vocabulary = set(model_data['vocabulary'])
        self.word_freq = Counter(model_data['word_freq'])
        
        # Load confusion matrix
        confusion_path = filepath.replace('.pkl', '_confusion.pkl')
        self.confusion_matrix.load(confusion_path)


def run(corpus: str, output_dir: str = './outputs') -> WeightedSpellChecker:
    """
    Run Extra Task: Weighted spell checking with confusion matrix
    
    Args:
        corpus: Text corpus
        output_dir: Output directory
        
    Returns:
        Trained weighted spell checker
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Build weighted spell checker
    confusion_matrix = ConfusionMatrix()
    spell_checker = WeightedSpellChecker(confusion_matrix=confusion_matrix)
    spell_checker.build_vocabulary(corpus, min_frequency=2)
    
    # Generate test pairs for confusion matrix learning
    test_pairs = generate_test_pairs(spell_checker.vocabulary, num_pairs=100)
    
    # Learn confusion patterns
    spell_checker.confusion_matrix.learn_from_pairs(test_pairs)
    
    # Save model
    model_path = os.path.join(output_dir, 'weighted_spell_checker.pkl')
    spell_checker.save_model(model_path)
    
    # Evaluate
    evaluation = spell_checker.evaluate(test_pairs[:50], max_distance=2.0)
    
    # Save report
    report_path = os.path.join(output_dir, 'extra_task_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"""
=== WEIGHTED SPELL CHECKER WITH CONFUSION MATRIX ===

Model Configuration:
  Vocabulary size: {len(spell_checker.vocabulary):,}
  Total errors learned: {confusion_matrix.total_errors}
  Confusion pairs: {len(confusion_matrix.confusion_counts)}

Evaluation Results:
  Test pairs: {evaluation['total']}
  Top-1 accuracy: {evaluation['accuracy_top1']:.2%}
  Top-5 accuracy: {evaluation['accuracy_top5']:.2%}

Top Character Confusions:
""")
        
        # Show top confusions
        all_confusions = []
        for char1, char2_dict in confusion_matrix.confusion_counts.items():
            for char2, count in char2_dict.items():
                if count > 0:
                    all_confusions.append((char1, char2, count))
        
        all_confusions.sort(key=lambda x: x[2], reverse=True)
        
        for i, (char1, char2, count) in enumerate(all_confusions[:15], 1):
            char1_display = char1 if char1 else '<empty>'
            char2_display = char2 if char2 else '<empty>'
            f.write(f"  {i}. '{char1_display}' -> '{char2_display}': {count} times\n")
    
    print(f"Extra task completed. Model saved to {model_path}")
    print(f"Confusion matrix learned {confusion_matrix.total_errors} error patterns")
    print(f"Accuracy (Top-1): {evaluation['accuracy_top1']:.2%}")
    
    return spell_checker


if __name__ == "__main__":
    # Test
    sample_corpus = """
    Azərbaycan Respublikası Cənubi Qafqazda yerləşir.
    Azərbaycan dili türk dillər ailəsinə aiddir.
    """ * 20
    
    spell_checker = run(sample_corpus)
