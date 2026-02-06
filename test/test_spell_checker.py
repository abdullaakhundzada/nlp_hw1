"""
Spell Checker Evaluation Script
Tests pretrained spell checker on a test dataset
"""
import json
import os
import argparse
from typing import Dict, List, Tuple
import re
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Import spell checker modules
from src.spell_checking import SpellChecker
from src.extra_weighted import WeightedSpellChecker


class SpellCheckerEvaluator:
    def __init__(self, spell_checker, test_data_path: str):
        """
        Initialize evaluator
        
        Args:
            spell_checker: Loaded spell checker (regular or weighted)
            test_data_path: Path to test JSON file
        """
        self.spell_checker = spell_checker
        self.test_data_path = test_data_path
        self.test_cases = []
        self.results = []
        
    def load_test_data(self):
        """Load test cases from JSON file"""
        with open(self.test_data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.test_cases = data.get('test_cases', [])
        print(f"Loaded {len(self.test_cases)} test cases")
        
        return self.test_cases
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words"""
        # Same tokenization as in task1
        tokens = re.findall(r'\b[\wəıöüğşç]+\b', text.lower())
        return tokens
    
    def evaluate_single_case(self, test_case: Dict, max_distance: float = 2) -> Dict:
        """
        Evaluate a single test case
        
        Args:
            test_case: Dictionary with 'correct' and 'typo' keys
            max_distance: Maximum edit distance for corrections
            
        Returns:
            Dictionary with evaluation results
        """
        correct_text = test_case['correct']
        typo_text = test_case['typo']
        test_id = test_case.get('id', 'unknown')
        
        # Tokenize both texts
        correct_tokens = self.tokenize(correct_text)
        typo_tokens = self.tokenize(typo_text)
        
        # Track corrections
        corrections = []
        word_corrections = {
            'total_words': len(typo_tokens),
            'corrected': 0,
            'correct_corrections': 0,
            'incorrect_corrections': 0,
            'unchanged': 0
        }
        
        # Create mapping of correct tokens by position (if same length)
        if len(correct_tokens) == len(typo_tokens):
            aligned = True
            correct_map = {i: correct_tokens[i] for i in range(len(correct_tokens))}
        else:
            aligned = False
            correct_map = {}
        
        # Evaluate each typo word
        for i, typo_word in enumerate(typo_tokens):
            # Get correction from spell checker
            if isinstance(self.spell_checker, WeightedSpellChecker):
                suggestions = self.spell_checker.correct_word(typo_word, max_distance=max_distance, top_n=1)
            else:
                suggestions = self.spell_checker.correct_word(typo_word, max_distance=int(max_distance), top_n=1)
            
            if suggestions:
                corrected_word = suggestions[0][0]
                distance = suggestions[0][1]
                
                # Check if correction is correct (if aligned)
                if aligned and i in correct_map:
                    expected = correct_map[i]
                    is_correct = (corrected_word == expected)
                    
                    if corrected_word != typo_word:
                        word_corrections['corrected'] += 1
                        if is_correct:
                            word_corrections['correct_corrections'] += 1
                        else:
                            word_corrections['incorrect_corrections'] += 1
                    else:
                        word_corrections['unchanged'] += 1
                    
                    corrections.append({
                        'position': i,
                        'typo': typo_word,
                        'corrected': corrected_word,
                        'expected': expected,
                        'is_correct': is_correct,
                        'distance': distance
                    })
                else:
                    # Can't verify without alignment
                    corrections.append({
                        'position': i,
                        'typo': typo_word,
                        'corrected': corrected_word,
                        'distance': distance
                    })
            else:
                # No suggestion found
                corrections.append({
                    'position': i,
                    'typo': typo_word,
                    'corrected': typo_word,
                    'distance': 999
                })
                word_corrections['unchanged'] += 1
        
        # Reconstruct corrected text
        corrected_text = ' '.join([c['corrected'] for c in corrections])
        
        # Calculate metrics
        result = {
            'id': test_id,
            'original': typo_text,
            'corrected': corrected_text,
            'expected': correct_text,
            'aligned': aligned,
            'word_corrections': word_corrections,
            'corrections': corrections
        }
        
        # Calculate accuracy (if aligned)
        if aligned and word_corrections['corrected'] > 0:
            result['accuracy'] = word_corrections['correct_corrections'] / word_corrections['corrected']
        else:
            result['accuracy'] = None
        
        return result
    
    def evaluate_all(self, max_distance: float = 2) -> Dict:
        """
        Evaluate all test cases
        
        Args:
            max_distance: Maximum edit distance
            
        Returns:
            Dictionary with overall statistics
        """
        if not self.test_cases:
            self.load_test_data()
        
        self.results = []
        overall_stats = {
            'total_cases': len(self.test_cases),
            'total_words': 0,
            'corrected_words': 0,
            'correct_corrections': 0,
            'incorrect_corrections': 0,
            'unchanged_words': 0,
            'aligned_cases': 0
        }
        
        print(f"\nEvaluating {len(self.test_cases)} test cases...")
        print("=" * 80)
        
        for i, test_case in enumerate(self.test_cases, 1):
            result = self.evaluate_single_case(test_case, max_distance)
            self.results.append(result)
            
            # Update overall stats
            wc = result['word_corrections']
            overall_stats['total_words'] += wc['total_words']
            overall_stats['corrected_words'] += wc['corrected']
            overall_stats['correct_corrections'] += wc['correct_corrections']
            overall_stats['incorrect_corrections'] += wc['incorrect_corrections']
            overall_stats['unchanged_words'] += wc['unchanged']
            
            if result['aligned']:
                overall_stats['aligned_cases'] += 1
            
            # Print progress
            if i % 10 == 0 or i == len(self.test_cases):
                print(f"Processed {i}/{len(self.test_cases)} cases...")
        
        # Calculate overall accuracy
        if overall_stats['corrected_words'] > 0:
            overall_stats['accuracy'] = overall_stats['correct_corrections'] / overall_stats['corrected_words']
        else:
            overall_stats['accuracy'] = 0.0
        
        # Calculate precision and recall
        if overall_stats['corrected_words'] > 0:
            overall_stats['precision'] = overall_stats['correct_corrections'] / overall_stats['corrected_words']
        else:
            overall_stats['precision'] = 0.0
        
        total_errors = sum(1 for r in self.results for c in r['corrections'] if c.get('typo') != c.get('expected'))
        if total_errors > 0:
            overall_stats['recall'] = overall_stats['correct_corrections'] / total_errors
        else:
            overall_stats['recall'] = 0.0
        
        # F1 score
        if overall_stats['precision'] + overall_stats['recall'] > 0:
            overall_stats['f1_score'] = 2 * (overall_stats['precision'] * overall_stats['recall']) / (overall_stats['precision'] + overall_stats['recall'])
        else:
            overall_stats['f1_score'] = 0.0
        
        return overall_stats
    
    def print_results(self, overall_stats: Dict, show_details: bool = True, num_examples: int = 5):
        """
        Print evaluation results
        
        Args:
            overall_stats: Overall statistics dictionary
            show_details: Whether to show detailed examples
            num_examples: Number of examples to show
        """
        print("\n" + "=" * 80)
        print("SPELL CHECKER EVALUATION RESULTS")
        print("=" * 80)
        
        print(f"\nOverall Statistics:")
        print(f"  Total test cases: {overall_stats['total_cases']}")
        print(f"  Aligned cases: {overall_stats['aligned_cases']}")
        print(f"  Total words: {overall_stats['total_words']}")
        print(f"  Words corrected: {overall_stats['corrected_words']}")
        print(f"  Correct corrections: {overall_stats['correct_corrections']}")
        print(f"  Incorrect corrections: {overall_stats['incorrect_corrections']}")
        print(f"  Unchanged words: {overall_stats['unchanged_words']}")
        
        print(f"\nPerformance Metrics:")
        print(f"  Accuracy: {overall_stats['accuracy']:.2%}")
        print(f"  Precision: {overall_stats['precision']:.2%}")
        print(f"  Recall: {overall_stats['recall']:.2%}")
        print(f"  F1 Score: {overall_stats['f1_score']:.4f}")
        
        if show_details and self.results:
            print(f"\n" + "=" * 80)
            print(f"SAMPLE RESULTS (showing {min(num_examples, len(self.results))} examples)")
            print("=" * 80)
            
            for i, result in enumerate(self.results[:num_examples], 1):
                print(f"\nTest Case #{result['id']}:")
                print(f"  Typo:      {result['original']}")
                print(f"  Corrected: {result['corrected']}")
                print(f"  Expected:  {result['expected']}")
                
                if result['aligned'] and result['accuracy'] is not None:
                    print(f"  Accuracy:  {result['accuracy']:.2%}")
                
                # Show word-level details
                if result['corrections']:
                    print(f"  Word corrections:")
                    for corr in result['corrections'][:10]:  # Show first 10 words
                        if 'expected' in corr:
                            status = "✓" if corr.get('is_correct') else "✗"
                            print(f"    {status} '{corr['typo']}' → '{corr['corrected']}' (expected: '{corr['expected']}')")
                        else:
                            print(f"    '{corr['typo']}' → '{corr['corrected']}'")
    
    def save_results(self, output_path: str, overall_stats: Dict):
        """
        Save results to JSON file
        
        Args:
            output_path: Path to save results
            overall_stats: Overall statistics
        """
        output_data = {
            'overall_statistics': overall_stats,
            'test_results': self.results
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"\nResults saved to: {output_path}")
    
    def generate_report(self, output_path: str, overall_stats: Dict):
        """
        Generate detailed text report
        
        Args:
            output_path: Path to save report
            overall_stats: Overall statistics
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("SPELL CHECKER EVALUATION REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("Overall Statistics:\n")
            f.write(f"  Total test cases: {overall_stats['total_cases']}\n")
            f.write(f"  Aligned cases: {overall_stats['aligned_cases']}\n")
            f.write(f"  Total words: {overall_stats['total_words']}\n")
            f.write(f"  Words corrected: {overall_stats['corrected_words']}\n")
            f.write(f"  Correct corrections: {overall_stats['correct_corrections']}\n")
            f.write(f"  Incorrect corrections: {overall_stats['incorrect_corrections']}\n")
            f.write(f"  Unchanged words: {overall_stats['unchanged_words']}\n\n")
            
            f.write("Performance Metrics:\n")
            f.write(f"  Accuracy: {overall_stats['accuracy']:.2%}\n")
            f.write(f"  Precision: {overall_stats['precision']:.2%}\n")
            f.write(f"  Recall: {overall_stats['recall']:.2%}\n")
            f.write(f"  F1 Score: {overall_stats['f1_score']:.4f}\n\n")
            
            f.write("=" * 80 + "\n")
            f.write("DETAILED RESULTS\n")
            f.write("=" * 80 + "\n\n")
            
            for result in self.results:
                f.write(f"Test Case #{result['id']}:\n")
                f.write(f"  Typo:      {result['original']}\n")
                f.write(f"  Corrected: {result['corrected']}\n")
                f.write(f"  Expected:  {result['expected']}\n")
                
                if result['aligned'] and result['accuracy'] is not None:
                    f.write(f"  Accuracy:  {result['accuracy']:.2%}\n")
                
                f.write("\n")
        
        print(f"Report saved to: {output_path}")
    
    def generate_character_confusion_matrix(self, output_path: str = None):
        """
        Generate and plot character-level confusion matrix from test results
        
        Args:
            output_path: Path to save the plot (optional)
        
        Returns:
            Dictionary with confusion statistics
        """
        # Collect character-level errors
        char_confusions = defaultdict(lambda: defaultdict(int))
        
        for result in self.results:
            if not result['aligned']:
                continue
            
            for correction in result['corrections']:
                if 'expected' not in correction:
                    continue
                
                typo_word = correction['typo']
                expected_word = correction['expected']
                
                # Align characters and count substitutions
                max_len = max(len(typo_word), len(expected_word))
                
                for i in range(max_len):
                    char_typo = typo_word[i] if i < len(typo_word) else ''
                    char_expected = expected_word[i] if i < len(expected_word) else ''
                    
                    if char_typo != char_expected:
                        char_confusions[char_typo][char_expected] += 1
        
        # Get top confused characters
        all_confusions = []
        for char1, char2_dict in char_confusions.items():
            for char2, count in char2_dict.items():
                if count > 0:
                    all_confusions.append((char1, char2, count))
        
        all_confusions.sort(key=lambda x: x[2], reverse=True)
        
        # Take top N confusions for visualization
        top_n = min(20, len(all_confusions))
        top_confusions = all_confusions[:top_n]
        
        if not top_confusions:
            print("No character confusions found in test data")
            return {}
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Left plot: Bar chart of top confusions
        labels = [f"'{c1}' → '{c2}'" if c1 and c2 else 
                 f"'{c1}' → <del>" if c1 else 
                 f"<ins> → '{c2}'" 
                 for c1, c2, _ in top_confusions]
        counts = [count for _, _, count in top_confusions]
        
        y_pos = np.arange(len(labels))
        ax1.barh(y_pos, counts, color='steelblue', alpha=0.7)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(labels, fontsize=10)
        ax1.invert_yaxis()
        ax1.set_xlabel('Frequency', fontsize=12)
        ax1.set_title('Top 20 Character Confusions in Test Data', fontsize=14, fontweight='bold')
        ax1.grid(axis='x', alpha=0.3)
        
        # Right plot: Confusion matrix heatmap (for most common characters)
        # Get unique characters involved in confusions
        unique_chars = set()
        for c1, c2, _ in top_confusions[:15]:  # Top 15 for matrix
            if c1: unique_chars.add(c1)
            if c2: unique_chars.add(c2)
        
        char_list = sorted(list(unique_chars))
        if len(char_list) > 1:
            matrix = np.zeros((len(char_list), len(char_list)))
            
            for c1, c2, count in all_confusions:
                if c1 in char_list and c2 in char_list:
                    i = char_list.index(c1)
                    j = char_list.index(c2)
                    matrix[i][j] = count
            
            # Plot heatmap
            im = ax2.imshow(matrix, cmap='YlOrRd', aspect='auto')
            
            # Set ticks
            ax2.set_xticks(np.arange(len(char_list)))
            ax2.set_yticks(np.arange(len(char_list)))
            ax2.set_xticklabels(char_list, fontsize=11)
            ax2.set_yticklabels(char_list, fontsize=11)
            
            # Rotate x labels
            plt.setp(ax2.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax2)
            cbar.set_label('Frequency', fontsize=11)
            
            # Add text annotations for non-zero values
            for i in range(len(char_list)):
                for j in range(len(char_list)):
                    if matrix[i][j] > 0:
                        text_color = 'white' if matrix[i][j] > matrix.max()/2 else 'black'
                        ax2.text(j, i, int(matrix[i][j]), ha="center", va="center", 
                               color=text_color, fontsize=9, fontweight='bold')
            
            ax2.set_title('Character Confusion Matrix\n(Typo → Expected)', 
                         fontsize=14, fontweight='bold')
            ax2.set_xlabel('Expected Character', fontsize=12)
            ax2.set_ylabel('Typo Character', fontsize=12)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix plot saved to: {output_path}")
        
        plt.close()
        
        # Return statistics
        return {
            'total_confusions': len(all_confusions),
            'top_confusions': top_confusions,
            'confusion_matrix': char_confusions
        }


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Test spell checker on test dataset')
    parser.add_argument('--test-data', type=str, required=True,
                       help='Path to test JSON file')
    parser.add_argument('--model', type=str, default='./outputs/spell_checker.pkl',
                       help='Path to spell checker model (default: ./outputs/spell_checker.pkl)')
    parser.add_argument('--weighted', action='store_true',
                       help='Use weighted spell checker instead of regular')
    parser.add_argument('--max-distance', type=float, default=2.0,
                       help='Maximum edit distance (default: 2.0)')
    parser.add_argument('--output', type=str, default='./test_results',
                       help='Output directory for results (default: ./test_results)')
    parser.add_argument('--examples', type=int, default=10,
                       help='Number of examples to show (default: 10)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Load spell checker
    print(f"Loading spell checker from: {args.model}")
    
    if args.weighted:
        spell_checker = WeightedSpellChecker()
        spell_checker.load_model(args.model)
        checker_type = "Weighted"
    else:
        spell_checker = SpellChecker()
        spell_checker.load_model(args.model)
        checker_type = "Regular"
    
    print(f"Loaded {checker_type} spell checker")
    print(f"Vocabulary size: {len(spell_checker.vocabulary):,}")
    
    # Create evaluator
    evaluator = SpellCheckerEvaluator(spell_checker, args.test_data)
    
    # Evaluate
    overall_stats = evaluator.evaluate_all(max_distance=args.max_distance)
    
    # Print results
    evaluator.print_results(overall_stats, show_details=True, num_examples=args.examples)
    
    # Save results
    results_json = os.path.join(args.output, 'evaluation_results.json')
    evaluator.save_results(results_json, overall_stats)
    
    # Generate report
    report_txt = os.path.join(args.output, 'evaluation_report.txt')
    evaluator.generate_report(report_txt, overall_stats)
    
    # Generate confusion matrix plot
    print("\n" + "=" * 80)
    print("GENERATING CONFUSION MATRIX VISUALIZATION")
    print("=" * 80)
    
    confusion_plot = os.path.join(args.output, 'confusion_matrix.png')
    confusion_stats = evaluator.generate_character_confusion_matrix(confusion_plot)
    
    if confusion_stats:
        print(f"\nTop 5 Character Confusions:")
        for i, (c1, c2, count) in enumerate(confusion_stats['top_confusions'][:5], 1):
            c1_display = c1 if c1 else '<empty>'
            c2_display = c2 if c2 else '<empty>'
            print(f"  {i}. '{c1_display}' → '{c2_display}': {count} occurrences")
    
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    print(f"Results directory: {args.output}")
    print(f"  - evaluation_results.json (detailed JSON)")
    print(f"  - evaluation_report.txt (text report)")
    print(f"  - confusion_matrix.png (character confusion visualization)")


if __name__ == "__main__":
    main()
