"""
Enhanced Spell Checker Testing with Comparison
Tests and compares regular vs weighted spell checkers
"""
import json
import os
import argparse
from test_spell_checker import SpellCheckerEvaluator
from src.spell_checking import SpellChecker
from src.extra_weighted import WeightedSpellChecker


def compare_spell_checkers(test_data_path: str, 
                          regular_model_path: str,
                          weighted_model_path: str,
                          output_dir: str,
                          max_distance: float = 2.0):
    """
    Compare regular and weighted spell checkers
    
    Args:
        test_data_path: Path to test data JSON
        regular_model_path: Path to regular spell checker
        weighted_model_path: Path to weighted spell checker
        output_dir: Output directory
        max_distance: Maximum edit distance
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load both spell checkers
    print("=" * 80)
    print("LOADING SPELL CHECKERS")
    print("=" * 80)
    
    print(f"\n1. Loading Regular Spell Checker...")
    regular_checker = SpellChecker()
    regular_checker.load_model(regular_model_path)
    print(f"   Vocabulary size: {len(regular_checker.vocabulary):,}")
    
    print(f"\n2. Loading Weighted Spell Checker...")
    weighted_checker = WeightedSpellChecker()
    weighted_checker.load_model(weighted_model_path)
    print(f"   Vocabulary size: {len(weighted_checker.vocabulary):,}")
    print(f"   Confusion patterns: {weighted_checker.confusion_matrix.total_errors}")
    
    # Evaluate regular checker
    print("\n" + "=" * 80)
    print("EVALUATING REGULAR SPELL CHECKER")
    print("=" * 80)
    
    regular_evaluator = SpellCheckerEvaluator(regular_checker, test_data_path)
    regular_stats = regular_evaluator.evaluate_all(max_distance=max_distance)
    regular_evaluator.print_results(regular_stats, show_details=False)
    
    # Save regular results
    regular_evaluator.save_results(
        os.path.join(output_dir, 'regular_checker_results.json'),
        regular_stats
    )
    regular_evaluator.generate_report(
        os.path.join(output_dir, 'regular_checker_report.txt'),
        regular_stats
    )
    
    # Evaluate weighted checker
    print("\n" + "=" * 80)
    print("EVALUATING WEIGHTED SPELL CHECKER")
    print("=" * 80)
    
    weighted_evaluator = SpellCheckerEvaluator(weighted_checker, test_data_path)
    weighted_stats = weighted_evaluator.evaluate_all(max_distance=max_distance)
    weighted_evaluator.print_results(weighted_stats, show_details=False)
    
    # Save weighted results
    weighted_evaluator.save_results(
        os.path.join(output_dir, 'weighted_checker_results.json'),
        weighted_stats
    )
    weighted_evaluator.generate_report(
        os.path.join(output_dir, 'weighted_checker_report.txt'),
        weighted_stats
    )
    
    # Generate comparison report
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    
    comparison = {
        'regular': regular_stats,
        'weighted': weighted_stats,
        'improvements': {}
    }
    
    # Calculate improvements
    for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
        regular_val = regular_stats[metric]
        weighted_val = weighted_stats[metric]
        improvement = weighted_val - regular_val
        improvement_pct = (improvement / regular_val * 100) if regular_val > 0 else 0
        
        comparison['improvements'][metric] = {
            'absolute': improvement,
            'percentage': improvement_pct
        }
    
    # Print comparison table
    print(f"\n{'Metric':<20} {'Regular':<15} {'Weighted':<15} {'Improvement':<15}")
    print("-" * 70)
    
    metrics_display = {
        'accuracy': 'Accuracy',
        'precision': 'Precision',
        'recall': 'Recall',
        'f1_score': 'F1 Score'
    }
    
    for metric, display_name in metrics_display.items():
        regular_val = regular_stats[metric]
        weighted_val = weighted_stats[metric]
        improvement = comparison['improvements'][metric]['absolute']
        improvement_pct = comparison['improvements'][metric]['percentage']
        
        print(f"{display_name:<20} {regular_val:>6.2%}        {weighted_val:>6.2%}        {improvement:>+6.2%} ({improvement_pct:>+6.1f}%)")
    
    print("\n" + "-" * 70)
    print(f"{'Total Words':<20} {regular_stats['total_words']:>14,} {weighted_stats['total_words']:>14,}")
    print(f"{'Corrected Words':<20} {regular_stats['corrected_words']:>14,} {weighted_stats['corrected_words']:>14,}")
    print(f"{'Correct Corrections':<20} {regular_stats['correct_corrections']:>14,} {weighted_stats['correct_corrections']:>14,}")
    
    # Save comparison
    comparison_path = os.path.join(output_dir, 'comparison_results.json')
    with open(comparison_path, 'w', encoding='utf-8') as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)
    
    print(f"\nComparison saved to: {comparison_path}")
    
    # Generate side-by-side examples
    print("\n" + "=" * 80)
    print("SIDE-BY-SIDE EXAMPLES")
    print("=" * 80)
    
    num_examples = min(5, len(regular_evaluator.results))
    
    for i in range(num_examples):
        regular_result = regular_evaluator.results[i]
        weighted_result = weighted_evaluator.results[i]
        
        print(f"\nTest Case #{regular_result['id']}:")
        print(f"  Original:  {regular_result['original']}")
        print(f"  Expected:  {regular_result['expected']}")
        print(f"  Regular:   {regular_result['corrected']}")
        print(f"  Weighted:  {weighted_result['corrected']}")
        
        if regular_result['aligned']:
            reg_acc = regular_result.get('accuracy', 0) or 0
            wgt_acc = weighted_result.get('accuracy', 0) or 0
            print(f"  Accuracy:  Regular={reg_acc:.1%}, Weighted={wgt_acc:.1%}")
    
    print("\n" + "=" * 80)
    print("COMPARISON COMPLETE")
    print("=" * 80)
    print(f"Output directory: {output_dir}")
    print(f"  - regular_checker_results.json")
    print(f"  - weighted_checker_results.json")
    print(f"  - comparison_results.json")
    print(f"  - regular_checker_report.txt")
    print(f"  - weighted_checker_report.txt")


def main():
    parser = argparse.ArgumentParser(description='Compare spell checkers on test data')
    parser.add_argument('--test-data', type=str, required=True,
                       help='Path to test JSON file')
    parser.add_argument('--regular-model', type=str, default='./outputs/spell_checker.pkl',
                       help='Path to regular spell checker model')
    parser.add_argument('--weighted-model', type=str, default='./outputs/weighted_spell_checker.pkl',
                       help='Path to weighted spell checker model')
    parser.add_argument('--max-distance', type=float, default=2.0,
                       help='Maximum edit distance')
    parser.add_argument('--output', type=str, default='./comparison_results',
                       help='Output directory')
    
    args = parser.parse_args()
    
    compare_spell_checkers(
        test_data_path=args.test_data,
        regular_model_path=args.regular_model,
        weighted_model_path=args.weighted_model,
        output_dir=args.output,
        max_distance=args.max_distance
    )


if __name__ == "__main__":
    main()
