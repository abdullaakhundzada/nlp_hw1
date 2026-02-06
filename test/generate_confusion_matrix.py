"""
Generate Confusion Matrix Visualization from Test Results
This script can be run after test_spell_checker.py to generate the confusion matrix visualization
"""
import json
import argparse
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def generate_confusion_matrix_from_results(results_json_path: str, output_path: str):
    """
    Generate confusion matrix from evaluation results JSON
    
    Args:
        results_json_path: Path to evaluation_results.json
        output_path: Path to save confusion matrix plot
    """
    # Load results
    with open(results_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    results = data['test_results']
    
    # Collect character-level errors
    char_confusions = defaultdict(lambda: defaultdict(int))
    
    for result in results:
        if not result.get('aligned'):
            continue
        
        for correction in result.get('corrections', []):
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
        return
    
    print(f"Found {len(all_confusions)} character confusion patterns")
    print(f"\nTop 10 Character Confusions:")
    for i, (c1, c2, count) in enumerate(top_confusions[:10], 1):
        c1_display = c1 if c1 else '<empty>'
        c2_display = c2 if c2 else '<empty>'
        print(f"  {i}. '{c1_display}' → '{c2_display}': {count} occurrences")
    
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
    
    # Right plot: Confusion matrix heatmap
    unique_chars = set()
    for c1, c2, _ in top_confusions[:15]:
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
        
        # Add text annotations
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
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nConfusion matrix saved to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Generate confusion matrix visualization from test results'
    )
    parser.add_argument('--results', type=str, required=True,
                       help='Path to evaluation_results.json file')
    parser.add_argument('--output', type=str, default='confusion_matrix.png',
                       help='Output path for confusion matrix image')
    
    args = parser.parse_args()
    
    generate_confusion_matrix_from_results(args.results, args.output)


if __name__ == "__main__":
    main()
