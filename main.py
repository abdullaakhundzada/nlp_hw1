"""
Main Script - Run All Tasks
Executes all NLP tasks on Azerbaijani corpus
"""
import os, sys, json, time
import src.data_loader as data_loader
import src.extra_weighted as extra_weighted
import src.heaps_law as heaps_law
import src.bpe as bpe
import src.tokenization as tokenization
import src.sentence_segmentation as sentence_segmentation
import src.spell_checking as spell_checking

def run_all_tasks(data_folder: str, output_dir: str = './outputs', use_sample: bool = False):
    """
    Run all NLP tasks
    
    Args:
        data_folder: Path to folder containing JSON files
        output_dir: Directory for outputs
        use_sample: If True, create and use sample data
    """
    print("=" * 80)
    print("AZERBAIJANI NLP PROCESSING PIPELINE")
    print("=" * 80)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print("\n[1/7] Loading data...")
    
    if use_sample:
        print("Creating sample data...")
        data_loader.create_sample_data(data_folder, num_files=3)
    
    loader = data_loader.DataLoader(data_folder)
    
    try:
        loader.load_data()
        corpus = loader.get_text_corpus()
        stats = loader.get_statistics()
        
        print(f"✓ Loaded {stats['total_files']} files")
        print(f"✓ Total pages: {stats['total_pages']}")
        print(f"✓ Total characters: {stats['total_characters']:,}")
        
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return False
    
    # Task 1: Tokenization
    print("\n[2/7] Task 1: Tokenization and Frequency Analysis...")
    try:
        start_time = time.time()
        tokenizer = tokenization.Tokenizer()
        task1_stats = tokenizer.process_corpus(corpus)
        tokenizer.save_vocabulary(os.path.join(output_dir, 'vocabulary.pkl'))
        
        print(f"✓ Total tokens: {task1_stats['total_tokens']:,}")
        print(f"✓ Total types: {task1_stats['total_types']:,}")
        print(f"✓ Type-Token Ratio: {task1_stats['type_token_ratio']:.4f}")
        print(f"✓ Completed in {time.time() - start_time:.2f}s")
        
    except Exception as e:
        print(f"✗ Error in Task 1: {e}")
        return False
    
    # Task 2: Heaps' Law
    print("\n[3/7] Task 2: Heaps' Law Analysis...")
    try:
        start_time = time.time()
        task2_result = heaps_law.run(tokenizer.tokens, output_dir)
        
        print(f"✓ k = {task2_result['k']:.4f}")
        print(f"✓ β = {task2_result['beta']:.4f}")
        print(f"✓ R² = {task2_result['r_squared']:.4f}")
        print(f"✓ Completed in {time.time() - start_time:.2f}s")
        
    except Exception as e:
        print(f"✗ Error in Task 2: {e}")
        return False
    
    # Task 3: BPE
    print("\n[4/7] Task 3: Byte Pair Encoding...")
    try:
        start_time = time.time()
        # Use fewer merges for small datasets
        num_merges = min(1000, max(100, task1_stats['total_types'] // 10))
        BPE = bpe.run(corpus, num_merges=num_merges, output_dir=output_dir)
        
        print(f"✓ Vocabulary size: {len(BPE.token_to_id)}")
        print(f"✓ Number of merges: {len(BPE.merges)}")
        print(f"✓ Completed in {time.time() - start_time:.2f}s")
        
    except Exception as e:
        print(f"✗ Error in Task 3: {e}")
        return False
    
    # Task 4: Sentence Segmentation
    print("\n[5/7] Task 4: Sentence Segmentation...")
    try:
        start_time = time.time()
        task4_stats = sentence_segmentation.run(corpus, output_dir)
        
        print(f"✓ Total sentences: {task4_stats['total_sentences']}")
        print(f"✓ Avg sentence length: {task4_stats['avg_sentence_length']:.2f} words")
        print(f"✓ Completed in {time.time() - start_time:.2f}s")
        
    except Exception as e:
        print(f"✗ Error in Task 4: {e}")
        return False
    
    # Task 5: Spell Checker
    print("\n[6/7] Task 5: Spell Checker (Levenshtein Distance)...")
    try:
        start_time = time.time()
        spell_checker = spell_checking.run(corpus, output_dir)
        
        print(f"✓ Vocabulary size: {len(spell_checker.vocabulary)}")
        print(f"✓ Completed in {time.time() - start_time:.2f}s")
        
    except Exception as e:
        print(f"✗ Error in Task 5: {e}")
        return False
    
    # Extra Task: Weighted Edit Distance
    print("\n[7/7] Extra Task: Weighted Edit Distance with Confusion Matrix...")
    try:
        start_time = time.time()
        weighted_checker = extra_weighted.run(corpus, output_dir)
        
        print(f"✓ Confusion patterns learned: {weighted_checker.confusion_matrix.total_errors}")
        print(f"✓ Completed in {time.time() - start_time:.2f}s")
        
    except Exception as e:
        print(f"✗ Error in Extra Task: {e}")
        return False
    
    # Generate summary report
    print("\n" + "=" * 80)
    print("GENERATING FINAL REPORT...")
    print("=" * 80)
    
    summary = {
        'data_statistics': stats,
        'task1': {
            'total_tokens': task1_stats['total_tokens'],
            'total_types': task1_stats['total_types'],
            'type_token_ratio': task1_stats['type_token_ratio']
        },
        'task2': {
            'k': task2_result['k'],
            'beta': task2_result['beta'],
            'r_squared': task2_result['r_squared']
        },
        'task3': {
            'vocabulary_size': len(BPE.token_to_id),
            'num_merges': len(BPE.merges)
        },
        'task4': {
            'total_sentences': task4_stats['total_sentences'],
            'avg_sentence_length': task4_stats['avg_sentence_length']
        },
        'task5': {
            'vocabulary_size': len(spell_checker.vocabulary)
        },
        'extra_task': {
            'confusion_patterns': weighted_checker.confusion_matrix.total_errors
        }
    }
    
    # Save summary
    summary_path = os.path.join(output_dir, 'summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ All tasks completed successfully!")
    print(f"✓ Results saved to: {output_dir}")
    print(f"✓ Summary saved to: {summary_path}")
    
    return True


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Azerbaijani NLP tasks')
    parser.add_argument('--data', type=str, default='./data',
                       help='Path to data folder')
    parser.add_argument('--output', type=str, default='./outputs',
                       help='Path to output folder')
    parser.add_argument('--sample', action='store_true',
                       help='Use sample data')
    
    args = parser.parse_args()
    
    success = run_all_tasks(
        data_folder=args.data,
        output_dir=args.output,
        use_sample=args.sample
    )
    
    if success:
        print("\n✓ Pipeline completed successfully!")
        return 0
    else:
        print("\n✗ Pipeline failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
