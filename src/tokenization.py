"""
Task 1: Tokenization and Frequency Analysis
Tokenize dataset and calculate token/type frequencies
"""
import re, json, pickle, os
from collections import Counter
from typing import Dict, List


class Tokenizer:
    def __init__(self):
        """
        Initialize tokenizer

        :rtype: None
        """
        self.tokens = []
        self.token_freq = Counter()
        self.type_freq = Counter()
        
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words

        :param text: Input text
        :type text: str
            
        :returns: List of tokens
        :rtype: List[str]
        """
        # Convert to lowercase and split by whitespace and punctuation
        # Keep Azerbaijani characters: ə, ı, ö, ü, ğ, ş, ç
        text = text.lower()
        
        # Split on whitespace and punctuation, but keep words intact
        tokens = re.findall(r'\b[\wəıöüğşç]+\b', text)
        
        return tokens
    
    def process_corpus(self, corpus: str) -> Dict:
        """
        Process entire corpus and calculate statistics
        
        :param corpus: Text corpus
        :type corpus: str
            
        :returns: Dictionary with tokenization statistics
        :rtype: Dict
        """
        self.tokens = self.tokenize(corpus)
        self.token_freq = Counter(self.tokens)
        
        # Types are unique tokens
        types = set(self.tokens)
        
        # Calculate type frequencies (same as token freq for types)
        self.type_freq = self.token_freq
        
        stats = {
            'total_tokens': len(self.tokens),
            'total_types': len(types),
            'type_token_ratio': len(types) / len(self.tokens) if self.tokens else 0,
            'token_frequencies': dict(self.token_freq.most_common(50)),
            'top_10_tokens': self.token_freq.most_common(10),
            'vocabulary_size': len(types)
        }
        
        return stats
    
    def get_frequency_distribution(self) -> Dict[int, int]:
        """
        Get frequency distribution (how many words appear n times)
        
        :returns: Dictionary mapping frequency to count of words with that frequency
        :rtype: Dict[int, int]
        """
        freq_dist = Counter(self.token_freq.values())
        return dict(sorted(freq_dist.items()))
    
    def save_vocabulary(self, filepath: str):
        """
        Save vocabulary and frequencies to file
        
        :param filepath: Path to save vocabulary
        :type filepath: str

        :rtype: None
        """
        vocab_data = {
            'tokens': self.tokens,
            'token_freq': dict(self.token_freq),
            'vocabulary': list(set(self.tokens))
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(vocab_data, f)
        
        # Also save as JSON for readability
        json_path = filepath.replace('.pkl', '.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump({
                'token_freq': dict(self.token_freq.most_common(100)),
                'total_tokens': len(self.tokens),
                'vocabulary_size': len(set(self.tokens))
            }, f, ensure_ascii=False, indent=2)
    
    def load_vocabulary(self, filepath: str):
        """
        Load vocabulary from file
        
        :param filepath: Path to vocabulary file
        :type filepath: str

        :rtype: None
        """
        with open(filepath, 'rb') as f:
            vocab_data = pickle.load(f)
        
        self.tokens = vocab_data['tokens']
        self.token_freq = Counter(vocab_data['token_freq'])
    
    def get_statistics_report(self) -> str:
        """
        Generate a formatted statistics report
        
        :returns: Formatted string with statistics
        :rtype: str
        """
        stats = self.process_corpus(' '.join(self.tokens)) if self.tokens else {}
        
        report = f"""
=== TOKENIZATION STATISTICS ===

Total Tokens: {stats.get('total_tokens', 0):,}
Total Types (Unique Tokens): {stats.get('total_types', 0):,}
Type-Token Ratio: {stats.get('type_token_ratio', 0):.4f}

Top 10 Most Frequent Tokens:
"""
        
        for i, (token, freq) in enumerate(stats.get('top_10_tokens', []), 1):
            report += f"{i:2d}. {token:20s} - {freq:6,} occurrences\n"
        
        return report


def run(corpus: str, output_dir: str = './outputs') -> Dict:
    """
    Runs Tokenization and frequency analysis
    
    Args:
    :param corpus: Text corpus to process
    :param output_dir: Directory to save outputs
    
    :type corpus: str
    :type output_dir: str = './outputs'
    
    :returns: Dictionary with statistics
    :rtype: Dict
    """
    os.makedirs(output_dir, exist_ok=True)
    
    tokenizer = Tokenizer()
    stats = tokenizer.process_corpus(corpus)
    
    # Save vocabulary
    vocab_path = os.path.join(output_dir, 'vocabulary.pkl')
    tokenizer.save_vocabulary(vocab_path)
    
    # Save statistics report
    report_path = os.path.join(output_dir, 'task1_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(tokenizer.get_statistics_report())
    
    print(f"Task 1 completed. Vocabulary saved to {vocab_path}")
    print(tokenizer.get_statistics_report())
    
    return stats
