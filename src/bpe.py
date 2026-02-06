"""
Task 3: Byte Pair Encoding (BPE) Tokenization
Implement BPE algorithm for subword tokenization
"""
import re, os, json, pickle
from collections import Counter
from typing import Dict, List, Tuple


class BPETokenizer:
    def __init__(self, num_merges: int = 1000):
        """
        Initialize BPE tokenizer
        
        :param num_merges: Number of merge operations to perform
        :type num_merges: int

        :rtype: None
        """
        self.num_merges = num_merges
        self.vocab = {}
        self.merges = []
        self.token_to_id = {}
        self.id_to_token = {}
        
    def get_stats(self, word_freqs: Dict[Tuple[str, ...], int]) -> Counter:
        """
        Count frequency of adjacent pairs in vocabulary
        
        :param word_freqs: Dictionary mapping word tuples to frequencies
        :type word_freqs: Dict[Tuple[str, ...], int]
            
        :returns: Counter of pair frequencies
        :rtype: Counter
        """
        pairs = Counter()
        
        for word, freq in word_freqs.items():
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                pairs[pair] += freq
        
        return pairs
    
    def merge_vocab(self, pair: Tuple[str, str], word_freqs: Dict[Tuple[str, ...], int]) -> Dict[Tuple[str, ...], int]:
        """
        Merge the most frequent pair in vocabulary
        
        :param pair: Pair to merge
        :param word_freqs: Current vocabulary
        
        :type pair: Tuple[str, str]
        :type word_freqs: Dict[Tuple[str, ...], int]

        :returns: Updated vocabulary
        :rtype: Dict[Tuple[str, ...], int]
        """
        new_word_freqs = {}
        replacement = ''.join(pair)
        
        for word, freq in word_freqs.items():
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
                    new_word.append(replacement)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            
            new_word_freqs[tuple(new_word)] = freq
        
        return new_word_freqs
    
    def train(self, corpus: str):
        """
        Train BPE on corpus
        
        :param corpus: Training text
        :type corpus: str

        :rtype: None
        """
        # Tokenize into words
        words = re.findall(r'\b[\wəıöüğşç]+\b', corpus.lower())
        
        # Initialize vocabulary with character-level tokens
        word_freqs = Counter(words)
        
        # Split words into characters with end-of-word marker
        vocab = {}
        for word, freq in word_freqs.items():
            # Add space between characters and </w> at the end
            vocab[tuple(list(word) + ['</w>'])] = freq
        
        # Perform merges
        self.merges = []
        
        for i in range(self.num_merges):
            pairs = self.get_stats(vocab)
            
            if not pairs:
                break
            
            # Get most frequent pair
            best_pair = max(pairs, key=pairs.get)
            
            # Merge the pair
            vocab = self.merge_vocab(best_pair, vocab)
            self.merges.append(best_pair)
            
            if (i + 1) % 100 == 0:
                print(f"Completed {i + 1} merges...")
        
        # Build final vocabulary
        self.vocab = vocab
        
        # Create token mappings
        all_tokens = set()
        for word in vocab.keys():
            all_tokens.update(word)
        
        # Add base characters
        base_chars = set('abcdefghijklmnopqrstuvwxyzəıöüğşç</w>')
        all_tokens.update(base_chars)
        
        self.token_to_id = {token: idx for idx, token in enumerate(sorted(all_tokens))}
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}
        
        print(f"Training completed. Vocabulary size: {len(self.token_to_id)}")
    
    def encode_word(self, word: str) -> List[str]:
        """
        Encode a single word using learned BPE merges
        
        :param word: Word to encode
        :type word: str
        
        :returns: List of subword tokens
        :type: List[str]
        """
        word = word.lower()
        word = tuple(list(word) + ['</w>'])
        
        # Apply merges in order
        for pair in self.merges:
            if len(word) < 2:
                break
            
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
                    new_word.append(''.join(pair))
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            
            word = tuple(new_word)
        
        return list(word)
    
    def encode(self, text: str) -> List[str]:
        """
        Encode text into BPE tokens
        
        :param text: Text to encode
        :type text: str

        :returns: List of BPE tokens
        :rtype: List[str]
        """
        words = re.findall(r'\b[\wəıöüğşç]+\b', text.lower())
        
        tokens = []
        for word in words:
            tokens.extend(self.encode_word(word))
        
        return tokens
    
    def decode(self, tokens: List[str]) -> str:
        """
        Decode BPE tokens back to text
        
        :param tokens: List of BPE tokens
        :type tokens: List[str]
            
        :returns: Decoded text
        :rtype: str
        """
        text = ''.join(tokens)
        text = text.replace('</w>', ' ')
        return text.strip()
    
    def save_model(self, filepath: str):
        """
        Save BPE model to file
        
        :param filepath: Path to save model
        :type filepath: str

        :rtype: None
        """
        model_data = {
            'num_merges': self.num_merges,
            'merges': self.merges,
            'vocab': {' '.join(k): v for k, v in self.vocab.items()},
            'token_to_id': self.token_to_id,
            'id_to_token': self.id_to_token
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        # Also save as JSON for inspection
        json_path = filepath.replace('.pkl', '.json')
        json_data = {
            'num_merges': self.num_merges,
            'vocabulary_size': len(self.token_to_id),
            'sample_merges': [f"{p[0]}+{p[1]}" for p in self.merges[:20]],
            'total_merges': len(self.merges)
        }
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        
        print(f"BPE model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """
        Load BPE model from file
        
        :param filepath: Path to model file
        :type filepath: str

        :rtype: None
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.num_merges = model_data['num_merges']
        self.merges = model_data['merges']
        self.vocab = {tuple(k.split()): v for k, v in model_data['vocab'].items()}
        self.token_to_id = model_data['token_to_id']
        self.id_to_token = model_data['id_to_token']
        
        print(f"BPE model loaded from {filepath}")
    
    def get_statistics(self) -> Dict:
        """
        Get BPE model statistics
        
        :returns: Dictionary with statistics
        :rtype: Dict
        """
        return {
            'num_merges': len(self.merges),
            'vocabulary_size': len(self.token_to_id),
            'num_words_in_vocab': len(self.vocab),
            'sample_tokens': list(self.token_to_id.keys())[:20]
        }


def run(corpus: str, num_merges: int = 1000, output_dir: str = './outputs') -> BPETokenizer:
    """
    Run Task 3: BPE tokenization
    
    Args:
    :param corpus: Text corpus
    :param num_merges: Number of BPE merges
    :param output_dir: Output directory

    :type corpus: str        
    :type num_merges: int = 1000        
    :type output_dir: str = "./outputs"

    :returns: Trained BPE tokenizer
    :rtype: BPETokenizer
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Training BPE with {num_merges} merges...")
    bpe = BPETokenizer(num_merges=num_merges)
    bpe.train(corpus)
    
    # Save model
    model_path = os.path.join(output_dir, 'bpe_model.pkl')
    bpe.save_model(model_path)
    
    # Test encoding
    sample_text = corpus[:200] if len(corpus) > 200 else corpus
    encoded = bpe.encode(sample_text)
    decoded = bpe.decode(encoded)
    
    # Save report
    report_path = os.path.join(output_dir, 'task3_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"""
=== BYTE PAIR ENCODING (BPE) REPORT ===

Configuration:
  Number of merges: {num_merges}
  Vocabulary size: {len(bpe.token_to_id)}

Sample Encoding/Decoding:
  Original: {sample_text[:100]}...
  Encoded tokens: {encoded[:20]}...
  Decoded: {decoded[:100]}...

Top 10 Merges:
""")
        for i, (a, b) in enumerate(bpe.merges[:10], 1):
            f.write(f"  {i}. '{a}' + '{b}' -> '{a}{b}'\n")
    
    print(f"Task 3 completed. BPE model saved to {model_path}")
    
    return bpe


if __name__ == "__main__":
    # Test with sample text
    sample_corpus = """
    Azərbaycan Respublikası Cənubi Qafqazda yerləşir. 
    Azərbaycan dili türk dillər ailəsinə aiddir.
    Bakı Azərbaycanın paytaxtıdır.
    """ * 10
    
    bpe = run(sample_corpus, num_merges=50)
    
    # Test encoding
    test_text = "Azərbaycan gözəldir"
    encoded = bpe.encode(test_text)
    decoded = bpe.decode(encoded)
    print(f"\nTest: '{test_text}'")
    print(f"Encoded: {encoded}")
    print(f"Decoded: '{decoded}'")
