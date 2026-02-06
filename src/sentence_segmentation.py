"""
Task 4: Sentence Segmentation Algorithm
Develop and test sentence segmentation for Azerbaijani text
"""
import re, os, json, pickle
from typing import List, Dict


class SentenceSegmenter:
    def __init__(self):
        """
        Initialize sentence segmenter
        
        :rtype: None
        """
        # Common abbreviations in Azerbaijani that shouldn't end sentences
        self.abbreviations = {
            'dr', 'prof', 'mr', 'mrs', 'ms', 'inc', 'ltd', 'co',
            'etc', 'vs', 'i.e', 'e.g', 'cf', 'no', 'st', 'ave',
            'blvd', 'rd', 'govt', 'dept', 'univ', 'resp', 'yəni', 'məs'
        }
        
        # Azerbaijani honorifics and titles
        self.azerbaijani_titles = {
            'cənab', 'xanım', 'müəllim', 'rəis', 'nazir', 
            'direktor', 'akademik', 'general'
        }
        
        self.abbreviations.update(self.azerbaijani_titles)
        
    def is_abbreviation(self, text: str) -> bool:
        """
        Check if text is a known abbreviation
        
        :param text: Text to check
        :type text: str
            
        :returns: whether the text is an abbreviation or not
        :rtype: bool
        """
        text_clean = text.lower().rstrip('.')
        return text_clean in self.abbreviations
    
    def segment(self, text: str) -> List[str]:
        """
        Segment text into sentences
        
        :param text: Input text
        :type text: str
            
        :returns: List of sentences
        :rtype: List[str]
        """
        # Replace newlines with spaces
        text = text.replace('\n', ' ')
        
        # Handle multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        sentences = []
        current_sentence = []
        
        # Split by potential sentence boundaries
        tokens = re.split(r'([.!?]+\s+|[.!?]+$)', text)
        
        i = 0
        while i < len(tokens):
            token = tokens[i]
            
            if not token or token.isspace():
                i += 1
                continue
            
            # Check if this is a sentence boundary
            if re.match(r'^[.!?]+\s*$', token):
                # Check if previous token is an abbreviation
                if current_sentence:
                    last_word = current_sentence[-1].strip().rstrip('.,!?')
                    
                    # If it's an abbreviation and only one period, continue
                    if self.is_abbreviation(last_word) and token.strip() == '.':
                        current_sentence.append(token)
                    else:
                        # End of sentence
                        current_sentence.append(token)
                        sentence = ''.join(current_sentence).strip()
                        if sentence:
                            sentences.append(sentence)
                        current_sentence = []
            else:
                current_sentence.append(token)
            
            i += 1
        
        # Add remaining sentence
        if current_sentence:
            sentence = ''.join(current_sentence).strip()
            if sentence:
                sentences.append(sentence)
        
        return sentences
    
    def segment_advanced(self, text: str) -> List[str]:
        """
        Advanced segmentation with better handling of edge cases
        
        :param text: Input text
        :type text: str
            
        :returns: List of sentences
        :rtype: List[str]
        """
        # Normalize text
        text = text.replace('\n', ' ').strip()
        text = re.sub(r'\s+', ' ', text)
        
        sentences = []
        
        # Split on sentence-ending punctuation followed by space and capital letter
        # or end of string
        pattern = r'([.!?]+)(\s+)(?=[A-ZƏÖÜĞŞÇI]|$)'
        
        parts = re.split(pattern, text)
        
        current = ""
        i = 0
        
        while i < len(parts):
            if i + 2 < len(parts):
                # We have: text, punctuation, space
                text_part = parts[i]
                punct = parts[i + 1]
                space = parts[i + 2]
                
                current += text_part + punct
                
                # Check if this should end a sentence
                words = text_part.strip().split()
                if words:
                    last_word = words[-1].lower().rstrip('.,!?')
                    
                    # Don't break on abbreviations with single period
                    if self.is_abbreviation(last_word) and punct == '.':
                        current += space
                    else:
                        # End sentence
                        if current.strip():
                            sentences.append(current.strip())
                        current = ""
                
                i += 3
            else:
                current += parts[i]
                i += 1
        
        if current.strip():
            sentences.append(current.strip())
        
        return sentences
    
    def evaluate(self, text: str, reference_sentences: List[str]) -> Dict:
        """
        Evaluate segmentation against reference
        
        :param text: Input text
        :param reference_sentences: Correct sentence segmentation

        :type text: str
        :type reference_sentences: List[str]
            
        :returns: Dictionary with evaluation metrics
        :retype: Dict
        """
        predicted_sentences = self.segment_advanced(text)
        
        # Calculate metrics
        correct = 0
        for pred, ref in zip(predicted_sentences, reference_sentences):
            if pred.strip() == ref.strip():
                correct += 1
        
        precision = correct / len(predicted_sentences) if predicted_sentences else 0
        recall = correct / len(reference_sentences) if reference_sentences else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'num_predicted': len(predicted_sentences),
            'num_reference': len(reference_sentences),
            'correct': correct,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
    
    def save_model(self, filepath: str):
        """
        Save segmenter configuration
        
        :param filepath: Path to save the model to
        :type filepath: str

        :rtype: None
        """
        with open(filepath, 'wb') as f:
            pickle.dump({
                'abbreviations': self.abbreviations,
                'azerbaijani_titles': self.azerbaijani_titles
            }, f)
    
    def load_model(self, filepath: str):
        """
        Load segmenter configuration
        
        :param filepath: Path to load the model from
        :type filepath: str

        :rtype: None
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.abbreviations = data['abbreviations']
            self.azerbaijani_titles = data.get('azerbaijani_titles', set())


def run(corpus: str, output_dir: str = './outputs') -> Dict:
    """
    Run Task 4: Sentence segmentation
    
    
    :param corpus: Text corpus
    :param output_dir: Output directory

    :type corpus: str
    :type output_dir: str = './outputs'
        
    :returns: Dictionary with statistics
    :rtype: Dict
    """
    os.makedirs(output_dir, exist_ok=True)
    
    segmenter = SentenceSegmenter()
    sentences = segmenter.segment_advanced(corpus)
    
    # Save model
    model_path = os.path.join(output_dir, 'sentence_segmenter.pkl')
    segmenter.save_model(model_path)
    
    # Statistics
    stats = {
        'total_sentences': len(sentences),
        'avg_sentence_length': sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0,
        'sample_sentences': sentences[:10]
    }
    
    # Save report
    report_path = os.path.join(output_dir, 'task4_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"""
=== SENTENCE SEGMENTATION REPORT ===

Statistics:
  Total sentences: {stats['total_sentences']}
  Average sentence length: {stats['avg_sentence_length']:.2f} words

Sample Sentences:
""")
        for i, sent in enumerate(sentences[:10], 1):
            f.write(f"\n{i}. {sent}\n")
    
    # Save sentences to JSON
    sentences_path = os.path.join(output_dir, 'sentences.json')
    with open(sentences_path, 'w', encoding='utf-8') as f:
        json.dump({'sentences': sentences}, f, ensure_ascii=False, indent=2)
    
    print(f"Task 4 completed. Found {len(sentences)} sentences.")
    print(f"Segmenter saved to {model_path}")
    
    return stats


if __name__ == "__main__":
    # Test with sample text
    sample_text = """
    Azərbaycan Respublikası Cənubi Qafqazda yerləşir. Ölkənin paytaxtı Bakı şəhəridir. 
    Prof. Əliyev bu mövzuda tədqiqat aparır. Azərbaycan dili türk dillər ailəsinə aiddir!
    Xəzər dənizi nə qədər gözəldir? Bəli, həqiqətən gözəldir.
    """
    
    segmenter = SentenceSegmenter()
    sentences = segmenter.segment_advanced(sample_text)
    
    print("Segmented sentences:")
    for i, sent in enumerate(sentences, 1):
        print(f"{i}. {sent}")
