"""
Data Loader Module
Loads and processes JSON files from the dataset
"""
import json
import os
from pathlib import Path
from typing import List, Dict, Tuple


class DataLoader:
    def __init__(self, data_folder: str):
        """
        Initialize DataLoader with path to JSON files
        
        Args:
            data_folder: Path to folder containing JSON files
        """
        self.data_folder = Path(data_folder)
        self.raw_data = []
        self.text_corpus = ""
        
    def load_data(self) -> List[Dict]:
        """
        Load all JSON files from the data folder
        
        Returns:
            List of dictionaries with page_number and content
        """
        if not self.data_folder.exists():
            raise FileNotFoundError(f"Data folder not found: {self.data_folder}")
        
        json_files = list(self.data_folder.glob("*.json"))
        
        if not json_files:
            raise ValueError(f"No JSON files found in {self.data_folder}")
        
        all_data = []
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    # Handle both list and single dict formats
                    if isinstance(data, list):
                        all_data.extend(data)
                    else:
                        all_data.append(data)
                        
            except Exception as e:
                print(f"Error loading {json_file}: {e}")
                continue
        
        self.raw_data = all_data
        return all_data
    
    def get_text_corpus(self) -> str:
        """
        Extract all text content from loaded data
        
        Returns:
            Combined text corpus
        """
        if not self.raw_data:
            self.load_data()
        
        texts = []
        for item in self.raw_data:
            if 'content' in item:
                texts.append(str(item['content']))
        
        self.text_corpus = '\n'.join(texts)
        return self.text_corpus
    
    def get_statistics(self) -> Dict:
        """
        Get basic statistics about the dataset
        
        Returns:
            Dictionary with dataset statistics
        """
        if not self.raw_data:
            self.load_data()
        
        corpus = self.get_text_corpus()
        
        return {
            'total_files': len(list(self.data_folder.glob("*.json"))),
            'total_pages': len(self.raw_data),
            'total_characters': len(corpus),
            'total_lines': corpus.count('\n') + 1
        }


def create_sample_data(output_folder: str, num_files: int = 3):
    """
    Create sample Azerbaijani JSON data for testing
    
    Args:
        output_folder: Path to save sample files
        num_files: Number of sample files to create
    """
    sample_texts = [
        [
            {"page_number": 1, "content": "Azərbaycan Respublikası Cənubi Qafqazda yerləşir. Ölkənin paytaxtı Bakı şəhəridir."},
            {"page_number": 2, "content": "Azərbaycan dili türk dillər ailəsinə aiddir. Bu dil milyonlarla insan tərəfindən danışılır."},
        ],
        [
            {"page_number": 1, "content": "Xəzər dənizi dünyanın ən böyük gölüdür. Azərbaycan sahilləri çox gözəldir."},
            {"page_number": 2, "content": "Bakının Qədim şəhəri UNESCO-nun ümumdünya irsi siyahısındadır."},
        ],
        [
            {"page_number": 1, "content": "Azərbaycan xalqı qonaqpərvərliyi ilə məşhurdur. Milli mətbəx çox zəngindir."},
            {"page_number": 2, "content": "Plov, dolma, qutab ən məşhur milli yeməklərdir. Çay içmək mədəniyyətin bir hissəsidir."},
        ]
    ]
    
    os.makedirs(output_folder, exist_ok=True)
    
    for i in range(min(num_files, len(sample_texts))):
        filepath = os.path.join(output_folder, f"sample_{i+1}.json")
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(sample_texts[i], f, ensure_ascii=False, indent=2)
    
    print(f"Created {num_files} sample files in {output_folder}")
