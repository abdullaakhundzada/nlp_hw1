"""
Task 2: Heaps' Law Analysis
Test Heaps' law: V(n) = k * n^β
where V(n) is vocabulary size and n is the number of tokens
"""
import numpy as np
from typing import Dict, List, Tuple
import matplotlib, json, os

matplotlib.use('Agg')  # Use non-interactive backend

import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


class HeapsLawAnalyzer:
    def __init__(self, tokens: List[str]):
        """
        Initialize Heaps' Law analyzer
        
        :param tokens: List of tokens from corpus
        :type tokens: List[str]

        :rtype: None
        """
        self.tokens = tokens
        self.k = None
        self.beta = None
        self.vocabulary_growth = []
        self.token_counts = []
        
    def calculate_vocabulary_growth(self, sample_points: int = 100) -> Tuple[List[int], List[int]]:
        """
        Calculate vocabulary size at different corpus sizes
        
        :param sample_points: Number of points to sample
        :type sample_points: int = 100

        :returns: Tuple of (token_counts, vocabulary_sizes)
        :rtype: Tuple[List[int], List[int]]
        """
        total_tokens = len(self.tokens)
        
        # Create sampling points (logarithmic scale for better distribution)
        if total_tokens < sample_points:
            sample_indices = list(range(1, total_tokens + 1))
        else:
            sample_indices = np.logspace(0, np.log10(total_tokens), sample_points, dtype=int)
            sample_indices = sorted(set(sample_indices))  # Remove duplicates and sort
        
        vocabulary_sizes = []
        vocab_set = set()
        
        current_idx = 0
        for target_idx in sample_indices:
            # Add tokens up to target_idx
            while current_idx < target_idx and current_idx < total_tokens:
                vocab_set.add(self.tokens[current_idx])
                current_idx += 1
            
            vocabulary_sizes.append(len(vocab_set))
        
        self.token_counts = sample_indices
        self.vocabulary_growth = vocabulary_sizes
        
        return sample_indices, vocabulary_sizes
    
    def heaps_law_function(self, n, k, beta):
        """
        Heaps' Law function: V(n) = k * n^β
        
        :param n: Number of tokens
        :param k: Constant
        :param beta: Exponent

        :type n: int
        :type k: float
        :type beta: float
            
        :returns: Vocabulary size
        :rtype: np.float
        """
        return k * np.power(n, beta)
    
    def fit_heaps_law(self) -> Tuple[float, float, float]:
        """
        Fit Heaps' Law to the data and find k and β
        
        :returns: Tuple of (k, beta, r_squared)
        :rtype: Tuple[float, float, float]
        """
        if not self.vocabulary_growth:
            self.calculate_vocabulary_growth()
        
        # Convert to numpy arrays
        n = np.array(self.token_counts)
        V = np.array(self.vocabulary_growth)
        
        # Fit the curve
        try:
            # Initial guess for parameters
            popt, pcov = curve_fit(
                self.heaps_law_function, 
                n, 
                V, 
                p0=[10, 0.5],
                maxfev=10000
            )
            
            self.k, self.beta = popt
            
            # Calculate R-squared
            V_predicted = self.heaps_law_function(n, self.k, self.beta)
            ss_res = np.sum((V - V_predicted) ** 2)
            ss_tot = np.sum((V - np.mean(V)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            
            return self.k, self.beta, r_squared
            
        except Exception as e:
            print(f"Error fitting Heaps' Law: {e}")
            return None, None, None
    
    def plot_heaps_law(self, output_path: str):
        """
        Plot vocabulary growth and Heaps' Law fit
        
        :param output_path: Path to save the plot
        :type output_path: str

        :rtype: None
        """
        if not self.vocabulary_growth:
            self.calculate_vocabulary_growth()
        
        if self.k is None or self.beta is None:
            self.fit_heaps_law()
        
        n = np.array(self.token_counts)
        V = np.array(self.vocabulary_growth)
        V_predicted = self.heaps_law_function(n, self.k, self.beta)
        
        plt.figure(figsize=(12, 6))
        
        # Linear scale plot
        plt.subplot(1, 2, 1)
        plt.plot(n, V, 'b.', label='Actual Data', markersize=4, alpha=0.6)
        plt.plot(n, V_predicted, 'r-', label=f'Heaps\' Law Fit\nV(n) = {self.k:.2f} * n^{self.beta:.3f}', linewidth=2)
        plt.xlabel('Number of Tokens (n)')
        plt.ylabel('Vocabulary Size V(n)')
        plt.title('Heaps\' Law - Linear Scale')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Log-log scale plot
        plt.subplot(1, 2, 2)
        plt.loglog(n, V, 'b.', label='Actual Data', markersize=4, alpha=0.6)
        plt.loglog(n, V_predicted, 'r-', label=f'Heaps\' Law Fit\nlog(V) = log({self.k:.2f}) + {self.beta:.3f}*log(n)', linewidth=2)
        plt.xlabel('Number of Tokens (n)')
        plt.ylabel('Vocabulary Size V(n)')
        plt.title('Heaps\' Law - Log-Log Scale')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Plot saved to {output_path}")
    
    def generate_report(self) -> str:
        """
        Generate a detailed report of Heaps' Law analysis
        
        :returns: Formatted report string
        :rtype: str
        """
        if self.k is None or self.beta is None:
            k, beta, r_squared = self.fit_heaps_law()
        else:
            n = np.array(self.token_counts)
            V = np.array(self.vocabulary_growth)
            V_predicted = self.heaps_law_function(n, self.k, self.beta)
            ss_res = np.sum((V - V_predicted) ** 2)
            ss_tot = np.sum((V - np.mean(V)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
        
        report = f"""
=== HEAPS' LAW ANALYSIS ===

Heaps' Law Formula: V(n) = k * n^β

Parameters Found:
  k (constant):     {self.k:.4f}
  β (beta):         {self.beta:.4f}
  R² (goodness):    {r_squared:.4f}

Interpretation:
- k represents the vocabulary richness
- β represents the rate of vocabulary growth (typically 0.4-0.6 for natural language)
- β = {self.beta:.4f} indicates {'typical' if 0.4 <= self.beta <= 0.6 else 'atypical'} vocabulary growth

Data Points:
  Total tokens analyzed:     {len(self.tokens):,}
  Final vocabulary size:     {self.vocabulary_growth[-1]:,}
  Sample points:             {len(self.token_counts)}

Predicted vs Actual (last 5 points):
"""
        
        for i in range(max(0, len(self.token_counts) - 5), len(self.token_counts)):
            n = self.token_counts[i]
            actual = self.vocabulary_growth[i]
            predicted = self.heaps_law_function(n, self.k, self.beta)
            report += f"  n={n:8,}: Actual={actual:6,}, Predicted={predicted:8.1f}, Error={abs(actual-predicted):6.1f}\n"
        
        return report


def run(tokens: List[str], output_dir: str = './outputs') -> Dict:
    """
    Run Task 2: Heaps' Law analysis
    
    :param tokens: List of tokens from corpus
    :param output_dir: Directory to save outputs

    :type tokens: List[str]
    :type output_dir: str = './outputs'
        
    :returns: Dictionary with Heaps' Law parameters
    :rtype: Dict
    """
    os.makedirs(output_dir, exist_ok=True)
    
    analyzer = HeapsLawAnalyzer(tokens)
    k, beta, r_squared = analyzer.fit_heaps_law()
    
    # Generate plot
    plot_path = os.path.join(output_dir, 'heaps_law_plot.png')
    analyzer.plot_heaps_law(plot_path)
    
    # Save report
    report_path = os.path.join(output_dir, 'task2_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(analyzer.generate_report())
    
    # Save parameters as JSON
    params_path = os.path.join(output_dir, 'heaps_law_params.json')
    with open(params_path, 'w', encoding='utf-8') as f:
        json.dump({
            'k': k,
            'beta': beta,
            'r_squared': r_squared,
            'total_tokens': len(tokens),
            'vocabulary_size': len(set(tokens))
        }, f, indent=2)
    
    print(f"Task 2 completed.")
    print(analyzer.generate_report())
    
    return {
        'k': k,
        'beta': beta,
        'r_squared': r_squared
    }


if __name__ == "__main__":
    # Test with sample tokens
    sample_tokens = ['the', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog'] * 100
    result = run(sample_tokens)
    print(f"k={result['k']:.4f}, beta={result['beta']:.4f}")
