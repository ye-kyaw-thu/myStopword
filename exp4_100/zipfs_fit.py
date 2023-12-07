## Written by Ye Kyaw Thu, LU Lab., Myanmar

import argparse
from collections import Counter
import numpy as np
from sklearn.linear_model import LinearRegression

def extract_word_frequencies(corpus_filename):
    """Extract word frequencies from the corpus."""
    with open(corpus_filename, 'r', encoding='utf-8') as f:
        words = [word for line in f for word in line.split()]
        frequencies = Counter(words)
    return frequencies.most_common()

def fit_zipfs_law(frequencies):
    """Fit Zipf's law to the data and return C and alpha values."""
    ranks = np.arange(1, len(frequencies) + 1).reshape(-1, 1)
    freqs = np.array([freq for word, freq in frequencies]).reshape(-1, 1)

    # Take log of both ranks and frequencies
    log_ranks = np.log(ranks)
    log_freqs = np.log(freqs)

    # Linear regression
    model = LinearRegression().fit(log_ranks, log_freqs)
    alpha = -model.coef_[0][0]
    C = np.exp(model.intercept_[0])
    
    return C, alpha

def main():
    parser = argparse.ArgumentParser(description="Fit Zipf's law to a corpus and determine C and alpha values.")
    parser.add_argument('-c', '--corpus_filename', required=True, help='the corpus file to read')
    args = parser.parse_args()

    # Extract word frequencies
    frequencies = extract_word_frequencies(args.corpus_filename)

    # Fit Zipf's law and get C and alpha
    C, alpha = fit_zipfs_law(frequencies)

    print(f"C: {C:.4f}")
    print(f"alpha: {alpha:.4f}")

if __name__ == "__main__":
    main()

