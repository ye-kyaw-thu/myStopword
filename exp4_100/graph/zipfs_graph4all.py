## Written by Ye Kyaw Thu, LU Lab., Myanmar

import argparse
from collections import Counter
import matplotlib.pyplot as plt
import os
import glob

def read_file(filename):
    """Read the file and return its content as a list of lines."""
    with open(filename, 'r', encoding='utf-8') as f:
        return f.read().splitlines()

def extract_top_words(corpus_filename, num):
    """Extract the top num words from the corpus."""
    with open(corpus_filename, 'r', encoding='utf-8') as f:
        words = [word for line in f for word in line.split()]
        frequencies = Counter(words).most_common(num)
    return frequencies

def extract_stopword_frequencies(corpus_filename, stopwords):
    """Extract the frequencies of the given stopwords from the corpus."""
    with open(corpus_filename, 'r', encoding='utf-8') as f:
        word_counts = Counter(word for line in f for word in line.split())
    return [(word, word_counts[word]) for word in stopwords]

def main():
    parser = argparse.ArgumentParser(description="Compare Zipf's law, top words in corpus, and extracted stopwords.")
    parser.add_argument('-c', '--corpus_filename', required=True, help='the corpus file to read')
    parser.add_argument('-s', '--stopword_filename_pattern', required=True, help='the wildcard pattern for stopwords files (e.g. *.txt)')
    parser.add_argument('-o', '--output_filename', default='comparison.png', help='filename to save the graph (default: comparison.png)')
    parser.add_argument('-p', '--plot_type', choices=['log', 'linear'], default='log', help='Type of plot: log-log or linear (default: log)')

    args = parser.parse_args()

    # Gather the list of stopword filenames based on the wildcard pattern
    stopword_filenames = glob.glob(args.stopword_filename_pattern)
    
    # Extract top X words from the corpus
    number = len(read_file(stopword_filenames[0]))  # Assuming all stopword files have the same number of stopwords
    frequencies_corpus = extract_top_words(args.corpus_filename, number)

    plt.figure(figsize=(10, 6))

    # Loop through each stopwords file and plot their frequencies
    for stopword_filename in stopword_filenames:
        stopwords = set(read_file(stopword_filename))
        stopword_label = os.path.basename(stopword_filename).replace(".txt", "").replace("_", " ").capitalize()
        frequencies_stopwords = extract_stopword_frequencies(args.corpus_filename, stopwords)
        ranks = range(1, len(frequencies_corpus) + 1)
        freq_stopwords = [freq for word, freq in frequencies_stopwords]

        if args.plot_type == 'log':
            plt.loglog(ranks, freq_stopwords, marker="x", label=stopword_label)
        else:
            plt.plot(ranks, freq_stopwords, marker="x", label=stopword_label)

    # Zipf's law calculation
    C = 4766211.8692
    alpha = 1.3720 
    zipf = [C / (rank**alpha) for rank in ranks]
    if args.plot_type == 'log':
        plt.loglog(ranks, [freq for word, freq in frequencies_corpus], marker="o", label="Top 100 words in corpus")
        plt.loglog(ranks, zipf, linestyle="-", color="r", label="Zipf's Law")
    else:
        plt.plot(ranks, [freq for word, freq in frequencies_corpus], marker="o", label="Top 100 words in corpus")
        plt.plot(ranks, zipf, linestyle="-", color="r", label="Zipf's Law")

    # Plot settings
    plt.title(f"Comparison of Top 100 Words, Extracted Stopwords, and Zipf's Law")
    plt.xlabel('Rank')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig(args.output_filename)
    print(f"Graph saved as '{args.output_filename}'.")

if __name__ == "__main__":
    main()

