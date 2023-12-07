import argparse
from collections import Counter
import matplotlib.pyplot as plt
import os

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

def draw_comparison(frequencies_corpus, frequencies_stopwords, output_filename, stopword_label, plot_type='log'):
    """Plot and save the comparison graph based on word frequencies."""
    ranks = range(1, len(frequencies_corpus) + 1)
    freq_corpus = [freq for word, freq in frequencies_corpus]

    # For stopwords
    freq_stopwords = [freq for word, freq in frequencies_stopwords]

    # Zipf's law calculation
    #zipf = [freq_corpus[0] / rank for rank in ranks]

    # from the paper:
    #C = 0.017
    #alpha = 0.7

    # I calculated for Myanmar corpus
    C = 4766211.8692
    alpha = 1.3720 
    zipf = [C / (rank**alpha) for rank in ranks]

    plt.figure(figsize=(10, 6))
    
    if plot_type == 'log':
        plt.loglog(ranks, freq_corpus, marker="o", label="Top 100 words in corpus")
        plt.loglog(ranks, freq_stopwords, marker="x", label=stopword_label)
        plt.loglog(ranks, zipf, linestyle="-", color="r", label="Zipf's Law")
    else:
        plt.plot(ranks, freq_corpus, marker="o", label="Top 100 words in corpus")
        plt.plot(ranks, freq_stopwords, marker="x", label=stopword_label)
        plt.plot(ranks, zipf, linestyle="-", color="r", label="Zipf's Law")

    plt.title(f"Comparison of Top 100 Words, Extracted Stopwords, and Zipf's Law")
    plt.xlabel('Rank')
    plt.ylabel('Frequency')
    plt.legend()

    plt.savefig(output_filename)
    print(f"Graph saved as '{output_filename}'.")

def main():
    parser = argparse.ArgumentParser(description="Compare Zipf's law, top words in corpus, and extracted stopwords.")
    parser.add_argument('-c', '--corpus_filename', required=True, help='the corpus file to read')
    parser.add_argument('-s', '--stopword_filename', required=True, help='the stopwords file to read')
    parser.add_argument('-o', '--output_filename', default='comparison.png', help='filename to save the graph (default: comparison.png)')
    parser.add_argument('-p', '--plot_type', choices=['log', 'linear'], default='log', help='Type of plot: log-log or linear (default: log)')

    args = parser.parse_args()

    # Load stopwords
    stopwords = set(read_file(args.stopword_filename))
    number = len(stopwords)

    # Create a label for the stopwords using the filename
    stopword_label = os.path.basename(args.stopword_filename).replace(".txt", "").replace("_", " ").capitalize()

    # Extract top X words from the corpus
    frequencies_corpus = extract_top_words(args.corpus_filename, number)

    # Extract frequencies of the given stopwords from the corpus
    frequencies_stopwords = extract_stopword_frequencies(args.corpus_filename, stopwords)

    # Draw and save the comparison graph
    draw_comparison(frequencies_corpus, frequencies_stopwords, args.output_filename, stopword_label, args.plot_type)


if __name__ == "__main__":
    main()

