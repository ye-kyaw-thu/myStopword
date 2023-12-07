# -*- coding: utf-8 -*-
#print("Script started!")
"""
Created on Thu Aug 10 16:04:01 2023

@author: Ye Kyaw Thu
How to run:
python stopword_with_5methods.py corpus.txt stopwords.txt --method normalised_tf

# for document level
python stopword_with_5methods.py ..\corpus\word-segmentation\segmentation-data-updated2.1k-group.txt stopwords_idf.txt --method idf
python stopword_with_5methods.py ..\corpus\word-segmentation\segmentation-data-updated2.1k-group.txt stopwords_normalised_idf.txt --method normalised_idf

# with values
python stopword_with_5methods.py ..\corpus\word-segmentation\segmentation-data-updated2 stopwords_tf_freq.txt --method tf --frequency
python stopword_with_5methods.py ..\corpus\word-segmentation\segmentation-data-updated2 stopwords_normalised_tf_freq.txt --method normalised_tf --frequency

ဒီ idf, normalized_idf က တစ်ခါ run ရင် ရလဒ်က တစ်မျိုး ပြောင်းနိုင်တယ်ဆိုတာကို မမေ့နဲ့
ပြီးတော့ လိုင်း ၁၀၀၀ ကို စာကြောင်း တစ်ကြောင်း လုပ်ထားတဲ့ ဖိုင်နဲ့ run ဖို့ကိုလည်း မမေ့နဲ့
python stopword_with_5methods.py ..\corpus\word-segmentation\segmentation-data-updated2.1k-group.txt stopwords_idf_freq.txt --method idf --frequency
python stopword_with_5methods.py ..\corpus\word-segmentation\segmentation-data-updated2.1k-group.txt stopwords_normalised_idf_freq.txt --method normalised_idf --frequency

tf-idf ကို run မယ်ဆိုရင်လည်း 1000 as one sentence corpus နဲ့ run ရမယ်။
python stopword_with_5methods.py ..\corpus\word-segmentation\segmentation-data-updated2.1k-group.txt stopwords_tfidf_freq.txt --method tf_idf --frequency

python stopword_with_5methods.py ..\corpus\word-segmentation\segmentation-data-updated2 stopwords_term_based_random_sampling_freq.txt --method term_based_random_sampling --frequency
python stopword_with_5methods.py ..\corpus\word-segmentation\segmentation-data-updated2 stopwords_pmi_freq.txt --method pmi_based_stopwords --frequency
python stopword_with_5methods.py ..\corpus\word-segmentation\segmentation-data-updated2 stopwords_entropy_freq.txt --method entropy --frequency
python stopword_with_5methods.py ..\corpus\word-segmentation\segmentation-data-updated2 stopwords_variance_freq.txt --method variance --frequency
python stopword_with_5methods.py ..\corpus\word-segmentation\segmentation-data-updated2 stopwords_word2vec_freq.txt --method word2vec_based --frequency

You need to install: pip install networkx

## Combinations

tf+entropy
python stopword_with_5methods.py ..\corpus\word-segmentation\segmentation-data-updated2 stopwords_tf-entropy_freq.txt --method tf entropy --frequency



Note for me:
According to my understanding/running results, the following methods
can be used for stopword extraction for Myanmar language (with the corpus formatted with sentence by sentece)

tf, normalised_tf, term_based_random_sampling, 
entropy, cooccurrence_network, lm_based (1gram)

Evaluation: paraphrasing? hatespeech?

"""

import argparse
from collections import Counter, defaultdict
import math
import random
import networkx as nx
from gensim.models import Word2Vec
#from gensim.models.keyedvectors import WordEmbeddingsKeyedVectors
import fasttext
import numpy as np

def term_frequency(corpus_filename):
    with open(corpus_filename, 'r', encoding='utf-8') as file:
        words = file.read().split()
    tf = Counter(words)
    return tf

def normalised_tf(corpus_filename):
    tf = term_frequency(corpus_filename)
    max_tf = max(tf.values())
    normalised_tf = {word: freq / max_tf for word, freq in tf.items()}
    return normalised_tf

def idf(corpus_filename):
    with open(corpus_filename, 'r', encoding='utf-8') as file:
        documents = file.readlines()
    total_documents = len(documents)
    word_document_count = Counter()
    for document in documents:
        words = set(document.split())
        word_document_count.update(words)
    idf_values = {word: math.log(total_documents / count) for word, count in word_document_count.items()}
    return idf_values

def normalised_idf(corpus_filename):
    idf_values = idf(corpus_filename)
    max_idf = max(idf_values.values())
    normalised_idf = {word: value / max_idf for word, value in idf_values.items()}
    return normalised_idf

def tf_idf(corpus_filename):
    with open(corpus_filename, 'r', encoding='utf-8') as file:
        documents = file.readlines()

    all_tf = [Counter(doc.split()) for doc in documents]
    idf_values = idf(corpus_filename)
    
    tfidf = defaultdict(float)
    for tf in all_tf:
        for word, freq in tf.items():
            tfidf[word] += freq * idf_values[word]  # Aggregate over all documents

    return tfidf

def term_based_random_sampling(corpus_filename):
    with open(corpus_filename, 'r', encoding='utf-8') as file:
        words = file.read().split()
    random_words = random.sample(words, 1000)
    return Counter(random_words)

def pmi_based_stopwords(corpus_filename):
    with open(corpus_filename, 'r', encoding='utf-8') as file:
        words = file.read().split()

    total_words = len(words)
    word_counts = Counter(words)
    co_occur_counts = defaultdict(Counter)

    for i, word in enumerate(words[:-1]):
        for j in range(i+1, min(i+6, total_words)):  # considering a window of next 5 words for co-occurrence
            co_occur_counts[word][words[j]] += 1

    pmi_scores = defaultdict(float)
    for word, co_occur in co_occur_counts.items():
        for co_word, count in co_occur.items():
            p_x = word_counts[word] / total_words
            p_y = word_counts[co_word] / total_words
            p_xy = count / total_words
            pmi = math.log(p_xy / (p_x * p_y), 2) if p_xy != 0 else 0
            pmi_scores[word] += pmi

    # get average pmi for each word
    avg_pmi_scores = {word: score / len(co_occur_counts[word]) for word, score in pmi_scores.items()}

    return avg_pmi_scores

def entropy_based_stopwords(corpus_filename):
    with open(corpus_filename, 'r', encoding='utf-8') as file:
        documents = file.readlines()

    term_frequencies = defaultdict(Counter)
    num_docs = len(documents)

    for doc in documents:
        counter = Counter(doc.split())
        for term, freq in counter.items():
            term_frequencies[term][freq] += 1

    term_entropies = {}
    for term, freq_dist in term_frequencies.items():
        entropy = -sum([(freq / num_docs) * math.log2(freq / num_docs) for freq in freq_dist.values()])
        term_entropies[term] = entropy

    return term_entropies

def variance_based_stopwords(corpus_filename):
    with open(corpus_filename, 'r', encoding='utf-8') as file:
        documents = file.readlines()

    # Initialize a dictionary to store word frequencies across documents
    word_freqs_across_docs = defaultdict(list)

    # Count word frequencies for each document
    for doc in documents:
        word_counts = Counter(doc.split())
        for word in word_counts:
            word_freqs_across_docs[word].append(word_counts[word])

    # Compute variance for each word
    word_variances = {}
    for word, freqs in word_freqs_across_docs.items():
        mean = sum(freqs) / len(freqs)
        variance = sum([(f - mean) ** 2 for f in freqs]) / len(freqs)
        word_variances[word] = variance

    return word_variances

def cooccurrence_network_stopwords(corpus_filename, window_size=5):
    G = nx.Graph()

    with open(corpus_filename, 'r', encoding='utf-8') as file:
        documents = file.readlines()

    for doc in documents:
        words = doc.split()
        for i, word in enumerate(words):
            start = max(0, i - window_size)
            end = min(len(words), i + window_size)
            neighbors = words[start:i] + words[i + 1:end]
            for neighbor in neighbors:
                if G.has_edge(word, neighbor):
                    G[word][neighbor]['weight'] += 1
                else:
                    G.add_edge(word, neighbor, weight=1)

    # Use degree centrality to identify potential stopwords
    centrality_scores = nx.degree_centrality(G)

    return centrality_scores

def train_ngram_model(corpus_filename, n=2):
    """
    Train an n-gram language model.
    Returns a dictionary of n-grams and their probabilities.
    """
    with open(corpus_filename, 'r', encoding='utf-8') as file:
        words = file.read().split()

    # Get n-grams from the corpus
    ngrams = [tuple(words[i:i+n]) for i in range(len(words)-n+1)]
    
    # Count n-grams and their contexts
    ngram_counts = Counter(ngrams)
    context_counts = defaultdict(int)
    for ngram in ngrams:
        context = ngram[:-1]
        context_counts[context] += 1

    # Calculate probabilities
    ngram_probs = {ngram: count/context_counts[ngram[:-1]] for ngram, count in ngram_counts.items()}

    return ngram_probs

def lm_based_stopwords(corpus_filename, n=1):
    """
    Use an n-gram language model to score words.
    Returns a dictionary of words and their average probabilities.
    """
    ngram_probs = train_ngram_model(corpus_filename, n=n)

    word_scores = defaultdict(float)
    word_counts = defaultdict(int)

    for ngram, prob in ngram_probs.items():
        word = ngram[-1]
        word_scores[word] += prob
        word_counts[word] += 1

    # Calculate average probabilities
    avg_word_probs = {word: score/word_counts[word] for word, score in word_scores.items()}

    return avg_word_probs

def word2vec_based_stopwords(corpus_filename, vector_size=100, window=5, min_count=1, workers=4):
    # Read the corpus and split it into sentences
    with open(corpus_filename, 'r', encoding='utf-8') as file:
        sentences = [line.strip().split() for line in file]

    # Train the Word2Vec model
    model = Word2Vec(sentences, vector_size=vector_size, window=window, min_count=min_count, workers=workers)

    # List of unique words in corpus
    unique_words = model.wv.index_to_key

    # Compute the average embedding of the entire corpus
    avg_vector = sum([model.wv[word] for word in unique_words]) / len(unique_words)

    # Compute cosine similarities of each word's embedding with the average embedding
    cosine_similarities = {}
    for word in unique_words:
        word_vector = model.wv[word]
        cosine_similarity = (word_vector @ avg_vector) / (model.wv.vector_size) # '@' is the dot product
        cosine_similarities[word] = cosine_similarity

    return cosine_similarities

def word2vec_frequency_based_stopwords(corpus_filename, vector_size=100, window=5, min_count=1, workers=4, alpha=0.5):
    """
    Combine Word2Vec embeddings with term frequency to compute a new measure for stopwords.
    alpha: A hyperparameter to decide the weight of frequency vs. embedding similarity. (0 <= alpha <= 1)
    """
    # Train Word2Vec and get cosine similarities
    cosine_similarities = word2vec_based_stopwords(corpus_filename, vector_size=vector_size, window=window, min_count=min_count, workers=workers)
    
    # Get term frequencies
    term_frequencies = term_frequency(corpus_filename)
    max_tf = max(term_frequencies.values())

    # Normalize term frequencies
    normalized_term_frequencies = {word: freq / max_tf for word, freq in term_frequencies.items()}

    # Combine the two measures
    combined_scores = {}
    for word in cosine_similarities:
        combined_scores[word] = alpha * normalized_term_frequencies.get(word, 0) + (1 - alpha) * cosine_similarities[word]

    return combined_scores


def fasttext_based_stopwords(corpus_filename):
    # Train the FastText model
    model = fasttext.train_unsupervised(corpus_filename, model='skipgram', minn=3, maxn=6, dim=100)

    # List of unique words in corpus
    with open(corpus_filename, 'r', encoding='utf-8') as file:
        words = list(set(file.read().split()))

    # Compute the average embedding of the entire corpus
    avg_embedding = np.mean([model[word] for word in words], axis=0)

    # Compute cosine similarities of each word's embedding with the average embedding
    cosine_similarities = {}
    for word in words:
        word_embedding = model[word]
        cosine_similarity = np.dot(word_embedding, avg_embedding) / (np.linalg.norm(word_embedding) * np.linalg.norm(avg_embedding))
        cosine_similarities[word] = cosine_similarity

    return cosine_similarities

def fasttext_frequency_based_stopwords(corpus_filename):
    # 1. Calculate term frequency
    tf = term_frequency(corpus_filename)
    max_tf = max(tf.values())

    # 2. Train FastText model
    model = fasttext.train_unsupervised(corpus_filename, model='skipgram', minn=3, maxn=6, dim=100)

    # 3. Calculate the average embedding of the entire corpus
    with open(corpus_filename, 'r', encoding='utf-8') as file:
        words = list(set(file.read().split()))
    avg_embedding = np.mean([model[word] for word in words], axis=0)

    # 4. Compute cosine similarities of each word's embedding with the average embedding
    cosine_similarities = {}
    for word in words:
        word_embedding = model[word]
        cosine_similarity = np.dot(word_embedding, avg_embedding) / (np.linalg.norm(word_embedding) * np.linalg.norm(avg_embedding))
        
        # Combining frequency with cosine similarity
        composite_score = (tf[word] / max_tf) * cosine_similarity
        cosine_similarities[word] = composite_score

    return cosine_similarities


def hybrid_stopwords(corpus_filename, methods, method_map):
    all_stopwords = {}
    
    for method in methods:
        method_stopwords = method_map[method](corpus_filename)
        for word, score in method_stopwords.items():
            all_stopwords[word] = all_stopwords.get(word, 0) + score

    return all_stopwords


#def hybrid_method(corpus_filename, methods):
def hybrid_method(corpus_filename, methods, method_map):
    print("Entering hybrid_stopwords function...")

    """Combine multiple methods to derive a consensus list of stopwords."""
    all_results = []
    
    # Get stopwords for each method
    for method in methods:
        stopwords = method_map[method](corpus_filename)
        if method == 'pmi_based_stopwords':
            sorted_stopwords = sorted(stopwords, key=stopwords.get)[:100]
        else:
            sorted_stopwords = sorted(stopwords, key=stopwords.get, reverse=True)[:100]
        all_results.append(set(sorted_stopwords))
    
    # Take intersection of stopwords from all methods
    consensus_stopwords = set.intersection(*all_results)
    print(f"Exiting hybrid_stopwords with {len(stopwords)} stopwords.")

    return consensus_stopwords

def main():
    #print("Starting main function...")
    # Define the method map at the top of the main() function
    method_map = {
        'tf': term_frequency,
        'normalised_tf': normalised_tf,
        'idf': idf,
        'normalised_idf': normalised_idf,
        'tf_idf': tf_idf,
        'term_based_random_sampling': term_based_random_sampling,
        'pmi': pmi_based_stopwords,
        'entropy': entropy_based_stopwords,
        'variance': variance_based_stopwords,
        'cooccurrence_network': cooccurrence_network_stopwords,
        'lm': lm_based_stopwords,
        'word2vec': word2vec_based_stopwords,
        'word2vec_frequency': word2vec_frequency_based_stopwords,
        'fasttext': fasttext_based_stopwords,
        'fasttext_frequency': fasttext_frequency_based_stopwords,
    }

    parser = argparse.ArgumentParser(description='Retrieve the stopword list from a given corpus.')
    parser.add_argument('corpus_filename', type=str, help='Path to the input corpus file.')
    parser.add_argument('output_filename', type=str, help='Path to the output stopword list file.')
    
    # Now the method_map keys can be accessed without error
    parser.add_argument('--methods', nargs='+', default=['tf'], choices=method_map.keys(), help='Methods to use for stopword retrieval. Can be one or multiple for hybrid approach.')
    parser.add_argument('--frequency', action='store_true', help='If set, outputs the stopword followed by its frequency (or method-based value).')
    parser.add_argument('-n', '--number', type=int, default=100, help='Number of stopwords to extract. Default is 100.')

    args = parser.parse_args()
    print(args)  # this will display the namespace with all parsed arguments.

    # If a single method is specified, use it directly
    if len(args.methods) == 1:
        method = method_map[args.methods[0]]
        stopwords = method(args.corpus_filename)
    # If multiple methods are specified, use the hybrid approach
    else:
        #stopwords = hybrid_stopwords(args.corpus_filename, methods=args.methods)
        print("Generating stopwords...")
        stopwords = hybrid_stopwords(args.corpus_filename, methods=args.methods, method_map=method_map)
        print(f"Generated {len(stopwords)} stopwords.")


    if 'pmi_based_stopwords' in args.methods:
        sorted_stopwords = sorted(stopwords, key=stopwords.get)[:args.number]
    else:
        sorted_stopwords = sorted(stopwords, key=stopwords.get, reverse=True)[:args.number]
    
    print(f"Writing to {args.output_filename}")
    with open(args.output_filename, 'w', encoding='utf-8') as file:
        for word in sorted_stopwords:
            if args.frequency:
                file.write(f'{word}\t{stopwords[word]}\n')
            else:
                file.write(f'{word}\n')
    #print("End of main function.")

if __name__ == '__main__':
    main()
