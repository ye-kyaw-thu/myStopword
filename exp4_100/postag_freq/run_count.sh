#!/bin/bash

python count_highest_pos_sorted.py -f ./stopwords_cooccurrence_network_pos.txt > count_cooccurrence.txt

python count_highest_pos_sorted.py -f ./stopwords_entropy_pos.txt > count_entropy.txt

python count_highest_pos_sorted.py -f ./stopwords_fasttext_frequency_pos.txt > count_fasttext.txt

python count_highest_pos_sorted.py -f ./stopwords_lm_pos.txt > count_lm.txt

python count_highest_pos_sorted.py -f ./stopwords_normalised_tf_pos.txt > count_normalised_tf.txt

python count_highest_pos_sorted.py -f ./stopwords_term_based_random_sampling_pos.txt > count_term.txt

python count_highest_pos_sorted.py -f ./stopwords_tf_pos.txt > count_tf.txt

python count_highest_pos_sorted.py -f ./stopwords_word2vec_frequency_pos.txt > count_word2vec.txt
