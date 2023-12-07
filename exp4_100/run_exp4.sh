#!/bin/bash

## Written by Ye Kyaw Thu, LU Lab., Myanmar

# Define the methods in an array
methods=("tf" "normalised_tf" "term_based_random_sampling" "entropy" "cooccurrence_network" "lm" "word2vec_frequency" "fasttext_frequency")

# Input and output filenames
input_file="./segmentation-data-updated2.cleaned"
output_prefix="stopwords_"

# Loop through each method
for method in "${methods[@]}"; do
    # With --frequency
    output_file="${output_prefix}${method}_freq.txt"
    echo "Processing for method: $method with --frequency"
    time python extract_stopword.py "$input_file" "$output_file" --method "$method" --frequency --number 100

    # Without --frequency
    output_file="${output_prefix}${method}.txt"
    echo "Processing for method: $method without --frequency"
    time python extract_stopword.py "$input_file" "$output_file" --method "$method" --number 100
done

echo "All commands executed successfully!"

