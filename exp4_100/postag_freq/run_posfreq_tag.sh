#!/bin/bash

# The corpus filename
CORPUS="mypos-ver.3.0.txt"

# List of stopwords filenames
STOPWORDS_LIST=("stopwords_cooccurrence_network.txt" "stopwords_normalised_tf.txt" 
                "stopwords_entropy.txt" "stopwords_term_based_random_sampling.txt"
                "stopwords_fasttext_frequency.txt" "stopwords_tf.txt"
                "stopwords_lm.txt" "stopwords_word2vec_frequency.txt")

# Loop through each stopwords file and run the Python program
for STOPWORDS in "${STOPWORDS_LIST[@]}"; do
    # Construct the output filename by replacing ".txt" with "_pos.txt"
    OUTPUT="${STOPWORDS%.txt}_pos.txt"
    echo "Processing ${STOPWORDS} -> ${OUTPUT}..."
    python pos_freq_tagger.py -c "$CORPUS" -s "$STOPWORDS" > "$OUTPUT"
done

echo "All done!"

