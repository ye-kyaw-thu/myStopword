#!/bin/bash

# The directory containing the segmentation data
CORPUS="../segmentation-data-updated2.cleaned"

# List of stopword filenames
STOPWORD_FILES=(
    "stopwords_cooccurrence_network.txt"
    "stopwords_normalised_tf.txt"
    "stopwords_entropy.txt"
    "stopwords_term_based_random_sampling.txt"
    "stopwords_fasttext_frequency.txt"
    "stopwords_tf.txt"
    "stopwords_lm.txt"
    "stopwords_word2vec_frequency.txt"
)

# Types of plots
PLOT_TYPES=("linear" "log")

# Loop through each stopword file and run the command
for STOPWORD_FILE in "${STOPWORD_FILES[@]}"; do
    for PLOT_TYPE in "${PLOT_TYPES[@]}"; do
        # Construct the output filename based on the stopword file's name and the plot type
        OUTPUT_FILENAME="${STOPWORD_FILE%.*}_$PLOT_TYPE.png"
        echo "Processing $STOPWORD_FILE ($PLOT_TYPE plot) -> $OUTPUT_FILENAME"

        # Execute the Python script
        python zipfs_graph.py -c "$CORPUS" -s "$STOPWORD_FILE" -o "$OUTPUT_FILENAME" -p "$PLOT_TYPE"
    done
done

echo "All done!"

