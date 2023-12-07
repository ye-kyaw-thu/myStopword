#!/bin/bash

python zipfs_graph4all.py -c ../segmentation-data-updated2.cleaned -s 'stopwords_*.txt' -o comparison_log.png -p log
python zipfs_graph4all.py -c ../segmentation-data-updated2.cleaned -s 'stopwords_*.txt' -o comparison_linear.png -p linear

