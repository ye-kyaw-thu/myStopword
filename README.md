# myStopWord
Stopword extraction for Burmese.

## Project Overview

In this study, we present a comprehensive exploration of stopword extraction techniques tailored for the Burmese language. Utilizing a manually segmented corpus of 212,836 Burmese sentences, we benchmark traditional methods such as term frequency and entropy against our novel proposals, word2vec_frequency and fasttext_frequency. Our findings showcase that these innovative embeddingfrequency approaches not only align with the established behavior of Zipf’s law but also offer promising avenues for enhancing text analysis in Burmese and potentially other under-resourced languages.

## File Information

1. [run_exp4.sh](https://github.com/ye-kyaw-thu/myStopWord/blob/main/exp4_100/run_exp4.sh)  
   The shell script I used for experiment number 4, which extracts only 100 stopwords.)
2. [extract_stopword.py](https://github.com/ye-kyaw-thu/myStopWord/blob/main/exp4_100/extract_stopword.py)  
   Python code for extracting Burmese stopwords using various approaches
3. [zipfs_fit.py](https://github.com/ye-kyaw-thu/myStopWord/blob/main/exp4_100/zipfs_fit.py)  
   Python code for generating graphs
4. [stopword_talk.pdf](https://github.com/ye-kyaw-thu/myStopWord/blob/main/slide/stopword_talk.pdf)  
   Presentation slide that I used at the iSAI-NLP 2023 conference.
5. [pos_tagger.py](https://github.com/ye-kyaw-thu/myStopWord/blob/main/exp4_100/postag/pos_tagger.py)  
   Python code for creating a POS-tag dictionary using the myPOS ver.3 corpus and performing rough POS tagging on extracted Burmese stopwords.  
6. [extract_highest_uniq_pos_ver2.py](https://github.com/ye-kyaw-thu/myStopWord/blob/main/exp4_100/postag_freq/extract_highest_uniq_pos_ver2.py)  
   Python code for extracting the unique highest-ranked POS tags from the input stopword file.
7. [zipfs_graph4all.py](https://github.com/ye-kyaw-thu/myStopWord/blob/main/exp4_100/graph/zipfs_graph4all.py)   
   Python code for drawing Zipf's Law graphs.
   
## Citation

If you intend to utilize any code snippets or utilize the extracted stopwords in your research, we kindly request that you acknowledge and cite the following paper: 

Ye Kyaw Thu and Thepchai Supnithi, "Embedding Meets Frequency: Novel Approaches to Stopword Identification in Burmese", In Proceedings of the 18th International Joint Symposium on Artificial Intelligence and Natural Language Processing (iSAI-NLP 2023), Nov 27 to 29, 2023, Bangkok, Thailand, pp. xx-xx [to appear]  
