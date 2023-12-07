import argparse
from collections import defaultdict

def build_word_tag_dictionary(corpus_filename):
    """
    Build a dictionary from the corpus where each word is mapped to its possible POS tags and their frequencies.
    """
    word_tag_dict = defaultdict(lambda: defaultdict(int))

    with open(corpus_filename, 'r', encoding='utf-8') as file:
        for line in file:
            # Replace | with a space for compound words
            line = line.replace('|', ' ')
            word_tags = line.strip().split()

            for word_tag in word_tags:
                if '/' in word_tag:
                    word, tag = word_tag.rsplit('/', 1)
                    word_tag_dict[word][tag] += 1

    return word_tag_dict

def tag_stopwords(stopwords_filename, word_tag_dict):
    """
    Tag each stopword with its possible POS tags and their frequencies using the word-tag dictionary.
    """
    tagged_stopwords = []

    with open(stopwords_filename, 'r', encoding='utf-8') as file:
        for word in file:
            word = word.strip()
            tag_freqs = word_tag_dict.get(word, {})
            # Sort the tags by frequency
            sorted_tags = sorted(tag_freqs.items(), key=lambda x: x[1], reverse=True)
            tagged_word = word + ''.join([f"/{tag}:{freq}" for tag, freq in sorted_tags])
            tagged_stopwords.append(tagged_word)

    return tagged_stopwords

def main():
    parser = argparse.ArgumentParser(description="Tag stopwords with their possible POS tags and their frequencies")
    parser.add_argument('-c', '--corpus_filename', type=str, required=True, help='Path to the corpus file.')
    parser.add_argument('-s', '--stopword_filename', type=str, required=True, help='Path to the stopwords file.')

    args = parser.parse_args()

    word_tag_dict = build_word_tag_dictionary(args.corpus_filename)
    tagged_stopwords = tag_stopwords(args.stopword_filename, word_tag_dict)

    # Print tagged stopwords
    for tagged_word in tagged_stopwords:
        print(tagged_word)

if __name__ == '__main__':
    main()

