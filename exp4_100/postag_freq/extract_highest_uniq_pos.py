import argparse
from collections import defaultdict

def extract_unique_highest_pos_tags(filename):
    pos_frequency_dict = defaultdict(int)
    unique_highest_pos_tags = {}

    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('/')
            word = parts[0]
            pos_tags = [tag.split(':') for tag in parts[1:]]

            # Sort by frequency and then get the highest frequency pos tag
            sorted_pos_tags = sorted(pos_tags, key=lambda x: int(x[1]), reverse=True)
            highest_pos_tag, frequency = sorted_pos_tags[0]
            
            # Check if the pos tag is already in our dictionary with a higher frequency
            if highest_pos_tag not in pos_frequency_dict or pos_frequency_dict[highest_pos_tag] < int(frequency):
                pos_frequency_dict[highest_pos_tag] = int(frequency)
                unique_highest_pos_tags[highest_pos_tag] = f"{highest_pos_tag}:{frequency}"

    return unique_highest_pos_tags

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract the unique highest POS tags from the input stopword file")
    parser.add_argument('-f', '--filename', type=str, required=True, help='Path to the input stopword file.')
    args = parser.parse_args()

    result = extract_unique_highest_pos_tags(args.filename)
    
    for pos, freq in result.items():
        print(freq)


