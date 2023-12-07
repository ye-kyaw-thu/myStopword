## Written by Ye Kyaw Thu, LU Lab., Myanmar

import argparse
from collections import defaultdict

def count_highest_pos_tags(filename):
    # Dictionary to store count of each POS tag
    pos_counts = defaultdict(int)

    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('/')
            
            if len(parts) < 2:  # skip lines with no pos tags
                continue

            # The left-most POS tag is always the first after the word, so we can just split by ':' and get the tag
            highest_pos_tag = parts[1].split(':')[0]

            # Increment the count for the POS tag
            pos_counts[highest_pos_tag] += 1

    # Sort the dictionary by value in descending order
    sorted_pos_counts = sorted(pos_counts.items(), key=lambda x: x[1], reverse=True)
    return sorted_pos_counts

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Count the occurrences of the highest (left-most) POS tags from the input stopword file")
    parser.add_argument('-f', '--filename', type=str, required=True, help='Path to the input stopword file.')
    args = parser.parse_args()

    result = count_highest_pos_tags(args.filename)
    
    for pos, count in result:
        print(f"{pos} = {count}")


