#!/usr/bin/env python3
import sys

current_word = None
current_count = 0

# Read from standard input (stdin)
for line in sys.stdin:
    word, count = line.strip().split("\t")
    count = int(count)
    
    if current_word == word:
        current_count += count
    else:
        if current_word is not None:
            # Print the word and its aggregated count
            print(f"{current_word}\t{current_count}")
        current_word = word
        current_count = count

# Print the last word if necessary
if current_word is not None:
    print(f"{current_word}\t{current_count}")
    
