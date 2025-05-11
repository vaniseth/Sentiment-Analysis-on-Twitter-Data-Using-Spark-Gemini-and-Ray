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
    
#### why 2nd part have some issue????
#!/usr/bin/env python3
# import sys
# import matplotlib.pyplot as plt

# word_counts = {}

# # Read input from the mapper
# for line in sys.stdin:
#     line = line.strip()
#     if not line:
#         continue

#     try:
#         word, count = line.split("\t")
#         count = int(count)
#         word_counts[word] = word_counts.get(word, 0) + count
#     except ValueError:
#         sys.stderr.write(f"Skipping malformed line: {line}\n")

# # Sort words by count (descending)
# sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)

# # Output word counts
# for word, count in sorted_words:
#     print(f"{word}\t{count}")

# # Generate graph for top 20 words
# top_words = sorted_words[:20]
# words, counts = zip(*top_words) if top_words else ([], [])

# plt.figure(figsize=(12, 6))
# plt.bar(words, counts, color='skyblue')
# plt.xlabel('Words')
# plt.ylabel('Count')
# plt.title('Top 20 Words Frequency')
# plt.xticks(rotation=45, ha='right')
# plt.tight_layout()

# # Save graph
# plt.savefig("top_20_words.png")
# plt.show()
