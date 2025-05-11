#!/usr/bin/env python3
import sys
import json

def extract_hashtags_urls(line):
    """Extracts hashtags and URLs from a JSON tweet record."""
    try:
        tweet = json.loads(line)  # Parse JSON
        hashtags = [ht['text'] for ht in tweet.get('entities', {}).get('hashtags', [])]
        urls = [url['expanded_url'] for url in tweet.get('entities', {}).get('urls', [])]
        
        # Emit words (hashtags and URLs)
        for hashtag in hashtags:
            print(f"{hashtag.lower()}\t1")  # Convert to lowercase for consistency
        for url in urls:
            print(f"{url.lower()}\t1") 
    except json.JSONDecodeError:
        sys.stderr.write(f"Error decoding JSON: {line}\n")

# Read JSON lines from standard input
for line in sys.stdin:
    extract_hashtags_urls(line.strip())
