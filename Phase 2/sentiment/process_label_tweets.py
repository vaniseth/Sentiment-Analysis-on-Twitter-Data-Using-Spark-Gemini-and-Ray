import os
import time
import re
import json
import pickle # To save vocabulary
from collections import Counter
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, StructType, StructField
from dotenv import load_dotenv
import google.generativeai as genai
from google.generativeai import types as genai_types
from google.api_core import exceptions as google_exceptions

# --- Configuration ---
load_dotenv()

# === INPUT: Adjust path to your raw data ===
# Make sure this points to the file containing the JSON with "full_text"
TWITTER_DATA_PATH = "data/out.json" # INPUT: Adjust path to your raw data
LABELED_OUTPUT_PARQUET_PATH = "data/labeled_tweets.parquet" # OUTPUT: Labeled data
VOCAB_OUTPUT_PATH = "data/sentiment_vocab.pkl" # OUTPUT: Vocabulary file

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL_NAME = "gemini-2.0-flash"

# === Data Limit ===
DATA_LIMIT = 1500 # Limit the number of records to process for labeling/vocab

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]
GENERATION_CONFIG = genai.types.GenerationConfig(
    response_mime_type="application/json",
    temperature=0.2
)

BATCH_SIZE = 20
API_RETRY_DELAY = 2
MAX_RETRIES = 3

# Vocabulary Configuration
VOCAB_SIZE = 10000 # Max number of words in the vocabulary
MIN_WORD_FREQ = 2 # Lower min frequency slightly for smaller dataset

# --- Helper Functions ---

def clean_tweet_text(text):
    """Basic cleaning of tweet text."""
    if text is None:
        return None
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+', '', text)
    # Keep hashtags as part of text for sentiment, remove '#' symbol
    text = re.sub(r'#', '', text)
    # Remove most punctuation but keep basic contractions/apostrophes and maybe some common symbols?
    text = re.sub(r'[^\w\s\'!?.,]', '', text) # Keep basic punctuation for now
    # Remove numbers (optional)
    # text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'^rt[\s]+', '', text)
    # Limit length slightly for prompt construction
    return text[:1000] if len(text) > 0 else None # Allow empty strings for now, filter later


# (get_sentiment_batch_from_gemini and process_single_batch remain the same)
def get_sentiment_batch_from_gemini(batch_iterator):
    """Processes partitions of tweets in batches using the Gemini API."""
    if not GOOGLE_API_KEY:
        print("GOOGLE_API_KEY not configured. Cannot label data.")
        for record_id, text in batch_iterator:
             yield (record_id, text, "Error: API Key Missing")
        return

    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel(
            GEMINI_MODEL_NAME,
            safety_settings=SAFETY_SETTINGS,
            generation_config=GENERATION_CONFIG
        )
    except Exception as e:
        print(f"Error configuring Gemini client in partition: {e}")
        for record_id, text in batch_iterator:
             yield (record_id, text, "Error: Client Setup Failed")
        return

    prompt_template = """Analyze the sentiment for each tweet listed below. Classify each as Positive, Negative, or Neutral.
Respond ONLY with a valid JSON object containing a single key "sentiments", where the value is a list of strings.
Each string in the list should be the sentiment (Positive, Negative, or Neutral) corresponding to the tweet at the same position in the input list.
If a sentiment cannot be determined, use "Neutral". The list must contain exactly {num_tweets} items.

Tweets:
{tweet_list_json}

JSON Response:
"""

    batch = []
    for record_id, text in batch_iterator:
        if text: # Only add valid text
            batch.append({"id": record_id, "text": text})
        else: # Yield Neutral for invalid/short text directly
             yield (record_id, text, "Neutral")

        if len(batch) >= BATCH_SIZE:
            yield from process_single_batch(model, batch, prompt_template)
            batch = []
    if batch:
        yield from process_single_batch(model, batch, prompt_template)


def process_single_batch(model, batch, prompt_template):
    """Sends a single batch to Gemini API and parses the response."""
    tweet_texts_for_prompt = [item['text'] for item in batch]
    try:
        tweet_list_json_str = json.dumps(tweet_texts_for_prompt, ensure_ascii=False)
    except Exception as e:
        print(f"Error creating JSON for prompt: {e}")
        for item in batch:
            yield (item['id'], item['text'], "Error: Prompt Creation Failed")
        return

    prompt = prompt_template.format(num_tweets=len(batch), tweet_list_json=tweet_list_json_str)
    sentiments = ["Error: API Call Failed"] * len(batch)

    for attempt in range(MAX_RETRIES):
        response = None # Initialize response
        try:
            response = model.generate_content(prompt)
            cleaned_response_text = response.text.strip()
            if cleaned_response_text.startswith("```json"):
                cleaned_response_text = cleaned_response_text[7:]
            if cleaned_response_text.endswith("```"):
                cleaned_response_text = cleaned_response_text[:-3]

            result_json = json.loads(cleaned_response_text)

            if isinstance(result_json, dict) and "sentiments" in result_json:
                raw_sentiments = result_json["sentiments"]
                if isinstance(raw_sentiments, list) and len(raw_sentiments) == len(batch):
                    valid_sentiments = ["Positive", "Negative", "Neutral"]
                    sentiments = [s.strip().capitalize() if isinstance(s, str) and s.strip().capitalize() in valid_sentiments else "Neutral" for s in raw_sentiments]
                    break
                else:
                    print(f"Warning: Parsed JSON 'sentiments' list length mismatch ({len(raw_sentiments)} vs {len(batch)}) or invalid type. Attempt {attempt+1}/{MAX_RETRIES}")
                    sentiments = ["Error: Response Length Mismatch"] * len(batch)
            else:
                print(f"Warning: Parsed JSON has incorrect structure. Expected {{'sentiments': [...]}}. Attempt {attempt+1}/{MAX_RETRIES}")
                sentiments = ["Error: Response Format Invalid"] * len(batch)

        except (json.JSONDecodeError, Exception) as parse_e:
             response_text_snippet = response.text[:200] if response else "N/A"
             print(f"Warning: Failed to parse response: {parse_e}. Response text: '{response_text_snippet}...'. Attempt {attempt+1}/{MAX_RETRIES}")
             sentiments = ["Error: Response Parsing Failed"] * len(batch)

        except (google_exceptions.ResourceExhausted, google_exceptions.InternalServerError, google_exceptions.DeadlineExceeded) as e:
            print(f"Warning: Rate limit or server error processing batch (Attempt {attempt+1}/{MAX_RETRIES}): {e}. Retrying after {API_RETRY_DELAY * (attempt + 1)}s...")
            sentiments = [f"Error: {type(e).__name__}"] * len(batch)
            time.sleep(API_RETRY_DELAY * (attempt + 1))
        except Exception as e:
            block_reason = ""
            if response and hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                 block_reason = f" BlockReason: {response.prompt_feedback}"
            # Check if the error is specifically a content block
            if "response was blocked" in str(e).lower() or (response and response.prompt_feedback):
                 print(f"Warning: Content blocked for batch.{block_reason}. Assigning Neutral.")
                 sentiments = ["Neutral"] * len(batch) # Assign Neutral if blocked
                 break # Stop retrying if blocked
            else:
                 print(f"Error calling Gemini API for batch: {e}.{block_reason}")
                 sentiments = [f"Error: {type(e).__name__}"] * len(batch)
                 break # Don't retry on other errors

    for i, item in enumerate(batch):
        yield (item['id'], item['text'], sentiments[i])
    time.sleep(0.1)


# --- Spark Processing ---

def main():
    """
    Main function for processing and labeling tweets using Apache Spark.

    This function performs the following steps:
    1. Initializes a Spark session and checks the default parallelism.
    2. Loads raw tweet data from a specified JSON file path.
        - Validates the existence of the input path.
        - Uses a predefined schema to load the data.
        - Prints the schema and a sample of the loaded data.
    3. Cleans the tweet text using a UDF and filters out invalid or short text.
        - Ensures the cleaned text is not null and has a minimum length.
        - Optionally limits the dataset size for further processing.
    4. Builds a vocabulary from the cleaned text.
        - Extracts words, counts their occurrences, and filters by minimum frequency.
        - Saves the vocabulary to a specified output path.
    5. Labels the tweets using a Gemini sentiment labeling function.
        - Applies batch processing for sentiment labeling.
        - Displays the distribution of sentiment labels and errors.
        - Filters out rows with labeling errors.
    6. Saves the labeled data to a Parquet file.
        - Ensures the output directory exists before saving.
        - Only saves successfully labeled tweets.

    The function includes error handling for data loading, vocabulary saving, and labeling steps.
    It also caches intermediate results for performance optimization and releases resources upon completion.
    """
    spark = SparkSession.builder \
        .appName("TwitterSentimentLabelingAndVocab") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .getOrCreate()

    sc = spark.sparkContext
    print(f"Spark Initialized. Default Parallelism: {sc.defaultParallelism}")

    # 1. Load Raw Data
    print(f"Attempting to load data from: {TWITTER_DATA_PATH}")
    if not os.path.exists(TWITTER_DATA_PATH):
         print(f"Error: Input path does not exist: {TWITTER_DATA_PATH}")
         spark.stop()
         return

    # === CORRECTED SCHEMA: Use full_text instead of text ===
    raw_schema = StructType([
        StructField("id_str", StringType(), True),
        StructField("full_text", StringType(), True), # Read from "full_text"
    ])
    try:
        raw_tweets_df = spark.read.schema(raw_schema).json(TWITTER_DATA_PATH)
        print("Schema used for loading:")
        raw_tweets_df.printSchema()
        print("Sample raw data:")
        raw_tweets_df.show(5, truncate=80) # Show fewer rows initially
        raw_count = raw_tweets_df.count()
        print(f"Total raw records loaded: {raw_count}")
        if raw_count == 0:
             print("Error: No records loaded from the source.")
             spark.stop()
             return
    except Exception as e:
         print(f"Error loading data from {TWITTER_DATA_PATH}: {e}")
         spark.stop()
         return

    # 2. Clean Text
    clean_text_udf = F.udf(clean_tweet_text, StringType())
    partially_cleaned_df = raw_tweets_df.withColumn("cleaned_text", clean_text_udf(F.col("full_text"))) \
                                        .select("id_str", "cleaned_text") # Select only needed columns earlier

    # Filter out rows where cleaning resulted in None OR text is too short
    # Combine filtering steps
    cleaned_df = partially_cleaned_df.filter(F.col("cleaned_text").isNotNull()) \
                                     .filter(F.length(F.col("cleaned_text")) > 20) \


    print(f"Count after cleaning and length filter (> 20 chars): {cleaned_df.count()}")

    # === ADDED: Limit the DataFrame ===
    if cleaned_df.count() > DATA_LIMIT:
        print(f"Limiting data to approximately {DATA_LIMIT} records.")
        # Use sample for randomness or limit for speed
        # limited_df = cleaned_df.sample(fraction=DATA_LIMIT/cleaned_df.count(), seed=42) # Use sample for random subset
        limited_df = cleaned_df.limit(DATA_LIMIT) # Use limit for faster subsetting
        print(f"Actual count after limit: {limited_df.count()}")
    else:
        print("Dataset size is within the limit, processing all valid records.")
        limited_df = cleaned_df # Use the full cleaned dataset if it's small enough

    # Cache the limited data as it's used for vocab and labeling
    limited_df.cache()
    final_cleaned_count = limited_df.count()

    if final_cleaned_count == 0:
        print("Error: No tweets remaining after cleaning and limiting. Exiting.")
        spark.stop()
        return

    print(f"Processing {final_cleaned_count} records for vocabulary and labeling.")
    print("Sample of data to be processed:")
    limited_df.show(10, truncate=50)


    # 3. Build Vocabulary (from the limited, cleaned text)
    print("Building vocabulary...")
    words_rdd = limited_df.select("cleaned_text") \
                          .rdd.flatMap(lambda row: row.cleaned_text.split())
    word_counts = words_rdd.map(lambda word: (word, 1)) \
                           .reduceByKey(lambda a, b: a + b) \
                           .filter(lambda item: item[1] >= MIN_WORD_FREQ)
    top_words = word_counts.sortBy(lambda item: item[1], ascending=False) \
                           .map(lambda item: item[0]) \
                           .take(VOCAB_SIZE)
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for i, word in enumerate(top_words):
        vocab[word] = i + 2
    print(f"Vocabulary built. Size: {len(vocab)} words.")
    print(f"Saving vocabulary to: {VOCAB_OUTPUT_PATH}")
    try:
        # Ensure directory exists before saving
        os.makedirs(os.path.dirname(VOCAB_OUTPUT_PATH), exist_ok=True)
        with open(VOCAB_OUTPUT_PATH, 'wb') as f:
            pickle.dump(vocab, f)
    except Exception as e:
        print(f"Error saving vocabulary: {e}")

    # 4. Apply Gemini Labeling using mapPartitions with Batching (on limited data)
    print("Starting Gemini labeling...")
    labeled_schema = StructType([
        StructField("id_str", StringType(), True),
        StructField("cleaned_text", StringType(), True),
        StructField("sentiment_label", StringType(), True)
    ])

    # Use the cached limited RDD
    limited_rdd = limited_df.rdd
    labeled_rdd = limited_rdd.mapPartitions(get_sentiment_batch_from_gemini)
    labeled_df = spark.createDataFrame(labeled_rdd, schema=labeled_schema)

    print("Sample data after Gemini batch labeling:")
    labeled_df.show(10, truncate=50)

    # Show counts per sentiment label
    print("Sentiment label distribution (excluding errors):")
    labeled_df.filter(~F.col("sentiment_label").startswith("Error:")) \
              .groupBy("sentiment_label") \
              .count() \
              .show()
    print("Distribution of errors during labeling:")
    labeled_df.filter(F.col("sentiment_label").startswith("Error:")) \
              .groupBy("sentiment_label") \
              .count() \
              .show(truncate=False)

    # Filter out rows where labeling failed before saving
    final_labeled_df_to_save = labeled_df.filter(~F.col("sentiment_label").startswith("Error:"))
    final_labeled_count = final_labeled_df_to_save.count()
    print(f"Final count of successfully labeled tweets to save: {final_labeled_count}")

    # 5. Save Labeled Data to Parquet
    if final_labeled_count > 0:
        print(f"Saving {final_labeled_count} labeled records to: {LABELED_OUTPUT_PARQUET_PATH}")
        # Ensure directory exists before saving
        os.makedirs(os.path.dirname(LABELED_OUTPUT_PARQUET_PATH), exist_ok=True)
        final_labeled_df_to_save.write.mode("overwrite").parquet(LABELED_OUTPUT_PARQUET_PATH)
    else:
        print("No successfully labeled data to save.")

    limited_df.unpersist() # Release cache
    print("Processing complete.")
    spark.stop()

if __name__ == "__main__":
    main()
