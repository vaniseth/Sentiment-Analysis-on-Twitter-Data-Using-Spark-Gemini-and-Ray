# Principles-of-Big-Data-Mangement

## Twitter Sentiment Analysis Project

This project is divided into **two phases**, each focusing on different aspects of large-scale data processing and machine learning using distributed computing frameworks.

---
## Phase 1: Hashtag and URL Extraction + WordCount

### Goal

The objective of Phase 1 is to process a collection of tweets and perform basic distributed analytics using Hadoop and Spark. The key tasks include:
- **Using provided Twitter datasets** (available on the course Canvas).
- **Extracting all hashtags and URLs** from the tweet text. This is to be implemented manually (i.e., without using external libraries that perform this extraction automatically).
- **Running the WordCount example**:
  - Execute the classic WordCount program on both **Apache Hadoop** and **Apache Spark**, using the extracted hashtags and URLs as input.
  - Collect and document the **output files** and **log files** from the Hadoop and Spark executions for comparison and reporting.

### Implementation Details

- The extraction code can be written in **any programming language** of your choice.
- Ensure that extracted hashtags and URLs are saved to intermediate files suitable for WordCount input.
- The WordCount job should count frequency of hashtags and URLs separately and store the output for analysis.

### Output Artifacts (Phase 1)

- Extracted hashtags file: `hashtags.txt`
- Extracted URLs file: `urls.txt`
- Hadoop WordCount outputs: `hadoop_output/`
- Spark WordCount outputs: `spark_output/`
- Log files from Hadoop and Spark executions: `logs/hadoop.log`, `logs/spark.log`

---
## Phase 2: Sentiment Analysis with Spark, Gemini API, and Ray

### Goal

The objective of Phase 2 is to design and implement a scalable sentiment analysis pipeline using distributed systems and deep learning frameworks. The key goals include:

- **Design and implement creative analytics using Apache Spark, Ray, and Twitter data.**
- **Store and retrieve tweets in Spark** and clean the raw text data.
- **Generate sentiment labels** using the **Google Gemini API**.
- **Train a deep learning model** using **Ray Train** and **PyTorch** to classify tweets as Positive, Negative, or Neutral.
- **Run Spark SQL queries** and **apply machine learning** using Ray for insights and classification.
- **Develop visualizations** such as pie charts, bar graphs, and confusion matrices for model evaluation.
- **Document the entire pipeline** from preprocessing to model evaluation.

### Implementation Stages

1. **Data Cleaning (Spark):** Normalize and clean raw tweet text.
2. **Labeling (Gemini API):** Automatically assign sentiment labels to tweets.
3. **Vocabulary Building (Spark):** Create token-index mappings for LSTM input.
4. **Model Definition (PyTorch):** LSTM model with attention mechanism.
5. **Training (Ray):** Distributed training using Ray Train with multiple workers.
6. **Evaluation & Visualization:** Generate metrics, plots, and reports on model performance.

### Output Artifacts (Part 2)

- Cleaned and labeled dataset: `labeled_tweets.parquet`
- Vocabulary: `sentiment_vocab.pkl`
- Trained model checkpoint: `trainer.pkl`
- Evaluation metrics: `evaluation_metrics_lstm.json`
- Visuals: `confusion_matrix.png`, `loss_history.png`, `accuracy_history.png`
- Final predictions: `test_predictions_lstm.csv`
- Classification report: `classification_report_lstm.txt`

---

This project created as a class project by Vani Seth, Rayhan Mahady, and Divya Reddy
