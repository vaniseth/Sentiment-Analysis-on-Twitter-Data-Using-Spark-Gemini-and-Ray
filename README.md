# Principles-of-Big-Data-Mangement

## Phase 2: Twitter Sentiment Analysis using Spark, Gemini, and Ray

This project implements a scalable pipeline to perform sentiment analysis on Twitter data using Apache Spark for preprocessing, Google Gemini API for automated labeling, and Ray for distributed training with an LSTM-based PyTorch model.

---

```plaintext
Phase-2/sentiment/
├── lstm_model.py                     # Defines LSTM model with attention
├── process_label_tweets.py           # Spark-based preprocessing and Gemini labeling
├── train_sentiment_model.py          # Ray-based training and evaluation
├── training_history.txt              # Raw logs from model training
├── sentiment_model_lstm_output/
│   ├── plots/                        # Evaluation and training visualizations
│   └── twitter_sentiment_lstm_training/  # Ray Tune and TorchTrainer artifacts
│       ├── basic-variant-state-*.json
│       ├── experiment_state-*.json
│       ├── TorchTrainer_*/
│       ├── trainer.pkl
│       └── tuner.pkl
├── ray_dashboard/                    # Screenshots from Ray dashboard 
```

### Features
* Data Preprocessing (Spark):
Cleans raw Twitter JSON data and filters invalid or short tweets. Generates a vocabulary for training.

* Labeling with Gemini API:
Automatically labels tweets with Positive, Negative, or Neutral sentiment using LLM API prompts in batches.

* Model Architecture (PyTorch):
Implements a Bi-LSTM with attention mechanism for sentiment classification.

* Distributed Training (Ray):
Trains the model across multiple CPUs using Ray Train, supports checkpointing and metric reporting.

* Evaluation & Visualization:
Confusion matrix, classification report, and training loss/accuracy curves are generated and stored.

---

Project Created by Vani Seth, Rayhan Mahady, and Divya Reddy
