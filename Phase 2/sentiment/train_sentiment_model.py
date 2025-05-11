import os
import pickle
import time
import json # Added for saving metrics
import ray
from ray import train, data
# Use TorchTrainer for PyTorch models
from ray.train.torch import TorchTrainer, TorchCheckpoint
from ray.air.config import ScalingConfig, RunConfig

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader # DataLoader might not be needed now
from torch.nn.utils.rnn import pad_sequence # For padding

import evaluate as hf_evaluate # Can still use evaluate library for metrics
import numpy as np
# Import specific metrics functions from sklearn
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import Counter # For potential use if vocab wasn't pre-built

# === Import the model class from the separate file ===
# Ensure lstm_model.py is in the same directory or Python path
try:
    from lstm_model import LSTMSentimentClassifier
except ImportError:
    print("Error: Could not import LSTMSentimentClassifier from lstm_model.py.")
    print("Ensure lstm_model.py is in the same directory as this script.")
    exit()


# --- Configuration ---
# Use absolute paths for better compatibility, especially with Ray AIR
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # Get directory of the script
DATA_DIR = os.path.join(BASE_DIR, "..", "data") # Assumes data dir is one level up

# Make sure these paths exist and point to the output of the Spark job
LABELED_DATA_PATH = os.path.join(DATA_DIR, "labeled_tweets.parquet") # INPUT: Labeled data (pointing to dir)
VOCAB_PATH = os.path.join(DATA_DIR, "sentiment_vocab.pkl") # INPUT: Vocabulary file
OUTPUT_DIR = os.path.join(BASE_DIR, "sentiment_model_lstm_output") # Directory for checkpoints, logs, plots
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots") # Subdir for plots/reports

# Model Hyperparameters
VOCAB_SIZE = 10000 + 2 # From Spark script + <PAD> + <UNK> (Will be updated from loaded vocab)
EMBEDDING_DIM = 128
HIDDEN_DIM = 256
NUM_LSTM_LAYERS = 2
DROPOUT_PROB = 0.5
NUM_CLASSES = 3 # Positive, Negative, Neutral

# Training parameters
NUM_EPOCHS = 50 # LSTMs might need more epochs
LEARNING_RATE = 0.001 # Common starting point for LSTMs with Adam
BATCH_SIZE_PER_WORKER = 64 # This is now the manual batch size
NUM_WORKERS = 2 # Adjust based on your Ray cluster resources
USE_GPU = torch.cuda.is_available() # Auto-detect GPU

# Label mapping (Consistent with Spark script is crucial)
label2id = {"Negative": 0, "Neutral": 1, "Positive": 2}
id2label = {v: k for k, v in label2id.items()}

# --- Data Preprocessing and Collation ---

# Load vocabulary globally once in the main script
try:
    print(f"Attempting to load vocabulary from: {VOCAB_PATH}")
    # Ensure the path exists before trying to open
    if not os.path.exists(VOCAB_PATH):
        raise FileNotFoundError(f"Vocabulary file not found at {VOCAB_PATH}")
    with open(VOCAB_PATH, 'rb') as f:
        vocab = pickle.load(f)
    print(f"Vocabulary loaded successfully from {VOCAB_PATH}. Size: {len(vocab)}")
    PAD_IDX = vocab['<PAD>']
    UNK_IDX = vocab['<UNK>']
    # Update VOCAB_SIZE based on loaded vocab if different from config
    ACTUAL_VOCAB_SIZE = len(vocab)
    if ACTUAL_VOCAB_SIZE != VOCAB_SIZE:
        print(f"Warning: Loaded vocab size ({ACTUAL_VOCAB_SIZE}) differs from config ({VOCAB_SIZE}). Using loaded size.")
        VOCAB_SIZE = ACTUAL_VOCAB_SIZE
except FileNotFoundError:
    print(f"Error: Vocabulary file not found at {VOCAB_PATH}. Run Spark script first.")
    exit()
except Exception as e:
    print(f"Error loading vocabulary: {e}")
    exit()

def simple_tokenizer(text):
    """Simple space-based tokenizer."""
    if not isinstance(text, str):
        return []
    return text.lower().split()

def collate_batch(batch: list[dict], device: torch.device) -> dict:
    """
    Manually collates a list of row dictionaries into padded tensors.
    Also returns original text for evaluation saving.
    Args:
        batch: A list of dictionaries, where each dict has 'cleaned_text' and 'sentiment_label'.
        device: The torch device to move tensors to.
    Returns:
        A dictionary containing 'text_ids', 'lengths', 'labels' tensors for the batch,
        and 'original_texts' list.
    """
    if not batch: # Handle empty batch case
        return {"text_ids": torch.empty(0, 0, dtype=torch.long, device=device),
                "lengths": torch.empty(0, dtype=torch.long), # Lengths stay on CPU
                "labels": torch.empty(0, dtype=torch.long, device=device),
                "original_texts": []}

    processed_texts = []
    lengths = []
    labels = []
    original_texts = [] # Store original text for valid items

    for item in batch:
        text = item.get("cleaned_text") # Use .get for safety
        label_str = item.get("sentiment_label")
        label_id = label2id.get(label_str, -1)

        # Skip items with invalid labels or missing text
        if label_id == -1 or text is None:
            continue

        if not isinstance(text, str):
            tokens = []
        else:
            tokens = simple_tokenizer(text)

        numericalized = [vocab.get(token, UNK_IDX) for token in tokens]
        processed_texts.append(torch.tensor(numericalized, dtype=torch.long)) # Create tensor here
        lengths.append(len(numericalized))
        labels.append(label_id)
        original_texts.append(text) # Keep original text

    # If all items in the batch had invalid labels or text
    if not processed_texts:
        return {"text_ids": torch.empty(0, 0, dtype=torch.long, device=device),
                "lengths": torch.empty(0, dtype=torch.long),
                "labels": torch.empty(0, dtype=torch.long, device=device),
                "original_texts": []}

    # Pad sequences within the batch
    padded_texts_tensor = pad_sequence(processed_texts, batch_first=True, padding_value=PAD_IDX)

    # Return dictionary of tensors, moved to the correct device
    return {
        "text_ids": padded_texts_tensor.to(device),
        "lengths": torch.tensor(lengths, dtype=torch.long), # Keep lengths on CPU
        "labels": torch.tensor(labels, dtype=torch.long).to(device),
        "original_texts": original_texts # Return list of original texts
    }


# --- Ray Train Loop for PyTorch ---

def train_loop_per_worker(config):
    """The training function Ray Train executes on each worker."""
    # Get worker configuration
    lr = config.get("lr", 0.001)
    epochs = config.get("epochs", 3)
    manual_batch_size = config.get("batch_size", 64) # Renamed for clarity
    vocab_size = config.get("vocab_size")
    embedding_dim = config.get("embedding_dim")
    hidden_dim = config.get("hidden_dim")
    n_layers = config.get("n_layers")
    dropout = config.get("dropout")
    pad_idx = config.get("pad_idx")
    num_classes = config.get("num_classes")

    # Get the Ray Dataset shard for this worker
    # These shards contain the *original* data (dict rows)
    train_data_shard = train.get_dataset_shard("train")
    eval_data_shard = train.get_dataset_shard("evaluation")

    # Instantiate the model - Uses the imported class
    model = LSTMSentimentClassifier(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        output_dim=num_classes,
        n_layers=n_layers,
        bidirectional=True,
        dropout=dropout,
        pad_idx=pad_idx
    )
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Prepare model for distributed training *after* moving to device if needed
    # model.to(device) # Let prepare_model handle placement
    model = train.torch.prepare_model(model)

    # Define Loss and Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Training Loop - Manual Batching using iter_rows
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_steps = 0
        epoch_start_time = time.time()

        # Manually iterate and batch
        current_batch = []
        # Use iter_rows() to get individual dictionaries
        for row in train_data_shard.iter_rows():
            current_batch.append(row)
            if len(current_batch) >= manual_batch_size:
                # Process the batch using our collate function
                batch_data = collate_batch(current_batch, device)
                current_batch = [] # Reset batch

                # Skip if collation resulted in empty batch (e.g., all invalid labels)
                if batch_data["labels"].numel() == 0: continue

                ids = batch_data["text_ids"]
                lengths = batch_data["lengths"]
                labels = batch_data["labels"]

                optimizer.zero_grad()
                predictions = model(ids, lengths)
                loss = criterion(predictions, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                train_steps += 1

        # Process the last partial batch if any
        if current_batch:
            batch_data = collate_batch(current_batch, device)
            if batch_data["labels"].numel() > 0:
                ids = batch_data["text_ids"]
                lengths = batch_data["lengths"]
                labels = batch_data["labels"]

                optimizer.zero_grad()
                predictions = model(ids, lengths)
                loss = criterion(predictions, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                train_steps += 1

        avg_train_loss = train_loss / train_steps if train_steps > 0 else 0.0

        # Evaluation Loop - Manual Batching using iter_rows
        eval_loss = 0.0
        eval_steps = 0
        all_preds = []
        all_labels = []
        model.eval()
        if eval_data_shard:
            with torch.no_grad():
                current_eval_batch = []
                for row in eval_data_shard.iter_rows():
                    current_eval_batch.append(row)
                    if len(current_eval_batch) >= manual_batch_size:
                        batch_data = collate_batch(current_eval_batch, device)
                        current_eval_batch = []

                        if batch_data["labels"].numel() == 0: continue

                        ids = batch_data["text_ids"]
                        lengths = batch_data["lengths"]
                        labels = batch_data["labels"] # Labels are already on device from collate

                        predictions = model(ids, lengths)
                        loss = criterion(predictions, labels) # Calculate eval loss
                        eval_loss += loss.item()
                        eval_steps += 1

                        preds = torch.argmax(predictions, dim=1)
                        all_preds.extend(preds.cpu().numpy())
                        all_labels.extend(labels.cpu().numpy()) # Collect labels

                # Process the last partial evaluation batch
                if current_eval_batch:
                     batch_data = collate_batch(current_eval_batch, device)
                     if batch_data["labels"].numel() > 0:
                        ids = batch_data["text_ids"]
                        lengths = batch_data["lengths"]
                        labels = batch_data["labels"]

                        predictions = model(ids, lengths)
                        loss = criterion(predictions, labels)
                        eval_loss += loss.item()
                        eval_steps += 1

                        preds = torch.argmax(predictions, dim=1)
                        all_preds.extend(preds.cpu().numpy())
                        all_labels.extend(labels.cpu().numpy())


        avg_eval_loss = eval_loss / eval_steps if eval_steps > 0 else float('inf')
        eval_accuracy = hf_evaluate.load("accuracy").compute(predictions=all_preds, references=all_labels)["accuracy"] if all_labels else 0.0
        eval_f1 = hf_evaluate.load("f1").compute(predictions=all_preds, references=all_labels, average="weighted")["f1"] if all_labels else 0.0

        epoch_duration = time.time() - epoch_start_time
        # Report metrics and checkpoint back to Ray Train
        metrics = {
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "evaluation_loss": avg_eval_loss,
            "evaluation_accuracy": eval_accuracy,
            "evaluation_f1": eval_f1,
            "epoch_duration_s": epoch_duration,
        }
        # === CORRECTED: Save unwrapped model's state_dict ===
        # model is potentially wrapped by DDP, access state via .module
        unwrapped_model = model.module if hasattr(model, "module") else model
        checkpoint = TorchCheckpoint.from_state_dict(unwrapped_model.state_dict())
        train.report(metrics=metrics, checkpoint=checkpoint)

        print(f"Epoch {epoch} finished in {epoch_duration:.2f}s. Train Loss: {avg_train_loss:.4f}, Eval Loss: {avg_eval_loss:.4f}, Eval Acc: {eval_accuracy:.4f}")


# --- Plotting and Evaluation Helper Functions ---
# (plot_training_history and plot_confusion_matrix remain the same as before)
def plot_training_history(history, save_dir):
    """Plots training and validation loss and accuracy."""
    os.makedirs(save_dir, exist_ok=True)
    df_data = []
    if isinstance(history, pd.DataFrame):
         history_list = history.to_dict(orient='records')
    elif isinstance(history, list):
         history_list = history
    else:
         print("Warning: Unexpected history format. Cannot plot.")
         return

    for log in history_list:
        epoch = log.get('epoch')
        step = log.get('step')
        train_loss = log.get('train_loss')
        eval_loss = log.get('evaluation_loss') # Key reported by train_loop_per_worker
        eval_acc = log.get('evaluation_accuracy') # Key reported by train_loop_per_worker

        if train_loss is not None:
            df_data.append({'epoch': epoch, 'step': step, 'value': train_loss, 'metric': 'train_loss'})
        if eval_loss is not None:
            df_data.append({'epoch': epoch, 'step': step, 'value': eval_loss, 'metric': 'eval_loss'})
        if eval_acc is not None:
            df_data.append({'epoch': epoch, 'step': step, 'value': eval_acc, 'metric': 'eval_accuracy'})

    if not df_data:
        print("No plottable history data found.")
        return

    history_df = pd.DataFrame(df_data)
    plot_col = 'epoch' if 'epoch' in history_df.columns and history_df['epoch'].notna().any() else 'step'
    if plot_col not in history_df.columns:
         print(f"Cannot plot history, missing '{plot_col}' column.")
         return
    # Ensure plot_col has numeric data for plotting
    history_df[plot_col] = pd.to_numeric(history_df[plot_col], errors='coerce')
    history_df = history_df.dropna(subset=[plot_col])
    if history_df.empty:
         print(f"No numeric data found in '{plot_col}' column to plot.")
         return


    plt.style.use('seaborn-v0_8-whitegrid')

    loss_df = history_df[history_df['metric'].isin(['train_loss', 'eval_loss'])]
    if not loss_df.empty:
        plt.figure(figsize=(10, 5))
        sns.lineplot(data=loss_df, x=plot_col, y='value', hue='metric', marker='o')
        plt.title('Training and Validation Loss')
        plt.xlabel(plot_col.capitalize())
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(save_dir, "loss_history.png"))
        plt.close()
        print(f"Loss history plot saved to {os.path.join(save_dir, 'loss_history.png')}")
    else:
        print("No loss data found to plot.")

    acc_df = history_df[history_df['metric'] == 'eval_accuracy']
    if not acc_df.empty:
        plt.figure(figsize=(10, 5))
        sns.lineplot(data=acc_df, x=plot_col, y='value', marker='o')
        plt.title('Validation Accuracy')
        plt.xlabel(plot_col.capitalize())
        plt.ylabel('Accuracy')
        plt.savefig(os.path.join(save_dir, "accuracy_history.png"))
        print(f"Accuracy history plot saved to {os.path.join(save_dir, 'accuracy_history.png')}")
    else:
        print("No evaluation accuracy data found to plot.")
    plt.close()


def plot_confusion_matrix(y_true, y_pred, labels, save_dir):
    """Plots the confusion matrix."""
    os.makedirs(save_dir, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"))
    plt.close()
    print(f"Confusion matrix plot saved to {os.path.join(save_dir, 'confusion_matrix.png')}")


# --- Main Ray Training Logic ---

def main():
    # Ensure the input paths exist
    if not os.path.exists(LABELED_DATA_PATH) or not os.path.isdir(LABELED_DATA_PATH):
         print(f"Error: Labeled data directory not found at {LABELED_DATA_PATH}. Run process_label_tweets.py first.")
         return
    if not os.path.exists(VOCAB_PATH):
         print(f"Error: Vocabulary file not found at {VOCAB_PATH}. Run process_label_tweets.py first.")
         return

    # Initialize Ray
    if ray.is_initialized():
        print("Shutting down existing Ray session...")
        ray.shutdown()
    ray.init(ignore_reinit_error=True)
    print("Ray Initialized.")
    print(f"Ray Cluster Resources: {ray.cluster_resources()}")

    # 1. Load Labeled Data using Ray Data
    print(f"Loading labeled data from: {LABELED_DATA_PATH}")
    try:
        dataset = ray.data.read_parquet(LABELED_DATA_PATH)
        print(f"Initial records loaded: {dataset.count()}")
        print(f"Dataset schema: {dataset.schema()}")
        required_cols = ["cleaned_text", "sentiment_label"]
        actual_cols = dataset.columns()
        if not all(col in actual_cols for col in required_cols):
             missing_cols = [col for col in required_cols if col not in actual_cols]
             raise ValueError(f"Dataset missing required columns: {missing_cols}. Found: {actual_cols}")

        # Filter data (similar to previous version)
        original_count = dataset.count()
        dataset = dataset.filter(lambda row: row["cleaned_text"] is not None and isinstance(row["cleaned_text"], str) and len(row["cleaned_text"]) > 0)
        dataset = dataset.filter(lambda row: row["sentiment_label"] in label2id)
        final_count = dataset.count()
        print(f"Records after filtering: {final_count}")
        if final_count == 0:
            print("Error: No valid data remaining after filtering.")
            ray.shutdown()
            return
        if final_count < original_count * 0.5:
             print(f"Warning: Significant data loss during filtering ({original_count} -> {final_count}).")

    except Exception as e:
        print(f"Error loading or validating dataset: {e}")
        ray.shutdown()
        return

    # 2. Split Data
    print("Splitting data...")
    train_dataset, temp_dataset = dataset.train_test_split(test_size=0.2, shuffle=True, seed=42)
    validation_dataset, test_dataset = temp_dataset.train_test_split(test_size=0.5, shuffle=True, seed=42)
    print(f"Train size: {train_dataset.count()}, Val size: {validation_dataset.count()}, Test size: {test_dataset.count()}")

    # 3. NO Preprocessing with map_batches needed here
    # Preprocessing (tokenization, numericalization, padding) is now done inside train_loop_per_worker via collate_batch
    print("Skipping map_batches preprocessing. Collation done in training loop.")
    train_dataset_processed = train_dataset # Pass raw datasets
    validation_dataset_processed = validation_dataset
    test_dataset_processed = test_dataset

    # --- Verify *Original* Dataset Schema (before collation) ---
    print("Sample original training batch (before collation):")
    try:
        sample = train_dataset_processed.take(limit=1)
        if sample:
            print(sample[0])
        else:
             print("Warning: Could not retrieve sample from original data.")
    except Exception as e:
        print(f"Error verifying original data: {e}")


    # 4. Configure Ray AIR Trainer (TorchTrainer)
    scaling_config = ScalingConfig(
        num_workers=NUM_WORKERS,
        use_gpu=USE_GPU,
    )

    absolute_output_dir = os.path.abspath(OUTPUT_DIR)
    # --- Ensure output directories exist before training ---
    os.makedirs(absolute_output_dir, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    print(f"Using absolute path for Ray AIR storage: {absolute_output_dir}")

    run_config = RunConfig(
        name="twitter_sentiment_lstm_training",
        storage_path=absolute_output_dir, # Use absolute path
        checkpoint_config=train.CheckpointConfig(
             num_to_keep=2,
             checkpoint_score_attribute="evaluation_loss", # Metric reported from train loop
             checkpoint_score_order="min"
        )
    )

    # Define the configuration dictionary to pass to train_loop_per_worker
    train_loop_config = {
        "lr": LEARNING_RATE,
        "epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE_PER_WORKER, # Manual batch size used in loop
        "vocab_size": ACTUAL_VOCAB_SIZE,
        "embedding_dim": EMBEDDING_DIM,
        "hidden_dim": HIDDEN_DIM,
        "n_layers": NUM_LSTM_LAYERS,
        "dropout": DROPOUT_PROB,
        "pad_idx": PAD_IDX,
        "num_classes": NUM_CLASSES,
    }

    # Instantiate TorchTrainer
    trainer = TorchTrainer(
        train_loop_per_worker=train_loop_per_worker,
        train_loop_config=train_loop_config,
        scaling_config=scaling_config,
        run_config=run_config,
        datasets={ # Pass original Ray datasets
            "train": train_dataset_processed,
            "evaluation": validation_dataset_processed
        },
    )

    # 5. Run Training
    print("Starting distributed LSTM training...")
    result = None # Initialize result
    try:
        result = trainer.fit()
        print("Training finished.")
        print(f"Training Result Metrics: {result.metrics}")
    except Exception as train_exc:
        print(f"FATAL: Training failed with exception: {train_exc}")
        # Attempt to plot history even if training failed partially
        if hasattr(trainer, 'latest_results_df'): # Check if results df available
             try:
                  history_df = trainer.latest_results_df # Use latest_results_df
                  plot_training_history(history_df, PLOTS_DIR)
             except Exception as plot_err:
                  print(f"Could not plot partial history after error: {plot_err}")
        ray.shutdown()
        return # Exit if training fails

    # --- Evaluation on Test Set ---
    print("\n--- Evaluating on Test Set ---")
    # Ensure result object exists from successful training
    if result is None:
        print("Training did not produce a result object. Skipping evaluation.")
        ray.shutdown()
        return

    best_checkpoint = result.checkpoint
    if best_checkpoint:
        print(f"Loading best model from checkpoint: {best_checkpoint.path}")
        try:
            # === CORRECTED: Load state dict from checkpoint object/path ===
            # Re-create the model architecture using the correct vocab size
            eval_model = LSTMSentimentClassifier(
                vocab_size=ACTUAL_VOCAB_SIZE, embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM,
                output_dim=NUM_CLASSES, n_layers=NUM_LSTM_LAYERS, bidirectional=True,
                dropout=DROPOUT_PROB, pad_idx=PAD_IDX
            )

            # Load the state dict using the checkpoint object's method if available,
            # otherwise load manually from path
            if hasattr(best_checkpoint, 'get_state_dict'):
                 model_state = best_checkpoint.get_state_dict()
                 print("Loaded state dict using checkpoint.get_state_dict()")
            else:
                 # Fallback: Load manually from path
                 with best_checkpoint.as_directory() as checkpoint_dir:
                      # Try common filenames for saved state dict
                      potential_paths = [
                           os.path.join(checkpoint_dir, TorchCheckpoint.MODEL_FILENAME), # Preferred
                           os.path.join(checkpoint_dir, "model.pt"),
                           os.path.join(checkpoint_dir, "torch_state_dict.pt")
                      ]
                      model_state_path = None
                      for p in potential_paths:
                           if os.path.exists(p):
                                model_state_path = p
                                break
                      if model_state_path is None:
                           raise FileNotFoundError(f"Could not find model state file in checkpoint directory: {checkpoint_dir}")

                      print(f"Loading state dict manually from: {model_state_path}")
                      # Set weights_only=True if possible and safe
                      try:
                          model_state = torch.load(model_state_path, map_location='cpu', weights_only=True)
                          print("Loaded state dict with weights_only=True")
                      except Exception as e_weights_only:
                          print(f"Could not load with weights_only=True ({e_weights_only}), trying default...")
                          # Ensure map_location is set for CPU loading consistency
                          model_state = torch.load(model_state_path, map_location='cpu')
                          print("Loaded state dict with weights_only=False")

            # Load the state dict into the evaluation model
            # Check if loaded object is the state_dict or the full model
            if isinstance(model_state, dict):
                 eval_model.load_state_dict(model_state)
            elif isinstance(model_state, nn.Module):
                 eval_model = model_state # Use the loaded model directly
            else:
                 raise TypeError(f"Loaded checkpoint data is not a state_dict or nn.Module: {type(model_state)}")

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            eval_model.to(device)
            eval_model.eval() # Set model to evaluation mode

            print("Predicting on test set...")
            all_preds = []
            all_true_labels = []
            all_texts = [] # Store texts for saving predictions
            # Manually iterate, batch, and collate the test set
            current_test_batch = []
            # Use iter_rows on the original test dataset split
            for row in test_dataset_processed.iter_rows():
                 current_test_batch.append(row)
                 # Use a potentially larger batch size for evaluation efficiency
                 if len(current_test_batch) >= BATCH_SIZE_PER_WORKER * 2:
                      batch_data = collate_batch(current_test_batch, device)
                      current_test_batch = []
                      if batch_data["labels"].numel() == 0: continue

                      ids = batch_data["text_ids"]
                      lengths = batch_data["lengths"]
                      labels = batch_data["labels"] # Labels are on device from collate
                      texts = batch_data["original_texts"] # Get original texts

                      with torch.no_grad():
                           predictions = eval_model(ids, lengths)
                           preds = torch.argmax(predictions, dim=1)
                           all_preds.extend(preds.cpu().numpy())
                           all_true_labels.extend(labels.cpu().numpy()) # Collect labels
                           all_texts.extend(texts) # Collect texts

            # Process the last partial test batch
            if current_test_batch:
                 batch_data = collate_batch(current_test_batch, device)
                 if batch_data["labels"].numel() > 0:
                      ids = batch_data["text_ids"]
                      lengths = batch_data["lengths"]
                      labels = batch_data["labels"]
                      texts = batch_data["original_texts"]
                      with torch.no_grad():
                           predictions = eval_model(ids, lengths)
                           preds = torch.argmax(predictions, dim=1)
                           all_preds.extend(preds.cpu().numpy())
                           all_true_labels.extend(labels.cpu().numpy())
                           all_texts.extend(texts)


            # --- Added Debug Prints ---
            print(f"Evaluation complete. Found {len(all_true_labels)} true labels and {len(all_preds)} predictions.")
            print(f"Number of texts collected: {len(all_texts)}")

            if len(all_preds) == len(all_true_labels) and len(all_true_labels) > 0:
                # --- Save Classification Report ---
                print("\nClassification Report (Test Set):")
                # Use labels argument to ensure order matches id2label
                class_labels = list(id2label.values())
                class_ids = list(id2label.keys())
                try:
                    report = classification_report(
                        all_true_labels,
                        all_preds,
                        labels=class_ids, # Ensure order
                        target_names=class_labels,
                        digits=4
                    )
                    print(report)
                    report_path = os.path.join(PLOTS_DIR, "classification_report_lstm.txt")
                    # Ensure directory exists
                    os.makedirs(PLOTS_DIR, exist_ok=True)
                    with open(report_path, "w") as f:
                        f.write("Classification Report (Test Set):\n")
                        f.write(report)
                    print(f"Classification report saved to {report_path}")
                except Exception as report_err:
                    print(f"Error generating/saving classification report: {report_err}")


                # --- Calculate and Save Other Metrics ---
                try:
                    metrics_dict = {
                        "accuracy": accuracy_score(all_true_labels, all_preds),
                        "f1_weighted": f1_score(all_true_labels, all_preds, average="weighted", labels=class_ids),
                        "precision_weighted": precision_score(all_true_labels, all_preds, average="weighted", labels=class_ids, zero_division=0),
                        "recall_weighted": recall_score(all_true_labels, all_preds, average="weighted", labels=class_ids, zero_division=0),
                        "f1_macro": f1_score(all_true_labels, all_preds, average="macro", labels=class_ids),
                        "precision_macro": precision_score(all_true_labels, all_preds, average="macro", labels=class_ids, zero_division=0),
                        "recall_macro": recall_score(all_true_labels, all_preds, average="macro", labels=class_ids, zero_division=0),
                    }
                    print("\nEvaluation Metrics:")
                    print(json.dumps(metrics_dict, indent=2))
                    metrics_path = os.path.join(PLOTS_DIR, "evaluation_metrics_lstm.json")
                    # Ensure directory exists
                    os.makedirs(PLOTS_DIR, exist_ok=True)
                    with open(metrics_path, "w") as f:
                        json.dump(metrics_dict, f, indent=2)
                    print(f"Evaluation metrics saved to {metrics_path}")
                except Exception as metrics_err:
                    print(f"Error calculating/saving evaluation metrics: {metrics_err}")

                # --- Plot Confusion Matrix ---
                try:
                    plot_confusion_matrix(all_true_labels, all_preds, class_labels, PLOTS_DIR)
                except Exception as plot_cm_err:
                    print(f"Error plotting confusion matrix: {plot_cm_err}")


                # --- Save True vs Predicted Labels with Text ---
                if len(all_texts) == len(all_true_labels): # Ensure text list matches label lists
                    try:
                        predictions_df = pd.DataFrame({
                            'cleaned_text': all_texts,
                            'true_label_id': all_true_labels,
                            'predicted_label_id': all_preds
                        })
                        # Map IDs back to string labels
                        predictions_df['true_label'] = predictions_df['true_label_id'].map(id2label)
                        predictions_df['predicted_label'] = predictions_df['predicted_label_id'].map(id2label)
                        # Save to CSV
                        predictions_csv_path = os.path.join(OUTPUT_DIR, "test_predictions_lstm.csv")
                        predictions_df.to_csv(predictions_csv_path, index=False)
                        print(f"Test predictions saved to {predictions_csv_path}")
                    except Exception as csv_err:
                        print(f"Error saving predictions CSV: {csv_err}")
                else:
                    print(f"Warning: Length mismatch between collected texts ({len(all_texts)}) and labels ({len(all_true_labels)}). Skipping saving predictions CSV.")


            elif len(all_true_labels) == 0:
                 print("Error: No valid true labels found for evaluation.")
            else:
                print(f"Error: Mismatch between predicted ({len(all_preds)}) and true ({len(all_true_labels)}) label counts during evaluation.")

        except Exception as eval_exc:
            print(f"Error during evaluation on test set: {eval_exc}")

    else:
        print("No checkpoint found from training result. Cannot evaluate test set.")

    # --- Plot Training History ---
    history_data = None
    if result:
        # Check multiple potential locations for metrics history
        if hasattr(result, 'metrics_dataframe') and result.metrics_dataframe is not None:
            print("\nPlotting training history from metrics_dataframe...")
            history_data = result.metrics_dataframe
        elif hasattr(result, 'metrics') and isinstance(result.metrics, list) and len(result.metrics) > 0 and isinstance(result.metrics[0], dict):
             print("\nPlotting training history from metrics list...")
             history_data = result.metrics # Ray may return list of dicts
        elif hasattr(result, 'metrics') and isinstance(result.metrics, dict):
             print("\nPlotting training history from final metrics dict (limited)...")
             history_data = [result.metrics] # Wrap final dict in list
        elif hasattr(result, 'best_result') and hasattr(result.best_result, 'metrics_dataframe'):
             print("\nPlotting training history from best_result.metrics_dataframe...")
             history_data = result.best_result.metrics_dataframe

    if history_data is not None:
         try:
              plot_training_history(history_data, PLOTS_DIR)
         except Exception as plot_hist_err:
              print(f"Error plotting training history: {plot_hist_err}")
    else:
        print("No training history data found in result object to plot.")


    print(f"\nModel training output, checkpoints, and plots saved in: {OUTPUT_DIR}")
    ray.shutdown()
    print("Ray shutdown.")

if __name__ == "__main__":
    main()
