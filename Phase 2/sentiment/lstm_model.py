import torch
import torch.nn as nn
import torch.nn.functional as F  # Needed for softmax

# --- PyTorch LSTM Model Definition with Attention ---
class LSTMSentimentClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim,
                 n_layers, bidirectional, dropout, pad_idx):
        super().__init__()

        # Initialize model parameters
        self.vocab_size = int(vocab_size)
        self.embedding_dim = int(embedding_dim)
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)
        self.n_layers = int(n_layers)
        self.bidirectional = bidirectional
        self.pad_idx = int(pad_idx)
        self.dropout_prob = dropout

        # Embedding layer: Converts input tokens to dense vectors
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=self.pad_idx)

        # LSTM layer: Processes sequences of embeddings
        self.lstm = nn.LSTM(self.embedding_dim,
                            self.hidden_dim,
                            num_layers=self.n_layers,
                            bidirectional=self.bidirectional,
                            dropout=self.dropout_prob if self.n_layers > 1 else 0,  # Dropout only if more than 1 layer
                            batch_first=True)  # Input/output tensors have batch size as the first dimension

        # --- Attention Layer ---
        # Attention dimension depends on whether the LSTM is bidirectional
        self.attention_dim = self.hidden_dim * 2 if self.bidirectional else self.hidden_dim

        # Learnable weight matrix for attention mechanism
        self.attention_W = nn.Linear(self.attention_dim, self.attention_dim, bias=False)
        # Learnable vector for scoring attention
        self.attention_v = nn.Linear(self.attention_dim, 1, bias=False)
        # --- End Attention Layer ---

        # Fully connected layer for final classification
        self.fc = nn.Linear(self.attention_dim, self.output_dim)
        # Dropout layer to prevent overfitting
        self.dropout = nn.Dropout(self.dropout_prob)

    def attention(self, lstm_output, final_hidden_state):
        """
        Applies an attention mechanism.
        lstm_output shape: (batch_size, seq_len, hidden_dim * num_directions)
        final_hidden_state shape: (batch_size, hidden_dim * num_directions) - Usually the last relevant hidden state
        """
        # Project LSTM outputs using a learnable weight matrix
        # u = tanh(W * lstm_output)
        # W * lstm_output shape: (batch_size, seq_len, attention_dim)
        u = torch.tanh(self.attention_W(lstm_output))

        # Calculate attention scores using a learnable vector
        # v * u shape: (batch_size, seq_len, 1) -> squeeze -> (batch_size, seq_len)
        attn_scores = self.attention_v(u).squeeze(2)

        # Apply softmax to normalize attention scores into probabilities
        # attn_weights shape: (batch_size, seq_len)
        attn_weights = F.softmax(attn_scores, dim=1)

        # Calculate context vector as a weighted sum of LSTM outputs
        # Unsqueeze attn_weights to (batch_size, 1, seq_len) for batch matrix multiplication
        # context = attn_weights * lstm_output -> (batch_size, 1, seq_len) * (batch_size, seq_len, attention_dim)
        # Resulting context shape: (batch_size, attention_dim)
        context = torch.bmm(attn_weights.unsqueeze(1), lstm_output).squeeze(1)

        return context, attn_weights

    def forward(self, text_ids, text_lengths):
        # text_ids = [batch size, seq len]

        # Pass input through embedding layer
        # embedded = [batch size, seq len, emb dim]
        embedded = self.dropout(self.embedding(text_ids))

        # Pass embeddings through LSTM
        # lstm_output = [batch size, seq len, hid dim * num directions]
        # hidden = [n layers * num directions, batch size, hid dim]
        # cell = [n layers * num directions, batch size, hid dim]
        lstm_output, (hidden, cell) = self.lstm(embedded)

        # --- Use Attention ---
        # Create the final hidden state to be used in attention calculation
        # For bidirectional LSTM, concatenate the last forward and backward hidden states
        if self.lstm.bidirectional:
            # hidden shape: (batch_size, hidden_dim * 2)
            final_hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        else:
            # hidden shape: (batch_size, hidden_dim)
            final_hidden = hidden[-1,:,:]

        # Calculate context vector using attention mechanism
        # context shape: (batch_size, attention_dim)
        context_vector, attention_weights = self.attention(lstm_output, final_hidden)
        # --- End Use Attention ---

        # Apply dropout to the context vector before the final layer
        context_vector_dropout = self.dropout(context_vector)

        # Pass the context vector through the final fully connected layer
        # output = [batch size, out dim]
        return self.fc(context_vector_dropout)
