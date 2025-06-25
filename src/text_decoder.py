# src/text_decoder.py
import torch
import torch.nn as nn
import logging

# Import the centralized initialize_weights from utils
from src.utils import initialize_weights

# Get logger for this module
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO) # Set to INFO for less verbose output during normal runs


class TextDecoder(nn.Module):
    """
    A GRU-based text decoder that takes an embedding and generates a sequence of tokens.
    This version has been refactored to perform a single decoding step at a time,
    allowing for more flexible decoding strategies (e.g., greedy, sampling) in the main model.
    """
    def __init__(self, embedding_dim: int, hidden_size: int, vocab_size: int, sos_token_id: int, embedding_dropout_rate: float = 0.1):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.sos_token_id = sos_token_id

        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.embedding_dropout = nn.Dropout(embedding_dropout_rate)

        self.linear_to_hidden = nn.Linear(embedding_dim, hidden_size)
        self.gru = nn.GRUCell(hidden_size, hidden_size)
        self.hidden_to_vocab = nn.Linear(hidden_size, vocab_size)

        initialize_weights(self) # Apply weight initialization from utils

        logger.debug(f"Initialized TextDecoder with embedding_dim={embedding_dim}, hidden_size={hidden_size}, vocab_size={vocab_size}, sos_token_id={sos_token_id}")

    def get_initial_hidden(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        Projects the input embedding to an initial hidden state for the GRU.
        """
        return torch.relu(self.linear_to_hidden(embedding))

    def forward(self, input_token: torch.Tensor, hidden_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Performs a single step of decoding.

        Args:
            input_token (torch.Tensor): The input token for the current timestep (batch_size).
            hidden_state (torch.Tensor): The current hidden state of the GRU (batch_size, hidden_size).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - token_logits (torch.Tensor): The logits for the next token prediction (batch_size, vocab_size).
                - new_hidden_state (torch.Tensor): The updated hidden state (batch_size, hidden_size).
        """
        batch_size = input_token.shape[0]
        device = input_token.device

        # Embed the current input token
        embedded_input = self.embedding_dropout(self.token_embedding(input_token))
        logger.debug(f"TextDecoder single step, embedded_input shape: {embedded_input.shape}")

        # Pass through GRU cell
        new_hidden_state = self.gru(embedded_input, hidden_state)
        logger.debug(f"TextDecoder single step, new GRU hidden state shape: {new_hidden_state.shape}")

        # Predict logits for the next token
        token_logits = self.hidden_to_vocab(new_hidden_state)
        logger.debug(f"TextDecoder single step, output logits shape: {token_logits.shape}")

        return token_logits, new_hidden_state

    def forward_teacher_forced(self, embedding: torch.Tensor, max_len: int, target_sequence: torch.Tensor) -> torch.Tensor:
        """
        Generates a sequence of token logits using teacher forcing.
        Used during training where the ground truth is known.
        """
        batch_size = embedding.shape[0]
        device = embedding.device

        hidden = self.get_initial_hidden(embedding)
        all_token_logits = []

        # Start with the <sos> token
        input_token = torch.full((batch_size,), self.sos_token_id, dtype=torch.long, device=device)

        for t in range(max_len):
            token_logits, hidden = self.forward(input_token, hidden)
            all_token_logits.append(token_logits.unsqueeze(1))

            # Teacher forcing: Use the actual next token from the target sequence
            # This is safe because target_sequence includes the <sos> token at the start
            # and we are predicting the token for the *next* step.
            if t < target_sequence.shape[1]:
                 input_token = target_sequence[:, t]
            else:
                 # Should not happen if max_len aligns with sequence length, but as a fallback:
                 input_token = torch.argmax(token_logits, dim=-1)


        output_logits = torch.cat(all_token_logits, dim=1)
        logger.debug(f"TextDecoder teacher-forced output logits shape: {output_logits.shape}")
        return output_logits