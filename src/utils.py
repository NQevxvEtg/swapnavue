# src/utils.py
import json
import torch
import torch.nn as nn
from transformers import AutoTokenizer
import logging
import math

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class Vocabulary:
    """
    Manages the vocabulary for tokenization and provides utilities for
    converting between words/subwords and their numerical IDs.
    Integrates with Hugging Face tokenizers.
    """
    def __init__(self, tokenizer=None):
        """
        Initializes the Vocabulary.

        Args:
            tokenizer: An optional Hugging Face tokenizer instance (e.g., BertTokenizerFast).
                       If None, a default 'bert-base-uncased' tokenizer will be used.
        """
        self.word2idx = {}
        self.idx2word = {}
        self.manifest: dict = {} # New: Stores the file manifest used to build this vocabulary
        
        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            logger.info("Initialized default 'bert-base-uncased' tokenizer in Vocabulary.")
        else:
            self.tokenizer = tokenizer

        # Add special tokens first to ensure they get the first sequential IDs
        self.add_word('<pad>') # ID 0
        self.add_word('<sos>') # ID 1
        self.add_word('<eos>') # ID 2
        self.add_word('<unk>') # ID 3

        # Store their IDs for easy access from our custom word2idx
        self.pad_token_id = self.word2idx['<pad>']
        self.sos_token_id = self.word2idx['<sos>']
        self.eos_token_id = self.word2idx['<eos>']
        self.unk_token_id = self.word2idx['<unk>']

    def add_word(self, word: str) -> int:
        """Adds a word/token to the vocabulary if it doesn't already exist."""
        if word not in self.word2idx:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word
            return idx
        return self.word2idx[word]

    def _build_vocab_from_batch(self, texts: list[str]):
        """
        Builds vocabulary by tokenizing a large batch of texts at once,
        leveraging the tokenizer's built-in optimized parallelism.
        """
        if not self.tokenizer:
            logger.warning("No tokenizer set for Vocabulary. Cannot build vocabulary.")
            return

        logger.info(f"Tokenizing {len(texts)} lines of text in a single batch...")
        # The tokenizer call on a list of texts is highly optimized to run in parallel.
        # We set add_special_tokens=False because we handle SOS/EOS manually later.
        tokenized_output = self.tokenizer(texts, add_special_tokens=False)

        logger.info("Building vocabulary from batched tokenization output...")
        # Process the results to add words to our custom vocabulary
        for token_id_list in tokenized_output['input_ids']:
            tokens = self.tokenizer.convert_ids_to_tokens(token_id_list)
            for token in tokens:
                self.add_word(token)

    def build_vocab(self, texts: list[str], current_manifest: dict): # Modified signature
        """
        Builds the vocabulary from a list of text strings using an efficient batch method.
        Updates the internal manifest.
        """
        self._build_vocab_from_batch(texts)
        self.manifest = current_manifest # Store the manifest used to build this vocab
        logger.info(f"Vocabulary built with {len(self.word2idx)} unique custom tokens.")

    def update_vocab(self, texts: list[str], current_data_manifest: dict) -> bool: # Modified signature
        """
        Updates the vocabulary with new tokens from a list of texts using an efficient batch method.
        Returns True if the vocabulary was modified, False otherwise.
        Now intelligently checks the provided manifest to avoid unnecessary tokenization.
        """
        if self.manifest == current_data_manifest:
            logger.info("Vocabulary manifest matches current data manifest. No update needed.")
            return False

        initial_vocab_size = self.vocab_size
        logger.info(f"Starting vocabulary update check. Initial size: {initial_vocab_size}")
        logger.info("Manifest mismatch detected. Re-tokenizing all current data to update vocabulary.")
        
        self._build_vocab_from_batch(texts) # Use the new, faster batch method
        self.manifest = current_data_manifest # Update the stored manifest

        final_vocab_size = self.vocab_size
        if final_vocab_size > initial_vocab_size:
            logger.info(f"Vocabulary updated. Added {final_vocab_size - initial_vocab_size} new tokens. Final size: {final_vocab_size}")
            return True
        else:
            logger.info("Vocabulary is already up-to-date with current data. No new tokens found after re-tokenization.")
            return False

    def encode(self, text: str) -> list[int]:
        """
        Encodes a text string into a list of token IDs using the internal tokenizer
        to get subword strings, and then mapping to the custom vocabulary's IDs.
        """
        if not self.tokenizer:
            logger.warning("No tokenizer set for Vocabulary. Cannot encode text.")
            return []
        token_strings = self.tokenizer.tokenize(text)
        encoded_ids = [self.word2idx.get(token_str, self.unk_token_id) for token_str in token_strings]
        return encoded_ids

    def decode(self, token_ids: list[int]) -> str:
        """
        Decodes a list of custom vocabulary token IDs back into a string.
        """
        if not self.tokenizer:
            logger.warning("No tokenizer set for Vocabulary. Cannot decode text.")
            return ""
        
        token_strings = [self.idx2word.get(idx, self.tokenizer.unk_token) for idx in token_ids if idx in self.idx2word]
        hf_token_ids = self.tokenizer.convert_tokens_to_ids(token_strings)
        decoded_text = self.tokenizer.decode(hf_token_ids, skip_special_tokens=True)
        return decoded_text

    @property
    def vocab_size(self) -> int:
        """Returns the current size of the custom vocabulary."""
        return len(self.word2idx)

    def save_vocab(self, path: str):
        """Saves the vocabulary (word2idx mapping) and its manifest to a JSON file."""
        data_to_save = {
            'word2idx': self.word2idx,
            'manifest': self.manifest # New: Save the manifest
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f, ensure_ascii=False, indent=4)
        logger.info(f"Vocabulary and manifest saved to {path}. Size: {self.vocab_size}")

    def load_vocab(self, path: str):
        """Loads the vocabulary from a JSON file."""
        with open(path, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)
            self.word2idx = loaded_data['word2idx']
            # New: Load the manifest, handle old formats without it
            self.manifest = loaded_data.get('manifest', {}) 
            self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        
        # Ensure special tokens are correctly set or re-added if missing/lost
        self.pad_token_id = self.word2idx.get('<pad>')
        self.sos_token_id = self.word2idx.get('<sos>')
        self.eos_token_id = self.word2idx.get('<eos>')
        self.unk_token_id = self.word2idx.get('<unk>')
        
        if self.pad_token_id is None: self.add_word('<pad>'); self.pad_token_id = self.word2idx['<pad>']
        if self.sos_token_id is None: self.add_word('<sos>'); self.sos_token_id = self.word2idx['<sos>']
        if self.eos_token_id is None: self.add_word('<eos>'); self.eos_token_id = self.word2idx['<eos>']
        if self.unk_token_id is None: self.add_word('<unk>'); self.unk_token_id = self.word2idx['<unk>']

        logger.info(f"Vocabulary loaded from {path}. Size: {self.vocab_size}")


def initialize_weights(m: nn.Module):
    """
    Initializes weights of linear layers and biases.
    Applies Xavier uniform initialization to weights and sets biases to zero.
    """
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1)
    logger.debug(f"Initialized weights for {m.__class__.__name__}")


def decode_sequence(token_ids: list[int], vocab: Vocabulary, eos_token_id: int) -> str:
    """
    Decodes a list of token IDs into a human-readable string.
    Stops decoding at the first End-of-Sequence token.
    """
    # Truncate the sequence at the first <eos> token, if present
    if eos_token_id in token_ids:
        eos_index = token_ids.index(eos_token_id)
        token_ids = token_ids[:eos_index]
        
    decoded_string = vocab.decode(token_ids)
    return decoded_string


def get_optimal_tm_config(
    user_config: dict,
    available_vram: int,
    vram_budget_fraction: float = 0.0618
) -> dict:
    """
    Adjusts Temporal Memory configuration to fit within a VRAM budget.

    Args:
        user_config (dict): The user-defined TM parameters.
        available_vram (int): Available GPU VRAM in bytes.
        vram_budget_fraction (float): The fraction of available VRAM to use.

    Returns:
        dict: A new configuration that is guaranteed to fit in memory.
    """
    final_config = user_config.copy()
    bytes_per_element = 2  # Using float16 for permanences

    num_cells = final_config['num_cells']
    max_segments = final_config['max_segments_per_cell']
    max_synapses = final_config['max_synapses_per_segment']

    # There are two large permanence tensors (volatile and consolidated)
    required_mem = 2 * num_cells * max_segments * max_synapses * bytes_per_element
    vram_budget = available_vram * vram_budget_fraction

    if required_mem > vram_budget:
        logger.warning(
            f"TM config from env.txt requires {required_mem / 1024**3:.2f} GB VRAM. "
            f"This exceeds the budget of {vram_budget / 1024**3:.2f} GB."
        )

        # Solve for the largest possible 's' where s*s = new max_segments * new max_synapses
        s_squared = (vram_budget / 2) / (num_cells * bytes_per_element)
        new_s = int(math.sqrt(s_squared)) if s_squared > 0 else 0

        # Clamp to a reasonable minimum to ensure functionality
        new_s = max(16, new_s)

        logger.warning(
            f"Auto-tuning TM to fit. New max_segments/synapses per cell: {new_s}"
        )

        final_config['max_segments_per_cell'] = new_s
        final_config['max_synapses_per_segment'] = new_s
    else:
        logger.info("User-defined TM configuration fits within VRAM budget.")

    return final_config
