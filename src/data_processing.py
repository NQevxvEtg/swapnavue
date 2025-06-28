# src/data_processing.py
import torch
from torch.utils.data import DataLoader, Dataset
from src.utils import Vocabulary # Import Vocabulary from our utilities
import glob
import os
import logging
import json

logger = logging.getLogger(__name__)

def create_file_manifest(directory: str) -> dict[str, dict]:
    """
    Creates a manifest of data files in a directory, recording their
    last modification time and size for change detection.
    """
    manifest = {}
    file_patterns = ['*.txt', '*.tokens', '*.md', '*.csv']
    
    for pattern in file_patterns:
        for filepath in glob.glob(os.path.join(directory, pattern)):
            try:
                stats = os.stat(filepath)
                # Use a normalized path for consistency across OS
                normalized_path = os.path.normpath(filepath)
                manifest[normalized_path] = {
                    "mtime": stats.st_mtime,
                    "size": stats.st_size
                }
            except FileNotFoundError:
                # File might be deleted during scan, just skip it
                continue
    return manifest

def get_changed_files(directory: str, old_manifest: dict[str, dict]) -> list[str]:
    """
    Compares the current state of a directory to an old manifest and returns
    a list of new or modified file paths.
    """
    changed_files = []
    current_manifest = create_file_manifest(directory)

    for filepath, current_stats in current_manifest.items():
        if filepath not in old_manifest:
            # File is new
            changed_files.append(filepath)
        else:
            # File exists, check for modification
            old_stats = old_manifest[filepath]
            if current_stats['mtime'] != old_stats['mtime'] or current_stats['size'] != old_stats['size']:
                changed_files.append(filepath)
                
    # Also returns the generated current_manifest to avoid doing the work twice
    return changed_files, current_manifest

def load_texts_from_specific_files(filepaths: list[str]) -> list[str]:
    """Loads all lines from a specific list of text files."""
    all_texts = []
    for file_path in filepaths:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                all_texts.extend(lines)
            logger.info(f"Loading {len(lines)} lines from changed file: {os.path.basename(file_path)}")
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
    return all_texts

def load_texts_from_directory(directory: str) -> list[str]:
    """
    Loads all lines from text files (.txt, .tokens, .md, .csv) in a given directory.
    This is a synchronous function suitable for startup or background thread execution.
    """
    all_texts = []
    # Define file patterns to search for
    file_patterns = ['*.txt', '*.tokens', '*.md', '*.csv']
    text_files = []
    for pattern in file_patterns:
        text_files.extend(glob.glob(os.path.join(directory, pattern)))

    if not text_files:
        logger.warning(f"No text files matching patterns {file_patterns} found in {directory}.")
        return []

    return load_texts_from_specific_files(text_files)

# --- Dataset and Dataloader ---
class TextDataset(Dataset):
    """
    A PyTorch Dataset for handling text data, performing tokenization,
    and preparing target sequences for the CRP model.
    """
    def __init__(self, texts: list[str], vocab: Vocabulary, max_seq_len: int):
        self.texts = texts
        self.vocab = vocab
        self.max_seq_len = max_seq_len

    def __len__(self):
        """Returns the total number of text samples."""
        return len(self.texts)

    def __getitem__(self, idx: int):
        """
        Retrieves a single text sample and processes it into input text
        for embedding and a tokenized target sequence.
        """
        text = self.texts[idx]

        # The original text is passed for sentence embedding by the model itself
        input_embedding_text = text

        # Tokenize the text using our custom vocabulary's encode method
        tokens = self.vocab.encode(text)

        # Manually add <SOS> and <EOS> tokens to the target sequence
        processed_tokens = [self.vocab.sos_token_id] + tokens + [self.vocab.eos_token_id]

        # Truncate if the sequence is longer than max_seq_len
        # (max_seq_len includes space for SOS and EOS)
        if len(processed_tokens) > self.max_seq_len:
            # Ensure EOS token is always present if truncation occurs
            processed_tokens = processed_tokens[:self.max_seq_len - 1] + [self.vocab.eos_token_id]

        # Pad if the sequence is shorter than max_seq_len
        if len(processed_tokens) < self.max_seq_len:
            target_sequence = processed_tokens + [self.vocab.pad_token_id] * (self.max_seq_len - len(processed_tokens))
        else:
            target_sequence = processed_tokens

        return {
            'input_text': input_embedding_text, # Original text for sentence embedding
            'target_sequence': torch.tensor(target_sequence, dtype=torch.long)
        }

def collate_fn(batch: list[dict]) -> tuple[list[str], torch.Tensor]:
    """
    Collation function for DataLoader.
    Stacks target sequences and collects input texts for batch processing.
    """
    input_texts = [item['input_text'] for item in batch]
    target_sequences = torch.stack([item['target_sequence'] for item in batch])
    return input_texts, target_sequences