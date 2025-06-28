# src/data_processing.py
import torch
from torch.utils.data import DataLoader, Dataset
from src.utils import Vocabulary # Import Vocabulary from our utilities
import glob
import os
import logging
import json
import re # Import regex module
import io # For StringIO
from concurrent.futures import ThreadPoolExecutor, as_completed # Import for parallel loading
import hashlib # For creating unique cache directory names

# Import necessary components from pdfminer.six
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage

logger = logging.getLogger(__name__)

# --- Cache Configuration - Centralized Management ---
# Constants for filenames within the unique dataset cache directory
TEXT_CACHE_FILE = "texts.json"
MANIFEST_CACHE_FILE = "manifest.json"

def _get_unique_dataset_cache_paths(data_directory: str, base_cache_dir: str) -> tuple[str, str, str]:
    """
    Returns paths for the unique dataset's cache directory, text cache file,
    and manifest cache file within the given base_cache_dir.
    Uses a hash of the data_directory path to ensure uniqueness.
    """
    # Normalize the data_directory path to ensure consistent hashing
    normalized_data_dir = os.path.normpath(data_directory)
    # Create a unique identifier for this specific data directory
    dir_hash = hashlib.md5(normalized_data_dir.encode('utf-8')).hexdigest()
    
    unique_dataset_cache_dir = os.path.join(base_cache_dir, dir_hash)
    text_cache_path = os.path.join(unique_dataset_cache_dir, TEXT_CACHE_FILE)
    manifest_cache_path = os.path.join(unique_dataset_cache_dir, MANIFEST_CACHE_FILE)
    return unique_dataset_cache_dir, text_cache_path, manifest_cache_path

def _save_data_to_cache(data_directory: str, base_cache_dir: str, texts: list[str], current_manifest: dict):
    """Saves loaded texts and the corresponding manifest to cache."""
    unique_dataset_cache_dir, text_cache_path, manifest_cache_path = _get_unique_dataset_cache_paths(data_directory, base_cache_dir)
    os.makedirs(unique_dataset_cache_dir, exist_ok=True)
    
    try:
        with open(text_cache_path, 'w', encoding='utf-8') as f:
            json.dump(texts, f, ensure_ascii=False, indent=2)
        with open(manifest_cache_path, 'w', encoding='utf-8') as f:
            json.dump(current_manifest, f, ensure_ascii=False, indent=2)
        logger.info(f"Data and manifest saved to cache in {unique_dataset_cache_dir}")
    except Exception as e:
        logger.error(f"Failed to save data to cache {unique_dataset_cache_dir}: {e}")

def _load_data_from_cache(data_directory: str, base_cache_dir: str) -> list[str] | None:
    """Loads texts from cache."""
    _, text_cache_path, _ = _get_unique_dataset_cache_paths(data_directory, base_cache_dir)
    try:
        with open(text_cache_path, 'r', encoding='utf-8') as f:
            texts = json.load(f)
        logger.info(f"Loaded data from cache: {text_cache_path}")
        return texts
    except Exception as e:
        logger.error(f"Failed to load data from cache {text_cache_path}: {e}")
        return None

def _is_cache_valid(data_directory: str, base_cache_dir: str, current_manifest: dict) -> bool:
    """Checks if the cached data is valid based on the manifest."""
    _, _, manifest_cache_path = _get_unique_dataset_cache_paths(data_directory, base_cache_dir)
    
    if not os.path.exists(manifest_cache_path):
        logger.debug(f"Cache manifest not found at {manifest_cache_path}. Cache invalid.")
        return False
    
    try:
        with open(manifest_cache_path, 'r', encoding='utf-8') as f:
            cached_manifest = json.load(f)
        
        # Compare manifests. If any file's mtime or size differs, or files are added/removed, cache is invalid.
        if cached_manifest == current_manifest:
            logger.debug(f"Cache manifest matches current manifest for {data_directory}. Cache is valid.")
            return True
        else:
            logger.info(f"Manifest mismatch for {data_directory}. Cache invalid.")
            return False
            
    except Exception as e:
        logger.error(f"Error validating cache manifest for {data_directory}: {e}")
        return False


def create_file_manifest(directory: str) -> dict[str, dict]:
    """
    Creates a manifest of data files in a directory, recording their
    last modification time and size for change detection.
    """
    manifest = {}
    # Include .pdf in the file patterns
    file_patterns = ['*.txt', '*.tokens', '*.md', '*.csv', '*.pdf']
    
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

def clean_text(text: str) -> str:
    """
    Cleans extracted text by removing common unwanted elements like page numbers
    and excessive whitespace.
    """
    # Remove common page number patterns (e.g., numbers at start/end of line, surrounded by spaces)
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE) # Lines containing only numbers
    text = re.sub(r'\bPage\s+\d+\b', '', text, flags=re.IGNORECASE) # "Page X"
    text = re.sub(r'-\s*\d+\s*-', '', text) # "- X -"

    # Replace multiple newlines with a single one, and strip leading/trailing whitespace
    text = re.sub(r'\n\s*\n', '\n', text).strip()
    return text

def load_texts_from_pdf(filepath: str) -> list[str]:
    """
    Loads all text from a single PDF file using pdfminer.six,
    extracting content page by page and applying cleaning.
    """
    all_pdf_texts = []
    try:
        rsrcmgr = PDFResourceManager()
        retstr = io.StringIO()
        laparams = LAParams() # Layout analysis parameters
        
        # Create a PDF device object that writes text into our StringIO object
        device = TextConverter(rsrcmgr, retstr, laparams=laparams)
        
        with open(filepath, 'rb') as fp:
            interpreter = PDFPageInterpreter(rsrcmgr, device)
            num_pages = 0
            for page_num, page in enumerate(PDFPage.get_pages(fp)):
                num_pages += 1
                interpreter.process_page(page)
                text = retstr.getvalue()
                
                # Clear the StringIO buffer for the next page
                retstr.truncate(0)
                retstr.seek(0)
                
                if text:
                    cleaned_text = clean_text(text)
                    if cleaned_text: # Only add if cleaned text is not empty
                        all_pdf_texts.append(cleaned_text)
            
            logger.info(f"Extracted text from PDF: {os.path.basename(filepath)} ({num_pages} pages)")
        
        device.close()
        retstr.close()
        
    except Exception as e:
        logger.error(f"Error loading PDF {filepath}: {e}")
    return all_pdf_texts

def _process_single_file(file_path: str) -> list[str]:
    """Helper function to load texts from a single file (PDF or text)."""
    try:
        if file_path.lower().endswith('.pdf'):
            lines = load_texts_from_pdf(file_path)
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f if line.strip()] 
        logger.info(f"Loaded {len(lines)} non-empty lines from file: {os.path.basename(file_path)}")
        return lines
    except Exception as e:
        logger.error(f"Error loading {file_path}: {e}")
        return []

def load_texts_from_specific_files(filepaths: list[str]) -> list[str]:
    """
    Loads all lines from a specific list of text files in parallel.
    """
    all_texts = []
    # Use ThreadPoolExecutor to process files in parallel
    # You can adjust max_workers based on your CPU cores and I/O capabilities
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        # Submit each file processing to the executor
        future_to_file = {executor.submit(_process_single_file, fp): fp for fp in filepaths}
        
        for future in as_completed(future_to_file):
            file_path = future_to_file[future]
            try:
                lines = future.result()
                all_texts.extend(lines)
            except Exception as e:
                logger.error(f"Error processing file {file_path} in parallel: {e}")
    return all_texts


def load_texts_from_directory(directory: str, base_cache_dir: str) -> list[str]:
    """
    Loads all lines from text files (.txt, .tokens, .md, .csv, .pdf) in a given directory,
    leveraging a cache for faster re-loads if data hasn't changed.
    The cache is managed within a central base_cache_dir.
    """
    logger.info(f"Attempting to load texts from directory: {directory} with cache from {base_cache_dir}.")
    
    # 1. Generate current manifest for the directory
    current_manifest = create_file_manifest(directory)

    # 2. Check if cache is valid
    if _is_cache_valid(directory, base_cache_dir, current_manifest):
        cached_texts = _load_data_from_cache(directory, base_cache_dir)
        if cached_texts is not None:
            logger.info(f"Successfully loaded {len(cached_texts)} texts from cache for {directory}.")
            return cached_texts
        else:
            logger.warning(f"Cache for {directory} was deemed valid but failed to load. Re-reading from disk.")
            # Fall through to re-read from disk if loading from valid cache fails

    logger.info(f"Cache for {directory} is invalid or non-existent. Re-reading all data from disk.")
    
    # 3. If cache is invalid or non-existent, read all files from disk
    all_text_files = []
    file_patterns = ['*.txt', '*.tokens', '*.md', '*.csv', '*.pdf']
    for pattern in file_patterns:
        all_text_files.extend(glob.glob(os.path.join(directory, pattern)))

    if not all_text_files:
        logger.warning(f"No text files matching patterns {file_patterns} found in {directory}. Returning empty list.")
        return []

    # Use the parallel file loading function
    loaded_texts = load_texts_from_specific_files(all_text_files)
    
    # 4. Save newly loaded data and its manifest to cache
    _save_data_to_cache(directory, base_cache_dir, loaded_texts, current_manifest)
    
    return loaded_texts


# --- Dataset and Dataloader (rest of the file remains the same) ---
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
        Retrieves a single text sample and processes it into raw input text
        and a tokenized target sequence.
        """
        text = self.texts[idx]

        # The raw text is passed as input for the model's internal SDR encoding
        input_text = text

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
            'input_text': input_text, # Raw text for model's internal SDR encoding
            'target_sequence': torch.tensor(target_sequence, dtype=torch.long)
        }

def collate_fn(batch: list[dict]) -> tuple[list[str], torch.Tensor]:
    """
    Collation function for DataLoader.
    Stacks target sequences and collects raw input texts for batch processing.
    """
    input_texts = [item['input_text'] for item in batch]
    target_sequences = torch.stack([item['target_sequence'] for item in batch])
    return input_texts, target_sequences