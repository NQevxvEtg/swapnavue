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
TEXT_CACHE_FILE = "texts_map.json" # Changed to reflect it's a map
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

# Modified to save a dictionary (filepath -> lines)
def _save_data_to_cache(data_directory: str, base_cache_dir: str, texts_map: dict[str, list[str]], current_manifest: dict):
    """Saves loaded texts_map (filepath -> lines) and the corresponding manifest to cache."""
    unique_dataset_cache_dir, text_cache_path, manifest_cache_path = _get_unique_dataset_cache_paths(data_directory, base_cache_dir)
    os.makedirs(unique_dataset_cache_dir, exist_ok=True)
    
    try:
        with open(text_cache_path, 'w', encoding='utf-8') as f:
            json.dump(texts_map, f, ensure_ascii=False, indent=2)
        with open(manifest_cache_path, 'w', encoding='utf-8') as f:
            json.dump(current_manifest, f, ensure_ascii=False, indent=2)
        logger.debug(f"Data map and manifest saved to cache in {unique_dataset_cache_dir}") # Changed to debug for less verbosity
    except Exception as e:
        logger.error(f"Failed to save data to cache {unique_dataset_cache_dir}: {e}")

# Modified to load a dictionary (filepath -> lines)
def _load_data_from_cache(data_directory: str, base_cache_dir: str) -> dict[str, list[str]] | None:
    """Loads texts_map (filepath -> lines) from cache."""
    _, text_cache_path, _ = _get_unique_dataset_cache_paths(data_directory, base_cache_dir)
    try:
        with open(text_cache_path, 'r', encoding='utf-8') as f:
            texts_map = json.load(f)
        logger.info(f"Loaded data map from cache: {text_cache_path}")
        return texts_map
    except Exception as e:
        logger.debug(f"Failed to load data map from cache {text_cache_path}: {e}. This is normal if cache is new/invalid.")
        return None

def _load_manifest_from_cache(data_directory: str, base_cache_dir: str) -> dict | None:
    """Loads manifest from cache."""
    _, _, manifest_cache_path = _get_unique_dataset_cache_paths(data_directory, base_cache_dir)
    try:
        if os.path.exists(manifest_cache_path):
            with open(manifest_cache_path, 'r', encoding='utf-8') as f:
                manifest = json.load(f)
            logger.debug(f"Loaded manifest from cache: {manifest_cache_path}")
            return manifest
        return None
    except Exception as e:
        logger.debug(f"Failed to load manifest from cache {manifest_cache_path}: {e}. This is normal if cache is new/invalid.")
        return None

def _is_file_metadata_changed(current_stats: dict, cached_stats: dict) -> bool:
    """Checks if a single file's metadata (mtime, size, content_hash) has changed."""
    return (current_stats['mtime'] != cached_stats['mtime'] or
            current_stats['size'] != cached_stats['size'] or
            current_stats.get('content_hash') != cached_stats.get('content_hash'))

def create_file_manifest(directory: str) -> dict[str, dict]:
    """
    Creates a manifest of data files in a directory, recording their
    last modification time, size, AND content hash for robust change detection.
    """
    manifest = {}
    file_patterns = ['*.txt', '*.tokens', '*.md', '*.csv', '*.pdf']
    
    for pattern in file_patterns:
        for filepath in glob.glob(os.path.join(directory, pattern)):
            try:
                stats = os.stat(filepath)
                normalized_path = os.path.normpath(filepath)
                
                # Calculate content hash
                file_hash = hashlib.sha256()
                with open(filepath, 'rb') as f:
                    # Read in chunks to handle large files efficiently
                    while chunk := f.read(8192):
                        file_hash.update(chunk)
                
                manifest[normalized_path] = {
                    "mtime": stats.st_mtime,
                    "size": stats.st_size,
                    "content_hash": file_hash.hexdigest() # Store the hash
                }
            except FileNotFoundError:
                continue
    return manifest

def get_changed_files(directory: str, old_manifest: dict[str, dict]) -> tuple[list[str], dict[str, dict]]:
    """
    Compares the current state of a directory to an old manifest and returns
    a list of new or modified file paths, and the new full manifest.
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
            if _is_file_metadata_changed(current_stats, old_stats):
                changed_files.append(filepath)
                
    return changed_files, current_manifest # Also return current_manifest

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

def _process_single_file_for_cache(file_path: str) -> list[str]:
    """Helper function to load texts from a single file (PDF or text)."""
    try:
        logger.debug(f"Caching file: {os.path.basename(file_path)}")
        if file_path.lower().endswith('.pdf'):
            lines = load_texts_from_pdf(file_path)
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f if line.strip()] 
        logger.debug(f"Finished processing {len(lines)} non-empty lines from: {os.path.basename(file_path)}") # Updated message
        return lines
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {e}")
        return []

def load_texts_from_directory(directory: str, base_cache_dir: str) -> list[str]:
    """
    Loads all lines from text files (.txt, .tokens, .md, .csv, .pdf) in a given directory,
    leveraging a cache for faster re-loads if data hasn't changed.
    This version supports true incremental loading and resuming by updating per-file.
    """
    logger.info(f"Attempting to load texts from directory: {directory} with cache from {base_cache_dir}.")
    
    # 1. Load existing cache data (as a map) and manifest, if any
    cached_texts_map = _load_data_from_cache(directory, base_cache_dir)
    if cached_texts_map is None:
        cached_texts_map = {} # Initialize empty if no cache exists

    cached_manifest = _load_manifest_from_cache(directory, base_cache_dir)
    if cached_manifest is None:
        cached_manifest = {} # Initialize empty if no manifest exists
    
    # 2. Get the current state of files on disk
    # This also returns the full current manifest of files on disk
    files_to_process, disk_manifest = get_changed_files(directory, cached_manifest)

    # 3. Identify files that were in cache but are no longer on disk (deleted files)
    deleted_files = [fp for fp in cached_manifest if fp not in disk_manifest]

    # --- Process deleted files ---
    if deleted_files:
        logger.info(f"Removing {len(deleted_files)} deleted files from cache for {directory}.")
        for fp in deleted_files:
            if fp in cached_texts_map:
                del cached_texts_map[fp]
            if fp in cached_manifest:
                del cached_manifest[fp]
        # Persist deletion changes
        _save_data_to_cache(directory, base_cache_dir, cached_texts_map, cached_manifest)

    # --- Process new/modified files iteratively ---
    if files_to_process:
        logger.info(f"Processing {len(files_to_process)} new/modified files for {directory}. Saving incrementally.")
        # Use ThreadPoolExecutor for processing to still benefit from parallelism
        # but save results per file as they complete.
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            future_to_file = {executor.submit(_process_single_file_for_cache, fp): fp for fp in files_to_process}
            
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    lines = future.result()
                    logger.info(f"Caching content for file: {os.path.basename(file_path)}") # New INFO log for each file
                    if lines: # Only cache if lines were successfully extracted
                        cached_texts_map[file_path] = lines
                        # Update manifest entry for this specific file with disk_manifest's data
                        cached_manifest[file_path] = disk_manifest[file_path] 
                        # Save after each file to ensure resume capability
                        _save_data_to_cache(directory, base_cache_dir, cached_texts_map, cached_manifest)
                    else:
                        logger.warning(f"File {os.path.basename(file_path)} had no extractable text. Skipping from cache.")
                        # If file previously had content and now doesn't, ensure it's removed
                        if file_path in cached_texts_map:
                            del cached_texts_map[file_path]
                        if file_path in cached_manifest:
                            del cached_manifest[file_path]
                        _save_data_to_cache(directory, base_cache_dir, cached_texts_map, cached_manifest)

                except Exception as e:
                    logger.error(f"Error processing and caching file {file_path}: {e}")
                    # Even on error for one file, we save the progress for others
                    _save_data_to_cache(directory, base_cache_dir, cached_texts_map, cached_manifest)


    if not files_to_process and not deleted_files:
        logger.info(f"All files in {directory} are up-to-date with cache. Loaded {sum(len(v) for v in cached_texts_map.values())} total lines.")
    else:
        logger.info(f"Incremental processing complete for {directory}. Total texts after update: {sum(len(v) for v in cached_texts_map.values())} lines.")

    # Convert the map of texts into a flat list for the TextDataset
    all_final_texts = []
    # Ensure stable order by sorting file paths
    for filepath in sorted(cached_texts_map.keys()):
        all_final_texts.extend(cached_texts_map[filepath])
    
    return all_final_texts


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