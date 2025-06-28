# src/text_sdr_encoder.py
import torch
import numpy as np
from typing import List, Union

from src.rdse_encoder import RDSEInstance, create_rdse, encode_scalar, overlap
from src.utils import Vocabulary # Assuming Vocabulary handles tokenizing to integer IDs

class TextSdrEncoder:
    """
    A Python implementation of the TextSdrEncoder from C++,
    using RDSE for token encoding and src.utils.Vocabulary for tokenization.
    """
    def __init__(self, vocab: Vocabulary, sdr_size: int, sdr_active_bits: int, rdse_seed: int = 42):
        """
        Initializes the TextSdrEncoder.

        Args:
            vocab (Vocabulary): An instance of the Vocabulary class for tokenization.
            sdr_size (int): The total number of bits in the SDR.
            sdr_active_bits (int): The number of active bits (sparsity) in the SDR.
            rdse_seed (int): Seed for the RDSE random number generator.
        """
        self.vocab = vocab
        self.sdr_size = sdr_size
        self.sdr_active_bits = sdr_active_bits
        
        # Mirroring C++: token_rdse_ = create_rdse(sdr_size, sdr_active_bits, sp_processor_->GetPieceSize(), 42);
        # In Python, vocab.vocab_size is equivalent to sp_processor_->GetPieceSize()
        self.token_rdse: RDSEInstance = create_rdse(
            n=sdr_size, 
            w=sdr_active_bits, 
            resolution=float(self.vocab.vocab_size), # Max token ID determines resolution
            seed=rdse_seed
        )
        print(f"Initialized TextSdrEncoder with SDR size: {sdr_size}, active bits: {sdr_active_bits}")
        print(f"RDSE Resolution (vocab size): {self.vocab.vocab_size}")

        # --- FIX: Pre-calculate SDRs for all vocab tokens once during initialization ---
        self.vocab_sdr_map = {}
        for token_id in range(self.vocab.vocab_size):
            self.vocab_sdr_map[token_id] = self.encode_single_token_id(token_id)
        print("Pre-calculated vocab SDR map for efficient decoding.")


    def encode(self, text: str) -> List[np.ndarray]:
        """
        Encodes a text string into a sequence of SDRs.
        Each token ID from the tokenizer is converted into an SDR.

        Args:
            text (str): The input text to encode.

        Returns:
            List[np.ndarray]: A list of SDRs, one for each token.
        """
        token_ids = self.vocab.encode(text)
        sdr_sequence = []
        for token_id in token_ids:
            sdr = encode_scalar(self.token_rdse, float(token_id))
            sdr_sequence.append(sdr)
        return sdr_sequence

    def encode_batch(self, texts: List[str]) -> List[List[np.ndarray]]:
        """
        Encodes a batch of text strings into sequences of SDRs.
        """
        batch_sdr_sequences = []
        for text in texts:
            batch_sdr_sequences.append(self.encode(text))
        return batch_sdr_sequences

    def encode_single_token_id(self, token_id: int) -> np.ndarray:
        """
        Encodes a single token ID into an SDR.
        Mirroring C++ TextSdrEncoder::encodeSingleToken.
        """
        return encode_scalar(self.token_rdse, float(token_id))

    def decode_sdr_sequence(self, sdr_sequence: List[np.ndarray]) -> str:
        """
        Decodes a sequence of SDRs back into a text string.
        This is a heuristic approach: for each SDR, find the closest token ID
        by calculating overlap with all possible token SDRs.
        """
        decoded_token_ids = []
        # --- FIX: Use the pre-calculated vocab_sdr_map ---
        for sdr in sdr_sequence:
            max_overlap = -1
            best_token_id = self.vocab.unk_token_id # Default to unknown token

            for token_id, token_sdr in self.vocab_sdr_map.items():
                current_overlap = overlap(sdr, token_sdr)
                if current_overlap > max_overlap:
                    max_overlap = current_overlap
                    best_token_id = token_id
            decoded_token_ids.append(best_token_id)
            
        return self.vocab.decode(decoded_token_ids)
    
    def get_sdr_size(self) -> int:
        """Returns the size of the generated SDRs."""
        return self.sdr_size

    def get_sdr_active_bits(self) -> int:
        """Returns the number of active bits in the generated SDRs."""
        return self.sdr_active_bits