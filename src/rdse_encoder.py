# src/rdse_encoder.py

import random
import math
import numpy as np

class RDSEInstance:
    """
    A class to hold the RDSE parameters, mirroring the C++ struct.
    """
    def __init__(self, size: int, active_bits: int, resolution: float, prototypes: list[float]):
        self.size = size
        self.active_bits = active_bits
        self.resolution = resolution
        self.prototypes = prototypes

def create_rdse(n: int, w: int, resolution: float, seed: int = -1) -> RDSEInstance:
    """
    Creates a Random Distributed Scalar Encoder (RDSE) instance,
    mirroring the C++ create_rdse function.
    """
    if seed != -1:
        random.seed(seed)
    else:
        # For true randomness, could use os.urandom or similar,
        # but for direct porting, Python's default random is sufficient.
        pass # random module initializes with system time by default

    prototypes = [random.uniform(0.0, resolution) for _ in range(n)]
    
    return RDSEInstance(n, w, resolution, prototypes)

def encode_scalar(rdse_instance: RDSEInstance, value: float) -> np.ndarray:
    """
    Encodes a scalar value into a Sparse Distributed Representation (SDR),
    mirroring the C++ encode_scalar function.
    """
    if rdse_instance.active_bits > rdse_instance.size:
        # In Python, we might raise an error or log a warning
        raise ValueError("active_bits cannot be greater than size")
    
    distances = []
    for prototype in rdse_instance.prototypes:
        distances.append(abs(prototype - value))
    
    # Get indices of the smallest distances
    # np.argsort returns the indices that would sort an array
    # We take the first `active_bits` indices
    sorted_indices = np.argsort(distances)[:rdse_instance.active_bits]
    
    sdr = np.zeros(rdse_instance.size, dtype=np.int8)
    for idx in sorted_indices:
        sdr[idx] = 1
        
    return sdr

def overlap(sdr1: np.ndarray, sdr2: np.ndarray) -> int:
    """
    Computes the overlap (dot product) between two binary SDRs,
    mirroring the C++ overlap function.
    """
    if sdr1.shape != sdr2.shape:
        raise ValueError("SDRs must have the same size to compute overlap")
        
    # Element-wise multiplication and then sum
    return int(np.sum(sdr1 * sdr2))