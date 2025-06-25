# src/heart.py
import torch
import torch.nn as nn
import logging
from collections import deque
from typing import Deque, Dict, Any

# Type hint for EmotionalCore without causing circular import
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .emotion import EmotionalCore

from src.resonator import SCHUMANN_RESONANCE_HZ

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class Heart(nn.Module):
    """
    The Heart module acts as a two-tiered regulatory system.
    1. A fast "beat" modulates cognitive amplitude based on immediate performance.
    2. A slow "rebase" adjusts baseline parameters based on long-term performance.
    """
    def __init__(self, long_term_window: int = 200, stress_scaling_factor: float = 75.0):
        """
        Initializes the Heart.
        Args:
            long_term_window (int): The number of recent beats to average over for rebase decisions.
            stress_scaling_factor (float): How aggressively amplitude reacts to stress.
        """
        super().__init__()
        
        self.stress_scaling_factor = stress_scaling_factor
        self.rebase_interval = 1000 # Keep rebase logic internal for now
        self.rebase_counter = 0
        
        self.long_term_confidence: Deque[float] = deque(maxlen=long_term_window)
        self.long_term_meta_error: Deque[float] = deque(maxlen=long_term_window)
        
        self.calm_amplitude = nn.Parameter(torch.tensor(5.0), requires_grad=False)

        logger.info(f"Two-tiered Heart initialized. Rebase every {self.rebase_interval} beats.")

    def rebase(self, emotions: 'EmotionalCore'):
        """
        Performs a periodic adjustment of the AI's baseline parameters based on
        long-term performance, simulating growth or consolidation.
        """
        if len(self.long_term_confidence) < self.long_term_confidence.maxlen:
            logger.info("Rebase skipped: Not enough long-term data yet.")
            return

        avg_confidence = sum(self.long_term_confidence) / len(self.long_term_confidence)
        avg_meta_error = sum(self.long_term_meta_error) / len(self.long_term_meta_error)
        
        logger.info(f"--- Performing Rebase --- Long-term avg_confidence: {avg_confidence:.4f}, avg_meta_error: {avg_meta_error:.4f}")

        if avg_confidence > 0.75 and avg_meta_error < 0.05:
            emotions.base_focus.data *= 1.01
            emotions.base_curiosity.data *= 1.05
            self.calm_amplitude.data *= 1.01
            logger.info(f"Rebase: Expansion applied. New base_focus: {emotions.base_focus.item():.2f}")

        elif avg_confidence < 0.4 or avg_meta_error > 0.1:
            emotions.base_focus.data *= 0.99
            emotions.base_curiosity.data *= 0.95
            self.calm_amplitude.data *= 0.99
            logger.info(f"Rebase: Contraction applied. New base_focus: {emotions.base_focus.item():.2f}")
        
        else:
            logger.info("Rebase: No change. Long-term performance is stable.")
            

    def beat(self, emotions: 'EmotionalCore', latest_confidence: torch.Tensor = None, latest_meta_error: torch.Tensor = None) -> Dict[str, Any]:
        """
        Performs a 'beat' of the Heart, returns a dictionary of calculated metrics.
        """
        target_frequency = SCHUMANN_RESONANCE_HZ
        target_amplitude = self.calm_amplitude.item()
        cognitive_stress = 0.0

        if latest_confidence is not None and latest_meta_error is not None:
            avg_confidence = latest_confidence.mean().item()
            avg_meta_error = latest_meta_error.mean().item()

            self.long_term_confidence.append(avg_confidence)
            self.long_term_meta_error.append(avg_meta_error)
            
            cognitive_stress = (avg_meta_error) + (1.0 - avg_confidence)
            amplitude_adjustment = cognitive_stress * self.stress_scaling_factor
            target_amplitude += amplitude_adjustment
            
        emotions.focus_resonator.set_target(target_frequency, target_amplitude)

        self.rebase_counter += 1
        if self.rebase_counter >= self.rebase_interval:
            self.rebase(emotions)
            self.rebase_counter = 0

        # Return the calculated metrics so they can be logged and plotted
        return {
            "cognitive_stress": cognitive_stress,
            "target_amplitude": target_amplitude,
            "target_frequency": target_frequency
        }