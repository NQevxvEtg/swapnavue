# src/emotion.py
import torch
import torch.nn as nn
import logging

from src.resonator import CognitiveResonator, SCHUMANN_RESONANCE_HZ

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class EmotionalCore(nn.Module):
    """
    The EmotionalCore now holds its core baseline parameters as learnable/adjustable
    Parameters, allowing the Heart's rebase mechanism to modify them over time.
    """
    def __init__(self):
        super().__init__()

        # --- Base Values for Emotional Parameters ---
        # These are now nn.Parameters so they can be adjusted by the Heart's
        # rebase mechanism and their state is saved with the model.
        self.base_focus = nn.Parameter(torch.tensor([50.0], dtype=torch.float32), requires_grad=False)
        self.base_curiosity = nn.Parameter(torch.tensor([0.001], dtype=torch.float32), requires_grad=False)

        # --- Cognitive Resonators ---
        # The resonators use the initial values of the base parameters.
        self.focus_resonator = CognitiveResonator(
            base_frequency=SCHUMANN_RESONANCE_HZ,
            base_amplitude=5.0,
            smoothing_factor=0.05
        )

        self.curiosity_resonator = CognitiveResonator(
            base_frequency=0.02, 
            base_amplitude=0.0005,
            smoothing_factor=0.02
        )

        # --- Active Emotional Parameters ---
        # These are the final, live values that the rest of the model will use.
        self.focus = nn.Parameter(torch.tensor([self.base_focus.item()], dtype=torch.float32), requires_grad=True)
        self.curiosity = nn.Parameter(torch.tensor([self.base_curiosity.item()], dtype=torch.float32), requires_grad=True)

        # --- Fixed Cognitive Parameters ---
        self.discipline = nn.Parameter(torch.tensor([1.0], dtype=torch.float32), requires_grad=False)
        self.contentment_threshold = nn.Parameter(torch.tensor([0.05], dtype=torch.float32), requires_grad=False)
        
        # --- Model Architecture Parameters ---
        self.model_dims = nn.Parameter(torch.tensor([512.], dtype=torch.float32), requires_grad=False)
        self.knowledge_dims = nn.Parameter(torch.tensor([512.], dtype=torch.float32), requires_grad=False)
        self.time_embedding_dim = nn.Parameter(torch.tensor([128.], dtype=torch.float32), requires_grad=False)
        self.max_seq_len = nn.Parameter(torch.tensor([256.], dtype=torch.float32), requires_grad=False)

        logger.info(f"EmotionalCore initialized with adjustable baseline parameters.")

    def update(self):
        """
        This is the main "tick" for the entire emotional system. It's called once
        per reasoning cycle to advance the rhythms and update the final emotion values.
        """
        focus_wave_value = self.focus_resonator.tick()
        curiosity_wave_value = self.curiosity_resonator.tick()

        # The final emotion value is its baseline + the current value from its rhythmic wave.
        new_focus = self.base_focus.item() + focus_wave_value
        new_curiosity = self.base_curiosity.item() + curiosity_wave_value
        
        # Apply the new values to the parameters.
        self.focus.data.fill_(new_focus)
        self.curiosity.data.fill_(new_curiosity)
        
        logger.debug(f"EmotionalCore updated: Focus={self.focus.item():.2f}, Curiosity={self.curiosity.item():.6f}")


    def get_focus(self) -> int:
        """Returns the current reasoning focus as an integer."""
        return max(1, int(self.focus.item()))

    def get_curiosity(self) -> float:
        """Returns the current curiosity (learning rate) as a float."""
        return self.curiosity.item()
    
    def get_discipline(self) -> float:
        """Returns the current discipline (gradient clipping value) as a float."""
        return self.discipline.item()

    def __repr__(self):
        return (f"EmotionalCore(base_focus={self.base_focus.item():.2f}, base_curiosity={self.base_curiosity.item():.6f}, "
                f"focus={self.focus.item():.2f}, curiosity={self.curiosity.item():.6f})")