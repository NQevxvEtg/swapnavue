# src/resonator.py

# The Schumann Resonance fundamental frequency, which acts as the constant,
# foundational "field" or "hum" for the entire system.
#
SCHUMANN_RESONANCE_HZ = 7.83


import torch
import torch.nn as nn
import math

class CognitiveResonator(nn.Module):
    """
    Manages the state and generation of a single cognitive rhythm (a wave).
    This component is responsible for smoothly interpolating its frequency
    and amplitude towards target values, creating a graceful "glide" effect.
    """
    def __init__(self, base_frequency: float, base_amplitude: float, smoothing_factor: float = 0.1):
        """
        Initializes the Cognitive resonator.

        Args:
            base_frequency (float): The default frequency (in Hz) of the wave.
            base_amplitude (float): The default amplitude of the wave.
            smoothing_factor (float): How quickly the wave glides to new targets. 
                                      Smaller values mean a smoother, slower glide.
        """
        super().__init__()
        
        # --- Configuration ---
        self.smoothing_factor = smoothing_factor

        # --- State Buffers ---
        # We register these as non-trainable buffers so they are part of the model's
        # state_dict and persist across sessions.
        self.register_buffer('time_step', torch.tensor(0.0, dtype=torch.float32))
        
        self.register_buffer('current_frequency', torch.tensor(base_frequency, dtype=torch.float32))
        self.register_buffer('target_frequency', torch.tensor(base_frequency, dtype=torch.float32))
        
        self.register_buffer('current_amplitude', torch.tensor(base_amplitude, dtype=torch.float32))
        self.register_buffer('target_amplitude', torch.tensor(base_amplitude, dtype=torch.float32))


    def _lerp(self, current: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Linearly interpolates between current and target values for a smooth transition.
        """
        return current + self.smoothing_factor * (target - current)

    def set_target(self, new_target_frequency: float, new_target_amplitude: float):
        """
        Sets a new target for the resonator's frequency and amplitude.
        The resonator will then smoothly glide towards these new targets on each tick.
        Called by the Heart.
        """
        self.target_frequency.fill_(new_target_frequency)
        self.target_amplitude.fill_(new_target_amplitude)

    def tick(self) -> float:
        """
        Advances the resonator by one step.
        1. Smoothly glide the current frequency/amplitude towards their targets.
        2. Advance the time step.
        3. Return the new value of the wave.
        """
        # 1. Glide frequency and amplitude towards their targets
        self.current_frequency = self._lerp(self.current_frequency, self.target_frequency)
        self.current_amplitude = self._lerp(self.current_amplitude, self.target_amplitude)

        # 2. Advance the time step
        self.time_step += 1.0
        
        # 3. Calculate the sine wave value using the *current* (smoothed) parameters
        # Formula: y = amplitude * sin(2 * pi * f * t)
        # We don't use a time_scale_factor here because the frequency is now dynamic.
        value = self.current_amplitude * math.sin(
            2 * math.pi * self.current_frequency * self.time_step.item() * 0.01 # Use a small multiplier to make Hz values practical
        )
        
        return value

    def forward(self) -> float:
        """Allows calling the resonator like a regular torch module."""
        return self.tick()