import torch
import torch.nn as nn

class SelfReflectionModule(nn.Module):
    """
    A module designed to provide the Continuously Reasoning Predictor (CRP) with
    self-reflection and meta-cognition capabilities. It processes the model's
    internal states (reasoned state, fast state, slow state) and potentially
    emotional states to generate feedback signals such as confidence and a meta-error.
    """
    def __init__(self, model_dims: int, reflection_dims: int):
        """
        Initializes the SelfReflectionModule.

        Args:
            model_dims (int): The primary dimension of the model's internal states
                              (e.g., model_dims from EmotionalCore).
            reflection_dims (int): The hidden dimension for the self-reflection processor.
        """
        super().__init__()

        # Define a processor network that takes concatenated internal states
        # The input dimension is derived from:
        # model_dims (for reasoned_state/z_0)
        # model_dims (for fast_state/S)
        # model_dims (for slow_state/K - assuming K is also model_dims)
        # We might also incorporate a representation of emotions here if needed,
        # but for simplicity, we'll focus on core state variables first.
        input_processor_dim = model_dims * 3 # z_0, S, K
        
        self.processor = nn.Sequential(
            nn.Linear(input_processor_dim, reflection_dims),
            nn.LayerNorm(reflection_dims), # Normalization helps stabilize training
            nn.ReLU(),
            nn.Linear(reflection_dims, reflection_dims // 2),
            nn.ReLU(),
            nn.Linear(reflection_dims // 2, 2) # Output: confidence score, meta-error signal
        )

        # Optional: A learnable parameter to represent inherent meta-knowledge or bias
        # This parameter would be updated during the self-reflection learning process.
        self.meta_knowledge_bias = nn.Parameter(torch.randn(reflection_dims // 2))


    def forward(self, reasoned_state: torch.Tensor, fast_state: torch.Tensor, slow_state: torch.Tensor, emotions_context: torch.Tensor = None):
        """
        Performs the self-reflection process.

        Args:
            reasoned_state (torch.Tensor): The output of the _reason_to_predict method (z_0).
                                           Shape: (batch_size, model_dims)
            fast_state (torch.Tensor): The current fast state (working memory).
                                       Shape: (batch_size, model_dims)
            slow_state (torch.Tensor): The current slow state (consolidated knowledge).
                                       Shape: (1, model_dims) - needs to be expanded for batch
            emotions_context (torch.Tensor, optional): A tensor representing relevant
                                                        emotional states (e.g., current focus, curiosity).
                                                        Can be concatenated into input_processor_dim if desired.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - confidence (torch.Tensor): A score (0-1) indicating the model's confidence in its reasoning.
                - meta_error (torch.Tensor): A signal (e.g., -1 to 1) indicating perceived internal inconsistency or error.
        """
        batch_size = reasoned_state.shape[0]

        # Expand slow_state to match the batch_size for concatenation
        expanded_slow_state = slow_state.squeeze(0).repeat(batch_size, 1)

        # Concatenate relevant internal states for processing by the reflection network
        # If emotions_context is provided and its dimension is consistent, it can be added here
        reflection_input = torch.cat([reasoned_state, fast_state, expanded_slow_state], dim=-1)

        # Pass through the processor network
        raw_reflection_output = self.processor(reflection_input)

        # Apply activation functions to get interpretable signals
        # Confidence: Sigmoid to squash between 0 and 1
        confidence = torch.sigmoid(raw_reflection_output[:, 0])
        # Meta-error: Tanh to squash between -1 and 1
        meta_error = torch.tanh(raw_reflection_output[:, 1])

        return confidence, meta_error