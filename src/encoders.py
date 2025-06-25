
import torch.nn as nn

class SensorimotorEncoder(nn.Module):
    """
    Encodes raw sensorimotor data into a high-dimensional dense representation.
    This dense representation will then be sparsified by the Spatial Pooler.

    This is a placeholder and can be replaced with more sophisticated
    encoders, such as those for vision, audio, or grid-cell-like location.
    """
    def __init__(self, input_dims, output_dims):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dims, output_dims),
            nn.ReLU()
        )

    def forward(self, x):
        return self.encoder(x)