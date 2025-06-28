import torch
import torch.nn as nn
from encoders import SensorimotorEncoder
from spatial_pooler import SpatialPooler
from temporal_memory import TemporalMemory

class HTMNetwork(nn.Module):
    """
    The complete HTM Network, assembling the Encoder, Spatial Pooler,
    and Temporal Memory into a single system.
    """
    def __init__(self, input_dims, encoder_dims, sp_dims, tm_params):
        super().__init__()
        self.encoder = SensorimotorEncoder(input_dims, encoder_dims)
        self.spatial_pooler = SpatialPooler(encoder_dims, sparsity=0.02)
        self.temporal_memory = TemporalMemory(
            input_dims=sp_dims,
            **tm_params
        )

    def forward(self, x, modulation_signal=0.0):
        """
        A full forward pass through the network. Handles batched input.
        """
        # Encoder and Spatial Pooler are batch-compatible.
        encoded_data = self.encoder(x)
        sdr_output_batch = self.spatial_pooler(encoded_data)

        # --- FIX: Process SDRs sequentially through the stateful Temporal Memory ---
        # The TM processes one SDR at a time to update its state.
        # We loop through the batch dimension.
        batch_predictions = []
        for sdr_output in sdr_output_batch:
            predictions = self.temporal_memory(sdr_output, modulation_signal)
            batch_predictions.append(predictions)

        # Stack predictions for each item in the batch into a single tensor.
        return torch.stack(batch_predictions)