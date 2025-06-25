# src/model_parts.py
import torch
import torch.nn as nn
import math
import logging

# Import the centralized initialize_weights from utils
from src.utils import initialize_weights 

# Get logger for this module
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO) # Set to INFO for less verbose output during normal runs


class CombineContext(nn.Module):
    def __init__(self, current_input_dims: int, fast_state_dims: int, slow_state_dims: int, context_dims: int):
        super().__init__()
        self.total_input_dims = current_input_dims + fast_state_dims + slow_state_dims
        self.linear = nn.Linear(self.total_input_dims, context_dims)
        self.relu = nn.ReLU() # Added ReLU for non-linearity
        initialize_weights(self) # Apply weight initialization from utils
        logger.debug(f"Initialized CombineContext with total_input_dims={self.total_input_dims}, context_dims={context_dims}")

    def forward(self, current_input: torch.Tensor, fast_state: torch.Tensor, slow_state: torch.Tensor) -> torch.Tensor:
        logger.debug(f"CombineContext input shapes - current_input: {current_input.shape}, fast_state: {fast_state.shape}, slow_state: {slow_state.shape}")
        
        if current_input.dim() == 1:
            current_input = current_input.unsqueeze(0)
        if fast_state.dim() == 1:
            fast_state = fast_state.unsqueeze(0)
        if slow_state.dim() == 1:
            slow_state = slow_state.unsqueeze(0)

        combined_tensor = torch.cat([current_input, fast_state, slow_state], dim=-1)
        logger.debug(f"CombineContext concatenated shape: {combined_tensor.shape}")
        context = self.relu(self.linear(combined_tensor))
        logger.debug(f"CombineContext output shape (context): {context.shape}")
        return context


class DenoiseNet(nn.Module):
    def __init__(self, model_dims: int, context_dims: int, time_embedding_dim: int = 64):
        super().__init__()
        self.model_dims = model_dims
        self.time_embedding_dim = time_embedding_dim

        # Sinusoidal time embedding
        self.time_embed_linear1 = nn.Linear(time_embedding_dim, time_embedding_dim * 4)
        self.time_embed_linear2 = nn.Linear(time_embedding_dim * 4, time_embedding_dim)
        
        # Input layer: z + time_embedding + context
        self.input_dim = model_dims + time_embedding_dim + context_dims
        
        self.norm_input = nn.LayerNorm(self.input_dim) # LayerNorm after concatenation

        self.fc1 = nn.Linear(self.input_dim, model_dims * 4)
        self.norm1 = nn.LayerNorm(model_dims * 4) # LayerNorm after fc1
        self.fc2 = nn.Linear(model_dims * 4, model_dims * 2)
        self.norm2 = nn.LayerNorm(model_dims * 2) # LayerNorm after fc2
        self.fc3 = nn.Linear(model_dims * 2, model_dims)

        initialize_weights(self) # Apply weight initialization from utils
        logger.debug(f"Initialized DenoiseNet with model_dims={model_dims}, context_dims={context_dims}, time_embedding_dim={time_embedding_dim}")

    def _time_embedding(self, tau: torch.Tensor) -> torch.Tensor:
        if tau.dim() == 0:
            tau = tau.unsqueeze(0)
        elif tau.dim() == 1 and tau.shape[0] > 1:
            pass
        elif tau.dim() == 1 and tau.shape[0] == 1:
            tau = tau.expand(1) 
        
        tau_expanded = tau.float() * 1000

        div_term = torch.exp(torch.arange(0, self.time_embedding_dim, 2).float() * (-math.log(10000.0) / self.time_embedding_dim)).to(tau.device)
        
        pe = torch.zeros(tau.shape[0], self.time_embedding_dim).to(tau.device)
        pe[:, 0::2] = torch.sin(tau_expanded.unsqueeze(1) * div_term)
        pe[:, 1::2] = torch.cos(tau_expanded.unsqueeze(1) * div_term)
        
        time_emb = torch.relu(self.time_embed_linear1(pe))
        time_emb = self.time_embed_linear2(time_emb)
        
        logger.debug(f"Time embedding shape for tau={tau.shape}: {time_emb.shape}")
        return time_emb


    def forward(self, z: torch.Tensor, tau: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        logger.debug(f"DenoiseNet forward input shapes - z: {z.shape}, tau: {tau.shape}, context: {context.shape}")

        time_emb = self._time_embedding(tau)

        if z.shape[0] != time_emb.shape[0]:
             time_emb = time_emb.expand(z.shape[0], -1)

        x = torch.cat([z, context, time_emb], dim=-1)
        logger.debug(f"DenoiseNet input concatenated shape: {x.shape}")

        x = self.norm_input(x)

        x = self.fc1(x)
        x = torch.relu(x)
        x = self.norm1(x)

        x = self.fc2(x)
        x = torch.relu(x)
        x = self.norm2(x)

        output = self.fc3(x)
        logger.debug(f"DenoiseNet output shape: {output.shape}")
        return output

class ProjectHead(nn.Module):
    def __init__(self, model_dims: int, output_dims: int):
        super().__init__()
        self.linear = nn.Linear(model_dims, output_dims)
        initialize_weights(self) # Apply weight initialization from utils
        logger.debug(f"Initialized ProjectHead with model_dims={model_dims}, output_dims={output_dims}")

    def forward(self, reasoned_state_z0: torch.Tensor) -> torch.Tensor:
        predicted_output = self.linear(reasoned_state_z0)
        logger.debug(f"ProjectHead output shape: {predicted_output.shape}")
        return predicted_output

class UpdateFastState(nn.Module):
    def __init__(self, fast_state_dims: int, current_input_dims: int, reasoned_state_z0_dims: int):
        super().__init__()
        self.input_dims = fast_state_dims + current_input_dims + reasoned_state_z0_dims
        self.linear = nn.Linear(self.input_dims, fast_state_dims)
        self.relu = nn.ReLU() # Added ReLU
        initialize_weights(self) # Apply weight initialization from utils
        logger.debug(f"Initialized UpdateFastState with input_dims={self.input_dims}, output_dims={fast_state_dims}")

    def forward(self, fast_state: torch.Tensor, current_input: torch.Tensor, reasoned_state_z0: torch.Tensor) -> torch.Tensor:
        logger.debug(f"UpdateFastState input shapes - fast_state: {fast_state.shape}, current_input: {current_input.shape}, reasoned_state_z0: {reasoned_state_z0.shape}")
        
        if current_input.dim() == 1:
            current_input = current_input.unsqueeze(0)
        if fast_state.dim() == 1:
            fast_state = fast_state.unsqueeze(0)
        if reasoned_state_z0.dim() == 1:
            reasoned_state_z0 = reasoned_state_z0.unsqueeze(0)

        combined_input = torch.cat([fast_state, current_input, reasoned_state_z0], dim=-1)
        logger.debug(f"UpdateFastState concatenated input shape: {combined_input.shape}")
        
        new_fast_state = self.relu(self.linear(combined_input))
        logger.debug(f"New fast state shape: {new_fast_state.shape}")
        return new_fast_state