# src/spatial_pooler.py
import torch
import torch.nn as nn
import functions as F # Use our own functions module

class SpatialPooler(nn.Module):
    """
    The Spatial Pooler (SP) creates a sparse distributed representation (SDR)
    from its input.

    This is a self-contained implementation of the KWinners layer from nupic.torch.
    It uses a k-winners-take-all operation, which includes built-in duty cycle
    and boosting mechanisms for robust learning.
    """
    def __init__(self,
                 input_dims,
                 sparsity=0.02,
                 boost_strength=1.0,
                 duty_cycle_period=1000):
        """
        Args:
            input_dims (int): The number of features from the encoder.
            sparsity (float): The target sparsity of the output SDR (e.g., 0.02 for 2%).
            boost_strength (float): Strength of the boosting mechanism. 0.0 means no boosting.
            duty_cycle_period (int): The period over which to average duty cycles.
        """
        super().__init__()
        self.n = input_dims
        self.k = int(round(self.n * sparsity))
        self.boost_strength = boost_strength
        self.duty_cycle_period = duty_cycle_period
        
        # We use register_buffer for states that should be part of the module
        # but are not model parameters to be trained by an optimizer.
        self.register_buffer("duty_cycle", torch.zeros(self.n))
        self.register_buffer("learning_iterations", torch.tensor(0, dtype=torch.int64))

    def forward(self, x):
        """
        Applies the K-Winner function to the input tensor.

        Args:
            x (torch.Tensor): The dense input from an encoder of shape [batch_size, input_dims].
        Returns:
            torch.Tensor: The sparse output SDR.
        """
        if self.training:
            # The core k-winners function call from our own `functions.py`
            x = F.kwinners(x, self.duty_cycle, self.k, self.boost_strength)
            self.update_duty_cycle(x)
        else:
            # During inference, boosting is still applied, but duty cycles aren't updated.
            # You could optionally use a different 'k' for inference if desired.
            x = F.kwinners(x, self.duty_cycle, self.k, self.boost_strength)

        return x

    def update_duty_cycle(self, x):
        """
        Updates our duty cycle estimates with the new value.
        """
        batch_size = x.shape[0]
        self.learning_iterations += batch_size
        period = min(float(self.duty_cycle_period), self.learning_iterations.float())
        
        # Calculate new duty cycle using a moving average
        self.duty_cycle.mul_(period - batch_size)
        self.duty_cycle.add_(x.gt(0).sum(dim=0, dtype=torch.float))
        self.duty_cycle.div_(period)

    def extra_repr(self):
        return (f"n={self.n}, k={self.k}, boost_strength={self.boost_strength}, "
                f"duty_cycle_period={self.duty_cycle_period}")