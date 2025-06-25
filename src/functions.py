# src/functions.py
import torch

# This code is adapted from the nupic.torch library to be self-contained.

@torch.jit.script
def boost_activations(x, duty_cycles, boost_strength: float):
    """
    Boosts activations based on their duty cycles to encourage participation.
    """
    if boost_strength > 0.0:
        return x.detach() * torch.exp(-boost_strength * duty_cycles)
    else:
        return x.detach()


@torch.jit.script
def kwinners(x, duty_cycles, k: int, boost_strength: float):
    """
    A simple K-winner take all function for creating layers with sparse output.
    It uses boosting to determine the winning units, then sets all others to zero.
    """
    if k == 0:
        return torch.zeros_like(x)

    boosted_x = boost_activations(x, duty_cycles, boost_strength)

    # Find the threshold value that separates the top k units.
    # We find the (n-k+1)th value, which is the k-th largest value.
    # Using `kthvalue` is generally faster than a full sort or topk.
    threshold = boosted_x.kthvalue(x.shape[1] - k + 1, dim=1, keepdim=True)[0]

    # Create a mask for all values less than the threshold.
    off_mask = boosted_x < threshold

    # Apply the mask to the original tensor.
    return x.masked_fill(off_mask, 0)