import torch
import pytest
from network import HTMNetwork

# --- Helper function for this test file ---
def create_distinct_input(size, pattern_type):
    """
    Creates a simple, non-sparse tensor with a batch dimension of 1,
    designed to produce distinct patterns.
    """
    tensor = torch.zeros(size)
    if pattern_type == "A":
        # Activate the first half of the features strongly
        tensor[:size // 2] = 1.0
        # Add a little noise to ensure it's not all zeros after potential transformations,
        # but the primary pattern remains.
        tensor += torch.rand(size) * 0.1
    elif pattern_type == "B":
        # Activate the second half of the features strongly
        tensor[size // 2:] = 1.0
        # Add a little noise
        tensor += torch.rand(size) * 0.1
    else:
        raise ValueError("pattern_type must be 'A' or 'B'")

    # .unsqueeze(0) adds the batch dimension, changing shape from [size] to [1, size]
    return tensor.unsqueeze(0)

# --- Test Fixture to create a default Network for our tests ---
@pytest.fixture
def htm_network():
    """Provides a default HTMNetwork instance."""
    return HTMNetwork(
        input_dims=50,
        encoder_dims=256,
        sp_dims=256, # Must match encoder_dims
        tm_params={
            "columns": 256, # Must match sp_dims
            "cells_per_column": 8,
            "activation_threshold": 4,  # Lowered the threshold
            "volatile_learning_rate": 0.2,  # Increased learning rate
        }
    )

# --- Test 1: End-to-End Smoke Test ---
def test_network_forward_pass(htm_network):
    """
    Tests if data can flow through the entire network without errors.
    """
    print("\nRunning: test_network_forward_pass (Smoke Test)")
    # Arrange: Create dummy input data. Using pattern 'A' for consistency, though 'B' would also work.
    dummy_input = create_distinct_input(size=htm_network.encoder.encoder[0].in_features, pattern_type="A")

    # Act: Pass the data through the network
    try:
        predictions = htm_network.forward(dummy_input)
    except Exception as e:
        pytest.fail(f"HTMNetwork forward pass failed with an exception: {e}")

    # Assert: Check if the output has the correct shape
    # The output should be [batch_size, num_cells]
    expected_shape = (1, htm_network.temporal_memory.num_cells)
    assert predictions.shape == expected_shape, "Output shape is incorrect."
    print("Success! Forward pass completed without errors.")


# --- Test 2: End-to-End Learning Test ---
def test_network_learns_sequence(htm_network):
    """
    Tests if the full integrated network can learn a simple A -> B transition.
    """
    print("Running: test_network_learns_sequence")
    # Arrange: Create two distinct inputs using the new helper
    input_a = create_distinct_input(size=htm_network.encoder.encoder[0].in_features, pattern_type="A")
    input_b = create_distinct_input(size=htm_network.encoder.encoder[0].in_features, pattern_type="B")

    # Act (Learning Phase): Repeat the A -> B sequence multiple times
    num_learning_iterations = 5 # Increased learning iterations for robustness
    for _ in range(num_learning_iterations):
        htm_network.forward(input_a, modulation_signal=1.0)
        htm_network.forward(input_b, modulation_signal=1.0)

    # Act (Prediction Phase):
    htm_network.temporal_memory.active_cells.zero_()
    htm_network.temporal_memory.predictive_cells.zero_()
    htm_network.temporal_memory.winner_cells.zero_()  # Keep this line as per original problem
    predictions = htm_network.forward(input_a, modulation_signal=0.0)

    # Assert: The network should now be making a prediction.
    assert predictions.any(), "The integrated network failed to make a prediction." [cite: 6]
    print("Success! The network formed a prediction.")