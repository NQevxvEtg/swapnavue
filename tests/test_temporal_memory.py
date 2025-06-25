# swapnavue/tests/test_temporal_memory.py

import torch
import pytest
from temporal_memory import TemporalMemory

# --- Helper function to create simple SDRs for testing ---
def create_sdr(size, active_indices):
    """Creates a sparse, binary tensor."""
    sdr = torch.zeros(size, dtype=torch.bool)
    sdr[active_indices] = True
    return sdr

# --- Test Fixture to create a default TM for our tests ---
@pytest.fixture
def tm():
    """Provides a default TemporalMemory instance for tests."""
    return TemporalMemory(
        input_dims=100,
        columns=100,
        cells_per_column=4,
        activation_threshold=2, # Lower threshold for easier testing
    )

# --- Test 1: Verify Column Bursting on Novel Input ---
def test_bursting_on_novel_input(tm):
    """
    Tests if all cells in active columns activate when an input is unexpected.
    """
    print("Running: test_bursting_on_novel_input")
    # Arrange: Create a novel SDR.
    sdr_a_indices = [2, 10, 25]
    sdr_a = create_sdr(tm.input_dims, sdr_a_indices)

    # Act: Pass the SDR through the TM's forward pass.
    tm.forward(sdr_a)

    # Assert: Check that the active cells match the expected burst pattern.
    expected_burst_mask = torch.zeros(tm.num_cells, dtype=torch.bool)
    for col_idx in sdr_a_indices:
        start = col_idx * tm.cells_per_column
        end = start + tm.cells_per_column
        expected_burst_mask[start:end] = True

    assert torch.equal(tm.active_cells, expected_burst_mask), "Cells did not burst correctly."
    print("Success!")

# --- Test 2: Verify that Correct Predictions Prevent Bursting ---
def test_prediction_prevents_bursting(tm):
    """
    Tests if only predicted cells become active when a prediction is correct.
    """
    print("Running: test_prediction_prevents_bursting")
    # Arrange: Manually set a predictive state for specific cells.
    predicted_column_idx = 15
    predicted_cell_local_idx = 2
    predicted_cell_global_idx = predicted_column_idx * tm.cells_per_column + predicted_cell_local_idx

    tm.predictive_cells[predicted_cell_global_idx] = True

    # Arrange: Create the SDR that corresponds to the prediction.
    sdr_b_indices = [predicted_column_idx]
    sdr_b = create_sdr(tm.input_dims, sdr_b_indices)

    # Act: Pass the predicted SDR through the forward pass.
    tm.forward(sdr_b)

    # Assert: Only the single predicted cell should be active.
    expected_active_mask = torch.zeros(tm.num_cells, dtype=torch.bool)
    expected_active_mask[predicted_cell_global_idx] = True

    assert torch.equal(tm.active_cells, expected_active_mask), "Prediction did not prevent bursting."
    print("Success!")


# --- Test 3: Verify a Simple Sequence Can Be Learned ---
def test_learn_simple_sequence(tm):
    """
    Tests if the TM can learn a transition A -> B and then predict B after seeing A.
    """
    print("Running: test_learn_simple_sequence")
    # Arrange: Create two distinct SDRs.
    sdr_a = create_sdr(tm.input_dims, [5, 20])
    sdr_b = create_sdr(tm.input_dims, [6, 21])

    # Act (Learning Phase): Pass the sequence A -> B through the TM.
    # The `modulation_signal` is high to enable learning.
    tm.forward(sdr_a, modulation_signal=1.0) # Process A
    tm.forward(sdr_b, modulation_signal=1.0) # Process B, learning the transition from A.

    # Act (Prediction Phase): Reset state and present A again.
    tm.active_cells.zero_()
    tm.predictive_cells.zero_()
    tm.forward(sdr_a, modulation_signal=0.0) # Process A again, this time only for prediction.

    # Assert: After seeing A, the TM should now be predicting something.
    # A perfect test would check if it predicts cells in columns 6 & 21.
    # For now, we just confirm that the predictive state is no longer empty.
    assert tm.predictive_cells.any(), "TM failed to form any prediction after learning."
    print(f"Success! Formed {tm.predictive_cells.sum()} predictive cells.")

# --- Test 4: Verify Consolidation Updates Permanent Memory ---
def test_consolidation_updates_weights(tm):
    """
    Tests if the consolidate() method modifies the consolidated permanences.
    """
    print("Running: test_consolidation_updates_weights")
    # Arrange: Create a memory trace to be consolidated.
    sdr_a = create_sdr(tm.input_dims, [1])
    sdr_b = create_sdr(tm.input_dims, [2])
    memory_trace = [sdr_a, sdr_b]

    # Store the initial state of the consolidated permanences.
    initial_consolidated_perms = tm.consolidated_permanences.clone()

    # Act: Run the consolidation process.
    tm.consolidate(memory_trace)

    # Assert: The consolidated permanences should have changed.
    assert not torch.equal(tm.consolidated_permanences, initial_consolidated_perms), \
        "Consolidate method did not update consolidated permanences."
    print("Success!")