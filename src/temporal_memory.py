# src/temporal_memory.py

import torch
import torch.nn as nn
import threading

class TemporalMemory(nn.Module):
    """
    A Temporal Memory (TM) capable of lifelong consolidated learning.

    This module learns sequences of Sparse Distributed Representations (SDRs) and
    makes predictions about future inputs. It implements a two-phase plasticity
    Adel model for robust long-term memory.
    """
    def __init__(self,
                 input_dims,
                 columns,
                 cells_per_column,
                 distal_segments_per_cell=32,
                 synapses_per_segment=128,
                 permanence_threshold=0.5,
                 connected_permanence=0.8,
                 volatile_learning_rate=0.1,
                 consolidated_learning_rate=0.01,
                 activation_threshold=10,
                 device=None):

        super().__init__()
        self.input_dims = input_dims
        self.columns = columns
        self.cells_per_column = cells_per_column
        self.num_cells = columns * cells_per_column
        self.distal_segments_per_cell = distal_segments_per_cell
        self.synapses_per_segment = synapses_per_segment
        self.permanence_threshold = permanence_threshold
        self.connected_permanence = connected_permanence
        self.volatile_learning_rate = volatile_learning_rate
        self.consolidated_learning_rate = consolidated_learning_rate
        self.activation_threshold = activation_threshold
        self.device = device # Store the device passed to the constructor

        # Buffers for state
        self.register_buffer("active_cells", torch.zeros(self.num_cells, dtype=torch.bool, device=self.device))
        self.register_buffer("predictive_cells", torch.zeros(self.num_cells, dtype=torch.bool, device=self.device))
        self.register_buffer("winner_cells", torch.zeros(self.num_cells, dtype=torch.bool, device=self.device))

        # Parameters for distal connections and permanences
        # distal_connections maps cell_idx, segment_idx, synapse_idx to a global cell index
        self.distal_connections = nn.Parameter(
            torch.randint(0, self.num_cells,
                          (self.num_cells, self.distal_segments_per_cell, self.synapses_per_segment),
                          device=self.device),
            requires_grad=False)
        
        self.volatile_permanences = nn.Parameter(
            torch.rand_like(self.distal_connections, dtype=torch.float, device=self.device) * 0.1, requires_grad=False)
        self.consolidated_permanences = nn.Parameter(
            torch.zeros_like(self.distal_connections, dtype=torch.float, device=self.device), requires_grad=False)
        
        # Add this lock to coordinate the main thread and consolidation thread
        self.lock = threading.Lock()

        print("TemporalMemory: Fully initialized.")

    def forward(self, x: torch.Tensor, modulation_signal: float = 1.0) -> tuple[torch.Tensor, torch.Tensor, float]:
        """
        Processes an SDR input (active columns from SP) and updates TM state.

        Args:
            x (torch.Tensor): A 1D boolean/int tensor representing the active columns
                              from the Spatial Pooler (size: self.columns).
            modulation_signal (float): A scalar signal from EmotionalCore for learning modulation.

        Returns:
            tuple[torch.Tensor, torch.Tensor, float]:
                - active_cells (torch.Tensor): The currently active cells after processing x.
                - predictive_cells (torch.Tensor): The cells predicted to be active at the next time step.
                - current_batch_accuracy (float): Proportion of columns successfully predicted.
        """
        # Acquire lock to ensure state is not modified by consolidation during forward pass
        with self.lock:
            prev_active_cells = self.active_cells.clone() # Cells active from the *previous* time step
            
            # Initialize for current batch accuracy
            num_predicted_columns = 0
            num_active_columns = 0

            # new_active_cells will store the cells activated by the current input 'x'
            new_active_cells = torch.zeros_like(self.active_cells, dtype=torch.bool, device=self.device)
            
            # Find which columns are active in the input SDR 'x'
            # x is a 1D tensor where 1s indicate active columns
            active_column_indices_in_input = torch.nonzero(x.squeeze(), as_tuple=True)[0]

            if active_column_indices_in_input.numel() == 0:
                # If no columns are active in the input, just return current state
                # No new active cells, no predictions based on this input for next step
                # Accuracy is 1.0 because we made no incorrect predictions
                return self.active_cells, self.predictive_cells, 1.0

            # Process each active column in the input
            for col_idx in active_column_indices_in_input:
                num_active_columns += 1 # This column is active in the current input

                col_cells_indices = self.get_column_cell_indices(col_idx)

                # Step 1: Compute predictive state for THIS column based on PREVIOUS active cells
                # This checks if this column was correctly predicted by the TM from the previous step
                col_predictive_cells_mask_current_step = self._compute_predictive_state(col_idx, prev_active_cells)
                
                winner_cells_mask_for_column = torch.zeros(self.cells_per_column, dtype=torch.bool, device=self.device)
                
                if col_predictive_cells_mask_current_step.any():
                    # If there are predicted cells for the current column, activate only those
                    winner_cells_mask_for_column = col_predictive_cells_mask_current_step
                    num_predicted_columns += 1 # Increment because we successfully predicted for this column
                else:
                    # If no predictions for this column, activate all cells in the column (burst)
                    # And then select a "winner" cell for learning (if applicable)
                    winner_cells_mask_for_column = torch.ones(self.cells_per_column, dtype=torch.bool, device=self.device)
                    # Additional logic to select a single winner cell from bursting:
                    # This could be based on overlap with previous active cells, or simply random
                    # For simplicity, here we'll assume the _learn function can handle bursting
                    # cells and find the best segment, or we might need to adjust winner_cells_mask_for_column
                    # to be a single cell if the learning rule strictly expects one.
                    # Given the original code, it seems _learn uses self.winner_cells, which is the global winner set.

                # Update the new_active_cells tensor with the activations for this column
                new_active_cells[col_cells_indices] = winner_cells_mask_for_column
            
            # Set global winner_cells based on the batch's active columns for learning
            self.winner_cells = new_active_cells.clone() # These are the cells that were active for the current input

            # Step 2: Learn based on the current activations and previous active cells
            # This adjusts permanences for segments on the 'winner_cells' based on 'prev_active_cells'.
            self._learn(prev_active_cells, modulation_signal)
            
            # Step 3: Compute predictive state for the NEXT step based on CURRENTLY active cells (new_active_cells)
            # This loop populates new_predictive_cells_for_next_step by computing predictions for all columns
            # that were activated in this current step.
            new_predictive_cells_for_next_step = torch.zeros_like(self.predictive_cells, dtype=torch.bool, device=self.device)
            # Iterate over all columns that were active in the current input (new_active_cells)
            for col_idx_for_next_pred in active_column_indices_in_input:
                predictive_mask = self._compute_predictive_state(col_idx_for_next_pred, new_active_cells)
                col_cells_indices = self.get_column_cell_indices(col_idx_for_next_pred)
                new_predictive_cells_for_next_step[col_cells_indices] = predictive_mask

            # Update the internal state of the TM after processing all columns in the batch
            self.active_cells = new_active_cells.clone()
            self.predictive_cells = new_predictive_cells_for_next_step.clone() # This is the prediction for the *next* time step
            # self.prev_active_cells is not needed, active_cells handles it implicitly in the next forward pass

            # Calculate and return current batch accuracy (proportion of predicted columns)
            current_batch_accuracy = num_predicted_columns / num_active_columns if num_active_columns > 0 else 1.0
            
            return self.active_cells, self.predictive_cells, current_batch_accuracy

    def _get_matching_segments(self, cell_indices, prev_active_cells):
        if not cell_indices.numel() or not prev_active_cells.any():
            return None
        # Ensure prev_active_cells is on the same device as distal_connections
        presynaptic_activity = prev_active_cells[self.distal_connections[cell_indices]]
        return presynaptic_activity.sum(dim=-1)

    def _activate_cells(self, x, prev_active_cells):
        # This function seems to be part of the original TM logic that's being replaced
        # by the more granular processing in `forward`.
        # Its presence might indicate old or unused logic.
        # For now, I'll keep it as is, but it's not directly called by the main forward pass anymore.
        active_columns_mask = x > 0
        active_cells_mask = torch.zeros_like(self.active_cells)
        winner_cells_mask = torch.zeros_like(self.winner_cells)

        for col_idx in torch.where(active_columns_mask)[0]:
            col_cells_start = col_idx * self.cells_per_column
            col_cells_end = col_cells_start + self.cells_per_column
            col_cells_indices = torch.arange(col_cells_start, col_cells_end, device=x.device)
            col_predictive_cells_mask = self.predictive_cells[col_cells_start:col_cells_end]

            if torch.any(col_predictive_cells_mask):
                active_cells_mask[col_cells_indices] = col_predictive_cells_mask
                winner_cells_mask[col_cells_indices] = col_predictive_cells_mask
            else:
                active_cells_mask[col_cells_indices] = True
                segment_overlaps = self._get_matching_segments(col_cells_indices, prev_active_cells)
                if segment_overlaps is not None and segment_overlaps.numel() > 0:
                    best_flat_idx = torch.argmax(segment_overlaps)
                    winner_cell_local_idx = best_flat_idx // self.distal_segments_per_cell
                    winner_cell_global_idx = col_cells_start + winner_cell_local_idx
                    winner_cells_mask[winner_cell_global_idx] = True

        self.active_cells = active_cells_mask
        self.winner_cells = winner_cells_mask

    def _compute_predictive_state(self, col_idx: int, input_active_cells: torch.Tensor) -> torch.Tensor:
        """
        Computes the predictive state for a specific column based on provided active cells.
        This function is now designed to be called for a single column `col_idx`.
        """
        # Get global indices of cells within the current column
        col_cells_indices = self.get_column_cell_indices(col_idx)

        # Handle case where input_active_cells is entirely false (no active cells anywhere)
        if not input_active_cells.any():
            return torch.zeros(self.cells_per_column, dtype=torch.bool, device=self.device)

        # Get relevant distal connections and permanences for cells in this column
        # shape: (cells_per_column, max_segments_per_cell, max_synapses_per_segment)
        col_distal_connections = self.distal_connections[col_cells_indices]
        col_volatile_perms = self.volatile_permanences[col_cells_indices]
        col_consolidated_perms = self.consolidated_permanences[col_cells_indices]

        # Connected synapses: those above permanence_threshold
        volatile_connected = col_volatile_perms > self.permanence_threshold
        consolidated_connected = col_consolidated_perms > self.permanence_threshold
        combined_connected = volatile_connected | consolidated_connected

        # Presynaptic activity: which *input_active_cells* were active for these synapses
        # input_active_cells is the equivalent of prev_active_cells or new_active_cells from forward
        # This indexing creates a boolean mask of active presynaptic cells for each synapse on each segment.
        # Clamp distal connections to ensure indices are within valid bounds (defensive programming)
        col_distal_connections_clamped = torch.clamp(col_distal_connections, 0, self.num_cells - 1)
        presynaptic_was_active = input_active_cells[col_distal_connections_clamped]
        
        # An active segment has enough connected *and* active synapses
        # Sum of (connected & active) synapses along the last dimension (synapses_per_segment)
        active_synapses_on_segment = (combined_connected & presynaptic_was_active).sum(dim=-1)
        
        # A cell is predictive if any of its segments have enough active synapses to cross the threshold
        predictive_segments_mask = active_synapses_on_segment >= self.activation_threshold
        predictive_cells_mask = predictive_segments_mask.any(dim=-1) # True if any segment predicts for that cell
        
        return predictive_cells_mask

    def _learn(self, prev_active_cells: torch.Tensor, modulation_signal: float):
        """
        Performs learning based on the current winner cells and previous active cells.
        Updates volatile permanences.
        """
        if modulation_signal == 0.0 or not prev_active_cells.any():
            return

        # Get global indices of cells that are "winners" in the current step
        learning_cells_indices_global = torch.where(self.winner_cells)[0]
        if not learning_cells_indices_global.numel():
            return

        # Get relevant segment overlaps for all learning cells
        # This will return (num_learning_cells, max_segments_per_cell)
        segment_overlaps_for_learning_cells = self._get_matching_segments(learning_cells_indices_global, prev_active_cells)
        
        if segment_overlaps_for_learning_cells is None: 
            return # No segments to learn on

        # For each learning cell, select the best segment to strengthen.
        # This gives a 1D tensor of segment indices, one for each learning cell.
        best_segment_indices_for_learning = torch.argmax(segment_overlaps_for_learning_cells, dim=-1)

        # Use the global indices of learning cells and their best segment indices to select
        # the specific synapses to update within the `distal_connections` tensor.
        # This is a bit tricky with advanced indexing. We need the indices of the synapses
        # for each (cell_idx, best_segment_idx) pair.
        
        # Create a meshgrid to get all combinations of (cell_idx, best_segment_idx)
        # and then index into distal_connections and volatile_permanences
        
        # Method 1: Using gather (more direct if target indices are uniform)
        # This requires `best_segment_indices_for_learning` to be expanded for each synapse
        # Not ideal for non-uniform structures.

        # Method 2: Manual indexing (clearer for non-uniform structure)
        # Create lists of indices to flatten the operation
        all_selected_cell_indices = []
        all_selected_segment_indices = []
        for i, cell_global_idx in enumerate(learning_cells_indices_global):
            seg_local_idx = best_segment_indices_for_learning[i].item()
            # Add the global cell index and the selected segment index for each synapse on that segment
            all_selected_cell_indices.extend([cell_global_idx] * self.synapses_per_segment)
            all_selected_segment_indices.extend([seg_local_idx] * self.synapses_per_segment)

        # Convert to tensors for advanced indexing
        selected_cell_indices_tensor = torch.tensor(all_selected_cell_indices, device=self.device)
        selected_segment_indices_tensor = torch.tensor(all_selected_segment_indices, device=self.device)

        # Get the global indices of the presynaptic cells connected to these selected synapses
        synapses_to_update_global_indices = self.distal_connections[
            selected_cell_indices_tensor, selected_segment_indices_tensor, torch.arange(self.synapses_per_segment, device=self.device).repeat(len(learning_cells_indices_global))
        ]
        
        # Get the current permanences for these specific synapses.
        permanences_to_update_flat = self.volatile_permanences[
            selected_cell_indices_tensor, selected_segment_indices_tensor, torch.arange(self.synapses_per_segment, device=self.device).repeat(len(learning_cells_indices_global))
        ]
        
        # Determine which of these presynaptic cells were active
        presynaptic_was_active_mask_flat = prev_active_cells[synapses_to_update_global_indices]

        potentiation = self.volatile_learning_rate * modulation_signal
        punishment = potentiation * 0.1 # N.B.: original was 0.1, making it dynamic now

        # Apply potentiation to active synapses
        permanences_to_update_flat[presynaptic_was_active_mask_flat] += potentiation
        # Apply punishment to inactive synapses
        permanences_to_update_flat[~presynaptic_was_active_mask_flat] -= punishment
        
        # Clamp permanences to [0, 1] range
        permanences_to_update_flat = torch.clamp(permanences_to_update_flat, 0, 1)

        # Update the volatile_permanences tensor in place.
        self.volatile_permanences[
            selected_cell_indices_tensor, selected_segment_indices_tensor, torch.arange(self.synapses_per_segment, device=self.device).repeat(len(learning_cells_indices_global))
        ] = permanences_to_update_flat.clone() # Use .clone() to ensure no aliasing issues

    def consolidate(self, memory_trace: list[torch.Tensor]):
        """
        Consolidates a memory trace (sequence of SDRs) into long-term memory.
        """
        # Acquire lock to ensure state is not modified by the forward pass during consolidation
        with self.lock:
            print(f"Consolidating a memory trace of length {len(memory_trace)}...")
            temp_active_cells_for_consolidation = torch.zeros_like(self.active_cells)
            temp_predictive_cells_for_consolidation = torch.zeros_like(self.predictive_cells)

            for sdr in memory_trace:
                # sdr is the active columns output from spatial pooler
                # Need to convert this sdr (columns) into active cells for the current step of consolidation
                
                # Simulate `_activate_cells` logic specific to consolidation
                current_active_cells_for_consolidation = torch.zeros_like(self.active_cells, dtype=torch.bool, device=self.device)
                current_winner_cells_for_consolidation = torch.zeros_like(self.winner_cells, dtype=torch.bool, device=self.device)

                active_column_indices_in_sdr = torch.nonzero(sdr.squeeze(), as_tuple=True)[0]
                
                for col_idx in active_column_indices_in_sdr:
                    col_cells_indices = self.get_column_cell_indices(col_idx)
                    col_predictive_cells_mask = self._compute_predictive_state(col_idx, temp_active_cells_for_consolidation)

                    if torch.any(col_predictive_cells_mask):
                        current_active_cells_for_consolidation[col_cells_indices] = col_predictive_cells_mask
                        current_winner_cells_for_consolidation[col_cells_indices] = col_predictive_cells_mask
                    else:
                        current_active_cells_for_consolidation[col_cells_indices] = True
                        # For consolidation, if bursting, we just activate all cells in the column for now
                        # More advanced winner selection can be added later if needed for consolidation
                        current_winner_cells_for_consolidation[col_cells_indices] = torch.ones(self.cells_per_column, dtype=torch.bool, device=self.device)

                # Perform consolidated learning
                self._consolidated_learn(temp_active_cells_for_consolidation, current_winner_cells_for_consolidation)
                
                # Update temporary active cells for the next step of the trace
                temp_active_cells_for_consolidation = current_active_cells_for_consolidation.clone()
                # Update predictive cells for the next consolidation step
                # This assumes we are predicting based on the *currently active* cells for the *next* SDR in the trace
                for col_idx in active_column_indices_in_sdr: # Or iterate over all columns, depending on desired behavior
                    predictive_mask = self._compute_predictive_state(col_idx, temp_active_cells_for_consolidation)
                    col_cells_indices = self.get_column_cell_indices(col_idx)
                    temp_predictive_cells_for_consolidation[col_cells_indices] = predictive_mask
            
            # Reset the main TM active/predictive state after consolidation
            # This ensures the TM starts fresh for regular forward passes after a consolidation run.
            self.active_cells.zero_()
            self.predictive_cells.zero_()
            self.winner_cells.zero_()
            print("Consolidation complete.")
    
    # Removed _activate_cells_for_consolidation as its logic is now inline in consolidate

    def _consolidated_learn(self, prev_active_cells: torch.Tensor, winner_cells: torch.Tensor):
        """
        Performs learning using the consolidated learning rate.
        """
        learning_cells_indices = torch.where(winner_cells)[0]
        if not learning_cells_indices.numel(): return

        segment_overlaps = self._get_matching_segments(learning_cells_indices, prev_active_cells)
        if segment_overlaps is None: return
        
        # Select best segment for each learning cell based on overlap
        best_segment_indices = torch.argmax(segment_overlaps, dim=-1)
        
        # Prepare for advanced indexing
        all_selected_cell_indices = []
        all_selected_segment_indices = []
        for i, cell_idx_global in enumerate(learning_cells_indices):
            seg_local_idx = best_segment_indices[i].item()
            all_selected_cell_indices.extend([cell_idx_global] * self.synapses_per_segment)
            all_selected_segment_indices.extend([seg_local_idx] * self.synapses_per_segment)
        
        selected_cell_indices_tensor = torch.tensor(all_selected_cell_indices, device=self.device)
        selected_segment_indices_tensor = torch.tensor(all_selected_segment_indices, device=self.device)

        # Get global indices of presynaptic cells for the selected synapses
        synapses_to_update_global_indices = self.distal_connections[
            selected_cell_indices_tensor, selected_segment_indices_tensor, torch.arange(self.synapses_per_segment, device=self.device).repeat(len(learning_cells_indices))
        ]

        # Get current permanences for these synapses from consolidated_permanences
        permanences_to_update_flat = self.consolidated_permanences[
            selected_cell_indices_tensor, selected_segment_indices_tensor, torch.arange(self.synapses_per_segment, device=self.device).repeat(len(learning_cells_indices))
        ]

        # Determine which presynaptic cells were active
        presynaptic_was_active_mask_flat = prev_active_cells[synapses_to_update_global_indices]

        # Apply learning rule
        potentiation = self.consolidated_learning_rate
        punishment = potentiation * 0.25 # Can be different from volatile

        permanences_to_update_flat[presynaptic_was_active_mask_flat] += potentiation
        permanences_to_update_flat[~presynaptic_was_active_mask_flat] -= punishment
        
        # Clamp permanences
        permanences_to_update_flat = torch.clamp(permanences_to_update_flat, 0, 1)

        # Update the consolidated_permanences tensor in place
        self.consolidated_permanences[
            selected_cell_indices_tensor, selected_segment_indices_tensor, torch.arange(self.synapses_per_segment, device=self.device).repeat(len(learning_cells_indices))
        ] = permanences_to_update_flat.clone()


    def get_column_cell_indices(self, col_idx: int) -> torch.Tensor:
        """
        Returns the global indices of all cells belonging to a given column.
        """
        col_cells_start = col_idx * self.cells_per_column
        col_cells_end = col_cells_start + self.cells_per_column
        return torch.arange(col_cells_start, col_cells_end, device=self.device)
