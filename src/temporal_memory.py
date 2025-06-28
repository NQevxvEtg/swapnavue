import torch
import torch.nn as nn
import threading

class TemporalMemory(nn.Module):
    """
    A Temporal Memory (TM) capable of lifelong consolidated learning.

    This module learns sequences of Sparse Distributed Representations (SDRs) and
    makes predictions about future inputs. It implements a two-phase plasticity
    model for robust long-term memory.
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

        self.register_buffer("active_cells", torch.zeros(self.num_cells, dtype=torch.bool))
        self.register_buffer("predictive_cells", torch.zeros(self.num_cells, dtype=torch.bool))
        self.register_buffer("winner_cells", torch.zeros(self.num_cells, dtype=torch.bool)) # Keep for internal logic if needed
        self.register_buffer("prev_active_cells", torch.zeros(self.num_cells, dtype=torch.bool)) # To store active cells from previous step

        self.distal_connections = nn.Parameter(
            torch.randint(0, self.num_cells,
                          (self.num_cells, self.distal_segments_per_cell, self.synapses_per_segment)),
            requires_grad=False)
        self.volatile_permanences = nn.Parameter(
            torch.rand_like(self.distal_connections, dtype=torch.float) * 0.1, requires_grad=False)
        self.consolidated_permanences = nn.Parameter(
            torch.zeros_like(self.distal_connections, dtype=torch.float), requires_grad=False)
        
        # Add this lock to coordinate the main thread and consolidation thread
        self.lock = threading.Lock()

        print("TemporalMemory: Fully initialized.")

    def forward(self, x, modulation_signal=1.0):
        # Acquire lock to ensure state is not modified by consolidation during forward pass
        with self.lock:
            # Use self.prev_active_cells from the end of the last forward pass
            prev_active_cells = self.prev_active_cells.clone() 
            
            num_predicted_columns = 0
            num_active_columns = 0

            # Initialize temporary tensors for the current step's active and next step's predictive cells
            new_active_cells = torch.zeros_like(self.active_cells, dtype=torch.bool, device=self.device)
            # Initialize for global predictive state for the *next* step
            temp_predictive_cells_for_next_step = torch.zeros_like(self.predictive_cells, dtype=torch.bool, device=self.device)
            
            # Process active columns (SDRs) from the input x
            # x here is active_columns (from SP output)
            for col_idx in torch.nonzero(x.squeeze(), as_tuple=True)[0]:
                num_active_columns += 1 # This column is active in the current input

                col_cells_indices = self.get_column_cell_indices(col_idx)

                # Compute predictive state for THIS column based on PREVIOUS active cells
                col_predictive_cells_mask_current_step = self._compute_predictive_state_for_column(col_idx, prev_active_cells)
                
                winner_cells_mask_for_column = torch.zeros(self.cells_per_column, dtype=torch.bool, device=self.device)
                # Define burst_cells_mask as all cells in the current column
                burst_cells_mask = torch.ones(self.cells_per_column, dtype=torch.bool, device=self.device)

                if col_predictive_cells_mask_current_step.any():
                    # If there are predicted cells for the current column, activate only those
                    winner_cells_mask_for_column = col_predictive_cells_mask_current_step
                    num_predicted_columns += 1 # Increment because we successfully predicted for this column
                else:
                    # If no predictions for this column, burst. Select a best matching cell if possible for learning.
                    segment_overlaps = self._get_matching_segments(col_cells_indices, prev_active_cells)
                    if segment_overlaps is not None and segment_overlaps.numel() > 0:
                        # Find the best matching cell within this column for bursting
                        best_flat_idx_in_column = torch.argmax(segment_overlaps)
                        # Map this to the local cell index within the column
                        best_cell_local_idx = best_flat_idx_in_column // self.distal_segments_per_cell
                        winner_cells_mask_for_column[best_cell_local_idx] = True # Only this cell is the "winner" for learning
                    else:
                        # If no segments, simply activate the first cell (or random) for learning, or choose another policy
                        winner_cells_mask_for_column[0] = True # Fallback: activate first cell for learning in burst

                    # For *activation* (what goes into new_active_cells), if bursting, all cells in column become active
                    new_active_cells[col_cells_indices] = burst_cells_mask
                
                # Apply learning based on this column's winner cells
                # FIX: Pass the specific winner_cells_mask_for_column to _learn
                self._learn(prev_active_cells, modulation_signal, new_active_cells, col_cells_indices, winner_cells_mask_for_column) 
                
            # Update the global active_cells state for the CURRENT time step
            self.active_cells = new_active_cells.clone()

            # Now, compute the global predictive state for the NEXT step based on the *current* active cells
            # FIX: Iterate over all columns to compute the full predictive_cells state
            for col_idx_global in range(self.columns):
                col_predictive_mask_for_next_step = self._compute_predictive_state_for_column(col_idx_global, self.active_cells)
                global_col_indices = self.get_column_cell_indices(col_idx_global)
                temp_predictive_cells_for_next_step[global_col_indices] = col_predictive_mask_for_next_step
            
            # Update the global predictive_cells state for the NEXT time step
            self.predictive_cells = temp_predictive_cells_for_next_step.clone()
            self.prev_active_cells = self.active_cells.clone() # Store current active cells for next iteration's `prev_active_cells`

            # Calculate and return current batch accuracy (proportion of predicted columns)
            current_batch_accuracy = num_predicted_columns / num_active_columns if num_active_columns > 0 else 1.0
            
            # Return active_cells and predictive_cells of the current state, and the accuracy
            return self.active_cells, self.predictive_cells, current_batch_accuracy

    def _get_matching_segments(self, cell_indices, prev_active_cells):
        if not cell_indices.numel() or not prev_active_cells.any():
            return None
        # Clamp distal connections to ensure indices are within valid bounds (defensive programming)
        col_distal_connections_clamped = torch.clamp(self.distal_connections[cell_indices], 0, self.num_cells - 1)
        presynaptic_activity = prev_active_cells[col_distal_connections_clamped]
        # Sum of active presynaptic cells for each segment
        return presynaptic_activity.sum(dim=-1)

    def _compute_predictive_state_for_column(self, col_idx: int, input_active_cells: torch.Tensor) -> torch.Tensor:
        """
        Computes the predictive state for a specific column based on provided active cells (e.g., from previous step).
        """
        col_cells_indices = self.get_column_cell_indices(col_idx)

        if not input_active_cells.any():
            return torch.zeros(self.cells_per_column, dtype=torch.bool, device=self.device)

        # Get relevant distal connections and permanences for cells in this column
        col_distal_connections = self.distal_connections[col_cells_indices]
        col_volatile_perms = self.volatile_permanences[col_cells_indices]
        col_consolidated_perms = self.consolidated_permanences[col_cells_indices]

        volatile_connected = col_volatile_perms > self.permanence_threshold
        consolidated_connected = col_consolidated_perms > self.permanence_threshold
        combined_connected = volatile_connected | consolidated_connected

        col_distal_connections_clamped = torch.clamp(col_distal_connections, 0, self.num_cells - 1)
        presynaptic_was_active = input_active_cells[col_distal_connections_clamped]
        
        active_synapses_on_segment = (combined_connected & presynaptic_was_active).sum(dim=-1)
        
        predictive_segments_mask = active_synapses_on_segment >= self.activation_threshold
        predictive_cells_mask = predictive_segments_mask.any(dim=-1)
        
        return predictive_cells_mask


    # FIX: Modified _learn signature to accept specific winner_cells for learning
    def _learn(self, prev_active_cells, modulation_signal, current_active_cells_global, col_cells_indices, winner_cells_for_column):
        if modulation_signal == 0.0 or not prev_active_cells.any():
            return

        # Get the global indices of cells chosen as winners for learning within this column
        learning_cells_global_indices = col_cells_indices[winner_cells_for_column]
        
        if not learning_cells_global_indices.numel():
            return

        # For each learning cell, select the segment to strengthen.
        # This part assumes a best-matching segment approach for learning.
        segment_overlaps_for_learning_cells = self._get_matching_segments(learning_cells_global_indices, prev_active_cells)
        
        if segment_overlaps_for_learning_cells is None: 
            return

        # For each learning cell, select the segment with the most overlap.
        best_segment_indices_for_learning = torch.argmax(segment_overlaps_for_learning_cells, dim=-1)

        # Iterate over the learning cells and their best segments
        for i, cell_idx in enumerate(learning_cells_global_indices):
            best_seg_idx = best_segment_indices_for_learning[i]
            
            # Get the global indices of synapses on this specific best segment
            synapse_global_indices = self.distal_connections[cell_idx, best_seg_idx]

            # Get the current permanences for these specific synapses
            permanences_to_update = self.volatile_permanences[cell_idx, best_seg_idx]
            
            # Determine which of these presynaptic cells were active in the previous step
            presynaptic_was_active_mask = prev_active_cells[synapse_global_indices]

            potentiation = self.volatile_learning_rate * modulation_signal
            punishment = potentiation * 0.1

            # Apply potentiation to active synapses
            permanences_to_update[presynaptic_was_active_mask] += potentiation
            # Apply punishment to inactive synapses
            permanences_to_update[~presynaptic_was_active_mask] -= punishment
            
            # Clamp permanences to [0, 1] range
            permanences_to_update = torch.clamp(permanences_to_update, 0, 1)

            # Update the volatile_permanences tensor in place
            self.volatile_permanences[cell_idx, best_seg_idx] = permanences_to_update


    def consolidate(self, memory_trace):
        # Acquire lock to ensure state is not modified by the forward pass during consolidation
        with self.lock:
            print(f"Consolidating a memory trace of length {len(memory_trace)}...")
            
            # Temporarily store active and predictive cells for consolidation process
            temp_active_cells_consolidation = torch.zeros_like(self.active_cells)
            temp_predictive_cells_consolidation = torch.zeros_like(self.predictive_cells)

            for sdr in memory_trace:
                prev_active_cells_trace = temp_active_cells_consolidation.clone()
                
                # Activate cells for this SDR in the consolidation context
                # This should ideally call a method that updates temp_active_cells_consolidation
                # based on prediction or bursting, similar to how forward works.
                # For simplicity, if sdr represents active columns, and it's a replay:
                
                new_active_cells_replay = torch.zeros_like(self.active_cells)
                # Compute predictive state for each column in the replayed SDR
                for col_idx in torch.nonzero(sdr.squeeze(), as_tuple=True)[0]:
                    col_cells_indices = self.get_column_cell_indices(col_idx)
                    col_predictive_cells_mask = self._compute_predictive_state_for_column(col_idx, prev_active_cells_trace)
                    
                    if col_predictive_cells_mask.any():
                        new_active_cells_replay[col_cells_indices] = col_predictive_cells_mask
                        # If a prediction, the winning cells for learning are these predictive cells
                        winner_cells_for_consolidated_learn = col_predictive_cells_mask
                    else:
                        new_active_cells_replay[col_cells_indices] = torch.ones(self.cells_per_column, dtype=torch.bool, device=self.device) # Burst
                        # If bursting, select a single best-matching cell for learning
                        segment_overlaps = self._get_matching_segments(col_cells_indices, prev_active_cells_trace)
                        if segment_overlaps is not None and segment_overlaps.numel() > 0:
                            best_flat_idx_in_column = torch.argmax(segment_overlaps)
                            best_cell_local_idx = best_flat_idx_in_column // self.distal_segments_per_cell
                            winner_cells_for_consolidated_learn = torch.zeros(self.cells_per_column, dtype=torch.bool, device=self.device)
                            winner_cells_for_consolidated_learn[best_cell_local_idx] = True
                        else:
                            winner_cells_for_consolidated_learn = torch.zeros(self.cells_per_column, dtype=torch.bool, device=self.device)
                            winner_cells_for_consolidated_learn[0] = True # Fallback

                    # Apply consolidated learning for this column
                    self._consolidated_learn(prev_active_cells_trace, col_cells_indices, winner_cells_for_consolidated_learn)

                temp_active_cells_consolidation = new_active_cells_replay.clone()
                # Re-compute predictive state for next step of consolidation replay
                for col_idx_global in range(self.columns):
                    col_predictive_mask_next_replay = self._compute_predictive_state_for_column(col_idx_global, temp_active_cells_consolidation)
                    global_col_indices = self.get_column_cell_indices(col_idx_global)
                    temp_predictive_cells_consolidation[global_col_indices] = col_predictive_mask_next_replay

            # Reset the main TM states after consolidation is complete
            self.active_cells.zero_()
            self.predictive_cells.zero_()
            self.prev_active_cells.zero_() # Also reset prev_active_cells
            self.winner_cells.zero_() # This can effectively be removed if not directly used outside _learn

            print("Consolidation complete.")
    
    # FIX: Renamed from _activate_cells to a more specific name for internal use
    # and fixed its logic to reflect the new forward processing.
    # This function is implicitly replaced by the direct logic in `forward` and `consolidate`.

    # FIX: Modified _consolidated_learn signature
    def _consolidated_learn(self, prev_active_cells, col_cells_indices, winner_cells_for_column):
        learning_cells_global_indices = col_cells_indices[winner_cells_for_column]
        if not learning_cells_global_indices.numel(): return

        segment_overlaps = self._get_matching_segments(learning_cells_global_indices, prev_active_cells)
        if segment_overlaps is None: return

        best_segment_indices = torch.argmax(segment_overlaps, dim=-1)

        for i, cell_idx in enumerate(learning_cells_global_indices):
            best_seg_idx = best_segment_indices[i]
            segment_permanences = self.consolidated_permanences[cell_idx, best_seg_idx]
            synapse_global_indices = self.distal_connections[cell_idx, best_seg_idx]
            presynaptic_was_active = prev_active_cells[synapse_global_indices]
            
            segment_permanences[presynaptic_was_active] += self.consolidated_learning_rate
            segment_permanences[~presynaptic_was_active] -= self.consolidated_learning_rate * 0.25
            self.consolidated_permanences[cell_idx, best_seg_idx] = torch.clamp(segment_permanences, 0, 1)

    def get_column_cell_indices(self, col_idx: int) -> torch.Tensor:
        """
        Returns the global indices of all cells belonging to a given column.
        """
        col_cells_start = col_idx * self.cells_per_column
        col_cells_end = col_cells_start + self.cells_per_column
        return torch.arange(col_cells_start, col_cells_end, device=self.device)