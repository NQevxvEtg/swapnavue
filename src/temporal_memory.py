# src/temporal_memory.py

import torch
import torch.nn as nn
import threading
import logging 

logger = logging.getLogger(__name__) 

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
        self.input_dims = input_dims # This seems unused, as input is `columns`
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

        # Parameters for internal states - dynamically sized in forward.
        # Initialize with batch_size 1. Will be re-instantiated if batch_size changes.
        self.active_cells = nn.Parameter(torch.zeros(1, self.num_cells, dtype=torch.bool, device=self.device), requires_grad=False)
        self.predictive_cells = nn.Parameter(torch.zeros(1, self.num_cells, dtype=torch.bool, device=self.device), requires_grad=False)
        self.winner_cells = nn.Parameter(torch.zeros(1, self.num_cells, dtype=torch.bool, device=self.device), requires_grad=False)


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

        logger.info("TemporalMemory: Fully initialized.") # Changed print to logger.info

    def forward(self, 
                sdr_batch: torch.Tensor, # Changed from x to sdr_batch. Expected shape: [batch_size, num_columns]
                modulation_signal_batch: torch.Tensor = None # Changed from modulation_signal to modulation_signal_batch. Expected shape: [batch_size]
               ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: # Changed return type for accuracy
        """
        Processes a batch of SDR inputs (active columns from SP) and updates TM state.

        Args:
            sdr_batch (torch.Tensor): A batch of 1D boolean/int tensors representing the active columns
                                      from the Spatial Pooler. Shape: [batch_size, num_columns].
            modulation_signal_batch (torch.Tensor): A batch of scalar signals from EmotionalCore for learning modulation.
                                                Shape: [batch_size].

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - active_cells_this_step (torch.Tensor): The currently active cells after processing x. Shape: [batch_size, num_cells].
                - predictive_cells_for_next_step (torch.Tensor): The cells predicted to be active at the next time step. Shape: [batch_size, num_cells].
                - current_batch_accuracy (torch.Tensor): Proportion of columns successfully predicted for each item in batch. Shape: [batch_size].
        """
        batch_size = sdr_batch.shape[0]

        # Dynamically resize internal state parameters' data if batch_size changes
        if self.active_cells.shape[0] != batch_size:
            logger.debug(f"Resizing TM state parameters from {self.active_cells.shape[0]} to {batch_size} for batching.")

            # Re-initialize the nn.Parameter objects with the new shape when batch_size changes.
            self.active_cells = nn.Parameter(torch.zeros(batch_size, self.num_cells, dtype=torch.bool, device=self.device), requires_grad=False)
            self.predictive_cells = nn.Parameter(torch.zeros(batch_size, self.num_cells, dtype=torch.bool, device=self.device), requires_grad=False)
            self.winner_cells = nn.Parameter(torch.zeros(batch_size, self.num_cells, dtype=torch.bool, device=self.device), requires_grad=False)
            
        if modulation_signal_batch is None:
            modulation_signal_batch = torch.ones(batch_size, device=self.device) # Default to 1.0 if not provided

        # Acquire lock to ensure state is not modified by consolidation during forward pass
        with self.lock:
            prev_active_cells = self.active_cells.clone() # Cells active from the *previous* time step (shape: [batch_size, num_cells])
            
            # Initialize for current batch accuracy
            # These need to be batched
            num_predicted_columns = torch.zeros(batch_size, dtype=torch.int, device=self.device)
            num_active_columns = torch.zeros(batch_size, dtype=torch.int, device=self.device)

            # new_active_cells will store the cells activated by the current input 'sdr_batch'
            new_active_cells = torch.zeros_like(self.active_cells, dtype=torch.bool, device=self.device)
            
            # Find which columns are active in the input SDR 'sdr_batch'
            active_column_indices_in_batch = torch.nonzero(sdr_batch, as_tuple=True) # (batch_indices, column_indices)

            # Handle empty batches (though model_architecture should prevent this usually)
            if active_column_indices_in_batch[0].numel() == 0:
                return self.active_cells, self.predictive_cells, torch.ones(batch_size, device=self.device)

            # Iterate over each item in the batch
            for batch_item_idx in range(batch_size):
                active_columns_for_this_item = torch.nonzero(sdr_batch[batch_item_idx]).squeeze(1)
                
                if active_columns_for_this_item.numel() == 0:
                    continue # Skip if no active columns for this batch item

                num_active_columns[batch_item_idx] = active_columns_for_this_item.numel() # Update count for this batch item

                # Process each active column in the current batch item
                for col_idx in active_columns_for_this_item:
                    col_cells_indices = self.get_column_cell_indices(col_idx.item()) # .item() to convert tensor to int

                    # Step 1: Compute predictive state for THIS column based on PREVIOUS active cells
                    # Call _compute_predictive_state for the specific column and prev_active_cells for this batch item
                    col_predictive_cells_mask_current_step = self._compute_predictive_state(col_idx.item(), prev_active_cells[batch_item_idx])
                    
                    winner_cells_mask_for_column = torch.zeros(self.cells_per_column, dtype=torch.bool, device=self.device)
                    
                    if col_predictive_cells_mask_current_step.any():
                        # If there are predicted cells for the current column, activate only those
                        winner_cells_mask_for_column = col_predictive_cells_mask_current_step
                        num_predicted_columns[batch_item_idx] += 1 # Increment because we successfully predicted for this column
                    else:
                        # If no predictions for this column, activate all cells in the column (burst)
                        winner_cells_mask_for_column = torch.ones(self.cells_per_column, dtype=torch.bool, device=self.device)

                    # Update the new_active_cells tensor with the activations for this column and batch item
                    new_active_cells[batch_item_idx, col_cells_indices] = winner_cells_mask_for_column
            
            # Set global winner_cells based on the batch's active columns for learning
            self.winner_cells.data.copy_(new_active_cells) # FIXED: Use .data.copy_()
            
            # Step 2: Learn based on the current activations and previous active cells
            # This adjusts permanences for segments on the 'winner_cells' based on 'prev_active_cells'.
            # _learn will now process the entire batch
            self._learn(prev_active_cells, modulation_signal_batch)
            
            # Step 3: Compute predictive state for the NEXT step based on CURRENTLY active cells (new_active_cells)
            # This loop populates new_predictive_cells_for_next_step by computing predictions for all columns
            # that were activated in this current step.
            new_predictive_cells_for_next_step = torch.zeros_like(self.predictive_cells, dtype=torch.bool, device=self.device)
            
            # Iterate over each item in the batch again to compute predictions for the next step
            for batch_item_idx in range(batch_size):
                active_columns_for_this_item = torch.nonzero(sdr_batch[batch_item_idx]).squeeze(1) # Using current input's active columns

                for col_idx_for_next_pred in active_columns_for_this_item:
                    predictive_mask = self._compute_predictive_state(col_idx_for_next_pred.item(), new_active_cells[batch_item_idx])
                    col_cells_indices = self.get_column_cell_indices(col_idx_for_next_pred.item())
                    new_predictive_cells_for_next_step[batch_item_idx, col_cells_indices] = predictive_mask

            # Update the internal state of the TM after processing all columns in the batch
            self.active_cells.data.copy_(new_active_cells) # FIXED: Use .data.copy_()
            self.predictive_cells.data.copy_(new_predictive_cells_for_next_step) # FIXED: Use .data.copy_()
            
            # Calculate and return current batch accuracy (proportion of predicted columns)
            current_batch_accuracy = torch.where(
                num_active_columns > 0,
                num_predicted_columns.float() / num_active_columns.float(),
                torch.tensor(1.0, device=self.device, dtype=torch.float) # Handle division by zero
            )
            
            return self.active_cells, self.predictive_cells, current_batch_accuracy

    def _get_matching_segments(self, cell_indices: torch.Tensor, prev_active_cells: torch.Tensor):
        """
        Helper to find segments that have sufficient overlap with prev_active_cells.
        Args:
            cell_indices (torch.Tensor): 1D tensor of global cell indices.
            prev_active_cells (torch.Tensor): 1D boolean tensor of previous active cells for a SINGLE batch item.
        Returns:
            torch.Tensor: Overlap counts for each segment of the given cells. Shape: [len(cell_indices), distal_segments_per_cell]
                          Returns None if no active cells or no segments to match.
        """
        if not cell_indices.numel() or not prev_active_cells.any():
            return None
        
        # Clamp distal connections to ensure indices are within valid bounds (defensive programming)
        distal_connections_clamped = torch.clamp(self.distal_connections[cell_indices], 0, self.num_cells - 1)
        
        # Gather presynaptic activity for all synapses on all segments of given cells
        # This will be [len(cell_indices), distal_segments_per_cell, synapses_per_segment]
        presynaptic_activity = prev_active_cells[distal_connections_clamped]
        
        # Sum active presynaptic cells per segment
        # Result: [len(cell_indices), distal_segments_per_cell]
        segment_overlaps = presynaptic_activity.sum(dim=-1)
        return segment_overlaps

    def _activate_cells(self, x, prev_active_cells):
        # This function is kept as per user request, but is not directly used in the current forward pass
        # logic for batch processing. It serves as a reference to a previous activation mechanism.
        
        active_columns_mask = x > 0 # Assuming x is a single input [num_columns]
        active_cells_mask = torch.zeros_like(self.active_cells) # Shape: [1, num_cells]
        winner_cells_mask = torch.zeros_like(self.winner_cells) # Shape: [1, num_cells]

        for col_idx in torch.where(active_columns_mask)[0]:
            col_cells_start = col_idx * self.cells_per_column
            col_cells_end = col_cells_start + self.cells_per_column
            col_cells_indices = torch.arange(col_cells_start, col_cells_end, device=x.device)
            # Note: self.predictive_cells is now [batch_size, num_cells], so need to get relevant one
            # This old function assumes self.predictive_cells is [1, num_cells]
            col_predictive_cells_mask = self.predictive_cells[0, col_cells_start:col_cells_end] # Assuming batch 0 for this old function

            if torch.any(col_predictive_cells_mask):
                active_cells_mask[0, col_cells_indices] = col_predictive_cells_mask
                winner_cells_mask[0, col_cells_indices] = col_predictive_cells_mask
            else:
                active_cells_mask[0, col_cells_indices] = True
                # This call to _get_matching_segments expects 1D prev_active_cells and 1D col_cells_indices
                # Adapting this old function to work with batch-first tensors would be complex.
                # For now, it's left as is, as it's not part of the current batching pipeline.
                segment_overlaps = self._get_matching_segments(col_cells_indices, prev_active_cells.squeeze(0)) # Squeeze for 1D input
                if segment_overlaps is not None and segment_overlaps.numel() > 0:
                    best_flat_idx = torch.argmax(segment_overlaps)
                    winner_cell_local_idx = best_flat_idx // self.distal_segments_per_cell
                    winner_cell_global_idx = col_cells_start + winner_cell_local_idx
                    winner_cells_mask[0, winner_cell_global_idx] = True

        self.active_cells.data.copy_(active_cells_mask) # FIXED: Use .data.copy_()
        self.winner_cells.data.copy_(winner_cells_mask) # FIXED: Use .data.copy_()

    def _compute_predictive_state(self, col_idx: int, input_active_cells_single_item: torch.Tensor) -> torch.Tensor:
        """
        Computes the predictive state for a specific column based on provided active cells
        from a SINGLE batch item.
        
        Args:
            col_idx (int): The column index to compute predictions for.
            input_active_cells_single_item (torch.Tensor): 1D boolean tensor of active cells
                                                            from a single batch item. Shape: [num_cells].
        Returns:
            torch.Tensor: 1D boolean tensor of cells within the column that are predictive.
                          Shape: [cells_per_column].
        """
        # Get global indices of cells within the current column
        col_cells_indices = self.get_column_cell_indices(col_idx)

        # Handle case where input_active_cells_single_item is entirely false (no active cells anywhere for this item)
        if not input_active_cells_single_item.any():
            return torch.zeros(self.cells_per_column, dtype=torch.bool, device=self.device)

        # Get relevant distal connections and permanences for cells in this column
        # shape: (cells_per_column, distal_segments_per_cell, synapses_per_segment)
        col_distal_connections = self.distal_connections[col_cells_indices]
        col_volatile_perms = self.volatile_permanences[col_cells_indices]
        col_consolidated_perms = self.consolidated_permanences[col_cells_indices]

        # Connected synapses: those above permanence_threshold
        volatile_connected = col_volatile_perms > self.permanence_threshold
        consolidated_connected = col_consolidated_perms > self.permanence_threshold
        combined_connected = volatile_connected | consolidated_connected

        # Presynaptic activity: which *input_active_cells_single_item* were active for these synapses
        # This indexing creates a boolean mask of active presynaptic cells for each synapse on each segment.
        # Clamp distal connections to ensure indices are within valid bounds (defensive programming)
        col_distal_connections_clamped = torch.clamp(col_distal_connections, 0, self.num_cells - 1)
        presynaptic_was_active = input_active_cells_single_item[col_distal_connections_clamped]
        
        # An active segment has enough connected *and* active synapses
        # Sum of (connected & active) synapses along the last dimension (synapses_per_segment)
        active_synapses_on_segment = (combined_connected & presynaptic_was_active).sum(dim=-1)
        
        # A cell is predictive if any of its segments have enough active synapses to cross the threshold
        predictive_segments_mask = active_synapses_on_segment >= self.activation_threshold
        predictive_cells_mask = predictive_segments_mask.any(dim=-1) # True if any segment predicts for that cell
        
        return predictive_cells_mask

    def _learn(self, prev_active_cells: torch.Tensor, modulation_signal_batch: torch.Tensor):
        """
        Performs learning based on the current winner cells and previous active cells for the batch.
        Updates volatile permanences.
        
        Args:
            prev_active_cells (torch.Tensor): Batch of active cells from the previous time step. Shape: [batch_size, num_cells].
            modulation_signal_batch (torch.Tensor): Batch of modulation signals. Shape: [batch_size].
        """
        # This function still contains sequential processing per batch item.
        # Vectorizing this is the next significant step after the main forward pass.
        
        batch_size = prev_active_cells.shape[0]

        for batch_item_idx in range(batch_size):
            if modulation_signal_batch[batch_item_idx].item() == 0.0 or not prev_active_cells[batch_item_idx].any():
                continue

            # Get global indices of cells that are "winners" in the current step for this batch item
            learning_cells_indices_global = torch.where(self.winner_cells[batch_item_idx])[0]
            if not learning_cells_indices_global.numel():
                continue

            # Get relevant segment overlaps for all learning cells for this batch item
            segment_overlaps_for_learning_cells = self._get_matching_segments(learning_cells_indices_global, prev_active_cells[batch_item_idx])
            
            if segment_overlaps_for_learning_cells is None: 
                continue # No segments to learn on

            # For each learning cell, select the best segment to strengthen.
            best_segment_indices_for_learning = torch.argmax(segment_overlaps_for_learning_cells, dim=-1)

            # Create lists of indices to flatten the operation
            all_selected_cell_indices = []
            all_selected_segment_indices = []
            for i, cell_global_idx in enumerate(learning_cells_indices_global):
                seg_local_idx = best_segment_indices_for_learning[i].item()
                all_selected_cell_indices.extend([cell_global_idx] * self.synapses_per_segment)
                all_selected_segment_indices.extend([seg_local_idx] * self.synapses_per_segment)

            # Convert to tensors for advanced indexing
            selected_cell_indices_tensor = torch.tensor(all_selected_cell_indices, device=self.device)
            selected_segment_indices_tensor = torch.tensor(all_selected_segment_indices, device=self.device)

            # Get global indices of presynaptic cells for the selected synapses
            synapse_indices = torch.arange(self.synapses_per_segment, device=self.device).repeat(len(learning_cells_indices_global))
            synapses_to_update_global_indices = self.distal_connections[
                selected_cell_indices_tensor, selected_segment_indices_tensor, synapse_indices
            ]
            
            # Determine which presynaptic cells were active
            presynaptic_was_active_mask_flat = prev_active_cells[batch_item_idx][synapses_to_update_global_indices]

            potentiation = self.volatile_learning_rate * modulation_signal_batch[batch_item_idx].item()
            punishment = potentiation * 0.1 # N.B.: original was 0.1, making it dynamic now

            # --- Apply updates to volatile_permanences ---
            # These updates are currently still modifying the SHARED self.volatile_permanences
            # If each batch item should have its own set of permanences, this structure needs deeper thought.
            # Assuming shared permanences that are updated by the collective batch.
            self.volatile_permanences[
                selected_cell_indices_tensor, selected_segment_indices_tensor, synapse_indices
            ][presynaptic_was_active_mask_flat] += potentiation
            
            self.volatile_permanences[
                selected_cell_indices_tensor, selected_segment_indices_tensor, synapse_indices
            ][~presynaptic_was_active_mask_flat] -= punishment
            
            # Clamp permanences to [0, 1] range
            self.volatile_permanences.clamp_(0, 1) # Use in-place clamp for efficiency


    def consolidate(self, memory_trace: list[torch.Tensor]):
        """
        Consolidates a memory trace (sequence of SDRs) into long-term memory.
        """
        # This function still processes traces sequentially.
        # Batching consolidation would require memory_trace to be structured as a batch of traces.
        with self.lock:
            logger.info(f"Consolidating a memory trace of length {len(memory_trace)}...") 
            
            # Initialize temporary states for consolidation process
            # These will act as batch_size=1 internal states for the sequential consolidation loop
            temp_active_cells_for_consolidation = torch.zeros(1, self.num_cells, dtype=torch.bool, device=self.device)
            temp_predictive_cells_for_consolidation = torch.zeros(1, self.num_cells, dtype=torch.bool, device=self.device)

            for sdr in memory_trace: # sdr is typically [num_columns]
                # Ensure sdr is [1, num_columns] for consistent processing
                sdr_single_item = sdr.unsqueeze(0) if sdr.dim() == 1 else sdr

                current_active_cells_for_consolidation_single = torch.zeros(1, self.num_cells, dtype=torch.bool, device=self.device)
                current_winner_cells_for_consolidation_single = torch.zeros(1, self.num_cells, dtype=torch.bool, device=self.device)

                active_column_indices_in_sdr = torch.nonzero(sdr_single_item.squeeze(), as_tuple=True)[0]
                
                for col_idx in active_column_indices_in_sdr:
                    col_cells_indices = self.get_column_cell_indices(col_idx.item())
                    
                    # _compute_predictive_state expects a single item prev_active_cells
                    predictive_mask = self._compute_predictive_state(col_idx.item(), temp_active_cells_for_consolidation.squeeze(0))

                    if torch.any(predictive_mask):
                        current_active_cells_for_consolidation_single[0, col_cells_indices] = predictive_mask
                        current_winner_cells_for_consolidation_single[0, col_cells_indices] = predictive_mask
                    else:
                        current_active_cells_for_consolidation_single[0, col_cells_indices] = True
                        current_winner_cells_for_consolidation_single[0, col_cells_indices] = torch.ones(self.cells_per_column, dtype=torch.bool, device=self.device)

                # Perform consolidated learning
                # _consolidated_learn expects single item tensors, so squeeze batch dim
                self._consolidated_learn(temp_active_cells_for_consolidation.squeeze(0), current_winner_cells_for_consolidation_single.squeeze(0))
                
                # Update temporary active cells for the next step of the trace
                temp_active_cells_for_consolidation = current_active_cells_for_consolidation_single.clone()
                
                # Update predictive cells for the next consolidation step
                # This logic is complex and needs to be carefully vectorized if consolidation needs to be batched
                # For now, it's a placeholder to show where it would happen for single items in the trace.
                # It currently iterates over active columns for *this* SDR to compute next predictions for next SDR.
                for col_idx in active_column_indices_in_sdr:
                    predictive_mask = self._compute_predictive_state(col_idx.item(), temp_active_cells_for_consolidation.squeeze(0))
                    col_cells_indices = self.get_column_cell_indices(col_idx.item())
                    temp_predictive_cells_for_consolidation[0, col_cells_indices] = predictive_mask
            
            # Reset the main TM active/predictive state after consolidation
            # This ensures the TM starts fresh for regular forward passes after a consolidation run.
            self.active_cells.zero_()
            self.predictive_cells.zero_()
            self.winner_cells.zero_()
            logger.info("Consolidation complete.")
    
    def _consolidated_learn(self, prev_active_cells_single_item: torch.Tensor, winner_cells_single_item: torch.Tensor):
        """
        Performs learning using the consolidated learning rate for a single item.
        
        Args:
            prev_active_cells_single_item (torch.Tensor): 1D boolean tensor of active cells from the previous time step. Shape: [num_cells].
            winner_cells_single_item (torch.Tensor): 1D boolean tensor of winner cells for the current time step. Shape: [num_cells].
        """
        learning_cells_indices = torch.where(winner_cells_single_item)[0]
        if not learning_cells_indices.numel(): return

        segment_overlaps = self._get_matching_segments(learning_cells_indices, prev_active_cells_single_item)
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
        synapse_indices = torch.arange(self.synapses_per_segment, device=self.device).repeat(len(learning_cells_indices))
        synapses_to_update_global_indices = self.distal_connections[
            selected_cell_indices_tensor, selected_segment_indices_tensor, synapse_indices
        ]

        # Determine which presynaptic cells were active
        presynaptic_was_active_mask_flat = prev_active_cells_single_item[synapses_to_update_global_indices]

        potentiation = self.consolidated_learning_rate
        punishment = potentiation * 0.25 # Can be different from volatile

        # Apply updates to consolidated_permanences (shared across batch items)
        self.consolidated_permanences[
            selected_cell_indices_tensor, selected_segment_indices_tensor, synapse_indices
        ][presynaptic_was_active_mask_flat] += potentiation
        
        self.consolidated_permanences[
            selected_cell_indices_tensor, selected_segment_indices_tensor, synapse_indices
        ][~presynaptic_was_active_mask_flat] -= punishment
        
        # Clamp permanences
        self.consolidated_permanences.clamp_(0, 1) # Use in-place clamp for efficiency


    def get_column_cell_indices(self, col_idx: int) -> torch.Tensor:
        """
        Returns the global indices of all cells belonging to a given column.
        """
        col_cells_start = col_idx * self.cells_per_column
        col_cells_end = col_cells_start + self.cells_per_column
        return torch.arange(col_cells_start, col_cells_end, device=self.device)