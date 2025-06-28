# src/model_architecture.py
import torch
from torch import compile
import torch.nn as nn
import torch.nn.functional as F
import logging
import random
import asyncio 
import numpy as np 

from typing import TYPE_CHECKING, Optional, List
from src.websocket_manager import broadcast_memory_state

if TYPE_CHECKING:
    from src.websocket_manager import ConnectionManager

from src.model_parts import CombineContext, DenoiseNet, ProjectHead, UpdateFastState
from src.text_decoder import TextDecoder
from src.utils import initialize_weights, decode_sequence, Vocabulary 
from src.emotion import EmotionalCore
from src.heart import Heart
from src.self_reflection import SelfReflectionModule
from src.self_prompting import SelfPromptingModule
from src.encoders import SensorimotorEncoder
from src.spatial_pooler import SpatialPooler
from src.temporal_memory import TemporalMemory
from src.consolidation import ConsolidationManager
from src.text_sdr_encoder import TextSdrEncoder 

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class ContinuouslyReasoningPredictor(nn.Module):
    """
    The Continuously Reasoning Predictor (CRP) model.
    This model integrates a reasoning diffusion process with continuous learning,
    governed by a rhythmic EmotionalCore and regulated by a homeostatic Heart.
    """
    def __init__(self, 
                 text_decoder_config: dict = None, 
                 spatial_pooler_config: dict = None, 
                 temporal_memory_config: dict = None, 
                 sdr_size: int = 2048, 
                 sdr_active_bits: int = 40, 
                 device: torch.device = None,
                 vocab: Optional[Vocabulary] = None 
                 ):
        super().__init__()
        self.device = device
        
        # --- Text Encoding with Custom RDSE ---
        if vocab is None:
            raise ValueError("Vocabulary instance must be provided to ContinuouslyReasoningPredictor.")
        self.vocab = vocab
        self.sdr_size = sdr_size
        self.sdr_active_bits = sdr_active_bits
        self.text_sdr_encoder = TextSdrEncoder(
            vocab=self.vocab, 
            sdr_size=self.sdr_size, 
            sdr_active_bits=self.sdr_active_bits
        )
        self.embedding_dim = self.sdr_size 
        logger.info(f"Model embedding dimension (SDR size): {self.embedding_dim}")

        # --- Core Cognitive Components ---
        self.emotions = EmotionalCore().to(device)
        self.heart = Heart()
        self.model_dims = self.emotions.model_dims.int().item()
        self.knowledge_dims = self.emotions.knowledge_dims.int().item()
        
        if self.embedding_dim != self.model_dims:
            self.input_projection = nn.Linear(self.embedding_dim, self.model_dims).to(device)
            initialize_weights(self.input_projection)
        else:
            self.input_projection = nn.Identity().to(device)

        # --- HTM Components ---
        # Note: temporal_memory_config should contain 'input_dims', 'columns', 'cells_per_column' etc.
        # This will be passed directly to TemporalMemory's __init__
        self.sensorimotor_encoder = SensorimotorEncoder(self.model_dims, spatial_pooler_config['input_dims']).to(device)
        self.spatial_pooler = SpatialPooler(**spatial_pooler_config).to(device)
        self.temporal_memory = TemporalMemory(**temporal_memory_config, device=device).to(device)
        
        # --- Language and Reasoning Components ---
        text_decoder_config['embedding_dim'] = self.embedding_dim
        self.text_decoder = TextDecoder(**text_decoder_config).to(device)
        self.project_head = ProjectHead(self.model_dims, self.embedding_dim).to(device)
        self.combine_context = CombineContext(self.model_dims, self.model_dims, self.knowledge_dims, self.model_dims).to(device)
        self.denoise_net = DenoiseNet(self.model_dims, self.model_dims, self.emotions.time_embedding_dim.int().item()).to(device)
        htm_output_dims = temporal_memory_config['columns'] * temporal_memory_config['cells_per_column']
        self.htm_to_embedding_projection = ProjectHead(htm_output_dims, self.embedding_dim).to(device)

        # --- State Vectors and Update Functions ---
        self.slow_state = nn.Parameter(torch.randn(1, self.knowledge_dims, device=device) * 0.01)
        self.fast_state = nn.Parameter(torch.zeros(1, self.model_dims, device=device), requires_grad=False)
        self.update_fast_state_func = UpdateFastState(self.model_dims, self.model_dims, self.model_dims).to(device)

        # --- Other Modules and Managers ---
        self.self_reflection_module = SelfReflectionModule(self.model_dims, self.model_dims).to(device)
        self.self_prompting_module = SelfPromptingModule().to(device)
        self.consolidation_manager = ConsolidationManager(temporal_memory=self.temporal_memory)

        # --- State Variables ---
        self.pad_token_id = text_decoder_config.get('pad_token_id', 0)
        self.sos_token_id = text_decoder_config.get('sos_token_id', 1)
        self.eos_token_id = text_decoder_config.get('eos_token_id', 2)
        self.consolidation_counter = 0
        self.consolidation_interval = 10 
        self.latest_confidence = None
        self.latest_meta_error = None
        self.latest_state_drift = None
        self.latest_tm_predictive_accuracy = None
        self.latest_heart_metrics = {}
        self.latest_predictive_accuracy = None
        self.latest_tm_sparsity = None

        self.apply(initialize_weights)
        self.to(self.device)
        logger.info("ContinuouslyReasoningPredictor initialized with new configurable architecture.")

    async def _reason_to_predict(self, model_dims_input_embedding: torch.Tensor, stop_event: asyncio.Event = None) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = model_dims_input_embedding.shape[0]
        current_slow_state = self.slow_state.expand(batch_size, -1)
        current_fast_state = self.fast_state.expand(batch_size, -1)

        c_t = self.combine_context(model_dims_input_embedding, current_fast_state, current_slow_state)
        z_tau = torch.randn(batch_size, self.model_dims, device=self.device)

        T_DIFF_STEPS = self.emotions.get_focus()

        for tau_int in reversed(range(1, T_DIFF_STEPS + 1)):
            if stop_event and stop_event.is_set():
                raise asyncio.CancelledError("Training stopped by user.")
            tau_tensor = torch.tensor([tau_int], dtype=torch.float32, device=self.device)
            z_tau = self.denoise_net(z_tau, tau_tensor, c_t)
            await asyncio.sleep(0)
        z_0 = z_tau
        return z_0, c_t


    async def _get_predicted_embedding(self, input_embedding_sdr: torch.Tensor, stop_event: asyncio.Event = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        self.emotions.update()
        
        processed_input_embedding = self.input_projection(input_embedding_sdr)

        # HTM Integration Start
        encoded_input = self.sensorimotor_encoder(processed_input_embedding)
        sdr_batch = self.spatial_pooler(encoded_input) # sdr_batch is [batch_size, num_columns_in_sdr]
        
        # Derive modulation signal (e.g., from curiosity)
        modulation_scalar = self.emotions.get_curiosity()
        if torch.is_tensor(modulation_scalar):
            modulation_scalar = modulation_scalar.item()
        
        # Expand modulation_scalar to a batch_size tensor
        batch_size = sdr_batch.shape[0]
        modulation_signal_batch = torch.full((batch_size,), modulation_scalar, device=self.device)

        # --- START: Modified TM processing for batching ---
        # Pass the entire batch to the TemporalMemory's forward method
        active_cells_current_step, predictive_cells_for_next_step, tm_batch_accuracy = \
            self.temporal_memory(sdr_batch, modulation_signal_batch) # Pass full batch and batched signal
        
        # Update TM metrics by averaging across the batch
        self.latest_tm_predictive_accuracy = tm_batch_accuracy.mean().item() # tm_batch_accuracy is now a tensor [batch_size]
        
        # Calculate sparsity across the entire batch's active cells
        # tm_output_active_cells is [batch_size, num_cells]
        total_active_cells_batch = active_cells_current_step.float().sum(dim=-1) # Sum active cells per item [batch_size]
        total_possible_cells = active_cells_current_step.numel() / batch_size # Total cells for a single item
        
        # Calculate sparsity for each item, then average
        batch_sparsities = (total_active_cells_batch / total_possible_cells)
        self.latest_tm_sparsity = batch_sparsities.mean().item()
        # --- END: Modified TM processing for batching ---

        # For visualization, select the LAST item from the batch for display
        final_active_cells = active_cells_current_step[-1] # Take the last item in the batch
        final_predictive_cells_for_next_step = predictive_cells_for_next_step[-1] # Take the last item in the batch

        # Ensure final_active_cells is not None (e.g., if sdr_batch was empty)
        if final_active_cells is None or final_active_cells.numel() == 0:
            # Fallback to a zero tensor if no SDRs were processed or batch was empty
            htm_output_dense = torch.zeros(self.temporal_memory.num_cells, device=self.device)
        else:
            # Convert sparse active_cells (boolean tensor) to a dense float tensor for projection
            htm_output_dense = final_active_cells.float()
        
        # Project HTM output to embedding_dim
        htm_predicted_embedding = self.htm_to_embedding_projection(htm_output_dense)
        # HTM Integration End

        reasoned_state_z0, c_t_returned = await self._reason_to_predict(processed_input_embedding, stop_event)

        # batch_size is already defined from processed_input_embedding
        expanded_fast_state_for_reflection = self.fast_state.expand(batch_size, -1)
        current_confidence, current_meta_error = self.self_reflection_module(
            reasoned_state=reasoned_state_z0,
            fast_state=expanded_fast_state_for_reflection,
            slow_state=self.slow_state # slow_state is [1, knowledge_dims], self_reflection_module expands it
        )

        self.latest_confidence = current_confidence.mean().item() # Mean over batch for scalar update
        self.latest_meta_error = current_meta_error.mean().item() # Mean over batch for scalar update
        
        with torch.no_grad():
            # For cosine similarity, ensure both tensors have a batch dimension if needed
            # Here, slow_state is (1, knowledge_dims) and fast_state.mean is (1, model_dims)
            # They need to be compatible or projected to the same space for a meaningful similarity.
            # Assuming model_dims and knowledge_dims are intended to be comparable,
            # or perhaps this similarity is just for reporting state drift.
            # If fast_state and slow_state are different dimensions, direct cosine similarity might fail.
            # Assuming for now they can be compared after appropriate expansion/reduction.
            # If model_dims != knowledge_dims, this might need a projection.
            # For current context, assuming `fast_state.mean(dim=0, keepdim=True)` makes it compatible.
            similarity = F.cosine_similarity(self.slow_state, self.fast_state.mean(dim=0, keepdim=True))
            self.latest_state_drift = (1.0 - similarity.mean()).item()

        self.latest_heart_metrics = self.heart.beat(self.emotions, current_confidence, current_meta_error) # Pass batched confidence/meta_error to heart if it handles it, or average here. Current Heart expects scalars.
        # Adjusted: Heart.beat expects scalar confidence/meta_error, so pass the means.
        # If Heart.beat expects full tensors for internal logic (e.g., batch-wise stress), it needs refactoring.
        # Based on heart.py, it uses .mean().item(), so passing raw current_confidence, current_meta_error is fine if they are [batch_size] tensors.
        # Let's check heart.py again: it calls `latest_confidence.mean().item()`. So passing `current_confidence` (tensor) is correct.

        predicted_embedding = self.project_head(reasoned_state_z0) + htm_predicted_embedding

        with torch.no_grad():
            expanded_fast_state = self.fast_state.expand(batch_size, -1)
            new_fast_state = self.update_fast_state_func(
                expanded_fast_state,
                processed_input_embedding,
                reasoned_state_z0
            ).mean(dim=0, keepdim=True) # Averaging over batch if batch_size > 1
            self.fast_state.copy_(new_fast_state)

        # Return the final active cells state for potential visualization, and the sdr_batch
        return predicted_embedding, sdr_batch, final_active_cells, self.latest_tm_predictive_accuracy

    async def learn_one_step(self, text_t_batch: List[str], target_sequence_t_plus_1_batch: torch.Tensor, websocket_manager: "ConnectionManager" = None, stop_event: asyncio.Event = None):
            if stop_event and stop_event.is_set():
                raise asyncio.CancelledError("Training stopped by user.")

            # Encode the batch of text strings into sequences of SDRs
            # This returns List[List[np.ndarray]]
            batch_sdr_sequences = self.text_sdr_encoder.encode_batch(text_t_batch)
            
            # Convert list of numpy SDR sequences to a single torch.Tensor
            # Shape: [batch_size, sdr_size]
            input_embedding_sdr_batch = []
            for sdr_sequence in batch_sdr_sequences:
                if not sdr_sequence:
                    # Handle empty text case for an individual sample in the batch
                    input_embedding_sdr_batch.append(torch.zeros(self.sdr_size, dtype=torch.float32, device=self.device))
                    logger.warning("Empty text input received in learn_one_step batch. Using zero SDR for a sample.")
                else:
                    # Average the SDRs in the sequence to get a single embedding per text
                    input_embedding_sdr_batch.append(
                        torch.tensor(np.array(sdr_sequence), dtype=torch.float32, device=self.device).mean(dim=0)
                    )
            
            # Stack all individual embeddings into a batch tensor
            input_embedding_sdr_batch = torch.stack(input_embedding_sdr_batch, dim=0)


            predicted_embedding, spatial_pooler_output, active_cells_for_viz, tm_accuracy_metric = await self._get_predicted_embedding(input_embedding_sdr_batch, stop_event)
            self.latest_tm_predictive_accuracy = tm_accuracy_metric # Store it for later access

            # target_sequence_t_plus_1_batch is already [batch_size, max_len]
            max_len = target_sequence_t_plus_1_batch.shape[1]
            decoded_logits = self.text_decoder.forward_teacher_forced(predicted_embedding, max_len, target_sequence_t_plus_1_batch)

            loss_criterion = nn.CrossEntropyLoss(ignore_index=self.pad_token_id)
            loss_value = loss_criterion(
                decoded_logits.reshape(-1, self.text_decoder.vocab_size),
                target_sequence_t_plus_1_batch.reshape(-1)
            )

            # Consolidation Trigger Logic (remains largely the same, operates on latest states)
            # `current_confidence` and `current_meta_error` from _get_predicted_embedding are now tensors [batch_size]
            # Here we check their mean, which is suitable for a single global decision to consolidate.
            if self.latest_confidence is not None and self.latest_meta_error is not None:
                confidence_threshold = 0.8
                meta_error_threshold = 0.1
                
                # Use the scalar `self.latest_confidence` and `self.latest_meta_error` which are already means
                if self.latest_confidence > confidence_threshold and \
                   self.latest_meta_error < meta_error_threshold:
                    
                    logger.debug(f"Adding to consolidation queue: Confidence={self.latest_confidence:.4f}, MetaError={self.latest_meta_error:.4f}")
                    
                    # Add current HTM states to consolidation queue
                    # spatial_pooler_output is now a batch of SDRs [batch_size, sdr_size].
                    # Pass individual SDRs from batch for consolidation.
                    self.consolidation_manager.add_to_consolidation_queue(
                        memory_trace=[sdr for sdr in spatial_pooler_output.detach().cpu()] # Detach and move to CPU if consolidation manager is CPU-bound
                    )
                    
                    self.consolidation_counter += 1
                    if self.consolidation_counter % self.consolidation_interval == 0:
                        logger.info("Initiating consolidation of memories from queue.")
                        pass # Actual consolidation happens in a background thread via ConsolidationManager
                        self.consolidation_counter = 0 
            
            # Memory visualization (uses the last item from the batch for display)
            if websocket_manager and websocket_manager.active_connections:
                # Use the `active_cells_for_viz` returned by _get_predicted_embedding, which is already a single item
                sdr_for_viz = spatial_pooler_output[-1] # The last SDR from the batch
                sdr_list = sdr_for_viz.flatten().int().tolist()
                
                active_cells_list = active_cells_for_viz.flatten().int().tolist()
                predictive_cells_list = self.temporal_memory.predictive_cells[-1].flatten().int().tolist() # Get last item of current predictive state

                volatile_perms = self.temporal_memory.volatile_permanences
                consolidated_perms = self.temporal_memory.consolidated_permanences

                volatile_hist = torch.histogram(volatile_perms.flatten().cpu(), bins=32, range=(0.0, 1.0))
                consolidated_hist = torch.histogram(consolidated_perms.flatten().cpu(), bins=32, range=(0.0, 1.0))
                
                permanence_data = {
                    'volatile': {
                        'values': volatile_hist[0].tolist(), 
                        'bins': volatile_hist[1].tolist()
                    },
                    'consolidated': {
                        'values': consolidated_hist[0].tolist(), 
                        'bins': consolidated_hist[1].tolist()
                    }
                }

                sdr_total_bits = sdr_for_viz.numel()
                grid_cols_sdr = int(sdr_total_bits**0.5)
                grid_rows_sdr = (sdr_total_bits + grid_cols_sdr - 1) // grid_cols_sdr

                memory_state_payload = {
                    "sdr": sdr_list,
                    "activeCells": active_cells_list,
                    "predictiveCells": predictive_cells_list, 
                    "permanences": permanence_data,
                    "gridDimensions": {
                        "sdr": [grid_rows_sdr, grid_cols_sdr], 
                        "cells": [self.temporal_memory.columns, self.temporal_memory.cells_per_column]
                    }
                }
                await broadcast_memory_state(websocket_manager, memory_state_payload)
                
            return loss_value

    async def generate_text(self, input_texts: List[str], max_len: int = None, top_p: float = 0.9) -> List[List[int]]:
        """
        Generates text for a batch of input texts.
        input_texts is now a List[str].
        Returns List[List[int]] (list of token ID sequences).
        """
        if max_len is None:
            max_len = self.emotions.max_seq_len.int().item()

        self.eval()
        with torch.no_grad():
            # Encode the batch of text strings into SDRs
            batch_sdr_sequences_encoded = self.text_sdr_encoder.encode_batch(input_texts)
            
            # Convert list of numpy SDR sequences to a single torch.Tensor [batch_size, sdr_size]
            input_embedding_sdr_batch = []
            for sdr_sequence in batch_sdr_sequences_encoded:
                if not sdr_sequence:
                    input_embedding_sdr_batch.append(torch.zeros(self.sdr_size, dtype=torch.float32, device=self.device))
                else:
                    input_embedding_sdr_batch.append(
                        torch.tensor(np.array(sdr_sequence), dtype=torch.float32, device=self.device).mean(dim=0)
                    )
            input_embedding_sdr_batch = torch.stack(input_embedding_sdr_batch, dim=0)

            predicted_embedding, spatial_pooler_output, _, _ = await self._get_predicted_embedding(input_embedding_sdr_batch)
            
            batch_size = predicted_embedding.shape[0]
            hidden = self.text_decoder.get_initial_hidden(predicted_embedding)
            
            input_token = torch.full((batch_size,), self.sos_token_id, dtype=torch.long, device=self.device)
            generated_sequences = torch.zeros(batch_size, max_len, dtype=torch.long, device=self.device)
            
            for t in range(max_len):
                logits, hidden = self.text_decoder.forward(input_token, hidden)
                
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = torch.zeros_like(logits, dtype=torch.bool).scatter_(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float('Inf')
                
                next_token_probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(next_token_probs, 1).squeeze(1)

                generated_sequences[:, t] = next_token
                input_token = next_token

                # Stop generation for sequences that have reached EOS
                # Use a boolean mask to track which sequences are done
                if (next_token == self.eos_token_id).all():
                    break
        
        return generated_sequences.tolist()


    async def generate_internal_thought(self, vocab, max_len: int = 64, input_prompt_override: str = None) -> tuple[str, float, float, int, float, str]:
        self.eval()
        with torch.no_grad():
            if input_prompt_override is not None:
                self_reflection_prompt = input_prompt_override
            else:
                current_states = {
                    "confidence": self.latest_confidence.mean().item() if self.latest_confidence is not None else 0.0,
                    "meta_error": self.latest_meta_error.mean().item() if self.latest_meta_error is not None else 0.0,
                    "focus": self.emotions.get_focus(),
                    "curiosity": self.emotions.get_curiosity()
                }
                self_reflection_prompt = self.self_prompting_module.generate_prompt(current_states)

            # Now, generate_text expects a list of strings
            generated_ids_list_batch = await self.generate_text([self_reflection_prompt], max_len=max_len)
            thought_text = decode_sequence(generated_ids_list_batch[0], vocab, self.eos_token_id) # Take the first (and only) sequence

            confidence = self.latest_confidence # This is already a scalar mean from _get_predicted_embedding
            meta_error = self.latest_meta_error # This is already a scalar mean from _get_predicted_embedding
            focus = self.emotions.get_focus()
            curiosity = self.emotions.get_curiosity()

            logger.info(f"Internal Prompt: '{self_reflection_prompt}' -> Thought: '{thought_text}' - Conf: {confidence:.4f}, ME: {meta_error:.4f}, F: {focus}, C: {curiosity:.6f}")

            return thought_text, confidence, meta_error, focus, curiosity, self_reflection_prompt