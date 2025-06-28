# src/training_manager.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.amp import autocast
from torch.cuda.amp import GradScaler
import asyncio
import logging
import os
import glob

from src.model_architecture import ContinuouslyReasoningPredictor
from src.utils import Vocabulary, encode_long_text
from src.api_models import TrainingStatusResponse
from src.data_processing import TextDataset, collate_fn, load_texts_from_directory, create_file_manifest
from src.websocket_manager import ConnectionManager, broadcast_state_update

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# --- Training Related Configurations ---
DATA_DIR = os.getenv("DATA_DIR", "/app/data")
TRAIN_EPOCHS = int(os.getenv("TRAIN_EPOCHS", "500"))
TRAIN_BATCH_SIZE = int(os.getenv("TRAIN_BATCH_SIZE", "32"))
MAX_SEQ_LEN = int(os.getenv("MAX_SEQ_LEN", "256"))
INITIAL_LEARNING_RATE = float(os.getenv("INITIAL_LEARNING_RATE", "0.001"))
SAVE_INTERVAL_BATCHES = int(os.getenv("SAVE_INTERVAL_BATCHES", "100"))

# --- Global State for Training ---
class TrainingState:
    def __init__(self):
        self.is_training_active: bool = False
        self.training_task: asyncio.Task | None = None
        self.current_epoch: int = 0
        self.current_batch: int = 0
        self.total_batches_in_epoch: int = 0
        self.train_loss: float = 0.0
        self.val_loss: float = float('inf')
        self.best_val_loss: float = float('inf')
        self.stop_training_flag: asyncio.Event = asyncio.Event()

# --- Training Loop Functions ---
async def train_model_step(
    model: ContinuouslyReasoningPredictor,
    train_dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_dataloader: DataLoader,
    best_val_loss_ref: list,
    training_state: TrainingState,
    websocket_manager: ConnectionManager,
    model_checkpoint_path: str,
    device: torch.device,
    scaler: GradScaler,
    effective_save_interval: int,
    lock: asyncio.Lock,
    data_manifest: dict
) -> bool:
    total_loss = 0
    num_batches_in_epoch = len(train_dataloader)
    training_state.total_batches_in_epoch = num_batches_in_epoch

    for batch_idx, (input_texts, target_sequences) in enumerate(train_dataloader):
        if training_state.stop_training_flag.is_set():
            logger.info("Training stop signal received at batch start. Exiting current epoch.")
            return False

        async with lock:
            model.train()
            optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                # Directly encode the batch of texts using the SentenceTransformer's optimized method.
                # The `encode_long_text` utility is still useful for *single* very long texts,
                # but for batch processing, the model's direct `encode` is more efficient.
                input_embeddings = model.embedding_model.encode(
                    input_texts,
                    convert_to_tensor=True,
                    device=device,
                    show_progress_bar=False # Optional: disable if not needed
                )


            target_sequences = target_sequences.to(device)

            try:
                with autocast(device_type=device.type):
                    loss = await model.learn_one_step(input_embeddings, target_sequences, websocket_manager=websocket_manager, stop_event=training_state.stop_training_flag)
            except asyncio.CancelledError:
                logger.info("Training cancelled by signal during learn_one_step. Exiting current epoch.")
                return False

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

            if effective_save_interval > 0 and (batch_idx + 1) % effective_save_interval == 0:
                logger.info(f"--- Performing intermediate validation at Epoch {epoch}, Batch {batch_idx+1} ---")
                current_val_loss = await validate_model_step(model, val_dataloader, device, max_batches=100)
                training_state.val_loss = current_val_loss
                
                # --- Trigger the Rebase mechanism after validation ---
                model.heart.rebase(model.emotions)

                if current_val_loss < best_val_loss_ref[0]:
                    best_val_loss_ref[0] = current_val_loss
                    training_state.best_val_loss = current_val_loss
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_loss': current_val_loss,
                        'data_manifest': data_manifest
                    }, model_checkpoint_path)
                    logger.info(f"New best model saved with Validation Loss: {current_val_loss:.4f}")

        training_state.current_batch = batch_idx + 1
        training_state.train_loss = total_loss / (batch_idx + 1)

        await broadcast_state_update(
            manager=websocket_manager,
            model=model,
            is_training=True,
            message=f"Epoch {epoch}, Batch {batch_idx+1}: Loss {loss.item():.4f}",
            epoch=epoch,
            batch=training_state.current_batch,
            total_batches=num_batches_in_epoch,
            train_loss=training_state.train_loss,
            val_loss=training_state.val_loss if training_state.val_loss != float('inf') else None
        )

    logger.info(f"Epoch [{epoch}/{TRAIN_EPOCHS}] complete. Average Training Loss: {total_loss / num_batches_in_epoch:.4f}")
    return True

async def validate_model_step(
    model: ContinuouslyReasoningPredictor,
    val_dataloader: DataLoader,
    device: torch.device,
    max_batches: int | None = None
):
    model.eval()
    total_val_loss = 0
    num_batches = 0

    with torch.no_grad():
        for batch_idx, (input_texts, target_sequences) in enumerate(val_dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break

                # Directly encode the batch of texts using the SentenceTransformer's optimized method.
                # The `encode_long_text` utility is still useful for *single* very long texts,
                # but for batch processing, the model's direct `encode` is more efficient.
                input_embeddings = model.embedding_model.encode(
                    input_texts,
                    convert_to_tensor=True,
                    device=device,
                    show_progress_bar=False # Optional: disable if not needed
                )

            target_sequences = target_sequences.to(device)
            
            with autocast(device_type=device.type):
                # FIX: Unpack the tuple returned by _get_predicted_embedding
                predicted_embedding, _, _, _ = await model._get_predicted_embedding(input_embeddings)
                decoded_logits = model.text_decoder.forward_teacher_forced(predicted_embedding, target_sequences.size(1), target_sequences)

                criterion = nn.CrossEntropyLoss(ignore_index=model.pad_token_id)
                loss = criterion(
                    decoded_logits.reshape(-1, model.text_decoder.vocab_size),
                    target_sequences.reshape(-1)
                )

            total_val_loss += loss.item()
            num_batches += 1
            await asyncio.sleep(0.001)

    if num_batches == 0:
        return float('inf')

    avg_val_loss = total_val_loss / num_batches
    logger.info(f"Validation complete over {num_batches} batches. Average Validation Loss: {avg_val_loss:.4f}")
    return avg_val_loss

async def run_model_training(
    model: ContinuouslyReasoningPredictor,
    vocab: Vocabulary,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    training_state: TrainingState,
    websocket_manager: ConnectionManager,
    model_checkpoint_path: str,
    device: torch.device,
    scaler: GradScaler,
    lock: asyncio.Lock
):
    training_state.is_training_active = True
    training_state.stop_training_flag.clear()
    
    training_state.current_epoch = 0
    training_state.current_batch = 0
    training_state.train_loss = 0.0
    training_state.val_loss = float('inf')
    training_state.best_val_loss = float('inf')

    await broadcast_state_update(
        manager=websocket_manager, model=model, is_training=True, message="Training started..."
    )
    logger.info("Starting model training process...")

    train_data_dir = os.path.join(DATA_DIR, 'train')
    val_data_dir = os.path.join(DATA_DIR, 'val')
    
    current_train_manifest = await asyncio.to_thread(create_file_manifest, train_data_dir)
    current_val_manifest = await asyncio.to_thread(create_file_manifest, val_data_dir)
    current_data_manifest = {**current_train_manifest, **current_val_manifest}

    train_texts = await asyncio.to_thread(load_texts_from_directory, train_data_dir)
    val_texts = await asyncio.to_thread(load_texts_from_directory, val_data_dir)
    train_dataset = TextDataset(train_texts, vocab, MAX_SEQ_LEN)
    val_dataset = TextDataset(val_texts, vocab, MAX_SEQ_LEN)
    train_dataloader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    
    best_val_loss_ref = [float('inf')]
    if os.path.exists(model_checkpoint_path):
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache() # Clear CUDA cache before loading checkpoint
            checkpoint = torch.load(model_checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            
            if 'optimizer_state_dict' in checkpoint:
                try:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                except:
                    logger.warning("Could not load optimizer state_dict, possibly due to new model parameters. Optimizer will be reset.")

            old_manifest = checkpoint.get('data_manifest')
            if old_manifest and old_manifest == current_data_manifest:
                if 'val_loss' in checkpoint:
                    best_val_loss_ref[0] = checkpoint['val_loss']
                    training_state.best_val_loss = checkpoint['val_loss']
            else:
                logger.warning("Training data has changed since last checkpoint. Resetting best validation loss.")
                training_state.best_val_loss = float('inf')

            logger.info(f"Loaded existing model checkpoint. Best val loss is set to: {best_val_loss_ref[0]:.4f}")

        except Exception as e:
            logger.error(f"Could not load model checkpoint: {e}. Starting with a new model.", exc_info=True)


    num_batches_in_epoch = len(train_dataloader)
    dynamic_save_interval = SAVE_INTERVAL_BATCHES
    if num_batches_in_epoch < SAVE_INTERVAL_BATCHES:
        dynamic_save_interval = max(1, num_batches_in_epoch // 2)
        if dynamic_save_interval == 0:
            dynamic_save_interval = 1
        logger.info(f"Adjusting save interval to {dynamic_save_interval} due to small dataset size.")

    for epoch in range(TRAIN_EPOCHS):
        training_state.current_epoch = epoch
        if training_state.stop_training_flag.is_set():
            break

        should_continue = await train_model_step(
            model, train_dataloader, optimizer, epoch, val_dataloader, best_val_loss_ref, training_state,
            websocket_manager, model_checkpoint_path, device, scaler, dynamic_save_interval, lock, current_data_manifest
        )
        if not should_continue:
            break
        
        await broadcast_state_update(
            manager=websocket_manager,
            model=model,
            is_training=True,
            message=f"Epoch {epoch} complete. Avg Loss: {training_state.train_loss:.4f}, Val Loss: {training_state.val_loss if training_state.val_loss != float('inf') else 'N/A'}",
            epoch=epoch
        )

        if best_val_loss_ref[0] <= model.emotions.contentment_threshold.item():
            logger.info(f"Early stopping at epoch {epoch} due to val loss reaching contentment threshold.")
            break

        await asyncio.sleep(0.1)

    logger.info("Training simulation complete.")
    training_state.is_training_active = False
    await broadcast_state_update(
        manager=websocket_manager,
        model=model,
        is_training=False,
        message="Training session finished."
    )