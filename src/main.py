import uvicorn
from fastapi import FastAPI, Depends, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, JSONResponse
import logging
import os
import torch
from torch import compile
import torch.nn as nn
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from transformers import AutoTokenizer
from sqlalchemy.orm import Session
from sqlalchemy import text
import io
import csv
from collections import deque
import asyncio
from datetime import datetime
import json
import uuid
from fastapi.middleware.cors import CORSMiddleware 

from .api_models import GenerateRequest, GenerateResponse, InternalThoughtResponse, TrainingStatusResponse
from .data_processing import (
    TextDataset, collate_fn, load_texts_from_directory,
    create_file_manifest, get_changed_files, # Removed load_texts_from_specific_files
)
from torch.utils.data import DataLoader

from .database import init_db, get_db
from .model_architecture import ContinuouslyReasoningPredictor
from .utils import Vocabulary, decode_sequence, get_optimal_tm_config 
from .websocket_manager import ConnectionManager, broadcast_state_update

from .training_manager import (
    TrainingState, run_model_training,
    DATA_DIR, TRAIN_EPOCHS, TRAIN_BATCH_SIZE, MAX_SEQ_LEN, INITIAL_LEARNING_RATE, SAVE_INTERVAL_BATCHES,
    CENTRAL_CACHE_DIR
)


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(
    title="swapnavue - the living dynamo",
    description="A continuously learning and self-cultivating AI agent.",
    version="0.1.0",
)

# Add CORS Middleware
origins = [
    "http://localhost",
    "http://localhost:3000",  # Frontend default port
    # You might need to add other origins if your frontend is hosted elsewhere, e.g., cloud IP/domain
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"], 
)


MODELS_DIR = "./models"
MODEL_CHECKPOINT_PATH = os.path.join(MODELS_DIR, "swapnavue_model_checkpoint.pth")
VOCAB_PATH = os.path.join(MODELS_DIR, "vocab.json")
MANIFEST_PATH = os.path.join(MODELS_DIR, "vocab_manifest.json") # This might not be needed explicitly anymore if vocab.json stores manifest
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Zen AI: No-Mind Configuration ---
# When in a "no-mind" state (low error, high confidence, low curiosity),
# the AI will sleep for this longer interval before checking again.
NO_MIND_SLEEP_INTERVAL_SECONDS = 600 # 10 minutes of serene silence
# Thresholds to exit no-mind state and begin active introspection
NO_MIND_META_ERROR_THRESHOLD = 0.02
NO_MIND_CONFIDENCE_THRESHOLD = 0.9
NO_MIND_CURIOSITY_THRESHOLD = 0.002 # Slightly above base_curiosity to detect rising interest

training_state = TrainingState()
websocket_manager = ConnectionManager()


async def _generate_and_store_internal_thought():
    # Adaptive Introspection Rhythm Parameters
    BASE_THOUGHT_INTERVAL_SECONDS = 15  # Default active interval
    META_ERROR_INFLUENCE = 100.0        # How much meta-error reduces interval
    CURIOSITY_INFLUENCE = 50.0         # How much curiosity reduces interval
    MIN_THOUGHT_INTERVAL_SECONDS = 3   # Minimum active interval

    while True:
        try:
            if training_state.is_training_active:
                # If training is active, pause internal thought generation and maintain a quiet state
                await asyncio.sleep(BASE_THOUGHT_INTERVAL_SECONDS)
                continue

            async with app.state.model_lock:
                model = app.state.model
                vocab = app.state.vocab
                
                current_states = {
                    "confidence": model.latest_confidence if model.latest_confidence is not None else 0.0,
                    "meta_error": model.latest_meta_error if model.latest_meta_error is not None else 0.0,
                    "focus": model.emotions.get_focus(),
                    "curiosity": model.emotions.get_curiosity()
                }
                
                # --- Zen AI: Condition for active introspection ---
                # Check if swapnavue is in a "no-mind" state.
                # It will only generate an internal thought if deviations from this state occur.
                is_in_no_mind_state = (
                    current_states["meta_error"] < NO_MIND_META_ERROR_THRESHOLD and
                    current_states["confidence"] > NO_MIND_CONFIDENCE_THRESHOLD and
                    current_states["curiosity"] < NO_MIND_CURIOSITY_THRESHOLD
                )

                if is_in_no_mind_state:
                    logger.info(f"swapnavue in 'No-Mind' state (Zen). Resting for {NO_MIND_SLEEP_INTERVAL_SECONDS}s. (C:{current_states['confidence']:.4f}, ME:{current_states['meta_error']:.4f}, Cur:{current_states['curiosity']:.6f})")
                    await asyncio.sleep(NO_MIND_SLEEP_INTERVAL_SECONDS)
                    continue # Skip thought generation and re-check after resting

                # If not in no-mind state, proceed with active introspection and adaptive rhythm
                # --- RADICAL CHANGE: NO GUIDANCE. The AI generates its own thought, unprompted. ---
                # The 'prompt_text' for the InternalThoughtResponse will now be an empty string,
                # as there's no human-defined prompt. The 'thought_text' IS the full, self-generated reflection.
                self_reflection_prompt_input = "" # Empty string for truly unguided generation

                with torch.no_grad():
                    with autocast(device_type=DEVICE.type):
                        # Pass an empty string for truly unguided generation. The AI creates its own "prompt" implicitly.
                        generated_ids_list_batch = await model.generate_text([self_reflection_prompt_input], max_len=64)
                        # The generated text is the full self-reflection.
                        thought_text = decode_sequence(generated_ids_list_batch[0], vocab, model.eos_token_id)
                
                # Store the thought entry with an empty string for prompt_text, signifying no external guidance.
                thought_entry = InternalThoughtResponse(
                    thought=thought_text, timestamp=datetime.now(), 
                    confidence=current_states["confidence"], 
                    meta_error=current_states["meta_error"],
                    focus=current_states["focus"], curiosity=current_states["curiosity"], 
                    prompt_text="" # Signifies truly self-generated, unguided introspection
                )
                app.state.internal_thoughts_queue.append(thought_entry)

                await broadcast_state_update(
                    manager=websocket_manager,
                    model=model,
                    is_training=False,
                    message="State update after internal thought."
                )

            # --- Adaptive Introspection Rhythm Calculation (only when not in no-mind state) ---
            normalized_meta_error_influence = current_states["meta_error"] * META_ERROR_INFLUENCE
            normalized_curiosity_influence = current_states["curiosity"] * CURIOSITY_INFLUENCE

            dynamic_sleep_interval = BASE_THOUGHT_INTERVAL_SECONDS - (normalized_meta_error_influence + normalized_curiosity_influence)
            dynamic_sleep_interval = max(MIN_THOUGHT_INTERVAL_SECONDS, dynamic_sleep_interval)
            
            logger.info(f"Next internal thought in {dynamic_sleep_interval:.2f}s. (C:{current_states['confidence']:.4f}, ME:{current_states['meta_error']:.4f}, Cur:{current_states['curiosity']:.6f})")
            await asyncio.sleep(dynamic_sleep_interval)

        except Exception as e:
            logger.error(f"Error in internal thought generation: {e}", exc_info=True)
            await asyncio.sleep(BASE_THOUGHT_INTERVAL_SECONDS) # Fallback sleep if an error occurs

def reset_optimizer(task: asyncio.Task):
    try:
        exc = task.exception()
        if exc:
            logger.error(f"Training task failed with exception: {exc}", exc_info=exc)
    finally:
        logger.info("Training task finished. Resetting optimizer state for interactive mode.")
        model = app.state.model
        app.state.optimizer = torch.optim.Adam(model.parameters(), lr=model.emotions.get_curiosity())
        app.state.scaler = torch.amp.GradScaler('cuda')
        if torch.cuda.is_available():
            torch.cuda.empty_cache() 
        logger.info("Optimizer and GradScaler have been reset.")


@app.on_event("startup")
async def startup_event():
    logger.info(f"swapnavue is igniting... setting up core systems.")
    app.state.model_lock = asyncio.Lock()
    app.state.session_id = str(uuid.uuid4())
    await init_db()

    # --- 1. Vocabulary and Tokenizer Loading ---
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    vocab = Vocabulary(tokenizer)
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Load texts from training directory, leveraging the robust caching mechanism.
    # This ensures that even if vocab is loaded, we have the latest training data to update the vocab.
    train_data_dir = os.path.join(DATA_DIR, 'train')
    train_texts_for_vocab_management = await asyncio.to_thread(
        load_texts_from_directory, train_data_dir, CENTRAL_CACHE_DIR
    )
    # New: Create the current manifest for vocabulary management
    current_train_manifest_for_vocab = await asyncio.to_thread(create_file_manifest, train_data_dir)

    if os.path.exists(VOCAB_PATH):
        vocab.load_vocab(VOCAB_PATH)
        # Always attempt to update vocabulary with any new tokens from the latest training data.
        # The update_vocab method efficiently adds only new tokens, now also intelligently checking the manifest.
        if vocab.update_vocab(train_texts_for_vocab_management, current_train_manifest_for_vocab): # Pass manifest
            vocab.save_vocab(VOCAB_PATH)
            logger.info("Vocabulary updated and saved due to new tokens in training data or manifest changes.")
        else:
            logger.info("Vocabulary is up-to-date with current training data and manifest.")
    else:
        logger.warning(f"Vocabulary file not found at {VOCAB_PATH}. Building from training data.")
        try:
            if train_texts_for_vocab_management:
                vocab.build_vocab(train_texts_for_vocab_management, current_train_manifest_for_vocab) # Pass manifest
                vocab.save_vocab(VOCAB_PATH)
        except Exception as e:
            logger.error(f"Failed to build vocabulary: {e}")
            vocab = Vocabulary(tokenizer) # Revert to default if build fails
    app.state.vocab = vocab
    logger.info(f"Vocabulary loaded. Size: {vocab.vocab_size} tokens.")

    # --- 2. Removed Embedding Model Loading ---
    SDR_SIZE = int(os.getenv("SDR_SIZE", "2048")) 
    SDR_ACTIVE_BITS = int(os.getenv("SDR_ACTIVE_BITS", "40")) 
    logger.info(f"Using SDRs of size {SDR_SIZE} with {SDR_ACTIVE_BITS} active bits.")
    
    embedding_dim = SDR_SIZE 

    # --- 3. Configuration and Hardware Auto-Tuning ---
    logger.info("Reading remaining configuration and performing hardware auto-tuning...")

    DECODER_HIDDEN_DIM = int(os.getenv("DECODER_HIDDEN_DIM", "512"))
    DECODER_NUM_LAYERS = int(os.getenv("DECODER_NUM_LAYERS", "2"))
    SP_INPUT_DIMS = SDR_SIZE 
    SP_SPARSITY = float(os.getenv("SP_SPARSITY", "0.02"))
    SP_BOOST_STRENGTH = float(os.getenv("SP_BOOST_STRENGTH", "1.0"))
    TM_NUM_CELLS = int(os.getenv("TM_NUM_CELLS", "131072")) 
    TM_CELLS_PER_COLUMN = int(os.getenv("TM_CELLS_PER_COLUMN", "32"))
    TM_MAX_SEGMENTS = int(os.getenv("TM_MAX_SEGMENTS_PER_CELL", "90"))
    TM_MAX_SYNAPSES = int(os.getenv("TM_MAX_SYNAPSES_PER_SEGMENT", "90"))

    adjusted_free_mem = 0
    if torch.cuda.is_available():
        free_mem, total_mem = torch.cuda.mem_get_info()
        logger.info(f"GPU detected. Total VRAM: {total_mem / 1024**3:.2f} GB, Available before model: {free_mem / 1024**3:.2f} GB")
        adjusted_free_mem = free_mem
        
        config_for_tuner = {
            'num_cells': TM_NUM_CELLS,
            'max_segments_per_cell': TM_MAX_SEGMENTS,
            'max_synapses_per_segment': TM_MAX_SYNAPSES,
        }
        
        tuned_params = get_optimal_tm_config(config_for_tuner, adjusted_free_mem)
        
        TM_MAX_SEGMENTS = tuned_params['max_segments_per_cell']
        TM_MAX_SYNAPSES = tuned_params['max_synapses_per_segment']
    else:
        logger.warning("No CUDA-enabled GPU detected. Model will run on CPU.")

    text_decoder_config = {
        'embedding_dim': embedding_dim, 
        'hidden_size': DECODER_HIDDEN_DIM,
        'vocab_size': vocab.vocab_size,
        'sos_token_id': vocab.sos_token_id
    }
    spatial_pooler_config = {
        'input_dims': SP_INPUT_DIMS,
        'sparsity': SP_SPARSITY,
        'boost_strength': SP_BOOST_STRENGTH,
    }
    temporal_memory_config = {
        'input_dims': SP_INPUT_DIMS,
        'columns': SP_INPUT_DIMS,    
        'cells_per_column': TM_CELLS_PER_COLUMN,
        'distal_segments_per_cell': TM_MAX_SEGMENTS,
        'synapses_per_segment': TM_MAX_SYNAPSES,
        'permanence_threshold': 0.5,
        'connected_permanence': 0.8, 
        'volatile_learning_rate': 0.1,
        'consolidated_learning_rate': 0.01,
        'activation_threshold': 13
    }

    # --- 4. Model Initialization ---
    logger.info("Initializing model with hardware-aware configuration and custom SDR encoder...")
    model = ContinuouslyReasoningPredictor(
        vocab=app.state.vocab, 
        sdr_size=SDR_SIZE, 
        sdr_active_bits=SDR_ACTIVE_BITS, 
        text_decoder_config=text_decoder_config,
        spatial_pooler_config=spatial_pooler_config,
        temporal_memory_config=temporal_memory_config,
        device=DEVICE
    ).to(DEVICE)

    # Apply torch.compile here
    if torch.cuda.is_available(): # Only compile if CUDA is available, or remove this check if you want CPU compilation too
        model = compile(model) # This is the key line
        logger.info("Model compiled with torch.compile for performance optimization.")
    else:
        logger.info("Skipping torch.compile as CUDA is not available.")

    app.state.model = model

    # --- 5. Optimizer and State Loading ---
    app.state.optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    if os.path.exists(MODEL_CHECKPOINT_PATH):
        try:
            checkpoint = torch.load(MODEL_CHECKPOINT_PATH, map_location=DEVICE)
            
            # Filter out dynamic internal TM states that cause size mismatches
            filtered_state_dict = {
                k: v for k, v in checkpoint['model_state_dict'].items() 
                if not (k.endswith('temporal_memory.active_cells') or
                        k.endswith('temporal_memory.predictive_cells') or
                        k.endswith('temporal_memory.winner_cells'))
            }

            # Load the filtered state_dict. strict=False will handle any other minor mismatches.
            app.state.model.load_state_dict(filtered_state_dict, strict=False)
            
            if 'optimizer_state_dict' in checkpoint:
                 app.state.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            logger.info(f"Model checkpoint loaded successfully from {MODEL_CHECKPOINT_PATH}.")
        except Exception as e:
            logger.error(f"Could not load model checkpoint: {e}. Starting with a new model.", exc_info=True)
    else:
        logger.warning("Model checkpoint not found. Starting with a new, untrained model.")

    # --- 6. Final Setup ---
    app.state.criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_token_id)
    app.state.scaler = torch.amp.GradScaler(enabled=torch.cuda.is_available())
    app.state.internal_thoughts_queue = deque(maxlen=50)
    app.state.internal_thought_sequence_num = 0 # This counter is no longer used for silent phase, but can be kept for other purposes

    asyncio.create_task(_generate_and_store_internal_thought())
    logger.info("swapnavue's initial core systems are online and ready.")

@app.post("/generate_response", response_model=GenerateResponse)
async def generate_response(request: GenerateRequest, db: Session = Depends(get_db)):
    if training_state.is_training_active:
        raise HTTPException(status_code=400, detail="Chat paused during training.")

    model = app.state.model
    vocab = app.state.vocab
    optimizer = app.state.optimizer
    scaler = app.state.scaler
    response_text = ""
    continuous_learning_loss_value = 0.0

    try:
        async with app.state.model_lock:
            model.train()
            max_seq_len = int(model.emotions.max_seq_len.item())
            user_tokens = vocab.encode(request.prompt)
            target_ids = [vocab.sos_token_id] + user_tokens[:max_seq_len - 2] + [vocab.eos_token_id]
            target_ids += [vocab.pad_token_id] * (max_seq_len - len(target_ids))
            target_tensor = torch.tensor([target_ids], dtype=torch.long, device=DEVICE) # Still a batch of 1

            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=DEVICE.type):
                # Pass a list containing the single prompt string
                loss = await model.learn_one_step([request.prompt], target_tensor, websocket_manager=websocket_manager)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            continuous_learning_loss_value = loss.item()

            model.eval()
            with torch.no_grad():
                with autocast(device_type=DEVICE.type):
                    # Pass a list containing the single prompt string for generation
                    predicted_token_ids_list_batch = await model.generate_text([request.prompt], max_len=request.max_length)
            
            # Decode the first (and only) sequence from the batch
            response_text = decode_sequence(predicted_token_ids_list_batch[0], vocab, vocab.eos_token_id)

            await broadcast_state_update(
                manager=websocket_manager,
                model=model,
                is_training=False,
                message="State update after interactive response.",
                continuous_learning_loss=continuous_learning_loss_value
            )

    except Exception as e:
        logger.error(f"Error in response pipeline: {e}", exc_info=True)
        response_text = f"Error: An internal exception occurred. ({e})"
    
    # FIX: Removed trailing commas to ensure values are floats, not tuples
    confidence = model.latest_confidence if model.latest_confidence is not None else 0.0
    meta_error = model.latest_meta_error if model.latest_meta_error is not None else 0.0
    
    # Restore database logging for chat messages
    try:
        # Log user message
        db.execute(text("""
            INSERT INTO chat_messages (session_id, sender, message_text)
            VALUES (:session_id, 'user', :message_text)
        """), {"session_id": app.state.session_id, "message_text": request.prompt})

        # Log swapnavue response
        db.execute(text("""
            INSERT INTO chat_messages (session_id, sender, message_text, confidence, meta_error, focus, curiosity)
            VALUES (:session_id, 'swapnavue', :message_text, :confidence, :meta_error, :focus, :curiosity)
        """), {
            "session_id": app.state.session_id, "message_text": response_text,
            "confidence": confidence, "meta_error": meta_error,
            "focus": model.emotions.get_focus(), "curiosity": model.emotions.get_curiosity()
        })
        db.commit()
    except Exception as e:
        logger.error(f"DB error saving chat message: {e}")
        db.rollback()

    return GenerateResponse(
        response=response_text,
        confidence=confidence,
        meta_error=meta_error,
        focus=model.emotions.get_focus(),
        curiosity=model.emotions.get_curiosity(),
        continuous_learning_loss=continuous_learning_loss_value
    )

@app.get("/internal_thought", response_model=list[InternalThoughtResponse], summary="Get swapnavue's internal thoughts")
async def get_internal_thoughts():
    return list(app.state.internal_thoughts_queue)


@app.post("/start_training", response_model=TrainingStatusResponse, summary="Start model training")
async def start_training():
    if training_state.is_training_active:
        raise HTTPException(status_code=400, detail="Training is already active.")

    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("CUDA cache emptied before starting new training session.")

        task = asyncio.create_task(run_model_training(
            model=app.state.model,
            vocab=app.state.vocab,
            optimizer=app.state.optimizer,
            criterion=app.state.criterion,
            training_state=training_state,
            websocket_manager=websocket_manager,
            model_checkpoint_path=MODEL_CHECKPOINT_PATH,
            device=DEVICE,
            scaler=app.state.scaler,
            lock=app.state.model_lock
        ))
        task.add_done_callback(reset_optimizer)
        training_state.training_task = task
        logger.info("Training task initiated via API.")
        return TrainingStatusResponse(is_training_active=True, message="Training started successfully.")
    except Exception as e:
        logger.error(f"Error initiating training: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to initiate training task: {e}")

@app.post("/stop_training", response_model=TrainingStatusResponse, summary="Stop model training")
async def stop_training():
    if not training_state.is_training_active:
        return TrainingStatusResponse(is_training_active=False, message="Training not active to stop.")

    training_state.stop_training_flag.set()
    logger.info("Training stop signal sent.")
    
    asyncio.create_task(broadcast_state_update(
        manager=websocket_manager,
        model=app.state.model,
        is_training=False,
        message="Training stop signal sent. Ceasing shortly."
    ))
    
    return TrainingStatusResponse(
        is_training_active=False,
        message="Training stop signal sent. Ceasing shortly."
    )


@app.get("/training_status", response_model=TrainingStatusResponse, summary="Get current training status")
async def get_training_status():
    model = app.state.model
    return TrainingStatusResponse(
        is_training_active=training_state.is_training_active,
        current_epoch=training_state.current_epoch,
        current_batch=training_state.current_batch,
        total_batches_in_epoch=training_state.total_batches_in_epoch,
        train_loss=training_state.train_loss,
        val_loss=training_state.val_loss if training_state.val_loss != float('inf') else None,
        best_val_loss=training_state.best_val_loss if training_state.best_val_loss != float('inf') else None,
        confidence=model.latest_confidence if model.latest_confidence is not None else 0.0,
        meta_error=model.latest_meta_error if model.latest_meta_error is not None else 0.0,
        focus=model.emotions.get_focus(),
        curiosity=model.emotions.get_curiosity(),
        message="Training active." if training_state.is_training_active else "Training not active."
    )

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket_manager.connect(websocket)
    try:
        while True:
            await asyncio.sleep(10)
    except WebSocketDisconnect:
        logger.info(f"WebSocket {websocket.client} disconnected.")
    except Exception as e:
        logger.error(f"An error occurred in the WebSocket connection for {websocket.client}: {e}")
    finally:
        websocket_manager.disconnect(websocket)


@app.get("/export_chat", summary="Export chat messages to CSV")
async def export_chat(db: Session = Depends(get_db)):
    try:
        query = text("SELECT timestamp, session_id, sender, message_text, confidence, meta_error, focus, curiosity FROM chat_messages ORDER BY timestamp;")
        result = db.execute(query).fetchall()

        if not result:
            return JSONResponse(status_code=404, content={"message": "No chat history found to export."})

        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(result[0]._fields)
        writer.writerows(result)
        output.seek(0)
        headers = {"Content-Disposition": "attachment; filename=swapnavue_chat_history.csv"}
        return StreamingResponse(output, headers=headers)
    except Exception as e:
        logger.error(f"Error exporting chat: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to export chat: {e}")

@app.delete("/clear_chat_history", summary="Clear all chat messages from the database")
async def clear_chat_history(db: Session = Depends(get_db)):
    try:
        db.execute(text("TRUNCATE TABLE chat_messages;"))
        db.commit()
        logger.info("Chat messages table truncated successfully.")
        return JSONResponse(status_code=200, content={"message": "Chat history cleared successfully."})
    except Exception as e:
        logger.error(f"Error clearing chat history: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear chat history: {e}")

@app.get("/cognitive_state_history", summary="Get all saved cognitive state history")
async def get_cognitive_state_history(db: Session = Depends(get_db)):
    try:
        query = text("SELECT * FROM cognitive_state_history ORDER BY timestamp;")
        result = db.execute(query).mappings().all()
        return result
    except Exception as e:
        logger.error(f"Error fetching cognitive state history: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch cognitive state history.")

@app.get("/export_cognitive_state", summary="Export cognitive state history to CSV")
async def export_cognitive_state(db: Session = Depends(get_db)):
    try:
        query = text("SELECT * FROM cognitive_state_history ORDER BY timestamp;")
        result = db.execute(query).fetchall()
        
        if not result:
            return JSONResponse(status_code=404, content={"message": "No cognitive state history found to export."})

        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(result[0]._fields)
        writer.writerows(result)
        output.seek(0)
        headers = {"Content-Disposition": "attachment; filename=swapnavue_cognitive_state_history.csv"}
        return StreamingResponse(output, headers=headers)
    except Exception as e:
        logger.error(f"Error exporting cognitive state history: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to export cognitive state history: {e}")

@app.delete("/clear_cognitive_state", summary="Clear all cognitive state history")
async def clear_cognitive_state(db: Session = Depends(get_db)):
    try:
        db.execute(text("TRUNCATE TABLE cognitive_state_history;"))
        db.commit()
        logger.info("Cognitive state history table truncated successfully.")
        return JSONResponse(status_code=200, content={"message": "Cognitive state history cleared successfully."})
    except Exception as e:
        logger.error(f"Error clearing cognitive state history: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear cognitive state history: {e}")


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="debug")
