# src/websocket_manager.py
import asyncio
import logging
from fastapi import WebSocket, WebSocketDisconnect
from typing import List, Any, Dict, Optional
import json
# Database Imports
from sqlalchemy.orm import Session
from sqlalchemy import text
from .database import SessionLocal

# Forward-referencing for type hints to avoid circular import
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .model_architecture import ContinuouslyReasoningPredictor

from .api_models import TrainingStatusResponse

logger = logging.getLogger(__name__)

class ConnectionManager:
    """Manages active WebSocket connections for broadcasting training updates."""
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected: {websocket.client}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected: {websocket.client}")

    async def broadcast(self, message: str):
        """Sends a text message to all active WebSocket connections."""
        logger.debug(f"Attempting to broadcast message to {len(self.active_connections)} connections.")
        for connection in list(self.active_connections):
            try:
                await connection.send_text(message)
            except (WebSocketDisconnect, RuntimeError) as e:
                logger.warning(f"WebSocket {connection.client} disconnected during broadcast. Removing. Error: {e}")
                self.disconnect(connection)
            except Exception as e:
                logger.error(f"Error broadcasting to WebSocket {connection.client}: {e}")
                self.disconnect(connection)

async def broadcast_state_update(
    manager: ConnectionManager,
    model: 'ContinuouslyReasoningPredictor',
    is_training: bool,
    message: str,
    epoch: int = 0,
    batch: int = 0,
    total_batches: int = 0,
    train_loss: float | None = None,
    val_loss: float | None = None,
    continuous_learning_loss: float | None = None
):
    """
    Gathers all cognitive metrics, saves them to the database,
    and broadcasts them to all WebSocket clients.
    """
    if model is None:
        return
    
    # --- 1. Gather All Metrics from the model and its sub-components ---
    metrics = {
        "focus": model.emotions.get_focus(),
        "confidence": model.latest_confidence.mean().item() if model.latest_confidence is not None else 0.0,
        "meta_error": model.latest_meta_error.mean().item() if model.latest_meta_error is not None else 0.0,
        "curiosity": model.emotions.get_curiosity(),
        "cognitive_stress": model.latest_heart_metrics.get("cognitive_stress"),
        "target_amplitude": model.latest_heart_metrics.get("target_amplitude"),
        "current_amplitude": model.emotions.focus_resonator.current_amplitude.item(),
        "target_frequency": model.latest_heart_metrics.get("target_frequency"),
        "current_frequency": model.emotions.focus_resonator.current_frequency.item(),
        "base_focus": model.emotions.base_focus.item(),
        "base_curiosity": model.emotions.base_curiosity.item(),
        "state_drift": model.latest_state_drift,
        "predictive_accuracy": model.latest_predictive_accuracy if model.latest_predictive_accuracy is not None else 0.0,
        "tm_sparsity": model.latest_tm_sparsity if model.latest_tm_sparsity is not None else 0.0,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "continuous_learning_loss": continuous_learning_loss,
    }

    # --- 2. Save Metrics to Database ---
    db: Session = SessionLocal()
    try:
        # Using keys from the metrics dictionary for robustness
        columns = ", ".join(metrics.keys())
        placeholders = ", ".join(f":{key}" for key in metrics.keys())
        stmt = text(f"""
            INSERT INTO cognitive_state_history (is_training, message, {columns})
            VALUES (:is_training, :message, {placeholders})
        """)
        
        # Combine static params with dynamic metrics
        params_to_save = {"is_training": is_training, "message": message, **metrics}

        db.execute(stmt, params_to_save)
        db.commit()
    except Exception as e:
        logger.error(f"Failed to save cognitive state to DB: {e}")
        db.rollback()
    finally:
        db.close()

    # --- 3. Broadcast Metrics to Frontend ---
    status_update_data = {
        "is_training_active": is_training,
        "message": message,
        "current_epoch": epoch,
        "current_batch": batch,
        "total_batches_in_epoch": total_batches,
        **metrics
    }
    status_update = TrainingStatusResponse(**status_update_data)
    await manager.broadcast(status_update.json())

async def broadcast_memory_state(manager: ConnectionManager, data: dict):
    """
    Packages and broadcasts the detailed memory state for visualization.
    """
    payload = {
        "type": "memory_state",
        "data": data
    }
    await manager.broadcast(json.dumps(payload))    