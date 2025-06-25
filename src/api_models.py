# src/api_models.py
from pydantic import BaseModel
from datetime import datetime
import torch # For type hinting in TrainingStatusResponse

# Pydantic Models for Chat Response
class GenerateRequest(BaseModel):
    prompt: str
    max_length: int = 256

class GenerateResponse(BaseModel):
    response: str
    confidence: float
    meta_error: float
    focus: int
    curiosity: float
    continuous_learning_loss: float | None = None

# Pydantic Model for Internal Thoughts
class InternalThoughtResponse(BaseModel):
    thought: str
    timestamp: datetime
    confidence: float
    meta_error: float
    focus: int
    curiosity: float
    prompt_text: str


# Pydantic Model for Training Status & Cognitive State Updates
class TrainingStatusResponse(BaseModel):
    is_training_active: bool
    message: str = "Status Update"
    
    # Core Metrics
    focus: float = 0.0
    confidence: float = 0.0
    meta_error: float = 0.0
    curiosity: float = 0.0
    
    # Training-specific Metrics
    current_epoch: int = 0
    current_batch: int = 0
    total_batches_in_epoch: int = 0
    train_loss: float | None = None # Changed to optional
    val_loss: float | None = None
    best_val_loss: float | None = None
    
    # New Advanced Metrics for Charting
    cognitive_stress: float | None = None
    target_amplitude: float | None = None
    current_amplitude: float | None = None
    target_frequency: float | None = None
    current_frequency: float | None = None
    base_focus: float | None = None
    base_curiosity: float | None = None
    state_drift: float | None = None
    predictive_accuracy: float | None = None
    tm_sparsity: float | None = None
    
    # New Loss Metric
    continuous_learning_loss: float | None = None