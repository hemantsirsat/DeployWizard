from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Union
import numpy as np
import os
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Get model path from environment variable or use a default for local development
MODEL_PATH = os.getenv("MODEL_PATH")

# Framework-specific imports and model loading
model: Any = None
model_loaded = False
model_error = None

# Define possible model paths to check
model_name = "{{ model_name }}"
possible_paths = [
    os.path.join(os.path.dirname(__file__), model_name),  # app/model.pkl
    os.path.join(os.path.dirname(__file__), "..", "model", model_name),  # ../model/model.pkl
    os.path.join(os.path.dirname(__file__), "model", model_name)  # app/model/model.pkl
]

# If MODEL_PATH is not set, try to find the model in standard locations
if not MODEL_PATH:
    # Find the first existing path
    for path in possible_paths:
        if os.path.exists(path):
            MODEL_PATH = path
            break
    else:
        # If no existing path found, use the first one and log a warning
        MODEL_PATH = possible_paths[0]
        logger.warning(f"Model file not found in any standard location, will try: {MODEL_PATH}")

# Convert to absolute path for better error messages
MODEL_PATH = os.path.abspath(MODEL_PATH)
logger.info(f"Looking for model at: {MODEL_PATH}")

# Check if model file exists
if not os.path.exists(MODEL_PATH):
    error_msg = f"Model file not found at {MODEL_PATH}. "
    if 'possible_paths' in locals():
        error_msg += f"Searched in: {', '.join(possible_paths)}"
    logger.error(error_msg)
    model_error = error_msg
    if os.getenv("ENV") == "production":
        raise FileNotFoundError(error_msg)

{% if framework == 'sklearn' %}
import joblib
try:
    model = joblib.load(MODEL_PATH)
    model_loaded = True
    logger.info(f"Successfully loaded model from {MODEL_PATH}")
except Exception as e:
    model_error = str(e)
    logger.error(f"Error loading model: {model_error}")
    if os.getenv("ENV") == "production":
        raise

{% elif framework == 'pytorch' %}
import torch
try:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load(MODEL_PATH, map_location=device)
    model = model.to(device)
    model.eval()
    model_loaded = True
    logger.info(f"Successfully loaded PyTorch model on {device}")
except Exception as e:
    model_error = str(e)
    logger.error(f"Error loading PyTorch model: {model_error}")
    if os.getenv("ENV") == "production":
        raise

{% elif framework == 'tensorflow' %}
import tensorflow as tf
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    model_loaded = True
    logger.info("Successfully loaded TensorFlow model")
except Exception as e:
    model_error = str(e)
    logger.error(f"Error loading TensorFlow model: {model_error}")
    if os.getenv("ENV") == "production":
        raise
{% endif %}

class Input(BaseModel):
    features: List[float]
    
    class Config:
        json_schema_extra = {
            "example": {
                "features": [5.1, 3.5, 1.4, 0.2]  # Example for Iris dataset
            }
        }

class Prediction(BaseModel):
    prediction: Union[int, float, List[float]]
    probabilities: List[float] = None

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model_loaded,
        "model_path": MODEL_PATH,
        "error": model_error if not model_loaded else None
    }

@app.post("/predict", response_model=Prediction)
async def predict(data: Input):
    """
    Make predictions using the loaded model.
    
    Args:
        data: Input data containing features for prediction
        
    Returns:
        Dictionary containing the prediction and optionally class probabilities
    """
    if not model_loaded:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "Model not loaded",
                "message": model_error or "Model failed to load",
                "model_path": MODEL_PATH
            }
        )
    
    try:
        # Convert input to numpy array and reshape if needed
        features = np.array(data.features).reshape(1, -1)
        
        # Make prediction based on framework
        {% if framework == 'sklearn' %}
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(features)[0].tolist()
            prediction = model.predict(features)[0]
            # Convert numpy types to native Python types for JSON serialization
            if hasattr(prediction, 'item'):
                prediction = prediction.item()
            return {"prediction": prediction, "probabilities": proba}
        else:
            prediction = model.predict(features)[0]
            if hasattr(prediction, 'item'):
                prediction = prediction.item()
            return {"prediction": prediction}
                
        {% elif framework == 'pytorch' %}
        with torch.no_grad():
            features_tensor = torch.tensor(features, dtype=torch.float32).to(device)
            output = model(features_tensor)
            if hasattr(model, 'predict_proba'):
                proba = torch.softmax(output, dim=1)[0].tolist()
                prediction = output.argmax().item()
                return {"prediction": prediction, "probabilities": proba}
            else:
                return {"prediction": output.item()}
                
        {% elif framework == 'tensorflow' %}
        prediction = model.predict(features, verbose=0)
        if len(prediction.shape) > 1 and prediction.shape[1] > 1:  # Classification with multiple classes
            proba = prediction[0].tolist()
            return {"prediction": int(np.argmax(prediction[0])), "probabilities": proba}
        else:  # Regression or binary classification
            return {"prediction": float(prediction[0][0])}
        {% endif %}
            
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Prediction failed",
                "message": str(e),
                "model_path": MODEL_PATH
            }
        )

# Add CORS middleware if needed
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
