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
import os
import importlib.util
import sys
from pathlib import Path

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

# Try to load the model
try:
    model = None
    model_loaded = False
    model_error = None
    
    # Debug: Print model path and check if file exists
    logger.info(f"Looking for model at: {MODEL_PATH}")
    logger.info(f"File exists: {os.path.exists(MODEL_PATH)}")
    if os.path.exists(MODEL_PATH):
        logger.info(f"File size: {os.path.getsize(MODEL_PATH) / (1024*1024):.2f} MB")
    
    # Strategy 1: Try loading as a full PyTorch model
    try:
        logger.info("Attempting to load as full PyTorch model...")
        model = torch.load(MODEL_PATH, map_location=device)
        if isinstance(model, torch.nn.Module):
            model = model.to(device)
            model.eval()
            model_loaded = True
            logger.info("Successfully loaded as a full PyTorch model")
    except Exception as e:
        logger.warning(f"Failed to load as full model: {str(e)}")
    
    # Strategy 2: Try loading with model class from model.py
    if not model_loaded:
        try:
            logger.info("Attempting to load with model class...")
            model_dir = Path(MODEL_PATH).parent
            model_file = model_dir / "model.py"
            
            logger.info(f"Looking for model class in: {model_file}")
            
            if model_file.exists():
                # Import the model class
                spec = importlib.util.spec_from_file_location("model", str(model_file))
                model_module = importlib.util.module_from_spec(spec)
                sys.modules["model"] = model_module
                spec.loader.exec_module(model_module)
                
                # Find the model class (look for a class that inherits from nn.Module)
                model_class = None
                for name, obj in model_module.__dict__.items():
                    if (isinstance(obj, type) and 
                        issubclass(obj, torch.nn.Module) and 
                        obj != torch.nn.Module):
                        model_class = obj
                        break
                
                if model_class is None:
                    raise ValueError("No PyTorch model class found in model.py")
                
                logger.info(f"Found model class: {model_class.__name__}")
                
                # Load the model state
                checkpoint = torch.load(MODEL_PATH, map_location=device)
                
                # Handle different checkpoint formats
                if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                    # Create model with parameters from checkpoint if available
                    if hasattr(checkpoint, 'input_size') and hasattr(checkpoint, 'output_size'):
                        model = model_class(
                            input_size=checkpoint['input_size'],
                            output_size=checkpoint['output_size']
                        ).to(device)
                    else:
                        model = model_class().to(device)
                    
                    model.load_state_dict(state_dict)
                    model.eval()
                    model_loaded = True
                    logger.info("Successfully loaded model using provided model class and state dict")
                else:
                    # Assume the file is just the state dict
                    model = model_class().to(device)
                    model.load_state_dict(checkpoint)
                    model.eval()
                    model_loaded = True
                    logger.info("Successfully loaded model using provided model class with direct state dict")
            
        except Exception as e:
            logger.warning(f"Failed to load with model class: {str(e)}", exc_info=True)
    
    if not model_loaded:
        raise ValueError(
            "Failed to load model. Please ensure:\n"
            "1. The model file is a valid PyTorch model or state dictionary\n"
            "2. A model.py file with the model class is in the same directory\n"
            f"Error details: {str(e) if 'e' in locals() else 'Unknown error'}"
        )

except Exception as e:
    model_error = str(e)
    logger.error(f"Error loading PyTorch model: {model_error}", exc_info=True)
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
            # Convert input to tensor
            features_tensor = torch.tensor(features, dtype=torch.float32)
            
            # Check if we have a state_dict that needs model initialization
            if isinstance(model, dict) and 'state_dict' in model:
                raise HTTPException(
                    status_code=500,
                    detail={
                        "error": "Model initialization required",
                        "message": (
                            "The model was loaded as a state dictionary. "
                            "Please initialize the model class and load the state_dict before making predictions.\n"
                            "Example:\n"
                            "```python\n"
                            "from model import SimpleTorchModel  # Import your model class\n"
                            "model = SimpleTorchModel()  # Initialize model\n"
                            "model.load_state_dict(torch.load('path/to/model.pt'))  # Load weights\n"
                            "model.eval()  # Set to evaluation mode\n"
                            "```"
                        ),
                        "model_path": MODEL_PATH
                    }
                )
            
            # Move input to the same device as the model
            if hasattr(model, 'parameters') and next(model.parameters()).is_cuda:
                features_tensor = features_tensor.cuda()
            
            # Get model prediction
            output = model(features_tensor)
            
            # Handle different output types
            if isinstance(output, (list, tuple)):
                output = output[0]  # Take first output if model returns multiple values
                
            # Convert to numpy for easier handling
            if hasattr(output, 'cpu'):
                output = output.cpu()
            if hasattr(output, 'numpy'):
                output = output.numpy()
            
            # Handle different output shapes
            output = np.squeeze(output)
            
            # For binary classification, return probabilities for both classes
            if output.size == 1:  # Binary classification
                prob = float(output)
                return {
                    "prediction": round(prob),  # Class prediction (0 or 1)
                    "probabilities": [1 - prob, prob]  # [P(class=0), P(class=1)]
                }
            else:  # Multi-class
                # Apply softmax to get probabilities
                exp_scores = np.exp(output - np.max(output))  # For numerical stability
                probabilities = exp_scores / exp_scores.sum()
                predicted_class = int(np.argmax(probabilities))
                return {
                    "prediction": predicted_class,
                    "probabilities": probabilities.tolist()
                }
                
        {% elif framework == 'tensorflow' %}
        prediction = model.predict(features, verbose=0)
        if len(prediction.shape) > 1:
            if prediction.shape[1] > 1:  # Multi-class classification
                proba = prediction[0].tolist()
                return {"prediction": int(np.argmax(prediction[0])), "probabilities": proba}
            else:  # Binary classification
                prob = float(prediction[0][0])
                return {
                    "prediction": 1 if prob >= 0.5 else 0,
                    "probabilities": [1 - prob, prob]  # [prob_class_0, prob_class_1]
                }
        else:  # Regression
            return {"prediction": float(prediction[0]), "probabilities": None}
        {% endif %}
            
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
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
