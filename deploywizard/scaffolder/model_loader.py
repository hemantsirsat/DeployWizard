import joblib
import torch
from pathlib import Path
from typing import Any, Dict, Optional
import os

class ModelLoader:
    def __init__(self):
        self._loaders = {
            'sklearn': self._load_sklearn,
            'pytorch': self._load_pytorch,
            'tensorflow': self._load_tensorflow
        }

    def load(self, model_path: str, framework: str) -> Any:
        """Load a model based on its framework.
        
        Args:
            model_path: Path to the model file
            framework: Framework of the model ('sklearn', 'pytorch', or 'tensorflow')
            
        Returns:
            Loaded model
            
        Raises:
            FileNotFoundError: If model file doesn't exist
            ValueError: If framework is not supported
            ImportError: If required dependencies are missing
        """
        # Convert to absolute path if not already
        model_path = str(Path(model_path).absolute())
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
            
        if framework not in self._loaders:
            raise ValueError(f"Unsupported framework: {framework}. Must be one of {list(self._loaders.keys())}")
        
        return self._loaders[framework](model_path)

    def _load_sklearn(self, model_path: str) -> Any:
        """Load a scikit-learn model."""
        try:
            return joblib.load(model_path)
        except ImportError:
            raise ImportError("scikit-learn is required for sklearn models")
        except Exception as e:
            raise RuntimeError(f"Failed to load scikit-learn model: {str(e)}")

    def _load_pytorch(self, model_path: str) -> Any:
        """Load a PyTorch model."""
        try:
            return torch.load(model_path)
        except ImportError:
            raise ImportError("PyTorch is required for pytorch models")
        except Exception as e:
            raise RuntimeError(f"Failed to load PyTorch model: {str(e)}")

    def _load_tensorflow(self, model_path: str) -> Any:
        """Load a TensorFlow model."""
        try:
            from tensorflow.keras.models import load_model
            return load_model(model_path)
        except ImportError:
            raise ImportError("TensorFlow is required for tensorflow models")
        except Exception as e:
            raise RuntimeError(f"Failed to load TensorFlow model: {str(e)}")
