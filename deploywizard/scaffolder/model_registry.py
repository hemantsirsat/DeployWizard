import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone

class ModelRegistry:
    """
    A simple model registry that stores model metadata in a JSON file.
    """
    def __init__(self, registry_path: str = None):
        """
        Initialize the model registry.
        
        Args:
            registry_path: Path to the registry JSON file. If None, checks DEPLOYWIZARD_REGISTRY 
                          environment variable, otherwise defaults to "registry.json"
        """
        # Use provided path, then check environment variable, then default
        path = registry_path or os.environ.get("DEPLOYWIZARD_REGISTRY", "registry.json")
        self.registry_path = Path(path).absolute()
        self._registry = self._load_registry()

    def _load_registry(self) -> Dict[str, Any]:
        """Load the registry from the JSON file or create a new one if it doesn't exist."""
        if self.registry_path.exists():
            try:
                with open(self.registry_path, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                # If the file is corrupted, create a new registry
                return {"models": {}, "next_id": 1}
        return {"models": {}, "next_id": 1}

    def _save_registry(self) -> None:
        """Save the registry to the JSON file."""
        # Create parent directories if they don't exist
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.registry_path, 'w') as f:
            json.dump(self._registry, f, indent=2)

    def register_model(self, name: str, version: str, path: str, framework: str, 
                      description: str = "") -> Dict[str, Any]:
        """
        Register a new model version in the registry.
        
        Args:
            name: Name of the model
            version: Version string (e.g., "1.0.0")
            path: Path to the model file
            framework: Framework used (e.g., "sklearn", "pytorch", "tensorflow")
            description: Optional description of the model
            
        Returns:
            Dictionary containing the registered model's metadata
            
        Raises:
            ValueError: If a model with the same name and version already exists
        """
        # Check if model with same name and version already exists
        if name in self._registry.get("models", {}) and \
           version in self._registry["models"][name]:
            raise ValueError(f"Model '{name}' version '{version}' already exists in registry")
            
        model_id = str(self._registry["next_id"])
        self._registry["next_id"] += 1
        
        model_data = {
            "id": model_id,
            "name": name,
            "version": version,
            "path": str(Path(path).absolute()),
            "framework": framework,
            "description": description,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "tags": []
        }
        
        # Initialize models dictionary if it doesn't exist
        if "models" not in self._registry:
            self._registry["models"] = {}
            
        # Add model to registry
        if name not in self._registry["models"]:
            self._registry["models"][name] = {}
            
        self._registry["models"][name][version] = model_data
        self._save_registry()
        
        return model_data

    def get_model(self, name: str, version: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get model metadata by name and optionally version.
        
        Args:
            name: Name of the model
            version: Optional version string. If None, returns the latest version.
            
        Returns:
            Dictionary containing model metadata or None if not found
        """
        if name not in self._registry.get("models", {}):
            return None
            
        versions = self._registry["models"][name]
        
        if not versions:
            return None
            
        if version is None:
            # Return the latest version (alphabetically highest version string)
            latest_version = sorted(versions.keys(), reverse=True)[0]
            return versions[latest_version]
            
        return versions.get(version)

    def list_models(self) -> List[Dict[str, Any]]:
        """
        List all models in the registry.
        
        Returns:
            List of model metadata dictionaries
        """
        result = []
        for model_versions in self._registry.get("models", {}).values():
            for version_data in model_versions.values():
                result.append(version_data)
        return result
    
    def delete_model(self, name: str, version: Optional[str] = None) -> bool:
        """
        Delete a model or model version from the registry.
        
        Args:
            name: Name of the model
            version: Optional version string. If None, deletes all versions.
            
        Returns:
            True if any models were deleted, False otherwise
        """
        if name not in self._registry.get("models", {}):
            return False
            
        if version is None:
            # Delete all versions of the model
            del self._registry["models"][name]
            self._save_registry()
            return True
            
        if version in self._registry["models"][name]:
            del self._registry["models"][name][version]
            self._save_registry()
            return True
            
        return False
