from pathlib import Path
from typing import Dict, Optional, Union, List, Any
import shutil
import os
from datetime import datetime

from .model_loader import ModelLoader
from .api_generator import APIGenerator
from .docker_generator import DockerGenerator
from .model_registry import ModelRegistry
from .template_utils import get_template_vars

class Scaffolder:
    def __init__(self, registry_path: str = None):
        """
        Initialize the Scaffolder.
        
        Args:
            registry_path: Path to the registry JSON file. If None, checks DEPLOYWIZARD_REGISTRY 
                          environment variable, otherwise defaults to "registry.json"
        """
        # Use provided path, then check environment variable, then default
        path = registry_path or os.environ.get("DEPLOYWIZARD_REGISTRY", "registry.json")
        self._model_loader = ModelLoader()
        self._api_generator = APIGenerator()
        self._docker_generator = DockerGenerator()
        self._registry = ModelRegistry(registry_path=path)

    def register_model(self, name: str, version: str, model_path: str, 
                     framework: str, description: str = "") -> Dict[str, Any]:
        """
        Register a model in the model registry.
        
        Args:
            name: Name of the model
            version: Version string (e.g., "1.0.0")
            model_path: Path to the model file
            framework: Framework used (e.g., "sklearn", "pytorch", "tensorflow")
            description: Optional description of the model
            
        Returns:
            Dictionary containing the registered model's metadata
        """
        # Validate the model can be loaded
        try:
            model = self._model_loader.load(model_path, framework)
            print("[SUCCESS] Model loaded successfully:", name, "v", version)
        except Exception as e:
            print("[ERROR] Failed to load model:", str(e))
            raise
            
        # Register the model
        return self._registry.register_model(
            name=name,
            version=version,
            path=model_path,
            framework=framework,
            description=description
        )

    def get_model_info(self, name: str, version: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get model metadata from the registry.
        
        Args:
            name: Name of the model
            version: Optional version string. If None, gets the latest version.
            
        Returns:
            Dictionary containing model metadata or None if not found
        """
        return self._registry.get_model(name, version)

    def list_models(self) -> List[Dict[str, Any]]:
        """
        List all registered models.
        
        Returns:
            List of model metadata dictionaries
        """
        return self._registry.list_models()

    def generate_project(self, model_name: str, version: Optional[str] = None, 
                       output_dir: str = ".", api_type: str = "fastapi") -> None:
        """
        Generate a deployment project for a registered model.
        
        Args:
            model_name: Name of the registered model
            version: Optional version string. If None, uses the latest version.
            output_dir: Directory to generate the project in
            api_type: Type of API to generate (e.g., "fastapi", "flask")
        """
        # Get model info from registry
        model_info = self.get_model_info(model_name, version)
        if not model_info:
            raise ValueError(f"Model {model_name} (version: {version or 'latest'}) not found in registry")
            
        model_path = model_info["path"]
        framework = model_info["framework"]
        version = model_info["version"]
        
        print("Generating project for", model_name, "v", version)
        print("Output directory:", output_dir)
        
        # Create output directory structure
        output_path = Path(output_dir).absolute()
        app_dir = output_path / "app"
        app_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy model file to app directory
        model_file_name = Path(model_path).name
        model_output_path = app_dir / model_file_name
        
        try:
            shutil.copy2(model_path, model_output_path)
            print("[SUCCESS] Model file copied to", model_output_path)
            
            # Get template variables
            template_vars = get_template_vars(str(model_output_path), framework)
            template_vars['model_name'] = model_file_name
            template_vars['model_info'] = model_info  # Pass full model info to templates
            
            # Generate API code - pass the parent directory, not the app_dir
            self._api_generator.generate(
                model_path=str(model_output_path),
                framework=framework,
                output_dir=str(output_path),  # Pass the parent directory here
                api_type=api_type,
                template_vars=template_vars
            )
            
            # Generate Docker configuration
            self._docker_generator.generate(
                output_dir=str(output_path),
                template_vars=template_vars
            )
            
            print("[SUCCESS] Project generated successfully in", output_dir)
            
        except Exception as e:
            print("[ERROR] Failed to generate project:", str(e))
            # Clean up partial files if something went wrong
            if output_path.exists():
                shutil.rmtree(output_path)
            raise

    def _generate_readme(self, output_dir: str) -> None:
        """Generate a basic README file for the project."""
        readme_path = Path(output_dir) / "README.md"
        content = """# ML Model Deployment

This project contains the deployment code for your machine learning model.

## Running Locally

```bash
uvicorn app.main:app --reload
```

## Building Docker Image

```bash
docker build -t model-api .
```

## Running Docker Container

```bash
docker run -p 8000:8000 model-api
```

## API Endpoints

- POST /predict - Make predictions using the model
"""
        readme_path.write_text(content)
