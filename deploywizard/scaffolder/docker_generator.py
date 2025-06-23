from jinja2 import Environment, FileSystemLoader
from pathlib import Path
from typing import Dict, Optional, Any
from importlib import resources
import os
import shutil

class DockerGenerator:
    def __init__(self):
        template_dir = resources.files('deploywizard.templates')
        self._env = Environment(
            loader=FileSystemLoader(str(template_dir)),
            trim_blocks=True,
            lstrip_blocks=True
        )

    def generate(self, output_dir: str, template_vars: Optional[Dict[str, Any]] = None) -> None:
        """
        Generate Docker configuration files.
        
        Args:
            output_dir: Directory where the files will be created
            template_vars: Dictionary of template variables
        """
        if template_vars is None:
            template_vars = {}
            
        # Generate Dockerfile
        self.generate_dockerfile(
            model_name=template_vars.get('model_name', 'model.pkl'),
            output_dir=output_dir,
            python_version=template_vars.get('python_version', '3.10'),
            additional_deps=template_vars.get('additional_deps', {})
        )
        
        # Generate docker-compose.yml
        self.generate_docker_compose(
            output_dir=output_dir,
            service_name=template_vars.get('service_name', 'ml-service'),
            port=template_vars.get('port', 8000)
        )

    def generate_dockerfile(
        self, 
        model_name: str, 
        output_dir: str,
        python_version: str = "3.10",
        additional_deps: Optional[Dict[str, list]] = None
    ) -> None:
        """
        Generate a Dockerfile based on the template.
        
        Args:
            model_name: Name of the model file (e.g., 'model.pkl')
            output_dir: Directory where the Dockerfile will be created
            python_version: Python version for the base image
            additional_deps: Additional system dependencies to install
        """
        template = self._env.get_template('Dockerfile.tpl')
        
        # Default system dependencies
        system_deps = ["build-essential"]
        
        # Add any additional system dependencies
        if additional_deps and 'system' in additional_deps:
            system_deps.extend(additional_deps['system'])
        
        # Ensure model_name is just the filename, not a path
        model_name = os.path.basename(model_name)
        
        rendered = template.render(
            python_version=python_version,
            model_name=model_name,
            system_deps=system_deps
        )
        
        # Ensure output directory exists
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Write Dockerfile
        with open(output_path / 'Dockerfile', 'w') as f:
            f.write(rendered)
    
    def generate_docker_compose(
        self,
        output_dir: str,
        service_name: str = "ml-service",
        port: int = 8000
    ) -> None:
        """
        Generate a docker-compose.yml file.
        
        Args:
            output_dir: Directory where the docker-compose.yml will be created
            service_name: Name of the service in docker-compose
            port: Port to expose for the service
        """
        template = self._env.get_template('docker-compose.tpl')
        
        rendered = template.render(
            service_name=service_name,
            port=port
        )
        
        # Ensure output directory exists
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Write docker-compose.yml
        with open(output_path / 'docker-compose.yml', 'w') as f:
            f.write(rendered)
