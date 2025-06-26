from jinja2 import Environment, FileSystemLoader
from pathlib import Path
from typing import Dict, Optional, Any
from importlib import resources
import os
import shutil
import logging

# Set up logging
logger = logging.getLogger(__name__)

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
            
        Raises:
            PermissionError: If there are permission issues creating files or directories
            OSError: For other file system related errors
        """
        if template_vars is None:
            template_vars = {}
            
        try:
            # Ensure output directory exists and is writable
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Test if directory is writable
            test_file = output_path / '.deploywizard_test'
            try:
                test_file.touch()
                test_file.unlink()
            except (PermissionError, OSError) as e:
                logger.error(f"Cannot write to output directory {output_dir}: {e}")
                raise PermissionError(f"Cannot write to output directory {output_dir}") from e
            
            # Generate Dockerfile
            self.generate_dockerfile(
                model_name=template_vars.get('model_name', 'model.pkl'),
                output_dir=output_dir,
                python_version=template_vars.get('python_version', '3.10'),
                additional_deps=template_vars.get('additional_deps', {}),
                use_gpu=template_vars.get('use_gpu', False),
                requirements_file=template_vars.get('requirements_file')
            )
            
            # Generate docker-compose.yml
            self.generate_docker_compose(
                output_dir=output_dir,
                service_name=template_vars.get('service_name', 'ml-service'),
                port=template_vars.get('port', 8000)
            )
            
        except (PermissionError, OSError):
            # Re-raise permission and OS errors
            raise
        except Exception as e:
            # Wrap other exceptions in OSError for consistency
            logger.error(f"Failed to generate Docker configuration: {e}")
            raise OSError(f"Failed to generate Docker configuration: {e}") from e

    def generate_dockerfile(
        self, 
        model_name: str, 
        output_dir: str,
        python_version: str = "3.10",
        additional_deps: Optional[Dict[str, list]] = None,
        use_gpu: bool = False,
        requirements_file: Optional[str] = None
    ) -> None:
        """
        Generate a Dockerfile based on the template.
        
        Args:
            model_name: Name of the model file (e.g., 'model.pkl')
            output_dir: Directory where the Dockerfile will be created
            python_version: Python version for the base image
            additional_deps: Additional system dependencies to install
            use_gpu: Whether to configure the Dockerfile for GPU support
            requirements_file: Custom requirements file to use (if any)
            
        Raises:
            PermissionError: If there are permission issues writing the Dockerfile
            OSError: For other file system related errors
        """
        try:
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
                system_deps=system_deps,
                use_gpu=use_gpu,
                requirements_file=requirements_file
            )
            
            # Ensure output directory exists
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Write Dockerfile
            dockerfile_path = output_path / 'Dockerfile'
            try:
                with open(dockerfile_path, 'w') as f:
                    f.write(rendered)
            except (PermissionError, OSError) as e:
                logger.error(f"Failed to write Dockerfile to {dockerfile_path}: {e}")
                raise PermissionError(f"Cannot write to {dockerfile_path}") from e
                
        except Exception as e:
            logger.error(f"Failed to generate Dockerfile: {e}")
            raise OSError(f"Failed to generate Dockerfile: {e}") from e
    
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
            
        Raises:
            PermissionError: If there are permission issues writing the docker-compose file
            OSError: For other file system related errors
        """
        try:
            template = self._env.get_template('docker-compose.tpl')
            
            rendered = template.render(
                service_name=service_name,
                port=port
            )
            
            # Ensure output directory exists
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Write docker-compose.yml
            compose_path = output_path / 'docker-compose.yml'
            try:
                with open(compose_path, 'w') as f:
                    f.write(rendered)
            except (PermissionError, OSError) as e:
                logger.error(f"Failed to write docker-compose.yml to {compose_path}: {e}")
                raise PermissionError(f"Cannot write to {compose_path}") from e
                
        except Exception as e:
            logger.error(f"Failed to generate docker-compose.yml: {e}")
            raise OSError(f"Failed to generate docker-compose.yml: {e}") from e
