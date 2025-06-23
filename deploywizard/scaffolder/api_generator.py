from jinja2 import Environment, FileSystemLoader, TemplateNotFound
from pathlib import Path
from typing import Dict, Any, Optional, Union
from importlib import resources
import logging

# Set up logging
logger = logging.getLogger(__name__)

class APIGenerator:
    def __init__(self):
        template_dir = resources.files('deploywizard.templates')
        logger.debug(f"Template directory: {template_dir}")
        self._env = Environment(
            loader=FileSystemLoader(str(template_dir)),
            trim_blocks=True,
            lstrip_blocks=True
        )
        # Verify template exists
        try:
            self._env.get_template('fastapi_main.tpl')
            logger.debug("Successfully loaded fastapi_main.tpl template")
        except TemplateNotFound as e:
            logger.error(f"Template not found: {e}")
            raise

    def generate(self, model_path: str, framework: str, output_dir: str, 
                api_type: str = "fastapi", template_vars: Optional[Dict[str, Any]] = None) -> None:
        """
        Generate API code and requirements for the model.
        
        Args:
            model_path: Path to the model file
            framework: Framework used (e.g., "sklearn", "pytorch", "tensorflow")
            output_dir: Directory to write the generated files to
            api_type: Type of API to generate (e.g., "fastapi")
            template_vars: Additional template variables
        """
        logger.info(f"Generating API for {model_path} with framework {framework}")
        
        if template_vars is None:
            template_vars = {}
            
        # Add common template variables
        template_vars.update({
            'model_path': model_path,
            'framework': framework,
            'api_type': api_type,
            'model_name': Path(model_path).name  # Add model_name for the template
        })
        
        # Generate API code
        self.generate_api(template_vars, output_dir)
        
        # Generate requirements
        requirements = {
            'fastapi': '>=0.68.0',
            'uvicorn': '>=0.15.0',
            'python-multipart': '',  # For file uploads
            'pydantic': '>=1.8.2,<2.0.0',
            'numpy': '==1.23.5'  # Pin numpy version for compatibility
        }
        
        # Add framework-specific requirements
        if framework == 'sklearn':
            requirements['scikit-learn'] = '==1.2.2'  # Pin scikit-learn version
            requirements['joblib'] = ''
        elif framework == 'pytorch':
            requirements['torch'] = ''
        elif framework == 'tensorflow':
            requirements['tensorflow'] = ''
            
        self.generate_requirements(requirements, output_dir)

    def generate_api(self, model_info: Dict[str, Any], output_dir: str) -> None:
        """Generate FastAPI application code."""
        try:
            # Ensure output directory exists
            output_path = Path(output_dir) / 'app'
            output_path.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Output directory: {output_path.absolute()}")
            logger.info(f"Directory exists: {output_path.exists()}")
            logger.info(f"Directory contents: {list(output_path.glob('*')) if output_path.exists() else 'N/A'}")
            
            # Get the template
            logger.info("Loading template...")
            template = self._env.get_template('fastapi_main.tpl')
            logger.info(f"Rendering template with model_info: {model_info}")
            
            # Render the template
            rendered = template.render(**model_info)
            
            # Write the output file
            main_py = output_path / 'main.py'
            logger.info(f"Writing API code to {main_py.absolute()}")
            
            main_py.write_text(rendered, encoding='utf-8')
            logger.info(f"Successfully wrote {main_py.absolute()} (exists: {main_py.exists()})")
            
            # Verify the file was written
            if not main_py.exists():
                error_msg = f"Failed to create {main_py}"
                logger.error(error_msg)
                # Try to write to a different location to debug
                debug_path = Path.cwd() / 'debug_main.py'
                debug_path.write_text(rendered, encoding='utf-8')
                logger.error(f"Wrote debug file to {debug_path.absolute()}")
                raise RuntimeError(f"{error_msg}. Debug file written to {debug_path}")
                
        except Exception as e:
            logger.error(f"Error generating API code: {str(e)}", exc_info=True)
            raise

    def generate_requirements(self, requirements: Dict[str, str], output_dir: str) -> None:
        """
        Generate requirements.txt file.
        
        Args:
            requirements: Dictionary of package names to version specifications
            output_dir: Directory to write requirements.txt to
        """
        try:
            output_path = Path(output_dir) / 'app'
            output_path.mkdir(parents=True, exist_ok=True)
            
            requirements_path = output_path / 'requirements.txt'
            logger.info(f"Writing requirements to {requirements_path.absolute()}")
            
            with requirements_path.open('w', encoding='utf-8') as f:
                for pkg, version in requirements.items():
                    if version:
                        f.write(f"{pkg}{version}\n")
                    else:
                        f.write(f"{pkg}\n")
            
            logger.info(f"Successfully wrote requirements to {requirements_path.absolute()}")
            
            if not requirements_path.exists():
                error_msg = f"Failed to create {requirements_path}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)
                
        except Exception as e:
            logger.error(f"Error generating requirements: {str(e)}")
            raise
