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

    def generate(
        self, 
        model_path: str, 
        framework: str, 
        output_dir: str, 
        api_type: str = "fastapi",
        template_vars: Optional[Dict[str, Any]] = None
    ) -> None:
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
        
        # Set up output directories
        output_path = Path(output_dir)
        app_dir = output_path / "app"
        app_dir.mkdir(parents=True, exist_ok=True)
        
        # Default template variables
        default_vars = {
            'model_name': Path(model_path).name,
            'framework': framework,
            'model_class_available': False,
        }
        
        # Merge with provided template variables
        template_vars = {**default_vars, **(template_vars or {})}
        
        # Generate main application file
        self._generate_main(app_dir, framework, template_vars)
        
        # Generate requirements.txt
        self._generate_requirements(app_dir, framework)
        
        # Generate README if it doesn't exist
        readme_path = app_dir / "README.md"
        if not readme_path.exists():
            self._generate_readme(app_dir, framework, template_vars)

    def _generate_main(self, output_dir: Path, framework: str, template_vars: Dict[str, Any]) -> None:
        """Generate the main application file."""
        try:
            template = self._env.get_template('fastapi_main.tpl')
            output = template.render(
                framework=framework,
                model_name=template_vars['model_name'],
                model_class_available=template_vars.get('model_class_available', False),
            )
            
            with open(output_dir / "main.py", "w", encoding="utf-8") as f:
                f.write(output)
                
        except Exception as e:
            logger.error(f"Failed to generate main.py: {e}")
            raise

    def _generate_requirements(self, output_dir: Path, framework: str) -> None:
        """
        Generate requirements.txt file.
        
        Args:
            output_dir: Directory to write requirements.txt to
            framework: Framework used (e.g., "sklearn", "pytorch", "tensorflow")
        """
        try:
            requirements = {
                'fastapi': '>=0.68.0',
                'uvicorn': '>=0.15.0',
                'python-multipart': '',  # For file uploads
                'pydantic': '>=1.8.2,<3.0.0',
                'numpy': '>=1.21.0,<2.0.0'  # Compatible with most ML libraries
            }
            
            # Add framework-specific requirements
            if framework == 'sklearn':
                requirements['scikit-learn'] = '>=1.0.0,<2.0.0'  # Support a wide range of scikit-learn versions
                requirements['joblib'] = '>=1.0.0'  # Flexible joblib version
            elif framework == 'pytorch':
                requirements['torch'] = '>=1.9.0,<3.0.0'  # Support a wide range of PyTorch versions
            elif framework == 'tensorflow':
                requirements['tensorflow'] = '>=2.6.0,<3.0.0'  # Support TF 2.x
                
            with open(output_dir / "requirements.txt", "w", encoding="utf-8") as f:
                for pkg, version in requirements.items():
                    if version:
                        f.write(f"{pkg}{version}\n")
                    else:
                        f.write(f"{pkg}\n")
                    
        except Exception as e:
            logger.error(f"Failed to generate requirements.txt: {e}")
            raise

    def _generate_readme(self, output_dir: Path, framework: str, template_vars: Dict[str, Any]) -> None:
        """
        Generate README.md file.
        
        Args:
            output_dir: Directory to write README.md to
            framework: Framework used (e.g., "sklearn", "pytorch", "tensorflow")
            template_vars: Template variables
        """
        try:
            # Generate README content
            readme_content = f"# {template_vars['model_name']} API\n"
            readme_content += f"Generated using {framework} framework.\n"
            
            with open(output_dir / "README.md", "w", encoding="utf-8") as f:
                f.write(readme_content)
                
        except Exception as e:
            logger.error(f"Failed to generate README.md: {e}")
            raise
