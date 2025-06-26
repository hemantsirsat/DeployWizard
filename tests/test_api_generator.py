import pytest
import os
from pathlib import Path
from unittest.mock import patch, MagicMock, call
from deploywizard.scaffolder.api_generator import APIGenerator

def test_api_generator_init():
    """Test APIGenerator initialization."""
    generator = APIGenerator()
    assert generator is not None
    assert hasattr(generator, '_env')

@patch('deploywizard.scaffolder.api_generator.Environment')
@patch('deploywizard.scaffolder.api_generator.resources')
def test_template_loading(mock_resources, mock_env_class):
    """Test that templates are loaded correctly."""
    # Setup mocks
    mock_env = MagicMock()
    mock_env_class.return_value = mock_env
    
    # Mock the resources.files to return a Path-like object
    mock_template_dir = MagicMock()
    mock_resources.files.return_value = mock_template_dir
    mock_template_dir.__truediv__.return_value = mock_template_dir
    mock_template_dir.exists.return_value = True
    
    # Create the generator
    generator = APIGenerator()
    
    # Verify environment was set up correctly
    mock_env_class.assert_called_once()
    assert generator._env is not None

def test_generate_fastapi(tmp_path):
    """Test FastAPI code generation."""
    generator = APIGenerator()
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    app_dir = output_dir / "app"
    app_dir.mkdir()
    
    # Create a dummy requirements.txt
    (output_dir / "requirements.txt").write_text("fastapi\nuvicorn\n")
    
    # Mock the template rendering
    with patch.object(generator._env, 'get_template') as mock_get_template:
        mock_template = MagicMock()
        mock_template.render.return_value = "# Mock FastAPI app code"
        mock_get_template.return_value = mock_template
        
        # Call the method
        generator.generate(
            model_path="model.pkl",
            framework="sklearn",
            output_dir=str(output_dir),
            api_type="fastapi"
        )
        
        # Verify template was rendered
        mock_get_template.assert_called_with("fastapi_main.tpl")
        mock_template.render.assert_called_once()
    
    # Verify files were created
    assert (output_dir / "app" / "main.py").exists()
    assert (output_dir / "requirements.txt").exists()

def test_generate_with_custom_vars(tmp_path):
    """Test generation with custom template variables."""
    generator = APIGenerator()
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    app_dir = output_dir / "app"
    app_dir.mkdir()
    
    # Create a dummy requirements.txt
    (output_dir / "requirements.txt").write_text("fastapi\nuvicorn\n")
    
    # Mock the template rendering
    with patch.object(generator._env, 'get_template') as mock_get_template, \
         patch.object(generator, '_generate_requirements') as mock_gen_reqs:
        mock_template = MagicMock()
        mock_template.render.return_value = "# Mock FastAPI app code"
        mock_get_template.return_value = mock_template
        
        # Call with custom variables
        generator.generate(
            model_path="model.pkl",
            framework="pytorch",
            output_dir=str(output_dir),
            api_type="fastapi",
            template_vars={"custom_var": "value", "model_class_available": True}
        )
        
        # Verify template was rendered with the correct variables
        mock_get_template.assert_called_with("fastapi_main.tpl")
        mock_template.render.assert_called_once_with(
            framework="pytorch",
            model_name="model.pkl",
            model_class_available=True,
            custom_var="value"
        )
        
        # Verify requirements were generated
        mock_gen_reqs.assert_called_once_with(Path(str(output_dir)) / "app", "pytorch")

def test_template_loading_error():
    """Test error handling when template is not found."""
    generator = APIGenerator()
    
    # Mock template loading to raise an exception
    with patch.object(generator._env, 'get_template') as mock_get_template:
        mock_get_template.side_effect = Exception("Template not found")
        
        # Test that the exception is properly propagated
        with pytest.raises(Exception) as excinfo:
            generator.generate(
                model_path="model.pkl",
                framework="sklearn",
                output_dir="/tmp",
                api_type="fastapi"
            )
        assert "Template not found" in str(excinfo.value)
