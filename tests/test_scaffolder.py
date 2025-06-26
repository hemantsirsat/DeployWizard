import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, call
from deploywizard.scaffolder import Scaffolder

def test_scaffolder_initialization():
    """Test that Scaffolder initializes correctly."""
    scaffolder = Scaffolder()
    assert scaffolder is not None
    assert hasattr(scaffolder, '_model_loader')
    assert hasattr(scaffolder, '_api_generator')
    assert hasattr(scaffolder, '_docker_generator')
    assert hasattr(scaffolder, '_registry')

@patch('deploywizard.scaffolder.scaffolder.ModelLoader')
@patch('deploywizard.scaffolder.scaffolder.ModelRegistry')
def test_register_model(mock_registry, mock_loader, tmp_path):
    """Test model registration workflow."""
    # Setup mocks
    mock_model = MagicMock()
    mock_loader.return_value.load.return_value = mock_model
    
    mock_registry_instance = MagicMock()
    mock_registry.return_value = mock_registry_instance
    
    # Initialize scaffolder
    scaffolder = Scaffolder(registry_path=str(tmp_path / "registry.json"))
    
    # Create a dummy model file
    model_path = tmp_path / "model.pkl"
    model_path.write_text("dummy model data")
    
    # Call method
    result = scaffolder.register_model(
        name='test_model',
        version='1.0.0',
        model_path=str(model_path),
        framework='sklearn',
        description='Test model'
    )
    
    # Verify interactions
    mock_loader.return_value.load.assert_called_once_with(str(model_path), 'sklearn')
    mock_registry_instance.register_model.assert_called_once()
    assert result is not None

@patch('deploywizard.scaffolder.scaffolder.APIGenerator')
@patch('deploywizard.scaffolder.scaffolder.DockerGenerator')
@patch('deploywizard.scaffolder.scaffolder.ModelRegistry')
@patch('deploywizard.scaffolder.scaffolder.shutil')
def test_generate_project(mock_shutil, mock_registry, mock_docker, mock_api, tmp_path):
    """Test project generation."""
    # Setup
    scaffolder = Scaffolder()
    
    # Create necessary directories and files
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    app_dir = output_dir / "app"
    app_dir.mkdir()
    
    # Mock model info
    model_info = {
        'name': 'test_model',
        'version': '1.0.0',
        'path': str(tmp_path / 'model.pkl'),
        'framework': 'sklearn',
        'description': 'Test model'
    }
    
    # Setup registry mock
    mock_registry_instance = MagicMock()
    mock_registry_instance.get_model.return_value = model_info
    scaffolder._registry = mock_registry_instance
    
    # Setup API and Docker generator mocks
    mock_api_instance = MagicMock()
    mock_docker_instance = MagicMock()
    scaffolder._api_generator = mock_api_instance
    scaffolder._docker_generator = mock_docker_instance
    
    # Call method
    scaffolder.generate_project(
        model_name='test_model',
        version='1.0.0',
        output_dir=str(output_dir)
    )
    
    # Verify API and Docker generation
    mock_api_instance.generate.assert_called_once()
    mock_docker_instance.generate.assert_called_once()
    mock_registry_instance.get_model.assert_called_once_with("test_model", "1.0.0")

@patch('deploywizard.scaffolder.scaffolder.ModelRegistry')
def test_list_models(mock_registry, tmp_path):
    """Test listing registered models."""
    # Setup mock registry
    mock_registry_instance = MagicMock()
    mock_registry_instance.list_models.return_value = [
        {'name': 'model1', 'version': '1.0.0'},
        {'name': 'model2', 'version': '2.0.0'}
    ]
    mock_registry.return_value = mock_registry_instance
    
    # Test
    scaffolder = Scaffolder(registry_path=str(tmp_path / "registry.json"))
    models = scaffolder.list_models()
    
    # Verify
    assert len(models) == 2
    mock_registry_instance.list_models.assert_called_once()

@patch('deploywizard.scaffolder.scaffolder.ModelRegistry')
def test_get_model_info(mock_registry, tmp_path):
    """Test retrieving a specific model's info."""
    # Setup mock registry
    mock_registry_instance = MagicMock()
    mock_registry_instance.get_model.return_value = {
        'name': 'test_model',
        'version': '1.0.0',
        'path': str(tmp_path / 'model.pkl'),
        'framework': 'sklearn'
    }
    mock_registry.return_value = mock_registry_instance
    
    # Test
    scaffolder = Scaffolder(registry_path=str(tmp_path / "registry.json"))
    model = scaffolder.get_model_info('test_model', '1.0.0')
    
    # Verify
    assert model['name'] == 'test_model'
    mock_registry_instance.get_model.assert_called_once_with('test_model', '1.0.0')

def test_generate_readme(tmp_path):
    """Test README generation."""
    # Setup
    from deploywizard.scaffolder.scaffolder import Scaffolder
    scaffolder = Scaffolder()
    
    # Create output directory
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    # Call method
    scaffolder._generate_readme(str(output_dir))
    
    # Verify README was created with expected content
    readme_path = output_dir / "README.md"
    assert readme_path.exists()
    
    content = readme_path.read_text()
    assert "ML Model Deployment" in content
    assert "docker build" in content.lower()
    assert "uvicorn" in content.lower()
