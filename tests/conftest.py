import pytest
import os
import tempfile
from pathlib import Path
from scaffolder.model_registry import ModelRegistry

@pytest.fixture
def temp_registry():
    """Create a temporary registry file for testing."""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as tmp:
        registry_path = tmp.name
    
    # Create a registry instance with the temp file
    registry = ModelRegistry(registry_path=registry_path)
    
    yield registry
    
    # Clean up
    try:
        os.unlink(registry_path)
    except:
        pass

@pytest.fixture
def sample_model():
    """Return sample model data for testing."""
    return {
        'name': 'test_model',
        'version': '1.0.0',
        'path': '/path/to/model.pkl',
        'framework': 'sklearn',
        'description': 'A test model'
    }
