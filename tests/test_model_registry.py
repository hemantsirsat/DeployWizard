import os
import json
import pytest
from pathlib import Path

class TestModelRegistry:
    def test_register_model(self, temp_registry, sample_model):
        """Test registering a new model."""
        # Register a model
        model_info = temp_registry.register_model(
            name=sample_model['name'],
            version=sample_model['version'],
            path=sample_model['path'],
            framework=sample_model['framework'],
            description=sample_model['description']
        )
        
        # Verify the returned model info
        assert model_info['name'] == sample_model['name']
        assert model_info['version'] == sample_model['version']
        assert model_info['path'] == os.path.abspath(sample_model['path'])
        assert model_info['framework'] == sample_model['framework']
        assert model_info['description'] == sample_model['description']
        assert 'id' in model_info
        assert 'created_at' in model_info
        
        # Verify the model was saved to the registry
        saved_model = temp_registry.get_model(sample_model['name'], sample_model['version'])
        assert saved_model == model_info
    
    def test_register_duplicate_model(self, temp_registry, sample_model):
        """Test registering a duplicate model version raises an error."""
        # Register a model
        temp_registry.register_model(**sample_model)
        
        # Try to register the same model version again
        with pytest.raises(ValueError, match="already exists"):
            temp_registry.register_model(**sample_model)
    
    def test_get_model(self, temp_registry, sample_model):
        """Test retrieving a model by name and version."""
        # Register a model
        registered_model = temp_registry.register_model(**sample_model)
        
        # Get the model
        retrieved_model = temp_registry.get_model(
            sample_model['name'], 
            sample_model['version']
        )
        
        assert retrieved_model == registered_model
    
    def test_get_latest_version(self, temp_registry, sample_model):
        """Test getting the latest version of a model."""
        # Register multiple versions
        v1 = sample_model.copy()
        v1['version'] = '1.0.0'
        temp_registry.register_model(**v1)
        
        v2 = sample_model.copy()
        v2['version'] = '2.0.0'
        temp_registry.register_model(**v2)
        
        # Get latest version (should be 2.0.0)
        latest = temp_registry.get_model(sample_model['name'])
        assert latest['version'] == '2.0.0'
    
    def test_list_models(self, temp_registry, sample_model):
        """Test listing all registered models."""
        # Register multiple models
        model1 = sample_model.copy()
        model1['name'] = 'model1'
        model1['version'] = '1.0.0'
        
        model2 = sample_model.copy()
        model2['name'] = 'model1'
        model2['version'] = '2.0.0'
        
        model3 = sample_model.copy()
        model3['name'] = 'model2'
        model3['version'] = '1.0.0'
        
        # Register all models
        temp_registry.register_model(**model1)
        temp_registry.register_model(**model2)
        temp_registry.register_model(**model3)
        
        # List all models
        models = temp_registry.list_models()
        assert len(models) == 3
        
        # Verify model names and versions
        model_versions = {(m['name'], m['version']) for m in models}
        assert ('model1', '1.0.0') in model_versions
        assert ('model1', '2.0.0') in model_versions
        assert ('model2', '1.0.0') in model_versions
    
    def test_delete_model_version(self, temp_registry, sample_model):
        """Test deleting a specific model version."""
        # Register multiple versions
        v1 = sample_model.copy()
        v1['version'] = '1.0.0'
        temp_registry.register_model(**v1)
        
        v2 = sample_model.copy()
        v2['version'] = '2.0.0'
        temp_registry.register_model(**v2)
        
        # Delete version 1.0.0
        result = temp_registry.delete_model(sample_model['name'], '1.0.0')
        assert result is True
        
        # Verify version 1.0.0 is deleted
        assert temp_registry.get_model(sample_model['name'], '1.0.0') is None
        
        # Verify version 2.0.0 still exists
        assert temp_registry.get_model(sample_model['name'], '2.0.0') is not None
    
    def test_delete_all_versions(self, temp_registry, sample_model):
        """Test deleting all versions of a model."""
        # Register multiple versions
        v1 = sample_model.copy()
        v1['version'] = '1.0.0'
        temp_registry.register_model(**v1)
        
        v2 = sample_model.copy()
        v2['version'] = '2.0.0'
        temp_registry.register_model(**v2)
        
        # Delete all versions
        result = temp_registry.delete_model(sample_model['name'])
        assert result is True
        
        # Verify all versions are deleted
        assert temp_registry.get_model(sample_model['name'], '1.0.0') is None
        assert temp_registry.get_model(sample_model['name'], '2.0.0') is None
        assert temp_registry.list_models() == []
    
    def test_persists_to_disk(self, temp_registry, sample_model, tmp_path):
        """Test that the registry persists to disk."""
        # Register a model
        temp_registry.register_model(**sample_model)
        registry_path = temp_registry.registry_path
        
        # Create a new registry instance with the same path
        new_registry = type(temp_registry)(registry_path=registry_path)
        
        # Verify the model exists in the new registry
        assert new_registry.get_model(sample_model['name'], sample_model['version']) is not None
    
    def test_handles_corrupted_file(self, temp_registry, sample_model):
        """Test that the registry handles corrupted files gracefully."""
        # Register a model
        temp_registry.register_model(**sample_model)
        
        # Corrupt the registry file
        with open(temp_registry.registry_path, 'w') as f:
            f.write('{invalid json')
        
        # Create a new registry instance - should handle the corruption
        new_registry = type(temp_registry)(registry_path=temp_registry.registry_path)
        
        # Should be empty due to corruption handling
        assert new_registry.list_models() == []
