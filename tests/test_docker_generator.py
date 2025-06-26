import os
import sys
import stat
import platform
import pytest
from pathlib import Path
from unittest.mock import patch, mock_open, MagicMock
from deploywizard.scaffolder.docker_generator import DockerGenerator

def test_generate_dockerfile_defaults(tmp_path):
    """Test generating a Dockerfile with default parameters."""
    generator = DockerGenerator()
    
    # Set up test directories and files
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    # Create app directory and requirements.txt
    app_dir = output_dir / "app"
    app_dir.mkdir()
    (app_dir / "requirements.txt").write_text("fastapi\nuvicorn\n")
    (app_dir / "model.pkl").touch()
    
    # Generate the Dockerfile
    dockerfile_path = output_dir / "Dockerfile"
    generator.generate(
        output_dir=str(output_dir),
        template_vars={
            'model_name': 'model.pkl',
            'python_version': '3.10',
            'additional_deps': {}
        }
    )
    
    # Verify the Dockerfile was created
    assert dockerfile_path.exists(), "Dockerfile was not created"
    
    # Read the generated Dockerfile
    dockerfile_content = dockerfile_path.read_text()
    
    # Verify content in Dockerfile
    assert 'FROM python:3.10-slim' in dockerfile_content
    assert 'COPY --chown=appuser:appuser app/requirements.txt .' in dockerfile_content
    assert 'RUN pip install --no-cache-dir -r requirements.txt' in dockerfile_content
    assert 'COPY --chown=appuser:appuser app/ .' in dockerfile_content
    assert 'CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]' in dockerfile_content

@pytest.mark.parametrize("use_gpu,expected_base_image", [
    (False, 'python:3.10-slim'),
    (True, 'nvidia/cuda:11.8.0-base-ubuntu22.04')
])
def test_dockerfile_variations(tmp_path, use_gpu, expected_base_image):
    """Test different Dockerfile variations."""
    generator = DockerGenerator()
    
    # Set up test directories and files
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    # Create app directory and requirements.txt
    app_dir = output_dir / "app"
    app_dir.mkdir()
    (app_dir / "requirements.txt").write_text("fastapi\nuvicorn\n")
    (app_dir / "model.pkl").touch()
    
    # Generate the Dockerfile
    dockerfile_path = output_dir / "Dockerfile"
    generator.generate(
        output_dir=str(output_dir),
        template_vars={
            'model_name': 'model.pkl',
            'python_version': '3.10',
            'use_gpu': use_gpu,
            'additional_deps': {}
        }
    )
    
    # Verify the Dockerfile was created
    assert dockerfile_path.exists(), "Dockerfile was not created"
    
    # Read the generated Dockerfile
    dockerfile_content = dockerfile_path.read_text()
    
    # Verify the correct base image is used
    assert f'FROM {expected_base_image}' in dockerfile_content, \
        f"Expected base image {expected_base_image} not found in Dockerfile"
    
    # Verify GPU-specific settings if needed
    if use_gpu:
        assert 'NVIDIA_VISIBLE_DEVICES=all' in dockerfile_content
        assert 'NVIDIA_DRIVER_CAPABILITIES=compute,utility' in dockerfile_content

def test_custom_requirements(tmp_path):
    """Test with custom requirements file."""
    generator = DockerGenerator()
    
    # Set up test directories and files
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    # Create app directory and custom requirements
    app_dir = output_dir / "app"
    app_dir.mkdir()
    (app_dir / "custom_reqs.txt").write_text("fastapi\nuvicorn\ntorch\n")
    (app_dir / "model.pkl").touch()
    
    # Generate the Dockerfile with custom requirements
    dockerfile_path = output_dir / "Dockerfile"
    generator.generate(
        output_dir=str(output_dir),
        template_vars={
            'model_name': 'model.pkl',
            'python_version': '3.10',
            'requirements_file': 'custom_reqs.txt',
            'additional_deps': {}
        }
    )
    
    # Verify the Dockerfile was created
    assert dockerfile_path.exists(), "Dockerfile was not created"
    
    # Read the generated Dockerfile
    dockerfile_content = dockerfile_path.read_text()
    
    # Verify custom requirements are used
    assert '# Using custom requirements file' in dockerfile_content
    assert 'custom_reqs.txt' in dockerfile_content
    assert 'requirements.txt' in dockerfile_content

def test_file_write_errors(tmp_path):
    """Test error handling during file writing."""
    # Skip this test on Windows as the permission model is different
    if platform.system() == 'Windows':
        pytest.skip("Skipping permission test on Windows")
    
    generator = DockerGenerator()
    
    # Create a read-only directory
    read_only_dir = tmp_path / "readonly"
    read_only_dir.mkdir()
    
    # Make the directory read-only
    read_only_dir.chmod(0o400)
    
    # Try to generate in a read-only directory (should raise PermissionError)
    with pytest.raises((PermissionError, OSError)):
        generator.generate(
            output_dir=str(read_only_dir),
            template_vars={
                'model_name': 'model.pkl',
                'python_version': '3.10',
                'additional_deps': {}
            }
        )

def test_docker_generator_with_extra_args(tmp_path):
    """Test DockerGenerator with extra arguments."""
    generator = DockerGenerator()
    
    # Set up test directories and files
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    # Create app directory and requirements
    app_dir = output_dir / "app"
    app_dir.mkdir()
    (app_dir / "requirements.txt").write_text("fastapi\nuvicorn\n")
    (app_dir / "model.pkl").touch()
    
    # Generate the Dockerfile with extra system dependencies
    dockerfile_path = output_dir / "Dockerfile"
    generator.generate_dockerfile(
        model_name="model.pkl",
        output_dir=str(output_dir),
        python_version="3.10",
        additional_deps={'system': ['git', 'curl']},
        use_gpu=False,
        requirements_file="requirements.txt"
    )
    
    # Verify the Dockerfile was created
    assert dockerfile_path.exists(), "Dockerfile was not created"
    
    # Read the generated Dockerfile
    dockerfile_content = dockerfile_path.read_text()
    
    # Find the apt-get install section
    apt_get_section = ""
    in_apt_get = False
    for line in dockerfile_content.split('\n'):
        if 'apt-get install' in line:
            in_apt_get = True
            apt_get_section += line
        elif in_apt_get:
            if line.strip().endswith('\\'):
                # Remove trailing backslash and add to section
                apt_get_section += ' ' + line.strip(' \\')
            else:
                in_apt_get = False
                apt_get_section += ' ' + line.strip()
    
    # Normalize whitespace for easier checking
    apt_get_section = ' '.join(apt_get_section.split())
    
    # Check that all required packages are in the apt-get install command
    required_packages = ['build-essential', 'git', 'curl']
    for pkg in required_packages:
        assert pkg in apt_get_section, f"{pkg} not found in apt-get install command: {apt_get_section}"
    
    # Verify the template variables were used correctly
    assert 'COPY --chown=appuser:appuser app/requirements.txt .' in dockerfile_content
    assert 'COPY --chown=appuser:appuser app/model.pkl /app/' in dockerfile_content
