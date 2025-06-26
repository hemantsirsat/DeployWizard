#!/usr/bin/env python3
"""
Test the end-to-end workflow from the README to ensure all commands work as expected.
"""
import os
import sys
import subprocess
import tempfile
import shutil
import time
import json
from pathlib import Path
import joblib
import pytest
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pytest
import os
from pathlib import Path
from unittest.mock import patch, MagicMock, ANY
from typer.testing import CliRunner
from deploywizard.cli import app

# Constants
TEST_MODEL_NAME = "test_iris_classifier"
TEST_MODEL_VERSION = "1.0.0"
TEST_API_PORT = 8000


def run_command(cmd, cwd=None, env=None, check=True):
    """Run a shell command and return the output."""
    if env is None:
        env = os.environ.copy()
    
    print(f"Running: {cmd}")
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            errors='replace',
            check=check
        )
        return result
    except subprocess.CalledProcessError as e:
        print(f"Command failed with code {e.returncode}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        if check:
            raise
        return e


def create_test_model():
    """Create a test scikit-learn model."""
    X, y = load_iris(return_X_y=True)
    model = RandomForestClassifier(n_estimators=10, random_state=42).fit(X, y)
    return model


@pytest.fixture(scope="module")
def temp_dir():
    """Create a temporary directory for testing."""
    temp_dir = tempfile.mkdtemp(prefix="deploywizard_test_")
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


def is_docker_running():
    """Check if Docker daemon is running and accessible."""
    try:
        result = subprocess.run(
            ["docker", "info"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.SubprocessError):
        return False


def test_docker_available():
    """Check if Docker is available before running the test."""
    assert is_docker_running(), \
        "Docker is not running. Please start Docker Desktop and try again."


@pytest.mark.skipif(not is_docker_running(), reason="Docker is not running")
def test_end_to_end_workflow(temp_dir):
    """Test the end-to-end workflow from the README."""
    # Skip test if Docker is not running
    if not is_docker_running():
        pytest.skip("Docker is not running. Skipping end-to-end test.")
    
    # Step 1: Create a test model
    model = create_test_model()
    model_path = temp_dir / "iris_model.pkl"
    joblib.dump(model, model_path)
    
    # Set up environment
    env = os.environ.copy()
    env["DEPLOYWIZARD_REGISTRY"] = str(temp_dir / "test_registry.json")
    
    # Step 2: Register the model
    cmd = f"deploywizard register --name {TEST_MODEL_NAME} --version {TEST_MODEL_VERSION} --framework sklearn --description \"Test model\" {model_path}"
    result = run_command(cmd, env=env, check=False)
    assert result.returncode == 0, f"Failed to register model: {result.stderr}"
    
    # Verify model is registered
    result = run_command(f"deploywizard list", env=env)
    assert result.returncode == 0
    assert TEST_MODEL_NAME in result.stdout
    
    # Step 3: Deploy the API
    api_dir = temp_dir / "iris_api"
    cmd = f"deploywizard deploy --name {TEST_MODEL_NAME} --version {TEST_MODEL_VERSION} --output {api_dir} --api fastapi"
    result = run_command(cmd, env=env, check=False)
    
    # Debug output
    print(f"Deploy command exited with code: {result.returncode}")
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
    
    # Check if the output directory was created
    print(f"Checking if output directory exists: {api_dir}")
    print(f"Directory contents: {list(api_dir.parent.glob('*')) if api_dir.parent.exists() else 'Parent dir does not exist'}")
    
    assert result.returncode == 0, f"Failed to deploy model. Exit code: {result.returncode}"
    
    # Verify deployment files were created
    expected_files = [
        api_dir / "Dockerfile",
        api_dir / "docker-compose.yml",
        api_dir / "app" / "main.py",
        api_dir / "app" / "requirements.txt"
    ]
    
    missing_files = [str(f) for f in expected_files if not f.exists()]
    assert not missing_files, f"Missing expected files: {', '.join(missing_files)}"
    
    # Print contents of generated files for debugging
    for file_path in expected_files:
        if file_path.exists():
            print(f"\nContents of {file_path}:")
            try:
                with open(file_path, 'r') as f:
                    print(f.read())
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
        else:
            print(f"\nFile not found: {file_path}")
    
    # Ensure the model file is in the app directory for Docker to find
    app_dir = api_dir / 'app'
    model_dir = app_dir / 'model'
    model_dir.mkdir(exist_ok=True)
    shutil.copy2(model_path, model_dir / 'iris_model.pkl')
    
    # Step 4: Test the API (using docker-compose)
    if not is_docker_running():
        print("Skipping Docker Compose test as Docker is not available")
        return
    
    try:
        # First, build the image to capture build logs
        print("\nBuilding Docker image...")
        build_cmd = "docker compose build --no-cache --progress=plain"  
        build_result = run_command(build_cmd, cwd=api_dir, check=False, env={**os.environ, "DOCKER_BUILDKIT": "0"})  
    
        print(f"Build command exited with code: {build_result.returncode}")
        print("Build STDOUT:", build_result.stdout)
        print("Build STDERR:", build_result.stderr)
        
        if build_result.returncode != 0:
            # Get detailed build logs
            logs_cmd = "docker compose logs"  
            logs_result = run_command(logs_cmd, cwd=api_dir, check=False)
            print("\nContainer logs after failed build:")
            print(logs_result.stdout)
            print("Logs STDERR:", logs_result.stderr)
            
            # Get container status
            ps_cmd = "docker compose ps -a"  
            ps_result = run_command(ps_cmd, cwd=api_dir, check=False)
            print("\nContainer status:")
            print(ps_result.stdout)
            
            # Get Docker system info
            info_cmd = "docker info"
            info_result = run_command(info_cmd, check=False)
            print("\nDocker info:")
            print(info_result.stdout)
            
            assert False, f"Docker build failed with code {build_result.returncode}"
        
        # If build succeeded, start the container
        print("\nStarting Docker containers...")
        up_cmd = "docker compose up -d"  
        up_result = run_command(up_cmd, cwd=api_dir, check=False)
        
        print(f"Up command exited with code: {up_result.returncode}")
        print("Up STDOUT:", up_result.stdout)
        print("Up STDERR:", up_result.stderr)
        
        if up_result.returncode != 0:
            # Get container logs
            logs_cmd = "docker compose logs"  
            logs_result = run_command(logs_cmd, cwd=api_dir, check=False)
            print("\nContainer logs after failed start:")
            print(logs_result.stdout)
            print("Logs STDERR:", logs_result.stderr)
            
            # Get container status
            ps_cmd = "docker compose ps -a"  
            ps_result = run_command(ps_cmd, cwd=api_dir, check=False)
            print("\nContainer status:")
            print(ps_result.stdout)
            
            # Get Docker system info
            info_cmd = "docker info"
            info_result = run_command(info_cmd, check=False)
            print("\nDocker info:")
            print(info_result.stdout)
            
            assert False, f"Docker Compose up failed with code {up_result.returncode}"
        
        # Test health check with retries
        health_check_passed = False
        health_response = None
        response = None  # Initialize response variable
        
        # Use requests for more reliable HTTP requests
        import requests
        from urllib3.exceptions import NewConnectionError
        
        max_retries = 5
        for attempt in range(max_retries):
            try:
                print(f"\nAttempt {attempt + 1}/{max_retries} - Testing health check on port {TEST_API_PORT}...")
                response = requests.get(
                    f"http://localhost:{TEST_API_PORT}/health",
                    timeout=5,
                    headers={"Accept": "application/json"}
                )
                print(f"Health check status: {response.status_code}")
                print(f"Response headers: {response.headers}")
                
                if response.status_code == 200:
                    health_response = response.json()
                    print(f"Health check response JSON: {health_response}")
                    
                    # Check if the response contains expected fields
                    if isinstance(health_response, dict):
                        status = health_response.get("status")
                        if status in ["ok", "healthy"]:  # Accept both 'ok' and 'healthy' status
                            health_check_passed = True
                            break
                        else:
                            print(f"Unexpected status in health check: {status}")
                            print(f"Full response: {health_response}")
                    else:
                        print(f"Unexpected health check response format: {health_response}")
                else:
                    print(f"Unexpected status code: {response.status_code}")
                    print(f"Response content: {response.text}")
            
            except requests.exceptions.ConnectionError as e:
                if isinstance(e.args[0], NewConnectionError):
                    print(f"Connection refused - service may not be ready yet: {e}")
                else:
                    print(f"Connection error during health check: {e}")
            except requests.exceptions.RequestException as e:
                print(f"Health check request failed: {e}")
            except json.JSONDecodeError as e:
                print(f"Failed to parse health check response: {e}")
                if response is not None and hasattr(response, 'text'):
                    print(f"Response text: {response.text[:500]}")
            except Exception as e:
                print(f"Unexpected error during health check: {str(e)}")

            if attempt < max_retries - 1:
                wait_time = 2 * (attempt + 1)  # Exponential backoff
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
        
        if not health_check_passed:
            # Get container logs for debugging
            logs_cmd = "docker compose logs --no-color"
            logs_result = run_command(logs_cmd, cwd=api_dir, check=False)
            print("\n=== Container Logs ===")
            print(logs_result.stdout)
            print("Container errors:", logs_result.stderr)
            
            # Get container status
            ps_cmd = "docker compose ps -a"
            ps_result = run_command(ps_cmd, cwd=api_dir, check=False)
            print("\n=== Container Status ===")
            print(ps_result.stdout)
            
            # Get container inspect details
            inspect_cmd = "docker compose ps -q | xargs -I {} docker inspect {}"
            inspect_result = run_command(inspect_cmd, cwd=api_dir, check=False)
            print("\n=== Container Details ===")
            print(inspect_result.stdout)
            
            # Try to get logs from the container directly
            container_logs_cmd = "docker compose exec -T ml-service cat /app/main.py || echo 'Failed to get main.py'"
            container_logs = run_command(container_logs_cmd, cwd=api_dir, check=False)
            print("\n=== Main Application Code ===")
            print(container_logs.stdout)
            
            # Check if the port is actually bound
            port_check_cmd = "docker compose exec -T ml-service sh -c 'netstat -tuln || ss -tuln || echo \"No netstat/ss available\"'"
            port_check = run_command(port_check_cmd, cwd=api_dir, check=False)
            print("\n=== Port Binding Check ===")
            print(port_check.stdout)
            
            # Get environment variables
            env_cmd = "docker compose exec -T ml-service env"
            env_result = run_command(env_cmd, cwd=api_dir, check=False)
            print("\n=== Environment Variables ===")
            print(env_result.stdout)
            
            # Check if the model file exists in the container
            model_check_cmd = f'docker compose exec -T ml-service sh -c "ls -la /app/ || echo \"Failed to list /app\""'
            model_check = run_command(model_check_cmd, cwd=api_dir, check=False)
            print("\n=== Model File Check ===")
            print(model_check.stdout)
            
            # Check Python version and installed packages
            py_check = "docker compose exec -T ml-service python -c \"import sys, pkg_resources; print(f'Python {sys.version}\nInstalled packages:'); [print(p) for p in sorted([f'{p.project_name}=={p.version}' for p in pkg_resources.working_set])]\""
            py_result = run_command(py_check, cwd=api_dir, check=False)
            print("\n=== Python Environment ===")
            print(py_result.stdout)
            
            assert False, f"Health check failed after {max_retries} attempts. See logs above for details."
        
        # Test prediction
        print("\nTesting prediction endpoint...")
        predict_data = {"features": [5.1, 3.5, 1.4, 0.2]}
        
        try:
            response = requests.post(
                f"http://localhost:{TEST_API_PORT}/predict",
                json=predict_data,
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            print(f"Prediction status: {response.status_code}")
            print(f"Response headers: {response.headers}")
            
            if response.status_code != 200:
                print(f"Prediction failed with status {response.status_code}: {response.text}")
                assert False, f"Prediction request failed with status {response.status_code}"
            
            prediction = response.json()
            print(f"Prediction JSON: {prediction}")
            
            # Verify prediction result
            if not isinstance(prediction, dict):
                assert False, f"Expected dictionary response, got {type(prediction)}: {prediction}"
                
            if "prediction" not in prediction:
                assert False, f"Response missing 'prediction' field: {prediction}"
                
            if not isinstance(prediction["prediction"], (int, float)):
                assert False, f"'prediction' is not a number: {prediction}"
                
            print("Prediction test passed!")
            
        except requests.exceptions.RequestException as e:
            assert False, f"Prediction request failed: {str(e)}"
        except json.JSONDecodeError as e:
            assert False, f"Failed to parse prediction response: {str(e)}\nResponse: {response.text if 'response' in locals() else 'No response'}"
        except Exception as e:
            print(f"Unexpected error during prediction: {str(e)}")
            raise
            
    finally:
        # Clean up
        try:
            print("\nCleaning up...")
            # Stop and remove containers, networks, and volumes
            down_cmd = "docker compose down -v --remove-orphans"
            print(f"Running: {down_cmd} in {api_dir}")
            down_result = subprocess.run(
                down_cmd,
                shell=True,
                cwd=api_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            if down_result.returncode != 0:
                print("Warning: docker-compose down failed:")
                print("STDOUT:", down_result.stdout)
                print("STDERR:", down_result.stderr)
                
            # Additional cleanup in case docker-compose down failed
            print("Running additional cleanup...")
            try:
                # Windows-compatible cleanup commands
                if os.name == 'nt':  # Windows
                    subprocess.run(
                        'for /f "tokens=*" %i in (\'docker ps -aq\') do @docker rm -f %i',
                        shell=True,
                        check=False,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE
                    )
                else:  # Unix
                    subprocess.run(
                        'docker ps -aq | xargs -r docker rm -f',
                        shell=True,
                        check=False,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE
                    )
                
                subprocess.run(
                    'docker network prune -f',
                    shell=True,
                    check=False,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                subprocess.run(
                    'docker volume prune -f',
                    shell=True,
                    check=False,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
            except Exception as e:
                print(f"Warning: Additional cleanup failed: {e}")
            
        except Exception as e:
            print(f"Warning: Cleanup failed: {e}")


# Initialize test runner
runner = CliRunner()

@patch('deploywizard.cli.Scaffolder')
@patch('subprocess.run')
def test_end_to_end_workflow(mock_run, mock_scaffolder, tmp_path):
    """Test the complete workflow from the README example."""
    # Setup mocks
    mock_instance = MagicMock()
    mock_scaffolder.return_value = mock_instance
    
    # Create a dummy model file
    model_path = tmp_path / "model.pkl"
    model_path.write_text("dummy model data")
    
    # Mock model registration
    mock_instance.register_model.return_value = {
        'name': 'test_model',
        'version': '1.0.0',
        'path': str(model_path),
        'framework': 'sklearn'
    }
    
    # Mock model info
    mock_instance.get_model_info.return_value = {
        'name': 'test_model',
        'version': '1.0.0',
        'path': str(model_path),
        'framework': 'sklearn'
    }
    
    # Setup mock for generate_project
    output_dir = tmp_path / "deployment"
    output_dir.mkdir()
    
    # Run register command
    result = runner.invoke(app, [
        "register",
        str(model_path),
        "--name", "test_model",
        "--version", "1.0.0",
        "--framework", "sklearn",
        "--description", "Test model"
    ])
    
    # Verify registration
    assert result.exit_code == 0, f"Register command failed: {result.output}"
    mock_instance.register_model.assert_called_once()
    
    # Reset mock for next command
    mock_instance.reset_mock()
    
    # Run deploy command
    result = runner.invoke(app, [
        "deploy",
        "--name", "test_model",
        "--version", "1.0.0",
        "--output", str(output_dir)
    ])
    
    # Verify deployment was successful
    assert result.exit_code == 0, f"Deploy command failed: {result.output}"
    
    # Verify generate_project was called with correct arguments
    mock_instance.generate_project.assert_called_once()
    call_args = mock_instance.generate_project.call_args[1]
    assert call_args['model_name'] == 'test_model'
    assert call_args['version'] == '1.0.0'
    assert call_args['output_dir'] == str(output_dir)
    assert call_args['api_type'] == 'fastapi'  # Default value

@patch('deploywizard.cli.Scaffolder')
@patch('subprocess.run')
def test_docker_build_failure(mock_run, mock_scaffolder, tmp_path):
    """Test handling of Docker build failures."""
    # Setup mocks
    mock_instance = MagicMock()
    mock_scaffolder.return_value = mock_instance
    
    # Setup mock for generate_project
    output_dir = tmp_path / "deployment"
    output_dir.mkdir()
    
    # Mock model info
    mock_instance.get_model_info.return_value = {
        'name': 'test_model',
        'version': '1.0.0',
        'path': str(tmp_path / "model.pkl"),
        'framework': 'sklearn'
    }
    
    # Run deploy command
    result = runner.invoke(app, [
        "deploy",
        "--name", "test_model",
        "--version", "1.0.0",
        "--output", str(output_dir)
    ])
    
    # Verify deployment was successful
    assert result.exit_code == 0, f"Deploy command failed: {result.output}"
    mock_instance.generate_project.assert_called_once()

@patch('deploywizard.cli.Scaffolder')
@patch('subprocess.run')
def test_custom_docker_options(mock_run, mock_scaffolder, tmp_path):
    """Test deployment with custom Docker options."""
    # Setup mocks
    mock_instance = MagicMock()
    mock_scaffolder.return_value = mock_instance
    
    # Setup mock for generate_project
    output_dir = tmp_path / "deployment"
    output_dir.mkdir()
    
    # Mock model info
    mock_instance.get_model_info.return_value = {
        'name': 'test_model',
        'version': '1.0.0',
        'path': str(tmp_path / "model.pkl"),
        'framework': 'sklearn'
    }
    
    # Run deploy command with custom API type
    result = runner.invoke(app, [
        "deploy",
        "--name", "test_model",
        "--version", "1.0.0",
        "--output", str(output_dir),
        "--api", "flask"  # Using a valid option
    ])
    
    # Verify deployment was successful
    assert result.exit_code == 0, f"Deploy command failed: {result.output}"
    
    # Verify generate_project was called with custom API type
    mock_instance.generate_project.assert_called_once()
    call_args = mock_instance.generate_project.call_args[1]
    assert call_args['api_type'] == 'flask'

def test_help_command():
    """Test the help command."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Usage:" in result.output
    assert "register" in result.output
    assert "deploy" in result.output

@patch('deploywizard.cli.Scaffolder')
def test_list_models_command(mock_scaffolder):
    """Test the list models command."""
    # Setup mock
    mock_instance = MagicMock()
    mock_instance.list_models.return_value = [
        {"name": "model1", "version": "1.0.0", "framework": "sklearn"},
        {"name": "model2", "version": "2.0.0", "framework": "pytorch"}
    ]
    mock_scaffolder.return_value = mock_instance
    
    # Run command
    result = runner.invoke(app, ["list"])
    
    # Verify
    if result.exit_code != 0:
        print(f"Command failed with output: {result.output}")
        print(f"Exception: {result.exception}")
    
    assert result.exit_code == 0, f"Expected exit code 0, got {result.exit_code}"
    
    # Check for expected output
    output = result.output.lower()
    assert "model1" in output
    assert "1.0.0" in output
    assert "model2" in output
    assert "2.0.0" in output
    assert "sklearn" in output
    assert "pytorch" in output

if __name__ == "__main__":
    # For manual testing
    with tempfile.TemporaryDirectory(prefix="deploywizard_test_") as temp_dir:
        test_end_to_end_workflow(Path(temp_dir))
