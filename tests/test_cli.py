import os
import pytest
import uuid
import logging
import shutil
import tempfile
from pathlib import Path
from typer.testing import CliRunner
from deploywizard.cli import app
import numpy as np
from sklearn.linear_model import LinearRegression
import joblib

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_cli_help():
    """Test CLI help command."""
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Usage" in result.output

def test_cli_register_and_deploy(tmp_path):
    """Test the register and deploy workflow."""
    runner = CliRunner()
    
    # Create a unique model name for this test run to avoid conflicts
    model_name = f"test-model-{uuid.uuid4().hex[:8]}"
    model_version = "1.0.0"
    
    # Create a temporary directory for the registry that will persist across commands
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        registry_path = temp_dir_path / "registry.json"
        
        # Set environment variable for registry location
        os.environ["DEPLOYWIZARD_REGISTRY"] = str(registry_path)
        logger.info(f"Using registry path: {registry_path}")
        
        # Create a dummy model file with valid scikit-learn model data
        model_path = tmp_path / "model.pkl"
        logger.debug(f"Creating test model at {model_path}")
        model = LinearRegression()
        model.fit(np.random.rand(10, 1), np.random.rand(10))
        joblib.dump(model, model_path)
        
        # Verify model file was created
        assert model_path.exists(), f"Model file not created at {model_path}"
        
        # Test register command
        with runner.isolated_filesystem(temp_dir=temp_dir_path) as td:
            logger.info(f"Isolated test directory: {td}")
            
            # Verify registry path is accessible
            logger.info(f"Registry path before command: {registry_path}")
            logger.info(f"Registry file exists before: {registry_path.exists()}")
            
            logger.debug("Running register command...")
            register_cmd = [
                "register",
                str(model_path),
                "--name", model_name,
                "--version", model_version,
                "--framework", "sklearn",
                "--description", "Test model"
            ]
            logger.debug(f"Command: {' '.join(register_cmd)}")
            
            register_result = runner.invoke(app, register_cmd)
            logger.info(f"Register command output: {register_result.output}")
            logger.info(f"Register command exit code: {register_result.exit_code}")
            
            # Debug: List all files in the temp directory
            logger.info("Files in temp directory after register:")
            for f in Path(td).rglob('*'):
                logger.info(f"  {f}")
            
            # Check if registry file was created in the expected location
            registry_files = list(Path(td).rglob('registry.json'))
            logger.info(f"Found registry files: {registry_files}")
            
            # Verify registration was successful
            assert register_result.exit_code == 0, f"Registration failed: {register_result.output}"
            assert "Successfully registered" in register_result.output
            
            # Verify registry file was created (either in the original location or in the isolated fs)
            if registry_path.exists():
                logger.info(f"Registry file found at expected location: {registry_path}")
                registry_content = registry_path.read_text()
                logger.info(f"Registry content: {registry_content}")
                # Copy registry file to temp dir for next command
                shutil.copy2(registry_path, temp_dir_path / "registry.json.bak")
            else:
                # Check if registry was created in the isolated filesystem
                isolated_registry = Path(td) / "registry.json"
                if isolated_registry.exists():
                    logger.info(f"Registry file found in isolated filesystem: {isolated_registry}")
                    registry_content = isolated_registry.read_text()
                    logger.info(f"Registry content: {registry_content}")
                    # Copy to our known location for the next command
                    shutil.copy2(isolated_registry, registry_path)
                    shutil.copy2(registry_path, temp_dir_path / "registry.json.bak")
                else:
                    # Try to find the registry file anywhere in the temp directory
                    found = False
                    for f in Path(td).rglob('registry.json'):
                        logger.info(f"Found registry file at: {f}")
                        shutil.copy2(f, registry_path)
                        shutil.copy2(registry_path, temp_dir_path / "registry.json.bak")
                        found = True
                        break
                    if not found:
                        logger.error("Registry file not found in any expected location")
                        logger.error("Contents of temp directory:")
                        for entry in Path(td).rglob('*'):
                            logger.error(f"  {entry}")
                        assert False, "Registry file was not created"
        
        # Test deploy command - use direct subprocess to avoid test runner issues
        import subprocess
        import sys
        
        # Prepare the command
        cmd = [sys.executable, "-m", "deploywizard.cli", "deploy",
               "--name", model_name,
               "--version", model_version,
               "--output", str(temp_dir_path / "output"),
               "--api", "fastapi"]
        
        # Set environment variables
        env = os.environ.copy()
        env["DEPLOYWIZARD_REGISTRY"] = str(registry_path)
        
        logger.info(f"Running command: {' '.join(cmd)}")
        logger.info(f"Registry path: {registry_path}")
        logger.info(f"Registry exists: {registry_path.exists()}")
        
        # Run the command
        try:
            result = subprocess.run(
                cmd,
                env=env,
                cwd=str(temp_dir_path),  # Use the parent directory as working directory
                capture_output=True,
                text=True,
                check=True
            )
            
            # Log command output
            logger.info(f"Command stdout:\n{result.stdout}")
            if result.stderr:
                logger.warning(f"Command stderr:\n{result.stderr}")
                
            # Verify deployment was successful
            assert "Successfully deployed" in result.stdout, "Deployment success message not found"
            
            # Verify deployment files were created
            logger.info("Verifying output files...")
            output_dir = temp_dir_path / "output"
            app_dir = output_dir / "app"
            
            # Debug: Print directory structure
            def print_dir_tree(path: Path, indent: str = ""):
                if not path.exists():
                    return f"{indent}MISSING: {path.name}"
                if path.is_file():
                    return f"{indent}FILE: {path.name}"
                
                lines = [f"{indent}DIR: {path.name}/"]
                try:
                    for child in sorted(path.iterdir()):
                        lines.append(print_dir_tree(child, indent + "  "))
                except Exception as e:
                    lines.append(f"{indent}ERROR listing {path}: {e}")
                return "\n".join(lines)
            
            logger.info("Directory structure after deployment:")
            logger.info(print_dir_tree(output_dir))
            
            # Debug: List all files in the output directory
            logger.info(f"Files in {output_dir}:")
            for f in output_dir.rglob('*'):
                logger.info(f"  {f.relative_to(output_dir)}")
            
            required_files = [
                (app_dir, "app directory"),
                (app_dir / "main.py", "main.py"),
                (app_dir / "requirements.txt", "requirements.txt"),
                (output_dir / "Dockerfile", "Dockerfile"),
                (output_dir / "docker-compose.yml", "docker-compose.yml")
            ]
            
            missing_files = []
            for path, desc in required_files:
                if not path.exists():
                    missing_files.append((path, desc))
            
            if missing_files:
                error_msg = "Missing required files:\n"
                for path, desc in missing_files:
                    error_msg += f"- {desc} not found at {path}\n"
                logger.error(error_msg)
                assert False, error_msg
            
            # Verify files exist
            for path, desc in required_files:
                assert path.exists(), f"{desc} not found at {path}"
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Command failed with exit code {e.returncode}")
            logger.error(f"Stdout:\n{e.stdout}")
            logger.error(f"Stderr:\n{e.stderr}")
            raise
        
        # Test registering the same model again should fail
        with runner.isolated_filesystem(temp_dir=temp_dir_path) as td:
            # Restore registry file if it exists
            if (temp_dir_path / "registry.json.bak").exists():
                shutil.copy2(temp_dir_path / "registry.json.bak", registry_path)
            
            logger.info("Testing duplicate registration...")
            duplicate_result = runner.invoke(
                app,
                [
                    "register",
                    str(model_path),
                    "--name", model_name,
                    "--version", model_version,
                    "--framework", "sklearn"
                ]
            )
            logger.info(f"Duplicate register output: {duplicate_result.output}")
            logger.info(f"Duplicate register exit code: {duplicate_result.exit_code}")
            
            # Verify duplicate registration fails
            assert duplicate_result.exit_code == 1
            assert "already exists in registry" in duplicate_result.output
