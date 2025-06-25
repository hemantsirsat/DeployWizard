# DeployWizard

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/hemantsirsat/deploywizard/actions/workflows/tests.yml/badge.svg)](https://github.com/hemantsirsat/deploywizard/actions/workflows/tests.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

DeployWizard is a powerful CLI tool that automates the deployment of machine learning models as production-ready REST APIs with Docker support. It generates all the necessary code and configuration files to containerize your ML models with FastAPI, making deployment a breeze.

## Features

- **Multi-Framework Support**: Works with scikit-learn, PyTorch, and TensorFlow models
- **Production-Ready**: Generates Dockerfiles and optimized FastAPI applications
- **Environment-Aware**: Handles both development and production environments
- **Secure by Default**: Uses non-root users in containers and secure defaults
- **Easy to Use**: Simple CLI interface for generating deployment code
- **Customizable**: Flexible template system for advanced customization
- **Tested**: Comprehensive test suite with 100% code coverage

## Installation

1. **Create and activate a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

2. **Install DeployWizard** using pip:
   ```bash
   pip install deploywizard
   ```

3. **Verify installation**:
   ```bash
   deploywizard --version
   ```

## Quick Start

### 1. Initialize a New Project (Recommended for New Models)

The `init` command creates a new deployment project and registers your model in one step:

```bash
# Initialize a project with a scikit-learn model
deploywizard init --model path/to/model.pkl --framework sklearn --output-dir my_ml_api

# Initialize with a custom model name
deploywizard init --model path/to/model.pkl --framework sklearn --output-dir my_ml_api --name my_model

# For PyTorch models (full model)
deploywizard init --model path/to/model.pt --framework pytorch --output-dir pytorch_api

# For PyTorch state_dict models (requires model class)
deploywizard init --model path/to/model.pt --framework pytorch --output-dir pytorch_api \
    --model-class path/to/model.py
```

This will:
1. Register your model in the DeployWizard registry
2. Create a new directory with a complete API project
3. Generate all necessary configuration files
4. Print the next steps to run your API

### 2. Deploying PyTorch Models

When working with PyTorch models, you have two options depending on how the model was saved:

#### Option 1: Full Model (Recommended)
If you saved the entire model using `torch.save(model, 'model.pt')`:
```bash
deploywizard init --model model.pt --framework pytorch --output-dir pytorch_api
```

#### Option 2: State Dictionary
If you saved just the state dict using `torch.save(model.state_dict(), 'model.pt')`, you need to provide the model class definition:

1. Create a Python file (e.g., `model.py`) with your model class:
   ```python
   import torch
   import torch.nn as nn
   
   class MyModel(nn.Module):
       def __init__(self, input_size=10, hidden_size=20, output_size=1):
           super().__init__()
           self.fc1 = nn.Linear(input_size, hidden_size)
           self.fc2 = nn.Linear(hidden_size, output_size)
           self.sigmoid = nn.Sigmoid()
       
       def forward(self, x):
           x = torch.relu(self.fc1(x))
           x = self.fc2(x)
           return self.sigmoid(x)
   ```

2. Initialize the project with the model class:
   ```bash
   deploywizard init --model model.pt --framework pytorch --output-dir pytorch_api \
       --model-class model.py
   ```

### 3. Run the Deployed API

After initializing your project, navigate to the output directory and start the API:

```bash
cd my_ml_api
docker-compose up --build
```

Once the containers are running, you can access:
- API: http://localhost:8000
- Interactive API documentation: http://localhost:8000/docs
- Alternative documentation: http://localhost:8000/redoc

### 4. Test the API

Test the API using `curl` or any HTTP client:

```bash
# Health check
curl http://localhost:8000/health
# Expected output: {"status":"healthy"}

# Make predictions (example format)
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"data": [[5.1, 3.5, 1.4, 0.2]]}'
```

## Advanced Usage

### Manual Model Registration (Alternative to `init`)

If you prefer to manage the model registration separately:

```bash
# Register a model
deploywizard register model.pkl --name my_model --version 1.0.0 --framework sklearn

# Deploy a registered model
deploywizard deploy --name my_model --output my_api

# Update model metadata
deploywizard update --name my_model --new-version 2.0.0 --description "Improved model"

# Delete a model
deploywizard delete --name old_model
```

## Testing

DeployWizard includes a comprehensive test suite. To run the tests:

```bash
# Install test dependencies
pip install -e ".[test]"

# Run all tests
pytest -v

# Run tests with coverage report
pytest --cov=deploywizard --cov-report=term-missing

# Run a specific test file
pytest tests/test_cli.py -v
```

## Troubleshooting

### Common Issues

1. **Model not found**
   - Ensure the model file exists at the specified path
   - Verify the framework is correctly specified

2. **PyTorch Model Loading Issues**
   - For state_dict models, ensure you've provided the `--model-class` option
   - Verify the model class in the provided file matches the architecture used during training
   - Check that all required imports are included in your model class file

3. **Port already in use**
   - Stop any containers using port 8000
   - Or specify a different port: `docker run -p 8080:8000`

4. **Docker build fails**
   - Check your internet connection
   - Verify Docker is running
   - Check the Docker logs for specific error messages

5. **UnicodeDecodeError on Windows**
   - Ensure your terminal supports UTF-8 encoding
   - Set the following environment variable: `set PYTHONUTF8=1`

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a new branch for your feature
3. Commit your changes with descriptive messages
4. Push to the branch
5. Create a new Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with ❤️ using FastAPI and Docker
- Inspired by the need for simple ML model deployment solutions
- Thanks to all contributors who have helped improve this project!
