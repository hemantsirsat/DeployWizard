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

### 1. Register a Model

```bash
# Register a scikit-learn model
deploywizard register iris_model.pkl --name iris_classifier --version 1.0.0 --framework sklearn --description "Iris classifier with 95% accuracy"

# Register a PyTorch model
deploywizard register sentiment_model.pt --name sentiment_analyzer --version 2.1.0 --framework pytorch --description "BERT-based sentiment analysis"

# Register a TensorFlow model
deploywizard register image_model.h5 --name image_classifier --version 3.0.0 --framework tensorflow --description "CNN for image classification"
```

### 2. List and Inspect Models

```bash
# List all registered models
deploywizard list
# Output:
# ┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
# ┃ Name                ┃ Version ┃ Framework ┃ Description                                      ┃ Registered At ┃
# ┡━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
# │ test-model          │ 1.0.0   │ sklearn   │ Test model                                       │ 2025-06-23    │
# │ test-model-db0357e8 │ 1.0.0   │ sklearn   │ Test model                                       │ 2025-06-23    │
# │ sklearn_test_model  │ 1.0.0   │ sklearn   │ Logistic regression model trained on 100         │ 2025-06-23    │
# │                     │         │           │ samples, ...                                     │               │
# └─────────────────────┴─────────┴───────────┴──────────────────────────────────────────────────┴───────────────┘

# Get detailed info about a specific model
deploywizard info --name sklearn_test_model
# Output:
# 
# Model Information
#   • Name: sklearn_test_model
#   • Version: 1.0.0
#   • Framework: sklearn
#   • Path: /path/to/test_models/sklearn_model.pkl
#   • Registered: 2025-06-23T12:35:12.495113+00:00
#
# Description:
#   Logistic regression model trained on 100 samples, 4 features and 2 classes.
```

### 3. Deploy a Model

```bash
# Deploy a specific version of a model
deploywizard deploy --name sklearn_test_model --version 1.0.0 --output sklearn_test_model_api

# For PyTorch models saved as state_dict, provide the model class file
deploywizard deploy --name pytorch_model --model-class path/to/model.py --output pytorch_model_api

# Or deploy the latest version (omitting --version)
deploywizard deploy --name sklearn_test_model --output my_model_api
```

### 4. PyTorch Model Deployment

When deploying PyTorch models, you have several options:

1. **Full Model**: If you saved the entire model using `torch.save(model, 'model.pt')`, you can deploy it directly:
   ```bash
   deploywizard deploy --name my_pytorch_model --output pytorch_api
   ```

2. **State Dictionary**: If you saved just the state dict (`torch.save(model.state_dict(), 'model.pt')`), you need to provide the model class definition:
   ```bash
   deploywizard deploy --name my_pytorch_model --model-class path/to/model.py --output pytorch_api
   ```

   The model class file should define a PyTorch model that inherits from `torch.nn.Module`. For example:
   ```python
   import torch
   import torch.nn as nn
   
   class SimpleTorchModel(nn.Module):
       def __init__(self):
           super().__init__()
           self.fc1 = nn.Linear(input_size, hidden_size)
           self.fc2 = nn.Linear(hidden_size, 1)
           self.sigmoid = nn.Sigmoid()
       
       def forward(self, x):
           x = torch.relu(self.fc1(x))
           x = self.fc2(x)
           return self.sigmoid(x)
   ```

### 5. Run the Deployed API

After deploying, navigate to the output directory and start the API:

```bash
cd sklearn_test_model_api
docker-compose up --build
```

Once the containers are up and running, you can access:
- API: http://localhost:8000
- Interactive API documentation: http://localhost:8000/docs
- Alternative documentation: http://localhost:8000/redoc

### 6. Test the API

You can test the API using `curl` or any HTTP client:

```bash
# Health check
curl http://localhost:8000/health
# Expected output: {"status":"healthy"}

# Make predictions (example for a scikit-learn model)
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"data": [[5.1, 3.5, 1.4, 0.2]]}'
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
