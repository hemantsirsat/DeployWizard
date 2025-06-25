# ML Model Deployment

This project contains the deployment code for your machine learning model.

## Running Locally

```bash
uvicorn app.main:app --reload
```

## Building Docker Image

```bash
docker build -t model-api .
```

## Running Docker Container

```bash
docker run -p 8000:8000 model-api
```

## API Endpoints

- POST /predict - Make predictions using the model
