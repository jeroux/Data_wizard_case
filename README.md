# Fraudulent Text Detection System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.13](https://img.shields.io/badge/Python-3.13%2B-blue.svg)](https://www.python.org/)
[![Docker](https://badgen.net/badge/icon/docker?icon=docker&label)](https://docker.com)

An NLP-powered system for detecting fraudulent text content using PyTorch and Flask, with Docker support.

## Features

- 🚀 **DistilBERT-based Model**: Efficient and lightweight transformer model
- ⚖ **Class Imbalance Handling**: Weighted loss function for skewed datasets
- 🌐 **Web Interface**: User-friendly Flask web application
- 📦 **Docker Support**: Easy deployment with containerization
- 📊 **Confidence Visualization**: Clear probability display with progress bar
- 🔍 **Text Preservation**: Original text display with results

## Installation

### Prerequisites
- Docker (recommended) or Python 3.13+
- For GPU support: NVIDIA Docker runtime

## Installation (Docker Recommended)

### Pre-built Image
```bash
# Pull from Docker Hub
docker pull jeroux/fraud-detector:latest

# Run the container
docker run -p 5000:5000 --name fraud-container jeroux/fraud-detector
```


# Access the web interface
Open your web browser and navigate to:
1. Access the web interface at http://localhost:5000

2. Enter text in the textarea

3. Click "Analyze Text"

4. View results including:

    - Fraudulent/NON_FRAUDULENT prediction

    - Confidence percentage

    - Visual confidence bar

    - Original text display

## API Usage
### Endpoint
curl -X POST -d "text=Your suspicious text here" http://localhost:5000

The result is a HTML page with the prediction and confidence.
It's not a JSON response, but you can parse the HTML to extract the prediction and confidence.


# Model Training
## Key Design Choices
Model Architecture: DistilBERT for speed/efficiency tradeoff

Class Handling: Class weights calculated using sklearn's compute_class_weight

Training Optimization:

Sequence Length: 256 tokens

Batch Size: 64 with gradient accumulation

Mixed Precision (FP16) training

Evaluation Metrics: Focus on F1-score for imbalanced data

Training Configuration

NUM_EPOCHS = 2
BATCH_SIZE = 64
LEARNING_RATE = 3e-5
MAX_LENGTH = 128
EARLY_STOPPING_PATIENCE = 2

## Choices Explained
The goal was to create a model in a short time. But we still wanted to ensure it was effective for detecting fraudulent text. Here are the key design choices made during development:
- **DistilBERT**: Chosen for its balance between performance and speed, making it suitable for real-time applications.
- **Class Weights**: Addressed class imbalance by assigning higher weights to minority classes, improving model sensitivity.
- **Mixed Precision Training**: Enabled faster training and reduced memory usage, allowing larger batch sizes.