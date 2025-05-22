# Voice Assistant with Transformer-based NLP

This project implements a voice assistant using transformer-based natural language processing. It can understand voice commands, process them using state-of-the-art NLP techniques, and provide appropriate responses.

## Features
- Speech-to-text conversion
- Intent classification using transformers
- Text-to-speech response generation
- Performance evaluation metrics

## Setup
1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Install PyAudio system dependencies (if needed):
- Windows: No additional steps needed
- Linux: `sudo apt-get install portaudio19-dev`
- macOS: `brew install portaudio`

## Project Structure
- `src/` - Source code directory
  - `voice_recognition.py` - Speech recognition module
  - `model.py` - Transformer model implementation
  - `train.py` - Training script
  - `evaluate.py` - Evaluation metrics
  - `utils.py` - Utility functions
- `data/` - Dataset directory
- `models/` - Saved model checkpoints
- `notebooks/` - Jupyter notebooks for analysis

## Dataset
The project uses the [CLINC150](https://github.com/clinc/oos-eval) dataset for intent classification, which contains 150 intent classes across 10 domains, making it suitable for voice assistant applications.

## Model Architecture
The project implements a transformer-based architecture for intent classification with the following components:
- Pre-trained BERT model for text embeddings
- Custom classification head
- Fine-tuning on domain-specific data

## Evaluation Metrics
- Accuracy
- Confusion Matrix
- Precision, Recall, and F1 Score
- Intent Classification Report 