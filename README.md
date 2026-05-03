# Facial Emotion Recognition using CNN on FER2013

## Project Overview

This project implements a convolutional neural network (CNN) for facial emotion recognition using the FER2013 dataset. The objective is to classify 48×48 grayscale face images into seven emotion categories:

- angry
- disgust
- fear
- happy
- neutral
- sad
- surprise

This is a deep learning project built with TensorFlow/Keras. It includes dataset validation, exploratory analysis, model training, evaluation, and an integrated pipeline to run the full workflow from data preparation through evaluation.

## Team Contributions

- **Isaac:** Dataset preparation, validation, data exploration, and pipeline integration.
- **Alhagie:** Model architecture and integration.
- **Selvi:** Training pipeline.
- **Mahsa:** Evaluation and results visualization.

## Dataset

The dataset used in this project is **FER2013**, a well-known facial emotion recognition dataset. All images are grayscale and resized to **48×48 pixels**.

Expected dataset structure:

```bash
data/
├── train/
│   ├── angry/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── neutral/
│   ├── sad/
│   └── surprise/
└── test/
    ├── angry/
    ├── disgust/
    ├── fear/
    ├── happy/
    ├── neutral/
    ├── sad/
    └── surprise/
```

The dataset may be obtained from Kaggle:

https://www.kaggle.com/datasets/msambare/fer2013

If the `Data` folder is already included in the repository, please verify that it follows this expected structure before running any scripts.

## Dataset Preparation and Validation

The script `src/data_preparation.py` performs the following tasks:

- validates the presence of `Data/train` and `Data/test`
- verifies that all class folders exist
- counts images for each emotion class
- checks that image files are readable
- verifies grayscale compatibility and 48×48 dimensions
- optionally builds a processed dataset for training

Use the following commands:

```bash
python -m src.data_preparation --config configs/base.yaml
```

```bash
python -m src.data_preparation --config configs/base.yaml --build-processed
```

## Data Exploration

The script `src/data_exploration.py` analyzes the dataset and generates visual summaries. It typically:

- summarizes class distribution
- creates class distribution plots
- saves sample image visualizations
- stores outputs in `results/`

Run it with:

```bash
python -m src.data_exploration
```

## Model Training

The script `src/train.py` trains the CNN model using TensorFlow/Keras. Training includes data generator setup, model construction, and checkpointing.

Run training with:

```bash
python -m src.train
```

Training outputs, model checkpoints, and training history are saved in the `results/` directory.

## Evaluation

The script `src/evaluate.py` evaluates the trained model and generates performance metrics and visualizations. This may include confusion matrix plots and classification summaries.

Run evaluation with:

```bash
python -m src.evaluate
```

## Prediction

If implemented in this project, `src/predict.py` can be used to run inference on new images.

Use the following command:

```bash
python -m src.predict
```

## Running the Full Pipeline

The file `run_pipeline.py` executes the project in the correct order:

1. Data preparation
2. Data exploration
3. Model training
4. Evaluation

Run the full pipeline with:

```bash
python run_pipeline.py
```

Optional flags allow skipping long steps:

```bash
python run_pipeline.py --skip-train
python run_pipeline.py --skip-evaluate
python run_pipeline.py --skip-train --skip-evaluate
python run_pipeline.py --build-processed
```

Skipping training is useful when model training is time-consuming and a previously trained model is already available.

## Installation

Recommended environment setup for Windows PowerShell:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Requirements

Major dependencies include:

- TensorFlow/Keras
- NumPy
- pandas
- matplotlib
- scikit-learn
- Pillow
- PyYAML

## Results

Generated outputs are stored in the `results/` directory. Expected artifacts include:

- class distribution plots
- sample image visualizations
- training history plots
- confusion matrix visualizations
- saved model file

## Notes and Troubleshooting

- Always run commands from the project root directory.
- If imports fail, use `python -m src.script_name` rather than executing scripts directly.
- Ensure `configs/base.yaml` exists before running pipeline or script commands.
- Ensure `Data/train` and `Data/test` exist and follow the required folder structure.
- TensorFlow warnings related to oneDNN are usually informational and not fatal.

## Conclusion

This repository provides a full workflow for facial emotion recognition using a CNN and the FER2013 dataset. The process covers dataset preparation, exploratory analysis, model training, evaluation, and optional prediction, making it suitable for academic presentation and practical experimentation.
