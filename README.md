# Facial-Emotion-Recognition-using-CNN-

Facial emotion recognition project using a CNN trained on FER2013. The model classifies 48×48 grayscale face images into seven emotions and includes preprocessing, training, evaluation, confusion matrix analysis, and discussion of model limitations and dataset bias.

## Dataset Setup

This project uses the FER2013 dataset from Kaggle: https://www.kaggle.com/datasets/msambare/fer2013

The raw dataset is not stored in this repository because it is large. Download the images manually from Kaggle and place them in the local `Data/train` and `Data/test` folders.

Expected folder structure:

```
Data/
  train/
    angry/
    disgust/
    fear/
    happy/
    neutral/
    sad/
    surprise/
  test/
    angry/
    disgust/
    fear/
    happy/
    neutral/
    sad/
    surprise/
```

Once the dataset is available locally, run the data preparation script:

```bash
python src/data_preparation.py --build-processed
```

This validates the dataset structure, counts images per class, checks file types and image dimensions, and creates `Data/processed` with 48×48 grayscale images without changing the original data.
