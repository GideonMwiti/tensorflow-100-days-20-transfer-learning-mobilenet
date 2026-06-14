# Project 20: Transfer Learning with MobileNet

Use MobileNetV2 from TensorFlow Hub as a frozen feature extractor and train a custom dense classifier on a small dataset (Cats vs Dogs).

## Dataset
This project uses programmatically generated synthetic images of Cats and Dogs at 224x224 resolution:
- **Cats**: Round faces with whiskers and pointy triangular ears.
- **Dogs**: Round faces with a snout and floppy hanging ear ellipses on the sides.
- Total samples: 400 images (200 cats, 200 dogs).

The dataset is loaded and split dynamically (80/20 train/validation) using `tf_keras.utils.image_dataset_from_directory`.

## Model Architecture
The transfer learning model features:
1. **Input Layer**: Shape `(224, 224, 3)`.
2. **Rescaling Layer**: Normalizes pixel values from `[0, 255]` to `[0.0, 1.0]`.
3. **MobileNetV2 Feature Extractor**: Frozen weights loaded from TensorFlow Hub (`trainable=False`).
4. **Dense Classifier**:
   - `Dense` layer (64 units, ReLU activation)
   - `Dropout` (rate: 0.3)
   - `Dense` layer (1 unit, Sigmoid activation for binary classification)

## How to Run
Run the training and evaluation script:
```powershell
python main.py
```

## Results
The script saves a plot named `transfer_learning_results.png` showing:
- Loss and accuracy curves.
- A prediction showcase of 12 test samples with green/red borders indicating correctness.

## Repository Details
- **Name**: `tensorflow-100-days-20-transfer-learning-mobilenet`
- **Description**: Transfer learning implementation using MobileNetV2 from TensorFlow Hub as a frozen feature extractor and a custom classification head in tf_keras, trained on a synthetic Cats vs Dogs dataset.
- **Topics**: `tensorflow`, `deep-learning`, `transfer-learning`, `mobilenetv2`, `tf-hub`, `binary-classification`, `tf-keras`
