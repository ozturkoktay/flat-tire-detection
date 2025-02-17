# Flat Tire Detection using Deep Learning
This is the official repository for the ICAIAME (International Conference on Artificial Intelligence and Applied Mathematics in Engineering) 2022 paper "Transfer Learning Based Flat Tire Detection by Using RGB Images" by Oktay Ozturk and [Batuhan Hangun](https://github.com/batuhanhangun).


## Overview
This project implements a deep learning-based approach for detecting flat tires from images using transfer learning. The model is trained using the **Xception** architecture, leveraging **ImageNet** pre-trained weights to improve performance. The dataset is augmented to enhance generalization, and training is performed with an adaptive learning rate and early stopping.

## Dataset
The [Full vs Flat Tire](https://www.kaggle.com/datasets/rhammell/full-vs-flat-tire-images) dataset consists of images of **flat tires** and **full tires**, which are divided into three sets:
- **Training Set**: Used for model training
- **Validation Set**: Used for hyperparameter tuning
- **Test Set**: Used for performance evaluation

The dataset is stored as a ZIP file in Google Drive and extracted during execution.

## Dependencies
Ensure you have the following Python libraries installed:
```bash
pip install tensorflow torch numpy matplotlib imutils scikit-learn
```

## Model Architecture
The model uses the **Xception** architecture with the following modifications:
- The base model is **frozen** for initial layers and fine-tuned for deeper layers.
- **Global Average Pooling** is applied to the output layer.
- Fully connected layers with **Dropout** are added to reduce overfitting.
- The final layer is a **binary classifier** with a sigmoid activation function.

## Data Augmentation
Data augmentation techniques applied to the training images include:
- **Rotation** up to 270 degrees
- **Brightness variation**
- **Horizontal and vertical flipping**
- **Width and height shifts**

## Training
- Optimizer: **Adam** with a learning rate of **0.0001**
- Loss Function: **Binary Crossentropy**
- Metrics: **Accuracy and Mean Squared Error (MSE)**
- Callbacks:
  - **Early Stopping** (stops training when validation loss stops improving)
  - **Reduce Learning Rate on Plateau** (reduces learning rate when validation loss plateaus)

## Training Command
To train the model, simply execute the script in **Google Colab**:
```python
history = model.fit(
    trainGen,
    validation_data=valGen,
    callbacks=[early_stop, reduce_lr],
    epochs=T_EPOCHS
)
```

## Evaluation
After training, the model is evaluated on the test set using:
- **Classification Report** (Precision, Recall, F1-score, Accuracy)
- **Confusion Matrix**
- **Training and Validation Accuracy/Loss Plots**

## Output
- A confusion matrix and classification report are printed.
- A visualization of training and validation accuracy/loss is saved as `Xception-Results.png`.
- Sample augmented images are saved as `sample_augmented_images.png`.

## Usage
This code can be adapted for **real-time defect detection** using a camera in an industrial setup by:
- Replacing the dataset with live camera input.
- Deploying the trained model on an **edge device**.

## Citation
```
@InProceedings{10.1007/978-3-031-31956-3_22,
            author="Ozturk, Oktay
            and Hangun, Batuhan",
            title="Transfer Learning Based Flat Tire Detection by Using RGB Images",
            booktitle="4th International Conference on Artificial Intelligence and Applied Mathematics in Engineering",
            year="2023",
            publisher="Springer International Publishing",
            address="Cham",
            pages="264--273",
            isbn="978-3-031-31956-3"
}
```
