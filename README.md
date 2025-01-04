# Gender Classification Using CNNs

This project implements a **Convolutional Neural Network (CNN)** for gender classification based on facial images. The model is trained to differentiate between male and female faces using features extracted from facial images. The project includes preprocessing, model training, testing, and predictions.

## Features
- **Image Preprocessing**: Automated image processing for feature extraction using a CNN preprocessor.
- **Custom Neural Network**: Implementation of forward and backward propagation for gender classification.
- **Training & Testing**: Train the model on labeled data and evaluate its performance on unseen images.
- **Error Handling**: Robust handling of missing or improperly formatted images.

## Tools and Technologies
- **Python**: Main programming language.
- **NumPy**: For numerical operations and matrix computations.
- **Pickle**: To save and load trained model parameters.
- **Jupyter Notebook**: For project development and visualization.

## File Structure
- `model_training.ipynb`: Contains the training code and saves the model parameters.
- `model_testing.ipynb`: Used to test the trained model with new images.
- `cnn_preprocessor.py`: A module for preprocessing images using CNN features.
- `model_parameters.pkl`: Serialized file containing trained model parameters.

## Sample Result

The model predicts the gender of a given facial image:

- **Input**: An image of a face (e.g., `1 (10).png` from the dataset).
- **Output**: 
    - **Male** if the model identifies the person as male.
    - **Female** if the model identifies the person as female.

Example:
```bash
Prediction for image 'content/male_faces/1 (10).png': Male
