# Hand Gesture Recognition with TensorFlow

This repository contains a deep learning model for recognizing hand gestures using TensorFlow and OpenCV. The project includes a custom-built Convolutional Neural Network (CNN) that classifies various hand gestures in real-time from image or video input, making it ideal for interactive applications.

## Project Structure

- .ipynb_checkpoints/: Contains Jupyter notebook checkpoints.
- utils/: Utility functions used for preprocessing and model operations.
- gesture_recognition.py: Python script to run the hand gesture recognition model.
- Hand_Gesture_Recognition.ipynb: Jupyter notebook used to train, test, and analyze the gesture recognition model.
- requirements.txt: Contains all necessary dependencies to run the project.

## Example Gesture Classification

Here’s images of the demo:

![Demo Image #1](demo.png 'Examples of Hand Gesture Classification #1')

![Demo Image #2](demo-1.png 'Examples of Hand Gesture Classification #2')

![Demo Image #3](demo-2.png 'Examples of Hand Gesture Classification #3')

## Getting Started

Follow these instructions to set up and run the Hand Gesture Recognition model.

### Step 1: Clone the Repository

```
git clone https://github.com/ehvenga/handgesture.recognition-tensorflow.git
cd handgesture.recognition-tensorflow
```

### Step 2: Install Dependencies

Install the required dependencies listed in `requirements.txt` by running:

```
pip install -r requirements.txt
```

### Step 3: Train the Model

You can use the provided Jupyter notebook `Hand_Gesture_Recognition.ipynb` to train the model. Open the notebook in your favorite Jupyter environment and execute the cells to preprocess the data, train the model, and evaluate its performance.

Alternatively, if you have a pre-trained model, you can skip this step.

### Step 4: Run Gesture Recognition

To run the hand gesture recognition model on new images or video input, execute the Python script:

```
python gesture_recognition.py
```

You can modify the script to adjust the input source (e.g., from a video file or live webcam feed).

### Step 5: View Results

The results, such as predicted hand gestures and performance metrics, will be stored in the `results/` folder, and any image visualizations will be saved in the `resultsimg/` folder.

## Model Architecture

The model is built using a Convolutional Neural Network (CNN) architecture that efficiently processes image data for hand gesture classification. It has been designed to recognize multiple hand gestures with real-time performance using TensorFlow.

## Requirements

- Python 3.8+
- TensorFlow 2.x
- OpenCV (for video/image processing)
- NumPy
- Jupyter (if using the notebook)

GitHub Repository: [Hand Gesture Recognition](https://github.com/ehvenga/handgesture.recognition-tensorflow.git)
