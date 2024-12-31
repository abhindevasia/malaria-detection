# Malaria Detection

This project uses **Convolutional Neural Networks (CNN)** and **OpenCV** to detect the presence of malaria in blood smear images. The trained model classifies blood samples as either **parasitic** (malaria detected) or **non-parasitic**. The trained images were obtained from https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria.

## Tools & Technologies

- **Python**: The programming language used for the implementation.
- **TensorFlow**: Deep learning framework for training and deploying the CNN model.
- **OpenCV**: A computer vision library for image processing.
- **Convolutional Neural Networks (CNN)**: The deep learning model used to classify images of blood smears.

## Features

- **Pre-trained Model**: A CNN model trained on a large dataset of blood smear images to detect malaria.
- **User-Friendly GUI**: A graphical user interface built with Tkinter, allowing users to easily upload and classify images.
- **Real-time Prediction**: The model predicts whether a blood smear contains malaria parasites when an image is uploaded.

## Prerequisites

To run the project, you need Python 3.8 or higher installed along with the required libraries.

## Installation

1. **Clone the repository**: 
   - Click the "Code" button on the repository page, then copy the link and run this command in your terminal or command prompt:
     ```bash
     git clone https://github.com/abhindevasia/malaria-detection.git
     ```

2. **Install the required dependencies**:
   - Install the libraries by running:
     ```bash
     pip install -r requirements.txt
     ```

3. **Download the pre-trained model**:

## How to Run

1. After installing the dependencies, open the repository folder in your Python IDE or file explorer.

2. **Run the application**:
   - Open the script `malaria_detection_prediction.py` (or your main Python script).
   - Run the script with the command:
     ```bash
     python malaria_detection_prediction.py
     ```

3. **Use the GUI**:
   - Click the "Browse A File" button to select a blood smear image (PNG, JPG, or JPEG).
   - Click the "Submit" button to get the prediction.
   - The result will show either "Parasitic" (malaria detected) or "Non-Parasitic."

## Image Requirements

- The images should be clear, high-quality blood smear images.
- The model expects images to be resized to **64x64 pixels**. (If your images are larger, resize them before uploading or modify the code to handle different image sizes.)

## Model Training (Optional)

For those interested in training the model from scratch:

>> some of the images from the data folder may have been truncated so try downloading from https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria
1. Prepare your dataset with labeled images (Parasitized/Non-Parasitized).
2. Modify the `malaria_detection.ipynb` script to load your dataset.
3. Train the model by running:
   ```bash
   python malaria_detection.py
