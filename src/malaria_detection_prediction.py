# Importing Libraries
import tensorflow as tf
from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from PIL import Image, ImageOps
import numpy as np
import os

# Basic Interface to predict the status of a single image
class Root(Tk):
    def __init__(self):
        super(Root, self).__init__()
        self.title("Parasitic Detection GUI")
        self.minsize(640, 400)

        # Load the saved model
        self.model = self.load_model("models/malaria_detect_model.keras")

        # Set up the label frame
        self.labelFrame = ttk.LabelFrame(self, text="Open File")
        self.labelFrame.grid(column=0, row=1, padx=20, pady=20)

        # Initialize filename
        self.filename = ""

        # Add buttons and labels
        self.button()
        self.submit_button()
        self.result_label = ttk.Label(self.labelFrame, text="")
        self.result_label.grid(column=1, row=3, pady=10)

    def load_model(self, model_path):
        try:
            # Load the saved Keras model
            model = tf.keras.models.load_model(model_path)
            print("Model loaded successfully")
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            self.result_label.configure(text="Error loading model.")
            return None

    def button(self):
        self.browse_button = ttk.Button(self.labelFrame, text="Browse A File", command=self.fileDialog)
        self.browse_button.grid(column=1, row=1)

    def fileDialog(self):
        # Open file dialog to select an image
        self.filename = filedialog.askopenfilename(
            initialdir="/", title="Select An Image File",
            filetypes=(("Image files", "*.png;*.jpg;*.jpeg"), ("All files", "*.*"))
        )
        
        if self.filename:  # Check if a file is selected
            self.result_label.configure(text=f"Selected file: {os.path.basename(self.filename)}")
        else:
            self.result_label.configure(text="No file selected.")

    def preprocess_image(self, file_path):
        try:
            # Load the image and preprocess it
            image = Image.open(file_path)
            # Resize the image to the expected input size (e.g., 64x64 or 224x224 based on model)
            image = image.resize((64, 64), resample=Image.Resampling.LANCZOS)  # Change to 224 if needed
            image = np.array(image) / 255.0  # Normalize to [0, 1]
            image = np.expand_dims(image, axis=0)  # Add batch dimension
            return image
        except Exception as e:
            print(f"Error loading image: {e}")
            self.result_label.configure(text=f"Error loading image: {e}")
            return None


    
    def get_prediction(self):
        if not self.filename:
            self.result_label.configure(text="Please select an image file.")
            return

        self.result_label.configure(text="Predicting... Please wait.")  # Indicate progress

        try:
            # Preprocess the image
            image = self.preprocess_image(self.filename)
            if image is not None:
                # Perform prediction using the loaded model
                prediction = self.model.predict(image)
                result = "Non-Parasitic" if prediction[0][0] < 0.5 else "Parasitic"
                self.result_label.configure(text=f"Prediction: {result}")
        except Exception as e:
            print(f"Error during prediction: {e}")
            self.result_label.configure(text=f"Error: {e}")

    def submit_button(self):
        self.submit_button = ttk.Button(self.labelFrame, text="Submit", command=self.get_prediction)
        self.submit_button.grid(column=1, row=2, pady=10)

if __name__ == "__main__":
    root = Root()
    root.mainloop()
