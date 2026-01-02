# 🎵 Music Genre Classification System

An end-to-end Deep Learning application that classifies music into genres using **Convolutional Neural Networks (CNN)**. This project handles the entire pipeline: from raw audio preprocessing and feature extraction (MFCCs) to a real-time web interface for prediction.



## 🚀 Features
* **Audio Preprocessing:** Automated extraction of Mel-Frequency Cepstral Coefficients (MFCCs) using `librosa`.
* **Data Augmentation:** Segments long audio files into smaller chunks to expand the training dataset.
* **Deep Learning Model:** A 3-layer CNN architecture built with TensorFlow/Keras, featuring Batch Normalization and Dropout for robust learning.
* **Interactive Web App:** A Streamlit-based dashboard allowing users to:
    * Upload `.mp3` or `.wav` files.
    * Record live audio directly through the browser.
    * Visualize the audio's MFCC "fingerprint" via heatmaps.

---

## 🏗️ Project Structure
* `preprocess.py`: Processes the raw GTZAN dataset, segments audio, and saves MFCC features into a `data.json` file.
* `cnn_classifier.py`: Defines the CNN architecture, handles data loading, and contains the training logic.
* `web.py`: The Streamlit frontend script that integrates the trained model with a user-friendly UI.
* `music_genre_model.weights.h5`: Saved weights of the trained model (generated after training).

---

## 🛠️ Technical Stack
* **Audio Processing:** Librosa
* **Deep Learning:** TensorFlow, Keras
* **Numerical Computing:** NumPy, Scikit-learn
* **Frontend/UI:** Streamlit
* **Visualizations:** Matplotlib

---

## 🚦 Getting Started

### 1. Prerequisites
Ensure you have Python 3.8+ installed. Install the required dependencies:
```bash
pip install tensorflow librosa numpy scikit-learn streamlit audio-recorder-streamlit matplotlib
```
### 2. Dataset
This project is designed to work with the **GTZAN Genre Collection**.

1. **Download:** Obtain the dataset and place it in a folder named `genres_original1`.
2. **Structure:** The folder should contain subfolders for each genre as follows:
   ```text
   genres_original1/
   ├── blues/
   ├── classical/
   ├── country/
   ├── ...
   └── rock/
   ```
### 3. Preprocessing
  Run the preprocessing script to segment the audio and extract **Mel-Frequency Cepstral Coefficients (MFCCs)**. This script converts raw audio into a numerical format stored in `data.json`.

```bash
python preprocess.py
```
### 4. Training
   Train the CNN model using the extracted features. After training, the model weights will be saved as music_genre_model.weights.h5.
   ```bash
   python cnn_classifier.py
  ```
### 5. Running the Web App
  Launch the interactive Streamlit dashboard to test the model with your own audio files or live recordings.
```bash
streamlit run web.py
```
### 🧠 Model Architecture
The model interprets MFCC data as a grayscale image with the input shape (Time x Coefficients x 1).
  - **3 Convolutional Layers**: Extract spatial patterns and textures from the MFCC spectrograms.
  - **Max Pooling & Batch Normalization**: Used to downsample the feature maps and stabilize the learning process.
  - **Dense Hidden Layer**: 64 units with a Dropout (30%) layer to prevent overfitting.
  - **Softmax Output**: 10 units representing the probability distribution across the genres.

### 📊 Genres Covered
The system is trained to identify the following 10 musical genres:
🎸 Rock🎻 Classical🤠 Country🕺 Disco🎤 Hiphop🎷 
Jazz🤘 Metal🎹 Pop🏝️ Reggae🎷 Blues📝 

### About the MFCCs
Mel-Frequency Cepstral Coefficients (MFCCs) are a representation of the short-term power spectrum of a sound.Unlike a standard spectrogram, MFCCs map the powers onto the nonlinear Mel scale of frequency. This scale is designed to mimic the human ear's sensitivity to different frequencies (being more sensitive to changes at lower frequencies than higher ones), making it one of the most effective features for audio and speech classification tasks.
