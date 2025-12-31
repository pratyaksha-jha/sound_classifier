import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
import librosa
import io
import matplotlib.pyplot as plt
from audio_recorder_streamlit import audio_recorder
from cnn_classifier import build_model 

GENRES = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]
DATA_SHAPE = (130, 13) 
def process_input(audio_file):
    """Transforms raw audio into MFCC features for the CNN."""
    #ensure librosa reads everything
    audio_file.seek(0)
    
    signal, sr = librosa.load(audio_file, sr=22050)
    
    #MFCCs
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13, n_fft=2048, hop_length=512)
    mfcc = mfcc.T 
    if len(mfcc) > 130:
        mfcc = mfcc[:130, :]
    else:
        mfcc = np.pad(mfcc, ((0, 130 - len(mfcc)), (0, 0)), mode='constant')
        
    mfcc = mfcc[np.newaxis, ..., np.newaxis]
    return mfcc

st.set_page_config(page_title="Music Genre Classifier", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #F0F2F6; color: #1E1E1E; }
    h1, h2, h3 { color: #2E4053 !important; }
    .stButton>button { 
        width: 100%; border-radius: 10px; height: 3em; 
        background-color: #2E4053; color: white; font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🎵 Music Genre Classification System")
app_mode = st.sidebar.selectbox("Navigation", ["Home","Genre Classification", "About Us"])

@st.cache_resource
def load_trained_model():
    # Build architecture
    model = build_model(input_shape=(130, 13, 1))
    # Load weights
    try:
        model.load_weights("music_genre_model.weights.h5")
        return model
    except Exception as e:
        st.error(f"Error: Could not load 'music_genre_model.h5'. {e}")
        return None

model = load_trained_model()

if app_mode == "Home":
    st.markdown("## Welcome to the AI Music Lab!")
    st.write("This application uses Deep Learning to analyze audio signals and identify musical genres.")
    st.image("https://blog.neurotech.africa/content/images/2022/02/audio.jpg", use_container_width=True)

elif app_mode == "Genre Classification":
    st.header("Analyze Your Sound")
    
    if model is None:
        st.warning("Model weights not loaded. Please ensure 'music_genre_model.h5' is in the root directory.")
        st.stop()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("1. Upload Audio")
        uploaded_file = st.file_uploader("Upload an MP3 or WAV file", type=["mp3", "wav"])
    with col2:
        st.subheader("2. Record Audio")
        recorded_audio = audio_recorder(text="Click to record", icon_size="2x", neutral_color="#2E4053")

    final_audio = None
    if uploaded_file is not None:
        final_audio = uploaded_file
    elif recorded_audio is not None:
        final_audio = io.BytesIO(recorded_audio)

    if final_audio:
        st.audio(final_audio)
        
        if st.button("Predict Genre"):
            with st.spinner("Processing audio signal..."):
                try:
                    #feature extraction
                    input_mfcc = process_input(final_audio)
                    
                    #model prediction
                    prediction = model.predict(input_mfcc)
                    predicted_idx = np.argmax(prediction, axis=1)[0]
                    confidence = np.max(prediction) * 100
                    
                    #display results
                    st.success(f"### Predicted Genre: **{GENRES[predicted_idx].upper()}**")
                    st.metric("Confidence Level", f"{confidence:.2f}%")
                    st.progress(int(confidence))
                    
                    st.divider()
                    
                    # visualization
                    st.write("#### MFCC Spectrogram:")
                    
                    viz_data = input_mfcc[0, :, :, 0].T 
                    
                    fig, ax = plt.subplots(figsize=(10, 4))
                    img = ax.imshow(viz_data, aspect='auto', origin='lower', cmap='magma')
                    ax.set_title(f"MFCC Heatmap: {GENRES[predicted_idx].capitalize()}")
                    ax.set_xlabel("Time Frames")
                    ax.set_ylabel("MFCC Coefficients")
                    plt.colorbar(img, ax=ax)
                    
                    st.pyplot(fig)
                    
                except Exception as e:
                    st.error(f"Processing Error: {e}")

elif app_mode == "About Us":
    st.markdown("""
    ### Project Overview
    This tool extracts **Mel-Frequency Cepstral Coefficients (MFCCs)** from audio files. 
    MFCCs are essentially a "fingerprint" of the sound, capturing the power spectrum of the audio.
    """)
    
    st.markdown("""
    **Architecture:** Convolutional Neural Network (CNN)  
    **Framework:** TensorFlow/Keras & Librosa  
    **Dataset:** GTZAN Genre Collection
    """)