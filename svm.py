import streamlit as st
import soundfile as sf
import librosa
import numpy as np
import pickle
import  sounddevice as sd

st.markdown(
    """
    <style>
    .text-ke-tengah {
        display: flex;
        justify-content: center;
        align-items: center;
        text-align:center;
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown('<div class="text-ke-tengah"><h1>Analisis Audio Dengan Menggunakan SVM 🫶🏻 </h1></div>', unsafe_allow_html=True)

# File uploaded
uploaded_file = st.file_uploader("Uploud File Audio Anda", type=["wav", "mp3"])

# Function to extract MFCC features from audio files
def extract_features(file_path):
    audio, sample_rate = librosa.load(file_path)
    mfcc_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
    mfcc_scaled_features = np.mean(mfcc_features.T, axis=0)
    return mfcc_scaled_features

# Function to predict the emotion of a new audio file
def predict_emotion(file_path, model):
    # Extract features from the new audio file
    features = extract_features(file_path)

    # Reshape the features into a single sample
    features = features.reshape(1, -1)

    # Predict the emotion using the loaded SVM model
    emotion = model.predict(features)[0]

    return emotion

if uploaded_file is not None:
    audio, samplerate = sf.read(uploaded_file)
    st.audio(uploaded_file)
    # Define the path to your saved SVM model (.pkl file)
    model_path = "svm.pkl"

    # Path to the new audio file for prediction
    new_audio_path = uploaded_file

    # Load the trained SVM model from the .pkl file
    with open(model_path, 'rb') as file:
        trained_model = pickle.load(file)

    label_to_emotion = {0: 'sad', 1: 'happy'}

    # Predict the emotion of the new audio file using the loaded model
    predicted_emotion = predict_emotion(new_audio_path, trained_model)
    st.write("Predicted Emotion:", label_to_emotion[predicted_emotion])


