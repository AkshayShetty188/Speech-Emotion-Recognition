import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# === Page Config ===
st.set_page_config(
    page_title="Speech Emotion Recognition",
    layout="centered",
    page_icon="🎙️",
)

# === Set Background Image ===
page_bg_img = f"""
<style>
.stApp {{
background-image: url("https://images.unsplash.com/photo-1602524201433-8d3ee6c8c10b?auto=format&fit=crop&w=1500&q=80");
background-size: cover;
background-attachment: fixed;
background-position: center;
color: white;
}}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# === Load model ===
model = load_model("models/best_model_rich.h5")

# === Emotion mapping ===
emotions = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
emoji_map = {
    "angry": "😠", "calm": "😌", "disgust": "🤢", "fearful": "😨",
    "happy": "😄", "neutral": "😐", "sad": "😢", "surprised": "😲"
}

# === Header ===
st.title("🎙️ Speech Emotion Recognition")
st.markdown("Upload a `.wav` file and we'll detect the emotion using a deep learning model!")

# === File upload ===
audio_file = st.file_uploader("Upload Audio File", type=["wav"])

if audio_file:
    st.audio(audio_file, format='audio/wav')
    st.info("Processing audio...")

    # === Feature Extraction ===
    signal, sr = librosa.load(audio_file, sr=22050, duration=3)
    if len(signal) < sr * 3:
        signal = np.pad(signal, (0, sr * 3 - len(signal)))

    mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=signal, sr=sr)
    mel = librosa.feature.melspectrogram(y=signal, sr=sr)
    contrast = librosa.feature.spectral_contrast(y=signal, sr=sr)
    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(signal), sr=sr)

    features = np.concatenate((mfccs, chroma, mel, contrast, tonnetz), axis=0).T
    features = features[:130].reshape(1, 130, features.shape[1])

    # === Prediction ===
    probs = model.predict(features)[0]
    top_idx = np.argmax(probs)
    top_emotion = emotions[top_idx]
    emoji = emoji_map[top_emotion]

    # === Output ===
    st.markdown(f"## 🎯 Predicted Emotion: **{top_emotion.upper()}** {emoji}")
    
    # === Interactive Probability Plot ===
    st.markdown("### 📊 Emotion Probabilities")
    st.bar_chart({emotions[i]: probs[i] for i in range(len(emotions))})

    # === MFCC Display (Optional) ===
    st.markdown("### 🎧 MFCC Visualization")
    fig, ax = plt.subplots()
    librosa.display.specshow(mfccs, x_axis='time', ax=ax)
    ax.set_title("MFCC")
    st.pyplot(fig)
