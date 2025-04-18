# 🎧 Music Genre Classifier

This project is a deep learning-based classifier that predicts the **genre of a song** using its **Mel spectrogram**.  
The model was trained on a small dataset (~100 songs across 3 genres) and built using `TensorFlow` and `Librosa`.

---

## 🧠 What It Does

- Converts `.mp3` or `.wav` audio files into Mel spectrograms
- Splits spectrograms into 128×128 patches
- Trains a CNN (Convolutional Neural Network) to classify each patch
- Predicts the genre of an entire song via majority voting / average probability across slices

---

## 📁 Project Structure
genre-classifier/
├── model/                             # Trained model file (.keras)
├── data/                              # Audio files organized by genre
├── notebooks/
│   └── genre_classifier.ipynb         # Full training & testing code
├── utils/
│   └── audio_processing.py            # Slicing & prediction functions
├── requirements.txt                   # Dependencies
└── README.md                          # You’re reading it!

---

## 📦 Trained Model & Dataset

Due to GitHub's file size limitations, the **trained model** and **audio dataset** are hosted on Google Drive:

🔗 [Download model + data (Google Drive)](https://drive.google.com/file/d/1VgtlQ9kYjX--JJ0GM6oAGSgc1AD_D_V6/view?usp=share_link)

### After downloading:

1. Extract the `.zip` file
2. Place the folders into your project directory:
genre-classifier/
├── model/genre_classifier_model.keras
└── data/  # Folder with genre subfolders and songs

---

## 🚀 How to Run

### 1. Install dependencies
pip install -r requirements.txt
### 2. Run the notebook to train or test
jupyter notebook notebooks/genre_classifier.ipynb
### 3. Predict genre of a new song
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from utils.audio_processing import predict_genre_for_song

model = load_model("model/genre_classifier_model.keras")
encoder = LabelEncoder()
encoder.classes_ = ['jazz', 'rnb', 'rock']  # Use your genre order

genre = predict_genre_for_song("data/test_song.mp3", model, encoder)
print("Predicted genre:", genre)
