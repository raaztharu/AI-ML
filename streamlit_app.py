import streamlit as st
import numpy as np
import re
import contractions
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import pickle

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Load tokenizer and model
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

model = load_model("best_model.h5")

# Preprocessing function
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = contractions.fix(text)
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r"@\w+|#\w+", '', text)
    text = re.sub(r"[^a-z\s]", '', text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# GUI
st.title("🌟 Real-Time Hotel Rating Predictor")
st.write("Enter a review and get a **star rating (1–5)** prediction.")

user_input = st.text_area("Your Review")

if st.button("Predict Rating"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        cleaned = clean_text(user_input)
        sequence = tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(sequence, maxlen=200)  # use your actual max length
        prediction = model.predict(padded)
        predicted_rating = np.argmax(prediction) + 1  # Convert from 0-indexed

        st.success(f"⭐ Predicted Rating: **{predicted_rating}** star(s)")
