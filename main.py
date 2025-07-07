import streamlit as st
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
import re

# Download stopwords once
nltk.download('stopwords')

# Load the trained model and vectorizer
model = joblib.load("model.pkl")               # Trained classifier
vectorizer = joblib.load("vectorizer.pkl")     # CountVectorizer or TfidfVectorizer

# Initialize stemmer and stopwords
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Preprocessing functions
def clean_html(text):
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)

def convert_lower(text):
    return text.lower()

def remove_special(text):
    return ''.join([ch if ch.isalnum() else ' ' for ch in text])

def remove_stopwords(text):
    return [word for word in text.split() if word not in stop_words]

def stem_words(text):
    return [ps.stem(word) for word in text]

def join_back(words):
    return ' '.join(words)

def preprocess(text):
    text = clean_html(text)
    text = convert_lower(text)
    text = remove_special(text)
    words = remove_stopwords(text)
    words = stem_words(words)
    return join_back(words)

# Streamlit app UI
st.title("Sentiment Analysis App")
st.write("Enter a sentence and get its sentiment prediction.")

# Text input
user_input = st.text_area("Enter your sentence here:")

# Prediction button
if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        # Preprocess the user input
        processed_text = preprocess(user_input)

        # Vectorize the preprocessed text
        input_vector = vectorizer.transform([processed_text])

        # Predict sentiment
        prediction = model.predict(input_vector)[0]

        # Map numeric prediction to label
        sentiment_label = "Positive" if prediction == 1 else "Negative"

        # Output
        st.success(f"Predicted Sentiment: **{sentiment_label}**")