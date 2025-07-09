import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
import re
import os


# Download stopwords once
nltk.download('stopwords')

# Get absolute path to models directory
base_path = os.path.dirname(__file__)
model_path = os.path.join(base_path, "models", "model1.pkl")
vectorizer_path = os.path.join(base_path, "models", "vectorizer.pkl")

# Load them safely
model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

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