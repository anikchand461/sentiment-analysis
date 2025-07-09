import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
import re


# Download stopwords once
nltk.download('stopwords')

# Load the trained model and vectorizer
model = joblib.load("models/model1.pkl")               # Trained classifier
vectorizer = joblib.load("models/vectorizer.pkl")     # CountVectorizer or TfidfVectorizer

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