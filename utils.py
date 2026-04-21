import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

def clean_text(text):
    """Professional NLP Preprocessing Pipeline"""
    # Lowercase
    text = str(text).lower()
    # Remove URLs, Mentions, and Hashtags
    text = re.sub(r'http\S+|@\w+|#\w+', '', text)
    # Remove Punctuation and special characters
    text = re.sub(r'[^\w\s]', '', text)
    # Tokenization and Stopword removal
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    words = text.split()
    cleaned_words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    
    return " ".join(cleaned_words)