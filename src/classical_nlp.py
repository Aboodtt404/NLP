import re
import string
from typing import List, Dict, Tuple
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')


class ClassicalNLPProcessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.tfidf = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 3),
            stop_words='english'
        )
        self.classical_classifier = MultinomialNB()

        # Regular expression patterns for common intents
        self.patterns = {
            'search_images': r'(?:show|find|search|get|display).*(?:image|picture|photo|pic)s?\s+(?:of|about|for)?\s+(.*)',
            'search_videos': r'(?:show|find|search|get|display).*(?:video|clip|movie)s?\s+(?:of|about|for)?\s+(.*)',
            'search_news': r'(?:show|find|search|get).*(?:news|article)s?\s+(?:of|about|for)?\s+(.*)',
            'weather': r'(?:what|how).*(?:weather|temperature|forecast).*(?:in|at|for)?\s*(.*)',
            'time': r'(?:what|tell|show).*(?:time|clock).*(?:is it|now|current)',
            'reminder': r'(?:remind|remember|notification).*(?:to|about|that)\s+(.*)',
            'play_music': r'(?:play|start|begin).*(?:music|song|track|playlist)\s*(.*)',
            'volume': r'(?:change|adjust|set|increase|decrease).*(?:volume|sound|audio)\s*(.*)',
        }

    def preprocess_text(self, text: str) -> str:
        """Apply classical NLP preprocessing steps."""
        # Convert to lowercase
        text = text.lower()

        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))

        # Tokenize
        tokens = word_tokenize(text)

        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(
            token) for token in tokens if token not in self.stop_words]

        return ' '.join(tokens)

    def extract_entities(self, text: str, intent: str) -> Dict[str, str]:
        """Extract entities based on intent using regex patterns."""
        entities = {}

        if intent in self.patterns:
            match = re.match(self.patterns[intent], text.lower())
            if match and match.groups():
                entities['target'] = match.groups()[0].strip()

        return entities

    def pattern_match(self, text: str) -> Tuple[str, float, Dict[str, str]]:
        """Match text against regex patterns to determine intent."""
        text = text.lower()

        for intent, pattern in self.patterns.items():
            match = re.match(pattern, text)
            if match:
                entities = self.extract_entities(text, intent)
                return intent, 0.9, entities  # High confidence for pattern matches

        return None, 0.0, {}

    def train_classical_model(self, texts: List[str], labels: List[str]):
        """Train a classical ML model (TF-IDF + Naive Bayes)."""
        # Preprocess all texts
        processed_texts = [self.preprocess_text(text) for text in texts]

        # Create TF-IDF features
        X = self.tfidf.fit_transform(processed_texts)

        # Train the classifier
        self.classical_classifier.fit(X, labels)

    def predict_classical(self, text: str) -> Tuple[str, float]:
        """Make predictions using the classical ML model."""
        # Preprocess the input text
        processed_text = self.preprocess_text(text)

        # Transform using TF-IDF
        X = self.tfidf.transform([processed_text])

        # Get prediction and probability
        prediction = self.classical_classifier.predict(X)[0]
        probability = max(self.classical_classifier.predict_proba(X)[0])

        return prediction, probability

    def analyze_text(self, text: str) -> Dict:
        """Comprehensive text analysis using classical NLP techniques."""
        # Preprocess text
        processed_text = self.preprocess_text(text)

        # Try pattern matching first
        pattern_intent, pattern_conf, entities = self.pattern_match(text)

        # Get classical ML prediction
        ml_intent, ml_conf = self.predict_classical(text)

        # Basic text statistics
        stats = {
            'word_count': len(word_tokenize(text)),
            'char_count': len(text),
            'processed_text': processed_text
        }

        # Combine results
        result = {
            'pattern_match': {
                'intent': pattern_intent,
                'confidence': pattern_conf,
                'entities': entities
            },
            'ml_classification': {
                'intent': ml_intent,
                'confidence': ml_conf
            },
            'text_stats': stats
        }

        return result


def combine_predictions(classical_result: Dict, transformer_intent: str, transformer_conf: float) -> Tuple[str, float]:
    """Combine predictions from classical and transformer-based approaches."""
    pattern_intent = classical_result['pattern_match']['intent']
    pattern_conf = classical_result['pattern_match']['confidence']
    ml_intent = classical_result['ml_classification']['intent']
    ml_conf = classical_result['ml_classification']['confidence']

    # If pattern matching found a strong match, prefer it
    if pattern_intent and pattern_conf > 0.8:
        return pattern_intent, pattern_conf

    # If transformer is very confident, use its prediction
    if transformer_conf > 0.9:
        return transformer_intent, transformer_conf

    # If classical ML is more confident than transformer, use it
    if ml_conf > transformer_conf:
        return ml_intent, ml_conf

    # Default to transformer prediction
    return transformer_intent, transformer_conf
