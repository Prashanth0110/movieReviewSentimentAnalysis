import pickle
import re

def load_model():
    """Load the trained AI model and vectorizer"""
    try:
        with open('best_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        return model, vectorizer, True
    except FileNotFoundError:
        return None, None, False


def clean_text(text):
    """Clean text for model prediction"""
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    text = ' '.join(text.split())
    return text


def predict_sentiment(text, model, vectorizer):
    """Predict sentiment of given text"""
    if len(text.strip()) == 0:
        return None, None
    
    cleaned_text = clean_text(text)
    if len(cleaned_text.strip()) == 0:
        return None, None
    
    text_vector = vectorizer.transform([cleaned_text])
    prediction = model.predict(text_vector)[0]
    probability = model.predict_proba(text_vector)[0]
    
    sentiment = "Positive" if prediction == 1 else "Negative"
    confidence = max(probability) * 100
    
    return sentiment, confidence


def load_sample_data():
    """Load sample movie reviews for testing"""
    sample_reviews = {
        "Positive Reviews": [
            "This movie was absolutely incredible! The cinematography was breathtaking and the acting was phenomenal. I was completely captivated from start to finish.",
            "One of the best films I've ever seen! The storyline was compelling, the characters were well-developed, and the ending was perfect. Highly recommended!"
        ],
        "Negative Reviews": [
            "Terrible movie, complete waste of time and money. The plot was confusing, the acting was poor, and the dialogue was cringe-worthy.",
            "One of the worst films I've ever watched. Boring storyline, bad character development, and terrible special effects. Very disappointing."
        ],
        "Mixed/Neutral Reviews": [
            "The movie was okay, nothing special but not terrible either. Some parts were interesting while others were boring.",
            "Average film with decent acting but a predictable plot. Worth watching if you have nothing else to do."
        ]
    }
    return sample_reviews