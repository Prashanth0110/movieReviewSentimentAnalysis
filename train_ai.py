import pandas as pd
import re
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import time

def clean_text(text):
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = ' '.join(text.lower().split())
    return text

def load_and_prepare_data():
    df = pd.read_csv("IMDB Dataset.csv")
    df['cleaned_review'] = df['review'].apply(clean_text)
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', min_df=5, max_df=0.8)
    X = vectorizer.fit_transform(df['cleaned_review'])
    y = df['sentiment'].map({'positive': 1, 'negative': 0})
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    with open("vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    
    return X_train, X_test, y_train, y_test

def train_and_save_model():
    X_train, X_test, y_train, y_test = load_and_prepare_data()
    
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, solver='liblinear', random_state=42),
        "Naive Bayes": MultinomialNB(),
        "Random Forest": RandomForestClassifier(n_estimators=50, n_jobs=-1, random_state=42)
    }

    best_model = None
    best_acc = 0

    for name, model in models.items():
        print(f"Training {name}...")
        start_time = time.time()
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        print(f"{name} test accuracy: {acc*100:.2f}% (Time: {time.time()-start_time:.1f}s)")
        if acc > best_acc:
            best_acc = acc
            best_model = model

    with open("sentiment_model.pkl", "wb") as f:
        pickle.dump(best_model, f)

    print(f"✅ Best model saved with accuracy: {best_acc*100:.2f}%")

if __name__ == "__main__":
    train_and_save_model()