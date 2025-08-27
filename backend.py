from flask import Flask, request, jsonify
import pickle, re
from apscheduler.schedulers.background import BackgroundScheduler
import train_ai # import your train.py module

app = Flask(__name__)

# Load model & vectorizer at startup
def load_model():
    global model, vectorizer
    with open("sentiment_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    print("✅ Model loaded successfully!")

def clean_text(text):
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return ' '.join(text.lower().split())

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    text = data.get("review", "")
    if not text:
        return jsonify({"error": "No review provided"}), 400

    X = vectorizer.transform([clean_text(text)])
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0].max()
    sentiment = "Positive" if pred == 1 else "Negative"

    return jsonify({"sentiment": sentiment, "confidence": round(prob*100, 2)})

# Scheduler to retrain model every 1 hour
def retrain_job():
    print("🔄 Retraining AI model...")
    train_ai.train_and_save_model()
    load_model()

scheduler = BackgroundScheduler()
scheduler.add_job(retrain_job, 'interval', hours=1)
scheduler.start()

# Start server
if __name__ == "__main__":
    load_model()
    app.run(debug=True, port=5000)