# 🎬 Movie Review AI Analyzer

A web-based AI application for sentiment analysis of movie reviews using Streamlit and machine learning models.

# 🚀 Project Overview

The Movie Review AI Analyzer predicts whether a movie review is positive or negative and provides a confidence score for each prediction. Users can:

Test their own movie reviews through an interactive web interface.

View analytics (charts and summaries) of AI predictions.

Run the AI models in the backend without manually executing Python scripts.

This project demonstrates a complete AI deployment workflow from training to production-ready UI.

# 🧩 Features

Interactive Web UI built with Streamlit

AI Backend using Logistic Regression, Naive Bayes, Random Forest

Text Preprocessing: cleaning and vectorizing IMDB reviews

Real-time Predictions: get sentiment and confidence instantly

Visual Confidence Gauge using Plotly

Automatic Model Loading: Streamlit fetches trained models from backend

Easy-to-use interface for non-technical users

# 🛠️ Technologies Used

Python 3.10+

Streamlit — frontend for interactive UI

Flask — backend API for AI predictions

scikit-learn — machine learning models

pandas & numpy — data processing

pickle — model serialization

Plotly — interactive charts

IMDB Dataset — labeled movie reviews

# Optional scripts (used during development / training):

train.py — trains AI models and saves best model + vectorizer

test.py — interactive testing for AI predictions

analyze.py — exploratory data analysis and evaluation

# ⚙️ How to Run
# 1️⃣ Clone the repository
git clone <your-repo-url>
cd MovieReviewAI

# 2️⃣ Install dependencies
pip install -r requirements.txt

# 3️⃣ Run the backend API
python backend.py

# 4️⃣ Run the Streamlit UI
streamlit run ui_app.py

# 5️⃣ Open your browser

Navigate to: http://localhost:8501 to use the AI Analyzer.

# 🧠 How It Works

Text preprocessing: HTML tags, punctuation, and extra spaces removed.

Vectorization: Reviews converted into numerical features with TF-IDF.

Model prediction: Trained ML model predicts sentiment and returns confidence.

UI visualization: Streamlit shows sentiment and confidence gauge.

# 📊 Example Prediction
Review	Sentiment	Confidence
This movie was amazing! Fantastic storyline.	😊 Positive	96%
Terrible film. Waste of money.	😞 Negative	92%

Interactive visualization with a gauge chart shows the confidence score dynamically.

# 📈 Future Enhancements

Add real-time training triggers to update model automatically.

Expand analytics dashboard with more charts and metrics.

Deploy on cloud platform for global access.

Add multi-language support for reviews.
