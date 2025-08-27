import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import re

def load_model_and_data():
    """Load the trained model and original data"""
    try:
        # Load model and vectorizer
        with open('best_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        
        # Load original data
        df = pd.read_csv('IMDB Dataset.csv')
        
        print("✅ Model and data loaded successfully!")
        return model, vectorizer, df
    
    except FileNotFoundError:
        print("❌ Required files not found!")
        print("💡 Please run train_ai.py first")
        return None, None, None

def analyze_important_words(model, vectorizer):
    """Find out which words the AI thinks are most important"""
    
    print("🔍 Analyzing Important Words")
    print("=" * 35)
    
    # Get feature names (words)
    feature_names = vectorizer.get_feature_names_out()
    
    # Get model coefficients (importance scores)
    if hasattr(model, 'coef_'):
        # For linear models like Logistic Regression
        coef = model.coef_[0]
    elif hasattr(model, 'feature_importances_'):
        # For tree-based models like Random Forest
        coef = model.feature_importances_
    else:
        print("⚠️  This model type doesn't support feature importance analysis")
        return
    
    # Find most positive and negative words
    # Positive coefficients = words that indicate positive sentiment
    # Negative coefficients = words that indicate negative sentiment
    
    # Get indices of most important features
    n_features = 20
    top_positive_idx = coef.argsort()[-n_features:][::-1]
    top_negative_idx = coef.argsort()[:n_features]
    
    # Get the actual words and their scores
    top_positive_words = [(feature_names[i], coef[i]) for i in top_positive_idx]
    top_negative_words = [(feature_names[i], coef[i]) for i in top_negative_idx]
    
    print(f"📈 Top {n_features} words that indicate POSITIVE sentiment:")
    print("-" * 50)
    for i, (word, score) in enumerate(top_positive_words, 1):
        print(f"{i:2d}. {word:<15} (importance: {score:.4f})")
    
    print(f"\n📉 Top {n_features} words that indicate NEGATIVE sentiment:")
    print("-" * 50)
    for i, (word, score) in enumerate(top_negative_words, 1):
        print(f"{i:2d}. {word:<15} (importance: {score:.4f})")
    
    return top_positive_words, top_negative_words

def create_word_importance_chart(positive_words, negative_words):
    """Create a visualization of important words"""
    
    print("\n📊 Creating word importance visualization...")
    
    # Prepare data for plotting
    pos_words = [word for word, score in positive_words[:10]]
    pos_scores = [score for word, score in positive_words[:10]]
    neg_words = [word for word, score in negative_words[:10]]
    neg_scores = [abs(score) for word, score in negative_words[:10]]  # Make positive for plotting
    
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Positive words chart
    bars1 = ax1.barh(range(len(pos_words)), pos_scores, color='lightgreen')
    ax1.set_yticks(range(len(pos_words)))
    ax1.set_yticklabels(pos_words)
    ax1.set_xlabel('Importance Score')
    ax1.set_title('Top 10 Words for POSITIVE Sentiment', fontsize=14, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(pos_scores):
        ax1.text(v + 0.001, i, f'{v:.3f}', va='center', fontsize=10)
    
    # Negative words chart
    bars2 = ax2.barh(range(len(neg_words)), neg_scores, color='lightcoral')
    ax2.set_yticks(range(len(neg_words)))
    ax2.set_yticklabels(neg_words)
    ax2.set_xlabel('Importance Score (Absolute Value)')
    ax2.set_title('Top 10 Words for NEGATIVE Sentiment', fontsize=14, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(neg_scores):
        ax2.text(v + 0.001, i, f'{v:.3f}', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('word_importance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Chart saved as 'word_importance.png'")

def test_model_performance(model, vectorizer, df):
    """Test the model performance on sample data"""
    
    print("\n🎯 Testing Model Performance")
    print("=" * 32)
    
    # Take a sample of data for testing
    sample_size = min(1000, len(df))
    sample_df = df.sample(n=sample_size, random_state=42)
    
    # Clean and vectorize the sample
    def clean_text(text):
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = text.lower()
        text = ' '.join(text.split())
        return text
    
    sample_df['cleaned_review'] = sample_df['review'].apply(clean_text)
    X_sample = vectorizer.transform(sample_df['cleaned_review'])
    y_true = sample_df['sentiment'].map({'positive': 1, 'negative': 0})
    
    # Make predictions
    y_pred = model.predict(X_sample)
    accuracy = (y_pred == y_true).mean()
    
    print(f"📊 Sample accuracy: {accuracy*100:.1f}%")
    
    # Show confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted Negative', 'Predicted Positive'],
                yticklabels=['Actually Negative', 'Actually Positive'])
    plt.title('Confusion Matrix - How Well the AI Performs', fontsize=14, fontweight='bold')
    plt.ylabel('True Sentiment')
    plt.xlabel('Predicted Sentiment')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Confusion matrix saved as 'confusion_matrix.png'")
    
    # Explain the results
    true_pos = cm[1, 1]  # Correctly predicted positive
    true_neg = cm[0, 0]  # Correctly predicted negative
    false_pos = cm[0, 1]  # Incorrectly predicted positive
    false_neg = cm[1, 0]  # Incorrectly predicted negative
    
    print(f"\n📈 Detailed Results:")
    print(f"   ✅ Correctly identified positive reviews: {true_pos}")
    print(f"   ✅ Correctly identified negative reviews: {true_neg}")
    print(f"   ❌ Mistakenly called negative reviews positive: {false_pos}")
    print(f"   ❌ Mistakenly called positive reviews negative: {false_neg}")

def main():
    """Main analysis function"""
    print("🔬 AI Model Analyzer")
    print("=" * 25)
    
    # Load model and data
    model, vectorizer, df = load_model_and_data()
    if model is None:
        return
    
    # Analyze important words
    positive_words, negative_words = analyze_important_words(model, vectorizer)
    
    if positive_words and negative_words:
        # Create visualization
        create_word_importance_chart(positive_words, negative_words)
        
        # Test performance
        test_model_performance(model, vectorizer, df)
    
    print("\n🎉 Analysis complete!")
    print("📁 Check the generated PNG files to see the visualizations!")

if __name__ == "__main__":
    main()