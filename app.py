import streamlit as st
import plotly.graph_objects as go
import requests
import time

# ---------------- Gauge Chart ----------------
def create_confidence_gauge(confidence, sentiment):
    color = "#28a745" if sentiment == "Positive" else "#dc3545"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence,
        title={'text': "Confidence Score"},
        gauge={'axis': {'range': [None, 100]},
               'bar': {'color': color}}
    ))
    fig.update_layout(height=250)
    return fig

def predict_sentiment_via_api(review):
    url = "http://127.0.0.1:5000/predict"
    try:
        response = requests.post(url, json={"review": review})
        data = response.json()
        return data.get("sentiment"), data.get("confidence")
    except:
        st.error("❌ Could not reach backend API!")
        return None, None

# ---------------- UI Pages ----------------
def main():
    st.title("🎬 Movie Review AI Analyzer")

    page = st.sidebar.radio("Navigation", ["🏠 Home", "🧪 Test AI", "ℹ️ About"])

    if page == "🏠 Home":
        st.write("🚀 Welcome! Use sidebar to test AI.")
    elif page == "🧪 Test AI":
        show_test_page()
    elif page == "ℹ️ About":
        st.write("ℹ️ Sentiment Analysis on IMDB Reviews. Backend API serves predictions.")

def show_test_page():
    st.subheader("🧪 Test Your Movie Review AI")
    user_review = st.text_area("Enter a movie review:", height=150)
    if st.button("🔍 Analyze"):
        if user_review.strip():
            with st.spinner("Analyzing..."):
                time.sleep(1)
                sentiment, confidence = predict_sentiment_via_api(user_review)
                if sentiment:
                    st.write(f"**Prediction:** {sentiment} ({confidence:.1f}%)")
                    fig = create_confidence_gauge(confidence, sentiment)
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("⚠️ Review cannot be empty.")

if __name__ == "__main__":
    main()