import re

import nltk
import pandas as pd
import streamlit as st
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split


nltk.download("stopwords")
stop_words = set(stopwords.words("english"))
DATA_PATH = "IMDB Dataset.csv"


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z]", " ", text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["sentiment"] = df["sentiment"].map({"positive": 1, "negative": 0})
    df["cleaned"] = df["review"].apply(clean_text)
    return df


@st.cache_resource
def train_model():
    df = load_data()

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
    X = vectorizer.fit_transform(df["cleaned"])
    y = df["sentiment"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)
    cv_scores = cross_val_score(model, X, y, cv=5)

    return {
        "model": model,
        "vectorizer": vectorizer,
        "accuracy": accuracy,
        "confusion": confusion,
        "cv_scores": cv_scores,
    }


st.set_page_config(page_title="IMDB Sentiment Analyzer", page_icon="🎬", layout="wide")

assets = train_model()
model = assets["model"]
vectorizer = assets["vectorizer"]
accuracy = assets["accuracy"]
confusion = assets["confusion"]
cv_scores = assets["cv_scores"]

st.title("🎬 IMDB Sentiment Analyzer")
st.caption("Type a movie review and get an instant sentiment prediction.")

col1, col2, col3 = st.columns(3)
col1.metric("Accuracy", f"{accuracy:.2%}")
col2.metric("Average CV Score", f"{cv_scores.mean():.2%}")
col3.metric("TF-IDF Features", "5000")

with st.expander("Model Performance"):
    st.write("Cross-validation scores:", [round(score, 4) for score in cv_scores])
    st.write("Confusion matrix:")
    st.dataframe(
        pd.DataFrame(
            confusion,
            index=["Actual Negative", "Actual Positive"],
            columns=["Predicted Negative", "Predicted Positive"],
        ),
        use_container_width=True,
    )

st.divider()

sample_reviews = {
    "Positive sample": "This movie was brilliant, emotional, and beautifully acted from start to finish.",
    "Negative sample": "The plot was boring, the acting felt weak, and the whole film was a waste of time.",
    "Mixed sample": "The story started well, but the second half became slow and predictable.",
}

selected_sample = st.selectbox(
    "Try a sample review or write your own",
    options=["Custom review"] + list(sample_reviews.keys()),
)

default_review = (
    ""
    if selected_sample == "Custom review"
    else sample_reviews[selected_sample]
)

review = st.text_area("Enter a movie review", value=default_review, height=180)

if st.button("Predict Sentiment", type="primary"):
    if not review.strip():
        st.warning("Please enter a review before predicting.")
    else:
        cleaned_review = clean_text(review)
        vectorized_review = vectorizer.transform([cleaned_review])
        prediction = model.predict(vectorized_review)[0]
        probabilities = model.predict_proba(vectorized_review)[0]

        sentiment = "Positive" if prediction == 1 else "Negative"
        confidence = probabilities[prediction]

        result_col, detail_col = st.columns([1.1, 1])

        with result_col:
            if prediction == 1:
                st.success(f"Predicted Sentiment: {sentiment}")
            else:
                st.error(f"Predicted Sentiment: {sentiment}")
            st.progress(int(confidence * 100))
            st.write(f"Confidence: {confidence:.2%}")

        with detail_col:
            st.write("Prediction breakdown")
            st.write(f"Positive probability: {probabilities[1]:.2%}")
            st.write(f"Negative probability: {probabilities[0]:.2%}")

        with st.expander("Cleaned text used by the model"):
            st.write(cleaned_review)
