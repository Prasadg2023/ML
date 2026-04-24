import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

df = pd.read_csv("IMDB Dataset.csv")


# Label encoding
df["sentiment"] = df["sentiment"].map({"positive": 1, "negative": 0})

def clean_text(text):

    text = text.lower()

    text = re.sub(r'[^a-zA-Z]', ' ',text) 

    words = text.split()

    words = [w for w in words if w not in stop_words]

    return " ".join(words)

df["cleaned"] = df["review"].apply(clean_text)

# TFIDF
vectorizer = TfidfVectorizer(ngram_range = (1,2), max_features = 5000)
X = vectorizer.fit_transform(df["cleaned"])
y = df["sentiment"]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


model = LogisticRegression(max_iter = 1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Model evaluation Technique
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

cv_scores = cross_val_score(model, X, y, cv = 5)

st.title("🎬IMDB Movie Review Analysis")

st.subheader("✅ Model Performance")
st.write(f"Accuracy: {accuracy:.2f}")
st.write(f"Cross-Validation Scores: {cv_scores}")
st.write("Confusion Matrix")
st.write(cm)

st.divider()

review = st.text_area("Enter a movie review")
if st.button("Predict Sentiment"):
    if review.strip() == "":
        st.warning("Please enter a review to predict sentiment.")
    else:
        cleaned = clean_text(review)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        
        if prediction == 1:
            st.success("😊 Positive Review")
        else:
            st.error("💢 Negative Review")

       
