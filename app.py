import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

st.set_page_config(page_title="Email Spam Classifier", page_icon="✉️")
st.title("Email Spam Classifier")
st.write("Paste an email below and choose a machine learning model to classify it as Spam or Ham (legitimate).")

@st.cache_resource
def train_models():
    data = pd.read_csv("emails.csv")
    if not {"text", "spam"}.issubset(data.columns):
        raise ValueError("The dataset must contain 'text' and 'spam' columns.")
    data = data.dropna(subset=["text", "spam"])
    X = data["text"].astype(str)
    y = data["spam"].astype(int)

    vectorizer = TfidfVectorizer(stop_words="english")
    X_tfidf = vectorizer.fit_transform(X)

    nb_model = MultinomialNB()
    svm_model = LinearSVC()
    nb_model.fit(X_tfidf, y)
    svm_model.fit(X_tfidf, y)
    return vectorizer, nb_model, svm_model

try:
    vectorizer, nb_model, svm_model = train_models()
except Exception as exc:
    st.error(f"Could not load the dataset or train the models: {exc}")
    st.stop()

model_choice = st.selectbox("Choose classifier", ["Naive Bayes", "Support Vector Machine (SVM)"])
email_text = st.text_area("Email text", height=180, placeholder="Paste the email message here...")

if st.button("Classify Email", type="primary"):
    if not email_text.strip():
        st.warning("Please paste an email before clicking Classify Email.")
    else:
        features = vectorizer.transform([email_text])
        model = nb_model if model_choice == "Naive Bayes" else svm_model
        prediction = int(model.predict(features)[0])
        if prediction == 1:
            st.error("Prediction: SPAM")
        else:
            st.success("Prediction: HAM (LEGITIMATE EMAIL)")
        st.caption(f"Model used: {model_choice}")

st.divider()
st.caption("Academic demonstration system. Predictions are model-based and may not always be correct.")
