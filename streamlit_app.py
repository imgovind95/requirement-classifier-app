# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import re, random, os, warnings, shutil
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, LSTM, Dense

from transformers import pipeline

# ------------------------------------
# Streamlit UI
# ------------------------------------
st.title("Requirement Classification App ")

uploaded_file = st.file_uploader("Upload your dataset (CSV/TSV)", type=["csv", "tsv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file, sep="\t")
    except:
        df = pd.read_csv(uploaded_file, sep=None, engine="python")

    st.write("Preview of Data:", df.head())

    # Handle columns
    if len(df.columns) == 1:
        col0 = df.columns[0]
        splitted = df[col0].astype(str).str.split('\t', expand=True)
        if splitted.shape[1] >= 2:
            df = splitted.iloc[:, :3].copy()
            df.columns = ["RequirementText", "NFR", "niche"] if splitted.shape[1] >= 3 else ["RequirementText", "NFR"]
    if "niche" not in df.columns:
        df["niche"] = "unknown"

    # Text Cleaning
    def clean_text(text):
        text = str(text)
        text = re.sub(r"[^a-zA-Z\s]", " ", text).lower()
        return " ".join([w for w in text.split() if w not in set(["the","and","is","of"])])
    
    df["cleaned"] = df["RequirementText"].apply(clean_text)

    # Encode
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["NFR"].astype(str))
    X = df["cleaned"].values
    X_train_text, X_test_text, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model Selection
    model_choice = st.selectbox("Choose Model", ["Naive Bayes", "SVM", "Random Forest", "CNN", "LSTM", "Zero-Shot BART"])
    run_button = st.button("Run Model")

    if run_button:
        results = {}
        if model_choice in ["Naive Bayes", "SVM", "Random Forest"]:
            tfidf = TfidfVectorizer(max_features=5000)
            X_train_tfidf = tfidf.fit_transform(X_train_text)
            X_test_tfidf = tfidf.transform(X_test_text)

            if model_choice == "Naive Bayes":
                model = MultinomialNB()
            elif model_choice == "SVM":
                model = SVC(kernel="linear", random_state=42)
            else:
                model = RandomForestClassifier(n_estimators=100, random_state=42)

            model.fit(X_train_tfidf, y_train)
            preds = model.predict(X_test_tfidf)

        elif model_choice == "CNN":
            tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
            tokenizer.fit_on_texts(X_train_text)
            X_train_seq = tokenizer.texts_to_sequences(X_train_text)
            X_test_seq = tokenizer.texts_to_sequences(X_test_text)
            max_len = 50
            X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding="post", truncating="post")
            X_test_pad = pad_sequences(X_test_seq, maxlen=max_len, padding="post", truncating="post")

            vocab_size = min(5000, len(tokenizer.word_index) + 1)
            model = Sequential([
                Embedding(vocab_size, 100, input_length=max_len),
                Conv1D(128, 5, activation="relu"),
                GlobalMaxPooling1D(),
                Dense(64, activation="relu"),
                Dense(len(label_encoder.classes_), activation="softmax")
            ])
            model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
            model.fit(X_train_pad, y_train, epochs=3, batch_size=32, validation_split=0.1, verbose=0)
            preds = np.argmax(model.predict(X_test_pad), axis=1)

        elif model_choice == "LSTM":
            tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
            tokenizer.fit_on_texts(X_train_text)
            X_train_seq = tokenizer.texts_to_sequences(X_train_text)
            X_test_seq = tokenizer.texts_to_sequences(X_test_text)
            max_len = 50
            X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding="post", truncating="post")
            X_test_pad = pad_sequences(X_test_seq, maxlen=max_len, padding="post", truncating="post")

            vocab_size = min(5000, len(tokenizer.word_index) + 1)
            model = Sequential([
                Embedding(vocab_size, 100, input_length=max_len),
                LSTM(128, dropout=0.2),
                Dense(64, activation="relu"),
                Dense(len(label_encoder.classes_), activation="softmax")
            ])
            model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
            model.fit(X_train_pad, y_train, epochs=3, batch_size=32, validation_split=0.1, verbose=0)
            preds = np.argmax(model.predict(X_test_pad), axis=1)

        elif model_choice == "Zero-Shot BART":
            zsl = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
            preds = []
            for text in X_test_text:
                res = zsl(text, list(label_encoder.classes_), multi_label=False)
                preds.append(label_encoder.classes_.tolist().index(res["labels"][0]))
            preds = np.array(preds)

        acc = accuracy_score(y_test, preds)
        st.success(f" {model_choice} Accuracy: {acc:.2f}")
        st.text(classification_report(y_test, preds, target_names=label_encoder.classes_))

        # Save results
        results_df = pd.DataFrame({
            "RequirementText": X_test_text,
            "Actual": label_encoder.inverse_transform(y_test),
            "Predicted": label_encoder.inverse_transform(preds)
        })
        st.dataframe(results_df.head())
        st.download_button("Download Results", results_df.to_csv(index=False), "results.csv", "text/csv")
