# # streamlit_app.py
# import streamlit as st
# import pandas as pd
# import numpy as np
# import re, random, os, warnings
# import matplotlib.pyplot as plt
# import seaborn as sns

# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.preprocessing import LabelEncoder
# from sklearn.metrics import classification_report, accuracy_score
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.svm import SVC
# from sklearn.ensemble import RandomForestClassifier

# import tensorflow as tf
# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, LSTM, Dense

# # Lazy import transformers pipeline to avoid ImportError
# def get_pipeline():
#     from transformers import pipeline
#     return pipeline

# # ----------------------------
# # Streamlit UI
# # ----------------------------
# st.title("Requirement Classification App")

# uploaded_file = st.file_uploader("Upload your dataset (CSV/TSV)", type=["csv", "tsv"])

# if uploaded_file is not None:
#     try:
#         df = pd.read_csv(uploaded_file, sep="\t")
#     except:
#         df = pd.read_csv(uploaded_file, sep=None, engine="python")

#     st.write("Preview of Data:", df.head())

#     # Handle columns
#     if len(df.columns) == 1:
#         col0 = df.columns[0]
#         splitted = df[col0].astype(str).str.split('\t', expand=True)
#         if splitted.shape[1] >= 2:
#             df = splitted.iloc[:, :3].copy()
#             df.columns = ["RequirementText", "NFR", "niche"] if splitted.shape[1] >= 3 else ["RequirementText", "NFR"]
#     if "niche" not in df.columns:
#         df["niche"] = "unknown"

#     # Text Cleaning
#     def clean_text(text):
#         text = str(text)
#         text = re.sub(r"[^a-zA-Z\s]", " ", text).lower()
#         return " ".join([w for w in text.split() if w not in set(["the","and","is","of"])])
    
#     df["cleaned"] = df["RequirementText"].apply(clean_text)

#     # Encode labels
#     label_encoder = LabelEncoder()
#     y = label_encoder.fit_transform(df["NFR"].astype(str))
#     X = df["cleaned"].values
#     X_train_text, X_test_text, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     # Model Selection
#     model_choice = st.selectbox("Choose Model", ["Naive Bayes", "SVM", "Random Forest", "CNN", "LSTM", "Zero-Shot BART"])
#     run_button = st.button("Run Model")

#     if run_button:
#         preds = None

#         # Traditional ML models
#         if model_choice in ["Naive Bayes", "SVM", "Random Forest"]:
#             tfidf = TfidfVectorizer(max_features=5000)
#             X_train_tfidf = tfidf.fit_transform(X_train_text)
#             X_test_tfidf = tfidf.transform(X_test_text)

#             if model_choice == "Naive Bayes":
#                 model = MultinomialNB()
#             elif model_choice == "SVM":
#                 model = SVC(kernel="linear", random_state=42)
#             else:
#                 model = RandomForestClassifier(n_estimators=100, random_state=42)

#             model.fit(X_train_tfidf, y_train)
#             preds = model.predict(X_test_tfidf)

#         # CNN
#         elif model_choice == "CNN":
#             tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
#             tokenizer.fit_on_texts(X_train_text)
#             X_train_seq = tokenizer.texts_to_sequences(X_train_text)
#             X_test_seq = tokenizer.texts_to_sequences(X_test_text)
#             max_len = 50
#             X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding="post", truncating="post")
#             X_test_pad = pad_sequences(X_test_seq, maxlen=max_len, padding="post", truncating="post")

#             vocab_size = min(5000, len(tokenizer.word_index) + 1)
#             model = Sequential([
#                 Embedding(vocab_size, 100, input_length=max_len),
#                 Conv1D(128, 5, activation="relu"),
#                 GlobalMaxPooling1D(),
#                 Dense(64, activation="relu"),
#                 Dense(len(label_encoder.classes_), activation="softmax")
#             ])
#             model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
#             model.fit(X_train_pad, y_train, epochs=3, batch_size=32, validation_split=0.1, verbose=0)
#             preds = np.argmax(model.predict(X_test_pad), axis=1)

#         # LSTM
#         elif model_choice == "LSTM":
#             tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
#             tokenizer.fit_on_texts(X_train_text)
#             X_train_seq = tokenizer.texts_to_sequences(X_train_text)
#             X_test_seq = tokenizer.texts_to_sequences(X_test_text)
#             max_len = 50
#             X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding="post", truncating="post")
#             X_test_pad = pad_sequences(X_test_seq, maxlen=max_len, padding="post", truncating="post")

#             vocab_size = min(5000, len(tokenizer.word_index) + 1)
#             model = Sequential([
#                 Embedding(vocab_size, 100, input_length=max_len),
#                 LSTM(128, dropout=0.2),
#                 Dense(64, activation="relu"),
#                 Dense(len(label_encoder.classes_), activation="softmax")
#             ])
#             model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
#             model.fit(X_train_pad, y_train, epochs=3, batch_size=32, validation_split=0.1, verbose=0)
#             preds = np.argmax(model.predict(X_test_pad), axis=1)

#         # Zero-Shot BART
#         elif model_choice == "Zero-Shot BART":
#     try:
#         from transformers import pipeline
#         zsl = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
#         preds_list = []
#         for text in X_test_text:
#             res = zsl(text, list(label_encoder.classes_), multi_label=False)
#             preds_list.append(label_encoder.classes_.tolist().index(res["labels"][0]))
#         preds = np.array(preds_list)
#     except Exception as e:
#         st.error("Zero-Shot BART failed. Make sure transformers & torch are installed.")
#         st.stop()

#         # If predictions exist, show accuracy and results
#         if preds is not None:
#             acc = accuracy_score(y_test, preds)
#             st.success(f"{model_choice} Accuracy: {acc:.2f}")

#             st.text(classification_report(
#                 y_test,
#                 preds,
#                 labels=np.arange(len(label_encoder.classes_)),
#                 target_names=label_encoder.classes_
#             ))

#             # Full results dataframe
#             results_df = pd.DataFrame({
#                 "RequirementText": X_test_text,
#                 "Actual": label_encoder.inverse_transform(y_test),
#                 "Predicted": label_encoder.inverse_transform(preds)
#             })

#             st.dataframe(results_df)  # show full results

#             # Download button for full results
#             st.download_button(
#                 "Download Full Results",
#                 results_df.to_csv(index=False),
#                 "results.csv",
#                 "text/csv"
#             )
# streamlit_app.py
# import streamlit as st
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.svm import SVC
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, classification_report
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Embedding, LSTM, Conv1D, GlobalMaxPooling1D, Flatten
# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from transformers import pipeline

# # ------------------ Streamlit UI ------------------
# st.title("Requirement Classification App")
# st.write("Upload your dataset (CSV/TSV)")

# uploaded_file = st.file_uploader("Drag and drop file here", type=["csv","tsv"])
# if uploaded_file:
#     df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_csv(uploaded_file, sep="\t")
#     st.write("Preview of Data:")
#     st.dataframe(df.head())

#     text_column = st.selectbox("Select text column", df.columns)
#     label_column = st.selectbox("Select label column", df.columns)

#     X = df[text_column].astype(str).values
#     y = df[label_column].astype(str).values

#     # Label encoding
#     label_encoder = LabelEncoder()
#     y_encoded = label_encoder.fit_transform(y)

#     # Split data
#     X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

#     # ------------------ Load Transformers Models ------------------
#     @st.cache_resource
#     def load_transformers():
#         bert_pipeline = pipeline("text-classification", model="bert-base-uncased")
#         roberta_pipeline = pipeline("text-classification", model="roberta-base")
#         bart_pipeline = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
#         return bert_pipeline, roberta_pipeline, bart_pipeline

#     bert_pipeline, roberta_pipeline, bart_pipeline = load_transformers()

#     # ------------------ Model Selection ------------------
#     model_choice = st.selectbox("Choose Model", [
#         "NaiveBayes", "SVM", "RandomForest", "CNN", "LSTM", "BERT", "RoBERTa", "Zero-Shot BART"
#     ])

#     preds = []

#     if st.button("Classify"):
#         if model_choice in ["NaiveBayes", "SVM", "RandomForest"]:
#             # Simple TF-IDF for sklearn models
#             from sklearn.feature_extraction.text import TfidfVectorizer
#             vectorizer = TfidfVectorizer(max_features=5000)
#             X_train_vec = vectorizer.fit_transform(X_train)
#             X_test_vec = vectorizer.transform(X_test)

#             if model_choice == "NaiveBayes":
#                 model = MultinomialNB()
#             elif model_choice == "SVM":
#                 model = SVC()
#             else:
#                 model = RandomForestClassifier()

#             model.fit(X_train_vec, y_train)
#             preds = model.predict(X_test_vec)

#         elif model_choice in ["CNN", "LSTM"]:
#             tokenizer = Tokenizer(num_words=5000)
#             tokenizer.fit_on_texts(X_train)
#             X_train_seq = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=100)
#             X_test_seq = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=100)

#             vocab_size = 5000
#             if model_choice == "CNN":
#                 model = Sequential([
#                     Embedding(vocab_size, 128, input_length=100),
#                     Conv1D(64, 5, activation='relu'),
#                     GlobalMaxPooling1D(),
#                     Dense(64, activation='relu'),
#                     Dense(len(np.unique(y_train)), activation='softmax')
#                 ])
#             else:  # LSTM
#                 model = Sequential([
#                     Embedding(vocab_size, 128, input_length=100),
#                     LSTM(64),
#                     Dense(64, activation='relu'),
#                     Dense(len(np.unique(y_train)), activation='softmax')
#                 ])

#             model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#             model.fit(X_train_seq, y_train, epochs=3, batch_size=32, verbose=0)
#             preds = np.argmax(model.predict(X_test_seq), axis=1)

#         elif model_choice in ["BERT", "RoBERTa"]:
#             pipeline_model = bert_pipeline if model_choice == "BERT" else roberta_pipeline
#             preds = []
#             for text in X_test:
#                 res = pipeline_model(text)[0]
#                 # Find nearest label index
#                 label = res['label'].replace("LABEL_", "")
#                 if label.isdigit():
#                     preds.append(int(label))
#                 else:
#                     # fallback: map string label to encoded
#                     preds.append(label_encoder.transform([res['label']])[0])
#             preds = np.array(preds)

#         elif model_choice == "Zero-Shot BART":
#             candidate_labels = label_encoder.classes_.tolist()
#             preds = []
#             for text in X_test:
#                 res = bart_pipeline(text, candidate_labels)
#                 label = res['labels'][0]
#                 preds.append(label_encoder.transform([label])[0])
#             preds = np.array(preds)

#         # ------------------ Accuracy & Report ------------------
#         acc = accuracy_score(y_test, preds)
#         st.success(f"{model_choice} Accuracy: {acc:.2f}")

#         try:
#             st.text(classification_report(y_test, preds, target_names=label_encoder.classes_))
#         except:
#             st.warning("Classification report could not display (maybe mismatched labels)")

#         # ------------------ Save & Display Results ------------------
#         results_df = pd.DataFrame({
#             "Text": X_test,
#             "Predicted": label_encoder.inverse_transform(preds)
#         })
#         st.dataframe(results_df)
#         st.download_button("Download Results CSV", results_df.to_csv(index=False), "results.csv")
# import streamlit as st
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.svm import SVC
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, classification_report
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, LSTM, Dense
# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences

# # -------------------------------------------------------
# # Page Config
# # -------------------------------------------------------
# st.set_page_config(page_title="Requirement Classification App", layout="wide")
# st.title("📊 Requirement Classification App")

# # -------------------------------------------------------
# # File Upload
# # -------------------------------------------------------
# uploaded_file = st.file_uploader("Upload your dataset (CSV/TSV)", type=["csv", "tsv"])
# if uploaded_file:
#     sep = "," if uploaded_file.name.endswith(".csv") else "\t"
#     df = pd.read_csv(uploaded_file, sep=sep)
#     st.write("### Preview of Data:")
#     st.dataframe(df.head())

#     if "text" not in df.columns or "label" not in df.columns:
#         st.error("Dataset must contain 'text' and 'label' columns.")
#     else:
#         X = df["text"].astype(str)
#         y = df["label"].astype(str)

#         # Label encode
#         label_encoder = LabelEncoder()
#         y_encoded = label_encoder.fit_transform(y)

#         # Split data
#         X_train, X_test, y_train, y_test = train_test_split(
#             X, y_encoded, test_size=0.2, random_state=42
#         )

#         # -------------------------------------------------------
#         # Model Choice
#         # -------------------------------------------------------
#         model_choice = st.selectbox(
#             "Choose Model",
#             [
#                 "Naive Bayes",
#                 "SVM",
#                 "Random Forest",
#                 "CNN",
#                 "LSTM",
#                 "BERT",
#                 "RoBERTa",
#                 "Zero-Shot BART",
#             ],
#         )

#         results_df = None

#         # -------------------------------------------------------
#         # Traditional ML Models
#         # -------------------------------------------------------
#         if model_choice == "Naive Bayes":
#             model = MultinomialNB()
#             model.fit(X_train.str.len().values.reshape(-1, 1), y_train)
#             preds = model.predict(X_test.str.len().values.reshape(-1, 1))

#         elif model_choice == "SVM":
#             model = SVC()
#             model.fit(X_train.str.len().values.reshape(-1, 1), y_train)
#             preds = model.predict(X_test.str.len().values.reshape(-1, 1))

#         elif model_choice == "Random Forest":
#             model = RandomForestClassifier()
#             model.fit(X_train.str.len().values.reshape(-1, 1), y_train)
#             preds = model.predict(X_test.str.len().values.reshape(-1, 1))

#         # -------------------------------------------------------
#         # Deep Learning (CNN / LSTM)
#         # -------------------------------------------------------
#         elif model_choice in ["CNN", "LSTM"]:
#             max_words = 5000
#             max_len = 100
#             tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
#             tokenizer.fit_on_texts(X_train)

#             X_train_seq = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=max_len)
#             X_test_seq = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=max_len)

#             model = Sequential()
#             model.add(Embedding(max_words, 128, input_length=max_len))

#             if model_choice == "CNN":
#                 model.add(Conv1D(64, 5, activation="relu"))
#                 model.add(GlobalMaxPooling1D())
#             else:  # LSTM
#                 model.add(LSTM(64))

#             model.add(Dense(len(label_encoder.classes_), activation="softmax"))
#             model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

#             model.fit(X_train_seq, y_train, epochs=2, batch_size=32, validation_split=0.1, verbose=0)
#             preds = np.argmax(model.predict(X_test_seq), axis=1)

#         # -------------------------------------------------------
#         # HuggingFace Models (Lazy Load)
#         # -------------------------------------------------------
#         else:
#             from transformers import pipeline

#             if model_choice == "BERT":
#                 nlp = pipeline("text-classification", model="bert-base-uncased")
#                 preds = [label_encoder.transform([nlp(t)[0]["label"]])[0] if nlp(t)[0]["label"] in label_encoder.classes_ else 0 for t in X_test]

#             elif model_choice == "RoBERTa":
#                 nlp = pipeline("text-classification", model="roberta-base")
#                 preds = [label_encoder.transform([nlp(t)[0]["label"]])[0] if nlp(t)[0]["label"] in label_encoder.classes_ else 0 for t in X_test]

#             elif model_choice == "Zero-Shot BART":
#                 nlp = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
#                 preds = []
#                 for t in X_test:
#                     result = nlp(t, candidate_labels=list(label_encoder.classes_))
#                     label = result["labels"][0]
#                     preds.append(label_encoder.transform([label])[0])

#         # -------------------------------------------------------
#         # Results
#         # -------------------------------------------------------
#         acc = accuracy_score(y_test, preds)
#         st.success(f"{model_choice} Accuracy: {acc:.2f}")

#         report = classification_report(
#             y_test,
#             preds,
#             labels=np.unique(y_test),
#             target_names=label_encoder.inverse_transform(np.unique(y_test)),
#         )
#         st.text(report)

#         results_df = pd.DataFrame({
#             "Text": X_test,
#             "Actual": label_encoder.inverse_transform(y_test),
#             "Predicted": label_encoder.inverse_transform(preds),
#         })

#         st.write("### Results Preview")
#         st.dataframe(results_df)

#         # Full download
#         st.download_button(
#             "Download Results",
#             results_df.to_csv(index=False).encode("utf-8"),
#             "results.csv",
#             "text/csv",
#         )
# import streamlit as st
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.svm import SVC
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, classification_report
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Embedding, GlobalMaxPooling1D, Conv1D, LSTM
# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# import torch
# from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# st.title("Requirement Classification App")

# uploaded_file = st.file_uploader("Upload your dataset (CSV/TSV)", type=["csv", "tsv"])

# if uploaded_file is not None:
#     # Load dataset
#     try:
#         if uploaded_file.name.endswith(".csv"):
#             df = pd.read_csv(uploaded_file)
#         else:
#             df = pd.read_csv(uploaded_file, sep="\t")
#         st.success("Dataset loaded successfully!")
#     except Exception as e:
#         st.error(f"Error loading file: {e}")
#         st.stop()

#     st.dataframe(df.head())

#     # Select text and label columns
#     text_col = st.selectbox("Select text column", df.columns)
#     label_col = st.selectbox("Select label column", df.columns)

#     X = df[text_col].astype(str)
#     y = df[label_col]

#     # Encode labels
#     label_encoder = LabelEncoder()
#     y_encoded = label_encoder.fit_transform(y)

#     X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

#     model_choice = st.selectbox(
#         "Choose Model",
#         ["NaiveBayes", "SVM", "RandomForest", "CNN", "LSTM", "BERT", "RoBERTa", "Zero-Shot BART"]
#     )

#     results_df = pd.DataFrame()
#     preds = []

#     if model_choice in ["NaiveBayes", "SVM", "RandomForest"]:
#         vectorizer = TfidfVectorizer(max_features=5000)
#         X_train_tfidf = vectorizer.fit_transform(X_train)
#         X_test_tfidf = vectorizer.transform(X_test)

#         if model_choice == "NaiveBayes":
#             clf = MultinomialNB()
#         elif model_choice == "SVM":
#             clf = SVC()
#         else:
#             clf = RandomForestClassifier()

#         clf.fit(X_train_tfidf, y_train)
#         preds = clf.predict(X_test_tfidf)

#     elif model_choice in ["CNN", "LSTM"]:
#         tokenizer = Tokenizer(num_words=5000)
#         tokenizer.fit_on_texts(X_train)
#         X_train_seq = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=100)
#         X_test_seq = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=100)

#         vocab_size = 5000
#         embedding_dim = 50

#         model = Sequential()
#         model.add(Embedding(vocab_size, embedding_dim, input_length=100))
#         if model_choice == "CNN":
#             model.add(Conv1D(128, 5, activation="relu"))
#             model.add(GlobalMaxPooling1D())
#         else:
#             model.add(LSTM(128))
#         model.add(Dense(len(label_encoder.classes_), activation="softmax"))
#         model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
#         model.fit(X_train_seq, y_train, epochs=3, batch_size=32, verbose=0)
#         preds_probs = model.predict(X_test_seq)
#         preds = np.argmax(preds_probs, axis=1)

#     elif model_choice in ["BERT", "RoBERTa"]:
#         if model_choice == "BERT":
#             model_name = "bert-base-uncased"
#         else:
#             model_name = "roberta-base"

#         bert_tokenizer = AutoTokenizer.from_pretrained(model_name)
#         bert_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(label_encoder.classes_))
#         classifier = pipeline("text-classification", model=bert_model, tokenizer=bert_tokenizer)

#         preds = []
#         for text in X_test:
#             pred = classifier(text, truncation=True)
#             label = pred[0]["label"]
#             if label.startswith("LABEL_"):
#                 idx = int(label.split("_")[1])
#                 preds.append(idx)
#             else:
#                 # fallback in case
#                 preds.append(0)

#     elif model_choice == "Zero-Shot BART":
#         try:
#             zsl_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
#             candidate_labels = list(label_encoder.classes_)
#             preds = []
#             for text in X_test:
#                 pred = zsl_classifier(text, candidate_labels=candidate_labels)
#                 preds.append(label_encoder.transform([pred["labels"][0]])[0])
#         except Exception as e:
#             st.error(f"Zero-Shot BART error: {e}")
#             st.stop()

#     # Show accuracy & classification report safely
#     if len(preds) == len(y_test):
#         acc = accuracy_score(y_test, preds)
#         st.success(f"{model_choice} Accuracy: {acc:.2f}")
#         st.text(classification_report(y_test, preds, target_names=label_encoder.classes_))

#         results_df = pd.DataFrame({
#             "Text": X_test,
#             "Actual": label_encoder.inverse_transform(y_test),
#             "Predicted": label_encoder.inverse_transform(preds)
#         })

#         st.dataframe(results_df)
#         st.download_button("Download Results", results_df.to_csv(index=False), "results.csv")
#     else:
#         st.error("Prediction length does not match test data length.")
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.multiclass import unique_labels
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalMaxPooling1D, Conv1D, LSTM
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

st.title("Requirement Classification App")

uploaded_file = st.file_uploader("Upload your dataset (CSV/TSV)", type=["csv", "tsv"])

if uploaded_file is not None:
    # Load dataset
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file, sep="\t")
        st.success("Dataset loaded successfully!")
    except Exception as e:
        st.error(f"Error loading file: {e}")
        st.stop()

    st.dataframe(df.head())

    # Select text and label columns
    text_col = st.selectbox("Select text column", df.columns)
    label_col = st.selectbox("Select label column", df.columns)

    X = df[text_col].astype(str)
    y = df[label_col]

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    model_choice = st.selectbox(
        "Choose Model",
        ["NaiveBayes", "SVM", "RandomForest", "CNN", "LSTM", "BERT", "RoBERTa", "Zero-Shot BART"]
    )

    results_df = pd.DataFrame()
    preds = []

    if model_choice in ["NaiveBayes", "SVM", "RandomForest"]:
        vectorizer = TfidfVectorizer(max_features=5000)
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)

        if model_choice == "NaiveBayes":
            clf = MultinomialNB()
        elif model_choice == "SVM":
            clf = SVC()
        else:
            clf = RandomForestClassifier()

        clf.fit(X_train_tfidf, y_train)
        preds = clf.predict(X_test_tfidf)

    elif model_choice in ["CNN", "LSTM"]:
        tokenizer = Tokenizer(num_words=5000)
        tokenizer.fit_on_texts(X_train)
        X_train_seq = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=100)
        X_test_seq = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=100)

        vocab_size = 5000
        embedding_dim = 50

        model = Sequential()
        model.add(Embedding(vocab_size, embedding_dim, input_length=100))
        if model_choice == "CNN":
            model.add(Conv1D(128, 5, activation="relu"))
            model.add(GlobalMaxPooling1D())
        else:
            model.add(LSTM(128))
        model.add(Dense(len(label_encoder.classes_), activation="softmax"))
        model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        model.fit(X_train_seq, y_train, epochs=3, batch_size=32, verbose=0)
        preds_probs = model.predict(X_test_seq)
        preds = np.argmax(preds_probs, axis=1)

    elif model_choice in ["BERT", "RoBERTa"]:
        if model_choice == "BERT":
            model_name = "bert-base-uncased"
        else:
            model_name = "roberta-base"

        bert_tokenizer = AutoTokenizer.from_pretrained(model_name)
        bert_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(label_encoder.classes_))
        classifier = pipeline("text-classification", model=bert_model, tokenizer=bert_tokenizer)

        preds = []
        for text in X_test:
            pred = classifier(text, truncation=True)
            label = pred[0]["label"]
            if label.startswith("LABEL_"):
                idx = int(label.split("_")[1])
                preds.append(idx)
            else:
                preds.append(0)

    elif model_choice == "Zero-Shot BART":
        try:
            zsl_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
            candidate_labels = list(label_encoder.classes_)
            preds = []
            for text in X_test:
                pred = zsl_classifier(text, candidate_labels=candidate_labels)
                preds.append(label_encoder.transform([pred["labels"][0]])[0])
        except Exception as e:
            st.error(f"Zero-Shot BART error: {e}")
            st.stop()

    # ✅ Safe evaluation (Fix applied here)
    if len(preds) == len(y_test):
        acc = accuracy_score(y_test, preds)
        st.success(f"{model_choice} Accuracy: {acc:.2f}")

        labels_used = unique_labels(y_test, preds)
        st.text(classification_report(
            y_test,
            preds,
            labels=labels_used,
            target_names=label_encoder.classes_[labels_used]
        ))

        results_df = pd.DataFrame({
            "Text": X_test,
            "Actual": label_encoder.inverse_transform(y_test),
            "Predicted": label_encoder.inverse_transform(preds)
        })

        st.dataframe(results_df)
        st.download_button("Download Results", results_df.to_csv(index=False), "results.csv")
    else:
        st.error("Prediction length does not match test data length.")
