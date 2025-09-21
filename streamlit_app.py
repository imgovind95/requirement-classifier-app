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

# # ✅ CHANGE 1: Gemini SDK import & API setup
# import google.generativeai as genai
# from io import StringIO

# # ✅ CHANGE 2: Fix API key here (user ko key dalne ki need nahi)
# GEMINI_API_KEY = "AIzaSyByKSIsncltKAAhIg1NH7pPSCfGRqctxWc"   # <- yahan apna Gemini API key paste karo
# genai.configure(api_key="AIzaSyByKSIsncltKAAhIg1NH7pPSCfGRqctxWc")

# # ----------------------------
# # Streamlit UI
# # ----------------------------
# st.title("Requirement Classification App with Gemini Dataset Builder")

# # ✅ CHANGE 3: Add prompt input box
# prompt = st.text_area("Write your prompt for dataset (CSV format):", 
#                       "Generate 20 requirements in CSV format with columns: RequirementText,NFR")

# gen_button = st.button("Generate Dataset")

# df = None

# # ✅ CHANGE 4: If user clicks button, Gemini se dataset lao
# if gen_button and prompt:
#     try:
#         model = genai.GenerativeModel("gemini-1.5-flash")
#         response = model.generate_content(prompt)
#         raw_text = response.text

#         # Parse CSV output from Gemini
#         df = pd.read_csv(StringIO(raw_text))
#         st.success("✅ Dataset generated successfully from Gemini!")
#         st.dataframe(df.head())

#         # Download button
#         st.download_button(
#             "Download Generated Dataset",
#             df.to_csv(index=False),
#             "generated_dataset.csv",
#             "text/csv"
#         )

#     except Exception as e:
#         st.error(f"Error while parsing Gemini response: {e}")

# # ✅ CHANGE 5: Continue with classification if dataset available
# if df is not None:
#     if len(df.columns) == 1:
#         col0 = df.columns[0]
#         splitted = df[col0].astype(str).str.split('\t', expand=True)
#         if splitted.shape[1] >= 2:
#             df = splitted.iloc[:, :3].copy()
#             df.columns = ["RequirementText", "NFR", "niche"] if splitted.shape[1] >= 3 else ["RequirementText", "NFR"]
#     if "niche" not in df.columns:
#         df["niche"] = "unknown"

#     def clean_text(text):
#         text = str(text)
#         text = re.sub(r"[^a-zA-Z\s]", " ", text).lower()
#         return " ".join([w for w in text.split() if w not in set(["the","and","is","of"])])
    
#     df["cleaned"] = df["RequirementText"].apply(clean_text)

#     label_encoder = LabelEncoder()
#     y = label_encoder.fit_transform(df["NFR"].astype(str))
#     X = df["cleaned"].values
#     X_train_text, X_test_text, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     model_choice = st.selectbox("Choose Model", ["Naive Bayes", "SVM", "Random Forest", "CNN", "LSTM"])
#     run_button = st.button("Run Model")

#     if run_button:
#         preds = None
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

#         if preds is not None:
#             acc = accuracy_score(y_test, preds)
#             st.success(f"{model_choice} Accuracy: {acc:.2f}")

#             st.text(classification_report(
#                 y_test,
#                 preds,
#                 labels=np.arange(len(label_encoder.classes_)),
#                 target_names=label_encoder.classes_
#             ))

#             results_df = pd.DataFrame({
#                 "RequirementText": X_test_text,
#                 "Actual": label_encoder.inverse_transform(y_test),
#                 "Predicted": label_encoder.inverse_transform(preds)
#             })

#             st.dataframe(results_df)

#             st.download_button(
#                 "Download Full Results",
#                 results_df.to_csv(index=False),
#                 "results.csv",
#                 "text/csv"
#             )

# import streamlit as st
# import pandas as pd
# import numpy as np
# import re
# import google.generativeai as genai
# from io import StringIO

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

# # ----------------------------
# # Gemini API setup using Secrets
# # ----------------------------
# genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# st.title("Requirement Classification App Dataset Builder")

# # ----------------------------
# # Helpers
# # ----------------------------
# def clean_text(text):
#     text = str(text).lower()
#     text = re.sub(r"[^a-z0-9\s]", " ", text)
#     return re.sub(r"\s+", " ", text).strip()

# def fix_columns(df):
#     rename_map = {}
#     if "RequirementText" not in df.columns:
#         for col in df.columns:
#             if "require" in col.lower():
#                 rename_map[col] = "RequirementText"
#     if "NFR" not in df.columns:
#         for col in df.columns:
#             if "nfr" in col.lower() or "label" in col.lower() or "type" in col.lower():
#                 rename_map[col] = "NFR"
#     df.rename(columns=rename_map, inplace=True)
#     return df

# # ----------------------------
# # Initialize session state
# # ----------------------------
# if "df" not in st.session_state:
#     st.session_state.df = None

# # ----------------------------
# # Step 0: Local file upload
# # ----------------------------
# uploaded_file = st.file_uploader(
#     "Upload your dataset (CSV/TSV) or leave empty to generate :",
#     type=["csv", "tsv"]
# )

# if uploaded_file is not None:
#     try:
#         df = pd.read_csv(uploaded_file)
#     except pd.errors.ParserError:
#         df = pd.read_csv(uploaded_file, sep="\t")
#     df = fix_columns(df)
#     if "RequirementText" not in df.columns or "NFR" not in df.columns:
#         st.error("Uploaded file missing 'RequirementText' or 'NFR'.")
#     else:
#         df["cleaned"] = df["RequirementText"].apply(clean_text)
#         st.session_state.df = df
#         st.success("Uploaded dataset loaded successfully")
#         st.dataframe(df.head())

# # ----------------------------
# # Step 1: Gemini dataset generation
# # ----------------------------
# if st.session_state.df is None:
#     num_rows = st.number_input(
#         "How many requirements to generate?", min_value=1, value=20, step=1
#     )

#     prompt = st.text_area(
#         "Write your prompt for dataset (CSV format):",
#         f"Generate {num_rows} software requirements in CSV format with EXACTLY 2 columns: RequirementText and NFR. "
#         "Each row must have exactly 2 fields. Do not use commas or quotes inside the RequirementText or NFR fields. "
#         "Separate columns using a comma. Each row must be on a new line. Output only CSV content, no extra explanation or text."
#     )

#     gen_button = st.button("Generate Dataset")

#     if gen_button and prompt:
#         try:
#             model = genai.GenerativeModel("gemini-1.5-flash")
#             response = model.generate_content(prompt)
#             raw_text = response.text

#             # Ensure only lines with exactly 2 fields are used
#             lines = raw_text.strip().split("\n")
#             csv_lines = [line for line in lines if len(line.split(",")) == 2]
#             csv_text = "\n".join(csv_lines)

#             df = pd.read_csv(StringIO(csv_text), names=["RequirementText", "NFR"])
#             df = fix_columns(df)

#             if "RequirementText" not in df.columns or "NFR" not in df.columns:
#                 st.error("Dataset missing 'RequirementText' or 'NFR'. Please refine your prompt.")
#             else:
#                 df["cleaned"] = df["RequirementText"].apply(clean_text)
#                 st.session_state.df = df
#                 st.success("Dataset generated successfully via Gemini API")
#                 st.dataframe(df.head())

#         except Exception as e:
#             st.error(f"Error while generating dataset: {e}")

# # ----------------------------
# # Step 2: Train Models
# # ----------------------------
# if st.session_state.df is not None:
#     df = st.session_state.df
#     label_encoder = LabelEncoder()
#     y = label_encoder.fit_transform(df["NFR"].astype(str))
#     X = df["cleaned"].values

#     X_train_text, X_test_text, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42
#     )

#     model_choice = st.selectbox("Choose Model", ["Naive Bayes", "SVM", "Random Forest", "CNN", "LSTM"])
#     run_button = st.button("Run Model")

#     if run_button:
#         preds = None

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

#         if preds is not None:
#             acc = accuracy_score(y_test, preds)
#             st.success(f"{model_choice} Accuracy: {acc:.2f}")

#             st.text(classification_report(
#                 y_test,
#                 preds,
#                 labels=np.arange(len(label_encoder.classes_)),
#                 target_names=label_encoder.classes_
#             ))

#             results_df = pd.DataFrame({
#                 "RequirementText": X_test_text,
#                 "Actual": label_encoder.inverse_transform(y_test),
#                 "Predicted": label_encoder.inverse_transform(preds)
#             })

#             st.dataframe(results_df)

#             st.download_button(
#                 "Download Full Results",
#                 results_df.to_csv(index=False),
#                 "results.csv",
#                 "text/csv"
#             )
# import streamlit as st
# import pandas as pd
# import numpy as np
# import re
# import google.generativeai as genai
# from io import StringIO

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

# # ----------------------------
# # Gemini API setup using Secrets
# # ----------------------------
# genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# st.title("Requirement Classification App Dataset Builder")

# # ----------------------------
# # Helpers
# # ----------------------------
# def clean_text(text):
#     text = str(text).lower()
#     text = re.sub(r"[^a-z0-9\s]", " ", text)
#     return re.sub(r"\s+", " ", text).strip()

# # ----------------------------
# # Initialize session state
# # ----------------------------
# if "df" not in st.session_state:
#     st.session_state.df = None

# # ----------------------------
# # Step 0: Upload dataset
# # ----------------------------
# uploaded_file = st.file_uploader(
#     "Upload your dataset (CSV/TSV) or leave empty to generate :",
#     type=["csv", "tsv"]
# )

# if uploaded_file is not None:
#     try:
#         df = pd.read_csv(uploaded_file)
#     except pd.errors.ParserError:
#         df = pd.read_csv(uploaded_file, sep="\t")

#     if "RequirementText" not in df.columns or "NFR" not in df.columns:
#         st.error("Uploaded file must contain 'RequirementText' and 'NFR' columns.")
#     else:
#         df["cleaned"] = df["RequirementText"].apply(clean_text)
#         st.session_state.df = df
#         st.success("Uploaded dataset loaded successfully")
#         st.dataframe(df.head())

# # ----------------------------
# # Step 1: Gemini dataset generation (if no upload)
# # ----------------------------
# if st.session_state.df is None:
#     user_prompt = st.text_area(
#         "Write your prompt for dataset (CSV format):",
#         value="""Generate 20 software requirements in CSV format with EXACTLY 2 columns: RequirementText and NFR.
# Each row must have exactly 2 fields. Do not use commas or quotes inside the RequirementText or NFR fields.
# Separate columns using a comma. Each row must be on a new line. Output only CSV content, no extra explanation or text."""
#     )

#     if st.button("Generate Dataset"):
#         try:
#             model = genai.GenerativeModel("gemini-1.5-flash")
#             response = model.generate_content(user_prompt)
#             raw_text = response.text.strip()

#             # Keep only valid rows with 2 columns
#             lines = raw_text.split("\n")
#             csv_lines = [line for line in lines if len(line.split(",")) == 2]
#             csv_text = "\n".join(csv_lines)

#             df = pd.read_csv(StringIO(csv_text), names=["RequirementText", "NFR"])
#             df["cleaned"] = df["RequirementText"].apply(clean_text)

#             st.session_state.df = df
#             st.success("Dataset generated successfully via Gemini API")
#             st.dataframe(df.head())

#         except Exception as e:
#             st.error(f"Error while generating dataset: {e}")

# # ----------------------------
# # Step 2: Train Models
# # ----------------------------
# if st.session_state.df is not None:
#     df = st.session_state.df
#     label_encoder = LabelEncoder()
#     y = label_encoder.fit_transform(df["NFR"].astype(str))
#     X = df["cleaned"].values

#     X_train_text, X_test_text, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42
#     )

#     model_choice = st.selectbox("Choose Model", ["Naive Bayes", "SVM", "Random Forest", "CNN", "LSTM"])
#     run_button = st.button("Run Model")

#     if run_button:
#         preds = None

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

#         if preds is not None:
#             acc = accuracy_score(y_test, preds)
#             st.success(f"{model_choice} Accuracy: {acc:.2f}")

#             st.text(classification_report(
#                 y_test,
#                 preds,
#                 labels=np.arange(len(label_encoder.classes_)),
#                 target_names=label_encoder.classes_
#             ))

#             results_df = pd.DataFrame({
#                 "RequirementText": X_test_text,
#                 "Actual": label_encoder.inverse_transform(y_test),
#                 "Predicted": label_encoder.inverse_transform(preds)
#             })

#             st.dataframe(results_df)

#             st.download_button(
#                 "Download Full Results",
#                 results_df.to_csv(index=False),
#                 "results.csv",
#                 "text/csv"
#             )

import streamlit as st
import pandas as pd
import numpy as np
import re
import google.generativeai as genai
from io import StringIO

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

# ----------------------------
# Gemini API setup using Secrets
# ----------------------------
# It's recommended to handle API key errors gracefully
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
except (KeyError, AttributeError):
    st.error("GEMINI_API_KEY not found in Streamlit secrets. Please add it.")
    st.stop()


st.title("Requirement Classification App Dataset Builder")

# ----------------------------
# Helpers
# ----------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

# ----------------------------
# Initialize session state
# ----------------------------
if "df" not in st.session_state:
    st.session_state.df = None

# ----------------------------
# Step 0: Upload dataset
# ----------------------------
uploaded_file = st.file_uploader(
    "Upload your dataset (CSV/TSV) or leave empty to generate :",
    type=["csv", "tsv"]
)

if uploaded_file is not None:
    try:
        # Attempt to read as CSV, if it fails, try TSV
        try:
            df = pd.read_csv(uploaded_file)
        except pd.errors.ParserError:
            uploaded_file.seek(0) # Reset file pointer
            df = pd.read_csv(uploaded_file, sep="\t")

        st.session_state.raw_df = df # Store the original dataframe
        st.success("Uploaded dataset loaded successfully")
        st.dataframe(df.head())

    except Exception as e:
        st.error(f"Error loading file: {e}")
        st.session_state.raw_df = None


# New section to allow column selection
if "raw_df" in st.session_state and st.session_state.raw_df is not None:
    df = st.session_state.raw_df
    columns = df.columns.tolist()

    st.subheader("Select Columns")
    # Let the user select the text and label columns
    text_column = st.selectbox("Select the column containing the requirement text:", columns)
    nfr_column = st.selectbox("Select the column containing the NFR label:", columns)

    if st.button("Confirm Columns and Process Data"):
        # Create a new dataframe with the standard column names
        processed_df = pd.DataFrame({
            'RequirementText': df[text_column],
            'NFR': df[nfr_column]
        })
        processed_df["cleaned"] = processed_df["RequirementText"].apply(clean_text)
        st.session_state.df = processed_df
        st.success("Columns mapped and data processed!")
        st.dataframe(st.session_state.df.head())


# ----------------------------
# Step 1: Gemini dataset generation (if no upload)
# ----------------------------
if uploaded_file is None:
    user_prompt = st.text_area(
        "Write your prompt for dataset (CSV format):",
        value="""Generate 20 software requirements in CSV format with EXACTLY 2 columns: RequirementText and NFR.
Each row must have exactly 2 fields. Do not use commas or quotes inside the RequirementText or NFR fields.
Separate columns using a comma. Each row must be on a new line. Output only CSV content, no extra explanation or text."""
    )

    if st.button("Generate Dataset"):
        try:
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(user_prompt)
            raw_text = response.text.strip()

            # Keep only valid rows with 2 columns
            lines = raw_text.split("\n")
            csv_lines = [line for line in lines if len(line.split(",")) == 2]
            csv_text = "\n".join(csv_lines)

            df = pd.read_csv(StringIO(csv_text), names=["RequirementText", "NFR"])
            df["cleaned"] = df["RequirementText"].apply(clean_text)

            st.session_state.df = df
            st.success("Dataset generated successfully via Gemini API")
            st.dataframe(df.head())

        except Exception as e:
            st.error(f"Error while generating dataset: {e}")

# ----------------------------
# Step 2: Train Models
# ----------------------------
# if st.session_state.df is not None:
#     st.header("Train a Model")
#     df = st.session_state.df
#     label_encoder = LabelEncoder()
#     y = label_encoder.fit_transform(df["NFR"].astype(str))
#     X = df["cleaned"].values

#     X_train_text, X_test_text, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42
#     )

#     model_choice = st.selectbox("Choose Model", ["Naive Bayes", "SVM", "Random Forest", "CNN", "LSTM"])
#     run_button = st.button("Run Model")

#     if run_button:
#         preds = None

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

#         if preds is not None:
#             acc = accuracy_score(y_test, preds)
#             st.success(f"{model_choice} Accuracy: {acc:.2f}")

#             st.text(classification_report(
#                 y_test,
#                 preds,
#                 labels=np.arange(len(label_encoder.classes_)),
#                 target_names=label_encoder.classes_
#             ))

#             results_df = pd.DataFrame({
#                 "RequirementText": X_test_text,
#                 "Actual": label_encoder.inverse_transform(y_test),
#                 "Predicted": label_encoder.inverse_transform(preds)
#             })

#             st.dataframe(results_df)

#             st.download_button(
#                 "Download Full Results",
#                 results_df.to_csv(index=False),
#                 "results.csv",
#                 "text/csv"
#             )

# ----------------------------
# Step 2: Train Models
# ----------------------------
if st.session_state.df is not None:
    st.header("Train a Model")
    df = st.session_state.df
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["NFR"].astype(str))
    X = df["cleaned"].values

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model_choice = st.selectbox("Choose Model", ["Naive Bayes", "SVM", "Random Forest", "CNN", "LSTM"])
    run_button = st.button("Run Model")

    if run_button:
        preds = None
        full_preds = None # Variable to store full dataset predictions

        # --- Training the chosen model ---
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
            
            # Predict on the entire dataset
            X_full_tfidf = tfidf.transform(X)
            full_preds = model.predict(X_full_tfidf)

        elif model_choice in ["CNN", "LSTM"]:
            tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
            tokenizer.fit_on_texts(X_train_text)
            
            max_len = 50
            X_train_pad = pad_sequences(tokenizer.texts_to_sequences(X_train_text), maxlen=max_len, padding="post", truncating="post")

            vocab_size = min(5000, len(tokenizer.word_index) + 1)
            
            if model_choice == "CNN":
                model = Sequential([
                    Embedding(vocab_size, 100, input_length=max_len),
                    Conv1D(128, 5, activation="relu"), GlobalMaxPooling1D(),
                    Dense(64, activation="relu"), Dense(len(label_encoder.classes_), activation="softmax")
                ])
            else: # LSTM
                model = Sequential([
                    Embedding(vocab_size, 100, input_length=max_len),
                    LSTM(128, dropout=0.2), Dense(64, activation="relu"),
                    Dense(len(label_encoder.classes_), activation="softmax")
                ])

            model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
            model.fit(X_train_pad, y_train, epochs=3, batch_size=32, validation_split=0.1, verbose=0)
            
            # Predict on the entire dataset
            X_full_seq = tokenizer.texts_to_sequences(X)
            X_full_pad = pad_sequences(X_full_seq, maxlen=max_len, padding="post", truncating="post")
            full_preds = np.argmax(model.predict(X_full_pad), axis=1)

        # --- ⭐ NEW: Displaying results for the ENTIRE dataset ---
        # if full_preds is not None:
        #     st.header("Full Dataset Classification Results")
        #     st.info("Yeh aapke poore dataset (100%) ka result hai, jismein har row ke liye prediction dikhaya gaya hai.")

        #     # Calculate overall accuracy
        #     overall_acc = accuracy_score(y, full_preds)
        #     st.success(f"Overall Accuracy on Full Dataset: {overall_acc:.2f}")

        #     # Create a DataFrame with full results
        #     full_results_df = pd.DataFrame({
        #         "RequirementText": df["RequirementText"].values, # Original text
        #         "Actual Label": label_encoder.inverse_transform(y),
        #         "Predicted Label": label_encoder.inverse_transform(full_preds)
        #     })

        #     # Display the full results table
        #     st.dataframe(full_results_df)

        #     # Add a download button for the full results
        #     st.download_button(
        #         "Download Full Results",
        #         full_results_df.to_csv(index=False).encode('utf-8'),
        #         "full_classification_results.csv",
        #         "text/csv"
        #     )


        # if full_preds is not None:
        #     st.header("Full Dataset Classification Results")
        #     st.info("Yeh aapke poore dataset (100%) ka result hai, jismein har row ke liye prediction dikhaya gaya hai.")

        #     # --- NEW: Add Classification Report for the full dataset ---
        #     st.subheader("Overall Performance Metrics")
        #     report = classification_report(
        #         y,
        #         full_preds,
        #         labels=np.arange(len(label_encoder.classes_)),
        #         target_names=label_encoder.classes_,
        #         output_dict=True # Use this to easily convert to DataFrame
        #     )
            
        #     # Convert the report to a nice table and display it
        #     report_df = pd.DataFrame(report).transpose()
        #     st.dataframe(report_df)
        #     # --- END NEW ---

        #     # Create a DataFrame with full results
        #     full_results_df = pd.DataFrame({
        #         "RequirementText": df["RequirementText"].values, # Original text
        #         "Actual Label": label_encoder.inverse_transform(y),
        #         "Predicted Label": label_encoder.inverse_transform(full_preds)
        #     })

        #     # Display the full results table
        #     st.subheader("Row-by-Row Classification")
        #     st.dataframe(full_results_df)

        #     # Add a download button for the full results
        #     st.download_button(
        #         "Download Full Results",
        #         full_results_df.to_csv(index=False).encode('utf-8'),
        #         "full_classification_results.csv",
        #         "text/csv"
        #     )
# --- ⭐ NEW (UPDATED): Displaying results for the ENTIRE dataset ---
        if full_preds is not None:
            st.header("Full Dataset Classification Results")
            st.info("Yeh aapke poore dataset (100%) ka result hai, jismein har row ke liye prediction dikhaya gaya hai.")

            # --- HIGHLIGHTED ACCURACY ---
            overall_acc = accuracy_score(y, full_preds)
            st.success(f"✅ Overall Accuracy on Full Dataset: {overall_acc:.2f}")
            # --- END ---

            # --- Add Classification Report for the full dataset ---
            st.subheader("Overall Performance Metrics (Precision, Recall, F1-Score)")
            report = classification_report(
                y,
                full_preds,
                labels=np.arange(len(label_encoder.classes_)),
                target_names=label_encoder.classes_,
                output_dict=True
            )
            
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df)
            # --- END ---

            # Create a DataFrame with full results
            full_results_df = pd.DataFrame({
                "RequirementText": df["RequirementText"].values,
                "Actual Label": label_encoder.inverse_transform(y),
                "Predicted Label": label_encoder.inverse_transform(full_preds)
            })

            # Display the full results table
            st.subheader("Row-by-Row Classification")
            st.dataframe(full_results_df)

            # Add a download button for the full results
            st.download_button(
                "Download Full Results",
                full_results_df.to_csv(index=False).encode('utf-8'),
                "full_classification_results.csv",
                "text/csv"
            )


# import streamlit as st
# import pandas as pd
# import numpy as np
# import re
# import google.generativeai as genai
# from io import StringIO

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

# # ----------------------------
# # Gemini API setup using Secrets
# # ----------------------------
# genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# st.title("Requirement Classification App Dataset Builder")

# # ----------------------------
# # Helpers
# # ----------------------------
# def clean_text(text):
#     text = str(text).lower()
#     text = re.sub(r"[^a-z0-9\s]", " ", text)
#     return re.sub(r"\s+", " ", text).strip()

# # ----------------------------
# # Initialize session state
# # ----------------------------
# if "df" not in st.session_state:
#     st.session_state.df = None

# # ----------------------------
# # Step 0: Upload dataset
# # ----------------------------
# uploaded_file = st.file_uploader(
#     "Upload your dataset (CSV/TSV). If empty, dataset will be generated using Gemini:",
#     type=["csv", "tsv"]
# )

# if uploaded_file is not None:
#     try:
#         df = pd.read_csv(uploaded_file)
#     except pd.errors.ParserError:
#         df = pd.read_csv(uploaded_file, sep="\t")

#     if "RequirementText" not in df.columns or "NFR" not in df.columns:
#         st.error("Uploaded file must contain 'RequirementText' and 'NFR' columns.")
#     else:
#         df["cleaned"] = df["RequirementText"].apply(clean_text)
#         st.session_state.df = df
#         st.success("Uploaded dataset loaded successfully")
#         st.dataframe(df.head())

# # ----------------------------
# # Step 1: Gemini dataset generation (if no upload)
# # ----------------------------
# if st.session_state.df is None:
#     num_rows = st.number_input(
#         "How many rows to generate?", min_value=5, value=20, step=5
#     )

#     if st.button("Generate Dataset via Gemini"):
#         try:
#             model = genai.GenerativeModel("gemini-1.5-flash")

#             prompt = f"""
#             Generate {num_rows} software requirement records in CSV format
#             with EXACTLY these two columns: RequirementText, NFR.
#             - 'RequirementText' must contain realistic requirement statements.
#             - 'NFR' must contain the type/label of the requirement (like Performance, Security, Usability etc).
#             - Each row must have exactly 2 values separated by a comma.
#             - Do not add commas or quotes inside the fields.
#             - Output only raw CSV rows (no explanation).
#             """

#             response = model.generate_content(prompt)
#             raw_text = response.text.strip()

#             # Keep only valid rows with 2 columns
#             lines = raw_text.split("\n")
#             csv_lines = [line for line in lines if len(line.split(",")) == 2]
#             csv_text = "\n".join(csv_lines)

#             df = pd.read_csv(StringIO(csv_text), names=["RequirementText", "NFR"])
#             df["cleaned"] = df["RequirementText"].apply(clean_text)

#             st.session_state.df = df
#             st.success("Dataset generated successfully via Gemini API")
#             st.dataframe(df.head())

#         except Exception as e:
#             st.error(f"Error while generating dataset: {e}")

# # ----------------------------
# # Step 2: Train Models
# # ----------------------------
# if st.session_state.df is not None:
#     df = st.session_state.df
#     label_encoder = LabelEncoder()
#     y = label_encoder.fit_transform(df["NFR"].astype(str))
#     X = df["cleaned"].values

#     X_train_text, X_test_text, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42
#     )

#     model_choice = st.selectbox("Choose Model", ["Naive Bayes", "SVM", "Random Forest", "CNN", "LSTM"])
#     run_button = st.button("Run Model")

#     if run_button:
#         preds = None

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

#         if preds is not None:
#             acc = accuracy_score(y_test, preds)
#             st.success(f"{model_choice} Accuracy: {acc:.2f}")

#             st.text(classification_report(
#                 y_test,
#                 preds,
#                 labels=np.arange(len(label_encoder.classes_)),
#                 target_names=label_encoder.classes_
#             ))

#             results_df = pd.DataFrame({
#                 "RequirementText": X_test_text,
#                 "Actual": label_encoder.inverse_transform(y_test),
#                 "Predicted": label_encoder.inverse_transform(preds)
#             })

#             st.dataframe(results_df)

#             st.download_button(
#                 "Download Full Results",
#                 results_df.to_csv(index=False),
#                 "results.csv",
#                 "text/csv"
#             )







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
# from sklearn.utils.multiclass import unique_labels
# from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# st.title("Requirement Classification App")

# uploaded_file = st.file_uploader("Upload your dataset (CSV/TSV)", type=["csv", "tsv"])

# if uploaded_file is not None:
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

#     text_col = st.selectbox("Select text column", df.columns)
#     label_col = st.selectbox("Select label column", df.columns)

#     X = df[text_col].astype(str)
#     y = df[label_col]

#     # Drop classes with only 1 sample to avoid stratify issues
#     class_counts = y.value_counts()
#     valid_classes = class_counts[class_counts > 1].index
#     mask = y.isin(valid_classes)
#     X = X[mask]
#     y = y[mask]

#     label_encoder = LabelEncoder()
#     y_encoded = label_encoder.fit_transform(y)

#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
#     )

#     model_choice = st.selectbox(
#         "Choose Model",
#         ["NaiveBayes", "SVM", "RandomForest", "CNN", "LSTM", "BERT", "RoBERTa"]
#     )

#     preds = []

#     # Classic ML models
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

#     # Deep Learning
#     elif model_choice in ["CNN", "LSTM"]:
#         tokenizer = Tokenizer(num_words=5000)
#         tokenizer.fit_on_texts(X_train)
#         X_train_seq = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=100)
#         X_test_seq = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=100)

#         model = Sequential()
#         model.add(Embedding(5000, 50, input_length=100))
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

#     # Transformers
#     elif model_choice in ["BERT", "RoBERTa"]:
#         if model_choice == "BERT":
#             model_name = "bert-base-uncased"
#         else:
#             model_name = "roberta-base"

#         bert_tokenizer = AutoTokenizer.from_pretrained(model_name)
#         bert_model = AutoModelForSequenceClassification.from_pretrained(
#             model_name, num_labels=len(label_encoder.classes_)
#         )
#         classifier = pipeline("text-classification", model=bert_model, tokenizer=bert_tokenizer)

#         preds = []
#         for text in X_test:
#             pred = classifier(text, truncation=True)
#             label = pred[0]["label"]
#             # Convert label to index
#             if label.startswith("LABEL_"):
#                 idx = int(label.split("_")[1])
#                 # Ensure idx is in range
#                 idx = idx if idx < len(label_encoder.classes_) else 0
#                 preds.append(idx)
#             else:
#                 preds.append(0)

#     # Safe evaluation
#     if len(preds) == len(y_test):
#         acc = accuracy_score(y_test, preds)
#         st.success(f"{model_choice} Accuracy: {acc:.2f}")

#         labels_used = unique_labels(y_test, preds)
#         target_names = [label_encoder.classes_[i] for i in labels_used]

#         st.text(classification_report(y_test, preds, labels=labels_used, target_names=target_names))

#         results_df = pd.DataFrame({
#             "Text": X_test,
#             "Actual": label_encoder.inverse_transform(y_test),
#             "Predicted": label_encoder.inverse_transform(preds)
#         })

#         st.dataframe(results_df)
#         st.download_button("Download Results", results_df.to_csv(index=False), "results.csv")
#     else:
#         st.error("Prediction length does not match test data length.")
