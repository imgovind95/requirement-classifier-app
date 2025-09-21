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
# # It's recommended to handle API key errors gracefully
# try:
#     genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
# except (KeyError, AttributeError):
#     st.error("GEMINI_API_KEY not found in Streamlit secrets. Please add it.")
#     st.stop()


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
#         # Attempt to read as CSV, if it fails, try TSV
#         try:
#             df = pd.read_csv(uploaded_file)
#         except pd.errors.ParserError:
#             uploaded_file.seek(0) # Reset file pointer
#             df = pd.read_csv(uploaded_file, sep="\t")

#         st.session_state.raw_df = df # Store the original dataframe
#         st.success("Uploaded dataset loaded successfully")
#         st.dataframe(df.head())

#     except Exception as e:
#         st.error(f"Error loading file: {e}")
#         st.session_state.raw_df = None


# # New section to allow column selection
# if "raw_df" in st.session_state and st.session_state.raw_df is not None:
#     df = st.session_state.raw_df
#     columns = df.columns.tolist()

#     st.subheader("Select Columns")
#     # Let the user select the text and label columns
#     text_column = st.selectbox("Select the column containing the requirement text:", columns)
#     nfr_column = st.selectbox("Select the column containing the NFR label:", columns)

#     if st.button("Confirm Columns and Process Data"):
#         # Create a new dataframe with the standard column names
#         processed_df = pd.DataFrame({
#             'RequirementText': df[text_column],
#             'NFR': df[nfr_column]
#         })
#         processed_df["cleaned"] = processed_df["RequirementText"].apply(clean_text)
#         st.session_state.df = processed_df
#         st.success("Columns mapped and data processed!")
#         st.dataframe(st.session_state.df.head())


# # ----------------------------
# # Step 1: Gemini dataset generation (if no upload)
# # ----------------------------
# if uploaded_file is None:
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
#     st.header("Train a Model")
#     df = st.session_state.df

#     # --- Automatic label detection code ---
#     functional_labels = ['functionality', '0'] 
#     st.info(f"Automatically identifying labels like {functional_labels} as 'Functional'.")

#     df['Binary_NFR'] = df['NFR'].apply(
#         lambda label: 'Functional' if str(label).strip().lower() in functional_labels else 'Non-Functional'
#     )
    
#     label_encoder = LabelEncoder()
#     y = label_encoder.fit_transform(df["Binary_NFR"]) 
#     X = df["cleaned"].values
#     # --- End of automatic code ---

#     # --- Data Health Check ---
#     st.subheader("Data Health Check")
#     class_counts = pd.Series(y).value_counts()
    
#     if not class_counts.empty:
#         class_names = label_encoder.inverse_transform(class_counts.index)
#         st.write("Number of samples in each class before splitting:")
#         st.dataframe(pd.DataFrame({'Class': class_names, 'Count': class_counts.values}))

#         if class_counts.min() < 2:
#             st.error("ERROR: One of your classes has fewer than 2 samples. Please use a more balanced dataset.")
#             st.stop()
#     else:
#         st.error("Could not find any labels to train on.")
#         st.stop()
#     # --- End of Health Check ---

#     X_train_text, X_test_text, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42, stratify=y 
#     )

#     model_choice = st.selectbox("Choose Model", ["Naive Bayes", "SVM", "Random Forest", "CNN", "LSTM"])
#     run_button = st.button("Run Model")

#     # --- ⭐ ERROR DETECTIVE IS ADDED HERE ---
#     if run_button:
#         try:
#             st.write("'Run Model' button clicked. Starting process...")
#             preds = None
#             full_preds = None 

#             if model_choice in ["Naive Bayes", "SVM", "Random Forest"]:
#                 st.write(f"Debug: Training {model_choice}...")
#                 tfidf = TfidfVectorizer(max_features=5000)
#                 X_train_tfidf = tfidf.fit_transform(X_train_text)
                
#                 if model_choice == "Naive Bayes": model = MultinomialNB()
#                 elif model_choice == "SVM": model = SVC(kernel="linear", random_state=42)
#                 else: model = RandomForestClassifier(n_estimators=100, random_state=42)

#                 model.fit(X_train_tfidf, y_train)
                
#                 st.write("Predicting on the full dataset...")
#                 X_full_tfidf = tfidf.transform(X)
#                 full_preds = model.predict(X_full_tfidf)

#             elif model_choice in ["CNN", "LSTM"]:
#                 st.write(f"Training {model_choice}...")
#                 tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
#                 tokenizer.fit_on_texts(X_train_text)
                
#                 max_len = 50
#                 X_train_pad = pad_sequences(tokenizer.texts_to_sequences(X_train_text), maxlen=max_len, padding="post", truncating="post")
#                 vocab_size = min(5000, len(tokenizer.word_index) + 1)
                
#                 if model_choice == "CNN":
#                     model = Sequential([Embedding(vocab_size, 100, input_length=max_len), Conv1D(128, 5, activation="relu"), GlobalMaxPooling1D(), Dense(64, activation="relu"), Dense(len(label_encoder.classes_), activation="softmax")])
#                 else:
#                     model = Sequential([Embedding(vocab_size, 100, input_length=max_len), LSTM(128, dropout=0.2), Dense(64, activation="relu"), Dense(len(label_encoder.classes_), activation="softmax")])

#                 model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
#                 model.fit(X_train_pad, y_train, epochs=3, batch_size=32, validation_split=0.1, verbose=0)
                
#                 st.write("Predicting on the full dataset...")
#                 X_full_seq = tokenizer.texts_to_sequences(X)
#                 X_full_pad = pad_sequences(X_full_seq, maxlen=max_len, padding="post", truncating="post")
#                 full_preds = np.argmax(model.predict(X_full_pad), axis=1)

#             st.write("Debug: Model training and prediction complete. Now displaying results.")

#             if full_preds is not None:
#                 st.header("Full Dataset Classification Results")
                
#                 st.write("Calculating accuracy...")
#                 overall_acc = accuracy_score(y, full_preds)
#                 st.success(f"Overall Accuracy on Full Dataset: {overall_acc:.2f}")

#                 st.write("Generating performance report...")
#                 report = classification_report(y, full_preds, labels=np.arange(len(label_encoder.classes_)), target_names=label_encoder.classes_, output_dict=True)
#                 report_df = pd.DataFrame(report).transpose()
#                 st.subheader("Overall Performance Metrics (Precision, Recall, F1-Score)")
#                 st.dataframe(report_df)

#                 st.write("Creating final results table...")
#                 full_results_df = pd.DataFrame({
#                     "RequirementText": df["RequirementText"].values,
#                     "Actual Label": label_encoder.inverse_transform(y),
#                     "Predicted Label": label_encoder.inverse_transform(full_preds)
#                 })
#                 st.subheader("Row-by-Row Classification")
#                 st.dataframe(full_results_df)

#                 st.download_button("Download Full Results", full_results_df.to_csv(index=False).encode('utf-8'), "full_classification_results.csv", "text/csv")
#                 st.write("All results displayed successfully!")

#         except Exception as e:
#             st.error("❌ OOPS! Ek error aa gaya. Neeche error ki detail di gayi hai:")
#             st.exception(e)
# import streamlit as st
# import pandas as pd
# import numpy as np
# import re
# import google.generativeai as genai
# from io import StringIO

# # New imports for Transformer models
# from transformers import pipeline
# import torch

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
# try:
#     genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
# except (KeyError, AttributeError):
#     st.error("GEMINI_API_KEY not found in Streamlit secrets. Please add it.")
#     st.stop()


# st.title("Requirement Classification App & Dataset Builder")

# # ----------------------------
# # Caching function for loading models
# # ----------------------------
# @st.cache_resource
# def load_zsl_pipeline(model_name):
#     """Loads a Zero-Shot Classification pipeline and caches it."""
#     st.info(f"Downloading and caching Zero-Shot model: {model_name}. This may take a few minutes on first run.")
#     return pipeline("zero-shot-classification", model=model_name)

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
# if "raw_df" not in st.session_state:
#     st.session_state.raw_df = None

# # ----------------------------
# # Step 0: Upload dataset
# # ----------------------------
# uploaded_file = st.file_uploader(
#     "Upload your dataset (CSV/TSV) or leave empty to generate:",
#     type=["csv", "tsv"]
# )

# if uploaded_file is not None:
#     try:
#         try:
#             df = pd.read_csv(uploaded_file)
#         except pd.errors.ParserError:
#             uploaded_file.seek(0) # Reset file pointer
#             df = pd.read_csv(uploaded_file, sep="\t")
#         st.session_state.raw_df = df
#         st.success("Uploaded dataset loaded successfully")
#         st.dataframe(df.head())
#     except Exception as e:
#         st.error(f"Error loading file: {e}")
#         st.session_state.raw_df = None

# # Section to allow column selection
# if st.session_state.raw_df is not None:
#     df_raw = st.session_state.raw_df
#     columns = df_raw.columns.tolist()

#     st.subheader("Map Your Columns")
#     text_column = st.selectbox("Select the column containing the requirement text:", columns, index=0)
#     nfr_column = st.selectbox("Select the column containing the NFR label:", columns, index=min(1, len(columns)-1))

#     if st.button("Confirm Columns and Process Data"):
#         processed_df = pd.DataFrame({
#             'RequirementText': df_raw[text_column],
#             'NFR': df_raw[nfr_column]
#         })
#         processed_df["cleaned"] = processed_df["RequirementText"].apply(clean_text)
#         st.session_state.df = processed_df
#         st.success("Columns mapped and data processed!")
#         st.dataframe(st.session_state.df.head())

# # ----------------------------
# # Step 1: Gemini dataset generation
# # ----------------------------
# if uploaded_file is None and st.session_state.df is None:
#     user_prompt = st.text_area(
#         "Alternatively, write a prompt to generate a dataset (CSV format):",
#         value="""Generate 20 software requirements in CSV format with EXACTLY 2 columns: RequirementText and NFR.
# The NFR column should contain either 'functionality' or a non-functional requirement type like 'usability', 'performance', 'security'.
# Each row must have exactly 2 fields. Do not use commas or quotes inside the fields.
# Separate columns using a comma. Each row must be on a new line. Output only CSV content, no extra explanation or text."""
#     )

#     if st.button("Generate Dataset"):
#         try:
#             model = genai.GenerativeModel("gemini-1.5-flash")
#             response = model.generate_content(user_prompt)
#             raw_text = response.text.strip().replace('`', '')
            
#             lines = raw_text.split("\n")
#             csv_lines = [line for line in lines if len(line.split(",")) == 2]
#             csv_text = "\n".join(csv_lines)

#             if not csv_text:
#                 st.error("Gemini did not return a valid CSV format. Please try again or adjust the prompt.")
#             else:
#                 df_gen = pd.read_csv(StringIO(csv_text), names=["RequirementText", "NFR"])
#                 df_gen["cleaned"] = df_gen["RequirementText"].apply(clean_text)
#                 st.session_state.df = df_gen
#                 st.success("Dataset generated successfully via Gemini API")
#                 st.dataframe(df_gen)

#         except Exception as e:
#             st.error(f"Error while generating dataset: {e}")

# # ----------------------------
# # Step 2: Train Models
# # ----------------------------
# if st.session_state.df is not None:
#     st.header("Train and Evaluate a Model")
#     df = st.session_state.df

#     # --- Automatic binary label creation ---
#     functional_labels = ['functionality', 'functional', '0']
#     df['Binary_NFR'] = df['NFR'].apply(
#         lambda label: 'Functional' if str(label).strip().lower() in functional_labels else 'Non-Functional'
#     )
    
#     label_encoder = LabelEncoder()
#     y = label_encoder.fit_transform(df["Binary_NFR"])
#     X = df["cleaned"].values
    
#     # --- Data Health Check ---
#     st.subheader("Data Health Check")
#     class_counts = pd.Series(y).value_counts()
    
#     if not class_counts.empty:
#         class_names = label_encoder.inverse_transform(class_counts.index)
#         st.write("Number of samples in each class:")
#         st.dataframe(pd.DataFrame({'Class': class_names, 'Count': class_counts.values}))

#         if len(class_counts) < 2:
#              st.error("ERROR: Your dataset must contain at least two classes (e.g., 'Functional' and 'Non-Functional').")
#              st.stop()
#         if class_counts.min() < 2:
#             st.warning("WARNING: One of your classes has very few samples. Model performance may be unreliable. Consider stratifying.")
#     else:
#         st.error("Could not find any labels to train on.")
#         st.stop()

#     # --- ⭐️ CORRECTED MODEL SELECTION ⭐️ ---
#     model_choice = st.selectbox(
#         "Choose Model",
#         [
#             "Naive Bayes", "SVM", "Random Forest", "CNN", "LSTM",
#             "Zero-Shot (BART-Large)", "Zero-Shot (DeBERTa-small)"
#         ]
#     )
#     run_button = st.button("Run Model and Classify Full Dataset")

#     if run_button:
#         try:
#             full_preds = None

#             if model_choice in ["Naive Bayes", "SVM", "Random Forest", "CNN", "LSTM"]:
#                 X_train_text, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            
#             if model_choice in ["Naive Bayes", "SVM", "Random Forest"]:
#                 tfidf = TfidfVectorizer(max_features=5000)
#                 X_train_tfidf = tfidf.fit_transform(X_train_text)
                
#                 if model_choice == "Naive Bayes": model = MultinomialNB()
#                 elif model_choice == "SVM": model = SVC(kernel="linear", random_state=42, probability=True)
#                 else: model = RandomForestClassifier(n_estimators=100, random_state=42)

#                 model.fit(X_train_tfidf, y_train)
#                 X_full_tfidf = tfidf.transform(X)
#                 full_preds = model.predict(X_full_tfidf)

#             elif model_choice in ["CNN", "LSTM"]:
#                 tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
#                 tokenizer.fit_on_texts(X_train_text)
                
#                 max_len = 50
#                 X_train_pad = pad_sequences(tokenizer.texts_to_sequences(X_train_text), maxlen=max_len, padding="post")
#                 vocab_size = min(5000, len(tokenizer.word_index) + 1)
                
#                 if model_choice == "CNN":
#                     model = Sequential([Embedding(vocab_size, 100, input_length=max_len), Conv1D(128, 5, activation="relu"), GlobalMaxPooling1D(), Dense(64, activation="relu"), Dense(len(label_encoder.classes_), activation="softmax")])
#                 else:
#                     model = Sequential([Embedding(vocab_size, 100, input_length=max_len), LSTM(128, dropout=0.2), Dense(64, activation="relu"), Dense(len(label_encoder.classes_), activation="softmax")])

#                 model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
#                 model.fit(X_train_pad, y_train, epochs=5, batch_size=32, verbose=0)
                
#                 X_full_pad = pad_sequences(tokenizer.texts_to_sequences(X), maxlen=max_len, padding="post")
#                 full_preds = np.argmax(model.predict(X_full_pad), axis=1)

#             elif model_choice in ["Zero-Shot (BART-Large)", "Zero-Shot (DeBERTa-small)"]:
#                 st.info("Zero-Shot models classify based on label meanings, no training is needed. This might take a while...")
                
#                 candidate_labels = label_encoder.classes_.tolist()
#                 # --- ⭐️ CORRECTED MODEL MAP ⭐️ ---
#                 model_map = {
#                     "Zero-Shot (BART-Large)": "facebook/bart-large-mnli",
#                     "Zero-Shot (DeBERTa-small)": "Narsil/deberta-v3-small-mnli"
#                 }
#                 zsl_pipeline = load_zsl_pipeline(model_map[model_choice])

#                 temp_preds = []
#                 progress_bar = st.progress(0, text="Classifying requirements...")
#                 for i, text in enumerate(X):
#                     res = zsl_pipeline(text, candidate_labels, multi_label=False)
#                     temp_preds.append(res['labels'][0])
#                     progress_bar.progress((i + 1) / len(X), text=f"Classifying requirements... {i+1}/{len(X)}")
                
#                 label_to_int = {label: i for i, label in enumerate(label_encoder.classes_)}
#                 full_preds = np.array([label_to_int[pred] for pred in temp_preds])

#             if full_preds is not None:
#                 st.header("Full Dataset Classification Results")
                
#                 overall_acc = accuracy_score(y, full_preds)
#                 st.success(f"Overall Accuracy on Full Dataset: {overall_acc:.2f}")

#                 report = classification_report(y, full_preds, target_names=label_encoder.classes_, output_dict=True)
#                 report_df = pd.DataFrame(report).transpose()
#                 st.subheader("Overall Performance Metrics")
#                 st.dataframe(report_df)

#                 full_results_df = pd.DataFrame({
#                     "RequirementText": df["RequirementText"].values,
#                     "Actual_Label": label_encoder.inverse_transform(y),
#                     "Predicted_Label": label_encoder.inverse_transform(full_preds)
#                 })
#                 st.subheader("Row-by-Row Classification")
#                 st.dataframe(full_results_df)

#                 st.download_button(
#                     "Download Full Results", 
#                     full_results_df.to_csv(index=False).encode('utf-8'), 
#                     "full_classification_results.csv", 
#                     "text/csv"
#                 )

#         except Exception as e:
#             st.error("❌ An error occurred during the process.")
#             st.exception(e)

# import streamlit as st
# import pandas as pd
# import numpy as np
# import re
# import google.generativeai as genai
# from io import StringIO

# # New imports for Transformer models
# from transformers import pipeline
# import torch

# # Import for plotting
# import matplotlib.pyplot as plt

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
# try:
#     genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
# except (KeyError, AttributeError):
#     st.error("GEMINI_API_KEY not found in Streamlit secrets. Please add it.")
#     st.stop()


# st.title("Requirement Classification App & Dataset Builder")

# # ----------------------------
# # Caching function for loading models
# # ----------------------------
# @st.cache_resource
# def load_zsl_pipeline(model_name):
#     """Loads a Zero-Shot Classification pipeline and caches it."""
#     st.info(f"Downloading and caching Zero-Shot model: {model_name}. This may take a few minutes on first run.")
#     return pipeline("zero-shot-classification", model=model_name)

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
# if "raw_df" not in st.session_state:
#     st.session_state.raw_df = None
# # New session state to store model accuracies for comparison
# if "model_accuracies" not in st.session_state:
#     st.session_state.model_accuracies = {}


# # ----------------------------
# # Step 0: Upload dataset
# # ----------------------------
# uploaded_file = st.file_uploader(
#     "Upload your dataset (CSV/TSV) or leave empty to generate:",
#     type=["csv", "tsv"]
# )

# if uploaded_file is not None:
#     try:
#         try:
#             df = pd.read_csv(uploaded_file)
#         except pd.errors.ParserError:
#             uploaded_file.seek(0)
#             df = pd.read_csv(uploaded_file, sep="\t")
#         st.session_state.raw_df = df
#         st.success("Uploaded dataset loaded successfully")
#         st.dataframe(df.head())
#     except Exception as e:
#         st.error(f"Error loading file: {e}")
#         st.session_state.raw_df = None

# # Section to allow column selection
# if st.session_state.raw_df is not None:
#     df_raw = st.session_state.raw_df
#     columns = df_raw.columns.tolist()

#     st.subheader("Map Your Columns")
#     text_column = st.selectbox("Select the column containing the requirement text:", columns, index=0)
#     nfr_column = st.selectbox("Select the column containing the NFR label:", columns, index=min(1, len(columns)-1))

#     if st.button("Confirm Columns and Process Data"):
#         processed_df = pd.DataFrame({
#             'RequirementText': df_raw[text_column],
#             'NFR': df_raw[nfr_column]
#         })
#         processed_df["cleaned"] = processed_df["RequirementText"].apply(clean_text)
#         st.session_state.df = processed_df
#         st.session_state.model_accuracies = {} # Clear old results when new data is processed
#         st.success("Columns mapped and data processed!")
#         st.dataframe(st.session_state.df.head())

# # ----------------------------
# # Step 1: Gemini dataset generation
# # ----------------------------
# if uploaded_file is None and st.session_state.df is None:
#     user_prompt = st.text_area(
#         "Alternatively, write a prompt to generate a dataset (CSV format):",
#         value="""Generate 20 software requirements in CSV format with EXACTLY 2 columns: RequirementText and NFR.
# The NFR column should contain either 'functionality' or a non-functional requirement type like 'usability', 'performance', 'security'.
# Each row must have exactly 2 fields. Do not use commas or quotes inside the fields.
# Separate columns using a comma. Each row must be on a new line. Output only CSV content, no extra explanation or text."""
#     )

#     if st.button("Generate Dataset"):
#         try:
#             model = genai.GenerativeModel("gemini-1.5-flash")
#             response = model.generate_content(user_prompt)
#             raw_text = response.text.strip().replace('`', '')
            
#             lines = raw_text.split("\n")
#             csv_lines = [line for line in lines if len(line.split(",")) == 2]
#             csv_text = "\n".join(csv_lines)

#             if not csv_text:
#                 st.error("Gemini did not return a valid CSV format. Please try again or adjust the prompt.")
#             else:
#                 df_gen = pd.read_csv(StringIO(csv_text), names=["RequirementText", "NFR"])
#                 df_gen["cleaned"] = df_gen["RequirementText"].apply(clean_text)
#                 st.session_state.df = df_gen
#                 st.session_state.model_accuracies = {} # Clear old results
#                 st.success("Dataset generated successfully via Gemini API")
#                 st.dataframe(df_gen)

#         except Exception as e:
#             st.error(f"Error while generating dataset: {e}")

# # ----------------------------
# # Step 2: Train Models
# # ----------------------------
# if st.session_state.df is not None:
#     st.header("Train and Evaluate a Model")
#     df = st.session_state.df

#     functional_labels = ['functionality', 'functional', '0']
#     df['Binary_NFR'] = df['NFR'].apply(
#         lambda label: 'Functional' if str(label).strip().lower() in functional_labels else 'Non-Functional'
#     )
    
#     label_encoder = LabelEncoder()
#     y = label_encoder.fit_transform(df["Binary_NFR"])
#     X = df["cleaned"].values
    
#     st.subheader("Data Health Check")
#     class_counts = pd.Series(y).value_counts()
    
#     if not class_counts.empty:
#         class_names = label_encoder.inverse_transform(class_counts.index)
#         st.write("Number of samples in each class:")
#         st.dataframe(pd.DataFrame({'Class': class_names, 'Count': class_counts.values}))

#         if len(class_counts) < 2:
#              st.error("ERROR: Your dataset must contain at least two classes (e.g., 'Functional' and 'Non-Functional').")
#              st.stop()
#         if class_counts.min() < 2:
#             st.warning("WARNING: One of your classes has very few samples. Model performance may be unreliable.")
#     else:
#         st.error("Could not find any labels to train on.")
#         st.stop()

#     model_choice = st.selectbox(
#         "Choose Model",
#         [
#             "Naive Bayes", "SVM", "Random Forest", "CNN", "LSTM",
#             "Zero-Shot (BART-Large)", "Zero-Shot (DeBERTa-small)"
#         ]
#     )
#     run_button = st.button("Run Model and Classify Full Dataset")

#     if run_button:
#         try:
#             full_preds = None

#             if model_choice in ["Naive Bayes", "SVM", "Random Forest", "CNN", "LSTM"]:
#                 X_train_text, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            
#             if model_choice in ["Naive Bayes", "SVM", "Random Forest"]:
#                 with st.spinner(f"Training {model_choice}..."):
#                     tfidf = TfidfVectorizer(max_features=5000)
#                     X_train_tfidf = tfidf.fit_transform(X_train_text)
#                     if model_choice == "Naive Bayes": model = MultinomialNB()
#                     elif model_choice == "SVM": model = SVC(kernel="linear", random_state=42, probability=True)
#                     else: model = RandomForestClassifier(n_estimators=100, random_state=42)
#                     model.fit(X_train_tfidf, y_train)
#                     X_full_tfidf = tfidf.transform(X)
#                     full_preds = model.predict(X_full_tfidf)

#             elif model_choice in ["CNN", "LSTM"]:
#                 with st.spinner(f"Training {model_choice}..."):
#                     tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
#                     tokenizer.fit_on_texts(X_train_text)
#                     max_len = 50
#                     X_train_pad = pad_sequences(tokenizer.texts_to_sequences(X_train_text), maxlen=max_len, padding="post")
#                     vocab_size = min(5000, len(tokenizer.word_index) + 1)
#                     if model_choice == "CNN":
#                         model = Sequential([Embedding(vocab_size, 100, input_length=max_len), Conv1D(128, 5, activation="relu"), GlobalMaxPooling1D(), Dense(64, activation="relu"), Dense(len(label_encoder.classes_), activation="softmax")])
#                     else: # LSTM
#                         model = Sequential([Embedding(vocab_size, 100, input_length=max_len), LSTM(128, dropout=0.2), Dense(64, activation="relu"), Dense(len(label_encoder.classes_), activation="softmax")])
#                     model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
#                     model.fit(X_train_pad, y_train, epochs=5, batch_size=32, verbose=0)
#                     X_full_pad = pad_sequences(tokenizer.texts_to_sequences(X), maxlen=max_len, padding="post")
#                     full_preds = np.argmax(model.predict(X_full_pad), axis=1)

#             elif model_choice in ["Zero-Shot (BART-Large)", "Zero-Shot (DeBERTa-small)"]:
#                 candidate_labels = label_encoder.classes_.tolist()
                
#                 # --- ⭐️ CORRECTED MODEL MAP ⭐️ ---
#                 model_map = {
#                     "Zero-Shot (BART-Large)": "facebook/bart-large-mnli",
#                     "Zero-Shot (DeBERTa-small)": "MoritzLaurer/DeBERTa-v3-small-mnli"
#                 }
#                 zsl_pipeline = load_zsl_pipeline(model_map[model_choice])

#                 with st.spinner(f"Classifying with {model_choice}..."):
#                     temp_preds = []
#                     progress_bar = st.progress(0, text="Classifying requirements...")
#                     for i, text in enumerate(X):
#                         res = zsl_pipeline(text, candidate_labels, multi_label=False)
#                         temp_preds.append(res['labels'][0])
#                         progress_bar.progress((i + 1) / len(X), text=f"Classifying... {i+1}/{len(X)}")
#                     label_to_int = {label: i for i, label in enumerate(label_encoder.classes_)}
#                     full_preds = np.array([label_to_int[pred] for pred in temp_preds])

#             if full_preds is not None:
#                 st.header(f"Results for: {model_choice}")
                
#                 overall_acc = accuracy_score(y, full_preds)
#                 st.session_state.model_accuracies[model_choice] = overall_acc
                
#                 st.success(f"Overall Accuracy on Full Dataset: {overall_acc:.2f}")

#                 report = classification_report(y, full_preds, target_names=label_encoder.classes_, output_dict=True)
#                 report_df = pd.DataFrame(report).transpose()
#                 st.subheader("Overall Performance Metrics")
#                 st.dataframe(report_df)

#                 full_results_df = pd.DataFrame({
#                     "RequirementText": df["RequirementText"].values,
#                     "Actual_Label": label_encoder.inverse_transform(y),
#                     "Predicted_Label": label_encoder.inverse_transform(full_preds)
#                 })
#                 st.subheader("Row-by-Row Classification")
#                 st.dataframe(full_results_df)

#                 st.download_button("Download Results", full_results_df.to_csv(index=False).encode('utf-8'), f"{model_choice}_results.csv", "text/csv")

#         except Exception as e:
#             st.error("❌ An error occurred during the process.")
#             st.exception(e)
            
#     # --- ⭐️ NEW GRAPH SECTION ⭐️ ---
#     if st.session_state.model_accuracies:
#         st.header("Model Performance Comparison")
        
#         accuracies = pd.Series(st.session_state.model_accuracies).sort_values(ascending=True)
        
#         fig, ax = plt.subplots()
#         bars = ax.barh(accuracies.index, accuracies.values)
#         ax.set_title("Model Accuracy Comparison")
#         ax.set_xlabel("Accuracy")
#         ax.set_xlim(0, 1.0)
        
#         for bar in bars:
#             width = bar.get_width()
#             label_x_pos = width + 0.01
#             ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{width:.2f}', va='center')
            
#         st.pyplot(fig)

#         if st.button("Clear Comparison Chart"):
#             st.session_state.model_accuracies = {}
#             st.rerun()
import streamlit as st
import pandas as pd
import numpy as np
import re
import google.generativeai as genai
from io import StringIO

# New imports for Transformer models
from transformers import pipeline
import torch

# Import for plotting
import matplotlib.pyplot as plt

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
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
except (KeyError, AttributeError):
    st.error("GEMINI_API_KEY not found in Streamlit secrets. Please add it.")
    st.stop()


st.title("Requirement Classification App & Dataset Builder")

# ----------------------------
# Caching function for loading models
# ----------------------------
@st.cache_resource
def load_zsl_pipeline(model_name):
    """Loads a Zero-Shot Classification pipeline and caches it."""
    st.info(f"Downloading and caching Zero-Shot model: {model_name}. This may take a few minutes on first run.")
    return pipeline("zero-shot-classification", model=model_name)

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
if "raw_df" not in st.session_state:
    st.session_state.raw_df = None
# New session state to store model accuracies for comparison
if "model_accuracies" not in st.session_state:
    st.session_state.model_accuracies = {}


# ----------------------------
# Step 0: Upload dataset
# ----------------------------
uploaded_file = st.file_uploader(
    "Upload your dataset (CSV/TSV) or leave empty to generate:",
    type=["csv", "tsv"]
)

if uploaded_file is not None:
    try:
        try:
            df = pd.read_csv(uploaded_file)
        except pd.errors.ParserError:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, sep="\t")
        st.session_state.raw_df = df
        st.success("Uploaded dataset loaded successfully")
        st.dataframe(df.head())
    except Exception as e:
        st.error(f"Error loading file: {e}")
        st.session_state.raw_df = None

# Section to allow column selection
if st.session_state.raw_df is not None:
    df_raw = st.session_state.raw_df
    columns = df_raw.columns.tolist()

    st.subheader("Map Your Columns")
    text_column = st.selectbox("Select the column containing the requirement text:", columns, index=0)
    nfr_column = st.selectbox("Select the column containing the NFR label:", columns, index=min(1, len(columns)-1))

    if st.button("Confirm Columns and Process Data"):
        processed_df = pd.DataFrame({
            'RequirementText': df_raw[text_column],
            'NFR': df_raw[nfr_column]
        })
        processed_df["cleaned"] = processed_df["RequirementText"].apply(clean_text)
        st.session_state.df = processed_df
        st.session_state.model_accuracies = {} # Clear old results when new data is processed
        st.success("Columns mapped and data processed!")
        st.dataframe(st.session_state.df.head())

# ----------------------------
# Step 1: Gemini dataset generation
# ----------------------------
if uploaded_file is None and st.session_state.df is None:
    user_prompt = st.text_area(
        "Alternatively, write a prompt to generate a dataset (CSV format):",
        value="""Generate 20 software requirements in CSV format with EXACTLY 2 columns: RequirementText and NFR.
The NFR column should contain either 'functionality' or a non-functional requirement type like 'usability', 'performance', 'security'.
Each row must have exactly 2 fields. Do not use commas or quotes inside the fields.
Separate columns using a comma. Each row must be on a new line. Output only CSV content, no extra explanation or text."""
    )

    if st.button("Generate Dataset"):
        try:
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(user_prompt)
            raw_text = response.text.strip().replace('`', '')
            
            lines = raw_text.split("\n")
            csv_lines = [line for line in lines if len(line.split(",")) == 2]
            csv_text = "\n".join(csv_lines)

            if not csv_text:
                st.error("Gemini did not return a valid CSV format. Please try again or adjust the prompt.")
            else:
                df_gen = pd.read_csv(StringIO(csv_text), names=["RequirementText", "NFR"])
                df_gen["cleaned"] = df_gen["RequirementText"].apply(clean_text)
                st.session_state.df = df_gen
                st.session_state.model_accuracies = {} # Clear old results
                st.success("Dataset generated successfully via Gemini API")
                st.dataframe(df_gen)

        except Exception as e:
            st.error(f"Error while generating dataset: {e}")

# ----------------------------
# Step 2: Train Models
# ----------------------------
if st.session_state.df is not None:
    st.header("Train and Evaluate a Model")
    df = st.session_state.df

    functional_labels = ['functionality', 'functional', '0']
    df['Binary_NFR'] = df['NFR'].apply(
        lambda label: 'Functional' if str(label).strip().lower() in functional_labels else 'Non-Functional'
    )
    
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["Binary_NFR"])
    X = df["cleaned"].values
    
    st.subheader("Data Health Check")
    class_counts = pd.Series(y).value_counts()
    
    if not class_counts.empty:
        class_names = label_encoder.inverse_transform(class_counts.index)
        st.write("Number of samples in each class:")
        st.dataframe(pd.DataFrame({'Class': class_names, 'Count': class_counts.values}))

        if len(class_counts) < 2:
             st.error("ERROR: Your dataset must contain at least two classes (e.g., 'Functional' and 'Non-Functional').")
             st.stop()
        if class_counts.min() < 2:
            st.warning("WARNING: One of your classes has very few samples. Model performance may be unreliable.")
    else:
        st.error("Could not find any labels to train on.")
        st.stop()

    model_choice = st.selectbox(
        "Choose Model",
        [
            "Naive Bayes", "SVM", "Random Forest", "CNN", "LSTM",
            "Zero-Shot (BART-Large)" 
        ]
    )
    run_button = st.button("Run Model and Classify Full Dataset")

    if run_button:
        try:
            full_preds = None

            if model_choice in ["Naive Bayes", "SVM", "Random Forest", "CNN", "LSTM"]:
                X_train_text, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            
            if model_choice in ["Naive Bayes", "SVM", "Random Forest"]:
                with st.spinner(f"Training {model_choice}..."):
                    tfidf = TfidfVectorizer(max_features=5000)
                    X_train_tfidf = tfidf.fit_transform(X_train_text)
                    if model_choice == "Naive Bayes": model = MultinomialNB()
                    elif model_choice == "SVM": model = SVC(kernel="linear", random_state=42, probability=True)
                    else: model = RandomForestClassifier(n_estimators=100, random_state=42)
                    model.fit(X_train_tfidf, y_train)
                    X_full_tfidf = tfidf.transform(X)
                    full_preds = model.predict(X_full_tfidf)

            elif model_choice in ["CNN", "LSTM"]:
                with st.spinner(f"Training {model_choice}..."):
                    tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
                    tokenizer.fit_on_texts(X_train_text)
                    max_len = 50
                    X_train_pad = pad_sequences(tokenizer.texts_to_sequences(X_train_text), maxlen=max_len, padding="post")
                    vocab_size = min(5000, len(tokenizer.word_index) + 1)
                    if model_choice == "CNN":
                        model = Sequential([Embedding(vocab_size, 100, input_length=max_len), Conv1D(128, 5, activation="relu"), GlobalMaxPooling1D(), Dense(64, activation="relu"), Dense(len(label_encoder.classes_), activation="softmax")])
                    else: # LSTM
                        model = Sequential([Embedding(vocab_size, 100, input_length=max_len), LSTM(128, dropout=0.2), Dense(64, activation="relu"), Dense(len(label_encoder.classes_), activation="softmax")])
                    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
                    model.fit(X_train_pad, y_train, epochs=5, batch_size=32, verbose=0)
                    X_full_pad = pad_sequences(tokenizer.texts_to_sequences(X), maxlen=max_len, padding="post")
                    full_preds = np.argmax(model.predict(X_full_pad), axis=1)

            elif model_choice in ["Zero-Shot (BART-Large)"]:
                candidate_labels = label_encoder.classes_.tolist()
                
                model_map = {
                    "Zero-Shot (BART-Large)": "facebook/bart-large-mnli"
                }
                zsl_pipeline = load_zsl_pipeline(model_map[model_choice])

                with st.spinner(f"Classifying with {model_choice}..."):
                    temp_preds = []
                    progress_bar = st.progress(0, text="Classifying requirements...")
                    for i, text in enumerate(X):
                        res = zsl_pipeline(text, candidate_labels, multi_label=False)
                        temp_preds.append(res['labels'][0])
                        progress_bar.progress((i + 1) / len(X), text=f"Classifying... {i+1}/{len(X)}")
                    label_to_int = {label: i for i, label in enumerate(label_encoder.classes_)}
                    full_preds = np.array([label_to_int[pred] for pred in temp_preds])

            if full_preds is not None:
                st.header(f"Results for: {model_choice}")
                
                overall_acc = accuracy_score(y, full_preds)
                st.session_state.model_accuracies[model_choice] = overall_acc
                
                st.success(f"Overall Accuracy on Full Dataset: {overall_acc:.2f}")

                report = classification_report(y, full_preds, target_names=label_encoder.classes_, output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                st.subheader("Overall Performance Metrics")
                st.dataframe(report_df)

                full_results_df = pd.DataFrame({
                    "RequirementText": df["RequirementText"].values,
                    "Actual_Label": label_encoder.inverse_transform(y),
                    "Predicted_Label": label_encoder.inverse_transform(full_preds)
                })
                st.subheader("Row-by-Row Classification")
                st.dataframe(full_results_df)

                st.download_button("Download Results", full_results_df.to_csv(index=False).encode('utf-8'), f"{model_choice}_results.csv", "text/csv")

        except Exception as e:
            st.error("❌ An error occurred during the process.")
            st.exception(e)
            
    # --- GRAPH SECTION ---
    if st.session_state.model_accuracies:
        st.header("Model Performance Comparison")
        
        accuracies = pd.Series(st.session_state.model_accuracies).sort_values(ascending=True)
        
        fig, ax = plt.subplots()
        bars = ax.barh(accuracies.index, accuracies.values)
        ax.set_title("Model Accuracy Comparison")
        ax.set_xlabel("Accuracy")
        ax.set_xlim(0, 1.0)
        
        for bar in bars:
            width = bar.get_width()
            label_x_pos = width + 0.01
            ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{width:.2f}', va='center')
            
        st.pyplot(fig)

        if st.button("Clear Comparison Chart"):
            st.session_state.model_accuracies = {}
            st.rerun()
