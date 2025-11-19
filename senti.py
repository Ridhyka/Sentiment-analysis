import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
from joblib import dump
import numpy as np
import nltk
from nltk.corpus import stopwords
import string
import os

# --- Configuration ---
DATA_FILE = 'reviews.csv'
# UPDATED: These now reflect the column names found in your CSV file.
REVIEW_TEXT_COL = 'Text'      # Input column for the review text (from your CSV)
RATING_SCORE_COL = 'Score'    # Input column for the 1-5 star rating (from your CSV)

# The rest of the script uses these standard names for clarity in the modeling process.
TEXT_COLUMN = 'reviewText'    # Internal name for the text column
TARGET_COLUMN = 'sentiment'   # Internal name for the binary target column (0 or 1)

MODEL_FILENAME = 'sentiment_pipeline.joblib'
RANDOM_STATE = 42

# Download NLTK resources (only needed once)
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    print("Downloading NLTK stopwords...")
    nltk.download('stopwords')

STOPWORDS = set(stopwords.words('english'))
PUNCTUATION = string.punctuation

# --- Data Loading and Cleaning ---
def load_and_clean_data(file_path, text_col_in, score_col_in):
    """Loads CSV, performs initial cleaning, and converts 1-5 Scores to binary 0/1 sentiment."""
    try:
        # Attempt to load the file
        df = pd.read_csv(file_path, encoding='utf-8')
        
        # Check if the required columns exist using the input names
        if text_col_in not in df.columns or score_col_in not in df.columns:
            print(f"ERROR: The file must contain the text column ('{text_col_in}') and the score column ('{score_col_in}').")
            print(f"Available columns: {list(df.columns)}")
            return None

        # Drop rows with missing values in critical columns
        df.dropna(subset=[text_col_in, score_col_in], inplace=True)
        
        # --- Sentiment Conversion Logic (5-star to Binary) ---
        # 1. Map scores 4 and 5 to 1 (Positive)
        # 2. Map scores 1 and 2 to 0 (Negative)
        # 3. Filter out neutral score 3
        df[TARGET_COLUMN] = df[score_col_in].apply(lambda x: 1 if x >= 4 else (0 if x <= 2 else None))
        
        # Drop rows where sentiment is None (Score 3 - Neutral reviews)
        df.dropna(subset=[TARGET_COLUMN], inplace=True)
        
        # Rename the input text column to the internal standard name
        df.rename(columns={text_col_in: TEXT_COLUMN}, inplace=True)
        
        # Final cleanup and type cast
        df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)
        
        print(f"Successfully loaded {len(df)} records from {file_path}.")
        print(f"Converted {score_col_in} ratings to binary {TARGET_COLUMN} (excluding neutral Score 3 reviews).")
        return df

    except FileNotFoundError:
        print(f"FATAL ERROR: The file '{file_path}' was not found in the current directory: {os.getcwd()}")
        print("Please ensure your Python script and the CSV file are in the same folder.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during file loading: {e}")
        return None

def preprocess_text(text):
    """Clean the text by removing punctuation, converting to lowercase, and removing stopwords."""
    if not isinstance(text, str):
        return ""
    # Remove punctuation
    text = ''.join([char for char in text if char not in PUNCTUATION])
    # Convert to lowercase
    text = text.lower()
    # Remove stopwords
    text = ' '.join([word for word in text.split() if word not in STOPWORDS])
    return text

# --- Execution Starts Here ---
# Updated the call signature to use the new configuration variables
df = load_and_clean_data(DATA_FILE, REVIEW_TEXT_COL, RATING_SCORE_COL) 

if df is not None:
    print("Starting data preprocessing...")
    df['cleaned_review'] = df[TEXT_COLUMN].apply(preprocess_text)

    # 1. Define Features (X) and Target (y)
    X = df['cleaned_review']
    y = df[TARGET_COLUMN]

    # 2. Split data into training and testing sets (80/20 split)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

    print(f"Training set size: {len(X_train)} reviews.")
    print(f"Testing set size: {len(X_test)} reviews.")
    
    # 3. Create a Pipeline: TF-IDF Vectorizer + LinearSVC Classifier
    model_pipeline = Pipeline([
        # TfidfVectorizer: Converts text to feature vectors.
        ('tfidf', TfidfVectorizer(
            ngram_range=(1, 2),        # Include single words (unigrams) and pairs (bigrams)
            max_df=0.9,                # Ignore terms appearing in > 90% of documents (as a float proportion)
            min_df=0.005,              # Ignore terms appearing in < 0.5% of documents
            stop_words='english'
        )),
        # LinearSVC: Support Vector Classifier, highly effective and fast for text data.
        ('clf', LinearSVC(C=1.0, max_iter=10000, random_state=RANDOM_STATE))
    ])

    print("\nStarting model training...")
    try:
        # Train the model
        model_pipeline.fit(X_train, y_train)
        print("Model training complete.")

        # --- Evaluation ---
        y_pred = model_pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"\nModel Accuracy on Test Set: {accuracy:.4f}")
        print("\nClassification Report (Accuracy, Precision, Recall, F1-Score):")
        print(classification_report(y_test, y_pred))

        # --- Model Persistence (Saving) ---
        dump(model_pipeline, MODEL_FILENAME)
        print(f"\n✅ Model successfully trained and saved as: {MODEL_FILENAME}")
        
    except ValueError as e:
        print(f"\n*** TRAINING FAILED DUE TO VALUE ERROR ***")
        print(f"Error: {e}")
        print("\nSUGGESTION: The parameters max_df/min_df may still be too strict for your data size.")
        print("Try changing max_df to 1.0 and min_df to 1 in the TfidfVectorizer and run again.")
    except Exception as e:
        print(f"\nAn unexpected training error occurred: {e}")