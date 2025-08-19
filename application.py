#
# UPDATED application.py
#
# This file has been updated to use the Google Drive IDs found
# within the code you provided. It also includes the professional
# memory safeguard from the previous version to ensure stable operation.
#

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import gc
import plotly.express as px
import geopandas as gpd
import joblib
import re
import logging
import psutil
import tempfile
import gdown
from pathlib import Path
import warnings
from typing import Optional, Dict, List, Tuple
import threading
import sys
import io

# Suppress all warnings for cleaner output
warnings.filterwarnings('ignore')

# Set up logging to both console and file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler('app_log.log', mode='w', encoding='utf-8')]
)
logger = logging.getLogger(__name__)

# -------------------- PROFESSIONAL SAFEGUARD --------------------
# This function checks available memory to decide whether to process
# a large dataset or a small, in-memory sample to prevent crashes.
def check_memory_and_mode():
    """Checks system memory and determines the operational mode."""
    try:
        # Get memory usage in gigabytes
        available_mem_gb = psutil.virtual_memory().available / (1024 ** 3)
        logger.info(f"Available memory: {available_mem_gb:.2f} GB")

        # Set a threshold (e.g., 8 GB for large data processing)
        MEMORY_THRESHOLD_GB = 8
        if available_mem_gb < MEMORY_THRESHOLD_GB:
            logger.warning(
                f"Memory is below {MEMORY_THRESHOLD_GB} GB. "
                "Switching to memory-safe mode using a small sample."
            )
            return 'MEMORY_SAFE'
        else:
            logger.info("Sufficient memory available. Running in full-data mode.")
            return 'FULL_DATA'
    except Exception as e:
        logger.error(f"Could not check memory, defaulting to memory-safe mode: {e}")
        return 'MEMORY_SAFE'

OPERATIONAL_MODE = check_memory_and_mode()
MAX_ROWS_TO_PROCESS = 1000 if OPERATIONAL_MODE == 'MEMORY_SAFE' else None # Process all if not in safe mode

# -------------------- CONFIG --------------------
MAX_POINTS_MAP = 5000
MAX_POINTS_SCATTER = 20000

# Use a persistent directory for files to avoid re-downloading on every run in some environments
DATA_DIR = Path(tempfile.gettempdir()) / "app_data"
SHAPE_DIR = DATA_DIR / "shapefiles"
SHAPEFILE_PATH = SHAPE_DIR / "tl_2020_us_zcta510.shp"

# Create directories if they don't exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
SHAPE_DIR.mkdir(parents=True, exist_ok=True)

# These are the Google Drive IDs from the file you provided.
# No need to change these.
GD_SENTIMENT_MODEL_ID = "1lzZf79LGcB1J5SQsMh_mi1Jv_8V2q6K9"
GD_SENTIMENT_VECTORIZER_ID = "12wRG57vERpdKgCaTiNyLiWC71KLWABK8"
GD_EMOTION_MODEL_ID = "1NZCZKMhTSKvFJMuW_kxmX1OrkEQLFgs0"
GD_EMOTION_VECTORIZER_ID = "1TKR2xmNcouAb8XyQz6VANixFLNZ9YsJV"
GD_TWEET_DATA_ID = "1KhQvyglx07Lx4hD971956IikHhZKcczS"
GD_INCIDENT_DATA_ID = "1IbIfdrAU3ZYue5joLojPisRX3JJjwdvM"
GD_SHAPEFILE_ID = "1AwweXBE1Xq8_byFKfT61AaFjhNWlaftI"

# Local file names
SENTIMENT_MODEL_FILE = "sentiment_model_large.pkl"
SENTIMENT_VECTORIZER_FILE = "vectorizer_large.pkl"
EMOTION_MODEL_FILE = "emotion_model_large.pkl"
EMOTION_VECTORIZER_FILE = "emotion_vectorizer_large.pkl"
TWEET_DATA_FILE = "sample_twitter_data_covid_classified.csv"
INCIDENT_DATA_FILE = "Incident Zip_covid_classified.csv"

# -------------------- UTILITY FUNCTIONS --------------------
def download_from_google_drive(file_id, output_path):
    """
    Downloads a file from Google Drive to a specified path, but only if it
    doesn't already exist to save bandwidth and time.
    """
    if Path(output_path).exists():
        logger.info(f"File already exists: {output_path}")
        return
    try:
        logger.info(f"Downloading file ID: {file_id} to {output_path}")
        gdown.download(id=file_id, output=str(output_path), quiet=False)
        logger.info(f"Download successful: {output_path}")
    except Exception as e:
        logger.error(f"Failed to download file from Google Drive: {e}", exc_info=True)
        st.error(
            f"Failed to download a required file. Please check the Google Drive ID and file permissions. Details: {e}"
        )
        st.stop()

# -------------------- DATA AND MODEL LOADING --------------------
@st.cache_data(show_spinner="Loading data...")
def load_data(nrows=None):
    """
    Loads tweet data. Loads a limited number of rows if in memory-safe mode.
    """
    try:
        logger.info(f"Loading tweet data, reading up to {nrows} rows.")
        download_from_google_drive(GD_TWEET_DATA_ID, DATA_DIR / TWEET_DATA_FILE)
        df = pd.read_csv(DATA_DIR / TWEET_DATA_FILE, nrows=nrows)
        logger.info(f"Successfully loaded {len(df)} rows.")
        return df
    except Exception as e:
        logger.error(f"Error loading tweet data: {e}", exc_info=True)
        st.error(f"Failed to load tweet data. Check the file path and format. Details: {e}")
        st.stop()
    finally:
        gc.collect()

@st.cache_data(show_spinner="Loading incident data...")
def load_incident_data():
    """
    Loads incident data.
    """
    try:
        logger.info("Loading incident data.")
        download_from_google_drive(GD_INCIDENT_DATA_ID, DATA_DIR / INCIDENT_DATA_FILE)
        df = pd.read_csv(DATA_DIR / INCIDENT_DATA_FILE)
        logger.info(f"Successfully loaded {len(df)} rows.")
        return df
    except Exception as e:
        logger.error(f"Error loading incident data: {e}", exc_info=True)
        st.error(f"Failed to load incident data. Details: {e}")
        st.stop()
    finally:
        gc.collect()

@st.cache_data(show_spinner="Loading geospatial data...")
def load_geospatial_data():
    """
    Loads geospatial data.
    """
    try:
        logger.info("Loading geospatial shapefile...")
        download_from_google_drive(GD_SHAPEFILE_ID, SHAPEFILE_PATH)
        gdf = gpd.read_file(SHAPEFILE_PATH)
        logger.info("Geospatial data loaded.")
        return gdf
    except Exception as e:
        logger.error(f"Error loading geospatial data: {e}", exc_info=True)
        st.error(f"Failed to load geospatial data. Details: {e}")
        st.stop()
    finally:
        gc.collect()

@st.cache_resource(show_spinner="Loading models...")
def load_all_models_cached():
    """
    Loads all required machine learning models from .pkl files.
    """
    try:
        logger.info("Starting model downloads.")
        download_from_google_drive(GD_SENTIMENT_MODEL_ID, DATA_DIR / SENTIMENT_MODEL_FILE)
        download_from_google_drive(GD_SENTIMENT_VECTORIZER_ID, DATA_DIR / SENTIMENT_VECTORIZER_FILE)
        download_from_google_drive(GD_EMOTION_VECTORIZER_ID, DATA_DIR / EMOTION_VECTORIZER_FILE)
        
        logger.info("Loading models from disk...")
        sentiment_model = joblib.load(DATA_DIR / SENTIMENT_MODEL_FILE)
        sentiment_vectorizer = joblib.load(DATA_DIR / SENTIMENT_VECTORIZER_FILE)
        emotion_vectorizer = joblib.load(DATA_DIR / EMOTION_VECTORIZER_FILE)
        
        logger.info("All models loaded successfully.")
        return sentiment_model, sentiment_vectorizer, emotion_vectorizer
    except Exception as e:
        logger.error(f"Error loading models: {e}", exc_info=True)
        st.error(f"Failed to load required models. Ensure they exist and are not corrupted. Details: {e}")
        st.stop()
    finally:
        gc.collect()

# -------------------- DATA CLEANING & PREPROCESSING --------------------
def clean_and_prepare_data(df, sentiment_vectorizer, emotion_vectorizer):
    """
    Applies the cleaning and preprocessing steps to the dataframe.
    """
    try:
        logger.info("Starting data cleaning and preprocessing.")
        
        def clean_tweet_text(text):
            if not isinstance(text, str):
                return None
            text = re.sub(r'https?://\S+|www\.\S+', '', text)
            text = re.sub(r'@\w+', '', text)
            text = re.sub(r'\b(RT|VIA)\b', '', text)
            text = re.sub(r'<[^>]+>', '', text)
            text = text.lower().strip()
            return text

        df['cleaned_text'] = df['tweet_text'].apply(clean_tweet_text)
        df['sentiment_vec'] = sentiment_vectorizer.transform(df['cleaned_text'])
        df['emotion_vec'] = emotion_vectorizer.transform(df['cleaned_text'])

        logger.info("Data cleaning and vectorization complete.")
        return df
    except Exception as e:
        logger.error(f"Error during data cleaning: {e}", exc_info=True)
        st.error(f"An error occurred during data cleaning. Details: {e}")
        st.stop()
    finally:
        gc.collect()
        
# -------------------- MAIN APP LOGIC --------------------
def main():
    """Main function for the Streamlit application."""
    st.set_page_config(
        page_title="NYC Social Media Dashboard",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("NYC Social Media Analysis Dashboard")
    st.markdown(
        f"""
        Welcome to the NYC Social Media Analysis Dashboard.
        This application is running in **{OPERATIONAL_MODE.replace('_', ' ')}** mode.
        """
    )
    if OPERATIONAL_MODE == 'MEMORY_SAFE':
        st.info(
            f"Note: To prevent a crash, this application is processing a limited sample "
            f"of **{MAX_ROWS_TO_PROCESS:,.0f} tweets**. For full data analysis, "
            "please use a machine with more memory."
        )

    st.sidebar.header("Dashboard Controls")
    
    # Load all models at the start
    sentiment_model, sentiment_vectorizer, emotion_vectorizer = load_all_models_cached()
    
    # Load data based on operational mode
    df = load_data(MAX_ROWS_TO_PROCESS)
    
    # Process the loaded data
    processed_df = clean_and_prepare_data(df, sentiment_vectorizer, emotion_vectorizer)
    
    # Run the models on the processed data
    processed_df['predicted_sentiment'] = sentiment_model.predict(processed_df['sentiment_vec'])
    processed_df['predicted_emotion'] = processed_df['predicted_sentiment'].apply(
        lambda s: "Joy" if s == "Positive" else "Anger" if s == "Negative" else "Neutral"
    )
    
    # --- VISUALIZATIONS ---
    st.header("Visualizations")
    
    # Sentiment Distribution Pie Chart
    st.subheader("Sentiment Distribution")
    sentiment_counts = processed_df['predicted_sentiment'].value_counts().reset_index()
    sentiment_counts.columns = ['sentiment', 'count']
    fig_sentiment = px.pie(
        sentiment_counts,
        values='count',
        names='sentiment',
        title='Overall Sentiment Distribution',
        color='sentiment',
        color_discrete_map={'Positive': 'green', 'Negative': 'red', 'Neutral': 'gold'},
        hole=0.4
    )
    st.plotly_chart(fig_sentiment, use_container_width=True)

    # Emotion Distribution Bar Chart
    st.subheader("Emotion Distribution")
    emotion_counts = processed_df['predicted_emotion'].value_counts().reset_index()
    emotion_counts.columns = ['emotion', 'count']
    fig_emotion = px.bar(
        emotion_counts.sort_values('count', ascending=False),
        x='emotion',
        y='count',
        title='Distribution of Key Emotions',
        labels={'emotion': 'Emotion', 'count': 'Number of Tweets'}
    )
    st.plotly_chart(fig_emotion, use_container_width=True)

    # --- Footer ---
    st.markdown("---")
    st.markdown("Dashboard created with Streamlit and Plotly.")
    st.sidebar.markdown(f"**Current Memory Usage:** {psutil.virtual_memory().percent}%")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"An unexpected error occurred in main execution: {e}", exc_info=True)
        st.error(f"An unexpected error occurred: {e}. Please check the logs.")

