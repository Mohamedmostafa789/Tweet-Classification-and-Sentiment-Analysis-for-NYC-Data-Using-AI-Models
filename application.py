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
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import time
import os
import psutil
import tempfile
import gdown
from pathlib import Path
from queue import Queue
import warnings
from typing import Optional, Dict, List, Tuple
import threading
import sys

# Suppress all warnings for cleaner output
warnings.filterwarnings('ignore')

# Set up logging to both console and file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler('app_log.log', mode='w', encoding='utf-8')]
)
logger = logging.getLogger(__name__)

# -------------------- CONFIG --------------------
MAX_POINTS_MAP = 5000
MAX_POINTS_SCATTER = 20000

# Use a persistent directory for files to avoid re-downloading on every run in some environments
DATA_DIR = Path(tempfile.gettempdir()) / "app_data"
SHAPE_DIR = DATA_DIR / "shapefiles"
SHAPEFILE_PATH = SHAPE_DIR / "tl_2020_36_zcta520.shp"
SHAPE_URL = 'https://github.com/nyccla/nyc-zip-code-data/raw/master/tl_2020_36_zcta520.zip'
NYC_CSV_URL = 'https://docs.google.com/spreadsheets/d/1B0qf7r8s-y5X7_N_yC9C4X_x_y-V_y_C_x_y/export?format=csv&gid=0' # Placeholder URL
MODELS_DIR = DATA_DIR / "models"

# -------------------- DATA LOADING --------------------
@st.cache_data
def download_file(url, output_path):
    """
    Downloads a file from a URL to a specified path.
    Uses gdown for Google Drive files and requests for others.
    """
    try:
        if "drive.google.com" in url or "googledrive.com" in url:
            gdown.download(url, output_path, quiet=False)
        else:
            import requests
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        logger.info(f"Successfully downloaded {output_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to download {url}. Error: {e}")
        return False

@st.cache_data
def load_and_cache_zipcodes():
    """
    Loads and caches NYC zipcode shapefile data.
    """
    try:
        if not SHAPEFILE_PATH.exists():
            SHAPE_DIR.mkdir(parents=True, exist_ok=True)
            logger.info("Downloading NYC zipcode shapefile...")
            if not download_file(SHAPE_URL, SHAPE_DIR / "tl_2020_36_zcta520.zip"):
                return None
            
            import zipfile
            with zipfile.ZipFile(SHAPE_DIR / "tl_2020_36_zcta520.zip", 'r') as zip_ref:
                zip_ref.extractall(SHAPE_DIR)
            logger.info("Extracted shapefile.")
        
        gdf = gpd.read_file(SHAPEFILE_PATH)
        gdf = gdf.to_crs(epsg=4326)
        gdf['ZCTA5CE20'] = gdf['ZCTA5CE20'].astype(str)
        logger.info("Successfully loaded GeoDataFrame.")
        return gdf
    except Exception as e:
        st.error(f"Error loading geospatial data: {e}")
        logger.error(f"Error loading GeoDataFrame: {e}")
        return None

def download_model(file_id, model_name):
    """Downloads a model file from Google Drive."""
    model_path = MODELS_DIR / model_name
    if not MODELS_DIR.exists():
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        
    url = f"https://drive.google.com/uc?id={file_id}"
    logger.info(f"Downloading model {model_name} from {url}...")
    try:
        if not download_file(url, model_path):
            st.error(f"Failed to download {model_name}. Please check the file ID and your connection.")
            return None
        return joblib.load(model_path)
    except Exception as e:
        logger.error(f"Failed to load model {model_path}. Error: {e}")
        st.error(f"Failed to load model {model_name}. Error: {e}")
        return None

@st.cache_resource
def load_all_models_cached():
    """
    Caches and loads all ML models.
    """
    sentiment_model = download_model('1vS2S1Vd3G7zS0vT1T1T0T3T2T4T3T7T2', 'sentiment_model.joblib')
    sentiment_vectorizer = download_model('1vS2S1Vd3G7zS0vT1T1T0T3T2T4T3T7T2', 'sentiment_vectorizer.joblib') # Placeholder ID
    emotion_model = download_model('1tS3S2T0T3T2S1S1S1S1S1T1T1S1S0S1', 'emotion_model.joblib')
    emotion_vectorizer = download_model('1tS3S2T0T3T2S1S1S1S1S1T1T1S1S0S1', 'emotion_vectorizer.joblib') # Placeholder ID
    return sentiment_model, sentiment_vectorizer, emotion_model, emotion_vectorizer

# -------------------- DATA PROCESSING --------------------
def predict_sentiment_and_emotion(text_series, sentiment_model, sentiment_vectorizer, emotion_model, emotion_vectorizer):
    """
    Predicts sentiment and emotion for a series of texts.
    """
    if sentiment_model and sentiment_vectorizer:
        text_features_sentiment = sentiment_vectorizer.transform(text_series)
        sentiment_predictions = sentiment_model.predict(text_features_sentiment)
        sentiment_prob = sentiment_model.predict_proba(text_features_sentiment)
        max_sentiment_prob = np.max(sentiment_prob, axis=1)
        
    if emotion_model and emotion_vectorizer:
        text_features_emotion = emotion_vectorizer.transform(text_series)
        emotion_predictions = emotion_model.predict(text_features_emotion)
        emotion_prob = emotion_model.predict_proba(text_features_emotion)
        max_emotion_prob = np.max(emotion_prob, axis=1)
    
    return sentiment_predictions, max_sentiment_prob, emotion_predictions, max_emotion_prob

def clean_tweet(text):
    """
    A basic function to clean a single tweet text.
    """
    text = str(text)
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)
    text = re.sub(r'#', '', text)
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
    text = text.strip()
    return text

def process_data(df, sentiment_model, sentiment_vectorizer, emotion_model, emotion_vectorizer):
    """
    Processes the DataFrame by cleaning text and adding sentiment/emotion predictions.
    """
    if 'text' not in df.columns:
        st.error("The uploaded file must contain a 'text' column.")
        return None
    
    # Clean up the text data
    df['clean_text'] = df['text'].apply(clean_tweet)
    
    # Predict sentiments and emotions
    df['sentiment_pred'], df['sentiment_prob'], df['emotion_pred'], df['emotion_prob'] = \
        predict_sentiment_and_emotion(df['clean_text'], sentiment_model, sentiment_vectorizer, emotion_model, emotion_vectorizer)
        
    df = df.astype({'sentiment_pred': 'category', 'emotion_pred': 'category'})
    
    return df

# -------------------- VISUALIZATIONS --------------------
def general_sentiment_chart(df):
    st.subheader("General Sentiment Distribution")
    sentiment_counts = df['sentiment_pred'].value_counts().reset_index()
    sentiment_counts.columns = ['sentiment', 'count']
    fig = px.bar(
        sentiment_counts, 
        x='sentiment', 
        y='count', 
        color='sentiment',
        title="Distribution of Sentiment",
        color_discrete_map={'positive': 'green', 'negative': 'red', 'neutral': 'blue'}
    )
    st.plotly_chart(fig, use_container_width=True)

def emotion_sentiment_heatmap(df):
    st.subheader("Emotion vs Sentiment Heatmap")
    df_agg = df.groupby(['emotion_pred', 'sentiment_pred']).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df_agg, annot=True, fmt='d', cmap='YlGnBu', ax=ax)
    ax.set_title("Emotion vs Sentiment Heatmap")
    st.pyplot(fig)

def zip_code_maps(incident_df, nyc_gdf):
    st.subheader("Zip Code Sentiment Map")
    
    # Check for required columns before proceeding
    required_cols = ['zip', 'latitude', 'longitude']
    if not all(col in incident_df.columns for col in required_cols):
        st.error(
            "This visualization requires the uploaded data to have 'zip', 'latitude', "
            "and 'longitude' columns. Please upload a dataset with this information."
        )
        return

    st.write(f"Displaying up to {MAX_POINTS_MAP} points to ensure performance.")
    df_sample = incident_df.sample(n=min(len(incident_df), MAX_POINTS_MAP), random_state=42)
    
    fig = px.scatter_mapbox(
        df_sample,
        lat="latitude",
        lon="longitude",
        color="sentiment_pred",
        color_discrete_map={'positive': 'green', 'negative': 'red', 'neutral': 'blue'},
        hover_data=["text", "zip"],
        mapbox_style="carto-positron",
        zoom=9,
        title="Incident Sentiment by Location",
    )
    fig.update_layout(mapbox_center={"lat": 40.7128, "lon": -74.0060})
    st.plotly_chart(fig, use_container_width=True)

def zip_code_heatmap(incident_df, nyc_gdf):
    st.subheader("Zip Code Sentiment Heatmap")
    
    # Check for required columns before proceeding
    required_cols = ['zip']
    if not all(col in incident_df.columns for col in required_cols):
        st.error(
            "This visualization requires the uploaded data to have a 'zip' column. "
            "Please upload a dataset with this information."
        )
        return
        
    df_grouped = incident_df.groupby(['zip', 'sentiment_pred']).size().unstack(fill_value=0)
    df_grouped.columns = [f'{col}_count' for col in df_grouped.columns]
    
    df_merged = nyc_gdf.merge(
        df_grouped,
        left_on='ZCTA5CE20',
        right_on='zip',
        how='left'
    )
    df_merged = df_merged.fillna(0)
    
    sentiment_to_display = st.selectbox(
        "Select Sentiment to Display:", 
        ['positive', 'negative', 'neutral'],
        index=0,
        key='heatmap_sentiment_select'
    )
    
    column_name = f'{sentiment_to_display}_count'
    if column_name not in df_merged.columns:
        st.warning(f"No data available for the '{sentiment_to_display}' sentiment.")
        df_merged[column_name] = 0
    
    fig = px.choropleth_mapbox(
        df_merged,
        geojson=df_merged.geometry,
        locations=df_merged.index,
        color=column_name,
        color_continuous_scale="Viridis",
        mapbox_style="carto-positron",
        zoom=9,
        center={"lat": 40.7128, "lon": -74.0060},
        opacity=0.5,
        title=f"Distribution of {sentiment_to_display.capitalize()} Sentiment by ZIP Code"
    )
    st.plotly_chart(fig, use_container_width=True)

def borough_income_chart(df):
    st.subheader("Average Median Income by Borough")
    if 'median_income' not in df.columns or 'borough' not in df.columns:
        st.error(
            "This visualization requires the uploaded data to have 'median_income' "
            "and 'borough' columns. Please upload a dataset with this information."
        )
        return

    borough_income = df.groupby('borough')['median_income'].mean().reset_index()
    fig = px.bar(
        borough_income,
        x='borough',
        y='median_income',
        color='borough',
        title="Average Median Income by Borough"
    )
    st.plotly_chart(fig, use_container_width=True)

# -------------------- MAIN PAGE LOGIC --------------------
def main_dashboard():
    st.title("📊 Twitter Sentiment Analysis Dashboard")
    st.markdown("### Upload your Twitter data (CSV file) to analyze sentiment and emotion.")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("File uploaded successfully!")
            st.write("First 5 rows of your data:")
            st.dataframe(df.head())

            with st.spinner("Processing data... This may take a few minutes for large files."):
                processed_df = process_data(df, st.session_state.sentiment_model, st.session_state.sentiment_vectorizer, st.session_state.emotion_model, st.session_state.emotion_vectorizer)
                st.session_state.df = processed_df
                
                # Check for required columns for specific visualizations and create incident_df if they exist
                if all(col in processed_df.columns for col in ['zip', 'latitude', 'longitude']):
                    st.session_state.incident_df = processed_df
                else:
                    st.session_state.incident_df = pd.DataFrame(columns=['zip', 'latitude', 'longitude'])

                # Store the NYC GeoDataFrame for use in maps
                st.session_state.nyc_gdf = load_and_cache_zipcodes()
            
            st.success("Data processing complete!")

            if st.session_state.df is not None:
                st.sidebar.subheader("Dashboard Options")
                visualization_choice = st.sidebar.selectbox(
                    "Choose a visualization:",
                    [
                        "📈 General Sentiment Chart",
                        "🔥 Emotion vs Sentiment Heatmap",
                        "🗺 ZIP Code Sentiment Maps",
                        "🗺 ZIP Code Sentiment Heatmap",
                        "💰 Average Median Income by Borough"
                    ]
                )

                if visualization_choice == "📈 General Sentiment Chart":
                    general_sentiment_chart(st.session_state.df)
                elif visualization_choice == "🔥 Emotion vs Sentiment Heatmap":
                    emotion_sentiment_heatmap(st.session_state.df)
                elif visualization_choice == "🗺 ZIP Code Sentiment Maps":
                    # Check if incident_df and nyc_gdf are available and have required columns
                    if 'incident_df' in st.session_state and not st.session_state.incident_df.empty and 'nyc_gdf' in st.session_state and st.session_state.nyc_gdf is not None:
                        zip_code_maps(st.session_state.incident_df, st.session_state.nyc_gdf)
                    else:
                        st.warning("Please upload a dataset with 'zip', 'latitude', and 'longitude' columns to view this visualization.")
                elif visualization_choice == "🗺 ZIP Code Sentiment Heatmap":
                    # Check if incident_df and nyc_gdf are available and have required columns
                    if 'incident_df' in st.session_state and not st.session_state.incident_df.empty and 'nyc_gdf' in st.session_state and st.session_state.nyc_gdf is not None:
                        zip_code_heatmap(st.session_state.incident_df, st.session_state.nyc_gdf)
                    else:
                        st.warning("Please upload a dataset with a 'zip' column to view this visualization.")
                elif visualization_choice == "💰 Average Median Income by Borough":
                    borough_income_chart(st.session_state.df)

        except Exception as e:
            st.error(f"An error occurred: {e}")
            logger.error(f"Main dashboard error: {e}")

def combined_prediction_page():
    st.title("🔍 Combined Prediction")
    st.markdown("### Enter a sentence to predict its sentiment and emotion.")
    
    user_input = st.text_area("Enter your text here:", height=150, key='user_input_text_area')
    
    if st.button("Predict"):
        if not user_input.strip():
            st.warning("Please enter some text to analyze.")
            return

        with st.spinner("Predicting..."):
            text_series = pd.Series([user_input])
            sentiment_pred, sentiment_prob, emotion_pred, emotion_prob = predict_sentiment_and_emotion(
                text_series,
                st.session_state.sentiment_model,
                st.session_state.sentiment_vectorizer,
                st.session_state.emotion_model,
                st.session_state.emotion_vectorizer
            )
            
            st.success("Prediction Complete!")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Sentiment Prediction")
                st.metric("Sentiment", sentiment_pred[0].capitalize())
                st.progress(float(sentiment_prob[0]))
                st.caption(f"Confidence: {sentiment_prob[0]*100:.2f}%")
                
            with col2:
                st.subheader("Emotion Prediction")
                st.metric("Emotion", emotion_pred[0].capitalize())
                st.progress(float(emotion_prob[0]))
                st.caption(f"Confidence: {emotion_prob[0]*100:.2f}%")

def project_report_page():
    st.title("📄 Project Report")
    st.markdown("""
        ### Twitter Sentiment Analysis Project

        #### Introduction
        This project aims to analyze the sentiment and emotions expressed in Twitter data. Using machine learning models, we can classify tweets as 'positive', 'negative', or 'neutral' and also categorize them into different emotions such as 'joy', 'anger', 'sadness', etc. This dashboard provides a user-friendly interface to upload data, run predictions, and visualize the results.

        #### Methodology
        1.  **Data Ingestion:** The application allows users to upload a CSV file containing tweets. The file is expected to have a 'text' column.
        2.  **Text Preprocessing:** The tweets are cleaned to remove URLs, mentions, hashtags, and other non-essential characters. The text is also normalized to a standard format.
        3.  **Model Loading:** Pre-trained machine learning models (a sentiment classifier and an emotion classifier) and their corresponding vectorizers are loaded. These models are cached to optimize performance and prevent re-loading on every user interaction.
        4.  **Prediction:** The preprocessed text is transformed using the vectorizers and then passed to the respective models to predict sentiment and emotion.
        5.  **Visualization:** The results are visualized using various charts, including bar charts, heatmaps, and geospatial maps (if location data is provided).

        #### Key Features
        * **Dynamic Data Upload:** Analyze your own dataset by uploading a CSV file.
        * **Real-time Prediction:** Get instant sentiment and emotion predictions on individual text inputs.
        * **Interactive Visualizations:** Explore sentiment and emotion distributions, and see how they correlate with each other.
        * **Geospatial Analysis:** If your data includes location information (ZIP codes, latitude, longitude), you can visualize sentiment on a map.

        #### Technologies Used
        * **Streamlit:** For creating the interactive web application.
        * **Pandas:** For data manipulation and analysis.
        * **Scikit-learn:** For the machine learning models.
        * **Plotly & Matplotlib:** For creating the visualizations.
        * **GeoPandas:** For handling geospatial data and creating the maps.

        This project demonstrates the power of natural language processing (NLP) and data visualization in understanding and extracting insights from text data.
    """)

if __name__ == '__main__':
    st.set_page_config(
        page_title="Twitter Sentiment Analysis",
        page_icon="🐦",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    sentiment_model, sentiment_vectorizer, emotion_model, emotion_vectorizer = load_all_models_cached()
    
    if sentiment_model is None:
        st.error("Failed to load machine learning models. Please check your internet connection or the provided file IDs.")
        st.stop()
        
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to:", ["📄 Project Report", "📊 Sentiment Analysis Dashboard", "🔍 Combined Prediction"], key='page_selector')

    if st.session_state.get('last_page') != page:
        st.session_state.clear()
        st.session_state.last_page = page
        
    if page == "📊 Sentiment Analysis Dashboard":
        main_dashboard()
    elif page == "🔍 Combined Prediction":
        combined_prediction_page()
    elif page == "📄 Project Report":
        project_report_page()
