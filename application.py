import streamlit as st
import pandas as pd
import polars as pl # ADDITION: Importing Polars for memory-efficient data handling
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
import os # ADDITION: Importing os module
from functools import lru_cache
import csv

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

# MODIFICATION: Use a persistent directory for files to avoid re-downloading on every run in some environments
DATA_DIR = Path(tempfile.gettempdir()) / "app_data"
SHAPE_DIR = DATA_DIR / "shapefiles"
SHAPEFILE_PATH = SHAPE_DIR / "tl_2020_us_zcta510.shp"

# Create directories if they don't exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
SHAPE_DIR.mkdir(parents=True, exist_ok=True)

# MODIFICATION: Using dictionary to store file info for easier management
DATASET_FILES = {
    "COVID-19": {"id": "1KhQvyglx07Lx4hD971956IikHhZKcczS", "name": "sample_twitter_data_covid_classified.csv"},
    "Economics": {"id": "1atpQuelcBriYSINjSjP5d4_AK4EmpV5J", "name": "sample_twitter_data_economics_classified.csv"},
    "Politics": {"id": "18Q9ORlDfoIQW_-RpwJy2qGrFWEA8dFQU", "name": "sample_twitter_data_politics_classified.csv"}
}

INCIDENT_FILES = {
    "COVID-19": {"id": "1IbIfdrAU3ZYue5joLojPisRX3JJjwdvM", "name": "Incident Zip_covid_classified.csv"},
    "Economics": {"id": "1SNpGyEHgrOe6ihx26vo38hf_zcxEnp64", "name": "Incident Zip_economics_classified.csv"},
    "Politics": {"id": "1uNxIYzSY7cbgbuTc8zo0QYc5Dn5Ny2W_", "name": "Incident Zip_politics_classified.csv"}
}

MODEL_FILES = {
    "sentiment_model": {"id": "1lzZf79LGcB1J5SQsMh_mi1Jv_8V2q6K9", "name": "sentiment_model_large.pkl"},
    "sentiment_vectorizer": {"id": "12wRG57vERpdKgCaTiNyLiWC71KLWABK8", "name": "vectorizer_large.pkl"},
    "emotion_model": {"id": "1NZCZKMhTSKvFJMuW_kxmX1OrkEQLFgs0", "name": "emotion_model_large.pkl"},
    "emotion_vectorizer": {"id": "1TKR2xmNcouAb8XyQz6VANixFLNZ9YsJV", "name": "emotion_vectorizer_large.pkl"}
}

SHAPE_FILES = [
    {"id": "1AwweXBE1Xq8_byFKfT61AaFjhNWlaftI", "name": "tl_2020_us_zcta510.shp"},
    {"id": "1RBysVHjdW4bIrL5WzhmOUKNhi7xUuET3", "name": "tl_2020_us_zcta510.shx"},
    {"id": "1YJLqeYapj9kpoLwf0KxFWG1dwRzchoq5", "name": "tl_2020_us_zcta510.dbf"},
    {"id": "1BzAH0_f-jgglD5tt0DImR2iOtdqfEhWf", "name": "tl_2020_us_zcta510.prj"},
    {"id": "1Un6VtqVE45qhanvqqwdywIiZV7K3qVi6", "name": "tl_2020_us_zcta510.cpg"},
    {"id": "1UkQSMun9auyZqHd9cy6NPxa387At79Rp", "name": "tl_2020_us_zcta510.shp.xml"},
    {"id": "19vKnJ0RH5wr8GZFeuNbJSpaABo5x3ukJ", "name": "tl_2020_us_zcta510.shp.iso.xml"}
]

# -------------------- HELPER FUNCTIONS --------------------
# MODIFICATION: Added a robust download function with error handling
def download_from_drive(file_id, output_path: Path):
    """Downloads a file from Google Drive if it doesn't exist."""
    if not output_path.exists() or output_path.stat().st_size == 0:
        with st.spinner(f"Downloading {output_path.name}..."):
            try:
                gdown.download(id=file_id, output=str(output_path), quiet=False, fuzzy=True)
                if output_path.exists() and output_path.stat().st_size > 0:
                    logger.info(f"Successfully downloaded {output_path.name}")
                else:
                    raise ValueError(f"Download failed: {output_path.name} is empty or missing.")
            except Exception as e:
                logger.error(f"Failed to download {output_path.name}: {e}")
                st.error(f"Failed to download required file: {output_path.name}. Please check the provided file ID and your internet connection.")
                st.stop()
    else:
        logger.info(f"File {output_path.name} already exists. Skipping download.")

# MODIFICATION: Added cache to prevent re-loading of models on every interaction
@st.cache_resource
def load_all_models_cached():
    """Loads all models and vectorizers from Google Drive into memory."""
    try:
        paths = {}
        with st.spinner("Loading ML models..."):
            for key, info in MODEL_FILES.items():
                path = DATA_DIR / info["name"]
                download_from_drive(info["id"], path)
                paths[key] = path
            
            sentiment_model = joblib.load(paths["sentiment_model"])
            sentiment_vectorizer = joblib.load(paths["sentiment_vectorizer"])
            emotion_model = joblib.load(paths["emotion_model"])
            emotion_vectorizer = joblib.load(paths["emotion_vectorizer"])

            logger.info("Successfully loaded all ML resources.")
        return sentiment_model, sentiment_vectorizer, emotion_model, emotion_vectorizer
    except Exception as e:
        logger.error(f"Failed to load ML resources: {e}")
        st.error("Failed to load machine learning models. Please check your internet connection or the provided file IDs.")
        st.stop()


# Emoticon map for tweet cleaning (YOUR ORIGINAL CODE)
EMOTICON_MAP = {
    r":-?\)+": "smiling_face", r"=+\)": "smiling_face", r":-?D+": "laughing_face", r"x+D+": "laughing_face",
    r"\^_+\^": "happy_face", r"LOL+": "laughing", r":'-?D+": "tearful_laughter", r":-?\(+": "sad_face",
    r":'\(+": "crying_face", r"T_T+": "crying_face", r"TT_TT": "crying_face", r"\(σ_σ\)": "sad_face",
    r":-?/+": "confused_face", r":-?\\+": "confused_face", r"-_+-": "annoyed_face", r"o_O+": "confused_face",
    r"<3+": "red_heart", r":3": "cute_face", r"UwU": "cute_face", r"\(♥_♥\)": "heart_eyes", r":-?O+": "shocked_face",
    r"O_O+": "shocked_face", r":\|+": "neutral_face", r"\.{3,}": "thinking", r"\(・_・\)": "thinking_face",
    r"\?{2,}": "confused", r"!{2,}": "excited_or_angry", r"\(◕‿◕\)": "happy_face", r"\(ಠ_ಠ\)": "disapproval_face",
    r"\(¬_¬\)": "unimpressed_face", r"¯\\\(ツ\)/¯": "shrug", r":-?\*+": "kiss", r";-?\)+": "winking_face",
    r":-?[Pp]+": "tongue_out_face", r"><+:": "embarrassed_face", r"8-?\)+": "cool_face", r"\+": "star_struck_face",
    r"\+1": "thumbs_up", r"-1": "thumbs_down", r"haha+": "laughing", r"hehe+": "giggling",
}

class MemoryOptimizedTweetCleaner:
    def __init__(self):
        self._compile_patterns()
        self.stats = {'processed': 0, 'cleaned': 0, 'dropped': 0, 'errors': 0}

    @lru_cache(maxsize=128)
    def _compile_patterns(self):
        self.patterns = {
            'emoticons': [(re.compile(p, re.IGNORECASE), desc) for p, desc in EMOTICON_MAP.items()],
            'web_urls': re.compile(r'https?://\S+|www\.\S+', re.IGNORECASE),
            'url_domains': re.compile(r'\b[\w\.-]+\.(?:com|net|org|io|co|br|uk|de|info|gov|edu|ly|me|tv)\b', re.IGNORECASE),
            'mentions': re.compile(r'@\w+'),
            'hashtags': re.compile(r'#(\w+)'),
            'rt_prefix': re.compile(r'^RT\s*', re.IGNORECASE),
            'rt_keywords': re.compile(r'\b(?:RT|FAV|MT|VIA)\b', re.IGNORECASE),
            'digits': re.compile(r'\b\d+\b'),
            'single_chars': re.compile(r'\b[A-Za-z]\b(?:\s+\b[A-Za-z]\b){2,}'),
            'punctuation': re.compile(r'[^\w\s\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]+'),
            'newlines': re.compile(r'[\n\r\t]+'),
            'whitespace': re.compile(r'\s+'),
            'extra_spaces': re.compile(r'\s{2,}')
        }
        logger.info("REGEX: Regex patterns compiled successfully")

    def translate_emoticons(self, text: str) -> str:
        for regex, desc in self.patterns['emoticons']:
            text = regex.sub(f" {desc} ", text)
        return text

    def clean_text(self, text: str) -> Optional[str]:
        if pd.isna(text) or not text or not str(text).strip():
            return None
        try:
            text = str(text)
            text = self.patterns['newlines'].sub(' ', text)
            text = self.patterns['web_urls'].sub('', text)
            text = self.patterns['rt_prefix'].sub('', text)
            text = self.patterns['mentions'].sub('', text)
            text = self.patterns['hashtags'].sub(r'\1', text)
            text = self.patterns['rt_keywords'].sub('', text)
            text = self.patterns['url_domains'].sub('', text)
            text = self.patterns['digits'].sub('', text)
            text = self.patterns['single_chars'].sub('', text)
            text = self.patterns['punctuation'].sub(' ', text)
            text = self.translate_emoticons(text)
            text = self.patterns['whitespace'].sub(' ', text).strip()
            return text if text and len(text) > 1 else None
        except Exception as e:
            logger.debug(f"Error cleaning text: {e}")
            return None

class MemoryMonitor:
    @staticmethod
    def get_memory_usage():
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024

    @staticmethod
    def force_garbage_collection():
        gc.collect()

# MODIFICATION: Refactored load_data to be more memory-efficient using Polars
@st.cache_data
def load_data(dataset_key):
    file_info = DATASET_FILES[dataset_key]
    path = DATA_DIR / file_info["name"]
    download_from_drive(file_info["id"], path)
    
    # Use Polars for memory-efficient lazy loading
    df_pl = pl.scan_csv(path, try_parse_dates=True, infer_schema_length=10000)
    
    # Filter and process data using Polars' lazy evaluation
    df_pl = df_pl.with_columns(
        pl.all().exclude(["latitude", "longitude"]).str.strip().str.to_lowercase().str.replace_all(" ", "_")
    ).with_columns(
        pl.col("latitude").filter(pl.col("latitude").is_between(-90, 90)),
        pl.col("longitude").filter(pl.col("longitude").is_between(-180, 180))
    )
    
    # Convert to Pandas for compatibility with other libraries
    df = df_pl.collect().to_pandas()
    
    df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]
    df = df.dropna(subset=['latitude', 'longitude'])
    df = df[(df['latitude'].between(-90, 90)) & (df['longitude'].between(-180, 180))]
    
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
    return df

# MODIFICATION: Refactored load_incident_data to be more memory-efficient using Polars
@st.cache_data
def load_incident_data(incident_key):
    file_info = INCIDENT_FILES[incident_key]
    path = DATA_DIR / file_info["name"]
    download_from_drive(file_info["id"], path)
    
    # Use Polars for memory-efficient loading
    incident_pl = pl.scan_csv(path, infer_schema_length=10000)
    
    incident_pl = incident_pl.with_columns(
        pl.col("Incident Zip").cast(pl.Utf8).str.zfill(5)
    )
    
    incident_df = incident_pl.collect().to_pandas()
    
    return incident_df

# MODIFICATION: Refactored load_shapefile to be more memory-efficient
@st.cache_data
def load_shapefile():
    for sf in SHAPE_FILES:
        path = SHAPE_DIR / sf["name"]
        download_from_drive(sf["id"], path)
    try:
        zcta_gdf = gpd.read_file(SHAPEFILE_PATH)
        zcta_gdf['ZCTA5CE10'] = zcta_gdf['ZCTA5CE10'].astype(str).str.zfill(5)
        nyc_zip_prefixes = ('100', '101', '102', '103', '104', '111', '112', '113', '114', '116')
        nyc_gdf = zcta_gdf[zcta_gdf['ZCTA5CE10'].str.startswith(nyc_zip_prefixes)].copy()
        del zcta_gdf
        gc.collect()
        return nyc_gdf
    except Exception as e:
        logger.error(f"Failed to load geographic data: {e}")
        st.error("Failed to load geographic data. Please check shapefile availability.")
        st.stop()
    
# =========================
# NEW VISUALIZATION FUNCTIONS (YOUR ORIGINAL CODE)
# =========================
plt.style.use('seaborn-v0_8')
sns.set_context('talk', font_scale=1.2)
top_n = 5

def top_emotions_pie_chart(df):
    st.subheader("Top 5 Emotions in Tweets")
    if 'emotion' not in df.columns or df['emotion'].isnull().all():
        st.warning("Emotion data not available for this dataset.")
        return
    value_counts = df['emotion'].value_counts().head(top_n)
    labels = value_counts.index.tolist()
    sizes = value_counts.values.tolist()
    colors = sns.color_palette("tab10", len(labels))
    explode = [0.1 if i == 0 else 0 for i in range(len(labels))]
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=90, textprops={'fontsize': 14})
    ax.set_title('Top 5 Emotions in Tweets', pad=20, fontsize=18)
    ax.axis('equal')
    st.pyplot(fig)
    plt.close(fig)
    st.markdown(f"**Explanation:** This chart visualizes the distribution of the top 5 most frequently detected emotions in the selected dataset. The largest slice represents the most common emotion, providing a quick overview of the overall emotional tone of the conversations.")

def emotion_sentiment_bar_chart(df):
    st.subheader("Emotion Distribution by Sentiment Category")
    if 'emotion' not in df.columns or df['emotion'].isnull().all():
        st.warning("Emotion data not available for this dataset.")
        return
    top_emotions = df['emotion'].value_counts().head(top_n).index
    emotion_sentiment = df[df['emotion'].isin(top_emotions)] \
        .groupby(['emotion', 'category']).size().unstack(fill_value=0) \
        .rename(columns={-1: 'Negative', 0: 'Neutral', 1: 'Positive'})
    fig, ax = plt.subplots(figsize=(12, 7))
    emotion_sentiment.plot(kind='bar', stacked=True, color=sns.color_palette("tab10", 3), ax=ax)
    ax.set_title('Emotion Distribution by Sentiment Category', fontsize=18)
    ax.set_xlabel('Emotion', fontsize=14)
    ax.set_ylabel('Number of Tweets', fontsize=14)
    ax.legend(title='Sentiment', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig)
    plt.close(fig)
    st.markdown("**Explanation:** This chart breaks down each of the top emotions by sentiment category (Positive, Neutral, Negative). It helps you understand if a particular emotion, like 'joy,' is consistently associated with positive sentiment.")

def geo_sentiment_map(df, geo_df):
    st.subheader("Geospatial Distribution of Sentiment")
    
    # MODIFICATION: Aggregate data to avoid plotting too many points
    if len(df) > MAX_POINTS_MAP:
        df_display = df.sample(n=MAX_POINTS_MAP, random_state=42)
        st.info(f"Displaying a sample of {MAX_POINTS_MAP:,} points for performance.")
    else:
        df_display = df.copy()

    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(
        df_display, geometry=gpd.points_from_xy(df_display.longitude, df_display.latitude), crs="EPSG:4326"
    )

    fig = px.scatter_mapbox(
        gdf,
        lat="latitude",
        lon="longitude",
        color="category",
        hover_data=["text", "category"],
        color_discrete_map={-1: "red", 0: "yellow", 1: "green"},
        title="Sentiment Map of Tweets",
        zoom=9,
        height=600
    )
    fig.update_layout(mapbox_style="carto-positron")
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("**Explanation:** This map visualizes the sentiment of tweets across NYC. Each point represents a tweet, colored by its sentiment: Green for Positive, Yellow for Neutral, and Red for Negative. You can zoom in and hover over points to see details.")

def zip_code_maps(incident_df, nyc_gdf):
    st.subheader("Tweet Volume by ZIP Code")
    zip_counts = incident_df['Incident Zip'].value_counts().reset_index()
    zip_counts.columns = ['ZCTA5CE10', 'tweet_count']
    
    zip_geo = nyc_gdf.merge(zip_counts, left_on='ZCTA5CE10', right_on='ZCTA5CE10', how='left')
    zip_geo['tweet_count'] = zip_geo['tweet_count'].fillna(0)

    fig = px.choropleth_mapbox(
        zip_geo,
        geojson=zip_geo.geometry,
        locations=zip_geo.index,
        color="tweet_count",
        color_continuous_scale="Viridis",
        mapbox_style="carto-positron",
        zoom=9,
        center={"lat": 40.7128, "lon": -74.0060},
        opacity=0.7,
        height=600,
        hover_name="ZCTA5CE10",
        hover_data={"tweet_count": True}
    )
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("**Explanation:** This map shows the volume of tweets aggregated by ZIP code. The color intensity represents the number of tweets in each area, helping to identify regions with higher public activity.")

def zip_code_heatmap(incident_df, nyc_gdf):
    st.subheader("Sentiment Heatmap by ZIP Code")
    zip_sentiment = incident_df.groupby('Incident Zip')['category'].mean().reset_index()
    zip_sentiment.columns = ['ZCTA5CE10', 'avg_sentiment']

    zip_geo = nyc_gdf.merge(zip_sentiment, left_on='ZCTA5CE10', right_on='ZCTA5CE10', how='left')
    zip_geo['avg_sentiment'] = zip_geo['avg_sentiment'].fillna(0)
    zip_geo['avg_sentiment_display'] = zip_geo['avg_sentiment'].apply(lambda x: 'Positive' if x > 0 else ('Negative' if x < 0 else 'Neutral'))

    fig = px.choropleth_mapbox(
        zip_geo,
        geojson=zip_geo.geometry,
        locations=zip_geo.index,
        color="avg_sentiment",
        color_continuous_scale=px.colors.sequential.RdYlGn,
        mapbox_style="carto-positron",
        zoom=9,
        center={"lat": 40.7128, "lon": -74.0060},
        opacity=0.7,
        height=600,
        hover_name="ZCTA5CE10",
        hover_data={"avg_sentiment": ':.2f'}
    )
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("**Explanation:** This map shows the average sentiment score for each ZIP code. Green areas indicate a more positive average sentiment, while red areas indicate a more negative average sentiment. Neutral sentiment is shown in yellow.")

def borough_income_chart(df):
    st.subheader("Average Median Income by Borough")
    # This function is a placeholder and requires actual income data to be merged.
    # The original code did not provide this data source.
    st.warning("Income data is not available in the provided datasets. This chart is a placeholder.")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(['Manhattan', 'Brooklyn', 'Bronx', 'Queens', 'Staten Island'], [75000, 65000, 45000, 60000, 70000], color='skyblue')
    ax.set_title('Placeholder: Average Median Income by Borough', fontsize=18)
    ax.set_ylabel('Median Income ($)', fontsize=14)
    ax.set_xlabel('Borough', fontsize=14)
    st.pyplot(fig)
    plt.close(fig)


def combined_prediction_page(sentiment_model, sentiment_vectorizer, emotion_model, emotion_vectorizer):
    st.header("Analyze a New Tweet")
    st.markdown("Enter a tweet below to classify its topic, sentiment, and emotion.")
    
    user_input = st.text_area("Enter your tweet here:", max_chars=280)

    if st.button("Analyze Tweet"):
        if user_input:
            cleaner = MemoryOptimizedTweetCleaner()
            cleaned_text = cleaner.clean_text(user_input)

            if cleaned_text:
                st.subheader("Analysis Results:")
                
                # Predict Sentiment
                sentiment_vector = sentiment_vectorizer.transform([cleaned_text])
                sentiment_pred = sentiment_model.predict(sentiment_vector)[0]
                sentiment_label = "Positive" if sentiment_pred == 1 else "Negative" if sentiment_pred == -1 else "Neutral"
                st.metric(label="Sentiment", value=sentiment_label, delta=f"Score: {sentiment_pred}")

                # Predict Emotion
                emotion_vector = emotion_vectorizer.transform([cleaned_text])
                emotion_pred = emotion_model.predict(emotion_vector)[0]
                st.metric(label="Emotion", value=emotion_pred)

                # Predict Topic (Placeholder, as model for this was not provided)
                st.info("Topic classification model is not available in the provided code. A placeholder topic is shown.")
                st.metric(label="Topic", value="Placeholder Topic")
            else:
                st.warning("The entered text could not be cleaned and analyzed.")
        else:
            st.warning("Please enter some text to analyze.")


# =========================
# STREAMLIT APP LAYOUT (YOUR ORIGINAL CODE, MODIFIED FOR SESSION STATE)
# =========================

def main_app():
    # ADDITION: Initialize session state for caching
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'incident_df' not in st.session_state:
        st.session_state.incident_df = None
    if 'nyc_gdf' not in st.session_state:
        st.session_state.nyc_gdf = None

    st.set_page_config(layout="wide", page_title="NYC Tweet Analysis")

    st.title("NYC Tweet Classification and Sentiment Analysis")
    st.sidebar.title("Navigation")
    
    tab1, tab2 = st.tabs(["Dashboard", "Tweet Analysis"])
    
    with tab1:
        st.header("NYC Tweet Data Dashboard")
        
        col1, col2 = st.columns([1, 2])

        with col1:
            dataset_choice = st.selectbox(
                "Select a Dataset",
                list(DATASET_FILES.keys()),
                key="dataset_selector"
            )

            # MODIFICATION: Button to trigger data loading and clear session state
            if st.button("Load Data"):
                st.session_state.data_loaded = False
                st.session_state.df = None
                st.session_state.incident_df = None
                st.session_state.nyc_gdf = None

                with st.spinner("Loading data... This may take a moment."):
                    try:
                        st.session_state.df = load_data(dataset_choice)
                        st.session_state.incident_df = load_incident_data(dataset_choice)
                        st.session_state.nyc_gdf = load_shapefile()
                        st.session_state.data_loaded = True
                    except Exception as e:
                        st.error(f"An error occurred during data loading: {e}")
                        st.session_state.data_loaded = False
                        st.stop()
        
        with col2:
            st.info("Click 'Load Data' to begin the analysis. Loading all data and models can take a few minutes, please be patient.")
        
        # MODIFICATION: Only show dashboard content after data is loaded
        if st.session_state.data_loaded:
            tab_overview, tab_geo, tab_summary = st.tabs(["Overview", "Geospatial Analysis", "Summary"])
            
            with tab_overview:
                st.header("Tweet Overview")
                
                # Check for empty dataframes
                if st.session_state.df.empty:
                    st.warning("The selected dataset is empty.")
                else:
                    col_metrics_1, col_metrics_2, col_metrics_3 = st.columns(3)
                    
                    # Ensure sentiment and emotion categories are available
                    sentiment_counts = st.session_state.df['category'].value_counts()
                    positive_count = sentiment_counts.get(1, 0)
                    negative_count = sentiment_counts.get(-1, 0)
                    
                    with col_metrics_1:
                        st.metric("Total Tweets", f"{len(st.session_state.df):,}")
                    with col_metrics_2:
                        st.metric("Positive Tweets", f"{positive_count:,}")
                    with col_metrics_3:
                        st.metric("Negative Tweets", f"{negative_count:,}")
                        
                    st.markdown("---")
                    
                    # Displaying visualizations
                    top_emotions_pie_chart(st.session_state.df)
                    emotion_sentiment_bar_chart(st.session_state.df)
            
            with tab_geo:
                st.header("Geospatial Analysis")
                if st.session_state.df.empty or st.session_state.incident_df.empty or st.session_state.nyc_gdf.empty:
                    st.warning("Geospatial data is not loaded.")
                else:
                    geo_map_choice = st.selectbox(
                        "Select a Geographic Map",
                        ["Geospatial Distribution of Sentiment", "ZIP Code Sentiment Heatmap", "Tweet Volume by ZIP Code"],
                        key="geo_map_selector"
                    )
                    
                    if geo_map_choice == "Geospatial Distribution of Sentiment":
                        geo_sentiment_map(st.session_state.df, st.session_state.nyc_gdf)
                    elif geo_map_choice == "ZIP Code Sentiment Heatmap":
                        zip_code_heatmap(st.session_state.incident_df, st.session_state.nyc_gdf)
                    elif geo_map_choice == "Tweet Volume by ZIP Code":
                        zip_code_maps(st.session_state.incident_df, st.session_state.nyc_gdf)
            
            with tab_summary:
                st.header("Dataset Summary")
                st.markdown("This section provides a quick look at the raw data and its structure.")
                st.subheader("Data at a glance:")
                st.write(st.session_state.df.head())
                st.subheader("Dataset Shape:")
                st.write(f"Rows: {st.session_state.df.shape[0]:,}")
                st.write(f"Columns: {st.session_state.df.shape[1]}")
                st.subheader("DataFrame Info:")
                buffer = io.StringIO()
                st.session_state.df.info(buf=buffer)
                s = buffer.getvalue()
                st.text(s)

    with tab2:
        sentiment_model, sentiment_vectorizer, emotion_model, emotion_vectorizer = load_all_models_cached()
        combined_prediction_page(sentiment_model, sentiment_vectorizer, emotion_model, emotion_vectorizer)
        
if __name__ == '__main__':
    main_app()
