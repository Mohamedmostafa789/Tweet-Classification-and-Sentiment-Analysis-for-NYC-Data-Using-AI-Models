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

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler('app_log.log', mode='w', encoding='utf-8')]
)
logger = logging.getLogger(__name__)

# -------------------- CONFIG --------------------
MAX_POINTS_MAP = 5000
MAX_POINTS_SCATTER = 20000

DATA_DIR = Path(tempfile.gettempdir()) / "app_data"
SHAPE_DIR = DATA_DIR / "shapefiles"
SHAPEFILE_PATH = SHAPE_DIR / "tl_2020_us_zcta510.shp"

DATA_DIR.mkdir(parents=True, exist_ok=True)
SHAPE_DIR.mkdir(parents=True, exist_ok=True)

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
def download_from_drive(file_id, output_path: Path):
    """Downloads a file from Google Drive if it doesn't exist."""
    if not output_path.exists() or output_path.stat().st_size == 0:
        try:
            gdown.download(id=file_id, output=str(output_path), quiet=False)
            if output_path.exists() and output_path.stat().st_size > 0:
                logger.info(f"Successfully downloaded {output_path.name}")
            else:
                raise ValueError(f"Download failed: {output_path.name} is empty or missing.")
        except Exception as e:
            logger.error(f"Failed to download {output_path.name}: {e}")
            raise RuntimeError(f"Download failed for {output_path.name}") from e

@st.cache_resource
def load_all_models_cached():
    """Loads all models and vectorizers from Google Drive into memory."""
    try:
        paths = {}
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
        return None, None, None, None

# Emoticon map
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
        self.lock = threading.Lock()

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

    def update_stats(self, processed: int, cleaned: int, dropped: int, errors: int = 0):
        with self.lock:
            self.stats['processed'] += processed
            self.stats['cleaned'] += cleaned
            self.stats['dropped'] += dropped
            self.stats['errors'] += errors

class MemoryMonitor:
    @staticmethod
    def get_memory_usage():
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024

    @staticmethod
    def force_garbage_collection():
        gc.collect()

@st.cache_data(max_entries=1)
def load_data(dataset_key: str) -> pd.DataFrame:
    if MemoryMonitor.get_memory_usage() > 400:
        logger.warning("High memory usage before loading dataset. Clearing caches.")
        st.cache_data.clear()
        MemoryMonitor.force_garbage_collection()
    
    file_info = DATASET_FILES[dataset_key]
    path = DATA_DIR / file_info["name"]
    download_from_drive(file_info["id"], path)
    
    try:
        # Check available columns
        temp_df = pd.read_csv(path, nrows=1)
        available_columns = temp_df.columns.tolist()
        expected_columns = ['latitude', 'longitude', 'date', 'emotion', 'category', 'cleaned_tweet', 'hashtags', 'median_income']
        missing_columns = [col for col in expected_columns if col not in available_columns]
        
        if missing_columns:
            logger.warning(f"Missing columns in '{dataset_key}': {missing_columns}")
            st.warning(f"Dataset '{dataset_key}' is missing columns: {missing_columns}. Loading available columns.")
        
        # Load data with available columns
        df = pd.read_csv(path, low_memory=False)
        df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]
        
        # Apply filtering only if columns are present
        if 'latitude' in df.columns and 'longitude' in df.columns:
            df = df.dropna(subset=['latitude', 'longitude'])
            df = df[(df['latitude'].between(-90, 90)) & (df['longitude'].between(-180, 180))]
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        if df.empty:
            logger.warning(f"No valid data loaded for '{dataset_key}' after filtering")
            st.error(f"Failed to load dataset '{dataset_key}': No valid data after filtering. Available columns: {available_columns}")
            return pd.DataFrame()
        
        logger.info(f"Loaded dataset '{dataset_key}' with {len(df)} rows")
        MemoryMonitor.force_garbage_collection()
        return df
    
    except Exception as e:
        logger.error(f"Error loading dataset '{dataset_key}': {e}")
        st.error(f"Failed to load dataset '{dataset_key}': {e}")
        return pd.DataFrame()

@st.cache_data(max_entries=1)
def load_incident_data(incident_key: str) -> pd.DataFrame:
    if MemoryMonitor.get_memory_usage() > 400:
        logger.warning("High memory usage before loading incident data. Clearing caches.")
        st.cache_data.clear()
        MemoryMonitor.force_garbage_collection()
    
    file_info = INCIDENT_FILES[incident_key]
    path = DATA_DIR / file_info["name"]
    download_from_drive(file_info["id"], path)
    
    try:
        incident_df = pd.read_csv(path, low_memory=False)
        if 'Incident Zip' in incident_df.columns:
            incident_df['Incident Zip'] = incident_df['Incident Zip'].astype(str).str.zfill(5)
        else:
            logger.warning(f"No 'Incident Zip' column in incident data '{incident_key}'")
            st.warning(f"Incident dataset '{incident_key}' is missing 'Incident Zip' column.")
        
        logger.info(f"Loaded incident data '{incident_key}' with {len(incident_df)} rows")
        MemoryMonitor.force_garbage_collection()
        return incident_df
    
    except Exception as e:
        logger.error(f"Error loading incident data '{incident_key}': {e}")
        st.error(f"Failed to load incident data '{incident_key}': {e}")
        return pd.DataFrame()

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

# -------------------- VISUALIZATION FUNCTIONS --------------------
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
    ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
           shadow=True, startangle=90, textprops={'fontsize': 14})
    ax.set_title('Top 5 Emotions in Tweets', pad=20, fontsize=18)
    ax.axis('equal')
    st.pyplot(fig)
    plt.close(fig)
    st.markdown(f"Summary: The most frequent emotion is {labels[0]} with {sizes[0]} tweets "
                f"({sizes[0]/sum(sizes)*100:.1f}%). Top emotions together account for {sum(sizes)} tweets.")

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
    st.markdown("Summary: This chart shows how each of the top emotions is distributed across "
                "positive, neutral, and negative sentiments.")

def emotion_confidence_boxplot(df):
    st.subheader("Emotion Confidence Distribution by Top 5 Emotions")
    if 'emotion' not in df.columns or 'emotion_confidence' not in df.columns or df['emotion'].isnull().all():
        st.warning("Emotion or Emotion Confidence data not available for this dataset.")
        return
    top_emotions = df['emotion'].value_counts().head(top_n).index
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.boxplot(x='emotion', y='emotion_confidence',
                data=df[df['emotion'].isin(top_emotions)], palette='tab10', ax=ax)
    ax.set_title('Emotion Confidence Distribution by Top 5 Emotions', fontsize=18)
    ax.set_xlabel('Emotion', fontsize=14)
    ax.set_ylabel('Emotion Confidence', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig)
    plt.close(fig)
    st.markdown("Summary: Higher confidence scores indicate stronger certainty in emotion classification. "
                "This plot compares confidence levels for the top emotions.")

def geo_sentiment_scatterplot(df):
    st.subheader("Geographical Distribution of Tweet Sentiments")
    if 'latitude' not in df.columns or 'longitude' not in df.columns:
        st.warning("Latitude and Longitude data not available for this visualization.")
        return
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.scatterplot(
        x='longitude', y='latitude',
        hue='category', palette={1: '#ff9999', 0: '#66b3ff', -1: '#99ff99'},
        data=df, alpha=0.6, ax=ax
    )
    ax.set_title('Geographical Distribution of Tweet Sentiments', fontsize=18)
    st.pyplot(fig)
    plt.close(fig)
    st.markdown("Summary: Each point represents a tweet's location. Colors indicate sentiment category.")

def median_income_histogram(df):
    st.subheader("Distribution of Median Income in Tweet Locations")
    if 'median_income' not in df.columns:
        st.warning("Median Income data not available for this visualization.")
        return
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.histplot(df['median_income'].dropna(), bins=50, kde=True, color='purple', ax=ax)
    ax.set_title('Distribution of Median Income in Tweet Locations', fontsize=18)
    st.pyplot(fig)
    plt.close(fig)
    st.markdown("Summary: This distribution shows the range of median incomes for locations mentioned in tweets.")

def sentiment_trends_line_chart(df):
    st.subheader("Sentiment Trends Over Time")
    if 'date' not in df.columns:
        st.warning("Date data not available for this visualization.")
        return
    sentiment_over_time = df.groupby([df['date'].dt.date, 'category']) \
                           .size().unstack(fill_value=0) \
                           .rename(columns={-1: 'Negative', 0: 'Neutral', 1: 'Positive'})
    fig, ax = plt.subplots(figsize=(14, 7))
    for col in sentiment_over_time.columns:
        ax.plot(sentiment_over_time.index, sentiment_over_time[col], label=col, linewidth=2)
    ax.legend(title='Sentiment', fontsize=12)
    ax.set_title('Sentiment Trends Over Time', fontsize=18)
    st.pyplot(fig)
    plt.close(fig)
    st.markdown("Summary: This line chart tracks how the frequency of each sentiment "
                "has changed over time in the dataset.")

def emotion_sentiment_heatmap(df):
    st.subheader("Emotion vs Sentiment Correlation Heatmap")
    if 'emotion' not in df.columns or df['emotion'].isnull().all():
        st.warning("Emotion data not available for this visualization.")
        return
    top_emotions = df['emotion'].value_counts().head(top_n).index
    emotion_sentiment_counts = df[df['emotion'].isin(top_emotions)] \
        .groupby(['emotion', 'category']).size().unstack(fill_value=0) \
        .rename(columns={-1: 'Negative', 0: 'Neutral', 1: 'Positive'})
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(emotion_sentiment_counts, annot=True, fmt='d', cmap='YlGnBu', ax=ax)
    ax.set_title('Emotion vs Sentiment Correlation', fontsize=18)
    st.pyplot(fig)
    plt.close(fig)
    st.markdown("Summary: This heatmap shows the relationship between emotions and sentiment categories, "
                "with cell values representing tweet counts.")

def sentiment_pie_chart(df):
    st.subheader("Sentiment Distribution")
    if 'category' not in df.columns or df['category'].isnull().all():
        st.warning("Sentiment data not available for this dataset.")
        return
    value_counts = df['category'].value_counts()
    labels = ['Positive', 'Neutral', 'Negative']
    sizes = [value_counts.get(1, 0), value_counts.get(0, 0), value_counts.get(-1, 0)]
    if sum(sizes) == 0:
        st.warning("No valid sentiment data available for visualization.")
        return
    fig, ax = plt.subplots(figsize=(6, 6))
    colors = ['#4CAF50', '#2196F3', '#F44336']
    explode = (0.1, 0, 0)
    ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=90)
    ax.axis('equal')
    st.pyplot(fig)
    plt.close(fig)
    st.markdown("**Explanation:** This simple pie chart provides a clear and direct summary...")

def emotion_pie_chart(df):
    st.subheader("Emotion Distribution")
    if 'emotion' not in df.columns or df['emotion'].isnull().all():
        st.warning("Emotion data not available for this dataset.")
        return
    value_counts = df['emotion'].value_counts()
    labels = value_counts.index.tolist()
    sizes = value_counts.values.tolist()
    if sum(sizes) == 0:
        st.warning("No valid emotion data available for visualization.")
        return
    fig, ax = plt.subplots(figsize=(6, 6))
    colors = sns.color_palette("husl", len(labels))
    explode = [0.1] + [0] * (len(labels) - 1)
    ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=90)
    ax.axis('equal')
    st.pyplot(fig)
    plt.close(fig)

def sentiment_map(df):
    if 'latitude' not in df.columns or 'longitude' not in df.columns:
        st.warning("Latitude and Longitude data not available for this visualization.")
        return
    nyc_bbox = {'min_lon': -74.27, 'max_lon': -73.68, 'min_lat': 40.48, 'max_lat': 40.95}
    nyc_df = df[(df['longitude'].between(nyc_bbox['min_lon'], nyc_bbox['max_lon'])) &
                (df['latitude'].between(nyc_bbox['min_lat'], nyc_bbox['max_lat']))]
    if nyc_df.empty:
        st.warning("No valid data within NYC bounds for this visualization.")
        return
    sentiment_map = {-1: "#e74c3c", 0: "#f39c12", 1: "#2ecc71"}
    for cat, color in sentiment_map.items():
        sub_df = nyc_df[nyc_df['category'] == cat].sample(min(2000, len(nyc_df[nyc_df['category'] == cat])), random_state=42)
        fig = px.scatter_mapbox(sub_df, lat="latitude", lon="longitude",
                                color_discrete_sequence=[color], zoom=10, height=400,
                                hover_data=["emotion"])
        fig.update_layout(mapbox_style="carto-positron", margin={"r":0,"t":0,"l":0,"b":0})
        st.plotly_chart(fig, use_container_width=True)

def emotion_map(df):
    if 'latitude' not in df.columns or 'longitude' not in df.columns:
        st.warning("Latitude and Longitude data not available for this visualization.")
        return
    nyc_bbox = {'min_lon': -74.27, 'max_lon': -73.68, 'min_lat': 40.48, 'max_lat': 40.95}
    nyc_df = df[(df['longitude'].between(nyc_bbox['min_lon'], nyc_bbox['max_lon'])) &
                (df['latitude'].between(nyc_bbox['min_lat'], nyc_bbox['max_lat']))]
    if nyc_df.empty:
        st.warning("No valid data within NYC bounds for this visualization.")
        return
    emotion_colors = {"joy": "#FFD700", "anger": "#FF4500", "sadness": "#1E90FF", "fear": "#9400D3"}
    for emo, color in emotion_colors.items():
        sub_df = nyc_df[nyc_df['emotion'] == emo].sample(min(2000, len(nyc_df[nyc_df['emotion'] == emo])), random_state=42)
        fig = px.scatter_mapbox(sub_df, lat="latitude", lon="longitude",
                                color_discrete_sequence=[color], zoom=10, height=400,
                                hover_data=["category"])
        fig.update_layout(mapbox_style="carto-positron", margin={"r":0,"t":0,"l":0,"b":0})
        st.plotly_chart(fig, use_container_width=True)

def zip_code_maps(incident_df, nyc_gdf):
    st.subheader("ZIP Code Sentiment Maps")
    if 'Incident Zip' not in incident_df.columns:
        st.warning("Incident Zip data not available for this visualization.")
        return
    incident_sums = incident_df.groupby('Incident Zip')[['negative', 'positive', 'neutral']].sum().reset_index()
    incident_sums['total'] = incident_sums[['negative', 'positive', 'neutral']].sum(axis=1)
    for col in ['negative', 'positive', 'neutral']:
        incident_sums[col + '_pct'] = (incident_sums[col] / incident_sums['total']).fillna(0) * 100
    merged_gdf = nyc_gdf.merge(incident_sums, left_on='ZCTA5CE10', right_on='Incident Zip', how='left')
    merged_gdf[['negative', 'positive', 'neutral',
                'negative_pct', 'positive_pct', 'neutral_pct']] = merged_gdf[
                    ['negative', 'positive', 'neutral',
                     'negative_pct', 'positive_pct', 'neutral_pct']].fillna(0)
    fig, axes = plt.subplots(2, 3, figsize=(20, 14))
    categories_counts = ['negative', 'positive', 'neutral']
    cmaps_counts = ['Reds', 'Greens', 'Blues']
    for ax, cat, cmap in zip(axes[0], categories_counts, cmaps_counts):
        merged_gdf.plot(column=cat, cmap=cmap, linewidth=0.6, ax=ax, edgecolor='0.8', legend=True,
                        legend_kwds={'label': "Count", 'orientation': "vertical"})
        ax.set_title(f"NYC {cat.capitalize()} Incidents (Count)", fontsize=14)
        ax.axis('off')
    categories_pct = ['negative_pct', 'positive_pct', 'neutral_pct']
    cmaps_pct = ['Reds', 'Greens', 'Blues']
    for ax, cat, cmap in zip(axes[1], categories_pct, cmaps_pct):
        merged_gdf.plot(column=cat, cmap=cmap, linewidth=0.6, ax=ax, edgecolor='0.8', legend=True,
                        legend_kwds={'label': "Percentage", 'orientation': "vertical"})
        ax.set_title(f"NYC {cat.replace('_pct','').capitalize()} Incidents (%)", fontsize=14)
        ax.axis('off')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.markdown("### Map Descriptions")
    st.markdown("- Negative Incidents (Count): This map shows the raw count of negative sentiment incidents across NYC ZIP codes. Darker shades indicate higher counts.")
    st.markdown("- Positive Incidents (Count): This map shows the raw count of positive sentiment incidents across NYC ZIP codes. Darker shades indicate higher counts.")
    st.markdown("- Neutral Incidents (Count): This map shows the raw count of neutral sentiment incidents across NYC ZIP codes. Darker shades indicate higher counts.")
    st.markdown("- Negative Incidents (%): This map shows the percentage of negative sentiment incidents relative to total incidents in each NYC ZIP code. Darker shades indicate higher percentages.")
    st.markdown("- Positive Incidents (%): This map shows the percentage of positive sentiment incidents relative to total incidents in each NYC ZIP code. Darker shades indicate higher percentages.")
    st.markdown("- Neutral Incidents (%): This map shows the percentage of neutral sentiment incidents relative to total incidents in each NYC ZIP code. Darker shades indicate higher percentages.")

def zip_code_heatmap(incident_df, nyc_gdf):
    st.subheader("ZIP Code Sentiment Heatmap")
    if 'Incident Zip' not in incident_df.columns:
        st.warning("Incident Zip data not available for this visualization.")
        return
    incident_sums = incident_df.groupby('Incident Zip')[['negative', 'positive', 'neutral']].sum().reset_index()
    incident_sums['total'] = incident_sums[['negative', 'positive', 'neutral']].sum(axis=1)
    incident_sums['combined_sentiment'] = (incident_sums['positive'] - incident_sums['negative']) / incident_sums['total'].replace(0, 1)
    merged_gdf = nyc_gdf.merge(incident_sums, left_on='ZCTA5CE10', right_on='Incident Zip', how='left')
    merged_gdf['combined_sentiment'] = merged_gdf['combined_sentiment'].fillna(0)
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    merged_gdf.plot(column='combined_sentiment', cmap='RdYlBu_r', linewidth=0.6, ax=ax, edgecolor='0.8', legend=True,
                    legend_kwds={'label': "Sentiment Score", 'orientation': "vertical"})
    ax.set_title("NYC Sentiment Heatmap 2020", fontsize=16)
    ax.axis('off')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.markdown("### Heatmap Description")
    st.markdown("- Sentiment Heatmap: This map shows a combined sentiment score across NYC ZIP codes, calculated as (Positive - Negative) / Total incidents. Red indicates more negative sentiment, blue indicates more positive sentiment, and yellow/green represents neutral areas.")

def borough_income_chart(df):
    st.subheader("Average Median Income by NYC Borough")
    if 'median_income' not in df.columns:
        st.warning("Median Income data not available for this visualization.")
        return
    zip_column = 'Incident Zip' if 'Incident Zip' in df.columns else 'incident_zip' if 'incident_zip' in df.columns else 'zip_code'
    borough_map = {
        "Manhattan": ['100', '101', '102'],
        "Bronx": ['104'],
        "Brooklyn": ['112'],
        "Queens": ['110', '111', '113', '114', '116'],
        "Staten Island": ['103']
    }
    def get_borough(zipcode):
        zipcode = str(zipcode)
        for borough, prefixes in borough_map.items():
            if any(zipcode.startswith(pref) for pref in prefixes):
                return borough
        return None
    df['borough'] = df[zip_column].apply(get_borough)
    borough_income = (
        df.dropna(subset=['median_income', 'borough'])
          .groupby('borough')['median_income']
          .mean()
          .sort_values(ascending=False)
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    borough_income.plot(kind='bar', color='skyblue', edgecolor='black', ax=ax)
    ax.set_title('Average Median Income by NYC Borough', fontsize=16)
    ax.set_ylabel('Median Income (USD)', fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    for i, value in enumerate(borough_income):
        ax.text(i, value + 500, f"${value:,.0f}", ha='center', va='bottom', fontsize=10)
    st.pyplot(fig)
    plt.close(fig)

# -------------------- COMBINED PREDICTION PAGE --------------------
def combined_prediction_page(sentiment_model, sentiment_vectorizer, emotion_model, emotion_vectorizer):
    st.title("Tweet Sentiment and Emotion Predictor")
    cleaner = MemoryOptimizedTweetCleaner()
    tweet = st.text_area("Write a tweet:", key="tweet_input_combined")
    if st.button("Predict"):
        if tweet.strip():
            cleaned_tweet = cleaner.clean_text(tweet)
            if cleaned_tweet:
                sentiment_vec = sentiment_vectorizer.transform([cleaned_tweet])
                sentiment_prediction = sentiment_model.predict(sentiment_vec)[0]
                sentiment_proba = sentiment_model.predict_proba(sentiment_vec)[0]
                sentiment_conf = np.max(sentiment_proba)
                sentiment_label_map = {-1: "Negative 😡", 0: "Neutral 😐", 1: "Positive 😊"}
                emotion_vec = emotion_vectorizer.transform([cleaned_tweet])
                emotion_prediction = emotion_model.predict(emotion_vec)[0]
                emotion_proba = emotion_model.predict_proba(emotion_vec)[0]
                emotion_conf = np.max(emotion_proba)
                emotion_label_map = {
                    "joy": "Joy 😊", "anger": "Anger 😠", "sadness": "Sadness 😢",
                    "fear": "Fear 😨", "surprise": "Surprise 😲", "neutral": "Neutral 😐"
                }
                emotion_label = emotion_label_map.get(emotion_prediction, f"Unknown Emotion ({emotion_prediction})")
                st.write("Cleaned Tweet:", cleaned_tweet)
                st.write(f"Sentiment Prediction: {sentiment_label_map[sentiment_prediction]} (Confidence: {sentiment_conf:.2f})")
                st.write(f"Emotion Prediction: {emotion_label} (Confidence: {emotion_conf:.2f})")
            else:
                st.write("Cleaned Tweet: No valid content after cleaning")
                st.write("Prediction: Unable to predict (invalid or empty tweet after cleaning)")
        else:
            st.write("Please enter a tweet to predict.")

# -------------------- MAIN APP LOGIC --------------------
def main():
    st.set_page_config(
        page_title="Twitter Sentiment Analysis",
        page_icon="🐦",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🐦 Twitter Sentiment Analysis Dashboard")
    st.markdown("This application analyzes Twitter data to visualize sentiment and emotions related to different topics.")
    st.write(f"Current memory usage: {MemoryMonitor.get_memory_usage():.2f} MB")
    if MemoryMonitor.get_memory_usage() > 400:
        st.warning("High memory usage detected. Consider clearing memory or selecting a different dataset.")

    if 'df' not in st.session_state:
        st.session_state['df'] = None
    if 'incident_df' not in st.session_state:
        st.session_state['incident_df'] = None
    if 'nyc_gdf' not in st.session_state:
        st.session_state['nyc_gdf'] = None
    if 'current_dataset_choice' not in st.session_state:
        st.session_state['current_dataset_choice'] = None

    st.sidebar.title("Data and View Options")
    
    dataset_choice = st.sidebar.selectbox(
        "Select a dataset to load:", 
        list(DATASET_FILES.keys()),
        index=None,
        placeholder="Choose a dataset"
    )
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("Clear Memory"):
            st.session_state['df'] = None
            st.session_state['incident_df'] = None
            st.session_state['nyc_gdf'] = None
            st.session_state['current_dataset_choice'] = None
            st.cache_data.clear()
            MemoryMonitor.force_garbage_collection()
            st.success("Memory cleared! You can now load a new dataset.")
    
    with col2:
        if st.button("Load Dataset", disabled=not dataset_choice):
            if dataset_choice != st.session_state.get('current_dataset_choice'):
                st.session_state['current_dataset_choice'] = dataset_choice
                st.session_state['df'] = None
                st.session_state['incident_df'] = None
                st.session_state['nyc_gdf'] = None
                st.cache_data.clear()
                MemoryMonitor.force_garbage_collection()
            
            try:
                with st.spinner(f"Loading main dataset for '{dataset_choice}'..."):
                    st.session_state['df'] = load_data(dataset_choice)
                with st.spinner(f"Loading incident data for '{dataset_choice}'..."):
                    st.session_state['incident_df'] = load_incident_data(dataset_choice)
                st.write(f"**Main Dataset Info ({dataset_choice}):**")
                st.write(f"Rows: {st.session_state['df'].shape[0]:,}")
                st.write(f"Columns: {list(st.session_state['df'].columns)}")
                st.write(f"**Incident Dataset Info ({dataset_choice}):**")
                st.write(f"Rows: {st.session_state['incident_df'].shape[0]:,}")
                st.write(f"Columns: {list(st.session_state['incident_df'].columns)}")
                if st.session_state['df'].empty or st.session_state['incident_df'].empty:
                    st.error("One or both datasets are empty. Check Google Drive file accessibility or file content.")
                else:
                    st.success(f"Data for '{dataset_choice}' has been loaded!")
            except Exception as e:
                st.error(f"Failed to load dataset '{dataset_choice}': {e}")
                st.session_state['df'] = None
                st.session_state['incident_df'] = None
                st.session_state['current_dataset_choice'] = None

    if st.session_state['df'] is None or st.session_state['df'].empty:
        st.info("Please select a dataset from the sidebar and click 'Load Dataset' to begin.")
    else:
        st.sidebar.markdown("---")
        visualization_choice = st.sidebar.radio(
            "Choose a visualization:", 
            [
                "📈 Overall Sentiment Pie Chart", 
                "😄 Overall Emotion Pie Chart", 
                "🗺 Tweet Sentiment Map", 
                "🗺 Tweet Emotion Map",
                "--- Advanced Visualizations ---",
                "📊 Top Emotions Pie Chart",
                "📊 Emotion by Sentiment Bar Chart",
                "📊 Emotion Confidence Box Plot",
                "🗺 Geographical Sentiment Scatter Plot",
                "📊 Median Income Histogram",
                "📈 Sentiment Trends Line Chart",
                "🔥 Emotion vs Sentiment Heatmap",
                "--- Geo-Statistical Maps ---",
                "🗺 ZIP Code Sentiment Maps",
                "🗺 ZIP Code Sentiment Heatmap",
                "💰 Average Median Income by Borough"
            ]
        )
        st.header(f"Visualizing: {st.session_state['current_dataset_choice']}")

        if visualization_choice in [
            "🗺 ZIP Code Sentiment Maps", 
            "🗺 ZIP Code Sentiment Heatmap", 
            "💰 Average Median Income by Borough"
        ]:
            if st.session_state['nyc_gdf'] is None:
                with st.spinner("Loading geographic data..."):
                    st.session_state['nyc_gdf'] = load_shapefile()

        if visualization_choice == "📈 Overall Sentiment Pie Chart":
            sentiment_pie_chart(st.session_state['df'])
        elif visualization_choice == "😄 Overall Emotion Pie Chart":
            emotion_pie_chart(st.session_state['df'])
        elif visualization_choice == "🗺 Tweet Sentiment Map":
            sentiment_map(st.session_state['df'])
        elif visualization_choice == "🗺 Tweet Emotion Map":
            emotion_map(st.session_state['df'])
        elif visualization_choice == "📊 Top Emotions Pie Chart":
            top_emotions_pie_chart(st.session_state['df'])
        elif visualization_choice == "📊 Emotion by Sentiment Bar Chart":
            emotion_sentiment_bar_chart(st.session_state['df'])
        elif visualization_choice == "📊 Emotion Confidence Box Plot":
            emotion_confidence_boxplot(st.session_state['df'])
        elif visualization_choice == "🗺 Geographical Sentiment Scatter Plot":
            geo_sentiment_scatterplot(st.session_state['df'])
        elif visualization_choice == "📊 Median Income Histogram":
            median_income_histogram(st.session_state['df'])
        elif visualization_choice == "📈 Sentiment Trends Line Chart":
            sentiment_trends_line_chart(st.session_state['df'])
        elif visualization_choice == "🔥 Emotion vs Sentiment Heatmap":
            emotion_sentiment_heatmap(st.session_state['df'])
        elif visualization_choice == "🗺 ZIP Code Sentiment Maps":
            zip_code_maps(st.session_state['incident_df'], st.session_state['nyc_gdf'])
        elif visualization_choice == "🗺 ZIP Code Sentiment Heatmap":
            zip_code_heatmap(st.session_state['incident_df'], st.session_state['nyc_gdf'])
        elif visualization_choice == "💰 Average Median Income by Borough":
            borough_income_chart(st.session_state['incident_df'])

if __name__ == '__main__':
    sentiment_model, sentiment_vectorizer, emotion_model, emotion_vectorizer = load_all_models_cached()
    
    if sentiment_model is None:
        st.error("Failed to load machine learning models. Please check your internet connection or the provided file IDs.")
        st.stop()
        
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to:", ["📊 Sentiment Analysis Dashboard", "🔍 Combined Prediction"], key='page_selector')

    if st.session_state.get('last_page') != page:
        st.session_state.clear()
        st.session_state.last_page = page
        
    if page == "📊 Sentiment Analysis Dashboard":
        main()
    elif page == "🔍 Combined Prediction":
        combined_prediction_page(sentiment_model, sentiment_vectorizer, emotion_model, emotion_vectorizer)
