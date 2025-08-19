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
from pathlib import Path
import warnings
from typing import Optional, Dict, List, Tuple
import sys
import io
import os
import gdown
import time


# Suppress all warnings for cleaner output
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
def download_from_drive(file_id: str, output_path: Path) -> None:
    if not output_path.exists() or output_path.stat().st_size == 0:
        try:
            gdown.download(id=file_id, output=str(output_path), quiet=True, fuzzy=True)
            if output_path.exists() and output_path.stat().st_size > 0:
                logger.info(f"Successfully downloaded {output_path.name}")
            else:
                raise ValueError(f"Download failed: {output_path.name} is empty or missing.")
        except Exception as e:
            logger.error(f"Failed to download {output_path.name}: {e}")
            raise

@st.cache_resource(max_entries=1)
def load_all_models_cached() -> Tuple:
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
        raise

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
    def get_memory_usage() -> float:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024

    @staticmethod
    def force_garbage_collection() -> None:
        gc.collect()
        logger.info(f"Memory usage after GC: {MemoryMonitor.get_memory_usage():.2f} MB")

@st.cache_data(max_entries=1)
def load_data(dataset_key: str) -> pd.DataFrame:
    if MemoryMonitor.get_memory_usage() > 400:
        logger.warning("High memory usage before loading dataset. Clearing caches.")
        st.cache_data.clear()
        MemoryMonitor.force_garbage_collection()
    
    file_info = DATASET_FILES[dataset_key]
    path = DATA_DIR / file_info["name"]
    
    try:
        # Log download start
        logger.info(f"Starting download for '{dataset_key}' from Google Drive ID: {file_info['id']}")
        start_time = time.time()
        download_from_drive(file_info["id"], path)
        download_time = time.time() - start_time
        logger.info(f"Download completed for '{dataset_key}' in {download_time:.2f} seconds. File size: {path.stat().st_size / 1024 / 1024:.2f} MB")
        
        # Check file existence and size
        if not path.exists() or path.stat().st_size == 0:
            logger.error(f"Downloaded file '{path}' is missing or empty")
            st.error(f"Failed to download dataset '{dataset_key}': File is missing or empty.")
            return pd.DataFrame()
        
        # Load data with timeout
        chunk_size = 10000
        max_rows = 20000
        chunks = []
        rows_loaded = 0
        total_rows_before_filter = 0
        load_start_time = time.time()
        
        logger.info(f"Reading CSV file '{path}' in chunks of {chunk_size} rows")
        for chunk in pd.read_csv(
            path,
            chunksize=chunk_size,
            low_memory=False,
            dtype={'latitude': 'float32', 'longitude': 'float32'},
            usecols=['latitude', 'longitude', 'date', 'emotion', 'category', 'cleaned_tweet', 'hashtags', 'median_income']
        ):
            total_rows_before_filter += len(chunk)
            chunk = chunk.dropna(subset=['latitude', 'longitude'])
            chunk = chunk[(chunk['latitude'].between(-90, 90)) & (chunk['longitude'].between(-180, 180))]
            if 'date' in chunk.columns:
                chunk['date'] = pd.to_datetime(chunk['date'], errors='coerce')
            chunks.append(chunk)
            rows_loaded += len(chunk)
            logger.info(f"Processed chunk: {len(chunk)} rows (total loaded: {rows_loaded})")
            if rows_loaded >= max_rows or (time.time() - load_start_time) > 300:  # 5-minute timeout
                logger.warning(f"Stopping data load for '{dataset_key}' after {rows_loaded} rows or timeout")
                break
        
        logger.info(f"Total rows before filtering for '{dataset_key}': {total_rows_before_filter}")
        if chunks:
            df = pd.concat(chunks, ignore_index=True)
            if len(df) > max_rows:
                df = df.sample(n=max_rows, random_state=42)
            logger.info(f"Loaded dataset '{dataset_key}' with {len(df)} rows after filtering")
        else:
            logger.warning(f"No valid data loaded for '{dataset_key}' after filtering")
            st.error(f"Failed to load dataset '{dataset_key}': No valid data after filtering. Total rows before filtering: {total_rows_before_filter}")
            return pd.DataFrame()
        
        df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]
        MemoryMonitor.force_garbage_collection()
        logger.info(f"Memory usage after loading '{dataset_key}': {MemoryMonitor.get_memory_usage():.2f} MB")
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
    
    try:
        # Log download start
        logger.info(f"Starting download for incident data '{incident_key}' from Google Drive ID: {file_info['id']}")
        start_time = time.time()
        download_from_drive(file_info["id"], path)
        download_time = time.time() - start_time
        logger.info(f"Download completed for incident data '{incident_key}' in {download_time:.2f} seconds. File size: {path.stat().st_size / 1024 / 1024:.2f} MB")
        
        # Check file existence and size
        if not path.exists() or path.stat().st_size == 0:
            logger.error(f"Downloaded file '{path}' is missing or empty")
            st.error(f"Failed to download incident data '{incident_key}': File is missing or empty.")
            return pd.DataFrame()
        
        # Load data with timeout
        chunk_size = 10000
        max_rows = 20000
        chunks = []
        rows_loaded = 0
        total_rows_before_filter = 0
        load_start_time = time.time()
        
        logger.info(f"Reading CSV file '{path}' in chunks of {chunk_size} rows")
        for chunk in pd.read_csv(
            path,
            chunksize=chunk_size,
            low_memory=False,
            dtype={'Incident Zip': 'str'},
            usecols=['Incident Zip', 'negative', 'positive', 'neutral']
        ):
            total_rows_before_filter += len(chunk)
            chunk['Incident Zip'] = chunk['Incident Zip'].astype(str).str.zfill(5)
            chunks.append(chunk)
            rows_loaded += len(chunk)
            logger.info(f"Processed chunk: {len(chunk)} rows (total loaded: {rows_loaded})")
            if rows_loaded >= max_rows or (time.time() - load_start_time) > 300:  # 5-minute timeout
                logger.warning(f"Stopping incident data load for '{incident_key}' after {rows_loaded} rows or timeout")
                break
        
        logger.info(f"Total rows before filtering for incident data '{incident_key}': {total_rows_before_filter}")
        if chunks:
            incident_df = pd.concat(chunks, ignore_index=True)
            if len(incident_df) > max_rows:
                incident_df = incident_df.sample(n=max_rows, random_state=42)
            logger.info(f"Loaded incident data '{incident_key}' with {len(incident_df)} rows after filtering")
        else:
            logger.warning(f"No valid data loaded for incident data '{incident_key}'")
            st.error(f"Failed to load incident data '{incident_key}': No valid data. Total rows before filtering: {total_rows_before_filter}")
            return pd.DataFrame()
        
        MemoryMonitor.force_garbage_collection()
        logger.info(f"Memory usage after loading incident data '{incident_key}': {MemoryMonitor.get_memory_usage():.2f} MB")
        return incident_df
    
    except Exception as e:
        logger.error(f"Error loading incident data '{incident_key}': {e}")
        st.error(f"Failed to load incident data '{incident_key}': {e}")
        return pd.DataFrame()
def load_shapefile() -> gpd.GeoDataFrame:
    for sf in SHAPE_FILES:
        path = SHAPE_DIR / sf["name"]
        download_from_drive(sf["id"], path)
    try:
        zcta_gdf = gpd.read_file(SHAPEFILE_PATH)
        zcta_gdf['ZCTA5CE10'] = zcta_gdf['ZCTA5CE10'].astype(str).str.zfill(5)
        nyc_zip_prefixes = ('100', '101', '102', '103', '104', '111', '112', '113', '114', '116')
        nyc_gdf = zcta_gdf[zcta_gdf['ZCTA5CE10'].str.startswith(nyc_zip_prefixes)].copy()
        del zcta_gdf
        MemoryMonitor.force_garbage_collection()
        return nyc_gdf
    except Exception as e:
        logger.error(f"Failed to load geographic data: {e}")
        raise

# -------------------- VISUALIZATION FUNCTIONS --------------------
plt.style.use('seaborn-v0_8')
sns.set_context('talk', font_scale=1.2)
top_n = 5

def top_emotions_pie_chart(df: pd.DataFrame) -> None:
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
    del value_counts, labels, sizes, colors, explode
    MemoryMonitor.force_garbage_collection()
    st.markdown("**Explanation:** This chart visualizes the distribution of the top 5 most frequently detected emotions...")

def emotion_sentiment_bar_chart(df: pd.DataFrame) -> None:
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
    del emotion_sentiment, top_emotions
    MemoryMonitor.force_garbage_collection()
    st.markdown("**Explanation:** This chart breaks down each of the top emotions by sentiment category...")

def emotion_confidence_boxplot(df: pd.DataFrame) -> None:
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
    del top_emotions
    MemoryMonitor.force_garbage_collection()
    st.markdown("**Explanation:** A higher confidence score indicates that the model is more certain...")

def geo_sentiment_scatterplot(df: pd.DataFrame) -> None:
    st.subheader("Geographical Distribution of Tweet Sentiments")
    if 'latitude' not in df.columns or 'longitude' not in df.columns:
        st.warning("Latitude and Longitude data not available for this visualization.")
        return
    sample_df = df.sample(min(MAX_POINTS_SCATTER, len(df)), random_state=42)
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.scatterplot(
        x='longitude', y='latitude',
        hue='category', palette={1: '#ff9999', 0: '#66b3ff', -1: '#99ff99'},
        data=sample_df, alpha=0.6, ax=ax
    )
    ax.set_title('Geographical Distribution of Tweet Sentiments', fontsize=18)
    st.pyplot(fig)
    plt.close(fig)
    del sample_df
    MemoryMonitor.force_garbage_collection()
    st.markdown("**Explanation:** This scatter plot maps the location of each tweet...")

def median_income_histogram(df: pd.DataFrame) -> None:
    st.subheader("Distribution of Median Income in Tweet Locations")
    if 'median_income' not in df.columns:
        st.warning("Median Income data not available for this visualization.")
        return
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.histplot(df['median_income'].dropna(), bins=50, kde=True, color='purple', ax=ax)
    ax.set_title('Distribution of Median Income in Tweet Locations', fontsize=18)
    st.pyplot(fig)
    plt.close(fig)
    MemoryMonitor.force_garbage_collection()
    st.markdown("**Explanation:** This histogram shows the distribution of median income...")

def sentiment_trends_line_chart(df: pd.DataFrame) -> None:
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
    del sentiment_over_time
    MemoryMonitor.force_garbage_collection()
    st.markdown("**Explanation:** This chart tracks the number of positive, negative, and neutral tweets...")

def emotion_sentiment_heatmap(df: pd.DataFrame) -> None:
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
    del top_emotions, emotion_sentiment_counts
    MemoryMonitor.force_garbage_collection()
    st.markdown("**Explanation:** This heatmap shows the relationship between different emotions...")

def sentiment_pie_chart(df: pd.DataFrame) -> None:
    st.subheader("Sentiment Distribution")
    try:
        if 'category' not in df.columns:
            logger.error("Missing 'category' column in DataFrame")
            st.warning("Sentiment data not available for this dataset.")
            return
        value_counts = df['category'].value_counts()
        labels = ['Positive', 'Neutral', 'Negative']
        sizes = [value_counts.get(1, 0), value_counts.get(0, 0), value_counts.get(-1, 0)]
        if sum(sizes) == 0:
            logger.error("No valid sentiment data in 'category' column")
            st.warning("No valid sentiment data available for visualization.")
            return
        fig, ax = plt.subplots(figsize=(6, 6))
        colors = ['#4CAF50', '#2196F3', '#F44336']
        explode = (0.1, 0, 0)
        ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=90)
        ax.axis('equal')
        st.pyplot(fig)
        plt.close(fig)
        del value_counts, sizes, labels, colors, explode
        MemoryMonitor.force_garbage_collection()
        st.markdown("**Explanation:** This simple pie chart provides a clear and direct summary...")
    except Exception as e:
        logger.error(f"Error in sentiment_pie_chart: {e}")
        st.error(f"Failed to render sentiment pie chart: {e}")

def emotion_pie_chart(df: pd.DataFrame) -> None:
    st.subheader("Emotion Distribution")
    try:
        if 'emotion' not in df.columns or df['emotion'].isnull().all():
            logger.error("Missing or empty 'emotion' column in DataFrame")
            st.warning("Emotion data not available for this dataset.")
            return
        value_counts = df['emotion'].value_counts()
        if value_counts.empty:
            logger.error("No valid emotion data in 'emotion' column")
            st.warning("No valid emotion data available for visualization.")
            return
        labels = value_counts.index.tolist()
        sizes = value_counts.values.tolist()
        fig, ax = plt.subplots(figsize=(6, 6))
        colors = sns.color_palette("husl", len(labels))
        explode = [0.1] + [0] * (len(labels) - 1)
        ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=90)
        ax.axis('equal')
        st.pyplot(fig)
        plt.close(fig)
        del value_counts, labels, sizes, colors, explode
        MemoryMonitor.force_garbage_collection()
        st.markdown("**Explanation:** Similar to the sentiment chart, this pie chart provides a quick visual...")
    except Exception as e:
        logger.error(f"Error in emotion_pie_chart: {e}")
        st.error(f"Failed to render emotion pie chart: {e}")

def sentiment_map(df: pd.DataFrame) -> None:
    st.subheader("Geographical Sentiment Map (Sampled)")
    nyc_bbox = {'min_lon': -74.27, 'max_lon': -73.68, 'min_lat': 40.48, 'max_lat': 40.95}
    nyc_df = df[(df['longitude'].between(nyc_bbox['min_lon'], nyc_bbox['max_lon'])) &
                (df['latitude'].between(nyc_bbox['min_lat'], nyc_bbox['max_lat']))]
    sentiment_map_colors = {-1: "Negative 🔴", 0: "Neutral 🟡", 1: "Positive 🟢"}
    for cat, label in sentiment_map_colors.items():
        color = '#e74c3c' if cat == -1 else '#f39c12' if cat == 0 else '#2ecc71'
        sub_df = nyc_df[nyc_df['category'] == cat].sample(min(2000, len(nyc_df[nyc_df['category'] == cat])), random_state=42)
        st.write(f"### {label}")
        fig = px.scatter_mapbox(sub_df, lat="latitude", lon="longitude",
                                color_discrete_sequence=[color], zoom=10, height=400,
                                hover_data=["emotion"])
        fig.update_layout(mapbox_style="carto-positron", margin={"r":0,"t":0,"l":0,"b":0})
        st.plotly_chart(fig, use_container_width=True)
        del sub_df
        MemoryMonitor.force_garbage_collection()
        st.markdown(f"**Explanation:** This map shows the geographical distribution of tweets with **{label.replace('🔴','').replace('🟡','').replace('🟢','').strip()}** sentiment...")

def emotion_map(df: pd.DataFrame) -> None:
    st.subheader("Geographical Emotion Map (Sampled)")
    if 'emotion' not in df.columns or df['emotion'].isnull().all():
        st.warning("Emotion data not available for this dataset.")
        return
    nyc_bbox = {'min_lon': -74.27, 'max_lon': -73.68, 'min_lat': 40.48, 'max_lat': 40.95}
    nyc_df = df[(df['longitude'].between(nyc_bbox['min_lon'], nyc_bbox['max_lon'])) &
                (df['latitude'].between(nyc_bbox['min_lat'], nyc_bbox['max_lat']))]
    emotion_colors = {"joy": "Joy 😊", "anger": "Anger 😠", "sadness": "Sadness 😢", "fear": "Fear 😨"}
    for emo, label in emotion_colors.items():
        color_hex = {"joy": "#FFD700", "anger": "#FF4500", "sadness": "#1E90FF", "fear": "#9400D3"}.get(emo)
        if color_hex:
            sub_df = nyc_df[nyc_df['emotion'] == emo].sample(min(2000, len(nyc_df[nyc_df['emotion'] == emo])), random_state=42)
            st.write(f"### {label}")
            fig = px.scatter_mapbox(sub_df, lat="latitude", lon="longitude",
                                    color_discrete_sequence=[color_hex], zoom=10, height=400,
                                    hover_data=["category"])
            fig.update_layout(mapbox_style="carto-positron", margin={"r":0,"t":0,"l":0,"b":0})
            st.plotly_chart(fig, use_container_width=True)
            del sub_df
            MemoryMonitor.force_garbage_collection()
            st.markdown(f"**Explanation:** This map shows the locations of tweets expressing **{label.replace('😊','').replace('😠','').replace('😢','').replace('😨','').strip()}**...")

def zip_code_maps(incident_df: pd.DataFrame, nyc_gdf: gpd.GeoDataFrame) -> None:
    st.subheader("ZIP Code Sentiment Maps")
    map_type = st.selectbox("Select map type:", ["Count Maps", "Percentage Maps"], key="zip_code_map_type")
    sentiment_type = st.selectbox("Select sentiment:", ["Negative", "Positive", "Neutral"], key="zip_code_sentiment")
    
    incident_sums = incident_df.groupby('Incident Zip')[['negative', 'positive', 'neutral']].sum().reset_index()
    incident_sums['total'] = incident_sums[['negative', 'positive', 'neutral']].sum(axis=1)
    for col in ['negative', 'positive', 'neutral']:
        incident_sums[col + '_pct'] = (incident_sums[col] / incident_sums['total']).fillna(0) * 100
    merged_gdf = nyc_gdf.merge(incident_sums, left_on='ZCTA5CE10', right_on='Incident Zip', how='left')
    merged_gdf[['negative', 'positive', 'neutral', 'negative_pct', 'positive_pct', 'neutral_pct']] = merged_gdf[
        ['negative', 'positive', 'neutral', 'negative_pct', 'positive_pct', 'neutral_pct']].fillna(0)
    
    column = f"{sentiment_type.lower()}_pct" if map_type == "Percentage Maps" else sentiment_type.lower()
    title = f"{sentiment_type} Incidents {'%' if map_type == 'Percentage Maps' else 'Count'}"
    cmap = 'Reds' if sentiment_type == "Negative" else 'Greens' if sentiment_type == "Positive" else 'Blues'
    
    st.write(f"#### {title}")
    fig, ax = plt.subplots(figsize=(6, 6))
    merged_gdf.plot(column=column, cmap=cmap, linewidth=0.6, ax=ax, edgecolor='0.8', legend=True,
                    legend_kwds={'label': "Percentage" if map_type == "Percentage Maps" else "Count"})
    ax.axis('off')
    ax.set_title(title, fontsize=12)
    st.pyplot(fig)
    plt.close(fig)
    del incident_sums, merged_gdf
    MemoryMonitor.force_garbage_collection()
    st.markdown(f"**Explanation:** This map shows the {sentiment_type.lower()} sentiment {'percentage' if map_type == 'Percentage Maps' else 'count'} per ZIP code...")

def zip_code_heatmap(incident_df: pd.DataFrame, nyc_gdf: gpd.GeoDataFrame) -> None:
    st.subheader("ZIP Code Sentiment Heatmap")
    incident_sums = incident_df.groupby('Incident Zip')[['negative', 'positive', 'neutral']].sum().reset_index()
    incident_sums['total'] = incident_sums[['negative', 'positive', 'neutral']].sum(axis=1)
    incident_sums['combined_sentiment'] = (incident_sums['positive'] - incident_sums['negative']) / incident_sums['total'].replace(0, 1)
    merged_gdf = nyc_gdf.merge(incident_sums, left_on='ZCTA5CE10', right_on='Incident Zip', how='left')
    merged_gdf['combined_sentiment'] = merged_gdf['combined_sentiment'].fillna(0)
    fig, ax = plt.subplots(figsize=(12, 10))
    merged_gdf.plot(column='combined_sentiment', cmap='RdYlBu_r', linewidth=0.6, ax=ax, edgecolor='0.8', legend=True,
                    legend_kwds={'label': "Sentiment Score"})
    ax.set_title("NYC Sentiment Heatmap 2020", fontsize=16)
    ax.axis('off')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    del incident_sums, merged_gdf
    MemoryMonitor.force_garbage_collection()
    st.markdown("**Explanation:** This heatmap provides a single, easy-to-read view of sentiment across NYC ZIP codes...")

def borough_income_chart(df: pd.DataFrame) -> None:
    st.subheader("Average Median Income by NYC Borough")
    if 'median_income' not in df.columns:
        st.error("Error: The selected dataset does not contain 'median_income' data for this visualization.")
        return
    zip_column = next((col for col in ['Incident Zip', 'incident_zip', 'zip_code'] if col in df.columns), None)
    if not zip_column:
        st.error("Error: No ZIP code column found for borough mapping.")
        return
    borough_map = {
        "Manhattan": ['100', '101', '102'],
        "Bronx": ['104'],
        "Brooklyn": ['112'],
        "Queens": ['110', '111', '113', '114', '116'],
        "Staten Island": ['103']
    }
    def get_borough(zipcode: str) -> Optional[str]:
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
    del borough_income
    MemoryMonitor.force_garbage_collection()
    st.markdown("**Explanation:** This bar chart visualizes the average median income for each NYC borough...")

def top_hashtags_bar_chart(df: pd.DataFrame) -> None:
    st.subheader("Top 10 Most Frequent Hashtags")
    if 'hashtags' not in df.columns:
        st.warning("Hashtag data not available for this dataset.")
        return
    all_hashtags = df['hashtags'].astype(str).str.split(',')
    flat_hashtags = [item.strip().lower() for sublist in all_hashtags.dropna() for item in sublist]
    hashtag_counts = pd.Series(flat_hashtags).value_counts().head(10)
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.barplot(x=hashtag_counts.values, y=hashtag_counts.index, palette='viridis', ax=ax)
    ax.set_title('Top 10 Most Frequent Hashtags', fontsize=18)
    ax.set_xlabel('Count', fontsize=14)
    ax.set_ylabel('Hashtag', fontsize=14)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    del all_hashtags, flat_hashtags, hashtag_counts
    MemoryMonitor.force_garbage_collection()
    st.markdown("**Explanation:** This bar chart shows the 10 most used hashtags in the dataset...")

def tweet_length_vs_sentiment_boxplot(df: pd.DataFrame) -> None:
    st.subheader("Tweet Length vs. Sentiment")
    if 'cleaned_tweet' not in df.columns or 'category' not in df.columns:
        st.warning("Required 'cleaned_tweet' or 'category' data not available.")
        return
    df['tweet_length'] = df['cleaned_tweet'].astype(str).apply(len)
    sentiment_map = {1: 'Positive', 0: 'Neutral', -1: 'Negative'}
    df['sentiment_label'] = df['category'].map(sentiment_map)
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.boxplot(x='sentiment_label', y='tweet_length', data=df, ax=ax, palette='Set2')
    ax.set_title('Tweet Length Distribution by Sentiment', fontsize=18)
    ax.set_xlabel('Sentiment', fontsize=14)
    ax.set_ylabel('Tweet Length (characters)', fontsize=14)
    st.pyplot(fig)
    plt.close(fig)
    del df['tweet_length'], df['sentiment_label']
    MemoryMonitor.force_garbage_collection()
    st.markdown("**Explanation:** This box plot helps us understand if there's a relationship between the length of a tweet...")

def emotion_trends_line_chart(df: pd.DataFrame) -> None:
    st.subheader("Emotion Trends Over Time")
    if 'date' not in df.columns or 'emotion' not in df.columns:
        st.warning("Required 'date' or 'emotion' data not available.")
        return
    emotion_over_time = df.groupby([df['date'].dt.date, 'emotion']) \
                         .size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(14, 7))
    for col in emotion_over_time.columns:
        ax.plot(emotion_over_time.index, emotion_over_time[col], label=col, linewidth=2)
    ax.legend(title='Emotion', fontsize=12)
    ax.set_title('Emotion Trends Over Time', fontsize=18)
    ax.set_xlabel('Date', fontsize=14)
    ax.set_ylabel('Number of Tweets', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig)
    plt.close(fig)
    del emotion_over_time
    MemoryMonitor.force_garbage_collection()
    st.markdown("**Explanation:** This chart visualizes how the frequency of different emotions changes over time...")

def combined_prediction_page(sentiment_model, sentiment_vectorizer, emotion_model, emotion_vectorizer) -> None:
    st.title("Tweet Sentiment and Emotion Predictor")
    cleaner = MemoryOptimizedTweetCleaner()
    tweet = st.text_area("Write a tweet:", key="tweet_input_combined")
    if st.button("Predict"):
        if tweet.strip():
            with st.spinner("Analyzing tweet..."):
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
                    
                    st.markdown("### Prediction Results")
                    st.write(f"**Cleaned Tweet:** `{cleaned_tweet}`")
                    st.success(f"**Sentiment Prediction:** {sentiment_label_map[sentiment_prediction]} (Confidence: {sentiment_conf:.2f})")
                    st.success(f"**Emotion Prediction:** {emotion_label} (Confidence: {emotion_conf:.2f})")
                else:
                    st.write("Cleaned Tweet: No valid content after cleaning")
                    st.warning("Prediction: Unable to predict (invalid or empty tweet after cleaning)")
        else:
            st.warning("Please enter a tweet to predict.")

def load_models_with_progress_bar() -> None:
    if 'models_loaded' not in st.session_state or not st.session_state['models_loaded']:
        with st.status("Loading machine learning models...", expanded=True) as status:
            st.write("Checking model files...")
            total_steps = len(MODEL_FILES)
            progress_bar = st.progress(0, text="Downloading and loading files...")
            
            paths = {}
            for i, (key, info) in enumerate(MODEL_FILES.items()):
                path = DATA_DIR / info["name"]
                st.write(f"Downloading {info['name']}...")
                download_from_drive(info["id"], path)
                paths[key] = path
                progress_bar.progress((i + 1) / total_steps, text=f"Loading {key}...")
            
            st.write("Initializing models...")
            sentiment_model, sentiment_vectorizer, emotion_model, emotion_vectorizer = load_all_models_cached()
            
            st.session_state['sentiment_model'] = sentiment_model
            st.session_state['sentiment_vectorizer'] = sentiment_vectorizer
            st.session_state['emotion_model'] = emotion_model
            st.session_state['emotion_vectorizer'] = emotion_vectorizer
            st.session_state['models_loaded'] = True
            
            status.update(label="Models loaded successfully!", state="complete", expanded=False)
            st.success("All models are ready! You can now use the predictor tool.")
            MemoryMonitor.force_garbage_collection()

# -------------------- MAIN APP LOGIC --------------------
def main() -> None:
    st.set_page_config(
        page_title="Twitter Sentiment Analysis",
        page_icon="🐦",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("🐦 Twitter Sentiment & Emotion Analysis App")
    st.markdown("This application analyzes Twitter data to visualize sentiment and emotions related to different topics.")
    st.write(f"Current memory usage: {MemoryMonitor.get_memory_usage():.2f} MB")
    if MemoryMonitor.get_memory_usage() > 400:
        st.warning("High memory usage detected. Consider clearing memory or selecting a different dataset.")

    load_models_with_progress_bar()

    tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🔍 Predictor", "📄 Project Report"])
    
    with tab1:
        st.subheader("Dashboard")
        if 'current_dataset_choice' not in st.session_state:
            st.session_state['current_dataset_choice'] = None
        if 'df' not in st.session_state:
            st.session_state['df'] = None
        if 'incident_df' not in st.session_state:
            st.session_state['incident_df'] = None

        with st.container(border=True):
            st.subheader("1. Load Data")
            dataset_choice = st.selectbox(
                "Select a dataset to load:", 
                list(DATASET_FILES.keys()),
                index=None,
                placeholder="Choose a dataset",
                key="dataset_selector"
            )
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Clear Memory"):
                    st.session_state['df'] = None
                    st.session_state['incident_df'] = None
                    st.session_state['nyc_gdf'] = None
                    st.session_state['current_dataset_choice'] = None
                    st.cache_data.clear()
                    st.cache_resource.clear()
                    MemoryMonitor.force_garbage_collection()
                    st.success("Memory cleared! You can now load a new dataset.")
            
            with col2:
                if st.button("Load Dataset", disabled=not dataset_choice):
                    if dataset_choice != st.session_state['current_dataset_choice']:
                        st.session_state['df'] = None
                        st.session_state['incident_df'] = None
                        st.session_state['nyc_gdf'] = None
                        st.cache_data.clear()
                        st.cache_resource.clear()
                        MemoryMonitor.force_garbage_collection()
                        st.session_state['current_dataset_choice'] = dataset_choice
                    try:
                        with st.spinner(f"Loading main dataset for '{dataset_choice}'..."):
                            st.session_state['df'] = load_data(dataset_choice)
                        with st.spinner(f"Loading incident data for '{dataset_choice}'..."):
                            st.session_state['incident_df'] = load_incident_data(dataset_choice)
                        # Debugging output
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
                elif not dataset_choice:
                    st.warning("Please select a dataset first.")

        if st.session_state['df'] is None or st.session_state['df'].empty:
            st.info("Please select a dataset above and click 'Load Dataset' to begin visualizing the data.")
        else:
            st.header(f"Visualizing: {st.session_state['current_dataset_choice']} Data")
            
            tab_basic, tab_advanced, tab_geo, tab_statistical, tab_summary = st.tabs([
                "Basic Visualizations", 
                "Advanced Charts", 
                "Geographical Maps",
                "Geo-Statistical Maps",
                "Data Summary"
            ])

            with tab_basic:
                st.markdown("### Basic Visualizations")
                sentiment_pie_chart(st.session_state.df)
                emotion_pie_chart(st.session_state.df)

            with tab_advanced:
                st.markdown("### Advanced Visualizations")
                st.markdown("These charts provide deeper insights into the relationships between different data points.")
                
                advanced_chart_choice = st.selectbox(
                    "Select an advanced chart to display:",
                    [
                        "Top Emotions Pie Chart",
                        "Emotion by Sentiment Bar Chart",
                        "Emotion Confidence Box Plot",
                        "Median Income Histogram",
                        "Sentiment Trends Line Chart",
                        "Emotion vs Sentiment Heatmap",
                        "Top 10 Hashtags Bar Chart",
                        "Tweet Length vs. Sentiment",
                        "Emotion Trends over Time"
                    ],
                    key="advanced_chart_selector"
                )

                if advanced_chart_choice == "Top Emotions Pie Chart":
                    top_emotions_pie_chart(st.session_state.df)
                elif advanced_chart_choice == "Emotion by Sentiment Bar Chart":
                    emotion_sentiment_bar_chart(st.session_state.df)
                elif advanced_chart_choice == "Emotion Confidence Box Plot":
                    emotion_confidence_boxplot(st.session_state.df)
                elif advanced_chart_choice == "Median Income Histogram":
                    median_income_histogram(st.session_state.df)
                elif advanced_chart_choice == "Sentiment Trends Line Chart":
                    sentiment_trends_line_chart(st.session_state.df)
                elif advanced_chart_choice == "Emotion vs Sentiment Heatmap":
                    emotion_sentiment_heatmap(st.session_state.df)
                elif advanced_chart_choice == "Top 10 Hashtags Bar Chart":
                    top_hashtags_bar_chart(st.session_state.df)
                elif advanced_chart_choice == "Tweet Length vs. Sentiment":
                    tweet_length_vs_sentiment_boxplot(st.session_state.df)
                elif advanced_chart_choice == "Emotion Trends over Time":
                    emotion_trends_line_chart(st.session_state.df)
            
            with tab_geo:
                st.markdown("### Geographical Maps")
                geo_map_choice = st.selectbox(
                    "Select a map to view:", 
                    ["Tweet Sentiment Map", "Tweet Emotion Map", "Geographical Sentiment Scatter Plot"],
                    key="geo_map_selector"
                )
                
                if geo_map_choice == "Tweet Sentiment Map":
                    sentiment_map(st.session_state.df)
                elif geo_map_choice == "Tweet Emotion Map":
                    emotion_map(st.session_state.df)
                elif geo_map_choice == "Geographical Sentiment Scatter Plot":
                    geo_sentiment_scatterplot(st.session_state.df)

            with tab_statistical:
                if 'nyc_gdf' not in st.session_state or st.session_state['nyc_gdf'] is None:
                    with st.spinner("Loading geographic data..."):
                        st.session_state['nyc_gdf'] = load_shapefile()
                
                st.markdown("### Geo-Statistical Maps")
                st.markdown("These maps combine tweet data with geographic information at the ZIP code level.")
                geo_statistical_map_choice = st.selectbox(
                    "Select a map:", 
                    ["ZIP Code Sentiment Maps", "ZIP Code Sentiment Heatmap", "Average Median Income by Borough"],
                    key="geo_statistical_map_selector"
                )
                
                if geo_statistical_map_choice == "ZIP Code Sentiment Maps":
                    zip_code_maps(st.session_state.incident_df, st.session_state.nyc_gdf)
                elif geo_statistical_map_choice == "ZIP Code Sentiment Heatmap":
                    zip_code_heatmap(st.session_state.incident_df, st.session_state.nyc_gdf)
                elif geo_statistical_map_choice == "Average Median Income by Borough":
                    borough_income_chart(st.session_state.incident_df)

            with tab_summary:
                st.header("Dataset Summary")
                st.markdown("This section provides a quick look at the raw data and its structure.")
                st.subheader("Data at a glance:")
                st.write(st.session_state.df.head())
                st.subheader("Dataset Shape:")
                st.write(f"Rows: {st.session_state['df'].shape[0]:,}")
                st.write(f"Columns: {st.session_state['df'].shape[1]}")
                st.subheader("DataFrame Info:")
                buffer = io.StringIO()
                st.session_state.df.info(buf=buffer)
                s = buffer.getvalue()
                st.text(s)
                st.subheader("Incident Data Summary:")
                st.write(st.session_state.incident_df.head())
                st.write(f"Rows: {st.session_state['incident_df'].shape[0]:,}")
                st.write(f"Columns: {st.session_state['incident_df'].shape[1]}")
                buffer = io.StringIO()
                st.session_state.incident_df.info(buf=buffer)
                s = buffer.getvalue()
                st.text(s)

    with tab2:
        if 'models_loaded' in st.session_state and st.session_state['models_loaded']:
            combined_prediction_page(
                st.session_state['sentiment_model'],
                st.session_state['sentiment_vectorizer'],
                st.session_state['emotion_model'],
                st.session_state['emotion_vectorizer']
            )
        else:
            st.info("The models are still loading. Please wait a moment and then check this tab again.")

    with tab3:
        st.header("📄 Project Report")
        st.markdown("""
### Tweet Classification and Sentiment Analysis for NYC Data Using AI Models
**Mohamed Mostafa**
**August 12, 2025**
---

### Data Cleaning and Preprocessing

#### Cleaning Methodology
A `MemoryOptimizedTweetCleaner` class processed tweets in chunks, removing URLs, mentions, retweet prefixes, digits, punctuation, HTML, and extra whitespace using regex. Emojis and emoticons were converted to text (e.g., to “face_with_medical_mask”) for sentiment preservation.

#### Preservation of Sentiment-Relevant Features
Emojis, emoticons, hashtags (without #), and slang were retained or transformed to maintain emotional context for downstream analysis.

#### Implementation Details
Parallel processing (`ThreadPoolExecutor`, `max_workers=6–8`) handled chunks (~500MB), with memory monitoring (`psutil`) and garbage collection. Output was a “cleaned_tweets” column, with ~95% of rows retained after dropping nulls.

### Geospatial and Socioeconomic Enrichment

#### Adding Median Income
ACS data (`S1903_C03_001E`) and census tract shapefiles (`tl_2020_36_tract20.shp`) were merged via spatial joins (`geopandas`, `sjoin_nearest`). Missing incomes were imputed using neighbor averages.

#### Adding Incident ZIP Codes
The `uszipcode` library geocoded rounded lat/long pairs to ZIPs, with defaults (`-11414`) for invalid coordinates.

#### Implementation Details
Chunked processing assigned income and ZIPs to valid coordinates (~high% coverage). Outputs were saved as CSVs, with plots verifying assignments.

### Problems Faced, Solutions, and Unsolved Issues

#### Data Cleanliness (Solved)
Noise (~20–30% URLs/mentions) was removed, retaining ~95% cleaned rows.

#### Missing or Invalid Geospatial Data (Partially Solved)
Valid coordinates were enriched; invalid ones used defaults, leaving some nulls.

#### Multilingual Content (Solved)
LLMs handled multilingual text without translation, ensuring accurate sentiment.

#### Location Data Concentration (Unsolved)
Tweets from ~250 spots biased heatmaps, unresolved due to data limitations.

### Methodology for Topic Classification

#### Model Selection Process
Evaluated models included `twitter-roberta-topic-multi-all` (~0.765 F1), BART, BERTopic, fastText, and keyword-based methods. `twitter-roberta` was chosen for accuracy and efficiency.

#### Chosen Model: `twitter-roberta-topic-multi-all`

##### Model Description
A RoBERTa-based model fine-tuned on 11,267 tweets for multi-label classification (COVID-19, politics, economics).

##### Architecture
12 transformer layers, 768 hidden dimensions, ~125M parameters, with sigmoid logits for multi-label outputs.

##### Training Procedure and Data
Fine-tuned on `TweetTopic_multi` dataset (6,090 train, 1,679 test), lr=2e-5, batch=8.

##### Labels/Topics
19 topics mapped to COVID-19, politics, economics (e.g., “fitness_&_health” → COVID).

##### Performance Metrics
F1-micro=0.765, F1-macro=0.619, Accuracy=0.549 on `test_2021`.

##### Computational Requirements and Limitations
~500MB, CPU-friendly, ~0.1–0.5s/tweet. Limited by topic bias and temporal drift.

##### Intended Uses and How to Use
Used for tweet classification via `transformers` library, thresholding probabilities at 0.5.

### Implementation of Topic Classification
Batched inference (1,000 rows) on Google Colab classified “cleaned_tweets” into COVID-19, politics, and economics, adding columns (e.g., “category_covid”). Runtime was ~hours with GPU support.

### Sentiment and Emotion Analysis on Classified Tweets

#### Approach Using LLMs for Sentiment Analysis
Mistral-7B-Instruct-v0.1 was fine-tuned for sentiment (positive: 1, neutral: 0, negative: -1), capturing context and multilingual nuances.

#### Fine-Tuning Tutorial and Multilingual Handling
Fine-tuning used 4-bit quantization, LoRA (r=8), and a small dataset. Multilingual tweets were analyzed directly via LLM pretraining.

#### Emotion Classification Using Incremental Learning
To extend beyond sentiment polarity, an incremental learning approach classified emotions (e.g., joy, anger, sadness) on the large dataset, complementing the sentiment analysis by providing deeper emotional insights.

-   **Methodology:** The approach used a feature extraction technique to convert cleaned tweet text into numerical representations suitable for large-scale processing. A classifier was trained incrementally to handle the ~24 million tweets efficiently, accommodating the dataset’s size without requiring extensive computational resources. The process involved:

    -   **Validation Set Creation:** A sample of 500,000 tweets was used to create a balanced training and validation set, split 80:20, ensuring representation of all emotion categories (e.g., joy, anger, sadness, fear).
    -   **Incremental Training:** The dataset was processed in chunks of 100,000 tweets. Each chunk’s text was transformed into sparse feature vectors, capturing unigrams and bigrams for contextual understanding. The classifier was updated iteratively, learning from each chunk while maintaining memory efficiency. The first chunk established the set of emotion labels, with subsequent chunks refining the model.
    -   **Evaluation:** The trained classifier was evaluated on the validation set using metrics like accuracy, precision, recall, and F1-score to assess performance across emotion categories.
    -   **Model Persistence:** The trained classifier and feature extractor were saved for future use, ensuring reproducibility and scalability.

-   **Implementation Details:** The method leveraged standard Python libraries for data handling and machine learning. Feature extraction created compact representations of tweet text, optimized for large-scale processing. The classifier used a stochastic gradient descent approach with a log-loss objective, designed for iterative updates. Chunked processing and memory management ensured scalability on standard hardware, completing in ~hours. The model achieved an accuracy of 0.7743 on the validation set, with the following performance metrics across emotion classes (negative: -1, neutral: 0, positive: 1):

    **Table 1: Emotion Classification Performance Metrics**

| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| Negative (-1) | 0.89 | 0.22 | 0.36 | 14,489 |
| Neutral (0) | 0.76 | 0.95 | 0.84 | 50,721 |
| Positive (1) | 0.79 | 0.75 | 0.77 | 34,790 |

-   **Overall Accuracy:** 0.7743
-   **Macro Avg:** Precision: 0.81, Recall: 0.64, F1-Score: 0.66
-   **Weighted Avg:** Precision: 0.79, Recall: 0.77, F1-Score: 0.75

These results indicate strong performance for neutral and positive emotions, with high recall for neutral (0.95) and balanced precision/recall for positive (0.79/0.75). Negative emotions had high precision (0.89) but lower recall (0.22), suggesting challenges in detecting negative tweets, possibly due to class imbalance or nuanced expressions. The approach complemented LLM-based sentiment analysis by identifying nuanced emotions, e.g., anger in political tweets or sadness in COVID-19 discussions, enhancing insights into public mood.

### Overall Implementation and Workflow
The pipeline was:
1.  **Acquisition:** Load ~24M tweet CSV.
2.  **Cleaning:** `MemoryOptimizedTweetCleaner` for noise removal.
3.  **Enrichment:** Add income (spatial joins) and ZIP codes (geocoding).
4.  **Classification:** `twitter-roberta` for topic categorization.
5.  **Sentiment/Emotion Analysis:** `Mistral-7B` for sentiment polarity; incremental learning for emotions.

Tools included `pandas`, `geopandas`, `uszipcode`, `transformers`, and `sklearn`. GPU bursts via Colab optimized heavy steps. Verifications ensured quality at each stage.

### Results and Discussion
Cleaning retained ~95% rows. Enrichment assigned income/ZIPs to ~high % valid coordinates. Classification mapped tweets to categories, with overlaps (e.g., COVID-economics). Sentiment showed negative spikes (e.g., Q2 2020 COVID) and positive trends (e.g., vaccine rollouts). Emotion classification revealed nuanced patterns, e.g., anger in politics (precision 0.89 for negative), sadness in COVID tweets (high neutral recall). Geospatial concentration limited heatmap granularity. Future work could integrate real-time data and additional enrichments.

### Conclusion
This project processed ~24M NYC tweets, overcoming noise and multilingual challenges to reveal sentiment and emotion patterns tied to socioeconomic factors. The incremental learning approach, with 0.7743 accuracy, enhanced emotional insights, offering a scalable framework for urban social media analysis.

### References
-   Barbieri, F., et al. (2020). TweetEval: Unified Benchmark and Comparative Evaluation for Tweet Classification. Findings of EMNLP 2020. https://arxiv.org/abs/2010.12421
-   Barbieri, F., et al. (2022). TweetEval: Unified Benchmark and Comparative Evaluation for Tweet Classification. Proceedings of COLING 2022. https://huggingface.co/cardiffnlp/twitter-roberta-base-dec2021-tweet-topic-multi-all
-   Levy Abitbol, J.; Morales, A.J. (2021). Socioeconomic Patterns of Twitter User Activity. Entropy, 23, 780. https://doi.org/10.3390/e23060780
-   Gibbons J, et al. (2019). Twitter-based measures of neighborhood sentiment as predictors of residential population health. PLoS ONE, 14(7): e0219550. https://doi.org/10.1371/journal.pone.0219550
-   Zimbra, D., et al. (2018). The State-of-the-Art in Twitter Sentiment Analysis. ACM Trans. Manage. Inf. Syst., 9, 2, Article 5. https://doi.org/10.1145/3185045
-   Liu, Y., et al. (2019). RoBERTa: A Robustly Optimized BERT Pretraining Approach. arXiv preprint arXiv:1907.11692. https://arxiv.org/abs/1907.11692
-   Nagpal, M. (2024). How to use an LLM for Sentiment Analysis? ProjectPro. https://www.projectpro.io/article/llm-sentiment-analysis/1125
        """)

if __name__ == '__main__':
    main()



