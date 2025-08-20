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
import folium
from streamlit_folium import folium_static

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
SHAPEFILE_PATH = SHAPE_DIR / "tl_2020_us_zcta510.shp"

# Create directories if they don't exist
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

def download_from_drive(file_id, output_path: Path, progress_bar=None, message="Downloading..."):
    """Downloads a file from Google Drive if it doesn't exist, with an optional progress bar."""
    if not output_path.exists() or output_path.stat().st_size == 0:
        st.info(f"{message}")
        with st.spinner(f"Downloading {output_path.name}... This might take a while."):
            try:
                # Use gdown's built-in progress bar
                gdown.download(id=file_id, output=str(output_path), quiet=False)
                if output_path.exists() and output_path.stat().st_size > 0:
                    st.success(f"Successfully downloaded {output_path.name}")
                    logger.info(f"Successfully downloaded {output_path.name}")
                else:
                    raise ValueError(f"Download failed: {output_path.name} is empty or missing.")
            except Exception as e:
                st.error(f"Failed to download {output_path.name}: {e}")
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

@st.cache_data
def load_data(dataset_key):
    st.info("Loading tweet data...")
    file_info = DATASET_FILES[dataset_key]
    path = DATA_DIR / file_info["name"]
    download_from_drive(file_info["id"], path)
    
    with st.spinner("Reading data into DataFrame..."):
        df = pd.read_csv(path, low_memory=False)

    df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]
    
    # Check for and drop rows with missing location data
    if 'latitude' in df.columns and 'longitude' in df.columns:
        df = df.dropna(subset=['latitude', 'longitude'])
        df = df[(df['latitude'].between(-90, 90)) & (df['longitude'].between(-180, 180))]
    
    # Convert date column
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    st.success("Tweet data loaded successfully!")
    return df

@st.cache_data
def load_incident_data(incident_key):
    st.info("Loading incident data...")
    file_info = INCIDENT_FILES[incident_key]
    path = DATA_DIR / file_info["name"]
    download_from_drive(file_info["id"], path)
    
    with st.spinner("Reading incident data into DataFrame..."):
        incident_df = pd.read_csv(path, low_memory=False)
        
    incident_df['Incident Zip'] = incident_df['Incident Zip'].astype(str).str.zfill(5)
    st.success("Incident data loaded successfully!")
    return incident_df

@st.cache_data
def load_shapefile():
    with st.spinner("Preparing geographic data for maps..."):
        for i, sf in enumerate(SHAPE_FILES):
            path = SHAPE_DIR / sf["name"]
            download_from_drive(sf["id"], path, message=f"Downloading geographic file {i+1}/{len(SHAPE_FILES)}: {sf['name']}")
    
    try:
        st.info("Loading and processing shapefile...")
        with st.spinner("This may take a moment due to file size..."):
            zcta_gdf = gpd.read_file(SHAPEFILE_PATH)
            
        zcta_gdf['ZCTA5CE10'] = zcta_gdf['ZCTA5CE10'].astype(str).str.zfill(5)
        nyc_zip_prefixes = ('100', '101', '102', '103', '104', '111', '112', '113', '114', '116')
        
        st.info("Filtering for NYC zip codes...")
        nyc_gdf = zcta_gdf[zcta_gdf['ZCTA5CE10'].str.startswith(nyc_zip_prefixes)].copy()
        
        del zcta_gdf
        gc.collect()
        
        st.success("Geographic data loaded successfully!")
        return nyc_gdf
    except Exception as e:
        logger.error(f"Failed to load geographic data: {e}", exc_info=True)
        st.error(f"An error occurred while loading geographic data: {e}")
        st.warning("Please check the data files and try again.")
        st.stop()
    
# =========================
# VISUALIZATION FUNCTIONS
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
        x='longitude',
        y='latitude',
        hue='category',
        palette={1: '#ff9999', 0: '#66b3ff', -1: '#99ff99'},
        data=df,
        alpha=0.6,
        ax=ax
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
    st.markdown("Summary: This heatmap shows the relationship between different emotions and sentiment categories.")

def zip_code_maps(incident_df, nyc_gdf):
    st.subheader("NYC Zip Code Incident Maps")
    if incident_df.empty or 'Incident Zip' not in incident_df.columns:
        st.warning("Incident data is not available or missing 'Incident Zip' column.")
        return
    
    incident_counts = incident_df.groupby('Incident Zip').size().reset_index(name='count')
    merged_gdf = nyc_gdf.merge(incident_counts, left_on='ZCTA5CE10', right_on='Incident Zip', how='left')
    merged_gdf['count'] = merged_gdf['count'].fillna(0)
    
    # Create the map
    m = folium.Map(location=[40.7128, -74.0060], zoom_start=11, tiles="cartodbpositron")

    folium.Choropleth(
        geo_data=merged_gdf,
        data=merged_gdf,
        columns=['ZCTA5CE10', 'count'],
        key_on='feature.properties.ZCTA5CE10',
        fill_color='YlOrRd',
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name='Number of Incidents by ZIP Code',
        highlight=True
    ).add_to(m)
    
    style_function = lambda x: {'fillColor': '#ffffff', 'color':'#000000', 'fillOpacity': 0.1, 'weight': 0.1}
    highlight_function = lambda x: {'fillColor': '#000000', 'color':'#000000', 'fillOpacity': 0.50, 'weight': 0.1}
    
    tooltip_style = """
        background-color: white;
        border: 2px solid grey;
        border-radius: 3px;
        box-shadow: 3px;
    """
    
    def style_and_tooltip(feature):
        return {
            'tooltip': folium.Tooltip(
                f"ZIP: {feature['properties']['ZCTA5CE10']}<br>Incidents: {int(feature['properties']['count'])}",
                style=tooltip_style
            )
        }
    
    # Add tooltips to the map
    folium.GeoJson(
        merged_gdf,
        style_function=style_function,
        highlight_function=highlight_function,
        tooltip=folium.GeoJsonTooltip(
            fields=['ZCTA5CE10', 'count'],
            aliases=['ZIP Code', 'Incidents'],
            localize=True,
            sticky=False
        )
    ).add_to(m)
    
    folium_static(m)

    st.markdown("Summary: This map shows the geographical distribution of incidents reported in NYC zip codes, "
                "with darker shades indicating a higher number of incidents.")

def zip_code_heatmap(incident_df, nyc_gdf):
    st.subheader("NYC Zip Code Incident Heatmap")
    if incident_df.empty or 'latitude' not in incident_df.columns or 'longitude' not in incident_df.columns:
        st.warning("Incident data is not available or missing location columns.")
        return
    
    from folium.plugins import HeatMap
    
    st.info("Generating heatmap...")
    # Reduce data points if necessary for performance
    if len(incident_df) > MAX_POINTS_MAP:
        incident_df_small = incident_df.sample(MAX_POINTS_MAP, random_state=42)
        st.warning(f"Using a sample of {MAX_POINTS_MAP} points for the heatmap for better performance.")
    else:
        incident_df_small = incident_df
        
    m = folium.Map([40.7128, -74.0060], zoom_start=11)
    
    # Check if there's any data to plot
    if not incident_df_small[['latitude', 'longitude']].empty:
        heatmap_data = incident_df_small[['latitude', 'longitude']].values.tolist()
        HeatMap(heatmap_data, radius=15).add_to(m)
    else:
        st.warning("No location data to display on the heatmap.")
        
    folium_static(m)
    st.markdown("Summary: This heatmap visualizes the density of incidents based on their geographical coordinates.")


def borough_income_chart(df):
    st.subheader("Average Median Income by Borough")
    if 'borough' not in df.columns or 'median_income' not in df.columns:
        st.warning("Borough or Median Income data not available for this visualization.")
        return
    
    borough_income = df.groupby('borough')['median_income'].mean().sort_values(ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    borough_income.plot(kind='bar', color=sns.color_palette("viridis", len(borough_income)), ax=ax)
    
    ax.set_title("Average Median Income by Borough", fontsize=18)
    ax.set_xlabel("Borough", fontsize=14)
    ax.set_ylabel("Average Median Income", fontsize=14)
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("Summary: This chart compares the average median income across different boroughs based on tweet locations.")

# Main app logic
def project_report_page():
    st.title("Project Report: Twitter Sentiment Analysis")
    st.markdown("""
    ### Introduction
    This project is a comprehensive Streamlit application designed to analyze and visualize sentiment and emotion from Twitter data. It leverages machine learning models to classify tweets into sentiment categories (positive, neutral, negative) and emotions (e.g., happiness, sadness, anger). The application provides an interactive dashboard with various visualizations to explore the dataset's characteristics, including sentiment trends, emotion distributions, and geographical patterns.

    ### Data Sources
    The application uses a combination of data sources:
    - **Twitter Data:** CSV files containing tweets related to specific topics (e.g., COVID-19, Economics, Politics) with pre-classified sentiment and emotion labels.
    - **Geographic Data:** A US Census Bureau shapefile (`tl_2020_us_zcta510.shp`) is used to map sentiment and incident data to specific zip codes.
    - **Machine Learning Models:** Pre-trained models and vectorizers are stored in pickled files (`.pkl`) for sentiment and emotion classification.

    ### Key Features
    - **Dynamic Dashboard:** Users can select different topics and visualizations to explore the data in real-time.
    - **Interactive Visualizations:** The app includes charts and maps to represent data in an intuitive way. This includes pie charts for emotion distribution, bar charts for emotion-sentiment correlation, and geographical maps for sentiment and incident density.
    - **Combined Prediction Tool:** A dedicated section allows users to input custom text and receive instant sentiment and emotion predictions from the loaded models.
    - **Performance Optimization:** The application uses Streamlit's `@st.cache_data` and `@st.cache_resource` decorators to efficiently cache data and models, preventing redundant downloads and processing.

    ### Data Processing & Cleaning
    The application includes a `MemoryOptimizedTweetCleaner` class to preprocess raw tweet text. This class performs several cleaning steps:
    - Removal of URLs, mentions, and hashtags.
    - Translation of emoticons into descriptive text (e.g., ":)" becomes "smiling_face").
    - Removal of punctuation, extra whitespace, and single characters.
    This ensures that the text data is in a clean format suitable for model prediction.

    ### Challenges & Solutions
    - **Large File Handling:** Geographic data files are large and can cause memory issues. The solution was to use Streamlit's caching, and to immediately filter the large GeoDataFrame to only include relevant NYC zip codes, and then explicitly delete the original large dataframe to free up memory (`del zcta_gdf`).
    - **Model Loading:** Loading multiple large machine learning models can be slow. `@st.cache_resource` is used to load all models once and keep them in memory for the duration of the app session.
    - **User Experience:** Streamlit's native widgets and layouts were utilized to create a user-friendly interface. Progress bars and loading spinners were added to provide feedback during time-consuming operations like file downloads.

    ### Conclusion
    This project provides a robust and interactive tool for analyzing Twitter data. It demonstrates a practical application of machine learning, data visualization, and web development principles. The optimizations for data handling and caching ensure a smooth user experience, even with large datasets. The modular design makes it easy to extend with new topics, models, or visualizations in the future.
    """)

def dashboard_page():
    st.title("📊 Sentiment Analysis Dashboard")
    st.markdown("Explore Twitter sentiment and emotion data with interactive visualizations.")
    st.sidebar.header("Dashboard Filters")
    
    topic_choice = st.sidebar.selectbox("Select Topic:", list(DATASET_FILES.keys()), index=0)

    # State management for data loading
    # This block ensures all dataframes are loaded into session state at the beginning
    # of the dashboard page, preventing the AttributeError.
    if 'df' not in st.session_state or st.session_state.get('df_key') != topic_choice:
        st.session_state.df = load_data(topic_choice)
        st.session_state.df_key = topic_choice
    
    if 'incident_df' not in st.session_state or st.session_state.get('df_key') != topic_choice:
        st.session_state.incident_df = load_incident_data(topic_choice)
    
    if 'nyc_gdf' not in st.session_state:
        st.session_state.nyc_gdf = load_shapefile()

    st.sidebar.markdown("---")
    visualization_choice = st.sidebar.selectbox(
        "Select Visualization:",
        ["📈 Sentiment Trends Over Time", 
         "🗺 ZIP Code Sentiment Maps",
         "🗺 ZIP Code Sentiment Heatmap",
         "📍 Geographical Distribution of Tweet Sentiments",
         "📊 Distribution of Median Income",
         "💰 Average Median Income by Borough",
         "🎉 Top 5 Emotions Pie Chart",
         "📊 Emotion vs Sentiment Bar Chart",
         "📦 Emotion Confidence Box Plot",
         "🔥 Emotion vs Sentiment Heatmap"
        ]
    )

    df = st.session_state.df
    incident_df = st.session_state.incident_df
    nyc_gdf = st.session_state.nyc_gdf

    st.subheader(f"Analyzing: {topic_choice} Tweets")

    if visualization_choice == "📈 Sentiment Trends Over Time":
        sentiment_trends_line_chart(df)
    elif visualization_choice == "🎉 Top 5 Emotions Pie Chart":
        top_emotions_pie_chart(df)
    elif visualization_choice == "📊 Emotion vs Sentiment Bar Chart":
        emotion_sentiment_bar_chart(df)
    elif visualization_choice == "📦 Emotion Confidence Box Plot":
        emotion_confidence_boxplot(df)
    elif visualization_choice == "📍 Geographical Distribution of Tweet Sentiments":
        geo_sentiment_scatterplot(df)
    elif visualization_choice == "📊 Distribution of Median Income":
        median_income_histogram(df)
    elif visualization_choice == "💰 Average Median Income by Borough":
        borough_income_chart(df)
    elif visualization_choice == "🔥 Emotion vs Sentiment Heatmap":
        emotion_sentiment_heatmap(df)
    elif visualization_choice == "🗺 ZIP Code Sentiment Maps":
        zip_code_maps(incident_df, nyc_gdf)
    elif visualization_choice == "🗺 ZIP Code Sentiment Heatmap":
        zip_code_heatmap(incident_df, nyc_gdf)


def combined_prediction_page(sentiment_model, sentiment_vectorizer, emotion_model, emotion_vectorizer):
    st.title("🔍 Combined Prediction")
    st.markdown("Enter a short text and see its sentiment and emotion predictions instantly.")
    
    if any(m is None for m in [sentiment_model, sentiment_vectorizer, emotion_model, emotion_vectorizer]):
        st.error("Models failed to load. Please restart the app or check your internet connection.")
        return

    text_input = st.text_area("Enter your text here:", height=100)
    
    cleaner = MemoryOptimizedTweetCleaner()
    
    if st.button("Predict"):
        if text_input:
            with st.spinner("Classifying..."):
                cleaned_text = cleaner.clean_text(text_input)
                
                if cleaned_text:
                    # Sentiment prediction
                    sentiment_features = sentiment_vectorizer.transform([cleaned_text])
                    sentiment_prediction = sentiment_model.predict(sentiment_features)[0]
                    sentiment_category = "Positive" if sentiment_prediction == 1 else "Negative" if sentiment_prediction == -1 else "Neutral"
                    
                    # Emotion prediction
                    emotion_features = emotion_vectorizer.transform([cleaned_text])
                    emotion_prediction = emotion_model.predict(emotion_features)[0]

                    st.markdown("---")
                    st.subheader("Results:")
                    st.metric(label="Predicted Sentiment", value=sentiment_category)
                    st.metric(label="Predicted Emotion", value=emotion_prediction)
                else:
                    st.warning("Please enter some text to analyze.")
        else:
            st.warning("Please enter some text to analyze.")

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
        
    if page == "📄 Project Report":
        project_report_page()
    elif page == "📊 Sentiment Analysis Dashboard":
        dashboard_page()
    elif page == "🔍 Combined Prediction":
        combined_prediction_page(sentiment_model, sentiment_vectorizer, emotion_model, emotion_vectorizer)
