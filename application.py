import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import gc
import joblib
import re
import logging
import tempfile
import gdown
from pathlib import Path
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set up basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler('app_log.log', mode='w', encoding='utf-8')]
)
logger = logging.getLogger(__name__)

# -------------------- CONFIG --------------------
DATA_DIR = Path(tempfile.gettempdir()) / "app_data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

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

# -------------------- HELPER FUNCTIONS --------------------
def download_from_drive(file_id, output_path: Path):
    """Downloads a file from Google Drive if it doesn't exist."""
    if not output_path.exists() or output_path.stat().st_size == 0:
        try:
            gdown.download(id=file_id, output=str(output_path), quiet=True)
            if output_path.exists() and output_path.stat().st_size > 0:
                logger.info(f"Downloaded {output_path.name}")
            else:
                raise ValueError(f"Download failed: {output_path.name} is empty or missing.")
        except Exception as e:
            logger.error(f"Failed to download {output_path.name}: {e}")
            raise RuntimeError(f"Download failed for {output_path.name}") from e

@st.cache_resource
def load_all_models_cached():
    """Loads all models and vectorizers from Google Drive."""
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
        logger.info("Loaded all ML resources.")
        return sentiment_model, sentiment_vectorizer, emotion_model, emotion_vectorizer
    except Exception as e:
        logger.error(f"Failed to load ML resources: {e}")
        return None, None, None, None

# Simplified tweet cleaner
class MemoryOptimizedTweetCleaner:
    def __init__(self):
        self.patterns = {
            'web_urls': re.compile(r'https?://\S+|www\.\S+', re.IGNORECASE),
            'mentions': re.compile(r'@\w+'),
            'hashtags': re.compile(r'#(\w+)'),
            'rt_prefix': re.compile(r'^RT\s*', re.IGNORECASE),
            'punctuation': re.compile(r'[^\w\s]+'),
            'whitespace': re.compile(r'\s+'),
        }

    def clean_text(self, text: str) -> str:
        if pd.isna(text) or not text or not str(text).strip():
            return ""
        try:
            text = str(text)
            text = self.patterns['web_urls'].sub('', text)
            text = self.patterns['rt_prefix'].sub('', text)
            text = self.patterns['mentions'].sub('', text)
            text = self.patterns['hashtags'].sub(r'\1', text)
            text = self.patterns['punctuation'].sub(' ', text)
            text = self.patterns['whitespace'].sub(' ', text).strip()
            return text if text else ""
        except Exception as e:
            logger.debug(f"Error cleaning text: {e}")
            return ""

@st.cache_data
def load_data(dataset_key: str) -> pd.DataFrame:
    file_info = DATASET_FILES[dataset_key]
    path = DATA_DIR / file_info["name"]
    try:
        download_from_drive(file_info["id"], path)
        df = pd.read_csv(path, low_memory=False)
        df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]
        logger.info(f"Loaded dataset '{dataset_key}' with {len(df)} rows")
        if df.empty:
            st.error(f"Dataset '{dataset_key}' is empty. Check file content.")
            return pd.DataFrame()
        return df
    except Exception as e:
        logger.error(f"Error loading dataset '{dataset_key}': {e}")
        st.error(f"Failed to load dataset '{dataset_key}': {e}")
        return pd.DataFrame()

@st.cache_data
def load_incident_data(incident_key: str) -> pd.DataFrame:
    file_info = INCIDENT_FILES[incident_key]
    path = DATA_DIR / file_info["name"]
    try:
        download_from_drive(file_info["id"], path)
        incident_df = pd.read_csv(path, low_memory=False)
        if 'Incident Zip' in incident_df.columns:
            incident_df['Incident Zip'] = incident_df['Incident Zip'].astype(str).str.zfill(5)
        logger.info(f"Loaded incident data '{incident_key}' with {len(incident_df)} rows")
        if incident_df.empty:
            st.error(f"Incident dataset '{incident_key}' is empty. Check file content.")
            return pd.DataFrame()
        return incident_df
    except Exception as e:
        logger.error(f"Error loading incident data '{incident_key}': {e}")
        st.error(f"Failed to load incident data '{incident_key}': {e}")
        return pd.DataFrame()

# -------------------- VISUALIZATION FUNCTION --------------------
plt.style.use('seaborn-v0_8')
sns.set_context('talk', font_scale=1.2)

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
    st.markdown("**Explanation:** This pie chart shows the distribution of tweet sentiments.")

# -------------------- PREDICTION PAGE --------------------
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
    st.markdown("Analyze Twitter data for sentiment insights.")

    if 'df' not in st.session_state:
        st.session_state['df'] = None
    if 'incident_df' not in st.session_state:
        st.session_state['incident_df'] = None
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
        if st.button("Clear Cache"):
            st.session_state['df'] = None
            st.session_state['incident_df'] = None
            st.session_state['current_dataset_choice'] = None
            st.cache_data.clear()
            gc.collect()
            st.success("Cache cleared!")
    
    with col2:
        if st.button("Load Dataset", disabled=not dataset_choice):
            if dataset_choice != st.session_state.get('current_dataset_choice'):
                st.session_state['current_dataset_choice'] = dataset_choice
                st.session_state['df'] = None
                st.session_state['incident_df'] = None
                st.cache_data.clear()
                gc.collect()
            
            try:
                with st.spinner(f"Loading dataset '{dataset_choice}'..."):
                    st.session_state['df'] = load_data(dataset_choice)
                    st.session_state['incident_df'] = load_incident_data(dataset_choice)
                st.write(f"**Dataset Info ({dataset_choice}):**")
                st.write(f"Rows: {st.session_state['df'].shape[0]:,}")
                st.write(f"Columns: {list(st.session_state['df'].columns)}")
                st.write(f"**Incident Dataset Info ({dataset_choice}):**")
                st.write(f"Rows: {st.session_state['incident_df'].shape[0]:,}")
                st.write(f"Columns: {list(st.session_state['incident_df'].columns)}")
                if st.session_state['df'].empty or st.session_state['incident_df'].empty:
                    st.error("One or both datasets are empty. Check Google Drive file content.")
                else:
                    st.success(f"Data for '{dataset_choice}' loaded!")
            except Exception as e:
                st.error(f"Failed to load dataset '{dataset_choice}': {e}")
                st.session_state['df'] = None
                st.session_state['incident_df'] = None

    if st.session_state['df'] is not None and not st.session_state['df'].empty:
        st.sidebar.markdown("---")
        visualization_choice = st.sidebar.radio(
            "Choose a visualization:", 
            ["📈 Sentiment Pie Chart"]
        )
        st.header(f"Visualizing: {st.session_state['current_dataset_choice']}")
        if visualization_choice == "📈 Sentiment Pie Chart":
            sentiment_pie_chart(st.session_state['df'])

if __name__ == '__main__':
    sentiment_model, sentiment_vectorizer, emotion_model, emotion_vectorizer = load_all_models_cached()
    
    if sentiment_model is None:
        st.error("Failed to load machine learning models. Check internet or file IDs.")
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
