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
SHAPEFILE_PATH = SHAPE_DIR / "tl_2020_us_zcta510.shp"
NYC_ZIPS = {
    '10001', '10002', '10003', '10004', '10005', '10006', '10007', '10008', '10009', '10010',
    '10011', '10012', '10013', '10014', '10016', '10017', '10018', '10019', '10020', '10021',
    '10022', '10023', '10024', '10025', '10026', '10027', '10028', '10029', '10030', '10031',
    '10032', '10033', '10034', '10035', '10036', '10037', '10038', '10039', '10040', '10041',
    '10044', '10048', '10055', '10065', '10069', '10075', '10103', '10104', '10105', '10106',
    '10107', '10110', '10111', '10112', '10115', '10118', '10119', '10120', '10121', '10122',
    '10128', '10152', '10153', '10154', '10162', '10165', '10166', '10167', '10168', '10169',
    '10170', '10171', '10172', '10173', '10174', '10175', '10176', '10177', '10178', '10199',
    '10270', '10271', '10278', '10279', '10280', '10281', '10282', '10301', '10302', '10303',
    '10304', '10305', '10306', '10307', '10308', '10309', '10310', '10312', '10314', '10451',
    '10452', '10453', '10454', '10455', '10456', '10457', '10458', '10459', '10460', '10461',
    '10462', '10463', '10464', '10465', '10466', '10467', '10468', '10469', '10470', '10471',
    '10472', '10473', '10474', '10475', '11004', '11005', '11101', '11102', '11103', '11104',
    '11105', '11106', '11109', '11201', '11203', '11204', '11205', '11206', '11207', '11208',
    '11209', '11210', '11211', '11212', '11213', '11214', '11215', '11216', '11217', '11218',
    '11219', '11220', '11221', '11222', '11223', '11224', '11225', '11226', '11228', '11229',
    '11230', '11231', '11232', '11233', '11234', '11235', '11236', '11237', '11238', '11239',
    '11249', '11252', '11354', '11355', '11356', '11357', '11358', '11359', '11360', '11361',
    '11362', '11363', '11364', '11365', '11366', '11367', '11368', '11369', '11370', '11371',
    '11372', '11373', '11374', '11375', '11377', '11378', '11379', '11385', '11411', '11412',
    '11413', '11414', '11415', '11416', '11417', '11418', '11419', '11420', '11421', '11422',
    '11423', '11426', '11427', '11428', '11429', '11430', '11432', '11433', '11434', '11435',
    '11436', '11451', '11453', '11456', '11520', '11691', '11692', '11693', '11694', '11695',
    '11697', '12601', '12720', '12721', '12729', '12733', '12741', '12746', '12750', '12751',
    '12759', '12771', '12778', '12788', '12789', '12790'
}

# Mapping of topic choices to data filenames
TOPIC_DATA_MAP = {
    "COVID-19": {
        'df': 'sample_twitter_data_covid_classified.csv',
        'incident_df': 'Incident Zip_covid_classified.csv'
    },
    "Politics": {
        'df': 'sample_twitter_data_politics_classified.csv',
        'incident_df': 'Incident Zip_politics_classified.csv'
    },
    "Economics": {
        'df': 'sample_twitter_data_economics_classified.csv',
        'incident_df': 'Incident Zip_economics_classified.csv'
    }
}

# -------------------- UTILITY FUNCTIONS --------------------

@st.cache_resource(show_spinner=False)
def get_file_content(file_id: str, file_name: str) -> Path:
    """
    Downloads a file from Google Drive if it doesn't already exist.
    """
    filepath = DATA_DIR / file_name
    # Create the directory if it doesn't exist
    filepath.parent.mkdir(parents=True, exist_ok=True)
    if not filepath.exists():
        logger.info(f"Downloading {file_name}...")
        try:
            gdown.download(id=file_id, output=str(filepath), quiet=False)
            logger.info(f"Successfully downloaded {file_name}")
        except Exception as e:
            logger.error(f"Failed to download {file_name}: {e}")
            st.error(f"Failed to download {file_name}. Please check the file ID or your internet connection.")
            return None
    return filepath

def load_data(topic_choice: str) -> Optional[pd.DataFrame]:
    """
    Loads and caches the main dataset based on the topic choice.
    """
    data_file = TOPIC_DATA_MAP.get(topic_choice, {}).get('df')
    if data_file:
        data_filepath = DATA_DIR / data_file
        # Check if file exists, if not, try to download from a list of predefined IDs
        if not data_filepath.exists():
            st.error("Data file not found locally. Trying to download...")
            file_ids = {
                "COVID-19": "1KhQvyglx07Lx4hD971956IikHhZKcczS",
                "Politics": "18Q9ORlDfoIQW_-RpwJy2qGrFWEA8dFQU",
                "Economics": "1atpQuelcBriYSINjSjP5d4_AK4EmpV5J"
            }
            file_id = file_ids.get(topic_choice)
            if file_id:
                get_file_content(file_id, data_file)
            else:
                st.error(f"No download ID found for topic: {topic_choice}")
                return None

        if data_filepath.exists():
            try:
                # Use a specific, consistent method to read the data
                df = pd.read_csv(data_filepath)
                # Pre-processing steps
                df['created_at'] = pd.to_datetime(df['created_at'])
                df['month_year'] = df['created_at'].dt.to_period('M')
                # Clean up and normalize text
                df['text'] = df['text'].apply(lambda x: unicodedata.normalize('NFKD', x).encode('ascii', 'ignore').decode('utf-8'))
                # Handle missing 'Incident Zip' and convert to string
                df['Incident Zip'] = df['Incident Zip'].fillna('').astype(str)
                # Clean up zip codes to only include valid 5-digit ones
                df['Incident Zip'] = df['Incident Zip'].apply(lambda x: x if re.fullmatch(r'\d{5}', x) else None)
                return df
            except Exception as e:
                logger.error(f"Error loading {data_file}: {e}")
                st.error(f"Error loading data for {topic_choice}. Please check the file format.")
                return None
    return None

def load_incident_df(topic_choice: str) -> Optional[pd.DataFrame]:
    """
    Loads and caches the incident DataFrame based on the topic choice.
    """
    incident_file = TOPIC_DATA_MAP.get(topic_choice, {}).get('incident_df')
    if incident_file:
        incident_filepath = DATA_DIR / incident_file
        if not incident_filepath.exists():
            st.error("Incident data file not found locally. Trying to download...")
            file_ids = {
                "COVID-19": "1IbIfdrAU3ZYue5joLojPisRX3JJjwdvM",
                "Politics": "1uNxIYzSY7cbgbuTc8zo0QYc5Dn5Ny2W_",
                "Economics": "1SNpGyEHgrOe6ihx26vo38hf_zcxEnp64"
            }
            file_id = file_ids.get(topic_choice)
            if file_id:
                get_file_content(file_id, incident_file)
            else:
                st.error(f"No download ID found for incident data for topic: {topic_choice}")
                return None
        
        if incident_filepath.exists():
            try:
                incident_df = pd.read_csv(incident_filepath)
                return incident_df
            except Exception as e:
                logger.error(f"Error loading incident data {incident_file}: {e}")
                st.error(f"Error loading incident data for {topic_choice}. Please check the file format.")
                return None
    return None

@st.cache_resource
def load_all_models_cached():
    """
    Loads all ML models and vectorizers using caching.
    """
    try:
        # File IDs for the models and vectorizers on Google Drive
        model_ids = {
            'sentiment_model': '1lzZf79LGcB1J5SQsMh_mi1Jv_8V2q6K9',
            'sentiment_vectorizer': '12wRG57vERpdKgCaTiNyLiWC71KLWABK8',
            'emotion_model': '1NZCZKMhTSKvFJMuW_kxmX1OrkEQLFgs0',
            'emotion_vectorizer': '1TKR2xmNcouAb8XyQz6VANixFLNZ9YsJV'
        }

        # Download files if they don't exist locally
        for key, file_id in model_ids.items():
            file_name = f"{key}_large.pkl"
            filepath = DATA_DIR / file_name
            if not filepath.exists():
                get_file_content(file_id, file_name)

        # Load the models and vectorizers
        sentiment_model = joblib.load(DATA_DIR / 'sentiment_model_large.pkl')
        sentiment_vectorizer = joblib.load(DATA_DIR / 'vectorizer_large.pkl')
        emotion_model = joblib.load(DATA_DIR / 'emotion_model_large.pkl')
        emotion_vectorizer = joblib.load(DATA_DIR / 'emotion_vectorizer_large.pkl')

        logger.info("Successfully loaded all ML resources.")
        return sentiment_model, sentiment_vectorizer, emotion_model, emotion_vectorizer
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        st.error(f"Error loading machine learning models: {e}")
        return None, None, None, None

@st.cache_resource
def load_shapefile() -> gpd.GeoDataFrame:
    """
    Loads and filters the NYC zip code shapefile.
    """
    logger.info("Loading pre-filtered NYC shapefile.")
    shapefile_ids = {
        'shp': '1AwweXBE1Xq8_byFKfT61AaFjhNWlaftI',
        'shx': '1RBysVHjdW4bIrL5WzhmOUKNhi7xUuET3',
        'dbf': '1YJLqeYapj9kpoLwf0KxFWG1dwRzchoq5',
        'prj': '1BzAH0_f-jgglD5tt0DImR2iOtdqfEhWf',
        'cpg': '1Un6VtqVE45qhanvqqwdywIiZV7K3qVi6',
        'shp.xml': '1UkQSMun9auyZqHd9cy6NPxa387At79Rp',
        'shp.iso.xml': '19vKnJ0RH5wr8GZFeuNbJSpaABo5x3ukJ'
    }

    # Ensure shapefile directory exists
    SHAPE_DIR.mkdir(parents=True, exist_ok=True)

    # Download all necessary shapefile components
    for ext, file_id in shapefile_ids.items():
        file_name = f"tl_2020_us_zcta510.{ext}"
        filepath = SHAPE_DIR / file_name
        if not filepath.exists():
            get_file_content(file_id, file_name)

    try:
        # Load the shapefile
        gdf = gpd.read_file(SHAPEFILE_PATH)
        # Filter for NYC zip codes
        nyc_gdf = gdf[gdf['ZCTA5CE10'].isin(NYC_ZIPS)].copy()
        return nyc_gdf
    except Exception as e:
        logger.error(f"Error loading shapefile: {e}")
        st.error("Error loading geographical data for maps.")
        return None


def get_memory_usage() -> str:
    """
    Returns the current process memory usage in MB.
    """
    process = psutil.Process(os.getpid())
    return f"{process.memory_info().rss / 1024 / 1024:.2f} MB"

def create_pie_chart(df: pd.DataFrame, title: str):
    """
    Creates a pie chart from a DataFrame.
    """
    st.subheader(title)
    
    # Check if the DataFrame is empty before plotting
    if df.empty:
        st.info("No data available to display the pie chart.")
        return

    # Assuming the df has a 'sentiment' column
    sentiment_counts = df['sentiment'].value_counts()
    
    # Check if there are counts to plot
    if sentiment_counts.empty:
        st.info("No sentiment data found to display the pie chart.")
        return

    fig = px.pie(
        names=sentiment_counts.index,
        values=sentiment_counts.values,
        title=title,
        color_discrete_sequence=['red', 'green', 'blue']
    )
    st.plotly_chart(fig)
    gc.collect()

def create_bar_chart(df: pd.DataFrame, title: str, x_col: str, y_col: str):
    """
    Creates a bar chart from a DataFrame.
    """
    st.subheader(title)
    fig = px.bar(df, x=x_col, y=y_col, title=title)
    st.plotly_chart(fig)
    gc.collect()

def create_line_chart(df: pd.DataFrame, title: str, x_col: str, y_col: str, color_col: str):
    """
    Creates a line chart from a DataFrame.
    """
    st.subheader(title)
    fig = px.line(df, x=x_col, y=y_col, color=color_col, title=title)
    st.plotly_chart(fig)
    gc.collect()

def create_scatter_chart(df: pd.DataFrame, title: str):
    """
    Creates a scatter chart from a DataFrame.
    """
    st.subheader(title)
    if df.empty:
        st.info("No data available to display the scatter chart.")
        return
        
    df_sample = df.sample(min(len(df), MAX_POINTS_SCATTER))
    
    if not {'sentiment', 'emotion'}.issubset(df_sample.columns):
        st.error("DataFrame must contain 'sentiment' and 'emotion' columns for this chart.")
        return

    fig = px.scatter(df_sample, x='sentiment', y='emotion', title=title, color='sentiment')
    st.plotly_chart(fig)
    gc.collect()

def sentiment_emotion_correlation(df: pd.DataFrame):
    """
    Analyzes the correlation between sentiment and emotion.
    """
    st.title("Sentiment and Emotion Correlation Analysis")
    st.write("This section explores the relationship between the sentiment and emotion classifications.")
    
    if df.empty:
        st.info("Please select a topic and analyze the data to see this chart.")
        return

    if not {'sentiment', 'emotion'}.issubset(df.columns):
        st.error("The selected dataset does not contain both 'sentiment' and 'emotion' classifications.")
        return

    # Calculate the cross-tabulation of sentiment and emotion
    crosstab = pd.crosstab(df['sentiment'], df['emotion'])
    st.subheader("Sentiment vs. Emotion Cross-Tabulation")
    st.dataframe(crosstab)

    # Plot a stacked bar chart
    fig = px.bar(
        crosstab,
        x=crosstab.index,
        y=crosstab.columns,
        title='Sentiment Distribution by Emotion',
        labels={'value': 'Number of Tweets', 'x': 'Sentiment', 'color': 'Emotion'},
        color_discrete_map={
            'joy': 'gold', 'anger': 'darkred', 'surprise': 'purple', 'sadness': 'darkblue',
            'fear': 'gray', 'disgust': 'darkgreen'
        }
    )
    st.plotly_chart(fig)

    # Optionally, a grouped bar chart
    fig2 = px.bar(
        crosstab.T,
        barmode='group',
        title='Emotion Distribution by Sentiment',
        labels={'value': 'Number of Tweets', 'x': 'Emotion', 'color': 'Sentiment'},
        color_discrete_map={'positive': 'green', 'negative': 'red', 'neutral': 'blue'}
    )
    st.plotly_chart(fig2)
    gc.collect()

def emotion_sentiment_heatmap(df: pd.DataFrame):
    """
    Generates a heatmap of emotion vs sentiment.
    """
    st.title("Emotion vs Sentiment Heatmap")
    st.write("This heatmap visualizes the co-occurrence of emotions and sentiments.")
    
    if df.empty:
        st.info("Please select a topic and analyze the data to see this chart.")
        return

    if not {'sentiment', 'emotion'}.issubset(df.columns):
        st.error("The selected dataset does not contain both 'sentiment' and 'emotion' classifications.")
        return
        
    crosstab = pd.crosstab(df['emotion'], df['sentiment'], normalize='index')
    
    # Create the heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(crosstab, annot=True, cmap='YlGnBu', fmt=".2f", ax=ax)
    ax.set_title('Normalized Heatmap of Emotion by Sentiment')
    ax.set_xlabel('Sentiment')
    ax.set_ylabel('Emotion')
    st.pyplot(fig)
    plt.close(fig)
    gc.collect()

def zip_code_maps(incident_df: pd.DataFrame):
    """
    Generates an interactive map of NYC zip codes with sentiment data.
    """
    st.title("ZIP Code Sentiment Maps")
    st.write("This map shows the geographical distribution of sentiment across NYC zip codes.")
    
    # The function now handles the data loading internally
    nyc_gdf = load_shapefile()

    # Check for empty dataframes
    if incident_df is None or incident_df.empty:
        st.info("No incident data found for the selected topic. Please select a topic from the sidebar.")
        return
    if nyc_gdf is None or nyc_gdf.empty:
        st.error("Failed to load NYC geographical data.")
        return
        
    # Aggregate sentiment counts per zip code
    incident_sums = incident_df.groupby('Incident Zip')[['negative', 'positive', 'neutral']].sum().reset_index()
    incident_sums['total'] = incident_sums[['negative', 'positive', 'neutral']].sum(axis=1)

    for col in ['negative', 'positive', 'neutral']:
        incident_sums[col + '_pct'] = (incident_sums[col] / incident_sums['total'] * 100).round(2)
        
    # Merge the sentiment data with the geographical data
    merged_gdf = nyc_gdf.merge(incident_sums, left_on='ZCTA5CE10', right_on='Incident Zip', how='left')
    merged_gdf = merged_gdf.dropna(subset=['positive_pct', 'negative_pct'])

    # Create choropleth maps for each sentiment
    sentiment_map_fig = px.choropleth_mapbox(
        merged_gdf,
        geojson=merged_gdf.geometry,
        locations=merged_gdf.index,
        color='positive_pct',
        color_continuous_scale="Viridis",
        mapbox_style="carto-positron",
        zoom=9,
        center={"lat": 40.7128, "lon": -74.0060},
        opacity=0.7,
        labels={'positive_pct': 'Positive Sentiment %'},
        hover_data=['ZCTA5CE10', 'positive_pct', 'negative_pct', 'neutral_pct', 'total']
    )
    sentiment_map_fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    st.subheader("Positive Sentiment Map")
    st.plotly_chart(sentiment_map_fig)
    
    gc.collect()

def zip_code_heatmap(incident_df: pd.DataFrame):
    """
    Generates a heatmap of NYC zip codes with sentiment data.
    """
    st.title("ZIP Code Sentiment Heatmap")
    st.write("This map visualizes the concentration of sentiment across NYC by zip code.")
    
    # Load the optimized shapefile
    nyc_gdf = load_shapefile()
    
    if incident_df is None or incident_df.empty:
        st.info("No incident data found for the selected topic. Please select a topic from the sidebar.")
        return
    if nyc_gdf is None or nyc_gdf.empty:
        st.error("Failed to load NYC geographical data.")
        return
        
    incident_sums = incident_df.groupby('Incident Zip')[['negative', 'positive', 'neutral']].sum().reset_index()
    incident_sums['total'] = incident_sums[['negative', 'positive', 'neutral']].sum(axis=1)
    incident_sums['combined_sentiment'] = (incident_sums['positive'] - incident_sums['negative']) / incident_sums['total']
    
    merged_gdf = nyc_gdf.merge(incident_sums, left_on='ZCTA5CE10', right_on='Incident Zip', how='left')
    merged_gdf = merged_gdf.dropna(subset=['combined_sentiment'])
    
    # Create the heatmap
    heatmap_fig = px.choropleth_mapbox(
        merged_gdf,
        geojson=merged_gdf.geometry,
        locations=merged_gdf.index,
        color='combined_sentiment',
        color_continuous_scale="RdBu",
        range_color=(-1, 1),
        mapbox_style="carto-positron",
        zoom=9,
        center={"lat": 40.7128, "lon": -74.0060},
        opacity=0.7,
        labels={'combined_sentiment': 'Combined Sentiment (Positive-Negative)'},
        hover_data=['ZCTA5CE10', 'combined_sentiment', 'total']
    )
    heatmap_fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    st.plotly_chart(heatmap_fig)
    
    gc.collect()

def borough_income_chart(df: pd.DataFrame):
    """
    Analyzes and visualizes average median income by borough.
    """
    st.title("Average Median Income by Borough")
    
    if df is None or df.empty:
        st.info("No data available. Please select a topic from the sidebar.")
        return
    
    if 'Borough' not in df.columns or 'Median Income' not in df.columns:
        st.error("The selected dataset does not contain 'Borough' or 'Median Income' columns.")
        return

    # Assuming 'df' has a 'Borough' column and 'Median Income'
    borough_income = df.groupby('Borough')['Median Income'].mean().reset_index()
    
    fig = px.bar(
        borough_income,
        x='Borough',
        y='Median Income',
        title='Average Median Income by Borough',
        labels={'Median Income': 'Average Median Income ($)', 'Borough': 'Borough'}
    )
    st.plotly_chart(fig)
    gc.collect()


def project_report_page():
    """
    Displays the project report page with an overview of the project.
    """
    st.title("Project Report: Twitter Sentiment Analysis for NYC Data")
    st.markdown("""
    This project is a comprehensive analysis of Twitter data related to New York City, leveraging advanced AI models to classify tweets by sentiment and emotion. The goal is to provide a powerful, interactive dashboard for exploring public opinion on various topics, such as COVID-19, politics, and economics.

    ### 1. Data Collection and Preprocessing
    The foundation of this project is a large dataset of tweets related to NYC. The data was meticulously cleaned and preprocessed to handle missing values, normalize text, and standardize zip codes. This crucial step ensures the accuracy and reliability of the downstream analysis.

    ### 2. Machine Learning Models
    Two key machine learning models were developed and utilized:
    - **Sentiment Analysis Model:** A classification model trained to categorize tweets as 'positive', 'negative', or 'neutral'.
    - **Emotion Detection Model:** A multi-class classification model capable of identifying emotions such as 'joy', 'anger', 'sadness', and 'surprise'.

    These models, stored as joblib files, are efficiently loaded at the start of the application using Streamlit's caching mechanisms to avoid redundant computations.

    ### 3. Dashboard Functionality
    The Streamlit dashboard is designed for ease of use and powerful insights:
    - **Topic Selection:** Users can choose from different pre-classified datasets (COVID-19, Politics, Economics) to focus their analysis.
    - **Interactive Visualizations:** The dashboard features a variety of charts and maps to present complex data in an accessible format:
        - **Sentiment & Emotion Distribution:** Pie and bar charts show the overall distribution of sentiments and emotions.
        - **Temporal Analysis:** Line charts track how sentiments change over time, allowing for the identification of trends.
        - **Geospatial Mapping:** Using the `geopandas` and `plotly` libraries, interactive maps visualize sentiment and emotion heatmaps across NYC zip codes, revealing geographical patterns in public opinion.

    ### 4. Combined Prediction
    A unique feature of this application is the "Combined Prediction" page, which allows users to input their own text. The app then leverages both the sentiment and emotion models to provide a combined prediction, demonstrating the power and accuracy of the underlying AI.

    ### 5. Technical Architecture
    The application is built using a modern Python stack:
    - **Streamlit:** For creating the interactive web dashboard.
    - **Pandas & GeoPandas:** For data manipulation and geospatial analysis.
    - **Joblib:** For efficient serialization and loading of machine learning models.
    - **Plotly Express:** For generating interactive and aesthetically pleasing charts.
    - **gdown:** For securely and efficiently downloading large datasets and models from Google Drive.

    ### 6. Conclusion
    This project successfully combines data science, machine learning, and geospatial analysis to create a powerful tool for understanding public sentiment. The interactive dashboard provides a clear, intuitive way to explore complex data, making it valuable for researchers, policymakers, and anyone interested in the pulse of public opinion in New York City.
    """)
    gc.collect()

def combined_prediction_page():
    """
    Page for combined sentiment and emotion prediction on user input text.
    """
    st.title("Combined Sentiment & Emotion Prediction")
    st.write("Enter a tweet or a short text below to see its predicted sentiment and emotion.")

    # Load models
    sentiment_model, sentiment_vectorizer, emotion_model, emotion_vectorizer = load_all_models_cached()
    
    if sentiment_model is None:
        st.error("Models could not be loaded. Please try again.")
        return

    user_text = st.text_area("Enter your text here:", height=150)

    if st.button("Predict"):
        if user_text:
            try:
                # Preprocess the user text
                clean_text = unicodedata.normalize('NFKD', user_text).encode('ascii', 'ignore').decode('utf-8')
                
                # Predict sentiment
                sentiment_vectorized = sentiment_vectorizer.transform([clean_text])
                sentiment_prediction = sentiment_model.predict(sentiment_vectorized)[0]
                
                # Predict emotion
                emotion_vectorized = emotion_vectorizer.transform([clean_text])
                emotion_prediction = emotion_model.predict(emotion_vectorized)[0]

                st.subheader("Prediction Results")
                st.info(f"**Text:** {user_text}")
                
                # Display sentiment with color
                sentiment_color = "green" if sentiment_prediction == "positive" else "red" if sentiment_prediction == "negative" else "blue"
                st.markdown(f"**Sentiment:** <span style='color:{sentiment_color}'>**{sentiment_prediction.capitalize()}**</span>", unsafe_allow_html=True)
                
                # Display emotion with color
                emotion_color_map = {'joy': 'gold', 'anger': 'darkred', 'surprise': 'purple', 'sadness': 'darkblue', 'fear': 'gray', 'disgust': 'darkgreen'}
                emotion_color = emotion_color_map.get(emotion_prediction, 'black')
                st.markdown(f"**Emotion:** <span style='color:{emotion_color}'>**{emotion_prediction.capitalize()}**</span>", unsafe_allow_html=True)

            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
        else:
            st.warning("Please enter some text to get a prediction.")


def main():
    """
    Main function to run the dashboard.
    """
    st.title("🐦 Twitter Sentiment Analysis for NYC Data")
    st.write("Explore public sentiment and emotion on various topics across New York City.")
    st.info(f"Current memory usage: {get_memory_usage()}")

    # Sidebar for topic selection
    st.sidebar.title("Data Selection")
    topic_choice = st.sidebar.radio(
        "Select a topic:",
        list(TOPIC_DATA_MAP.keys()),
        key='topic_selector'
    )
    
    # Load data based on selection and cache it in session state
    if st.session_state.get('topic_choice') != topic_choice:
        st.session_state.topic_choice = topic_choice
        st.session_state.df = load_data(topic_choice)
        st.session_state.incident_df = load_incident_df(topic_choice)
        st.session_state.page_selector = "📊 Sentiment Analysis Dashboard"
        st.experimental_rerun()
    
    # Wait for data to be loaded
    if 'df' not in st.session_state or 'incident_df' not in st.session_state:
        st.info("Loading data. Please wait...")
        return
        
    df = st.session_state.df
    incident_df = st.session_state.incident_df
    
    # Sidebar for visualization choice
    st.sidebar.title("Visualization Options")
    visualization_choice = st.sidebar.radio(
        "Choose a visualization:",
        [
            "📈 Overall Sentiment Distribution",
            "📈 Overall Emotion Distribution",
            "📉 Temporal Sentiment Analysis",
            "🔗 Sentiment vs. Emotion Correlation",
            "🔥 Emotion vs Sentiment Heatmap",
            "🗺 ZIP Code Sentiment Maps",
            "🗺 ZIP Code Sentiment Heatmap",
            "💰 Average Median Income by Borough"
        ],
        key='visualization_selector'
    )

    # Display charts based on user choice
    with st.spinner("Generating chart..."):
        if visualization_choice == "📈 Overall Sentiment Distribution":
            create_pie_chart(df, "Overall Sentiment Distribution")
        elif visualization_choice == "📈 Overall Emotion Distribution":
            create_bar_chart(df['emotion'].value_counts().reset_index(), "Overall Emotion Distribution", 'emotion', 'count')
        elif visualization_choice == "📉 Temporal Sentiment Analysis":
            if not df.empty:
                sentiment_over_time = df.groupby(['month_year', 'sentiment']).size().reset_index(name='count')
                sentiment_over_time['month_year'] = sentiment_over_time['month_year'].astype(str)
                create_line_chart(sentiment_over_time, "Sentiment Trends Over Time", 'month_year', 'count', 'sentiment')
        elif visualization_choice == "🔗 Sentiment vs. Emotion Correlation":
            sentiment_emotion_correlation(df)
        elif visualization_choice == "🔥 Emotion vs Sentiment Heatmap":
            emotion_sentiment_heatmap(df)
        elif visualization_choice == "🗺 ZIP Code Sentiment Maps":
            # Check if incident_df is not None before passing it to the function
            if incident_df is not None:
                zip_code_maps(incident_df)
            else:
                st.error("Incident data not loaded. Please select a topic.")
        elif visualization_choice == "🗺 ZIP Code Sentiment Heatmap":
            # Check if incident_df is not None before passing it to the function
            if incident_df is not None:
                zip_code_heatmap(incident_df)
            else:
                st.error("Incident data not loaded. Please select a topic.")
        elif visualization_choice == "💰 Average Median Income by Borough":
            borough_income_chart(df)


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
    page = st.sidebar.radio("Go to:", ["📊 Sentiment Analysis Dashboard", "🔍 Combined Prediction", "📄 Project Report"], key='page_selector')

    if st.session_state.get('last_page') != page:
        st.session_state.clear()
        st.session_state.last_page = page

    if page == "📄 Project Report":
        project_report_page()
    elif page == "📊 Sentiment Analysis Dashboard":
        main()
    elif page == "🔍 Combined Prediction":
        combined_prediction_page()
