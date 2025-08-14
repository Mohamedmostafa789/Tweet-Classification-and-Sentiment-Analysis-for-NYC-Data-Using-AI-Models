# Tweet Classification and Sentiment Analysis for NYC Data Using AI Models
**Date:** August 12, 2025

---

## Contents
1. [Abstract](#1-abstract)  
2. [Introduction](#2-introduction)  
   - [Background](#21-background)  
   - [Objectives and Challenges](#22-objectives-and-challenges)  
3. [Data Acquisition and Initial Description](#3-data-acquisition-and-initial-description)  
4. [Data Understanding and Problem Identification](#4-data-understanding-and-problem-identification)  
5. [Data Cleaning and Preprocessing](#5-data-cleaning-and-preprocessing)  
   - [Cleaning Methodology](#51-cleaning-methodology)  
   - [Preservation of Sentiment-Relevant Features](#52-preservation-of-sentiment-relevant-features)  
   - [Implementation Details](#53-implementation-details)  
6. [Geospatial and Socioeconomic Enrichment](#6-geospatial-and-socioeconomic-enrichment)  
   - [Adding Median Income](#61-adding-median-income)  
   - [Adding Incident ZIP Codes](#62-adding-incident-zip-codes)  
   - [Implementation Details](#63-implementation-details)  
7. [Problems Faced, Solutions, and Unsolved Issues](#7-problems-faced-solutions-and-unsolved-issues)  
   - [Data Cleanliness (Solved)](#71-data-cleanliness-solved)  
   - [Missing or Invalid Geospatial Data (Partially Solved)](#72-missing-or-invalid-geospatial-data-partially-solved)  
   - [Multilingual Content (Solved)](#73-multilingual-content-solved)  
   - [Location Data Concentration (Unsolved)](#74-location-data-concentration-unsolved)  
8. [Methodology for Topic Classification](#8-methodology-for-topic-classification)  
   - [Model Selection Process](#81-model-selection-process)  
   - [Chosen Model: twitter-roberta-topic-multi-all](#82-chosen-model-twitter-roberta-topic-multi-all)  
     - [Model Description](#821-model-description)  
     - [Architecture](#822-architecture)  
     - [Training Procedure and Data](#823-training-procedure-and-data)  
     - [Labels/Topics](#824-labelstopics)  
     - [Performance Metrics](#825-performance-metrics)  
     - [Computational Requirements and Limitations](#826-computational-requirements-and-limitations)  
     - [Intended Uses and How to Use](#827-intended-uses-and-how-to-use)  
9. [Implementation of Topic Classification](#9-implementation-of-topic-classification)  
10. [Sentiment and Emotion Analysis on Classified Tweets](#10-sentiment-and-emotion-analysis-on-classified-tweets)  
    - [Approach Using LLMs for Sentiment Analysis](#101-approach-using-llms-for-sentiment-analysis)  
    - [Fine-Tuning Tutorial and Multilingual Handling](#102-fine-tuning-tutorial-and-multilingual-handling)  
    - [Emotion Classification Using Incremental Learning](#103-emotion-classification-using-incremental-learning)  
11. [Overall Implementation and Workflow](#11-overall-implementation-and-workflow)  
12. [Results and Discussion](#12-results-and-discussion)  
13. [Conclusion](#13-conclusion)  
14. [References](#14-references)  

---

## 1. Abstract
This report details the analysis of ∼24 million geo-located tweets from New York City (NYC) in 2020, exploring their relationship with socioeconomic indicators. The workflow involved acquiring raw tweet data, cleaning it to remove noise (e.g., URLs, mentions, emojis converted to text), and enriching it with geospatial (ZIP codes) and socioeconomic (median income) features. Tweets were classified into COVID-19, politics, and economics categories using the `twitter-roberta-topic-multi-all` model, followed by sentiment analysis (positive: 1, neutral: 0, negative: -1) with a fine-tuned Mistral-7B-Instruct-v0.1 LLM, handling multilingual content effectively. Additionally, an incremental learning approach classified emotions (e.g., joy, anger) on the large dataset, achieving an accuracy of **0.7743** with robust precision and recall across emotion classes.

---

## 2. Introduction

### 2.1 Background
NYC, a global hub, generated a rich Twitter dataset in 2020, reflecting public reactions to COVID-19, elections, and economic disruptions. The ∼24 million geo-located tweets, sourced via Twitter API with a 50km radius around NYC, captured diverse sentiments in a multilingual, noisy format, requiring advanced processing for urban insights.

### 2.2 Objectives and Challenges
**Objectives:**
- Clean tweets.
- Enrich with socioeconomic data.
- Classify into COVID-19, politics, and economics.
- Analyze sentiment and emotions.

**Challenges:**
- Noisy text (URLs, emojis).
- Missing/invalid geospatial data.
- Multilingual content.
- Location concentration.

---

## 3. Data Acquisition and Initial Description
The dataset, a CSV with ∼24 million tweets from NYC (2020), was sourced via Twitter API or public archives. Key columns included:
- `tweet` (noisy text)
- `Latitude`
- `Longitude`

---

## 4. Data Understanding and Problem Identification
Using `pandas` chunking, tweet length, lexical diversity, and multilingual content were analyzed. Geospatial issues included missing coordinates and concentration in ~250 spots.

---

## 5. Data Cleaning and Preprocessing

### 5.1 Cleaning Methodology
`MemoryOptimizedTweetCleaner` removed URLs, mentions, digits, punctuation, HTML, and whitespace using regex. Emojis and emoticons were converted to text.

### 5.2 Preservation of Sentiment-Relevant Features
Emojis, hashtags (without `#`), and slang were retained or transformed to preserve emotional context.

### 5.3 Implementation Details
Parallel processing with memory monitoring. Retained ~95% rows.

---

## 6. Geospatial and Socioeconomic Enrichment

### 6.1 Adding Median Income
Merged ACS data and census tract shapefiles via spatial joins.

### 6.2 Adding Incident ZIP Codes
Used `uszipcode` to geocode coordinates.

---

## 7. Problems Faced, Solutions, and Unsolved Issues

### 7.1 Data Cleanliness (Solved)
Removed ~20–30% noise.

### 7.2 Missing or Invalid Geospatial Data (Partially Solved)
Enriched valid coordinates; defaults used for invalid.

### 7.3 Multilingual Content (Solved)
LLMs handled without translation.

### 7.4 Location Data Concentration (Unsolved)
~250 spots biased heatmaps.

---

## 8. Methodology for Topic Classification

### 8.1 Model Selection Process
Evaluated multiple models; chose `twitter-roberta-topic-multi-all` for best F1 score.

### 8.2 Chosen Model: twitter-roberta-topic-multi-all

#### 8.2.1 Model Description
RoBERTa-based multi-label classification for COVID-19, politics, economics.

#### 8.2.5 Performance Metrics
- **F1-micro:** 0.765  
- **F1-macro:** 0.619  
- **Accuracy:** 0.549  

---

## 10. Sentiment and Emotion Analysis on Classified Tweets

### 10.3 Emotion Classification Using Incremental Learning
| Class        | Precision | Recall | F1-Score | Support  |
|--------------|-----------|--------|----------|----------|
| Negative (-1)| 0.89      | 0.22   | 0.36     | 14,489   |
| Neutral (0)  | 0.76      | 0.95   | 0.84     | 50,721   |
| Positive (1) | 0.79      | 0.75   | 0.77     | 34,790   |
| **Accuracy** |           |        | **0.7743** | 100,000 |
| Macro Avg    | 0.81      | 0.64   | 0.66     | 100,000 |
| Weighted Avg | 0.79      | 0.77   | 0.75     | 100,000 |

---

## 11. Overall Implementation and Workflow
1. Acquire data.  
2. Clean tweets.  
3. Add socioeconomic & geospatial data.  
4. Classify topics.  
5. Sentiment & emotion analysis.

---

## 12. Results and Discussion
- Cleaning retained 95% of rows.
- Classification mapped tweets into categories with overlaps.
- Sentiment showed trends tied to events.

---

## 13. Conclusion
Processed ∼24M tweets, revealing sentiment and emotion patterns tied to socioeconomic factors. The framework is scalable and effective.

---

## 14. References
1. Barbieri, F., et al. (2020). [TweetEval](https://arxiv.org/abs/2010.12421)  
2. Barbieri, F., et al. (2022). [TweetEval Model Card](https://huggingface.co/cardiffnlp/twitter-roberta-base-dec2021-tweet-topic-multi-all)  
3. Levy Abitbol, J.; Morales, A.J. (2021). [Socioeconomic Patterns of Twitter User Activity](https://doi.org/10.3390/e23060780)  
4. Gibbons J, et al. (2019). [Twitter-based measures of neighborhood sentiment](https://doi.org/10.1371/journal.pone.0219550)  
5. Zimbra, D., et al. (2018). [The State-of-the-Art in Twitter Sentiment Analysis](https://doi.org/10.1145/3185045)  
6. Liu, Y., et al. (2019). [RoBERTa](https://arxiv.org/abs/1907.11692)  
7. Nagpal, M. (2024). [How to use an LLM for Sentiment Analysis?](https://www.projectpro.io/article/llm-sentiment-analysis/1125)  
