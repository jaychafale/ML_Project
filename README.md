# Public Policy Sentiment & Awareness Analysis — India’s CoWIN Initiative

**Author:** Jay Chafale  
**Course:** Machine Learning & Deep Learning (Section B)  
**Instructor:** Dr. Deepak Asudani  
**Submission:** GitHub Repository (Notebook + Streamlit App + Report)

---

## Overview

This project focuses on analyzing **public sentiment and awareness around India’s CoWIN platform** — the official government portal for COVID-19 vaccination registration and tracking.

Using a large-scale, India-specific Twitter dataset, this work applies **Machine Learning (ML)** and **Deep Learning (DL)** techniques to classify public opinions (positive/negative) and visualize temporal trends, linguistic patterns, and engagement insights.

The goal is to demonstrate comparative performance across classical ML and neural models and deploy a **Streamlit-based interactive dashboard** for exploration.

---

##  Objectives

1. **To build a sentiment classification pipeline** using real-world Indian Twitter data related to public policy (CoWIN).
2. **To compare the performance** of ML and DL models on the same dataset.
3. **To visualize trends and insights** regarding public opinion, engagement, and awareness.
4. **To deploy an interactive dashboard** for exploration and single-tweet prediction.

---

##  Problem Definition

During India’s vaccination rollout, millions of citizens shared their experiences, concerns, and feedback about the **CoWIN platform** on Twitter.  
Analyzing these reactions helps policymakers and technologists understand:
- The public’s acceptance and awareness of government digital initiatives.
- Sentiment shifts over time in response to major vaccination events.
- Key issues that dominated social media conversations.

Thus, the **problem statement** is formulated as:

> “Perform sentiment analysis on India-specific public discourse related to the CoWIN vaccination platform using NLP-based ML and DL models.”

---

##  Dataset Description

### Source
- **Dataset Name:** *CoWIN Twitter Dataset — Public Sentiment & Awareness (India)*  
- **Origin:** [Mendeley Data (DOI: 10.17632/k5yr89ms8s.2)](https://data.mendeley.com/datasets/k5yr89ms8s/2)  
- **License:** CC-BY 4.0 (Permissible for academic and research use)

### Composition
| Feature | Description |
|----------|-------------|
| `timestamp` | Date and time of tweet creation |
| `text` | Tweet content (used as main input feature) |
| `sentiment` | Sentiment label derived from fine-tuned RoBERTa model (positive / negative) |
| `like_count` | Number of likes per tweet |
| `retweet_count` | Number of retweets per tweet |

### Dataset Stats
- **Total samples:** 470,854 tweets  
- **Language:** English (filtered)  
- **Time period:** January – December 2021  
- **Classes:** Positive / Negative  

### Preprocessing Steps
- Removed URLs, mentions, and emojis  
- Lowercased text and normalized whitespace  
- Removed duplicates and non-English tweets  
- Extracted sentiment labels (`predicted_sentiment_roberta`)  
- Final cleaned dataset: `tweets_with_sentiment.csv`

---

##  Methodology

### 1. Data Cleaning & Preprocessing
- Applied regex-based cleaning to remove hyperlinks, mentions, and special characters.
- Tokenized and normalized tweets for ML/DL compatibility.
- Stratified dataset split (80% train / 20% test).
- Converted categorical sentiment labels to numeric (for DL model training).

### 2. Model Implementations

| Model Type | Description | Libraries Used |
|-------------|--------------|----------------|
| **TF-IDF + Logistic Regression** | Baseline ML model using word frequency representations. | `scikit-learn` |
| **BiLSTM Neural Network** | Sequential model that captures contextual dependencies in text. | `TensorFlow / Keras` |

#### A. **TF-IDF + Logistic Regression**
- Vectorized text using bigrams (max 50,000 features)
- Solver: `saga`
- Regularization: `L2`
- Class weighting to handle imbalance
- **Output:** Sentiment label (`positive` / `negative`)

#### B. **Bidirectional LSTM**
- Word embeddings (128-dim)
- Bidirectional LSTM (128 units)
- Dense + Dropout regularization layers
- Sigmoid output neuron for binary classification
- Early stopping + checkpointing for training efficiency

### 3. Model Training & Evaluation
- Evaluation metrics: `Accuracy`, `Precision`, `Recall`, `F1-score`
- Visualizations: confusion matrices and accuracy comparison bar plots

### 4. Model Saving
- TF-IDF + Logistic Regression saved as `models/tfidf_lr.pkl`
- BiLSTM weights saved as `models/lstm_model.h5`
- Tokenizer and label encoder stored for deployment

---

##  Experimental Results

| Model | Accuracy | F1-Score | Notes |
|--------|-----------|----------|-------|
| Logistic Regression (TF-IDF) | ~0.83 | ~0.81 | Strong baseline, fast, interpretable |
| BiLSTM (Keras) | ~0.86 | ~0.84 | Better contextual understanding, slower training |
| Transformer (Roberta baseline) | ~0.88 | ~0.86 | (Reference result from dataset paper) |

**Observations:**
---

##  Streamlit Web App

An interactive interface was built using **Streamlit** for:
- Real-time tweet sentiment prediction
- Word cloud and sentiment distribution visualization
- Batch sentiment analysis via CSV upload

### How to Run:
```bash
streamlit run streamlit_app.py

- The LSTM model performs slightly better on complex sentence structures.
- Logistic Regression remains highly competitive for short-form tweets.
- Performance saturates beyond 50,000 vocabulary size.

---

##  Visualizations

- **Sentiment Distribution:** Proportion of positive vs negative tweets.
- **Temporal Trend:** Sentiment over time (spikes around major announcements).
- **Word Clouds:** Frequent words for positive and negative tweets.
- **Model Comparison Chart:** Bar chart for accuracy and F1-score.

---


