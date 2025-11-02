# Public Policy Sentiment Analysis — CoWIN Twitter Dataset

**Author:** <Jay Chafale>   
**Deliverables:** One Jupyter notebook `model_comparison.ipynb`, one Streamlit app `streamlit_app.py`, dataset `cowin_processed.csv`.

---

## Project summary
This project analyzes public sentiment toward India’s CoWIN vaccination platform using the CoWIN Twitter Dataset (Mendeley). Goals:
1. Preprocess tweets and build text classification pipelines.
2. Train & compare multiple models (classical ML, deep learning, transformer baseline).
3. Produce model evaluation, temporal sentiment analysis, and visualizations.
4. Provide a lightweight Streamlit app to explore outputs and perform live inference.

---

## Dataset
**Source:** CoWIN Twitter Dataset (Mendeley Data ID: 10.17632/k5yr89ms8s.2), CC-BY-4.0.  
**Files included:** `cowin_processed.csv` — English tweets cleaned and deduplicated (fields: tweet_id, text, timestamp, month, category, sentiment_label). If the dataset is too large for GitHub, include a sample and a download link in this README.

**Notes on usage and ethics**
- Tweets are public, anonymized; abide by Twitter/X terms when rehydrating.
- Please do not expose user IDs or PII.

---

## Models implemented (in `notebooks/model_comparison.ipynb`)
1. **Baseline:** TF-IDF + Logistic Regression  
2. **Deep Learning:** BiLSTM with pretrained word embeddings (or Keras embeddings)  

For each model we provide: training code, hyperparameters, confusion matrix, accuracy, precision, recall, F1, ROC/AUC (where applicable), inference code, and runtime notes.

---

## How this satisfies assignment requirements
- Single .ipynb containing all modeling, experiments, metrics, visualizations, and saving of model artifacts.
- Streamlit app that allows uploading text, selecting model, and seeing predictions + exploratory plots.
- README explains dataset, methods, how to run, reproducibility steps, and citations.

---

## How to run (local)
1. Clone repo:
```bash
git clone <your-repo-url>
cd public-policy-sentiment
