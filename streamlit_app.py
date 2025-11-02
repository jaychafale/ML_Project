import streamlit as st
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

st.set_page_config(page_title='CoWIN Sentiment Dashboard', layout='wide')
DATA_PATH = 'data/cowin_processed.csv'
MODEL_PATH = 'models/tfidf_lr.pkl'

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH, parse_dates=['timestamp'])

@st.cache_resource
def load_model():
    with open(MODEL_PATH, 'rb') as f:
        return pickle.load(f)

def show_wordcloud(df, sentiment):
    text = ' '.join(df[df['sentiment_label']==sentiment]['clean_text'].astype(str))
    wc = WordCloud(width=800, height=400, background_color='white').generate(text)
    fig, ax = plt.subplots(figsize=(10,5))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)

df = load_data()
model = load_model()

st.title('Public Policy Sentiment Analysis — CoWIN Twitter Data')
option = st.sidebar.radio('Select Mode', ['Overview', 'Predict'])

if option == 'Overview':
    st.subheader('Dataset Overview')
    st.dataframe(df.head(10))

    st.write('### Sentiment Distribution')
    fig, ax = plt.subplots()
    sns.countplot(data=df, x='sentiment_label', ax=ax)
    st.pyplot(fig)

    st.write('### Word Clouds')
    col1, col2 = st.columns(2)
    with col1:
        st.write('Positive Tweets')
        show_wordcloud(df, 'positive')
    with col2:
        st.write('Negative Tweets')
        show_wordcloud(df, 'negative')

elif option == 'Predict':
    st.subheader('Predict Tweet Sentiment')
    text = st.text_area('Enter a tweet text:')
    if st.button('Predict'):
        if text.strip():
            pred = model.predict([text])[0]
            st.success(f'Predicted Sentiment: {pred}')
        else:
            st.warning('Enter valid text before predicting.')