import streamlit as st
import pandas as pd
import numpy as np
import nltk
import contractions
import string
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download NLTK stopwords
nltk.download('stopwords')

# Load datasets
trial_data = pd.read_csv('data.csv')
trial_data.drop(348, inplace=True)
trial_data = trial_data[['event_name', 'datetime', 'description', 'host_org', 'event_perks', 'categories', 'location']]

data = pd.read_csv('preprocessed_data.csv')
data['host_org'] = data['host_org'].fillna('Unknown')
data.event_perks = data.event_perks.fillna('')
data.categories = data.categories.fillna('')
data['information'] = data['description'] + ' ' + data['host_org'] + ' ' + data['event_perks'] + ' ' + data['categories'] + ' ' + data['location']

# Text preprocessing function
def preprocess_text(data):
    if isinstance(data, pd.Series):
        data = data.astype(str).apply(lambda x: x.lower())
    else:
        data = data.lower()
    data = re.sub(r'<.*?>', '', data)
    data = re.sub(r'https?://[^\s]+', '', data)
    data = re.sub(r'@\w+', '', data)
    data = re.sub(r'#\w+', '', data)
    data = re.sub(r'[^a-zA-Z0-9\s]', '', data)
    data = data.translate(str.maketrans('', '', string.punctuation))
    data = ''.join([i for i in data if not i.isdigit()])
    data = re.sub(r'\s+', ' ', data.strip())
    data = contractions.fix(data)
    stop = nltk.corpus.stopwords.words('english')
    data = ' '.join([x for x in data.split() if x not in (stop)])
    return data

data.information = data.information.apply(preprocess_text)

# Event recommendation function
def recommend_event(user_input):
    user_data = preprocess_text(user_input)
    vectorizer = TfidfVectorizer(max_features=10000)
    vectorizer.fit(data.information)
    user_tfidf = vectorizer.transform([user_data])
    df_tfidf = vectorizer.transform(data.information)
    cosine_sim = cosine_similarity(user_tfidf, df_tfidf)
    top_indices = cosine_sim.argsort().flatten()[-5:][::-1]
    recommendations = trial_data.iloc[top_indices]
    return recommendations

# Streamlit app
st.title('ASU Event Recommender System')
st.write('Welcome to the ASU Event Recommender System! Here, we help you find events tailored to your interests.')

user_input = st.text_input('Enter your preferences:')
if user_input:
    recommendations = recommend_event(user_input)
    st.write('Here are your event recommendations:')
    for index, row in recommendations.iterrows():
        st.write(f"**{row['event_name']}** on {row['datetime']} at {row['location']}")
