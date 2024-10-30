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
    
    # Cleaning the text data
    data = re.sub(r'<.*?>', '', data)
    data = re.sub(r'https?://[^\s]+', '', data)
    data = re.sub(r'@\w+', '', data)
    data = re.sub(r'#\w+', '', data)
    data = re.sub(r'[^a-zA-Z0-9\s]', '', data)
    data = data.translate(str.maketrans('', '', string.punctuation))
    data = ''.join([i for i in data if not i.isdigit()])
    data = re.sub(r'\s+', ' ', data.strip())
    data = contractions.fix(data)
    
    # Remove stopwords
    stop = nltk.corpus.stopwords.words('english')
    data = ' '.join([x for x in data.split() if x not in stop])
    return data

# Preprocess information column
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
st.set_page_config(page_title='ASU Event Recommender System', layout='wide')

# Inject CSS for styling
st.markdown(
    """
    <style>
    /* General Reset */
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    
    /* Body Styling */
    body {
        font-family: 'Roboto', sans-serif;
        background-image: url('static/bg.jpg');
        background-repeat: no-repeat;
        background-size: cover;
        background-position: center;
        background-blend-mode: overlay;
        background-color: rgba(0, 0, 0, 0.5);
        color: #333333;
        padding: 20px;
    }
    
    /* Container Styling */
    .container {
        background-color: #ffffff;
        border-radius: 12px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
        padding: 30px;
        max-width: 90%;
        width: 100%;
        margin: 20px auto; /* Centering the container */
    }
    
    /* Headings */
    h1 {
        font-size: 28px;
        color: #8c1d40;
        text-align: center;
        margin-bottom: 20px;
    }

    h2 {
        font-size: 20px;
        color: #333333;
        margin-top: 30px;
        margin-bottom: 15px;
        border-bottom: 2px solid #8c1d40;
        padding-bottom: 5px;
    }

    /* Labels and Inputs */
    label {
        font-size: 20px;
        color: #333333;
        font-weight: 700;
        margin-bottom: 5px;
        display: block;
    }

    textarea {
        width: 100%;
        height: 50px;
        padding: 15px;
        border-radius: 8px;
        border: 2px solid #6a6565;
        margin-bottom: 20px;
        font-size: 14px;
        color: #000;
        resize: none;
        font-family: 'Roboto', sans-serif;
    }

    /* Button Styling */
    input[type="submit"] {
        background-color: #8c1d40;
        color: #ffffff;
        padding: 12px 20px;
        border: none;
        border-radius: 8px;
        cursor: pointer;
        font-size: 16px;
        font-weight: bold;
        display: block;
        width: 50%;
        max-width: 300px;
        transition: background-color 0.3s ease, transform 0.3s ease;
        box-shadow: 0 4px 15px rgba(140, 29, 64, 0.3);
        margin: 0 auto;
    }

    /* Button Hover Effects */
    input[type="submit"]:hover {
        background-color: #a84363;
        transform: translateY(-3px);
    }

    /* Paragraphs */
    p {
        font-size: 15px;
        color: #555555;
        line-height: 1.6;
        margin-top: 15px;
    }

    /* Keywords Section */
    .keywords {
        margin: 20px 0;
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
    }

    .keyword {
        background-color: #8c1d40;
        color: #ffffff;
        padding: 8px 12px;
        border-radius: 20px;
        font-size: 14px;
        transition: background-color 0.3s ease;
    }

    /* Recommendations List */
    ul {
        list-style: none;
        padding: 0;
    }

    .event-item {
        background-color: #e4e0e0;
        border: 1px solid #eeeeee;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 6px 8px rgba(0, 0, 0, 0.6);
    }

    .event-item h2 {
        margin: 0 0 10px 0;
        font-size: 22px;
        color: #8c1d40;
    }

    .event-item p {
        margin: 8px 0;
        color: #000;
        font-size: 15px;
    }

    strong {
        color: #333333;
        font-weight: bold;
    }

    /* Media Queries for Responsiveness */
    @media (max-width: 768px) {
        .container {
            padding: 20px;
            max-width: 95%;
        }

        h1 {
            font-size: 24px;
        }

        h2 {
            font-size: 18px;
        }

        input[type="submit"] {
            width: 70%;
            padding: 12px 18px;
        }

        label, textarea, input[type="submit"], p {
            font-size: 14px;
        }

        .event-item h2 {
            font-size: 20px;
        }

        textarea {
            height: 40px;
        }
    }

    @media (min-width: 1024px) {
        input[type="submit"] {
            width: 30%;
            padding: 16px 28px;
        }
    }
    </style>
    """, unsafe_allow_html=True
)

# App Title and Description
st.title('ASU Event Recommender System')
st.write('Welcome to the ASU Event Recommender System! Here, we help you find events tailored to your interests.')

# Input field for user preferences
user_input = st.text_area('Enter your preferences:', placeholder='e.g., I like music festivals, outdoor activities, and volunteer work.', height=100)

# Display recommendations based on user input
if st.button('Get Event Recommendations'):
    if user_input:
        recommendations = recommend_event(user_input)
        st.write('Here are your event recommendations:')
        for index, row in recommendations.iterrows():
            st.write(f"**{row['event_name']}** on {row['datetime']} at **{row['location']}**")
            st.write(f"*Description:* {row['description']}")
            st.write(f"*Hosted by:* {row['host_org']}")
            st.write(f"*Perks:* {row['event_perks']}")
            st.write(f"*Categories:* {row['categories']}")
            st.write('---')  # Divider for better readability
    else:
        st.warning('Please enter your preferences to get recommendations.')
