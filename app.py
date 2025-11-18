from nltk.corpus import stopwords
import nltk
import contractions
import string
import re
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download NLTK stopwords
nltk.download('stopwords')


# Another dataset to show to users.
trial_data = pd.read_csv('data.csv')
trial_data.drop(348, inplace=True)

trial_data = trial_data[['event_name', 'datetime', 'description',
                         'host_org', 'event_perks', 'categories', 'location']]

data = pd.read_csv('preprocessed_data.csv')

data['host_org'] = data['host_org'].fillna('Unknown')
data.event_perks = data.event_perks.fillna('')
data.categories = data.categories.fillna('')

data['information'] = data['description'] + ' ' + data['host_org'] + ' ' + \
    data['event_perks'] + ' ' + data['categories'] + ' ' + data['location']

nltk.download('stopwords')


def preprocess_text(data):

    # Convert to lowercase
    if isinstance(data, pd.Series):
        data = data.astype(str).apply(lambda x: x.lower())
    else:
        data = data.lower()

    # Remove HTML tags
    data = re.sub(r'<.*?>', '', data)

    # Remove URLs
    data = re.sub(r'https?://[^\s]+', '', data)

    # Remove mentions
    data = re.sub(r'@\w+', '', data)

    # Remove hashtags
    data = re.sub(r'#\w+', '', data)

    # Remove special characters and punctuation
    data = re.sub(r'[^a-zA-Z0-9\s]', '', data)

    # Remove punctuation
    data = data.translate(str.maketrans('', '', string.punctuation))

    # Remove digits
    data = ''.join([i for i in data if not i.isdigit()])

    # Remove extra whitespace
    data = re.sub(r'\s+', ' ', data.strip())

    # Performing contractions
    data = contractions.fix(data)

    # Remove stop words using NLTK
    stop = nltk.corpus.stopwords.words('english')
    data = ' '.join([x for x in data.split() if x not in (stop)])

    return data


data.information = data.information.apply(preprocess_text)


def recommned_event(user_input):

    # Preprocess user input
    user_data = preprocess_text(user_input)

    # Combining ifidf from data.
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(max_features=10000)

    # Fit the vectorizer on the combined text data
    vectorizer.fit(data.information)

    # Transform the user input
    user_ifidf = vectorizer.transform([user_data])

    # Transform the text data
    df_tfidf = vectorizer.transform(data.information)

    # Calculate cosine similarity
    from sklearn.metrics.pairwise import cosine_similarity
    cosine_sim = cosine_similarity(user_ifidf, df_tfidf)
    # Get top 5 recommendations
    top_indices = cosine_sim.argsort().flatten()[-5:][::-1]
    recommendations = trial_data.iloc[top_indices]
    return recommendations

# Inject CSS styles for button
st.markdown("""
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
            background-image: url('bg.jpg'); 
            background-repeat: no-repeat;
            background-size: cover;
            background-position: center;
            background-blend-mode: overlay;
            background-color: rgba(0, 0, 0, 0.5);
            color: #000000;
        }

        /* Container Styling */
        .stApp {
            background-color: #ffffff;
            border-radius: 12px;
            box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
            padding: 30px;
            max-width: 90%;
            width: 100%;
            margin: 20px auto;
        }

        /* Headings */
        h1 {
            font-size: 28px;
            color: #8c1d40;
            text-align: center;
            margin-bottom: 20px;
        }
        

        /* Input field styling */
        .stTextInput input {
            background-color: #ffffff !important;  /* Changed to white */
            color: #000000 !important;
            border: 1px solid #8c1d40 !important;
            border-radius: 8px !important;
            padding: 10px !important;
        }
        
        .stTextInput input::placeholder {
            color: #4f4e4e;  
        }
    
         /* Style for the label text */
        .stTextInput label {
            color: #000000;  
            font-size: 22px !important;  
            font-weight: bold;
        }
    
        /* Button Styling */
    .stButton button {
        background-color: #8c1d40;
        color: #ffffff;
        padding: 12px 20px;
        border: none;
        border-radius: 8px;
        cursor: pointer;
        font-size: 16px;
        font-weight: bold;
    
        /* Fix wrapping */
        white-space: nowrap !important;
        min-width: 250px !important;
    
        /* Layout */
        display: block;
        width: fit-content !important;
        margin: 20px auto;
    
        /* Effects */
        transition: background-color 0.3s ease, transform 0.3s ease;
        box-shadow: 0 4px 15px rgba(140, 29, 64, 0.3);
    }
    
    .stButton button:hover {
        background-color: #a84363;
        color: #ffffff;
        transform: translateY(-3px);
    }


        /* Recommendations List */
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
    </style>
""", unsafe_allow_html=True)
st.markdown("""
    <style>
        .stTextInput input::placeholder {
            color: #4f4e4e; 
        }
    </style>
""", unsafe_allow_html=True)
# Streamlit app
st.title('ASU Event Recommender System')
st.markdown("""
    <h3 style='font-size: 20px; color:#292828;'>Welcome to the ASU Event Recommender System! Here, we help you find events tailored to your interests.</h3>
""", unsafe_allow_html=True)
user_input = st.text_input('Enter your preferences:' , placeholder='e.g., I like music festivals, outdoor activities, and volunteer work.')
submit_button = st.button("Get Recommendations")

if submit_button and user_input:
    recommendations = recommned_event(user_input)
    st.markdown("""
    <h3 style='font-size: 20px; color:#292828;'>Here are your event recommendations:</h3>
""", unsafe_allow_html=True)
    for index, row in recommendations.iterrows():
        st.markdown(f"""
        <div class="event-item">
            <h2>{row['event_name']}</h2>
            <p><strong>Date:</strong> {row['datetime']}</p>
            <p><strong>Location:</strong> {row['location']}</p>
            <p><strong>Description:</strong> {row['description']}</p>
        </div>

        """, unsafe_allow_html=True)
