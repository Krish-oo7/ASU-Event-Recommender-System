from nltk.corpus import stopwords
import nltk
import contractions
import string
import re
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
app = Flask(__name__)

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


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/recommend', methods=['POST'])
def recommend():
    user_preferences = request.form['preferences']
    recommendations = recommned_event(user_preferences)
    return render_template('recommendations.html', events=recommendations)


if __name__ == '__main__':
    app.run(debug=True)
