from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
from bs4 import BeautifulSoup
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')

app = Flask(__name__)
# CORS(app)
# CORS(app, resources={r"/*": {"origins": "*"}})

CORS(app, resources={r"/*": {"origins": "https://webconttool.netlify.app/"}})
web_contents = {}  # Stores scraped content per URL

def scrape_content(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = [p.get_text() for p in soup.find_all('p')]
        web_contents[url] = " ".join(paragraphs)  # Store text per URL
        return "Content scraped successfully!"
    except Exception as e:
        return f"Error scraping content: {str(e)}"

@app.route('/ingest', methods=['POST'])
def ingest():
    data = request.json
    urls = data.get('urls', [])
    for url in urls:
        scrape_content(url)
    return jsonify({"message": "Content Ingested"})



@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    question = data.get('question', '').strip()

    if not web_contents:
        return jsonify({"answer": ["No content ingested yet!"]})

    all_sentences = []
    sentence_sources = {}  # Dictionary to map sentences to their sources

    for url, content in web_contents.items():
        sentences = nltk.sent_tokenize(content)
        for sentence in sentences:
            all_sentences.append(sentence)
            sentence_sources[sentence] = url  # Store which URL the sentence came from

    if not all_sentences:
        return jsonify({"answer": ["No relevant answer found."]})

    # Compute similarity
    vectorizer = TfidfVectorizer().fit_transform([question] + all_sentences)
    similarities = cosine_similarity(vectorizer[0:1], vectorizer[1:]).flatten()

    # Get top 3 most relevant sentences
    best_match_indices = similarities.argsort()[-3:][::-1]
    answers = [all_sentences[i] for i in best_match_indices if similarities[i] > 0.1]  # Filter out weak matches

    return jsonify({"answer": answers if answers else ["No relevant answer found."]})


# if __name__ == '__main__':
#     app.run(debug=True, port=5000)

import os

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Hosting provider ke port ka use karega
    app.run(debug=False, port=port, host="0.0.0.0")  # External access allow karega
