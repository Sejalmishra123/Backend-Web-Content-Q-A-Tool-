# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import requests
# from bs4 import BeautifulSoup
# import nltk
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity

# nltk.download('punkt')

# app = Flask(__name__)
# # CORS(app)
# # CORS(app, resources={r"/*": {"origins": "*"}})

# CORS(app, resources={r"/*": {"origins": "*"}})

# web_contents = {}  # Stores scraped content per URL

# def scrape_content(url):
#     try:
#         response = requests.get(url)
#         soup = BeautifulSoup(response.text, 'html.parser')
#         paragraphs = [p.get_text() for p in soup.find_all('p')]
#         web_contents[url] = " ".join(paragraphs)  # Store text per URL
#         return "Content scraped successfully!"
#     except Exception as e:
#         return f"Error scraping content: {str(e)}"

# @app.route('/ingest', methods=['POST'])
# def ingest():
#     data = request.json
#     urls = data.get('urls', [])
#     for url in urls:
#         scrape_content(url)
#     return jsonify({"message": "Content Ingested"})



# @app.route('/ask', methods=['POST'])
# def ask():
#     data = request.json
#     question = data.get('question', '').strip()

#     if not web_contents:
#         return jsonify({"answer": ["No content ingested yet!"]})

#     all_sentences = []
#     sentence_sources = {}  # Dictionary to map sentences to their sources

#     for url, content in web_contents.items():
#         sentences = nltk.sent_tokenize(content)
#         for sentence in sentences:
#             all_sentences.append(sentence)
#             sentence_sources[sentence] = url  # Store which URL the sentence came from

#     if not all_sentences:
#         return jsonify({"answer": ["No relevant answer found."]})

#     # Compute similarity
#     vectorizer = TfidfVectorizer().fit_transform([question] + all_sentences)
#     similarities = cosine_similarity(vectorizer[0:1], vectorizer[1:]).flatten()

#     # Get top 3 most relevant sentences
#     best_match_indices = similarities.argsort()[-3:][::-1]
#     answers = [all_sentences[i] for i in best_match_indices if similarities[i] > 0.1]  # Filter out weak matches

#     return jsonify({"answer": answers if answers else ["No relevant answer found."]})


# # if __name__ == '__main__':
# #     app.run(debug=True, port=5000)

# import os

# if __name__ == '__main__':
#     port = int(os.environ.get("PORT", 5000))  # Hosting provider ke port ka use karega
#     app.run(debug=False, port=port, host="0.0.0.0")  # External access allow karega

from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
from bs4 import BeautifulSoup
import nltk
import traceback
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download necessary NLTK data
nltk.download('punkt')

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Dictionary to store scraped content per URL
web_contents = {}

def scrape_content(url):
    """ Scrape content from a given URL and store it. """
    try:
        response = requests.get(url, timeout=10)  # Timeout to prevent hanging
        response.raise_for_status()  # Raise error for bad status codes (4xx, 5xx)

        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = [p.get_text() for p in soup.find_all('p')]

        if paragraphs:
            web_contents[url] = " ".join(paragraphs)  # Store text per URL
            return f"Content from {url} scraped successfully!"
        else:
            return f"No content found at {url}."

    except requests.exceptions.RequestException as e:
        return f"Error fetching URL {url}: {str(e)}"


@app.route('/ingest', methods=['POST'])
def ingest():
    """ Ingest content from provided URLs. """
    try:
        data = request.json
        urls = data.get('urls', [])

        if not urls:
            return jsonify({"error": "No URLs provided"}), 400  # Bad Request

        for url in urls:
            message = scrape_content(url)
            print(message)  # Log the scraping process

        return jsonify({"message": "Content Ingested", "stored_urls": list(web_contents.keys())})

    except Exception as e:
        error_msg = traceback.format_exc()
        print(f"ERROR in /ingest: {error_msg}")
        return jsonify({"error": "Internal Server Error", "details": str(e)}), 500


@app.route('/ask', methods=['POST'])
def ask():
    """ Process user question and find relevant answers from ingested content. """
    try:
        data = request.json
        question = data.get('question', '').strip()

        if not question:
            return jsonify({"error": "No question provided"}), 400  # Bad Request

        if not web_contents:
            return jsonify({"answer": ["No content ingested yet!"]})

        all_sentences = []
        sentence_sources = {}  # Map sentences to their sources

        for url, content in web_contents.items():
            sentences = nltk.sent_tokenize(content)
            for sentence in sentences:
                all_sentences.append(sentence)
                sentence_sources[sentence] = url

        if not all_sentences:
            return jsonify({"answer": ["No relevant answer found."]})

        # Compute similarity
        vectorizer = TfidfVectorizer(max_features=5000)  # Limit features to prevent memory issues
        tfidf_matrix = vectorizer.fit_transform([question] + all_sentences)
        similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

        # Get top 3 most relevant sentences
        best_match_indices = similarities.argsort()[-3:][::-1]
        answers = [all_sentences[i] for i in best_match_indices if similarities[i] > 0.1]  # Filter out weak matches

        if not answers:
            return jsonify({"answer": ["No relevant answer found."]})

        return jsonify({"answer": answers})

    except Exception as e:
        error_msg = traceback.format_exc()
        print(f"ERROR in /ask: {error_msg}")
        return jsonify({"error": "Internal Server Error", "details": str(e)}), 500


if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))  # Use the provided port
    app.run(debug=False, port=port, host="0.0.0.0")  # Allow external access

