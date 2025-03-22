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
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Ensure necessary NLTK resources are available
nltk_data_path = "/app/nltk_data"  # Change this if needed
os.makedirs(nltk_data_path, exist_ok=True)
nltk.data.path.append(nltk_data_path)

nltk.download('punkt', download_dir=nltk_data_path)
nltk.download('averaged_perceptron_tagger', download_dir=nltk_data_path)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Store scraped content
web_contents = {}

def scrape_content(url):
    """Scrapes content from a given URL and stores it."""
    try:
        response = requests.get(url, timeout=10)  # Added timeout for reliability
        if response.status_code != 200:
            return f"Error fetching URL ({response.status_code})"

        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = [p.get_text() for p in soup.find_all('p') if p.get_text()]
        
        if not paragraphs:
            return "No meaningful content found on page."

        web_contents[url] = " ".join(paragraphs)  # Store text per URL
        return "Content scraped successfully!"
    
    except requests.exceptions.RequestException as e:
        return f"Request error: {str(e)}"
    except Exception as e:
        return f"Unexpected error: {str(e)}"

@app.route('/ingest', methods=['POST'])
def ingest():
    """Ingests content from provided URLs."""
    try:
        data = request.json
        urls = data.get('urls', [])
        
        if not urls:
            return jsonify({"error": "No URLs provided"}), 400
        
        messages = {}
        for url in urls:
            messages[url] = scrape_content(url)

        return jsonify({"message": "Content Ingested", "stored_urls": list(web_contents.keys()), "status": messages})
    
    except Exception as e:
        return jsonify({"error": f"Unexpected server error: {str(e)}"}), 500

@app.route('/ask', methods=['POST'])
def ask():
    """Finds relevant answers based on the ingested content."""
    try:
        data = request.json
        question = data.get('question', '').strip()

        if not question:
            return jsonify({"error": "Question cannot be empty"}), 400

        if not web_contents:
            return jsonify({"answer": ["No content ingested yet!"]})

        all_sentences = []
        sentence_sources = {}

        for url, content in web_contents.items():
            sentences = nltk.sent_tokenize(content)
            for sentence in sentences:
                all_sentences.append(sentence)
                sentence_sources[sentence] = url

        if not all_sentences:
            return jsonify({"answer": ["No relevant answer found."]})

        # Compute similarity
        vectorizer = TfidfVectorizer().fit_transform([question] + all_sentences)
        similarities = cosine_similarity(vectorizer[0:1], vectorizer[1:]).flatten()

        # Get top 3 most relevant sentences
        best_match_indices = similarities.argsort()[-3:][::-1]
        answers = [all_sentences[i] for i in best_match_indices if similarities[i] > 0.1]

        return jsonify({"answer": answers if answers else ["No relevant answer found."]})
    
    except Exception as e:
        return jsonify({"error": f"Unexpected server error: {str(e)}"}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, port=port, host="0.0.0.0")
