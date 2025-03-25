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
import traceback
import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

web_contents = {}  # Dictionary to store scraped content per URL

def clean_text(text):
    """Remove unnecessary text, repeated words, and boilerplate content."""
    text = re.sub(r'\b(Comment|More info|Advertise with us|Next Article|Follow|Improve Article|Tags|Similar Reads|Machine Learning AI-ML-DS Tutorials)\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'\b(\w+) \1\b', '\1', text)  # Remove duplicate consecutive words
    return text

def scrape_content(url):
    """Scrape and clean content from a given URL."""
    try:
        print(f"Fetching content from: {url}")
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract text from important tags only
        elements = soup.find_all(['p', 'h1', 'h2', 'li'])
        paragraphs = [clean_text(elem.get_text()) for elem in elements if elem.get_text().strip()]
        
        if paragraphs:
            content = " ".join(paragraphs)
            web_contents[url] = content
            print(f"Content stored for {url}: {len(content)} characters")
            return f"Content from {url} scraped successfully!"
        else:
            print(f"No useful content found at {url}.")
            return f"No useful content found at {url}."
    
    except requests.exceptions.RequestException as e:
        print(f"Error fetching {url}: {str(e)}")
        return f"Error fetching URL {url}: {str(e)}"

def split_sentences(text):
    """Use regex to tokenize sentences for better accuracy."""
    return re.split(r'(?<=[.!?])\s+', text)

@app.route('/ingest', methods=['POST'])
def ingest():
    """Ingest content from provided URLs."""
    try:
        data = request.json
        urls = data.get('urls', [])
        
        if not urls:
            return jsonify({"error": "No URLs provided"}), 400

        for url in urls:
            message = scrape_content(url)
            print(message)
        
        print(f"Stored URLs: {list(web_contents.keys())}")
        return jsonify({"message": "Content Ingested", "stored_urls": list(web_contents.keys())})
    
    except Exception as e:
        error_msg = traceback.format_exc()
        print(f"ERROR in /ingest: {error_msg}")
        return jsonify({"error": "Internal Server Error", "details": str(e)}), 500

@app.route('/ask', methods=['POST'])
def ask():
    """Process user question and return relevant answers."""
    try:
        data = request.json
        question = data.get('question', '').strip()

        if not question:
            return jsonify({"error": "No question provided"}), 400

        if not web_contents:
            return jsonify({"answer": ["No content ingested yet!"]})
        
        all_sentences = []
        sentence_sources = {}
        
        print(f"🔍 Processing question: {question}")
        
        for url, content in web_contents.items():
            sentences = split_sentences(content)
            for sentence in sentences:
                clean_sentence = clean_text(sentence)
                if clean_sentence:
                    all_sentences.append(clean_sentence)
                    sentence_sources[clean_sentence] = url
        
        if not all_sentences:
            return jsonify({"answer": ["No relevant answer found."]})

        vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        tfidf_matrix = vectorizer.fit_transform([question] + all_sentences)
        similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
        
        best_match_index = similarities.argmax()  # Get the best match
        best_match_score = similarities[best_match_index]
        
        if best_match_score < 0.3:
            return jsonify({"answer": ["No relevant answer found."]})
        
        best_answer = all_sentences[best_match_index]
        source_url = sentence_sources[best_answer]
        
        return jsonify({"answer": [f"{best_answer.strip()} (Source: {source_url})"]})
    
    except Exception as e:
        error_msg = traceback.format_exc()
        print(f"❌ ERROR in /ask: {error_msg}")
        return jsonify({"error": "Internal Server Error", "details": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, port=port, host="0.0.0.0")                        