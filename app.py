'''
Flask-based web API for web scraping and answering questions based on scraped content.
Uses TF-IDF and cosine similarity to find relevant answers.
'''

# Importing necessary libraries
from flask import Flask, request, jsonify  # Flask for API creation
from flask_cors import CORS  # To allow cross-origin requests
import requests  # For making HTTP requests
from bs4 import BeautifulSoup  # For web scraping
import traceback  # For error handling
import os  # For handling environment variables
import re  # For text cleaning and regex operations
from sklearn.feature_extraction.text import TfidfVectorizer  # For text vectorization
from sklearn.metrics.pairwise import cosine_similarity  # For measuring text similarity

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Dictionary to store scraped web content
web_contents = {}

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
        
        # Parse the HTML content
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract relevant text from paragraphs, headers, and list items
        elements = soup.find_all(['p', 'h1', 'h2', 'li'])
        paragraphs = [clean_text(elem.get_text()) for elem in elements if elem.get_text().strip()]
        
        if paragraphs:
            content = " ".join(paragraphs)
            web_contents[url] = content  # Store scraped content
            print(f"Content stored for {url}: {len(content)} characters")
            return f"Content from {url} scraped successfully!"
        else:
            print(f"No useful content found at {url}.")
            return f"No useful content found at {url}."
    
    except requests.exceptions.RequestException as e:
        print(f"Error fetching {url}: {str(e)}")
        return f"Error fetching URL {url}: {str(e)}"

def split_sentences(text):
    """Use regex to tokenize text into sentences for better processing."""
    return re.split(r'(?<=[.!?])\s+', text)

@app.route('/ingest', methods=['POST'])
def ingest():
    """Endpoint to ingest content from provided URLs."""
    try:
        data = request.json
        urls = data.get('urls', [])
        
        if not urls:
            return jsonify({"error": "No URLs provided"}), 400

        # Scrape content from each provided URL
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
    """Endpoint to process user queries and return relevant answers."""
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
        
        # Tokenize content from stored URLs
        for url, content in web_contents.items():
            sentences = split_sentences(content)
            for sentence in sentences:
                clean_sentence = clean_text(sentence)
                if clean_sentence:
                    all_sentences.append(clean_sentence)
                    sentence_sources[clean_sentence] = url  # Store source URL for the sentence
        
        if not all_sentences:
            return jsonify({"answer": ["No relevant answer found."]})

        # Compute similarity between question and stored sentences
        vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        tfidf_matrix = vectorizer.fit_transform([question] + all_sentences)
        similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
        
        best_match_index = similarities.argmax()  # Get the best-matching sentence
        best_match_score = similarities[best_match_index]
        
        if best_match_score < 0.3:  # Threshold to ensure relevance
            return jsonify({"answer": ["No relevant answer found."]})
        
        best_answer = all_sentences[best_match_index]
        source_url = sentence_sources[best_answer]
        
        return jsonify({"answer": [f"{best_answer.strip()} (Source: {source_url})"]})
    
    except Exception as e:
        error_msg = traceback.format_exc()
        print(f"❌ ERROR in /ask: {error_msg}")
        return jsonify({"error": "Internal Server Error", "details": str(e)}), 500

if __name__ == '__main__':
    # Set up Flask app to run on specified port
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, port=port, host="0.0.0.0")
