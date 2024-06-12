from pathlib import Path
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory
import json
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

# # Download NLTK resources
# nltk.download('punkt')
# nltk.download('stopwords')

def preprocess_text(text):
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
    sentences = sent_tokenize(text)  # Tokenize into sentences
    stop_words = set(stopwords.words('english'))
    cleaned_sentences = []
    for sentence in sentences:
        words = word_tokenize(sentence.lower())  # Tokenize into words
        words = [word for word in words if word.isalnum() and word not in stop_words]  # Remove stopwords and non-alphanumeric
        cleaned_sentences.append(' '.join(words))
    return sentences, cleaned_sentences

def build_similarity_matrix(sentences, cleaned_sentences):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(cleaned_sentences)
    similarity_matrix = cosine_similarity(tfidf_matrix)
    return similarity_matrix

def generate_summary(text, top_n=3):
    original_sentences, cleaned_sentences = preprocess_text(text)
    if not cleaned_sentences:
        return ""

    similarity_matrix = build_similarity_matrix(original_sentences, cleaned_sentences)
    sentence_similarity_graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(sentence_similarity_graph)

    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(original_sentences)), reverse=True)
    summary = ' '.join([ranked_sentences[i][1] for i in range(min(top_n, len(ranked_sentences)))])
    return summary

if __name__ == "__main__":
    # Load the data
    tira = Client()
    df = tira.pd.inputs(
        "nlpbuw-fsu-sose-24", "summarization-validation-20240530-training"
    ).set_index("id")

    # Generate summaries for the dataset
    predictions = []
    for idx, story in df.iterrows():
        summary = generate_summary(story["story"])
        predictions.append({"id": idx, "summary": summary})

    # Save the predictions
    output_directory = get_output_directory(str(Path(__file__).parent))
    with open(Path(output_directory) / "predictions.jsonl", 'w') as file:
        for prediction in predictions:
            file.write(json.dumps(prediction) + '\n')
