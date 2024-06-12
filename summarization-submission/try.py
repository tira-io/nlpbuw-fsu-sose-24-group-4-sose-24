from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory
from nltk.tokenize import sent_tokenize
import pandas as pd

def main():
    try:
        # Initialize TIRA client and load data
        tira = Client()
        df = tira.pd.inputs("nlpbuw-fsu-sose-24", "summarization-validation-20240530-training")

        # Generate summaries for each story in the dataset
        df["summary"] = df["story"].apply(generate_summary)
        
        # Remove the original 'story' column
        df = df.drop(columns=["story"])

        # Save the predictions to a JSONL file
        output_directory = get_output_directory(str(Path(__file__).parent))
        output_path = Path(output_directory) / "predictions.jsonl"
        df.to_json(output_path, orient="records", lines=True)
        print(f"Predictions saved to {output_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

def generate_summary(text, num_sentences=2):
    sentences = sent_tokenize(text)
    if len(sentences) <= num_sentences:
        return text

    # Compute TF-IDF matrix
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)
    
    # Compute pairwise cosine similarity matrix
    similarity_matrix = cosine_similarity(tfidf_matrix)

    # Rank sentences based on similarity scores
    sentence_scores = similarity_matrix.sum(axis=1)
    ranked_sentences = [sentences[i] for i in sentence_scores.argsort()[-num_sentences:][::-1]]

    # form the summary
    return " ".join(ranked_sentences)

if __name__ == "__main__":
    main()
