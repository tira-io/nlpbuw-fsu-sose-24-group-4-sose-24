from pathlib import Path
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory
import json
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from collections import Counter, defaultdict
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

def preprocess_text(text):
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
    sentences = sent_tokenize(text)  # Tokenize into sentences
    return sentences

def extract_keywords(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text.lower())
    words = [word for word in words if word.isalnum() and word not in stop_words]
    word_freq = Counter(words)
    most_common_words = word_freq.most_common(10)  # Extract top 10 keywords
    return set(word for word, freq in most_common_words)

def score_sentences(sentences, keywords):
    sentence_scores = defaultdict(int)
    for sentence in sentences:
        for word in word_tokenize(sentence.lower()):
            if word in keywords:
                sentence_scores[sentence] += 1
    return sentence_scores

def generate_summary(text, num_sentences=2):
    sentences = preprocess_text(text)
    if len(sentences) <= num_sentences:
        return text

    keywords = extract_keywords(text)
    sentence_scores = score_sentences(sentences, keywords)
    ranked_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)
    summary = ' '.join(ranked_sentences[:num_sentences])
    return summary

if __name__ == "__main__":
    main()
