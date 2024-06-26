from pathlib import Path
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory
from transformers import pipeline
import pandas as pd
import json

def load_jsonl(file_path):
    with open(file_path, 'r') as file:
        data = [json.loads(line) for line in file]
    return data

def save_jsonl(data, file_path):
    with open(file_path, 'w') as file:
        for entry in data:
            file.write(json.dumps(entry) + '\n')

def get_labels(sentence, ner_model):
    ner_results = ner_model(sentence)
    labels = ["O"] * len(sentence.split())
    for entity in ner_results:
        word_index = sentence[:entity['start']].count(" ")
        labels[word_index] = f"B-{entity['entity_group']}"
        for i in range(word_index + 1, sentence[:entity['end']].count(" ") + 1):
            labels[i] = f"I-{entity['entity_group']}"
    return labels

if __name__ == "__main__":
    # Initialize TIRA client
    tira = Client()

    # Load validation data (automatically replaced by test data when run on TIRA)
    text_validation = tira.pd.inputs("nlpbuw-fsu-sose-24", "ner-validation-20240612-training")
    text_validation = text_validation.to_dict('records')

    # Initialize NER pipeline
    ner_pipeline = pipeline("ner", grouped_entities=True)

    # Apply NER pipeline to the validation data
    predictions = []
    for entry in text_validation:
        sentence = entry['sentence']
        labels = get_labels(sentence, ner_pipeline)
        predictions.append({"id": entry["id"], "tags": labels})

    # Save the predictions
    output_directory = get_output_directory(str(Path(__file__).parent))
    save_jsonl(predictions, Path(output_directory) / "predictions.jsonl")
