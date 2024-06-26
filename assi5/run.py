from pathlib import Path
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory
import spacy

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")

if __name__ == "__main__":

    tira = Client()

    # loading validation data (automatically replaced by test data when run on tira)
    text_validation = tira.pd.inputs(
        "nlpbuw-fsu-sose-24", "ner-validation-20240612-training"
    )
    targets_validation = tira.pd.truths(
        "nlpbuw-fsu-sose-24", "ner-validation-20240612-training"
    )

    # Function to apply NER tagging using spaCy
    def get_ner_tags(sentence):
        doc = nlp(sentence)
        tags = ["O"] * len(doc)
        for ent in doc.ents:
            tags[ent.start] = f"B-{ent.label_}"
            for i in range(ent.start + 1, ent.end):
                tags[i] = f"I-{ent.label_}"
        return tags

    # labeling the data
    predictions = text_validation.copy()
    predictions['tags'] = predictions['sentence'].apply(get_ner_tags)
    predictions = predictions[['id', 'tags']]

    # saving the prediction
    output_directory = get_output_directory(str(Path(__file__).parent))
    predictions.to_json(
        Path(output_directory) / "predictions.jsonl", orient="records", lines=True
    )
 