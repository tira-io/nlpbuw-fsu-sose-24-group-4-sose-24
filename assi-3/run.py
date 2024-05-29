from pathlib import Path
import pandas as pd
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory
import joblib
from submission.levenshtein import calculate_levenshtein_distance

if __name__ == "__main__":

    tira = Client()
    model = joblib.load('model.pkl')
    feature_names = joblib.load('features.pkl')

    # Load the data from TIRA platform
    df = tira.pd.inputs("nlpbuw-fsu-sose-24", "paraphrase-identification-validation-20240515-training").set_index("id")

    # Compute the Levenshtein distance
    df['levenshtein_distance'] = calculate_levenshtein_distance(df, 'sentence1', 'sentence2')
    
    # Predict labels
    X = df[feature_names]
    df['label'] = model.predict(X)
    
    # Drop unnecessary columns and reset index
    df = df.drop(columns=["sentence1", "sentence2"]).reset_index()

    # Save the predictions
    output_directory = get_output_directory(str(Path(__file__).parent))
    df.to_json(Path(output_directory) / "predictions.jsonl", orient="records", lines=True)
