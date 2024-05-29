from pathlib import Path
from joblib import load
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory
import pandas as pd

if __name__ == "__main__":
    # Load the data using tira client
    tira = Client()
    validation_data = tira.pd.inputs(
        "nlpbuw-fsu-sose-24", "paraphrase-identification-validation-20240515-training"
    ).set_index("id")

    # Load the model
    model = load(Path(__file__).parent / "model.joblib")

    # Make predictions
    predictions = model.predict(validation_data[['sentence1', 'sentence2']])
    validation_data['label'] = predictions

    # Prepare the predictions dataframe
    predictions_df = validation_data[['label']].reset_index()

    # Save predictions
    output_directory = get_output_directory(str(Path(__file__).parent))
    predictions_df.to_json(Path(output_directory) / "predictions.jsonl", orient="records", lines=True)
