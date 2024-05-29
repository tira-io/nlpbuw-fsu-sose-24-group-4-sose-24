from tira.rest_api_client import Client
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
from levenshtein import calculate_levenshtein_distance

if __name__ == "__main__":
    # Load the data from TIRA platform
    tira = Client()
    text = tira.pd.inputs("nlpbuw-fsu-sose-24", "paraphrase-identification-train-20240515-training").set_index("id")
    labels = tira.pd.truths("nlpbuw-fsu-sose-24", "paraphrase-identification-train-20240515-training").set_index("id")

    # Calculate Levenshtein distance
    text['levenshtein_distance'] = calculate_levenshtein_distance(text, 'sentence1', 'sentence2')
    df = text.join(labels)

    # Prepare the data for training
    X = df[['levenshtein_distance']]
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a logistic regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")

    # Save the model
    joblib.dump(model, 'model.pkl')
    joblib.dump(X.columns, 'features.pkl')
