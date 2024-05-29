from tira.rest_api_client import Client
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, matthews_corrcoef
from joblib import dump
from pathlib import Path
from custom_transformers import NGramFeatures, SemanticSimilarity
from sklearn.model_selection import GridSearchCV
import pandas as pd

if __name__ == "__main__":
    # Load the data using tira client
    tira = Client()
    text = tira.pd.inputs("nlpbuw-fsu-sose-24", "paraphrase-identification-train-20240515-training").set_index("id")
    labels = tira.pd.truths("nlpbuw-fsu-sose-24", "paraphrase-identification-train-20240515-training").set_index("id")
    train_data = text.join(labels)

    # Create a pipeline with n-gram features and semantic similarity
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('ngram', NGramFeatures()),
            ('semantic', SemanticSimilarity())
        ])),
        ('classifier', SVC(kernel='linear'))  # Using linear kernel for SVM
    ])

    # Grid search for hyperparameter tuning
    param_grid = {
        'classifier__C': [0.1, 1, 10]
    }
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')

    # Fit the model
    grid_search.fit(train_data[['sentence1', 'sentence2']], train_data['label'])

    # Save the best model
    dump(grid_search.best_estimator_, Path(__file__).parent / "model.joblib")

    # Predict and evaluate
    y_pred = grid_search.predict(train_data[['sentence1', 'sentence2']])
    accuracy = accuracy_score(train_data['label'], y_pred)
    mcc = matthews_corrcoef(train_data['label'], y_pred)

    print(f"Accuracy: {accuracy}")
    print(f"MCC: {mcc}")
