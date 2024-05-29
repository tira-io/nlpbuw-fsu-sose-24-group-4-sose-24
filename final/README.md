Summary "train.py"

1. import necessary modules and classes
2. Data loading

3. A scikit-learn Pipeline is created to streamline the preprocessing and modeling steps.
    -The FeatureUnion combines two feature extraction transformers: NGramFeatures (for n-gram TF-IDF features) and SemanticSimilarity (for semantic similarity features).
    -An SVM classifier with a linear kernel (SVC(kernel='linear')) is added to the pipeline.
4. A parameter grid (param_grid) is defined for hyperparameter tuning, specifying different values for the SVM's C parameter.
    -"GridSearchCV" is used to perform a grid search with 5-fold cross-validation to find the best hyperparameters based on accuracy.
5. The pipeline is fitted to the training data using the best hyperparameters found by the grid search.
6. Save model.
7. Additionally find accuracy. 

Summary "custom_transformers.py

1.  import necessary modules
2. CustomScaler:
    -Initialization (__init__): Takes a factor parameter which is used to scale the data.
    -Fit Method (fit): Computes the scaling factor using the range of values in X.
    -Transform Method (transform): Scales the input data X by the computed scaling factor.
3. CustomBinarizer:
    -Initialization (__init__): Takes a threshold parameter for binarization.
    -Fit Method (fit): No fitting needed for binarization, hence it returns self.
    -Transform Method (transform): Binarizes the data based on the threshold.
