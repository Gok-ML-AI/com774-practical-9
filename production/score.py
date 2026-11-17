import mlflow

# Sensible ranges for each input feature (in cm), agreed with the
# "subject matter expert". Anything outside these is treated as invalid.
FEATURE_RANGES = {
    'sepalLength': (3, 10),
    'sepalWidth': (1, 6),
}

def loadModel(modelPath):
    # Load a model saved with mlflow.sklearn.save_model
    return mlflow.sklearn.load_model(modelPath)

def isValid(row):
    # A row is valid if it has the right number of features
    # and every feature is inside its sensible range
    if len(row) != len(FEATURE_RANGES):
        return False
    for value, (low, high) in zip(row, FEATURE_RANGES.values()):
        if not (low <= value <= high):
            return False
    return True

def predict(model, X):
    # Guard-railed prediction: returns the predicted class for each
    # valid row, and None for any row with silly values
    return [model.predict([row])[0] if isValid(row) else None for row in X]
