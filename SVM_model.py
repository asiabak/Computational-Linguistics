import fasttext
import fasttext.util
import numpy as np
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# File paths
TRAIN_FILE = 'train.txt'
DEV_FILE = 'dev.txt'
TEST_FILE = 'test_blind.txt'

# Download and load FastText aligned vectors for German
print("Downloading and loading FastText model...")
fasttext.util.download_model('de', if_exists='ignore')  # Download aligned German vectors
ft = fasttext.load_model('cc.de.300.bin')

# Function to load training and development data (with labels)
def load_labeled_data(file_path):
    """Load data with latitude, longitude, and text."""
    sentences, labels = [], []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 3:
                try:
                    lat, lon = float(parts[0]), float(parts[1])
                    tweet = parts[2]
                    sentences.append(tweet)
                    labels.append((lat, lon))
                except ValueError:
                    continue
    X = np.array([ft.get_sentence_vector(tweet) for tweet in sentences])
    y = np.array(labels)
    return X, y

# Function to load test data (without labels)
def load_unlabeled_data(file_path):
    """Load data with only text (for predictions)."""
    sentences = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            tweet = line.strip()
            if tweet:  # Ensure the line is not empty
                sentences.append(tweet)
    X = np.array([ft.get_sentence_vector(tweet) for tweet in sentences])
    return X

# Load training and development data
print("Loading training and development data...")
X_train, y_train = load_labeled_data(TRAIN_FILE)
X_dev, y_dev = load_labeled_data(DEV_FILE)

# Train the SVM model
print("Training SVM model...")
svm = MultiOutputRegressor(SVR(kernel='rbf'))
svm.fit(X_train, y_train)

# Evaluate on the development set
print("Evaluating on development set...")
y_dev_pred = svm.predict(X_dev)
mse = mean_squared_error(y_dev, y_dev_pred, multioutput='raw_values')
print("Mean Squared Error for each label on dev set:", mse)
print("Average Mean Squared Error on dev set:", np.mean(mse))

# Load test data and make predictions
print("Loading test data and predicting...")
X_test = load_unlabeled_data(TEST_FILE)
predictions = svm.predict(X_test)

# Save predictions to a file
output_file = 'test_predictions.txt'
print(f"Saving predictions to {output_file}...")
with open(output_file, 'w', encoding='utf-8') as f:
    for pred in predictions:
        f.write(f"{pred[0]}\t{pred[1]}\n")
print("Predictions saved successfully.")
