import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib

def load_data(filepath):
    """Load the processed and augmented dataset."""
    return pd.read_csv(filepath)

def prepare_data(df):
    """Split features and encode target labels."""
    X = df.drop(columns=["label"])
    y = df["label"]
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    return X, y_encoded, label_encoder

def train_model(X_train, y_train):
    """Train a Random Forest classifier."""
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    return clf

def evaluate_model(model, X_test, y_test, label_encoder):
    """Evaluate the model and print performance metrics."""
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=label_encoder.classes_))

def save_model(model, label_encoder, model_dir="models"):
    """Save the trained model and label encoder."""
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(model_dir, "rf_model.pkl"))
    joblib.dump(label_encoder, os.path.join(model_dir, "label_encoder.pkl"))
    print("Model and label encoder saved.")

if __name__ == "__main__":
    dataset_path = os.path.join("data", "processed", "augmented_dataset.csv")
    df = load_data(dataset_path)

    X, y, label_encoder = prepare_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test, label_encoder)
    save_model(model, label_encoder)
