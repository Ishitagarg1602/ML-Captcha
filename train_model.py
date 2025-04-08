# import os
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.preprocessing import LabelEncoder
# from sklearn.metrics import accuracy_score, classification_report
# import joblib

# def load_data(filepath):
#     """Load the processed and augmented dataset."""
#     return pd.read_csv(filepath)

# def prepare_data(df):
#     """Split features and encode target labels."""
#     X = df.drop(columns=["label"])
#     y = df["label"]
#     label_encoder = LabelEncoder()
#     y_encoded = label_encoder.fit_transform(y)
#     return X, y_encoded, label_encoder

# def train_model(X_train, y_train):
#     """Train a Random Forest classifier."""
#     clf = RandomForestClassifier(n_estimators=100, random_state=42)
#     clf.fit(X_train, y_train)
#     return clf

# def evaluate_model(model, X_test, y_test, label_encoder):
#     """Evaluate the model and print performance metrics."""
#     y_pred = model.predict(X_test)
#     print("Accuracy:", accuracy_score(y_test, y_pred))
#     print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# def save_model(model, label_encoder, model_dir="models"):
#     """Save the trained model and label encoder."""
#     os.makedirs(model_dir, exist_ok=True)
#     joblib.dump(model, os.path.join(model_dir, "rf_model.pkl"))
#     joblib.dump(label_encoder, os.path.join(model_dir, "label_encoder.pkl"))
#     print("Model and label encoder saved.")

# if __name__ == "__main__":
#     dataset_path = os.path.join("data", "processed", "augmented_dataset.csv")
#     df = load_data(dataset_path)

#     X, y, label_encoder = prepare_data(df)
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     model = train_model(X_train, y_train)
#     evaluate_model(model, X_test, y_test, label_encoder)
#     save_model(model, label_encoder)


# import os
# import pandas as pd
# import numpy as np
# import argparse
# from pathlib import Path
# from tqdm import tqdm

# def augment_data(input_file="data/processed/features.csv", 
#                 output_file="data/processed/augmented_dataset.csv", 
#                 augmentation_factor=2,
#                 random_seed=42):
#     """
#     Augment the labeled dataset to improve model training.
    
#     Args:
#         input_file: Path to the labeled dataset CSV
#         output_file: Path to save the augmented dataset
#         augmentation_factor: How many times to augment each class
#         random_seed: For reproducibility
        
#     Returns:
#         DataFrame with augmented data
#     """
#     np.random.seed(random_seed)
    
#     if not os.path.exists(input_file):
#         raise FileNotFoundError(f"Input file not found: {input_file}")
        
#     print(f"Reading data from {input_file}")
#     try:
#         df = pd.read_csv(input_file)
#     except Exception as e:
#         print(f"Error reading input file: {str(e)}")
#         return None
    
#     if 'label' not in df.columns:
#         raise ValueError("Dataset must contain a 'label' column")
    
#     human_data = df[df['label'] == 'human']
#     bot_data = df[df['label'] == 'bot']
    
#     if len(human_data) == 0 or len(bot_data) == 0:
#         print("Warning: One or both classes have zero samples.")
    
#     print(f"Original dataset contains {len(human_data)} human samples and {len(bot_data)} bot samples")
    
#     # Identify numeric columns for augmentation
#     numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns 
#                    if col != 'label' and 'id' not in col.lower()]
    
#     print(f"Found {len(numeric_cols)} numeric features for augmentation")
    
#     augmented_data = [df]  # Start with original data

#     # Calculate statistics once for efficiency
#     human_stats = {col: {'mean': human_data[col].mean(), 'std': human_data[col].std()} 
#                   for col in numeric_cols}
#     bot_stats = {col: {'mean': bot_data[col].mean(), 'std': bot_data[col].std()}
#                 for col in numeric_cols}

#     # Augment human data with random noise
#     for i in tqdm(range(augmentation_factor), desc="Augmenting human data"):
#         augmented_humans = human_data.copy()

#         for col in numeric_cols:
#             std_dev = human_stats[col]['std']
#             if std_dev > 0:  # Avoid zero std dev
#                 noise = np.random.normal(0, 0.05 * std_dev, len(augmented_humans))
#                 # Ensure we don't create negative values for strictly positive features
#                 if augmented_humans[col].min() >= 0:
#                     augmented_humans[col] = np.maximum(0, augmented_humans[col] + noise)
#                 else:
#                     augmented_humans[col] += noise

#         augmented_data.append(augmented_humans)
    
#     # Augment bot data with systematic tweaks + noise
#     for i in tqdm(range(augmentation_factor), desc="Augmenting bot data"):
#         augmented_bots = bot_data.copy()

#         # Special handling for certain features based on domain knowledge
#         features_to_modify = {
#             'std_speed': (0.7, 0.9),
#             'avg_curvature': (0.6, 0.8),
#             'avg_speed': (0.8, 1.0),
#             'total_distance': (0.9, 1.1)
#         }
        
#         for col, (min_factor, max_factor) in features_to_modify.items():
#             if col in augmented_bots.columns:
#                 augmented_bots[col] *= np.random.uniform(min_factor, max_factor, len(augmented_bots))

#         # Add noise to all features
#         for col in numeric_cols:
#             std_dev = bot_stats[col]['std']
#             if std_dev > 0 and col not in features_to_modify:
#                 noise = np.random.normal(0, 0.03 * std_dev, len(augmented_bots))
#                 # Ensure we don't create negative values for strictly positive features
#                 if augmented_bots[col].min() >= 0:
#                     augmented_bots[col] = np.maximum(0, augmented_bots[col] + noise)
#                 else:
#                     augmented_bots[col] += noise

#         augmented_data.append(augmented_bots)
    
#     # Combine and save
#     augmented_df = pd.concat(augmented_data, ignore_index=True)
    
#     # Create output directory if it doesn't exist
#     Path(os.path.dirname(output_file)).mkdir(parents=True, exist_ok=True)
    
#     augmented_df.to_csv(output_file, index=False)

#     print(f"Augmented dataset saved to {output_file} with {len(augmented_df)} records.")
#     print(f"New label distribution: {augmented_df['label'].value_counts().to_dict()}")
    
#     return augmented_df

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Augment labeled data for CAPTCHA")
#     parser.add_argument("--input", default="data/processed/features.csv", help="Input CSV file with labeled data")
#     parser.add_argument("--output", default="data/processed/augmented_dataset.csv", help="Output CSV file for augmented data")
#     parser.add_argument("--factor", type=int, default=2, help="Augmentation factor")
#     parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
#     args = parser.parse_args()
    
#     augment_data(args.input, args.output, args.factor, args.seed)

import os
import pandas as pd
import numpy as np
import argparse
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def load_data(filepath):
    """
    Load the processed and augmented dataset.
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        DataFrame with the loaded data
    """
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        return None

def prepare_data(df, feature_cols=None, label_col='label', test_size=0.2, random_state=42):
    """
    Split features and encode target labels.
    
    Args:
        df: DataFrame with features and labels
        feature_cols: List of feature columns to use (None to use all except label)
        label_col: Column name for the labels
        test_size: Fraction of data to use for testing
        random_state: For reproducibility
    
    Returns:
        Tuple of (X_train, X_test, y_train, y_test, feature_columns, label_encoder)
    """
    if feature_cols is None:
        # Use all numeric columns except label and ID columns
        feature_cols = [col for col in df.select_dtypes(include=[np.number]).columns
                       if col != label_col and 'id' not in col.lower()]
    
    print(f"Using {len(feature_cols)} features for training")
    
    X = df[feature_cols]
    y = df[label_col]
    
    # Encode labels if they're strings
    label_encoder = None
    if y.dtype == 'object':
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        print(f"Encoded labels: {dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))}")
    else:
        y_encoded = y
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
    )
    
    print(f"Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")
    
    return X_train, X_test, y_train, y_test, feature_cols, label_encoder

def scale_features(X_train, X_test):
    """
    Scale features using StandardScaler.
    
    Args:
        X_train: Training features
        X_test: Test features
        
    Returns:
        Tuple of (scaled_X_train, scaled_X_test, scaler)
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler

def train_model(X_train, y_train, model_type='random_forest', use_grid_search=True, random_state=42):
    """
    Train a model on the provided data.
    
    Args:
        X_train: Training features
        y_train: Training labels
        model_type: Type of model to train ('random_forest' or 'gradient_boosting')
        use_grid_search: Whether to use grid search for hyperparameter tuning
        random_state: For reproducibility
        
    Returns:
        Trained model
    """
    if model_type == 'random_forest':
        if use_grid_search:
            print("Training Random Forest with GridSearchCV...")
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10]
            }
            model = GridSearchCV(
                RandomForestClassifier(random_state=random_state),
                param_grid,
                cv=5,
                scoring='f1',
                n_jobs=-1,
                verbose=1
            )
        else:
            print("Training Random Forest with default parameters...")
            model = RandomForestClassifier(n_estimators=100, random_state=random_state)
            
    elif model_type == 'gradient_boosting':
        if use_grid_search:
            print("Training Gradient Boosting with GridSearchCV...")
            param_grid = {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 5, 7]
            }
            model = GridSearchCV(
                GradientBoostingClassifier(random_state=random_state),
                param_grid,
                cv=5,
                scoring='f1',
                n_jobs=-1,
                verbose=1
            )
        else:
            print("Training Gradient Boosting with default parameters...")
            model = GradientBoostingClassifier(n_estimators=100, random_state=random_state)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    model.fit(X_train, y_train)
    
    # If we used grid search, get the best model
    if use_grid_search:
        print(f"Best parameters: {model.best_params_}")
        return model.best_estimator_
    
    return model

def evaluate_model(model, X_test, y_test, label_encoder=None):
    """
    Evaluate the model and calculate performance metrics.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        label_encoder: LabelEncoder for class names
        
    Returns:
        Dictionary of evaluation metrics
    """
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted'),
        'f1': f1_score(y_test, y_pred, average='weighted')
    }
    
    # If binary classification and predict_proba is available
    if len(np.unique(y_test)) == 2 and hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X_test)[:, 1]
        metrics['roc_auc'] = roc_auc_score(y_test, y_prob)
        
    # Print evaluation results
    print("Evaluation Metrics:")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")
    
    # Get class names
    if label_encoder is not None:
        target_names = label_encoder.classes_
    else:
        target_names = [str(i) for i in np.unique(y_test)]
    
    # Print classification report
    print("\nClassification Report:")
    report = classification_report(y_test, y_pred, target_names=target_names)
    print(report)
    
    return metrics, report

def plot_confusion_matrix(model, X_test, y_test, label_encoder=None, save_path=None):
    """
    Plot and optionally save confusion matrix.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        label_encoder: LabelEncoder for class names
        save_path: Path to save the plot (None to display only)
    """
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    # Get class names
    if label_encoder is not None:
        class_names = label_encoder.classes_
    else:
        class_names = [str(i) for i in np.unique(y_test)]
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
               xticklabels=class_names, 
               yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    
    if save_path:
        plt.savefig(save_path)
        print(f"Confusion matrix saved to {save_path}")
    else:
        plt.show()
    
    plt.close()

def plot_feature_importance(model, feature_names, save_path=None):
    """
    Plot and optionally save feature importance.
    
    Args:
        model: Trained model with feature_importances_ attribute
        feature_names: List of feature names
        save_path: Path to save the plot (None to display only)
    """
    if not hasattr(model, 'feature_importances_'):
        print("Model does not have feature importances")
        return
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 8))
    plt.title('Feature Importances')
    plt.bar(range(len(importances)), importances[indices])
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Feature importance plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()

def save_model(model, output_dir, model_name=None, label_encoder=None, scaler=None, feature_names=None):
    """
    Save the model and related objects.
    
    Args:
        model: Trained model
        output_dir: Directory to save the model
        model_name: Name for the model file
        label_encoder: LabelEncoder if used
        scaler: Scaler if used
        feature_names: List of feature names
    """
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = model_name or f"model_{timestamp}"
    
    # Save the model
    model_path = os.path.join(output_dir, f"{model_name}.pkl")
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    
    # Save label encoder if provided
    if label_encoder is not None:
        encoder_path = os.path.join(output_dir, f"label_encoder_{timestamp}.pkl")
        joblib.dump(label_encoder, encoder_path)
        print(f"Label encoder saved to {encoder_path}")
    
    # Save scaler if provided
    if scaler is not None:
        scaler_path = os.path.join(output_dir, f"scaler_{timestamp}.pkl")
        joblib.dump(scaler, scaler_path)
        print(f"Scaler saved to {scaler_path}")
    
    # Save feature names if provided
    if feature_names is not None:
        feature_path = os.path.join(output_dir, f"features_{timestamp}.txt")
        with open(feature_path, 'w') as f:
            f.write('\n'.join(feature_names))
        print(f"Feature names saved to {feature_path}")
    
    return {
        'model_path': model_path,
        'encoder_path': encoder_path if label_encoder is not None else None,
        'scaler_path': scaler_path if scaler is not None else None,
        'feature_path': feature_path if feature_names is not None else None
    }

def train_and_evaluate(input_file="data/processed/augmented_dataset.csv", 
                      output_dir="models/",
                      model_type="random_forest",
                      test_size=0.2,
                      use_grid_search=True,
                      scale_features_flag=True,
                      random_state=42):
    """
    Complete training pipeline: load data, prepare, train, evaluate and save model.
    
    Args:
        input_file: Path to the input dataset
        output_dir: Directory to save the model and related files
        model_type: Type of model to train
        test_size: Fraction of data to use for testing
        use_grid_search: Whether to use grid search for hyperparameter tuning
        scale_features_flag: Whether to scale features
        random_state: For reproducibility
        
    Returns:
        Dictionary with trained model and evaluation results
    """
    # Load the data
    df = load_data(input_file)
    if df is None:
        return None
    
    # Create a timestamp for this training run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Prepare the data
    X_train, X_test, y_train, y_test, feature_cols, label_encoder = prepare_data(
        df, test_size=test_size, random_state=random_state
    )
    
    # Scale features if requested
    if scale_features_flag:
        X_train, X_test, scaler = scale_features(X_train, X_test)
    else:
        scaler = None
    
    # Train the model
    model = train_model(X_train, y_train, model_type, use_grid_search, random_state)
    
    # Evaluate the model
    metrics, report = evaluate_model(model, X_test, y_test, label_encoder)
    
    # Create output directory for this run
    run_dir = os.path.join(output_dir, f"{model_type}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    # Save evaluation report
    report_path = os.path.join(run_dir, "evaluation_report.txt")
    with open(report_path, 'w') as f:
        f.write(f"Model: {model_type}\n")
        f.write(f"Training date: {timestamp}\n")
        f.write(f"Dataset: {input_file}\n")
        f.write(f"Samples: {len(df)}\n\n")
        f.write("Metrics:\n")
        for name, value in metrics.items():
            f.write(f"{name}: {value:.4f}\n")
        f.write("\nClassification Report:\n")
        f.write(report)
    
    # Plot confusion matrix
    cm_path = os.path.join(run_dir, "confusion_matrix.png")
    plot_confusion_matrix(model, X_test, y_test, label_encoder, cm_path)
    
    # Plot feature importance if available
    if hasattr(model, 'feature_importances_'):
        fi_path = os.path.join(run_dir, "feature_importance.png")
        plot_feature_importance(model, feature_cols, fi_path)
    
    # Save model and related objects
    save_paths = save_model(
        model, run_dir, f"{model_type}", label_encoder, scaler, feature_cols
    )
    
    # Return results
    return {
        'model': model,
        'metrics': metrics,
        'label_encoder': label_encoder,
        'scaler': scaler,
        'feature_names': feature_cols,
        'save_paths': save_paths
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate models on processed data")
    parser.add_argument("--input", default="data/processed/augmented_dataset.csv", help="Input dataset file")
    parser.add_argument("--output_dir", default="models/", help="Output directory for models")
    parser.add_argument("--model", default="random_forest", choices=["random_forest", "gradient_boosting"], 
                        help="Model type to train")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test set size ratio")
    parser.add_argument("--no_grid_search", action="store_true", help="Disable grid search")
    parser.add_argument("--no_scaling", action="store_true", help="Disable feature scaling")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    results = train_and_evaluate(
        input_file=args.input,
        output_dir=args.output_dir,
        model_type=args.model,
        test_size=args.test_size,
        use_grid_search=not args.no_grid_search,
        scale_features_flag=not args.no_scaling,
        random_state=args.seed
    )