#!/usr/bin/env python3
# train_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.pipeline import Pipeline
import joblib
import argparse
import os
import matplotlib.pyplot as plt
import seaborn as sns

def train_model(data_file, output_dir='models'):
    """Train the CAPTCHA detection model"""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading data from {data_file}...")
    df = pd.read_csv(data_file)
    
    # Check if 'label' column exists
    label_col = 'label' if 'label' in df.columns else 'is_bot'
    if label_col not in df.columns:
        raise ValueError(f"No label column found. Expected 'label' or 'is_bot' in dataframe columns: {df.columns}")
    
    print(f"Dataset shape: {df.shape}")
    print(f"Class distribution: {df[label_col].value_counts()}")
    
    # Prepare features and target
    X = df.drop(columns=[label_col] + [col for col in df.columns if col in ['file_name', 'ip', 'user_agent']])
    y = df[label_col]
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Define models to try
    models = {
        'gradient_boosting': GradientBoostingClassifier(
            n_estimators=200, 
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        ),
        'random_forest': RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            random_state=42
        )
    }
    
    best_auc = 0
    best_model_name = None
    results = {}
    
    # Train and evaluate models
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Create pipeline with scaler
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model)
        ])
        
        # Train model
        pipeline.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = pipeline.predict(X_test)
        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
        
        auc = roc_auc_score(y_test, y_pred_proba)
        report = classification_report(y_test, y_pred)
        
        print(f"{name} - Test AUC: {auc:.4f}")
        print(report)
        
        results[name] = {
            'pipeline': pipeline,
            'auc': auc,
            'report': report
        }
        
        if auc > best_auc:
            best_auc = auc
            best_model_name = name
    
    # Save best model
    if best_model_name:
        best_pipeline = results[best_model_name]['pipeline']
        model_path = os.path.join(output_dir, 'captcha_model.pkl')
        joblib.dump(best_pipeline, model_path)
        print(f"\nBest model ({best_model_name}) saved to {model_path}")
        
        # Save feature list for deployment
        features_path = os.path.join(output_dir, 'model_features.txt')
        with open(features_path, 'w') as f:
            for feature in X.columns:
                f.write(f"{feature}\n")
        print(f"Feature list saved to {features_path}")
        
        # Plot feature importance if it's a tree-based model
        if hasattr(best_pipeline.named_steps['model'], 'feature_importances_'):
            plt.figure(figsize=(12, 8))
            feature_importance = pd.DataFrame({
                'Feature': X.columns,
                'Importance': best_pipeline.named_steps['model'].feature_importances_
            }).sort_values('Importance', ascending=False)
            
            sns.barplot(x='Importance', y='Feature', data=feature_importance.head(20))
            plt.title(f'Top 20 Features - {best_model_name}')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'feature_importance.png'))
            print(f"Feature importance plot saved to {output_dir}/feature_importance.png")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train CAPTCHA detection model')
    parser.add_argument('--data', required=True, help='Path to processed features CSV file')
    parser.add_argument('--output', default='models', help='Output directory for model and artifacts')
    args = parser.parse_args()
    
    train_model(args.data, args.output)