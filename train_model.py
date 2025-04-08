import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, precision_recall_curve, precision_recall_fscore_support
from sklearn.pipeline import Pipeline
import joblib
import argparse
import os
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import shap

# Import optional packages with error handling
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_output_dirs(output_dir):
    """Create necessary output directories"""
    os.makedirs(output_dir, exist_ok=True)
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    return plots_dir

def load_and_validate_data(data_file):
    """Load and validate the dataset"""
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Input file not found: {data_file}")
    
    logger.info(f"Loading data from {data_file}...")
    df = pd.read_csv(data_file)
    
    # Check if label column exists
    label_col = 'is_bot' if 'is_bot' in df.columns else 'label'
    if label_col not in df.columns:
        raise ValueError(f"No label column found. Expected 'label' or 'is_bot' in dataframe columns: {df.columns}")
    
    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Class distribution: {df[label_col].value_counts()}")
    
    # Check for missing values
    missing_values = df.isnull().sum().sum()
    if missing_values > 0:
        logger.warning(f"Dataset contains {missing_values} missing values")
        # Simple imputation for missing values
        df = df.fillna(df.mean())
    
    return df, label_col

def prepare_features(df, label_col):
    """Prepare features and target variable"""
    # Columns to exclude from features
    exclude_cols = [label_col] + [col for col in df.columns if col in ['file_name', 'ip', 'user_agent', 'timestamp', 'session_id']]
    X = df.drop(columns=[col for col in exclude_cols if col in df.columns])
    y = df[label_col]
    
    return X, y

def augment_human_data(X_human, y_human, noise_factor=0.05, multiplier=2):
    """Augment human data with slight variations to increase diversity"""
    logger.info(f"Augmenting human data (multiplier={multiplier}, noise_factor={noise_factor})...")
    augmented_X = X_human.copy()
    augmented_y = y_human.copy()
    
    for i in range(multiplier - 1):
        noisy_X = X_human.copy()
        # Add Gaussian noise to numerical features
        for col in noisy_X.select_dtypes(include=[np.number]).columns:
            noise = np.random.normal(0, noise_factor * noisy_X[col].std(), size=len(noisy_X))
            noisy_X[col] = noisy_X[col] + noise
        
        augmented_X = pd.concat([augmented_X, noisy_X])
        augmented_y = pd.concat([augmented_y, y_human])
    
    logger.info(f"Augmented human data from {len(X_human)} to {len(augmented_X)} samples")
    return augmented_X, augmented_y

def generate_synthetic_bot_data(X_human, num_samples):
    """Generate synthetic bot data with more regular patterns"""
    logger.info(f"Generating {num_samples} synthetic bot samples...")
    synthetic_X = pd.DataFrame(columns=X_human.columns)
    
    for i in range(num_samples):
        sample = {}
        
        # Create synthetic features for common bot behaviors
        for col in X_human.columns:
            if 'mouse_path' in col or 'curvature' in col:
                # Bots have straighter mouse paths
                sample[col] = np.random.uniform(0.01, 0.1)
            elif 'speed_variance' in col or 'acceleration' in col:
                # Bots have more consistent speed
                sample[col] = np.random.uniform(0.005, 0.05)
            elif 'keystroke' in col or 'typing' in col:
                # Bots have more regular typing patterns
                sample[col] = np.random.uniform(0.001, 0.02)
            elif 'backspace' in col or 'correction' in col:
                # Bots make fewer corrections
                sample[col] = np.random.uniform(0, 0.01)
            elif 'time' in col or 'duration' in col:
                # Bots are often faster
                mean_val = X_human[col].mean() * 0.7
                std_val = X_human[col].std() * 0.3
                sample[col] = np.random.normal(mean_val, std_val)
            else:
                # For other features, use mean with reduced variance
                mean_val = X_human[col].mean()
                std_val = X_human[col].std() * 0.2
                sample[col] = np.random.normal(mean_val, std_val)
        
        synthetic_X = pd.concat([synthetic_X, pd.DataFrame([sample])], ignore_index=True)
    
    return synthetic_X

def balance_dataset(X_train, y_train, augment_threshold=0.3):
    """Balance dataset if needed using augmentation and synthetic data"""
    human_ratio = (y_train == 0).sum() / len(y_train)
    bot_ratio = (y_train == 1).sum() / len(y_train)
    
    logger.info(f"Original class distribution - Human: {human_ratio:.2f}, Bot: {bot_ratio:.2f}")
    
    # Check if balancing is needed
    if min(human_ratio, bot_ratio) >= augment_threshold:
        logger.info("Data is already sufficiently balanced. No augmentation needed.")
        return X_train, y_train
    
    # Get human and bot data
    human_mask = (y_train == 0)
    X_human = X_train[human_mask]
    y_human = y_train[human_mask]
    X_bot = X_train[~human_mask]
    y_bot = y_train[~human_mask]
    
    # Determine which class needs augmentation
    if human_ratio < bot_ratio:
        # Augment human data
        target_human_count = int(len(y_train) * 0.5)
        multiplier = max(1, target_human_count // len(X_human))
        X_human_aug, y_human_aug = augment_human_data(X_human, y_human, multiplier=multiplier)
        
        # Combine with original bot data
        X_balanced = pd.concat([X_human_aug, X_bot])
        y_balanced = pd.concat([y_human_aug, y_bot])
    else:
        # Generate synthetic bot data
        target_bot_count = int(len(y_train) * 0.5)
        num_synthetic_bots = target_bot_count - len(X_bot)
        if num_synthetic_bots > 0:
            X_synthetic_bots = generate_synthetic_bot_data(X_human, num_synthetic_bots)
            y_synthetic_bots = pd.Series([1] * len(X_synthetic_bots))
            
            # Combine with original data
            X_balanced = pd.concat([X_human, X_bot, X_synthetic_bots])
            y_balanced = pd.concat([y_human, y_bot, y_synthetic_bots])
        else:
            X_balanced, y_balanced = X_train, y_train
    
    logger.info(f"Balanced class distribution - Human: {(y_balanced == 0).sum() / len(y_balanced):.2f}, "
                f"Bot: {(y_balanced == 1).sum() / len(y_balanced):.2f}")
    return X_balanced, y_balanced

def plot_confusion_matrix(y_test, y_pred, model_name, plots_dir):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Human', 'Bot'], 
                yticklabels=['Human', 'Bot'])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'{model_name}_confusion_matrix.png'))
    plt.close()

def plot_pr_curve(y_test, y_pred_proba, model_name, plots_dir):
    """Plot and save precision-recall curve"""
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, marker='.')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {model_name}')
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, f'{model_name}_pr_curve.png'))
    plt.close()

def plot_feature_importance(pipeline, X, model_name, plots_dir):
    """Plot and save feature importance for a model"""
    if hasattr(pipeline.named_steps['model'], 'feature_importances_'):
        plt.figure(figsize=(12, 8))
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': pipeline.named_steps['model'].feature_importances_
        }).sort_values('Importance', ascending=False)
        
        # Plot top 20 features or all if less than 20
        top_n = min(20, len(feature_importance))
        sns.barplot(x='Importance', y='Feature', data=feature_importance.head(top_n), palette='viridis')
        plt.title(f'Top {top_n} Features - {model_name}')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'{model_name}_feature_importance.png'))
        plt.close()
        
        return feature_importance
    return None

def create_shap_plots(pipeline, X_test, model_name, plots_dir):
    """Create and save SHAP plots for model explainability"""
    try:
        model = pipeline.named_steps['model']
        # Check if the model is a tree-based model that SHAP can explain
        if hasattr(model, 'feature_importances_'):
            logger.info(f"Generating SHAP plots for {model_name}...")
            # Create explainer
            explainer = shap.TreeExplainer(model)
            
            # Calculate SHAP values
            # Use a smaller subset if dataset is large
            sample_size = min(500, X_test.shape[0])
            X_sample = X_test.sample(sample_size, random_state=42)
            shap_values = explainer.shap_values(X_sample)
            
            # Summary plot
            plt.figure(figsize=(12, 8))
            if isinstance(shap_values, list):
                # For multi-class, use class 1 (bot)
                shap.summary_plot(shap_values[1] if len(shap_values) > 1 else shap_values[0], 
                                X_sample, feature_names=X_test.columns.tolist(), show=False)
            else:
                shap.summary_plot(shap_values, X_sample, feature_names=X_test.columns.tolist(), show=False)
            
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f'{model_name}_shap_summary.png'))
            plt.close()
            
            # Dependence plots for top features
            feature_importance = pd.DataFrame({
                'Feature': X_test.columns,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            for feature in feature_importance.head(5)['Feature']:
                plt.figure(figsize=(10, 6))
                if isinstance(shap_values, list):
                    shap.dependence_plot(feature, shap_values[1] if len(shap_values) > 1 else shap_values[0], 
                                        X_sample, feature_names=X_test.columns.tolist(), show=False)
                else:
                    shap.dependence_plot(feature, shap_values, X_sample, 
                                        feature_names=X_test.columns.tolist(), show=False)
                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, f'{model_name}_shap_dependence_{feature}.png'))
                plt.close()
            
            logger.info(f"SHAP plots saved to {plots_dir}")
            return True
    except Exception as e:
        logger.warning(f"Could not create SHAP plots: {str(e)}")
    
    return False

def create_adversarial_samples(X_test, y_test, plots_dir):
    """Create adversarial samples to test model robustness"""
    logger.info("Creating adversarial samples...")
    
    # Select bot samples only
    bot_mask = (y_test == 1)
    X_bot = X_test[bot_mask].copy()
    
    if len(X_bot) == 0:
        logger.warning("No bot samples in test set to create adversarial examples")
        return None
    
    # Limit sample size for performance
    sample_size = min(100, len(X_bot))
    adv_samples = X_bot.sample(sample_size).copy()
    
    # Add human-like randomness to bot behavior
    for col in adv_samples.columns:
        if 'mouse' in col:
            # Introduce slight irregularities in mouse movements
            noise_factor = np.random.uniform(0.05, 0.15)
            adv_samples[col] += np.random.normal(0, noise_factor * adv_samples[col].std(), size=len(adv_samples))
        
        elif 'key' in col or 'type' in col:
            # Make typing patterns more human-like but still programmatic
            adv_samples[col] *= np.random.uniform(0.9, 1.1, size=len(adv_samples))
            
        elif 'click' in col:
            # Introduce slight variability in click patterns
            adv_samples[col] += np.random.normal(0, 0.1 * adv_samples[col].std(), size=len(adv_samples))
        
        else:
            # Minimal change to other features
            adv_samples[col] += np.random.normal(0, 0.05 * adv_samples[col].std(), size=len(adv_samples))
    
    # Plot comparison of original vs adversarial features
    plt.figure(figsize=(14, 8))
    for i, col in enumerate(np.random.choice(adv_samples.columns, 6, replace=False)):
        plt.subplot(2, 3, i+1)
        plt.hist(X_bot[col], alpha=0.5, label='Original')
        plt.hist(adv_samples[col], alpha=0.5, label='Adversarial')
        plt.title(col)
        plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'adversarial_comparison.png'))
    plt.close()
    
    return adv_samples

def setup_models():
    """Set up all models to evaluate"""
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'SVM': SVC(probability=True, random_state=42)
    }
    
    # Add optional models if available
    if XGBOOST_AVAILABLE:
        models['XGBoost'] = xgb.XGBClassifier(random_state=42)
    
    if LIGHTGBM_AVAILABLE:
        models['LightGBM'] = lgb.LGBMClassifier(random_state=42)
    
    return models

def setup_param_grids():
    """Set up hyperparameter grids for each model"""
    param_grids = {
        'Logistic Regression': {
            'C': [0.01, 0.1, 1, 10],
            'penalty': ['l2'],
            'solver': ['liblinear']
        },
        'Random Forest': {
            'n_estimators': [100, 200],
            'max_depth': [None, 15, 30],
            'min_samples_split': [2, 5]
        },
        'Gradient Boosting': {
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1],
            'max_depth': [3, 5, 7]
        },
        'SVM': {
            'C': [0.1, 1, 10],
            'gamma': ['scale', 'auto']
        }
    }
    
    # Add optional model hyperparameters if available
    if XGBOOST_AVAILABLE:
        param_grids['XGBoost'] = {
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1],
            'max_depth': [3, 5, 7],
            'reg_lambda': [0.1, 1.0]
        }
    
    if LIGHTGBM_AVAILABLE:
        param_grids['LightGBM'] = {
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1],
            'max_depth': [3, 5, 7],
            'reg_lambda': [0.1, 1.0]
        }
    
    return param_grids

def tune_hyperparameters(X_train, y_train, models, param_grids):
    """Perform hyperparameter tuning for each model using cross-validation"""
    tuned_models = {}
    
    for name, model in models.items():
        logger.info(f"Tuning hyperparameters for {name}...")
        
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model)
        ])
        
        # Prepare parameter grid
        if name in param_grids:
            param_grid = {'model__' + key: value for key, value in param_grids[name].items()}
        else:
            logger.warning(f"No parameter grid defined for {name}, using default parameters")
            param_grid = {}
        
        # Use smaller grid for quick testing if dataset is large
        if X_train.shape[0] > 10000:
            for param in param_grid:
                param_grid[param] = param_grid[param][:1]
        
        # Set up cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # Perform grid search with cross-validation
        grid_search = GridSearchCV(
            pipeline, 
            param_grid, 
            cv=cv, 
            scoring='f1',  # Using F1 score for imbalanced classification
            n_jobs=-1,
            verbose=1
        )
        
        try:
            grid_search.fit(X_train, y_train)
            logger.info(f"Best parameters for {name}: {grid_search.best_params_}")
            tuned_models[name] = grid_search.best_estimator_
        except Exception as e:
            logger.error(f"Error tuning {name}: {str(e)}")
            # Fall back to default model if tuning fails
            pipeline.fit(X_train, y_train)
            tuned_models[name] = pipeline
    
    return tuned_models

def evaluate_models(models, X_train, X_val, y_train, y_val, plots_dir):
    """Evaluate models on validation set and return results"""
    results = {}
    best_f1 = 0
    best_model_name = None
    
    for name, pipeline in models.items():
        logger.info(f"\nEvaluating {name}...")
        
        # Make predictions
        y_pred = pipeline.predict(X_val)
        y_pred_proba = pipeline.predict_proba(X_val)[:, 1]
        
        # Calculate metrics
        train_auc = roc_auc_score(y_train, pipeline.predict_proba(X_train)[:, 1])
        val_auc = roc_auc_score(y_val, y_pred_proba)
        precision, recall, f1, _ = precision_recall_fscore_support(y_val, y_pred, average='binary')
        report = classification_report(y_val, y_pred)
        
        logger.info(f"{name} - Train AUC: {train_auc:.4f}, Val AUC: {val_auc:.4f}")
        logger.info(f"{name} - Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        logger.info(f"Classification Report:\n{report}")
        
        # Create evaluation plots
        plot_confusion_matrix(y_val, y_pred, name, plots_dir)
        plot_pr_curve(y_val, y_pred_proba, name, plots_dir)
        
        # Store results
        results[name] = {
            'pipeline': pipeline,
            'train_auc': train_auc,
            'val_auc': val_auc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'report': report
        }
        
        # Check if this is the best model based on F1 score
        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name
    
    return results, best_model_name

def evaluate_best_model_on_test(best_model_name, results, X_test, y_test, plots_dir):
    """Evaluate the best model on the test set"""
    if not best_model_name:
        logger.warning("No best model found to evaluate.")
        return {}
    
    logger.info(f"\nEvaluating best model ({best_model_name}) on test set...")
    
    best_pipeline = results[best_model_name]['pipeline']
    
    # Make predictions
    y_pred = best_pipeline.predict(X_test)
    y_pred_proba = best_pipeline.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    test_auc = roc_auc_score(y_test, y_pred_proba)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    report = classification_report(y_test, y_pred)
    
    logger.info(f"Test AUC: {test_auc:.4f}")
    logger.info(f"Test Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    logger.info(f"Test Classification Report:\n{report}")
    
    # Create evaluation plots for test set
    plot_confusion_matrix(y_test, y_pred, f"{best_model_name}_test", plots_dir)
    plot_pr_curve(y_test, y_pred_proba, f"{best_model_name}_test", plots_dir)
    
    # Create SHAP plots for explainability
    create_shap_plots(best_pipeline, X_test, f"{best_model_name}_test", plots_dir)
    
    # Create and evaluate on adversarial samples
    adv_samples = create_adversarial_samples(X_test, y_test, plots_dir)
    if adv_samples is not None:
        adv_pred = best_pipeline.predict(adv_samples)
        adv_pred_proba = best_pipeline.predict_proba(adv_samples)[:, 1]
        detection_rate = adv_pred.mean()
        logger.info(f"Bot Detection Rate on Adversarial Samples: {detection_rate:.4f}")
        logger.info(f"Mean Confidence Score: {adv_pred_proba.mean():.4f}")
    
    test_results = {
        'auc': test_auc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'report': report
    }
    
    return test_results

def save_model_artifacts(best_model_name, results, X, test_results, output_dir, plots_dir, data_file):
    """Save model and related artifacts"""
    if not best_model_name:
        logger.warning("No best model found to save.")
        return
    
    best_pipeline = results[best_model_name]['pipeline']
    
    # Save model with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(output_dir, f'captcha_model_{best_model_name}_{timestamp}.pkl')
    joblib.dump(best_pipeline, model_path)
    
    # Also save with standard name for easier imports
    standard_model_path = os.path.join(output_dir, 'captcha_model.pkl')
    joblib.dump(best_pipeline, standard_model_path)
    
    logger.info(f"Best model ({best_model_name}) saved to {model_path}")
    
    # Save feature list
    features_path = os.path.join(output_dir, 'model_features.txt')
    with open(features_path, 'w') as f:
        for feature in X.columns:
            f.write(f"{feature}\n")
    logger.info(f"Feature list saved to {features_path}")
    
    # Save model metadata
    metadata_path = os.path.join(output_dir, 'model_metadata.txt')
    with open(metadata_path, 'w') as f:
        f.write(f"Model type: {best_model_name}\n")
        f.write(f"Training AUC: {results[best_model_name]['train_auc']:.4f}\n")
        f.write(f"Validation AUC: {results[best_model_name]['val_auc']:.4f}\n")
        f.write(f"Test AUC: {test_results['auc']:.4f}\n")
        f.write(f"Test Precision: {test_results['precision']:.4f}\n")
        f.write(f"Test Recall: {test_results['recall']:.4f}\n")
        f.write(f"Test F1 Score: {test_results['f1']:.4f}\n")
        f.write(f"Training data: {data_file}\n")
        f.write(f"Training date: {datetime.now()}\n")
        f.write(f"Feature count: {X.shape[1]}\n")
        f.write(f"Model parameters: {str(best_pipeline.get_params())}\n")
    logger.info(f"Model metadata saved to {metadata_path}")
    
    # Save detailed metrics
    metrics_path = os.path.join(output_dir, 'model_metrics.csv')
    metrics_df = pd.DataFrame([{
        'model': name,
        'train_auc': res['train_auc'],
        'val_auc': res['val_auc'],
        'val_precision': res['precision'],
        'val_recall': res['recall'],
        'val_f1': res['f1']
    } for name, res in results.items()])
    metrics_df.to_csv(metrics_path, index=False)
    logger.info(f"Model comparison metrics saved to {metrics_path}")
    
    # Plot feature importance
    feature_importance = plot_feature_importance(best_pipeline, X, best_model_name, plots_dir)
    if feature_importance is not None:
        importance_path = os.path.join(output_dir, 'feature_importance.csv')
        feature_importance.to_csv(importance_path, index=False)
        logger.info(f"Feature importance data saved to {importance_path}")

def train_model(data_file='data/processed/augmented_dataset.csv', output_dir='models'):
    """Train the CAPTCHA detection model with comprehensive ML pipeline"""
    try:
        # Create output directories
        plots_dir = create_output_dirs(output_dir)
        
        # Load and validate data
        df, label_col = load_and_validate_data(data_file)
        
        # Prepare features and target
        X, y = prepare_features(df, label_col)
        
        # Split data into train, validation, and test sets
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        )
        
        logger.info(f"Train set: {X_train.shape}, Validation set: {X_val.shape}, Test set: {X_test.shape}")
        
        # Balance training dataset if needed
        X_train, y_train = balance_dataset(X_train, y_train)
        
        # Setup models and hyperparameter grids
        models = setup_models()
        param_grids = setup_param_grids()
        
        # Tune hyperparameters
        tuned_models = tune_hyperparameters(X_train, y_train, models, param_grids)
        
        # Evaluate models on validation set
        results, best_model_name = evaluate_models(
            tuned_models, X_train, X_val, y_train, y_val, plots_dir
        )
        
        # Evaluate best model on test set
        test_results = evaluate_best_model_on_test(
            best_model_name, results, X_test, y_test, plots_dir
        )
        
        # Save model and related artifacts
        save_model_artifacts(
            best_model_name, results, X, test_results, output_dir, plots_dir, data_file
        )
        
        logger.info(f"Training completed successfully. Best model: {best_model_name}")
        return best_model_name, results
        
    except Exception as e:
        logger.error(f"Error during model training: {str(e)}")
        raise

def main():
    """Main function to run the model training pipeline"""
    parser = argparse.ArgumentParser(description='Train CAPTCHA detection model')
    parser.add_argument('--data', type=str, default='data/processed/augmented_dataset.csv',
                        help='Path to the input CSV dataset')
    parser.add_argument('--output', type=str, default='models',
                        help='Output directory for models and artifacts')
    args = parser.parse_args()
    
    logger.info("Starting CAPTCHA detection model training pipeline...")
    logger.info(f"Input data: {args.data}")
    logger.info(f"Output directory: {args.output}")
    
    try:
        best_model_name, results = train_model(args.data, args.output)
        logger.info(f"Pipeline completed successfully. Best model: {best_model_name}")
        
        # Print summary of model performance
        logger.info("\nModel Performance Summary:")
        for name, res in results.items():
            logger.info(f"{name}: F1={res['f1']:.4f}, AUC={res['val_auc']:.4f}")
        
        return 0
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        return 1

def predict_sample(model_path, sample_data):
    """Make predictions on new data using trained model"""
    logger.info(f"Loading model from {model_path}...")
    pipeline = joblib.load(model_path)
    
    # Convert sample data to DataFrame if needed
    if not isinstance(sample_data, pd.DataFrame):
        sample_data = pd.DataFrame([sample_data])
    
    # Make prediction
    pred_proba = pipeline.predict_proba(sample_data)[:, 1]
    pred_class = pipeline.predict(sample_data)
    
    # Return probabilities and classes
    return {
        'probability': pred_proba.tolist(),
        'class': pred_class.tolist(),
        'is_bot': pred_class.astype(bool).tolist()
    }

def evaluate_on_new_data(model_path, data_path, output_dir=None):
    """Evaluate trained model on new dataset"""
    logger.info(f"Evaluating model on new data: {data_path}")
    
    # Create output directory if specified
    if output_dir:
        plots_dir = create_output_dirs(output_dir)
    else:
        plots_dir = None
    
    # Load model
    pipeline = joblib.load(model_path)
    
    # Load data
    df, label_col = load_and_validate_data(data_path)
    X, y = prepare_features(df, label_col)
    
    # Make predictions
    y_pred = pipeline.predict(X)
    y_pred_proba = pipeline.predict_proba(X)[:, 1]
    
    # Calculate metrics
    auc = roc_auc_score(y, y_pred_proba)
    precision, recall, f1, _ = precision_recall_fscore_support(y, y_pred, average='binary')
    report = classification_report(y, y_pred)
    
    logger.info(f"Evaluation results on new data:")
    logger.info(f"AUC: {auc:.4f}")
    logger.info(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    logger.info(f"Classification Report:\n{report}")
    
    # Create plots if output directory is provided
    if plots_dir:
        model_name = "Evaluation_on_new_data"
        plot_confusion_matrix(y, y_pred, model_name, plots_dir)
        plot_pr_curve(y, y_pred_proba, model_name, plots_dir)
        
        # Extract model name from model path
        model_type = model_path.split('/')[-1].split('_')[2] if '_' in model_path else "Model"
        create_shap_plots(pipeline, X, model_type, plots_dir)
    
    return {
        'auc': auc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'report': report
    }

if __name__ == "__main__":
    sys.exit(main())