import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, classification_report
from sklearn.pipeline import Pipeline
import xgboost as xgb
import lightgbm as lgb
import shap

# Load preprocessed data
data = pd.read_csv('captcha_behavioral_features.csv')

# Separate features and target
X = data.drop('is_bot', axis=1)
y = data['is_bot']

# Split into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

def augment_human_data(X_human, y_human, noise_factor=0.05, multiplier=2):
    augmented_X = X_human.copy()
    augmented_y = y_human.copy()
    
    for _ in range(multiplier - 1):
        noisy_X = X_human.copy()
        # Add Gaussian noise to numerical features
        for col in noisy_X.select_dtypes(include=[np.number]).columns:
            noise = np.random.normal(0, noise_factor * noisy_X[col].std(), size=len(noisy_X))
            noisy_X[col] = noisy_X[col] + noise
        
        augmented_X = pd.concat([augmented_X, noisy_X])
        augmented_y = pd.concat([augmented_y, y_human])
    
    return augmented_X, augmented_y

def generate_synthetic_bot_data(X_human, num_samples):
    # Create synthetic bot data (more linear/regular patterns)
    synthetic_X = pd.DataFrame(columns=X_human.columns)
    
    # Generate perfectly linear mouse paths
    for i in range(num_samples):
        sample = {}
        # Example: Create perfect linear mouse movements
        sample['mouse_path_curvature'] = 0.05  # Almost straight lines
        sample['mouse_speed_variance'] = 0.02  # Very consistent speed
        sample['mouse_acceleration_variance'] = 0.01  # Very consistent acceleration
        
        # Example: Create regular typing patterns
        sample['keystroke_interval_variance'] = 0.01  # Very regular typing rhythm
        sample['backspace_frequency'] = 0.001  # Almost no corrections
        
        # Add other synthetic features
        for col in X_human.columns:
            if col not in sample:
                # Use mean values from human data with reduced variance
                mean_val = X_human[col].mean()
                std_val = X_human[col].std() * 0.2  # Reduced variance for bots
                sample[col] = np.random.normal(mean_val, std_val)
        
        synthetic_X = pd.concat([synthetic_X, pd.DataFrame([sample])], ignore_index=True)
    
    return synthetic_X

# Apply data augmentation if needed (e.g., class imbalance)
if (y_train == 0).sum() / len(y_train) < 0.3:  # If bots are less than 30% of data
    # Augment human data
    human_mask = y_train == 0
    X_human = X_train[human_mask]
    y_human = y_train[human_mask]
    
    X_human_aug, y_human_aug = augment_human_data(X_human, y_human)
    
    # Generate synthetic bot data
    num_synthetic_bots = int(len(X_train) * 0.5) - (y_train == 1).sum()
    if num_synthetic_bots > 0:
        X_synthetic_bots = generate_synthetic_bot_data(X_human, num_synthetic_bots)
        y_synthetic_bots = pd.Series([1] * len(X_synthetic_bots))
        
        # Combine with original training data
        X_train = pd.concat([X_train, X_synthetic_bots])
        y_train = pd.concat([y_train, y_synthetic_bots])


# Define models to evaluate
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'XGBoost': xgb.XGBClassifier(random_state=42),
    'LightGBM': lgb.LGBMClassifier(random_state=42),
    'SVM': SVC(probability=True, random_state=42)
}

# Define hyperparameter grids for each model
param_grids = {
    'Logistic Regression': {
        'C': [0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear']
    },
    'Random Forest': {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    },
    'Gradient Boosting': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    },
    'XGBoost': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'reg_lambda': [0.1, 1.0, 10.0]
    },
    'LightGBM': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'reg_lambda': [0.1, 1.0, 10.0]
    },
    'SVM': {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.1, 0.01]
    }
}

# Train and evaluate each model
results = {}
best_models = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Create a pipeline with preprocessing and model
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ])
    
    # Set up cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Perform grid search
    grid_search = GridSearchCV(
        pipeline,
        {'model__' + key: value for key, value in param_grids[name].items()},
        cv=cv,
        scoring='f1',
        n_jobs=-1
    )
    
    # Fit grid search
    grid_search.fit(X_train, y_train)
    
    # Get best model
    best_model = grid_search.best_estimator_
    best_models[name] = best_model
    
    # Evaluate on validation set
    y_pred = best_model.predict(X_val)
    y_pred_proba = best_model.predict_proba(X_val)[:, 1]
    
    precision, recall, f1, _ = precision_recall_fscore_support(y_val, y_pred, average='binary')
    auc = roc_auc_score(y_val, y_pred_proba)
    
    results[name] = {
        'best_params': grid_search.best_params_,
        'auc_roc': auc,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
    
    print(f"{name} - Best Parameters: {grid_search.best_params_}")
    print(f"{name} - Validation AUC-ROC: {auc:.4f}")
    print(f"{name} - Validation F1 Score: {f1:.4f}")

# Find best overall model based on AUC-ROC
best_model_name = max(results, key=lambda x: results[x]['auc_roc'])
print(f"\nBest overall model: {best_model_name}")

# Evaluate best model on test set
best_model = best_models[best_model_name]
y_test_pred = best_model.predict(X_test)
y_test_pred_proba = best_model.predict_proba(X_test)[:, 1]

test_auc = roc_auc_score(y_test, y_test_pred_proba)
test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(y_test, y_test_pred, average='binary')

print(f"\nTest Set Evaluation for {best_model_name}:")
print(f"AUC-ROC: {test_auc:.4f}")
print(f"Precision: {test_precision:.4f}")
print(f"Recall: {test_recall:.4f}")
print(f"F1 Score: {test_f1:.4f}")
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_test_pred))

# If best model is tree-based, we can analyze feature importance
if best_model_name in ['Random Forest', 'Gradient Boosting', 'XGBoost', 'LightGBM']:
    model = best_model.named_steps['model']
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\nFeature Importance:")
    print(feature_importance.head(10))
    
    # SHAP values for deeper insights
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    # Summary plot
    print("\nGenerating SHAP summary plot...")
    shap.summary_plot(shap_values, X_test, feature_names=X.columns.tolist(), show=False)
    
    # Save SHAP plot
    plt.savefig('shap_summary.png', bbox_inches='tight')
    print("SHAP summary saved as 'shap_summary.png'")



import joblib

# Save the best model
joblib.dump(best_model, f'captcha_model_{best_model_name}.pkl')
print(f"\nModel saved as 'captcha_model_{best_model_name}.pkl'")

# Save a feature list (important for deployment)
with open('model_features.txt', 'w') as f:
    for feature in X.columns:
        f.write(f"{feature}\n")
print("Feature list saved as 'model_features.txt'")


def create_adversarial_samples(X_test, num_samples=100):
    # Select a subset of test samples
    adv_samples = X_test.sample(num_samples).copy()
    
    # Add human-like randomness to bot behavior
    for col in adv_samples.columns:
        # Add varying degrees of humanlike noise
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
    
    return adv_samples

# Create adversarial samples
print("\nPerforming Adversarial Testing...")
adv_samples = create_adversarial_samples(X_test[y_test == 1])  # Create from bot samples

# Evaluate model on adversarial samples
adv_pred_proba = best_model.predict_proba(adv_samples)[:, 1]
adv_pred = best_model.predict(adv_samples)

# Calculate detection rate (should be high for good models)
detection_rate = adv_pred.mean()
print(f"Bot Detection Rate on Adversarial Samples: {detection_rate:.4f}")
print(f"Mean Confidence Score: {adv_pred_proba.mean():.4f}")