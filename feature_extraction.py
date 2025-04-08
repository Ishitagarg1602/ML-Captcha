import os
import json
import pandas as pd
import numpy as np
import argparse
from math import sqrt
from pathlib import Path
from tqdm import tqdm  # For progress bar

def calculate_features(mouse_movements):
    """
    Extract meaningful features from raw mouse movement data

    Args:
        mouse_movements: List of dictionaries with x, y, timestamp

    Returns:
        Dictionary of features
    """
    # Default empty features dictionary
    empty_features = {
        'movement_count': 0,
        'total_distance': 0,
        'avg_speed': 0,
        'std_speed': 0,
        'max_speed': 0,
        'avg_acceleration': 0,
        'std_acceleration': 0,
        'avg_curvature': 0,
        'duration_ms': 0
    }
    
    # Convert string to list of dictionaries if needed
    if isinstance(mouse_movements, str):
        try:
            mouse_movements = json.loads(mouse_movements)
        except json.JSONDecodeError:
            return empty_features

    # If empty or too short
    if not mouse_movements or len(mouse_movements) < 3:
        empty_features['movement_count'] = len(mouse_movements) if mouse_movements else 0
        return empty_features

    distances, speeds, accelerations, curvatures = [], [], [], []
    
    # Pre-allocate arrays for better performance
    n_points = len(mouse_movements)
    distances = np.zeros(n_points - 1)
    speeds = np.zeros(n_points - 1)
    accelerations = np.zeros(n_points - 2)
    curvatures = np.zeros(n_points - 2)

    for i in range(1, n_points):
        prev = mouse_movements[i - 1]
        curr = mouse_movements[i]

        dx = curr['x'] - prev['x']
        dy = curr['y'] - prev['y']
        distance = sqrt(dx**2 + dy**2)
        time_diff = (curr['timestamp'] - prev['timestamp']) / 1000  # seconds

        distances[i-1] = distance
        speeds[i-1] = distance / time_diff if time_diff > 0 else 0

        # Acceleration
        if i > 1 and speeds[i - 2] > 0 and time_diff > 0:
            accelerations[i-2] = (speeds[i-1] - speeds[i-2]) / time_diff

        # Curvature
        if i > 1:
            prev_prev = mouse_movements[i - 2]
            dx1 = prev['x'] - prev_prev['x']
            dy1 = prev['y'] - prev_prev['y']
            dx2 = curr['x'] - prev['x']
            dy2 = curr['y'] - prev['y']

            len1 = sqrt(dx1**2 + dy1**2)
            len2 = sqrt(dx2**2 + dy2**2)

            if len1 > 0 and len2 > 0:
                dot_product = (dx1 * dx2 + dy1 * dy2) / (len1 * len2)
                dot_product = max(-1, min(1, dot_product))  # Clamp
                curvatures[i-2] = abs(1 - dot_product)

    duration_ms = mouse_movements[-1]['timestamp'] - mouse_movements[0]['timestamp']
    
    # Filter out zeros/placeholders
    accelerations = accelerations[accelerations != 0]
    curvatures = curvatures[curvatures != 0]

    features = {
        'movement_count': n_points,
        'total_distance': np.sum(distances),
        'avg_speed': np.mean(speeds) if len(speeds) > 0 else 0,
        'std_speed': np.std(speeds) if len(speeds) > 0 else 0,
        'max_speed': np.max(speeds) if len(speeds) > 0 else 0,
        'avg_acceleration': np.mean(accelerations) if len(accelerations) > 0 else 0,
        'std_acceleration': np.std(accelerations) if len(accelerations) > 0 else 0,
        'avg_curvature': np.mean(curvatures) if len(curvatures) > 0 else 0,
        'duration_ms': duration_ms
    }

    return features

def process_dataset(input_file, output_file):
    """
    Process the raw dataset and extract features for machine learning

    Args:
        input_file: Path to the input CSV file with raw data
        output_file: Path to save the processed features
        
    Returns:
        DataFrame with extracted features
    """
    print(f"Loading raw data from {input_file}...")
    
    try:
        df = pd.read_csv(input_file)
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        return None

    print(f"Found {len(df)} records")
    print(f"Columns: {df.columns.tolist()}")

    # Detect mouse movement column
    mouse_columns = [col for col in df.columns if 'mouse' in col.lower()]
    if not mouse_columns:
        print("Error: No column containing mouse movement data found.")
        return None
    
    mouse_column = mouse_columns[0]
    print(f"Extracting features from column: {mouse_column}...")

    features_list = []
    
    # Use tqdm for progress bar
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting features"):
        try:
            mouse_data = row[mouse_column]
            features = calculate_features(mouse_data)
            
            # Preserve all other columns as metadata
            for col in df.columns:
                if col != mouse_column:
                    features[col] = row[col]
                    
            features_list.append(features)
        except Exception as e:
            print(f"Error processing row {idx}: {str(e)}")

    if not features_list:
        print("Error: No features were extracted.")
        return None
        
    features_df = pd.DataFrame(features_list)
    
    # Create output directory if it doesn't exist
    Path(os.path.dirname(output_file)).mkdir(parents=True, exist_ok=True)
    features_df.to_csv(output_file, index=False)

    print(f"Features saved to {output_file}")
    print(f"Feature columns: {features_df.columns.tolist()}")

    return features_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract features from mouse movement data")
    parser.add_argument("--input", default="data/processed/dataset.csv", help="Path to input dataset")
    parser.add_argument("--output", default="data/processed/features.csv", help="Path to save features")
    args = parser.parse_args()

    process_dataset(args.input, args.output)