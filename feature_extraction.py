# feature_extraction.py
import os
import json
import pandas as pd
import numpy as np
import argparse
from math import sqrt

def calculate_features(mouse_movements):
    """
    Extract meaningful features from raw mouse movement data

    Args:
        mouse_movements: List of dictionaries with x, y, timestamp

    Returns:
        Dictionary of features
    """
    # Convert string to list of dictionaries if needed
    if isinstance(mouse_movements, str):
        try:
            mouse_movements = json.loads(mouse_movements)
        except json.JSONDecodeError:
            return {  # fallback if JSON is malformed
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

    # If empty or too short
    if not mouse_movements or len(mouse_movements) < 3:
        return {
            'movement_count': len(mouse_movements) if mouse_movements else 0,
            'total_distance': 0,
            'avg_speed': 0,
            'std_speed': 0,
            'max_speed': 0,
            'avg_acceleration': 0,
            'std_acceleration': 0,
            'avg_curvature': 0,
            'duration_ms': 0
        }

    distances, speeds, accelerations, curvatures = [], [], [], []

    for i in range(1, len(mouse_movements)):
        prev = mouse_movements[i - 1]
        curr = mouse_movements[i]

        dx = curr['x'] - prev['x']
        dy = curr['y'] - prev['y']
        distance = sqrt(dx**2 + dy**2)
        time_diff = (curr['timestamp'] - prev['timestamp']) / 1000  # seconds

        speed = distance / time_diff if time_diff > 0 else 0
        distances.append(distance)
        speeds.append(speed)

        # Acceleration
        if i > 1 and speeds[i - 2] > 0 and time_diff > 0:
            acceleration = (speed - speeds[i - 2]) / time_diff
            accelerations.append(acceleration)

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
                curvature = abs(1 - dot_product)
                curvatures.append(curvature)

    duration_ms = mouse_movements[-1]['timestamp'] - mouse_movements[0]['timestamp']

    features = {
        'movement_count': len(mouse_movements),
        'total_distance': sum(distances),
        'avg_speed': np.mean(speeds) if speeds else 0,
        'std_speed': np.std(speeds) if speeds else 0,
        'max_speed': max(speeds) if speeds else 0,
        'avg_acceleration': np.mean(accelerations) if accelerations else 0,
        'std_acceleration': np.std(accelerations) if accelerations else 0,
        'avg_curvature': np.mean(curvatures) if curvatures else 0,
        'duration_ms': duration_ms
    }

    return features

def process_dataset(input_file, output_file):
    """
    Process the raw dataset and extract features for machine learning

    Args:
        input_file: Path to the input CSV file with raw data
        output_file: Path to save the processed features
    """
    print(f"Loading raw data from {input_file}...")
    df = pd.read_csv(input_file)

    print(f"Found {len(df)} records")
    print(f"Columns: {df.columns.tolist()}")

    # Detect mouse movement column
    mouse_column = next((col for col in df.columns if 'mouse' in col.lower()), None)
    if not mouse_column:
        print("Error: No column containing mouse movement data found.")
        return

    print(f"Extracting features from column: {mouse_column}...")

    features_list = []
    for idx, row in df.iterrows():
        mouse_data = row[mouse_column]
        features = calculate_features(mouse_data)

        # Add label if exists
        label = row.get('label', row.get('is_bot', None))
        if label is not None:
            features['label'] = label

        features_list.append(features)

        if (idx + 1) % 100 == 0 or idx == len(df) - 1:
            print(f"Processed {idx + 1} / {len(df)}")

    features_df = pd.DataFrame(features_list)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
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
