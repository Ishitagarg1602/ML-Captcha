# extract_features.py
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
            # If it's not valid JSON, return empty features
            return {
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
    
    # If empty data, return zeros
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
    
    # Calculate distances, speeds, and accelerations
    distances = []
    speeds = []
    accelerations = []
    curvatures = []
    
    for i in range(1, len(mouse_movements)):
        # Calculate distance between consecutive points
        prev = mouse_movements[i-1]
        curr = mouse_movements[i]
        
        dx = curr['x'] - prev['x']
        dy = curr['y'] - prev['y']
        distance = sqrt(dx*dx + dy*dy)
        time_diff = (curr['timestamp'] - prev['timestamp']) / 1000  # Convert to seconds
        
        # Handle zero time difference
        if time_diff > 0:
            speed = distance / time_diff
        else:
            speed = 0
            
        distances.append(distance)
        speeds.append(speed)
        
        # Calculate acceleration (change in speed)
        if i > 1 and speeds[i-2] > 0:
            acceleration = (speed - speeds[i-2]) / time_diff
            accelerations.append(acceleration)
        
        # Calculate curvature (needs 3 points)
        if i > 1:
            prev_prev = mouse_movements[i-2]
            # Simple approximation of curvature using angle change
            # Higher values mean sharper turns
            if distance > 0:
                dx1 = prev['x'] - prev_prev['x']
                dy1 = prev['y'] - prev_prev['y']
                dx2 = curr['x'] - prev['x']
                dy2 = curr['y'] - prev['y']
                
                # Normalize vectors
                len1 = sqrt(dx1*dx1 + dy1*dy1)
                len2 = sqrt(dx2*dx2 + dy2*dy2)
                
                if len1 > 0 and len2 > 0:
                    # Calculate dot product and determine angle
                    dx1, dy1 = dx1/len1, dy1/len1
                    dx2, dy2 = dx2/len2, dy2/len2
                    dot_product = dx1*dx2 + dy1*dy2
                    # Clamp to handle floating point errors
                    dot_product = max(-1, min(1, dot_product))
                    curvature = abs(1 - dot_product)  # 0 means straight line, 2 means 180° turn
                    curvatures.append(curvature)
    
    # Calculate duration in ms
    duration_ms = mouse_movements[-1]['timestamp'] - mouse_movements[0]['timestamp']
    
    # Compile features
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
    
    # Check for mouse_movements column
    mouse_column = next((col for col in df.columns if 'mouse' in col.lower()), None)
    if not mouse_column:
        print("Warning: No column containing mouse movement data found!")
        return
        
    print(f"Extracting features from {mouse_column}...")
    
    # Extract features
    features_list = []
    for idx, row in df.iterrows():
        # Get label
        label = row.get('label', row.get('is_bot', None))
        
        # Extract mouse movement features
        mouse_data = row[mouse_column]
        features = calculate_features(mouse_data)
        
        # Add label if it exists
        if label is not None:
            features['label'] = label
            
        features_list.append(features)
        
        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1} records...")
    
    # Create features dataframe
    features_df = pd.DataFrame(features_list)
    
    # Save to CSV
    features_df.to_csv(output_file, index=False)
    print(f"Features extracted and saved to {output_file}")
    print(f"Feature columns: {features_df.columns.tolist()}")
    
    return features_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract features from mouse movement data')
    parser.add_argument('--input', default='data/processed/dataset.csv', 
                        help='Path to input CSV file with raw data')
    parser.add_argument('--output', default='data/processed/features.csv', 
                        help='Path to save extracted features')
    args = parser.parse_args()
    
    process_dataset(args.input, args.output)