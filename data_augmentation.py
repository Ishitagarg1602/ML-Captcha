# data_augmentation.py
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import argparse

def augment_data(input_file="data/processed/features.csv", output_file="data/processed/augmented_dataset.csv", augmentation_factor=2):
    """
    Augment the labeled dataset to improve model training.
    
    Args:
        input_file: Path to the labeled dataset CSV (default: data/processed/dataset.csv)
        output_file: Path to save the augmented dataset
        augmentation_factor: How many times to augment each class
    """
    # Verify input file exists
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
        
    print(f"Reading data from {input_file}")
    df = pd.read_csv(input_file)
    
    # Ensure the label column exists
    if 'label' not in df.columns:
        raise ValueError("Dataset must contain a 'label' column")
    
    # Separate by label
    human_data = df[df['label'] == 'human']
    bot_data = df[df['label'] == 'bot']
    
    print(f"Original dataset contains {len(human_data)} human samples and {len(bot_data)} bot samples")
    
    augmented_data = []
    
    # Augment human data with random noise
    for i in range(augmentation_factor):
        augmented_humans = human_data.copy()
        numeric_cols = augmented_humans.select_dtypes(include=[np.number]).columns
        
        # Add random noise (±5%)
        for col in numeric_cols:
            if col not in ['label']:
                noise = np.random.normal(0, 0.05 * augmented_humans[col].std(), len(augmented_humans))
                augmented_humans[col] = augmented_humans[col] + noise
        
        augmented_data.append(augmented_humans)
        print(f"Generated human augmentation batch {i+1}")
    
    # Augment bot data with more systematic patterns
    for i in range(augmentation_factor):
        augmented_bots = bot_data.copy()
        numeric_cols = augmented_bots.select_dtypes(include=[np.number]).columns
        
        # Make some features more "bot-like"
        if 'mouse_std_speed' in numeric_cols:
            augmented_bots['mouse_std_speed'] *= np.random.uniform(0.7, 0.9, len(augmented_bots))
        
        if 'mouse_mean_curvature' in numeric_cols:
            augmented_bots['mouse_mean_curvature'] *= np.random.uniform(0.6, 0.8, len(augmented_bots))
        
        # Add some random noise to other features
        for col in numeric_cols:
            if col not in ['label', 'mouse_std_speed', 'mouse_mean_curvature']:
                noise = np.random.normal(0, 0.03 * augmented_bots[col].std(), len(augmented_bots))
                augmented_bots[col] = augmented_bots[col] + noise
        
        augmented_data.append(augmented_bots)
        print(f"Generated bot augmentation batch {i+1}")
    
    # Combine original and augmented data
    augmented_df = pd.concat([df] + augmented_data, ignore_index=True)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save augmented dataset
    augmented_df.to_csv(output_file, index=False)
    print(f"Augmented dataset saved to {output_file} with {len(augmented_df)} records.")
    print(f"New label distribution: {augmented_df['label'].value_counts().to_dict()}")
    
    return augmented_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Augment labeled data for CAPTCHA")
    parser.add_argument("--input", default="data/processed/dataset.csv", help="Input CSV file with labeled data")
    parser.add_argument("--output", default="data/processed/augmented_dataset.csv", help="Output CSV file for augmented data")
    parser.add_argument("--factor", type=int, default=2, help="Augmentation factor")
    args = parser.parse_args()
    
    augment_data(args.input, args.output, args.factor)