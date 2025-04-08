import os
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm

def augment_data(input_file="data/processed/features.csv", 
                output_file="data/processed/augmented_dataset.csv", 
                augmentation_factor=2,
                random_seed=42):
    """
    Augment the labeled dataset to improve model training.
    
    Args:
        input_file: Path to the labeled dataset CSV
        output_file: Path to save the augmented dataset
        augmentation_factor: How many times to augment each class
        random_seed: For reproducibility
        
    Returns:
        DataFrame with augmented data
    """
    np.random.seed(random_seed)
    
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
        
    print(f"Reading data from {input_file}")
    try:
        df = pd.read_csv(input_file)
    except Exception as e:
        print(f"Error reading input file: {str(e)}")
        return None
    
    if 'label' not in df.columns:
        raise ValueError("Dataset must contain a 'label' column")
    
    human_data = df[df['label'] == 'human']
    bot_data = df[df['label'] == 'bot']
    
    if len(human_data) == 0 or len(bot_data) == 0:
        print("Warning: One or both classes have zero samples.")
    
    print(f"Original dataset contains {len(human_data)} human samples and {len(bot_data)} bot samples")
    
    # Identify numeric columns for augmentation
    numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns 
                   if col != 'label' and 'id' not in col.lower()]
    
    print(f"Found {len(numeric_cols)} numeric features for augmentation")
    
    augmented_data = [df]  # Start with original data

    # Calculate statistics once for efficiency
    human_stats = {col: {'mean': human_data[col].mean(), 'std': human_data[col].std()} 
                  for col in numeric_cols}
    bot_stats = {col: {'mean': bot_data[col].mean(), 'std': bot_data[col].std()}
                for col in numeric_cols}

    # Augment human data with random noise
    for i in tqdm(range(augmentation_factor), desc="Augmenting human data"):
        augmented_humans = human_data.copy()

        for col in numeric_cols:
            std_dev = human_stats[col]['std']
            if std_dev > 0:  # Avoid zero std dev
                noise = np.random.normal(0, 0.05 * std_dev, len(augmented_humans))
                # Ensure we don't create negative values for strictly positive features
                if augmented_humans[col].min() >= 0:
                    augmented_humans[col] = np.maximum(0, augmented_humans[col] + noise)
                else:
                    augmented_humans[col] += noise

        augmented_data.append(augmented_humans)
    
    # Augment bot data with systematic tweaks + noise
    for i in tqdm(range(augmentation_factor), desc="Augmenting bot data"):
        augmented_bots = bot_data.copy()

        # Special handling for certain features based on domain knowledge
        features_to_modify = {
            'std_speed': (0.7, 0.9),
            'avg_curvature': (0.6, 0.8),
            'avg_speed': (0.8, 1.0),
            'total_distance': (0.9, 1.1)
        }
        
        for col, (min_factor, max_factor) in features_to_modify.items():
            if col in augmented_bots.columns:
                augmented_bots[col] *= np.random.uniform(min_factor, max_factor, len(augmented_bots))

        # Add noise to all features
        for col in numeric_cols:
            std_dev = bot_stats[col]['std']
            if std_dev > 0 and col not in features_to_modify:
                noise = np.random.normal(0, 0.03 * std_dev, len(augmented_bots))
                # Ensure we don't create negative values for strictly positive features
                if augmented_bots[col].min() >= 0:
                    augmented_bots[col] = np.maximum(0, augmented_bots[col] + noise)
                else:
                    augmented_bots[col] += noise

        augmented_data.append(augmented_bots)
    
    # Combine and save
    augmented_df = pd.concat(augmented_data, ignore_index=True)
    
    # Create output directory if it doesn't exist
    Path(os.path.dirname(output_file)).mkdir(parents=True, exist_ok=True)
    
    augmented_df.to_csv(output_file, index=False)

    print(f"Augmented dataset saved to {output_file} with {len(augmented_df)} records.")
    print(f"New label distribution: {augmented_df['label'].value_counts().to_dict()}")
    
    return augmented_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Augment labeled data for CAPTCHA")
    parser.add_argument("--input", default="data/processed/features.csv", help="Input CSV file with labeled data")
    parser.add_argument("--output", default="data/processed/augmented_dataset.csv", help="Output CSV file for augmented data")
    parser.add_argument("--factor", type=int, default=2, help="Augmentation factor")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()
    
    augment_data(args.input, args.output, args.factor, args.seed)
