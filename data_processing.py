import os
import json
import pandas as pd
import argparse
from pathlib import Path

def create_dataset(raw_human_directory='data/raw/human/', 
                  raw_bot_directory='data/raw/bot/',
                  processed_directory='data/processed/'):
    """
    Process raw human and bot data to create a labeled dataset
    
    Args:
        raw_human_directory: Path to directory containing human data JSON files
        raw_bot_directory: Path to directory containing bot data JSON files
        processed_directory: Path to save the processed dataset
    
    Returns:
        DataFrame of the processed dataset
    """
    # Create processed directory using pathlib (more robust)
    Path(processed_directory).mkdir(parents=True, exist_ok=True)
    
    all_data = []
    
    # Process human data
    human_path = Path(raw_human_directory)
    if human_path.exists():
        for file_path in human_path.glob('*.json'):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    data['label'] = 'human'
                    all_data.append(data)
            except json.JSONDecodeError:
                print(f"Warning: Could not parse JSON in {file_path}")
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
    
    # Process bot data
    bot_path = Path(raw_bot_directory)
    if bot_path.exists():
        for file_path in bot_path.glob('*.json'):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    data['label'] = 'bot'
                    all_data.append(data)
            except json.JSONDecodeError:
                print(f"Warning: Could not parse JSON in {file_path}")
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
    
    if not all_data:
        print("Warning: No data was loaded. Check your input directories.")
        return None
        
    # Convert to DataFrame
    df = pd.DataFrame(all_data)

    # Stringify the 'mouse_movements' column if it exists
    if 'mouse_movements' in df.columns:
        df['mouse_movements'] = df['mouse_movements'].apply(lambda x: json.dumps(x) if not isinstance(x, str) else x)

    # Save to CSV
    output_path = Path(processed_directory) / 'dataset.csv'
    df.to_csv(output_path, index=False)
    
    print(f"Created dataset with {len(df)} records")
    print(f"Label distribution: {df['label'].value_counts().to_dict()}")
    
    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process raw data to create labeled dataset")
    parser.add_argument("--human_dir", default="data/raw/human/", help="Directory with human data")
    parser.add_argument("--bot_dir", default="data/raw/bot/", help="Directory with bot data")
    parser.add_argument("--output_dir", default="data/processed/", help="Directory to save processed data")
    args = parser.parse_args()
    
    create_dataset(args.human_dir, args.bot_dir, args.output_dir)