import os
import json
import pandas as pd

def create_dataset():
    # Define directories
    raw_human_directory = 'data/raw/human/'
    raw_bot_directory = 'data/raw/bot/'
    processed_directory = 'data/processed/'
    
    # Create processed directory if it doesn't exist
    os.makedirs(processed_directory, exist_ok=True)
    
    all_data = []
    
    # Process human data
    if os.path.exists(raw_human_directory):
        for filename in os.listdir(raw_human_directory):
            if filename.endswith('.json'):
                with open(os.path.join(raw_human_directory, filename), 'r') as f:
                    data = json.load(f)
                    # Add label for human data
                    data['label'] = 'human'
                    all_data.append(data)
    
    # Process bot data
    if os.path.exists(raw_bot_directory):
        for filename in os.listdir(raw_bot_directory):
            if filename.endswith('.json'):
                with open(os.path.join(raw_bot_directory, filename), 'r') as f:
                    data = json.load(f)
                    # Add label for bot data
                    data['label'] = 'bot'
                    all_data.append(data)
    
    # Convert to DataFrame
    df = pd.DataFrame(all_data)

    # Stringify the 'mouse_movements' column if it exists
    if 'mouse_movements' in df.columns:
        df['mouse_movements'] = df['mouse_movements'].apply(json.dumps)

    # Save to CSV
    df.to_csv(os.path.join(processed_directory, 'dataset.csv'), index=False)
    print(f"Created dataset with {len(df)} records")
    print(f"Label distribution: {df['label'].value_counts().to_dict()}")

if __name__ == '__main__':
    create_dataset()