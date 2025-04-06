import os
import json
import pandas as pd

def create_dataset():
    data_directory = 'data/raw/'
    processed_directory = 'data/processed/'
    all_data = []

    # Create processed directory if it doesn't exist
    os.makedirs(processed_directory, exist_ok=True)

    for filename in os.listdir(data_directory):
        if filename.endswith('.json'):
            with open(os.path.join(data_directory, filename), 'r') as f:
                data = json.load(f)
                all_data.append(data)

    # Convert to DataFrame
    df = pd.DataFrame(all_data)

    # Save to CSV
    df.to_csv(os.path.join(processed_directory, 'dataset.csv'), index=False)

if __name__ == '__main__':
    create_dataset()