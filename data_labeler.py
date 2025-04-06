# data_labeler.py
import pandas as pd
import os
import argparse

def label_data(input_file, output_file, label_map=None):
    """
    Label the processed data based on file patterns or manually.
    
    Args:
        input_file: Path to the processed features CSV file
        output_file: Path to save the labeled dataset
        label_map: Dictionary mapping patterns to labels (e.g., {'bot': ['script', 'automated']})
    """
    df = pd.read_csv(input_file)
    
    # Initialize label column
    if 'label' not in df.columns:
        df['label'] = None
    
    # Automatic labeling based on patterns if label_map is provided
    if label_map:
        for label, patterns in label_map.items():
            for pattern in patterns:
                # Label based on file_name or user_agent containing pattern
                mask = (df['file_name'].str.contains(pattern, case=False, na=False) | 
                       df['user_agent'].str.contains(pattern, case=False, na=False))
                df.loc[mask, 'label'] = label
    
    # Identify unlabeled data
    unlabeled = df[df['label'].isna()]
    if not unlabeled.empty:
        print(f"{len(unlabeled)} records remain unlabeled.")
        
        # Save unlabeled data for manual review
        unlabeled_path = os.path.splitext(output_file)[0] + "_unlabeled.csv"
        unlabeled.to_csv(unlabeled_path, index=False)
        print(f"Unlabeled data saved to {unlabeled_path} for manual review.")
    
    # Save the labeled dataset (excluding unlabeled data)
    labeled = df.dropna(subset=['label'])
    labeled.to_csv(output_file, index=False)
    print(f"Labeled dataset saved to {output_file} with {len(labeled)} records.")
    
    return labeled

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Label behavioral data for CAPTCHA")
    parser.add_argument("--input", required=True, help="Input CSV file with processed features")
    parser.add_argument("--output", required=True, help="Output CSV file for labeled data")
    args = parser.parse_args()
    
    # Example label map - customize based on your data
    label_map = {
        'bot': ['selenium', 'chromedriver', 'phantomjs', 'automated', 'script'],
        'human': ['manual_verification']
    }
    
    label_data(args.input, args.output, label_map)