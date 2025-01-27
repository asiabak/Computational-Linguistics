import re
import pandas as pd

def clean_text(text):
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    # Remove @mentions
    text = re.sub(r'@\w+', '', text)
    # Handle hashtags (keep the text without #)
    text = re.sub(r'#(\w+)', r'\1', text)
    # Remove special characters and extra whitespace
    text = re.sub(r'[^\w\s]', ' ', text)
    # Normalize whitespace
    text = ' '.join(text.split())
    # Convert to lowercase
    text = text.lower()
    return text.strip()

def process_file(input_path, output_path):
    # Read the file using pandas, specifying no header and column names
    df = pd.read_csv(input_path,
                     sep='\t',
                     header=None,
                     names=['latitude', 'longitude', 'text'])

    # Clean the text column
    df['text'] = df['text'].astype(str).apply(clean_text)

    # Save to new file, preserving tab separation and removing index
    df.to_csv(output_path, sep='\t', index=False, header=False)

# Example usage
input_file = "test_blind.txt"
output_file = "test_blind_clean.txt"