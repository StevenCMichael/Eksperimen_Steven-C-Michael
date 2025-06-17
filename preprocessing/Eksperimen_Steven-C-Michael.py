import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os

def preprocess_encode_and_randomize(df):
    label_encoder = LabelEncoder()
    df['Quality_encoded'] = label_encoder.fit_transform(df['Quality'])
    df_processed = df.drop('Quality', axis=1)
    df_randomized = df_processed.sample(frac=1, random_state=42).reset_index(drop=True)
    return df_randomized

if __name__ == '__main__':
    try:
        input_path = 'banana_quality.csv'
        output_path = 'preprocessing/cleaned_banana_quality.csv'
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        raw_df = pd.read_csv(input_path)
        processed_df = preprocess_encode_and_randomize(raw_df.copy())
        processed_df.to_csv(output_path, index=False)

        print(f"Preprocessing complete. Saved to: {output_path}")
    except FileNotFoundError:
        print(f"Error: File {input_path} not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
