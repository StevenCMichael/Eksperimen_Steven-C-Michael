import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os

def count_outliers(df):
    outlier_counts = {}
    for col in df.select_dtypes(include=np.number).columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outlier_counts[col] = len(outliers)
    return outlier_counts

def remove_outliers_iqr(df):
    df_cleaned = df.copy()
    for col in df_cleaned.select_dtypes(include=np.number).columns:
        Q1 = df_cleaned[col].quantile(0.25)
        Q3 = df_cleaned[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_cleaned = df_cleaned[(df_cleaned[col] >= lower_bound) & (df_cleaned[col] <= upper_bound)]
    return df_cleaned

def preprocess_encode_and_randomize(df):
    label_encoder = LabelEncoder()
    df['Quality_encoded'] = label_encoder.fit_transform(df['Quality'])
    df_encoded = df.drop('Quality', axis=1)

    outlier_counts = count_outliers(df_encoded)

    df_no_outlier = remove_outliers_iqr(df_encoded)

    df_randomized = df_no_outlier.sample(frac=1, random_state=42).reset_index(drop=True)
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

# Steven C Michael
