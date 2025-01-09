import numpy as np
import pandas as pd
import random

file_path = '../dataset/Stars.csv'
stars_df = pd.read_csv(file_path)

# Synthetic data generator
def generate_synthetic_data(original_df, num_samples):
    synthetic_data = {}

    # Generate synthetic numerical data
    for column in ["Temperature (K)", "Luminosity (L/Lo)", "Radius (R/Ro)", "Absolute magnitude (Mv)"]:
        synthetic_data[column] = np.random.normal(
            loc=original_df[column].mean(),
            scale=original_df[column].std(),
            size=num_samples
        )

    # Generate synthetic categorical data
    for column in ["Star type", "Star category", "Star color", "Spectral Class"]:
        categories = original_df[column].unique()
        probabilities = original_df[column].value_counts(normalize=True).reindex(categories, fill_value=0).values
        synthetic_data[column] = np.random.choice(categories, size=num_samples, replace=True, p=probabilities)

    # Create a DataFrame for synthetic data
    synthetic_df = pd.DataFrame(synthetic_data)

    # Ensure data types match the original
    for column in ["Temperature (K)", "Star type"]:
        synthetic_df[column] = synthetic_df[column].astype(int)

    return synthetic_df

# Generate 100 synthetic samples
synthetic_samples = generate_synthetic_data(stars_df, 100)

# Save synthetic data to a new CSV file
synthetic_file_path = '../dataset/Synthetic_Stars.csv'
synthetic_samples.to_csv(synthetic_file_path, index=False)

print(f"Synthetic data generated and saved to: {synthetic_file_path}")
