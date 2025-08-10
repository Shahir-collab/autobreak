import pandas as pd

# Load the full malware dataset
df = pd.read_csv("malware_dataset.csv")

# Filter for a single benign sample
benign_sample = df[df['classification'] == 'benign'].sample(1, random_state=42)

# Drop non-feature columns
benign_sample_cleaned = benign_sample.drop(columns=['hash', 'classification'])

# Save to CSV
benign_sample_cleaned.to_csv("verified_benign.csv", index=False)

print("✅ Generated 'verified_benign.csv' successfully.")
