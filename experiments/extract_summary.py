import pandas as pd
import numpy as np

# Load your full dataset
df = pd.read_csv('data/raw/astram_events.csv')

with open('dataset_summary.txt', 'w', encoding='utf-8') as f:
    f.write("## DATASET STRUCTURE\n")
    f.write(f"Total Rows: {df.shape[0]} | Total Columns: {df.shape[1]}\n\n")

    f.write("## HIGH-SIGNAL DATA SAMPLING\n")
    f.write(df.sample(5).to_string() + "\n")

    f.write("\n## DATA TYPES & MISSING VALUES\n")
    missing_info = pd.DataFrame({
        'Data Type': df.dtypes,
        'Missing Values': df.isnull().sum(),
        'Missing %': (df.isnull().sum() / len(df)) * 100
    })
    f.write(missing_info.to_string() + "\n")

    f.write("\n## STATISTICAL SPREAD (Numerical Columns)\n")
    f.write(df.describe().to_string() + "\n")

    f.write("\n## CARDINALITY (Categorical Columns)\n")
    for col in df.select_dtypes(include=['object', 'category']).columns:
        f.write(f"\n### {col} Value Distribution (Top 10):\n")
        f.write(df[col].value_counts().head(10).to_string() + "\n")
