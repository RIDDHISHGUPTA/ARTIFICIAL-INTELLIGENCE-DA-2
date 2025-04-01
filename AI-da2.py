import os
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def preprocess_data(file_path, impute_strategy='mean', apply_pca=False, n_components=None):
    # Load the dataset
    df = pd.read_excel(file_path)
    
    # Convert to float to avoid dtype issues
    df = df.apply(pd.to_numeric, errors='coerce')
    
    # Handling missing values
    imputer = SimpleImputer(strategy=impute_strategy)
    df.iloc[:, :] = imputer.fit_transform(df)
    
    # Z-score Normalization
    scaler = StandardScaler()
    df.iloc[:, :] = scaler.fit_transform(df)
    
    # Dimensionality Reduction (PCA)
    if apply_pca and n_components is not None:
        pca = PCA(n_components=n_components)
        df = pd.DataFrame(pca.fit_transform(df))
    
    # Define output file path
    output_directory = "C:\\Users\\riddh\\OneDrive\\Desktop\\AI-DA2"
    output_file = os.path.join(output_directory, "processed_data.xlsx")
    
    # Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)
    
    # Save processed data
    df.to_excel(output_file, index=False, engine='openpyxl')
    
    return df

if __name__ == "__main__":
    file_path = "C:\\Users\\riddh\\OneDrive\\Desktop\\AI-DA2\\dataset.xlsx"
    print("Loading Dataset and Preprocessing...")
    processed_df = preprocess_data(file_path, impute_strategy='mean', apply_pca=True, n_components=2)
    print("\nPreprocessed Dataset:")
    print(processed_df)