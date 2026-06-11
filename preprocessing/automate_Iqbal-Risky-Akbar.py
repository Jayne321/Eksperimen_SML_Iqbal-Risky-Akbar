import os
import sys
import pandas as pd
from sklearn.preprocessing import StandardScaler

def run_preprocessing(raw_path, output_path):
    print(f"Starting preprocessing on: {raw_path}")
    if not os.path.exists(raw_path):
        print(f"Error: Raw data file not found at {raw_path}")
        sys.exit(1)
        
    df = pd.read_csv(raw_path)
    
    # 1. Handle Missing Values
    if 'total_bedrooms' in df.columns and df['total_bedrooms'].isnull().sum() > 0:
        df['total_bedrooms'] = df['total_bedrooms'].fillna(df['total_bedrooms'].median())

    # 2. Drop Duplicates
    df = df.drop_duplicates().reset_index(drop=True)
    
    # 3. Categorical Encoding
    if 'ocean_proximity' in df.columns:
        df = pd.get_dummies(df, columns=['ocean_proximity'], drop_first=True)
    
    # 4. Feature Scaling
    if 'median_house_value' in df.columns:
        X = df.drop(columns=['median_house_value'])
        y = df['median_house_value']
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        df_final = pd.DataFrame(X_scaled, columns=X.columns)
        df_final['median_house_value'] = y.values
    else:
        df_final = df
        
    # 5. Save directly as a file inside the correct path
    df_final.to_csv(output_path, index=False)
    print(f"Automation successful! Clean file saved directly at: {output_path}")

if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) 
    BASE_DIR = os.path.dirname(SCRIPT_DIR)                  
    
    RAW_PATH = os.path.join(BASE_DIR, "housing_raw.csv")
    OUTPUT_PATH = os.path.join(SCRIPT_DIR, "housing_preprocessing.csv") # Diperbarui
    
    run_preprocessing(RAW_PATH, OUTPUT_PATH)