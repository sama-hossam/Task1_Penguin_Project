import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import TargetEncoder

def preprocess_data(file_path, feat1, feat2, class1, class2):
   
    
    
    df = pd.read_csv(file_path)
   
    numeric_cols = ['CulmenLength', 'CulmenDepth', 'FlipperLength', 'BodyMass']
    for col in numeric_cols:
        df[col] = df[col].fillna(df.groupby('Species')[col].transform('mean'))
    
    
    df_filtered = df[df['Species'].isin([class1, class2])].copy()
    
    
    le = LabelEncoder()
    df_filtered['Target'] = le.fit_transform(df_filtered['Species'])
    
   
   # target encoding for OriginLocation
    te = TargetEncoder()
    df_filtered['OriginLocation'] = te.fit_transform( df_filtered['OriginLocation'],df_filtered['Species'])
    
   
    X = df_filtered[[feat1, feat2]].values.astype(float)
    y = df_filtered['Target'].values
    
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.4,
        stratify=y,
        random_state=42
    )
    
    return X_train, y_train, X_test, y_test