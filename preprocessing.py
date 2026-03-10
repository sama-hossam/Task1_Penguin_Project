import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split



def preprocess_data(file_path, feat1, feat2, class1, class2):
   
    
    
    df = pd.read_csv(file_path)
   
    numeric_cols = ['CulmenLength', 'CulmenDepth', 'FlipperLength', 'BodyMass']
    for col in numeric_cols:
        df[col] = df[col].fillna(df.groupby('Species')[col].transform('mean'))

    
    # label encoding for OriginLocation
    le = LabelEncoder()
    df['OriginLocation'] = le.fit_transform(df['OriginLocation'])
    df['OriginLocation']=df['OriginLocation']

    df_filtered = df[df['Species'].isin([class1, class2])].copy()
    
    
    le = LabelEncoder()
    df_filtered['Target'] = le.fit_transform(df_filtered['Species'])
    df_filtered['Target'] = df_filtered['Target'].replace(0, -1)
   
   
    X = df_filtered[[feat1, feat2]].values.astype(float)
    y = df_filtered['Target'].values
    
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.4,
        stratify=y,
        random_state=42
    )
    scale=StandardScaler()
    X_train=scale.fit_transform(X_train)
    X_test=scale.transform(X_test)
    
    return X_train, y_train, X_test, y_test