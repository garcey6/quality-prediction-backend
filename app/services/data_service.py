import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def preprocess_data(data, target_column=None):
    X = data.drop(columns=[target_column]) if target_column else data
    
    for col in X.select_dtypes(include=[np.number]).columns:
        q1, q3 = X[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        X[col] = np.where((X[col] < lower) | (X[col] > upper), np.nan, X[col])
    X.fillna(X.median(), inplace=True)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled