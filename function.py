import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import mean_squared_error, r2_score

# 读取数据
df = pd.read_csv("./data/data_50_time.csv")

# 1. 变量选择 - 增加时间列检查
time_columns = [col for col in df.columns if 'time' in col.lower() or 'date' in col.lower()]
X = df.drop(columns=["Penicillin concentration"] + time_columns)  # 同时移除时间列
y = df["Penicillin concentration"]

# 2. 异常处理 - 添加空值检查
if X.isnull().any().any() or y.isnull().any():
    print("数据包含空值，请先处理缺失值")
    exit()

# 2. 异常处理（如用 Z-score）
z_scores = np.abs((X - X.mean()) / X.std())
X = X[(z_scores < 3).all(axis=1)]
y = y.loc[X.index]

# 3. 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. 特征选择（剔除低方差特征）
selector = VarianceThreshold(threshold=0.01)
X_selected = selector.fit_transform(X_scaled)

# 5. 特征提取 - 添加PCA方法
from sklearn.decomposition import PCA
pca = PCA(n_components=0.98)  # 保留95%的方差
X_pca = pca.fit_transform(X_selected)

# 6. PLS 建模 - 使用PCA处理后的数据
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

pls = PLSRegression(n_components=5)
pls.fit(X_train, y_train)
y_pred = pls.predict(X_test)

# 评估
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2:", r2_score(y_test, y_pred))
