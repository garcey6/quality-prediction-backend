from flask import Blueprint, request, jsonify, current_app
import pandas as pd
import os
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from tensorflow.keras.optimizers import Adam

lstm_bp = Blueprint('lstm', __name__)

@lstm_bp.route('/api/lstm/predict', methods=['POST'])
def lstm_prediction():
    try:
        # 检查请求数据
        if not request.is_json:
            return jsonify({'status': 'error', 'message': '请求必须是JSON格式'}), 400
            
        data = request.json
        
        # 读取数据文件
        data_path = os.path.join(current_app.root_path,'uploads',f'latest_upload.csv')
        if not os.path.exists(data_path):
            return jsonify({
                'status': 'error',
                'message': '请先上传数据文件'
            }), 400
            
        # 加载数据
        df = pd.read_csv(data_path)
        
        # 检查数据列
        selection_path = os.path.join(current_app.root_path,'uploads',f'feature_selection_result.csv')
        if not os.path.exists(selection_path):
            return jsonify({
                'status': 'error',
                'message': '请先进行特征选择'
            }), 400
            
        selected_df = pd.read_csv(selection_path)
        target_column = selected_df['target'].iloc[0]
        
        if target_column not in df.columns:
            return jsonify({
                'status': 'error',
                'message': f'数据文件中缺少目标变量列: {target_column}'
            }), 400
            
        # 分离特征和标签
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # 数据标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 创建时间序列数据
        def create_dataset(X, y, time_step=50):
            X_data, y_data = [], []
            for i in range(len(X) - time_step):
                X_data.append(X[i:(i + time_step)])
                y_data.append(y[i + time_step])
            return np.array(X_data), np.array(y_data)
            
        time_step = 50
        X_data, y_data = create_dataset(X_scaled, y.values, time_step)
        
        # 检查并确保数据是三维的 (样本数, 时间步长, 特征数)
        if len(X_data.shape) != 3:
            return jsonify({
                'status': 'error',
                'message': f'数据维度错误，需要3维数据，当前维度: {X_data.shape}'
            }), 400
            
        # 不再进行数据分割，直接使用全部数据
        X_train = X_data
        y_train = y_data
        
        # 获取请求参数中的学习率，默认0.001
        learning_rate = float(data.get('learning_rate', 0.001))
        
        # 创建LSTM模型
        model = Sequential()
        model.add(LSTM(units=64, return_sequences=False, 
                      input_shape=(X_train.shape[1], X_train.shape[2]),
                      kernel_initializer='glorot_uniform'))  # 修改初始化方式
        model.add(Dense(units=1, activation='relu'))  # 添加relu激活函数
        
        # 调整学习率
        learning_rate = float(data.get('learning_rate', 0.01))  # 默认改为0.01
        
        # 使用传入的学习率配置优化器
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='mean_squared_error')
        
        # 增加训练轮次
        epochs = 50  # 增加到100轮
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=0)
        
        # 预测 - 直接使用训练数据进行预测
        y_pred = model.predict(X_train)
        
        # 处理预测结果 - 将小于0的值设为0
        y_pred = np.where(y_pred < 0, 0, y_pred)
        
        # 评估模型
        mse = mean_squared_error(y_train, y_pred)
        r2 = r2_score(y_train, y_pred)
        
        # 保存预测结果
        result_df = pd.DataFrame({
            '实际值': y_train.flatten(),
            '预测值': y_pred.flatten()
        })
        result_path = os.path.join(current_app.root_path,'uploads',f'lstm_predictions.csv')
        result_df.to_csv(result_path, index=False)
        
        # 检查标签数据是否为常量
        if len(np.unique(y)) == 1:
            return jsonify({
                'status': 'error',
                'message': '标签数据为常量值，无法进行有效预测'
            }), 400
            
        # 添加交叉验证评估
        if len(X_data) > 1000:  # 大数据集时不执行交叉验证
            cv_scores = []
        else:
            from sklearn.model_selection import KFold
            kfold = KFold(n_splits=5)
            cv_scores = []
            temp_model = Sequential()
            temp_model.add(LSTM(50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
            temp_model.add(Dense(1))
            temp_model.compile(optimizer='adam', loss='mse')
            
            for train, test in kfold.split(X_data):
                temp_model.fit(X_data[train], y_data[train], epochs=10, batch_size=32, verbose=0)
                score = temp_model.evaluate(X_data[test], y_data[test], verbose=0)
                cv_scores.append(score)
        
        return jsonify({
            'status': 'success',
            'message': 'LSTM预测完成',
            'data': {
                'mse': mse,
                'r2_score': r2,
                'epochs': epochs,  # 返回epochs数量
                'learning_rate': learning_rate,  # 返回使用的学习率
                'predictions': y_pred.tolist()
            }
        })
        
    except Exception as e:
        current_app.logger.error(f'LSTM预测错误: {str(e)}', exc_info=True)
        return jsonify({
            'status': 'error',
            'message': f'LSTM预测失败: {str(e)}'
        }), 500