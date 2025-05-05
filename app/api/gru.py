from flask import Blueprint, request, jsonify, current_app
import pandas as pd
import os
import numpy as np
from keras.models import Sequential
from keras.layers import GRU, Dense  # 主要修改：使用GRU层替代SimpleRNN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.optimizers import Adam

gru_bp = Blueprint('gru', __name__)

@gru_bp.route('/api/gru/predict', methods=['POST'])
def gru_prediction():
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
        
        # 检查数据维度
        if len(X_data.shape) != 3:
            return jsonify({
                'status': 'error',
                'message': f'数据维度错误，需要3维数据，当前维度: {X_data.shape}'
            }), 400
            
        # 创建GRU模型
        model = Sequential()
        model.add(GRU(units=64,  # GRU单元数可以比RNN多一些
                    return_sequences=False,
                    input_shape=(X_data.shape[1], X_data.shape[2])))
        model.add(Dense(units=1, activation='linear'))
        
        # 使用较小的学习率
        learning_rate = float(data.get('learning_rate', 0.001))
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='mean_squared_error')
        
        # 训练参数调整
        epochs = 50  # GRU需要比RNN更少的训练轮次
        batch_size = 32
        
        # 训练模型
        history = model.fit(X_data, y_data, 
                         epochs=epochs,
                         batch_size=batch_size,
                         verbose=0)
        
        # 预测
        y_pred = model.predict(X_data)
        y_pred = np.where(y_pred < 0, 0, y_pred)
        
        # 评估
        mse = mean_squared_error(y_data, y_pred)
        r2 = r2_score(y_data, y_pred)
        
        # 保存结果
        result_df = pd.DataFrame({
            '实际值': y_data.flatten(),
            '预测值': y_pred.flatten()
        })
        result_path = os.path.join(current_app.root_path,'uploads',f'gru_predictions.csv')
        result_df.to_csv(result_path, index=False)
        
        return jsonify({
            'status': 'success',
            'message': 'GRU预测完成',
            'data': {
                'mse': float(mse),
                'r2_score': float(r2),
                'predictions': y_pred.flatten().tolist()
            }
        })
        
    except Exception as e:
        current_app.logger.error(f'GRU预测错误: {str(e)}', exc_info=True)
        return jsonify({
            'status': 'error',
            'message': f'GRU预测失败: {str(e)}'
        }), 500