from flask import Blueprint, request, jsonify, current_app
import pandas as pd
import os
import numpy as np
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.optimizers import Adam

rnn_bp = Blueprint('rnn', __name__)

@rnn_bp.route('/api/rnn/predict', methods=['POST'])
def rnn_prediction():
    try:
        # 检查请求数据
        if not request.is_json:
            return jsonify({'status': 'error', 'message': '请求必须是JSON格式'}), 400
            
        data = request.json
        
        # 获取前端参数
        rnn_type = data.get('rnn_type', 'SimpleRNN')
        network_params = data.get('network_params', {})
        train_params = data.get('train_params', {})
        
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
        
        # 创建时间序列数据 - 修改后的版本
        def create_dataset(X, y, time_step=50):
            X_data, y_data = [], []
            # 为前time_step个点创建填充数据
            for i in range(time_step):
                pad_size = time_step - i - 1
                padded_X = np.vstack([np.zeros((pad_size, X.shape[1])), X[:i+1]])
                X_data.append(padded_X)
                y_data.append(y[i])
            
            # 正常时间窗口数据
            for i in range(len(X) - time_step):
                X_data.append(X[i:(i + time_step)])
                y_data.append(y[i + time_step])
            return np.array(X_data), np.array(y_data)
            
        time_step = 50
        X_data, y_data = create_dataset(X_scaled, y.values, time_step)
        
        # 构建RNN模型 - 使用前端参数
        model = Sequential()
        model.add(SimpleRNN(
            units=network_params.get('hidden_units', 50),
            return_sequences=False,
            input_shape=(X_data.shape[1], X_data.shape[2]),
            activation='tanh'
        ))
        model.add(Dense(units=1, activation='linear'))
        
        # 设置学习率与优化器 - 使用前端参数
        learning_rate = train_params.get('learning_rate', 0.001)
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='mean_squared_error')
        
        # 训练模型参数调整 - 使用前端参数
        epochs = train_params.get('epochs', 100)
        batch_size = train_params.get('batch_size', 32)
        
        # 训练模型
        history = model.fit(
            X_data, y_data,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0
        )
        
        # 预测
        y_pred = model.predict(X_data)
        y_pred = np.where(y_pred < 0, 0, y_pred)
        
        # 评估
        # 评估时只使用有完整时间窗口的数据
        eval_start = time_step
        mse = mean_squared_error(y_data[eval_start:], y_pred[eval_start:])
        r2 = r2_score(y_data[eval_start:], y_pred[eval_start:])
        
        # 保存结果
        result_df = pd.DataFrame({
            '实际值': y_data.flatten(),
            '预测值': y_pred.flatten()
        })
        result_path = os.path.join(current_app.root_path,'uploads',f'rnn_predictions.csv')
        result_df.to_csv(result_path, index=False)
        
        return jsonify({
            'status': 'success',
            'message': 'RNN预测完成',
            'data': {
                'mse': float(mse),
                'r2_score': float(r2),
                'epochs': epochs,  # 返回实际使用的epochs数
                'predictions': y_pred.flatten().tolist()
            }
        })
        
    except Exception as e:
        current_app.logger.error(f'RNN预测错误: {str(e)}', exc_info=True)
        return jsonify({
            'status': 'error',
            'message': f'RNN预测失败: {str(e)}'
        }), 500