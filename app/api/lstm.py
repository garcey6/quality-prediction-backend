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
        data_path = os.path.join(current_app.config['UPLOAD_FOLDER'], 'latest_upload.csv')
        if not os.path.exists(data_path):
            return jsonify({
                'status': 'error',
                'message': '请先上传数据文件'
            }), 400
            
        # 加载数据
        df = pd.read_csv(data_path)
        
        # 检查数据列
        selection_path = os.path.join(current_app.config['UPLOAD_FOLDER'], 'feature_selection_result.csv')
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
        def create_dataset(X, y, time_step=3):  # 修改时间步长为3
            X_data, y_data = [], []
            for i in range(len(X) - time_step):
                X_data.append(X[i:(i + time_step)])
                y_data.append(y[i + time_step])
            return np.array(X_data), np.array(y_data)
            
        time_step = 3  # 使用3个时间步长
        X_data, y_data = create_dataset(X_scaled, y.values, time_step)
        
        # 数据分割
        X_train, X_test, y_train, y_test = train_test_split(
            X_data, y_data, test_size=0.2, random_state=42
        )
        
        # 获取请求参数中的学习率，默认0.001
        learning_rate = float(data.get('learning_rate', 0.001))
        
        # 创建LSTM模型
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=False, input_shape=(X_train.shape[1], X_train.shape[2])))  # 修改为return_sequences=False
        model.add(Dense(units=1))
        
        # 使用传入的学习率配置优化器
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='mean_squared_error')  # 修改为mean_squared_error
        
        # 训练模型
        epochs = 20  # 增加epochs到100
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=0)
        
        # 预测
        y_pred = model.predict(X_test)
        
        # 评估模型
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # 保存模型
        model_dir = os.path.join(os.path.dirname(__file__), '../../models')
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, 'lstm_model.h5')
        model.save(model_path)
        
        # 保存预测结果
        result_df = pd.DataFrame({
            '实际值': y_test.flatten(),
            '预测值': y_pred.flatten()
        })
        result_path = os.path.join(current_app.config['UPLOAD_FOLDER'], 'lstm_predictions.csv')
        result_df.to_csv(result_path, index=False)
        
        # 检查标签数据是否为常量
        if len(np.unique(y)) == 1:
            return jsonify({
                'status': 'error',
                'message': '标签数据为常量值，无法进行有效预测'
            }), 400
            
        # 添加交叉验证评估
        # 修改交叉验证部分，避免重复训练
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
        
                # 在返回前添加调试打印
        print("\n===== LSTM预测调试信息 =====")
        print(f"数据集形状: X_train={X_train.shape}, y_train={y_train.shape}")
        print(f"训练参数: epochs={epochs}, learning_rate={learning_rate}")
        print(f"评估指标: MSE={mse:.4f}, R2={r2:.4f}")
        print("前5个样本预测结果:")
        print("===========================\n")
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