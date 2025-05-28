from flask import Blueprint, request, jsonify, current_app
import pandas as pd
import os
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.optimizers import Adam
from keras.layers import Input
from keras.models import Model
from tcn import TCN  # 新增TCN导入

# 修改Blueprint名称为tcn
tcn_bp = Blueprint('tcn', __name__)

@tcn_bp.route('/api/tcn/predict', methods=['POST'])  # 修改路由为/tcn
def tcn_prediction():  # 修改函数名
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
        
        # 创建时间序列数据 - 修改后的版本
        def create_dataset(X, y, time_step=50):
            X_data, y_data = [], []
            # 为前time_step-1个点创建填充数据
            for i in range(time_step):  # 修改为range(time_step)而不是range(time_step-1)
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
        
        # 检查数据维度
        if len(X_data.shape) != 3:
            return jsonify({
                'status': 'error',
                'message': f'数据维度错误，需要3维数据，当前维度: {X_data.shape}'
            }), 400
            
        # 创建TCN模型 - 调整后的版本
        inputs = Input(shape=(X_data.shape[1], X_data.shape[2]))
        tcn_layer = TCN(
            nb_filters=32,  # 减少过滤器数量
            kernel_size=2,  # 减小卷积核大小
            nb_stacks=3,    # 增加堆叠层数
            dilations=[1, 2, 4, 8, 16],  # 增加膨胀系数范围
            padding='causal',
            use_skip_connections=True,
            return_sequences=False,
            dropout_rate=0.1  # 添加dropout减少过拟合
        )(inputs)
        outputs = Dense(1, activation='linear')(tcn_layer)
        model = Model(inputs=inputs, outputs=outputs)
        
        # 调整学习率和优化器
        learning_rate = float(data.get('learning_rate', 0.0005))  # 降低学习率
        optimizer = Adam(learning_rate=learning_rate, clipnorm=1.0)  # 添加梯度裁剪
        
        # 使用更稳健的损失函数
        model.compile(optimizer=optimizer, loss='huber_loss')  # 改用huber损失
        
        # 调整训练参数
        epochs = 50  # 适当增加epochs
        batch_size = 64  # 增大batch size
        
        # 训练模型
        history = model.fit(X_data, y_data, 
                         epochs=epochs,
                         batch_size=batch_size,
                         verbose=0)
        
        # 预测
        y_pred = model.predict(X_data)
        y_pred = np.where(y_pred < 0, 0, y_pred)
        
        # 评估时只使用有完整时间窗口的数据
        eval_start = time_step - 1
        mse = mean_squared_error(y_data[eval_start:], y_pred[eval_start:])
        r2 = r2_score(y_data[eval_start:], y_pred[eval_start:])
        
        # 保存结果
        result_df = pd.DataFrame({
            '实际值': y_data.flatten(),
            '预测值': y_pred.flatten()
        })
        result_path = os.path.join(current_app.root_path,'uploads',f'tcn_predictions.csv')
        result_df.to_csv(result_path, index=False)
        
        return jsonify({
            'status': 'success',
            'message': 'TCN预测完成',  # 修改消息
            'data': {
                'mse': float(mse),
                'r2_score': float(r2),
                'predictions': y_pred.flatten().tolist()
            }
        })

    except Exception as e:
        current_app.logger.error(f'TCN预测错误: {str(e)}', exc_info=True)
        return jsonify({
            'status': 'error',
            'message': f'TCN预测失败: {str(e)}'  # 修改错误消息
        }), 500