from flask import Blueprint, request, jsonify, current_app
import pandas as pd
import os
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

transformer_bp = Blueprint('transformer', __name__)

@transformer_bp.route('/api/transformer/predict', methods=['POST'])
def transformer_prediction():
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
        def create_dataset(X, y, time_step=20):
            X_data, y_data = [], []
            for i in range(len(X) - time_step):
                X_data.append(X[i:(i + time_step)])
                y_data.append(y[i + time_step])
            return np.array(X_data), np.array(y_data)
            
        time_step = 20
        X_data, y_data = create_dataset(X_scaled, y.values, time_step)
        
        # 检查数据维度
        if len(X_data.shape) != 3:
            return jsonify({
                'status': 'error',
                'message': f'数据维度错误，需要3维数据，当前维度: {X_data.shape}'
            }), 400

        # Transformer模型构建
        def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
            # 多头注意力机制
            x = MultiHeadAttention(
                key_dim=head_size, num_heads=num_heads, dropout=dropout
            )(inputs, inputs)
            x = Dropout(dropout)(x)
            x = LayerNormalization(epsilon=1e-6)(x + inputs)

            # 前馈网络
            x = Dense(ff_dim, activation="relu")(x)
            x = Dropout(dropout)(x)
            x = Dense(inputs.shape[-1])(x)
            x = LayerNormalization(epsilon=1e-6)(x + inputs)
            return x

        inputs = Input(shape=(X_data.shape[1], X_data.shape[2]))
        x = transformer_encoder(inputs, head_size=64, num_heads=4, ff_dim=128)
        x = GlobalAveragePooling1D()(x)
        outputs = Dense(1)(x)

        model = Model(inputs=inputs, outputs=outputs)
        
        # 设置学习率
        learning_rate = 0.001
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='mean_squared_error')

        # 训练模型
        model.fit(X_data, y_data, epochs=100, batch_size=32, verbose=0)
        
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
        result_path = os.path.join(current_app.root_path,'uploads',f'transformer_predictions.csv')
        result_df.to_csv(result_path, index=False)
        
        return jsonify({
            'status': 'success',
            'message': 'Transformer预测完成',
            'data': {
                'mse': float(mse),
                'r2_score': float(r2),
                'predictions': y_pred.flatten().tolist()
            }
        })
        
    except Exception as e:
        current_app.logger.error(f'Transformer预测错误: {str(e)}', exc_info=True)
        return jsonify({
            'status': 'error',
            'message': f'Transformer预测失败: {str(e)}'
        }), 500