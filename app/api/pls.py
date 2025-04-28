from flask import Blueprint, request, jsonify, current_app
import pandas as pd
import os
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from sklearn.preprocessing import StandardScaler

pls_bp = Blueprint('pls', __name__)

@pls_bp.route('/api/pls/predict', methods=['POST'])
def pls_prediction():
    try:
        # 检查请求数据
        if not request.is_json:
            return jsonify({'status': 'error', 'message': '请求必须是JSON格式'}), 400
            
        data = request.json
        
        # 读取原始数据文件
        data_path = os.path.join(current_app.config['UPLOAD_FOLDER'], 'latest_upload.csv')
        if not os.path.exists(data_path):
            return jsonify({
                'status': 'error',
                'message': '请先上传数据文件'
            }), 400
            
        # 加载数据
        df = pd.read_csv(data_path)
        
        # 分离特征和标签
        if 'Penicillin concentration' not in df.columns:
            return jsonify({
                'status': 'error',
                'message': '数据文件中缺少"Penicillin concentration"列'
            }), 400
            
        X = df.drop(columns=['Penicillin concentration'])
        y = df['Penicillin concentration']
        
        # 数据标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 数据分割
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        # 获取请求参数或使用默认值
        n_components = data.get('n_components', 2)
        
        # 创建PLS模型
        pls = PLSRegression(n_components=n_components)
        pls.fit(X_train, y_train)
        
        # 预测
        y_pred = pls.predict(X_test)
        
        # 评估模型
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # 保存模型
        model_path = os.path.join(current_app.config['UPLOAD_FOLDER'], 'pls_model.pkl')
        pd.to_pickle(pls, model_path)
        
        # 保存预测结果
        result_df = pd.DataFrame({
            '实际值': y_test.values.flatten(),
            '预测值': y_pred.flatten()
        })
        result_path = os.path.join(current_app.config['UPLOAD_FOLDER'], 'pls_predictions.csv')
        result_df.to_csv(result_path, index=False)
        
        # 检查标签数据是否为常量
        if len(np.unique(y)) == 1:
            return jsonify({
                'status': 'error',
                'message': '标签数据为常量值，无法进行有效预测'
            }), 400
            
        # 添加交叉验证评估
        from sklearn.model_selection import cross_val_score
        cv_scores = cross_val_score(pls, X_scaled, y, cv=5, scoring='r2')
        
        return jsonify({
            'status': 'success',
            'message': 'PLS预测完成',
            'data': {
                'mse': mse,
                'r2_score': r2,
                'cv_r2_scores': cv_scores.tolist(),
                'n_components': n_components,
                'predictions': y_pred.tolist()
            }
        })
        
    except Exception as e:
        current_app.logger.error(f'PLS预测错误: {str(e)}', exc_info=True)
        return jsonify({
            'status': 'error',
            'message': f'PLS预测失败: {str(e)}'
        }), 500