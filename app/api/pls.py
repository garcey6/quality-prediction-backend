from flask import Blueprint, request, jsonify, current_app
import pandas as pd
import os
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

pls_bp = Blueprint('pls', __name__)

@pls_bp.route('/api/pls/predict', methods=['POST'])
def pls_prediction():
    try:
        # 检查请求数据
        if not request.is_json:
            return jsonify({'status': 'error', 'message': '请求必须是JSON格式'}), 400
            
        data = request.json
        
        # 读取PCA特征提取结果
        pca_path = os.path.join(current_app.config['UPLOAD_FOLDER'], 'feature_extraction_result.csv')
        if not os.path.exists(pca_path):
            return jsonify({
                'status': 'error',
                'message': '请先进行PCA特征提取'
            }), 400
            
        # 读取质量标签数据
        label_path = os.path.join(current_app.config['UPLOAD_FOLDER'], 'data_.csv')
        if not os.path.exists(label_path):
            return jsonify({
                'status': 'error',
                'message': '缺少质量标签文件quality_labels.csv'
            }), 400
            
        # 加载数据
        X = pd.read_csv(pca_path)
        y = pd.read_csv(label_path)
        
        # 确保数据形状匹配
        if len(X) != len(y):
            return jsonify({
                'status': 'error',
                'message': 'PCA特征与标签数据行数不匹配'
            }), 400
            
        # 确保标签数据是单列
        if len(y.columns) > 1:
            y = y.iloc[:, 0]  # 只取第一列作为标签
        
        # 数据分割
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
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
        cv_scores = cross_val_score(pls, X, y, cv=5, scoring='r2')
        
        return jsonify({
            'status': 'success',
            'message': 'PLS预测完成',
            'data': {
                'mse': mse,
                'r2_score': r2,
                'cv_r2_scores': cv_scores.tolist(),  # 添加交叉验证结果
                'n_components': n_components,
                'predictions': y_pred.tolist()
            }
        })
        
    except Exception as e:
        current_app.logger.error(f'PLS预测错误: {str(e)}', exc_info=True)
        return jsonify({
            'status': 'error',
            'message': f'PLS预测失败: {str(e)}',
            'debug_info': {
                'pca_path': pca_path,
                'label_path': label_path,
                'data_shapes': {
                    'X_shape': X.shape if 'X' in locals() else None,
                    'y_shape': y.shape if 'y' in locals() else None
                }
            }
        }), 500