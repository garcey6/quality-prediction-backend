from flask import Blueprint, request, jsonify, current_app
import pandas as pd
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from sklearn.feature_selection import mutual_info_regression
from scipy.stats import pearsonr

feature_selection_bp = Blueprint('feature_selection', __name__)

# 新增热力图生成接口
@feature_selection_bp.route('/api/feature-selection/heatmap', methods=['GET'])
def get_heatmap():
    try:
        file_path = os.path.join(current_app.root_path,'uploads',f'latest_upload.csv')
        if not os.path.exists(file_path):
            return jsonify({'status': 'error', 'message': '请先上传数据文件'}), 400
            
        df = pd.read_csv(file_path)
        
        # 计算相关系数矩阵
        numeric_cols = df.select_dtypes(include=['number']).columns
        corr_matrix = df[numeric_cols].corr()
        
        # 生成热力图
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', 
                   cbar=True, square=True)
        plt.title('Feature Correlation Heatmap')
        
        # 转换为base64编码
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()
        
        return jsonify({
            'status': 'success',
            'image': image_base64
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'生成热力图失败: {str(e)}'
        }), 500

# 修改特征选择接口
@feature_selection_bp.route('/api/feature-selection/select', methods=['POST'])
def feature_selection():
    try:
        if not request.is_json:
            return jsonify({'error': '请求必须是JSON格式'}), 400
            
        data = request.json
        threshold = data.get('threshold', 0.5)  # 默认阈值设为0.5
        method = data.get('method', 'pearson')  # 默认使用pearson方法
        
        if threshold is None:
            return jsonify({'error': '缺少threshold参数'}), 400
            
        try:
            threshold = float(threshold)
        except ValueError:
            return jsonify({'error': 'threshold必须是数值类型'}), 400
            
        # 从上传的文件获取数据
        filepath = os.path.join(current_app.root_path,'uploads',f'latest_upload.csv')
        if not os.path.exists(filepath):
            return jsonify({
                'status': 'error',
                'code': 400,
                'message': '请先上传数据文件',
                'data': None
            }), 400
            
        try:
            df = pd.read_csv(filepath, header=0, keep_default_na=False)
        except Exception as e:
            return jsonify({
                'status': 'error', 
                'code': 500,
                'message': f'读取数据文件失败: {str(e)}',
                'data': None
            }), 500

        # 读取特征选择结果文件获取目标变量
        selection_path = os.path.join(current_app.root_path,'uploads',f'feature_selection_result.csv')
        if not os.path.exists(selection_path):
            return jsonify({
                'status': 'error',
                'message': '请先进行特征选择'
            }), 400
            
        selected_df = pd.read_csv(selection_path)
        actual_target_var = selected_df['target'].iloc[0]
        
        # 分离特征和目标变量
        X = df.drop(columns=[actual_target_var])
        y = df[actual_target_var]
        
        # 移除非数值列（如batch_id, date等）
        numeric_cols = X.select_dtypes(include=['number']).columns
        X = X[numeric_cols]
        
        # 检查并处理缺失值
        if X.isnull().any().any() or y.isnull().any():
            X = X.fillna(X.mean())
            y = y.fillna(y.mean())
            
        # 过滤掉常量特征
        non_constant_cols = [col for col in X.columns if X[col].nunique() > 1]
        X = X[non_constant_cols]
        
        # 检查数据有效性
        if len(X.columns) == 0:
            return jsonify({
                'status': 'error',
                'code': 400,
                'message': '没有可用的数值型特征列',
                'data': {
                    'available_numeric_columns': numeric_cols.tolist()
                }
            }), 400

        # 根据选择的方法进行特征选择
        if method == 'pearson':
            selected_features = []
            feature_scores = []
            for col in X.columns:
                if X[col].nunique() == 1:
                    continue
                corr, _ = pearsonr(X[col], y)
                if abs(corr) >= threshold:
                    selected_features.append(col)
                    feature_scores.append(abs(corr))
                    
        elif method == 'mutual_info':
            mi_scores = mutual_info_regression(X, y)
            selected_features = [col for col, score in zip(X.columns, mi_scores) 
                               if score >= threshold]
            feature_scores = [score for score in mi_scores 
                            if score >= threshold]
        
        # 保存选择结果到文件
        result_path = os.path.join(current_app.root_path,'uploads',f'feature_selection_result.csv')
        result_df = pd.DataFrame({
            'feature': selected_features,
            'score': feature_scores,
            'target': [actual_target_var] * len(selected_features)
        })
        result_df.to_csv(result_path, index=False)

        return jsonify({
            'status': 'success',
            'message': f'使用{method}方法选择了{len(selected_features)}个特征',
            'selected_features': selected_features,
            'scores': feature_scores
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'特征选择过程中出错: {str(e)}'
        }), 500