from flask import Blueprint, request, jsonify, current_app
import pandas as pd
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

feature_extraction_bp = Blueprint('feature_extraction', __name__)

@feature_extraction_bp.route('/api/feature_extraction/extract', methods=['POST'])
def feature_extraction():
    try:
        # 检查请求数据
        if not request.is_json:
            return jsonify({'status': 'error', 'message': '请求必须是JSON格式'}), 400
            
        data = request.json
        required_fields = ['method']  # 修改为只要求method参数
        for field in required_fields:
            if field not in data:
                return jsonify({'status': 'error', 'message': f'缺少{field}参数'}), 400
        
        # 从上传的文件获取数据
        filepath = os.path.join(current_app.root_path,'uploads',f'latest_upload.csv')
        if not os.path.exists(filepath):
            return jsonify({
                'status': 'error',
                'message': '请先上传数据文件'
            }), 400
            
        try:
            df = pd.read_csv(filepath, header=0)
        except Exception as e:
            return jsonify({
                'status': 'error', 
                'message': f'读取数据文件失败: {str(e)}'
            }), 500

        # 读取特征选择结果文件
        selection_path = os.path.join(current_app.root_path,'uploads',f'feature_selection_result.csv')
        if os.path.exists(selection_path):
            selected_df = pd.read_csv(selection_path)
            selected_features = selected_df['feature'].tolist()
        else:
            selected_features = data.get('features', [])
            if not selected_features:  # 如果没有提供features参数
                return jsonify({
                    'status': 'error',
                    'message': '请先进行特征选择或提供features参数'
                }), 400

        # 检查特征是否存在
        missing_features = [f for f in selected_features if f not in df.columns]
        if missing_features:
            return jsonify({
                'status': 'error',
                'message': f'以下特征不存在: {", ".join(missing_features)}'
            }), 400
            
        # 提取选中的特征数据
        X = df[selected_features]
        
        # 标准化数据
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 从请求中获取方差阈值参数，默认为0.98
        variance_threshold = float(data.get('variance_threshold', 0.98))
        
        # PCA特征提取 - 使用前端传入的方差阈值
        pca = PCA(n_components=variance_threshold)
        principal_components = pca.fit_transform(X_scaled)
        
        # 获取实际保留的主成分数量
        n_components_retained = pca.n_components_
        
        # 创建结果DataFrame并保存
        pc_columns = [f'PC{i+1}' for i in range(n_components_retained)]
        result_df = pd.DataFrame(data=principal_components, columns=pc_columns)
        result_path = os.path.join(current_app.root_path,'uploads',f'feature_extraction_result.csv')
        result_df.to_csv(result_path, index=False)
        
        return jsonify({
            'status': 'success',
            'message': f'PCA提取完成，保留{n_components_retained}个主成分({variance_threshold*100}%方差)',
            'n_components_retained': int(n_components_retained),  # 转换为Python int
            'explained_variance_ratio': [float(x) for x in pca.explained_variance_ratio_]  # 转换为Python float列表
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'特征提取过程中出错: {str(e)}'
        }), 500