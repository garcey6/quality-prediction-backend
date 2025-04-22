from flask import Blueprint, request, jsonify, current_app
import pandas as pd
import numpy as np  # 添加numpy导入
import os
from sklearn.preprocessing import MinMaxScaler

standardize_bp = Blueprint('standardize', __name__)

@standardize_bp.route('/api/data/standardize', methods=['POST'])
def standardize():
    try:
        # 获取文件路径
        file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], 'latest_upload.csv')
        
        if not os.path.exists(file_path):
            return jsonify({'status': 'error', 'message': '请先上传数据文件'}), 400

        # 读取数据
        df = pd.read_csv(file_path)
        original_rows = len(df)
        
        # 处理数值列
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        scaler = MinMaxScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        
        # 保存处理后的数据
        df.to_csv(file_path, index=False)
        
        return jsonify({
            'status': 'success',
            'message': '数据标准化完成',
            'data': {
                'original_rows': original_rows,
                'processed_rows': len(df)
            }
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'标准化失败: {str(e)}'
        }), 500