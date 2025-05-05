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
        file_path = os.path.join(current_app.root_path,'uploads', f'latest_upload.csv')
        
        if not os.path.exists(file_path):
            return jsonify({'status': 'error', 'message': '请先上传数据文件'}), 400

        # 读取数据
        df = pd.read_csv(file_path)
        original_rows = len(df)
        
        # 读取目标变量
        target_path = os.path.join(current_app.root_path,'uploads',f'target_variable.txt')
        target_col = None
        if os.path.exists(target_path):
            with open(target_path, 'r') as f:
                target_col = f.read().strip()
        
        # 处理数值列，排除目标变量列（改进比较方式）
        numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns 
                       if str(col).strip().lower() != str(target_col).strip().lower()] if target_col else df.select_dtypes(include=[np.number]).columns
        
        if numeric_cols:
            scaler = MinMaxScaler()
            df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        
        # 保存处理后的数据
        df.to_csv(file_path, index=False)
        
        return jsonify({
            'status': 'success',
            'message': '数据标准化完成',
            'data': {
                'original_rows': original_rows,
                'processed_rows': len(df),
                'excluded_target': target_col if target_col else '无'
            }
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'标准化失败: {str(e)}'
        }), 500