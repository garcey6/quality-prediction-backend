from flask import Blueprint, jsonify, current_app
import pandas as pd
import numpy as np
import os

exception_handling_bp = Blueprint('exception_handling', __name__)

@exception_handling_bp.route('/api/exception/handle', methods=['POST'])
def handle_exceptions():
    try:
        # 获取文件路径
        file_path = os.path.join(current_app.root_path,'uploads',f'latest_upload.csv')
        
        if not os.path.exists(file_path):
            return jsonify({'error': '请先上传数据文件'}), 400

        # 读取数据
        df = pd.read_csv(file_path)
        original_rows = len(df)
        
        # 缺失值处理 - 填充均值
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        
        # 保存处理后的数据
        df.to_csv(file_path, index=False)
        
        return jsonify({
            'status': 'success',
            'message': '缺失值处理完成',
            'data': {
                'original_rows': original_rows,
                'processed_rows': len(df)
            }
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'缺失值处理失败: {str(e)}'
        }), 500