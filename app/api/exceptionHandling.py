from flask import Blueprint, jsonify, current_app
import pandas as pd
import numpy as np
import os

exception_handling_bp = Blueprint('exception_handling', __name__)

@exception_handling_bp.route('/api/exception/handle', methods=['POST'])
def handle_exceptions():
    try:
        # 获取文件路径
        file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], 'latest_upload.csv')
        
        if not os.path.exists(file_path):
            return jsonify({'error': '请先上传数据文件'}), 400

        # 读取数据
        df = pd.read_csv(file_path)
        original_rows = len(df)  # 先保存原始行数
        
        # 1. 缺失值处理 - 填充均值
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        
        # 2. 异常值处理 - 改进的Z-score方法
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        valid_cols = []
        z_scores = pd.DataFrame(index=df.index)
        
        for col in numeric_cols:
            if df[col].std() > 0:  # 只处理标准差不为零的列
                valid_cols.append(col)
                z_scores[col] = np.abs((df[col] - df[col].mean()) / df[col].std())
        
        if not valid_cols:
            return jsonify({'error': '没有有效的数值列可用于异常检测'}), 400
            
        mask = (z_scores[valid_cols] < 3).all(axis=1)
        removed_rows = len(df) - sum(mask)
        df = df[mask]
        
        # 保存处理后的数据
        df.to_csv(file_path, index=False)
        
        return jsonify({
            'status': 'success',
            'message': '异常处理完成',
            'data': {
                'original_rows': original_rows,
                'processed_rows': len(df),
                'removed_rows': removed_rows
            }
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'异常处理失败: {str(e)}'
        }), 500