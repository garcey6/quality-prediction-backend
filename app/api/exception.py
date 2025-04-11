from flask import Blueprint, request, jsonify
import numpy as np
import pandas as pd
import os
from ...app import app  # 导入app以获取上传目录配置

exception_bp = Blueprint('exception', __name__)

@exception_bp.route('/api/exception/process', methods=['POST'])
def process_data():
    req_data = request.json
    algorithm = req_data.get('algorithm')
    selected_columns = req_data.get('columns', [])  # 获取前端选择的列
    
    # 默认参数设置
    default_params = {
        'zscore': {'mean': 0, 'std': 1},
        'minmax': {'min': 0, 'max': 1}
    }
    
    try:
        # 读取最新上传的CSV文件
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'latest_upload.csv')
        if not os.path.exists(file_path):
            return jsonify({'error': '请先上传数据文件'}), 400
            
        df = pd.read_csv(file_path)
        
        # 如果指定了列，则只处理这些列
        if selected_columns:
            df = df[selected_columns]
            
        # 转换为numpy数组处理
        data = df.values
        
        if algorithm == 'zscore':
            # 使用默认参数的z-score处理
            processed = (data - default_params['zscore']['mean']) / default_params['zscore']['std']
        elif algorithm == 'minmax':
            # 使用默认参数的min-max处理
            processed = (data - default_params['minmax']['min']) / (default_params['minmax']['max'] - default_params['minmax']['min'])
        else:
            return jsonify({'error': '不支持的算法类型'}), 400
            
        # 将处理后的数据转换回DataFrame
        processed_df = pd.DataFrame(processed, columns=df.columns)
        
        return jsonify({
            'success': True,
            'algorithm': algorithm,
            'processed_data': processed_df.to_dict(orient='records'),  # 返回字典格式数据
            'columns': list(processed_df.columns)  # 返回列名
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500