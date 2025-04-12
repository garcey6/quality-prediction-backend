from flask import Blueprint, jsonify
import pandas as pd
import os

variables_bp = Blueprint('variables', __name__)

@variables_bp.route('/api/variables', methods=['GET'])
def get_variables():
    try:
        # 从上传的文件中读取变量信息
        file_path = 'uploads/latest_upload.csv'
        
        if not os.path.exists(file_path):
            return jsonify({'error': '请先上传数据文件'}), 400
        
        # 读取CSV文件获取列名和类型
        df = pd.read_csv(file_path)
        variables = []
        
        for idx, col in enumerate(df.columns):
            col_type = str(df[col].dtype)
            variables.append({
                'id': idx + 1,
                'name': col,
                'type': col_type,
                'status': 1  # 默认所有变量可用
            })
        
        return jsonify(variables)
        
    except Exception as e:
        return jsonify({
            'error': f'获取变量失败: {str(e)}'
        }), 500