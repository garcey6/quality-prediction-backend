from flask import Blueprint, jsonify, current_app
import pandas as pd
import os

variables_bp = Blueprint('variables', __name__)

@variables_bp.route('/api/variables', methods=['GET'])
def get_variables():
    # 从上传的文件中读取变量信息
    upload_folder = current_app.config['UPLOAD_FOLDER']
    file_path = os.path.join(upload_folder, 'latest_upload.csv')
    
    if not os.path.exists(file_path):
        return jsonify({'error': '文件不存在，请先上传数据'}), 404
    
    try:
        # 读取CSV文件获取列名
        df = pd.read_csv(file_path)
        columns = df.columns.tolist()
        
        # 构建变量数据
        variables = []
        for idx, col in enumerate(columns):
            # 这里可以根据实际需求添加更多变量属性
            col_type = str(df[col].dtype)
            variables.append({
                'id': idx + 1,
                'name': col,
                'type': col_type,
                'status': 1  # 1表示可用，0表示禁用
            })
        
        return jsonify(variables)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500