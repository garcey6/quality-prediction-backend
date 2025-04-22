from flask import Blueprint, request, jsonify,current_app
import pandas as pd
import os
from werkzeug.utils import secure_filename

data_bp = Blueprint('data', __name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@data_bp.route('/api/data/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': '未选择文件'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '未选择文件'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        if not os.path.exists(UPLOAD_FOLDER):
            os.makedirs(UPLOAD_FOLDER)
        
        # 保存原始文件
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        # 创建latest_upload.csv副本
        latest_path = os.path.join(UPLOAD_FOLDER, 'latest_upload.csv')
        df = pd.read_csv(filepath)
        df.to_csv(latest_path, index=False)  # 使用pandas保存确保格式一致
        
        try:
            # 读取原始数据（保留原始列名）
            df = pd.read_csv(filepath, header=0)  # 确保使用第一行作为列名
            original_data = df.to_dict(orient='records')
            
            # 创建处理后的数据副本（保持列名不变）
            processed_data = df.copy().to_dict(orient='records')
            
            return jsonify({
                'original': original_data,
                'processed': processed_data,
                'columns': list(df.columns)  # 返回列名列表
            }), 200
            
        except Exception as e:
            return jsonify({
                'error': f'文件处理失败: {str(e)}'
            }), 500
    
    return jsonify({'error': '不允许的文件类型'}), 400