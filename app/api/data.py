from flask import Blueprint, request, jsonify,current_app
import pandas as pd
import os
from werkzeug.utils import secure_filename

data_bp = Blueprint('data', __name__)

DATA_FOLDER = 'data'  # 修改为data文件夹
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
        
        # 保存原始文件到data文件夹
        data_dir = os.path.join(current_app.root_path, 'data')
        os.makedirs(data_dir, exist_ok=True)
        filepath = os.path.join(data_dir, filename)
        file.save(filepath)
        # 保存最新上传的文件到uploads文件夹
        latest_path = os.path.join(current_app.root_path, 'uploads',f'latest_upload.csv')
        df = pd.read_csv(filepath)
        df.to_csv(latest_path, index=False)
        
        try:
            # 读取原始数据
            df = pd.read_csv(filepath, header=0)
            original_data = df.to_dict(orient='records')
            
            # 创建处理后的数据副本
            processed_data = df.copy().to_dict(orient='records')
            
            return jsonify({
                'original': original_data,
                'processed': processed_data,
                'columns': list(df.columns)
            }), 200
            
        except Exception as e:
            return jsonify({
                'error': f'文件处理失败: {str(e)}'
            }), 500
    
    return jsonify({'error': '不允许的文件类型'}), 400