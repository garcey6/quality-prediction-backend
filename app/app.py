from flask import Flask, request, jsonify
from flask_cors import CORS
from api.variables import variables_bp
from api.exception import exception_bp 
import os

app = Flask(__name__)
CORS(app)

# 文件上传配置
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# 注册蓝图
app.register_blueprint(variables_bp)
app.register_blueprint(exception_bp)
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload/', methods=['POST'])
def upload_file():
    # 检查请求中是否有文件
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    # 检查是否选择了文件
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # 检查文件类型
    if file and allowed_file(file.filename):
        # 确保上传目录存在
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        
        # 保存文件
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        
        # 创建latest_upload.csv链接
        latest_path = os.path.join(app.config['UPLOAD_FOLDER'], 'latest_upload.csv')
        if os.path.exists(latest_path):
            os.remove(latest_path)
        os.link(filepath, latest_path)  # 使用硬链接节省空间
        
        return jsonify({
            'message': 'File uploaded successfully',
            'filename': file.filename,
            'path': filepath
        }), 200
    else:
        return jsonify({'error': 'Allowed file type is csv'}), 400

if __name__ == '__main__':
    app.run(debug=True)