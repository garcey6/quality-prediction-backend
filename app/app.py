from flask import Flask
from flask_cors import CORS
import os

def create_app():
    app = Flask(__name__)
    CORS(app)
    
    # 配置
    app.config['UPLOAD_FOLDER'] = 'uploads'
    app.config['MAX_CONTENT_LENGTH'] = 1000 * 1024 * 1024  # 16MB限制
    
    # 动态注册所有蓝图
    from api.data import data_bp
    from api.variables import variables_bp
    from api.exceptionHandling import exception_handling_bp
    from api.standardization import standardize_bp
    from api.visualization import visualization_bp
    from api.FeatureSelection import feature_selection_bp
    from api.FeatureExtraction import feature_extraction_bp
    from api.pls import pls_bp  
    
    blueprints = [
        data_bp,
        variables_bp,
        exception_handling_bp,
        standardize_bp,
        visualization_bp,
        feature_selection_bp,
        feature_extraction_bp,
        pls_bp  
    ]
    
    for bp in blueprints:
        app.register_blueprint(bp)
    
    return app

app = create_app()

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)