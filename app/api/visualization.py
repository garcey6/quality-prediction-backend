from flask import Blueprint, request, jsonify, current_app
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import base64
from io import BytesIO
import seaborn as sns

visualization_bp = Blueprint('visualization', __name__, url_prefix='/api/visualization')

@visualization_bp.route('/multivariate', methods=['POST'])
def multivariate_visualization():
    try:
        # 获取文件路径
        file_path = os.path.join(current_app.root_path,'uploads',f'latest_upload.csv')
        
        if not os.path.exists(file_path):
            return jsonify({'error': '请先上传数据文件'}), 400

        # 生成相关性热力图
        image_base64 = generate_correlation_heatmap(file_path)
        
        return jsonify({
            'image': image_base64,
            'message': '相关性热力图生成成功'
        })
        
    except Exception as e:
        return jsonify({'error': f'生成热力图时出错: {str(e)}'}), 500

@visualization_bp.route('/generate', methods=['GET'])  # 修改为GET方法以匹配前端
def generate_visualization():
    try:
        # 获取文件路径
        file_path = os.path.join(current_app.root_path,'uploads',f'latest_upload.csv')
        
        if not os.path.exists(file_path):
            return jsonify({'error': '请先上传数据文件'}), 400

        # 生成相关性热力图
        image_base64 = generate_correlation_heatmap(file_path)
        
        return jsonify({
            'status': 'success',
            'image': image_base64,
            'message': '相关性热力图生成成功'
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'生成热力图时出错: {str(e)}'
        }), 500

def generate_correlation_heatmap(file_path):
    """从文件生成参数相关性热力图"""
    # 读取CSV文件
    df = pd.read_csv(file_path)
    
    # 移除非数值列
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) < 2:
        raise ValueError("至少需要2个数值列才能计算相关性")
    df = df[numeric_cols]
    
    # 检查并处理缺失值
    if df.isnull().values.any():
        df = df.fillna(df.mean())  # 用均值填充缺失值
    
    # 计算相关系数矩阵
    corr_matrix = df.corr()
    
    # 创建热力图
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, 
                annot=True, 
                cmap='coolwarm', 
                fmt=".2f",
                linewidths=0.5,
                vmin=-1, 
                vmax=1)
    
    plt.title('参数相关性热力图', fontsize=14)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    
    # 确保static目录存在
    static_dir = os.path.join(current_app.root_path, 'static')
    os.makedirs(static_dir, exist_ok=True)
    
    # 保存到static目录
    static_path = os.path.join(static_dir, 'correlation_heatmap.png')
    plt.savefig(static_path, format='png', dpi=150, bbox_inches='tight')
    
    # 转换为base64
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')