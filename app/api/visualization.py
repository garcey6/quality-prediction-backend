from flask import Blueprint, request, jsonify
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import base64
from io import BytesIO
import seaborn as sns

visualization_bp = Blueprint('visualization', __name__, url_prefix='/api/visualization')

@visualization_bp.route('/multivariate', methods=['POST'])
def multivariate_visualization():
    try:
        # 检查请求数据
        if not request.is_json:
            return jsonify({'error': '请求必须是JSON格式'}), 400
            
        request_data = request.json
        if 'data' not in request_data:
            return jsonify({'error': '请求中缺少data字段'}), 400
            
        data = request_data['data']
        if not isinstance(data, list):
            return jsonify({'error': 'data必须是列表格式'}), 400
            
        if len(data) == 0:
            return jsonify({'error': '数据不能为空列表'}), 400
            
        # 验证数据项格式
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                return jsonify({'error': f'第{i+1}条数据必须是字典格式'}), 400
            if len(item) == 0:
                return jsonify({'error': f'第{i+1}条数据不能为空字典'}), 400

        # 记录接收到的数据信息
        print(f"接收到数据，记录数: {len(data)}")
        print(f"包含变量: {list(data[0].keys())}")
        
        # 生成相关性热力图
        image_base64 = generate_correlation_heatmap(data)
        
        return jsonify({
            'image': image_base64,
            'message': '相关性热力图生成成功'
        })
        
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': f'生成热力图时出错: {str(e)}'}), 500

def generate_correlation_heatmap(data):
    """生成参数相关性热力图"""
    # 转换为DataFrame并确保所有值为数值类型
    df = pd.DataFrame(data)
    
    # 移除非数值列（如Timestamp、batch_id等）
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
    
    # 转换为base64
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')