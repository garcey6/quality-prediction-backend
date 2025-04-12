from flask import Blueprint, request, jsonify
import matplotlib
matplotlib.use('Agg')  # 添加这行，使用非交互式后端
import matplotlib.pyplot as plt
import numpy as np
import base64
from io import BytesIO

visualization_bp = Blueprint('visualization', __name__, url_prefix='/api/visualization')

@visualization_bp.route('/multivariate', methods=['POST'])
def multivariate_visualization():
    try:
        data = request.json.get('data')
        if not data:
            return jsonify({'error': '无数据提供'}), 400
        
        # 生成多变量时间序列矩阵图
        image_base64 = generate_multivariate_plot(data)
        
        return jsonify({
            'image': image_base64,
            'message': '可视化生成成功'
        })
        
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': f'生成可视化时出错: {str(e)}'}), 500

def generate_multivariate_plot(data):
    """生成多变量时间序列矩阵图"""
    if not data or len(data) == 0:
        raise ValueError("无有效数据可供可视化")
    
    variables = list(data[0].keys())
    n = len(variables)
    
    # 创建图形时添加plt.figure()确保线程安全
    plt.figure()  # 添加这行
    fig, axes = plt.subplots(n, n, figsize=(15, 15))
    
    # 绘制每个变量组合
    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            if i == j:
                # 对角线显示变量名和直方图
                ax.hist([d[variables[i]] for d in data], bins=20, alpha=0.7)
                ax.set_title(variables[i], fontsize=8)
            else:
                # 绘制时间序列关系图
                x = [d[variables[j]] for d in data]
                y = [d[variables[i]] for d in data]
                ax.plot(x, y, 'b-', alpha=0.5, linewidth=0.5)
                ax.set_xlabel(variables[j], fontsize=6)
                ax.set_ylabel(variables[i], fontsize=6)
    
    plt.tight_layout()
    
    # 转换为base64
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')