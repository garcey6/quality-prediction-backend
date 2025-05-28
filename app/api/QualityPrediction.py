from flask import Blueprint, request, jsonify, current_app
import pandas as pd
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO
from datetime import datetime
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error  # 添加这行导入

# 在文件开头添加字体设置
import matplotlib.font_manager as fm

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows系统常用中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

quality_bp = Blueprint('quality_prediction', __name__)

@quality_bp.route('/api/quality/visualize', methods=['POST'])
def visualize_prediction():
    try:
        # 检查请求数据
        if not request.is_json:
            return jsonify({'status': 'error', 'message': '请求必须是JSON格式'}), 400
            
        data = request.json
        model_type = data.get('model_type', 'lstm')  # 支持多种模型
        
        # 根据模型类型获取预测结果文件
        pred_path = os.path.join(
            current_app.root_path,'uploads',
            f'{model_type}_predictions.csv'
        )
        
        if not os.path.exists(pred_path):
            return jsonify({
                'status': 'error',
                'message': f'请先进行{model_type.upper()}预测'
            }), 400
            
        # 读取预测结果
        df = pd.read_csv(pred_path)
        
        # 检查数据格式
        if '实际值' not in df.columns or '预测值' not in df.columns:
            return jsonify({
                'status': 'error',
                'message': '预测结果文件格式不正确'
            }), 400
            
        # 计算评估指标
        metrics = {
            'mse': mean_squared_error(df['实际值'], df['预测值']),
            'rmse': np.sqrt(mean_squared_error(df['实际值'], df['预测值'])),
            'r2': r2_score(df['实际值'], df['预测值']),
            'mae': mean_absolute_error(df['实际值'], df['预测值']),
            '样本数': len(df)
        }

        # 创建预测对比图
        plt.figure(figsize=(10, 6))
        sorted_indices = np.argsort(df['实际值'].values)
        plt.plot(np.sort(df['实际值'].values), color='blue', label='实际值', linewidth=2)
        plt.plot(df['预测值'].values[sorted_indices], color='red', label='预测值', linewidth=2)
        plt.title(f'{model_type.upper()}预测结果对比(按实际值排序)')
        
        plt.xlabel('样本(按实际值排序)')
        plt.ylabel('浓度值')
        plt.legend()
        plt.tight_layout()

        # 保存预测对比图
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        plt.close()
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')

        # 创建残差图
        plt.figure(figsize=(10, 6))
        residuals = df['实际值'] - df['预测值']
        
        # 计算纵坐标范围，使用实际值的范围乘以一个系数
        y_range = df['实际值'].max() - df['实际值'].min()
        y_padding = y_range * 0.2  # 添加20%的padding
        
        # 随机采样50%的数据点
        sample_size = int(len(df) * 0.5)
        indices = np.random.choice(len(df), size=sample_size, replace=False)
        sampled_predictions = df['预测值'].values[indices]
        sampled_residuals = residuals.values[indices]
        
        # 绘制采样后的散点图
        plt.scatter(sampled_predictions, sampled_residuals, color='green', alpha=0.2)
        plt.axhline(y=0, color='red', linestyle='--')
        plt.title(f'{model_type.upper()}预测残差图(50%采样)')
        plt.xlabel('预测值')
        plt.ylabel('残差(实际值-预测值)')
        
        # 设置纵坐标范围
        plt.ylim(-y_padding, y_padding)
        
        plt.tight_layout()

        # 保存残差图
        residual_buf = BytesIO()
        plt.savefig(residual_buf, format='png', dpi=150, bbox_inches='tight')
        plt.close()
        residual_buf.seek(0)
        residual_image_base64 = base64.b64encode(residual_buf.read()).decode('utf-8')

        # 保存图片文件到static文件夹
        img_path = os.path.join(
            current_app.root_path,
            'static',
            f'quality_prediction_{model_type}.png'
            
        )
        residual_img_path = os.path.join(
            current_app.root_path,
            'static',
            f'residual_plot_{model_type}.png'
        )

        # 确保static目录存在
        os.makedirs(os.path.dirname(img_path), exist_ok=True)
        with open(img_path, 'wb') as f:
            f.write(base64.b64decode(image_base64))
        with open(residual_img_path, 'wb') as f:
            f.write(base64.b64decode(residual_image_base64))

        return jsonify({
            'status': 'success',
            'message': f'{model_type.upper()}预测可视化完成',
            'data': {
                'metrics': metrics,
                'image': image_base64,  # 预测对比图
                'residual_image': residual_image_base64,  # 残差图
                'image_path': img_path,
                'residual_image_path': residual_img_path
            }
        })

    except Exception as e:
        current_app.logger.error(f'质量预测可视化错误: {str(e)}', exc_info=True)
        return jsonify({
            'status': 'error',
            'message': f'质量预测可视化失败: {str(e)}'
        }), 500


@quality_bp.route('/api/quality/export', methods=['POST'])
def export_prediction():
    try:
        data = request.json
        model_type = data.get('model_type', 'pls')
        export_format = data.get('format', 'png')  # 支持png/pdf/svg
        
        # 获取最新生成的图片
        img_files = [f for f in os.listdir(current_app.root_path,'static',) 
                    if f.startswith(f'quality_prediction_{model_type}')]
        
        if not img_files:
            return jsonify({
                'status': 'error',
                'message': '没有找到可导出的预测结果'
            }), 404
            
        latest_img = max(img_files, key=lambda x: os.path.getmtime(
            os.path.join(current_app.root_path,'uploads', x)
        ))
        
        img_path = os.path.join(current_app.root_path,'uploads', latest_img)
        
        # 这里可以添加格式转换逻辑
        # 实际项目中可以根据export_format参数转换图片格式
        
        return jsonify({
            'status': 'success',
            'message': '导出成功',
            'data': {
                'file_path': img_path,
                'file_name': latest_img
            }
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'导出失败: {str(e)}'
        }), 500