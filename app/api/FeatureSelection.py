from flask import Blueprint, request, jsonify, current_app
import pandas as pd
import os
from sklearn.feature_selection import mutual_info_regression
from scipy.stats import pearsonr

feature_selection_bp = Blueprint('feature_selection', __name__)

# 新增获取变量列表接口
@feature_selection_bp.route('/api/feature-selection/variables', methods=['GET'])
def get_feature_variables():
    try:
        file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], 'latest_upload.csv')
        if not os.path.exists(file_path):
            return jsonify({'status': 'error', 'message': '请先上传数据文件'}), 400
            
        df = pd.read_csv(file_path)
        
        # 修改返回数据结构，直接返回数组
        variables = [
            {
                'id': idx + 1,
                'name': col,
                'type': str(df[col].dtype),
                'status': 1  # 默认状态
            } 
            for idx, col in enumerate(df.columns)
        ]
        
        return jsonify(variables)  # 直接返回数组
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'获取变量列表失败: {str(e)}'
        }), 500

# 修改特征选择接口
@feature_selection_bp.route('/api/feature-selection/select', methods=['POST'])
def feature_selection():
    try:
        if not request.is_json:
            return jsonify({'error': '请求必须是JSON格式'}), 400
            
        data = request.json
        target_var = data.get('target_variable')  # 修改参数名
        threshold = data.get('threshold')
        
        if not target_var:
            return jsonify({'error': '缺少target_variable参数'}), 400
        if threshold is None:
            return jsonify({'error': '缺少threshold参数'}), 400
            
        try:
            threshold = float(threshold)
        except ValueError:
            return jsonify({'error': 'threshold必须是数值类型'}), 400
            
        method = data.get('method', 'pearson')
        
        # 从上传的文件获取数据
        filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], 'latest_upload.csv')
        if not os.path.exists(filepath):
            return jsonify({
                'status': 'error',
                'code': 400,
                'message': '请先上传数据文件',
                'data': None
            }), 400
            
        try:
            df = pd.read_csv(filepath, header=0, keep_default_na=False)
        except Exception as e:
            return jsonify({
                'status': 'error', 
                'code': 500,
                'message': f'读取数据文件失败: {str(e)}',
                'data': None
            }), 500

        # 检查目标变量是否存在
        if target_var not in df.columns:
            return jsonify({
                'status': 'error',
                'code': 400,
                'message': f'目标变量"{target_var}"不存在',
                'data': {
                    'available_columns': list(df.columns)
                }
            }), 400
            
        # 使用原始列名
        actual_target_var = target_var
        
        # 分离特征和目标变量
        X = df.drop(columns=[actual_target_var])
        y = df[actual_target_var]
        
        # 移除非数值列（如batch_id, date等）
        numeric_cols = X.select_dtypes(include=['number']).columns
        X = X[numeric_cols]
        
        # 检查并处理缺失值
        if X.isnull().any().any() or y.isnull().any():
            X = X.fillna(X.mean())
            y = y.fillna(y.mean())
            
        # 过滤掉常量特征
        non_constant_cols = [col for col in X.columns if X[col].nunique() > 1]
        X = X[non_constant_cols]
        
        # 检查数据有效性
        if len(X.columns) == 0:
            return jsonify({
                'status': 'error',
                'code': 400,
                'message': '没有可用的数值型特征列',
                'data': {
                    'available_numeric_columns': numeric_cols.tolist()
                }
            }), 400

        # 根据选择的方法进行特征选择
        if method == 'pearson':
            selected_features = []
            feature_scores = []
            for col in X.columns:
                if X[col].nunique() == 1:
                    continue
                corr, _ = pearsonr(X[col], y)
                if abs(corr) >= threshold:
                    selected_features.append(col)
                    feature_scores.append(abs(corr))
                    
        elif method == 'mutual_info':
            mi_scores = mutual_info_regression(X, y)
            selected_features = [col for col, score in zip(X.columns, mi_scores) 
                               if score >= threshold]
            feature_scores = [score for score in mi_scores 
                            if score >= threshold]
        
        # 保存选择结果到文件
        result_path = os.path.join(current_app.config['UPLOAD_FOLDER'], 'feature_selection_result.csv')
        result_df = pd.DataFrame({
            'feature': selected_features,
            'score': feature_scores
        })
        result_df.to_csv(result_path, index=False)

        return jsonify({
            'status': 'success',
            'message': f'使用{method}方法选择了{len(selected_features)}个特征'
        })
        
    except ValueError as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'特征选择过程中出错: {str(e)}'
        }), 500