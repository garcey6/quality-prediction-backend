from flask import Blueprint, jsonify, request
import pandas as pd
import os

variables_bp = Blueprint('variables', __name__)

@variables_bp.route('/api/variables/list', methods=['GET'])
def get_variables():
    try:
        file_path = 'uploads/latest_upload.csv'
        
        if not os.path.exists(file_path):
            return jsonify({'error': '请先上传数据文件'}), 400
        
        df = pd.read_csv(file_path)
        variables = []
        
        for idx, col in enumerate(df.columns):
            col_type = str(df[col].dtype)
            variables.append({
                'id': idx + 1,
                'name': col,
                'type': col_type,
                'status': 1
            })
        
        return jsonify(variables)
        
    except Exception as e:
        return jsonify({
            'error': f'获取变量失败: {str(e)}'
        }), 500

@variables_bp.route('/api/variables/select', methods=['POST'])
def select_variables():
    try:
        if not request.is_json:
            return jsonify({'error': '请求必须是JSON格式'}), 400
            
        selected_cols = request.json.get('selected_columns', [])
        if not selected_cols:
            return jsonify({'error': '未选择任何变量'}), 400
            
        file_path = 'uploads/latest_upload.csv'
        if not os.path.exists(file_path):
            return jsonify({'error': '请先上传数据文件'}), 400
            
        try:
            df = pd.read_csv(file_path)
            
            missing_cols = [col for col in selected_cols if col not in df.columns]
            if missing_cols:
                return jsonify({
                    'error': f'以下列不存在: {", ".join(missing_cols)}',
                    'available_columns': list(df.columns)
                }), 400
                
            df_selected = df[selected_cols]
            df_selected.to_csv(file_path, index=False)
            
            return jsonify({
                'message': '变量选择成功',
                'selected_columns': selected_cols
            })
            
        except pd.errors.EmptyDataError:
            return jsonify({'error': '数据文件为空'}), 400
        except pd.errors.ParserError:
            return jsonify({'error': '数据文件格式错误'}), 400
            
    except Exception as e:
        return jsonify({
            'error': f'变量选择失败: {str(e)}'
        }), 500