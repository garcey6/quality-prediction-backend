from flask import Blueprint, jsonify, request, current_app
import pandas as pd
import os

variables_bp = Blueprint('variables', __name__)

@variables_bp.route('/api/variables/list', methods=['GET'])
def get_variables():
    try:
        file_path = os.path.join(current_app.root_path,'uploads',f'latest_upload.csv')
        
        if not os.path.exists(file_path):
            return jsonify({'error': '请先上传数据文件'}), 400
        
        df = pd.read_csv(file_path)
        variables = []
        
        # 检查是否已有目标变量选择
        target_path = os.path.join(current_app.root_path,'uploads',f'target_variable.txt')
        current_target = None
        if os.path.exists(target_path):
            with open(target_path, 'r') as f:
                current_target = f.read().strip()
        
        for idx, col in enumerate(df.columns):
            col_type = str(df[col].dtype)
            variables.append({
                'id': idx + 1,
                'name': col,
                'type': col_type,
                'status': 1,
                'is_target': col == current_target  # 标记当前目标变量
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
            
        data = request.json
        selected_cols = data.get('selected_columns', [])
        target_col = data.get('target_variable')
        
        if not selected_cols:
            return jsonify({'error': '未选择任何变量'}), 400
            
        file_path = os.path.join(current_app.root_path,'uploads',f'latest_upload.csv')
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
                
            # 直接保存目标变量，无需检查是否存在
            if target_col and target_col in selected_cols:
                target_path = os.path.join(current_app.root_path,'uploads',f'target_variable.txt')
                with open(target_path, 'w') as f:
                    f.write(target_col)
                
                # 将目标变量移到最后一列
                selected_cols.remove(target_col)
                selected_cols.append(target_col)
            
            df_selected = df[selected_cols]
            df_selected.to_csv(file_path, index=False)
            
            return jsonify({
                'message': '变量选择成功',
                'selected_columns': selected_cols,
                'target_variable': target_col
            })
            
        except pd.errors.EmptyDataError:
            return jsonify({'error': '数据文件为空'}), 400
        except pd.errors.ParserError:
            return jsonify({'error': '数据文件格式错误'}), 400
            
    except Exception as e:
        return jsonify({
            'error': f'变量选择失败: {str(e)}'
        }), 500