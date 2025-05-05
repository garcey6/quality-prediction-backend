from flask import Blueprint, request, jsonify, current_app
import pandas as pd
import os
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split, cross_val_score  # 添加cross_val_score导入
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict


pls_bp = Blueprint('pls', __name__)

@pls_bp.route('/api/pls/predict', methods=['POST'])
def pls_prediction():
    try:
        # 检查请求数据
        if not request.is_json:
            return jsonify({'status': 'error', 'message': '请求必须是JSON格式'}), 400
            
        data = request.json
        
        # 读取原始数据文件
        data_path = os.path.join(current_app.root_path,'uploads',f'latest_upload.csv')
        if not os.path.exists(data_path):
            return jsonify({
                'status': 'error',
                'message': '请先上传数据文件'
            }), 400
            
        # 加载数据
        df = pd.read_csv(data_path)
        
        # 检查数据列
        selection_path = os.path.join(current_app.root_path,'uploads',f'feature_selection_result.csv')
        if not os.path.exists(selection_path):
            return jsonify({
                'status': 'error',
                'message': '请先进行特征选择'
            }), 400
            
        # 读取特征选择结果
        selected_df = pd.read_csv(selection_path)
        target_column = selected_df['target'].iloc[0]
        selected_features = selected_df['feature'].tolist()
        
        # 检查目标变量列是否存在（不区分大小写和空格）
        target_found = any(col.strip().lower() == target_column.strip().lower() for col in df.columns)
        if not target_found:
            return jsonify({
                'status': 'error',
                'message': f'数据文件中缺少目标变量列: {target_column}',
                'available_columns': list(df.columns)
            }), 400
            
        # 获取实际的目标列名（保持原始大小写，去除前后空格）
        actual_target_col = next(col for col in df.columns 
                               if col.strip().lower() == target_column.strip().lower())
        
        # 分离特征和标签 - 只使用选定的特征
        X = df[selected_features]
        y = df[actual_target_col]  # 使用处理后的列名
        
        # 取1/10的样本进行测试
        sample_fraction = 0.1
        sample_size = int(len(X) * sample_fraction)
        X = X.sample(n=sample_size, random_state=42)
        y = y.loc[X.index]
        
        # 检查特征选择结果中的特征是否存在于数据中
        missing_features = [f for f in selected_features if f not in df.columns]
        if missing_features:
            return jsonify({
                'status': 'error',
                'message': f'数据文件中缺少以下特征列: {", ".join(missing_features)}'
            }), 400
        # 数据标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 扩展主成分数选择范围
        max_components = min(20, len(selected_features))  # 从10增加到20
        
        # 使用更严格的交叉验证
        cv_folds = 10  # 从5折增加到10折
        
        # 自动选择最佳组件数 - 添加标准差判断
        best_n = 1
        best_score = -np.inf
        score_std_threshold = 0.05  # 分数标准差阈值
        
        for n in range(1, max_components+1):
            pls = PLSRegression(n_components=n, scale=True)
            scores = cross_val_score(pls, X_scaled, y, cv=cv_folds, scoring='r2')
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            
            # 选择分数高且稳定的主成分数
            if mean_score > best_score and std_score < score_std_threshold:
                best_score = mean_score
                best_n = n
        
        # 训练最终模型 - 添加更多正则化参数
        pls = PLSRegression(
            n_components=best_n,
            scale=True,
            max_iter=5000,  # 增加最大迭代次数
            tol=1e-09,     # 更严格的收敛条件
            copy=True
        )
        pls.fit(X_scaled, y)
        
        # 预测
        y_pred = pls.predict(X_scaled)
        
        # 将小于0的预测值设为0
        y_pred[y_pred < 0] = 0
        
        # 评估
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)

        # 保存预测结果到CSV文件
        result_df = pd.DataFrame({
            '实际值': y.values.flatten(),
            '预测值': y_pred.flatten()
        })
        result_path = os.path.join(current_app.root_path,'uploads',f'pls_predictions.csv')
        result_df.to_csv(result_path, index=False)

        # 打印调试信息
        print("返回前端的数据结构:")
        print({
            'status': 'success',
            'data': {
                'mse': float(mse),
                'r2_score': float(r2),
                'n_components': int(best_n),
                'predictions': [float(pred) for pred in y_pred.flatten().tolist()][:5]  # 只打印前5个预测值
            }
        })
        
        return jsonify({
            'status': 'success',
            'data': {
                'mse': float(mse),
                'r2_score': float(r2),
                'n_components': int(best_n),
                'predictions': [float(pred) for pred in y_pred.flatten().tolist()]
            }
        })
        
    except Exception as e:
        current_app.logger.error(f'PLS预测错误: {str(e)}', exc_info=True)
        return jsonify({
            'status': 'error',
            'message': f'PLS预测失败: {str(e)}'
        }), 500