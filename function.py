from flask import Flask,request,jsonify,send_file,url_for
import matplotlib.pyplot as plt
import pandas as pd
import os
from flask_cors import CORS
import csv
from io import TextIOWrapper
import json
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from io import BytesIO
import matplotlib
import seaborn as sns
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from sklearn import preprocessing

from io import StringIO
import io


matplotlib.use('agg')  # 设置Matplotlib的后端为非交互式后端

app = Flask(__name__, static_url_path='/static')
CORS(app, resources={r"/*": {"origins": "*","supports_credentials": True}})
# 全局变量用于存储CSV数据
# 全局变量用于存储 CSV 数据
csv_columns = []
csv_data = []

@app.route('/create', methods=['POST'])
def createProject():
    data = request.get_json()
    print(data)
    return "Hello:" + data['name']

@app.route('/upload', methods=['POST'])
def dataUpload():
    global csv_data, csv_columns
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file:
        filename, file_extension = os.path.splitext(file.filename)
        if file_extension == '.csv':
            # 使用 Pandas 读取 CSV 文件
            df = pd.read_csv(file)
            csv_data = df.values.tolist()
            csv_columns = df.columns.tolist()
            return jsonify({"status": "success", "message": "File uploaded successfully"})
        else:
            return 'Unsupported file extension'

@app.route('/columns', methods=['GET'])
def getColumns():
    global csv_columns
    return jsonify(csv_columns), 200

@app.route('/extract', methods=['POST'])
def extractColumns():
    global csv_data
    selected_columns = request.json.get('columns', [])
    if not selected_columns:
        return jsonify({"status": "error", "message": "No columns selected"}), 400

    # 创建新的 DataFrame，提取选定列的数据
    df = pd.DataFrame(csv_data, columns=csv_columns)
    extracted_data = df[selected_columns]

    # 保存到 CSV
    output = BytesIO()
    extracted_data.to_csv(output, index=False)
    output.seek(0)

    return send_file(output, mimetype='text/csv', as_attachment=True, download_name='extracted_data.csv')




""" 
特征提取
使用PCA算法对数据进行降维
返回降维后的数据分布图
"""
@app.route('/PCA',methods = ['GET'])
def pca():
    # 加载数据
    data = pd.read_csv(r'./data/data_.csv')

    # 假设CSV文件中每一列都是一个特征，并且没有缺失值
    # 标准化数据
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # 执行PCA
    pca = PCA(n_components=5)  # 调整n_components以查看更多的主成分
    principal_components = pca.fit_transform(data_scaled)

    # 创建一个新的DataFrame来保存降维后的数据
    columns = ['PC{}'.format(i+1) for i in range(pca.n_components_)]
    pca_df = pd.DataFrame(data=principal_components, columns=columns)

    # 将方差解释率绘制为柱状图
    plt.figure(figsize=(10, 5))
    plt.bar(columns, pca.explained_variance_ratio_, color='maroon', alpha=0.7)
    plt.ylabel('Percentage of Explained Variance')
    plt.xlabel('Principal Components')
    plt.title('PCA Explained Variance Ratio')
    # plt.savefig('pca_explained_variance_ratio.png')
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()

    return send_file(buf, mimetype='image/png')

"""  
数据可视化
参数:需要可视化的变量
返回变量可视化后的图片
"""
@app.route('/visualize/<column>',methods = ['GET'])
def visualize(column):
    print(column)
    # 加载CSV文件
    file_path = r'./data/data_50_time.csv'
    data = pd.read_csv(file_path)
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
    matplotlib.rcParams['axes.unicode_minus'] = False

    # 确保输入的列名有效
    if column  in data.columns:
        if 'time' in data.columns:
            data['time'] = pd.to_datetime(data['time'])
        else:
            print("找不到名为 'time' 的时间列，请检查列名。")
            exit()  # 退出程序
        
        # 批次和每个批次的数据点数量
        batch_count = 100
        points_per_batch = 300
        column_data = data[column]

        # 绘制折线图
        plt.figure(figsize=(12, 6))

        # 遍历每一个批次并绘制数据
        for i in range(batch_count):
            start_index = i * points_per_batch
            end_index = start_index + points_per_batch
            batch_data = column_data[start_index:end_index]
            plt.plot(range(points_per_batch), batch_data)

        plt.title(f' {column}')
        plt.xlabel('time')
        plt.ylabel('value')
        plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1), ncol=1, fontsize='small')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()  # 自动调整布局

        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()

    return send_file(buf, mimetype='image/png')

"""  
皮尔逊相关系数热力图
"""
@app.route('/pearson',methods = ['GET'])
def pearson():
    file_path = r'./data/data_50_time.csv'  
    # 读取CSV文件
    data = pd.read_csv(file_path)
    # 删除 'batch' 列，如果存在
    if 'batch' in data.columns:
        data = data.drop('batch', axis=1)
    
    data['time'] = pd.to_datetime(data['time'])  # 将 'time' 列转换为日期格式
    # 计算皮尔逊相关系数
    correlation_matrix = data.corr()
    # 设置matplotlib的显示中文和负号问题
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    # 绘制热力图
    plt.figure(figsize=(10, 8))  # 可以调整图形大小
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('皮尔逊相关系数热力图')

    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()

    return send_file(buf, mimetype='image/png')



file_path_rnn = r'./data/train_data.csv'
window = 24  # 模型输入序列长度
length_size = 1  # 预测结果的序列长度
epochs = 1  # 迭代次数
batch_size = 32
# 读取数据
data = pd.read_csv(file_path_rnn)  # 读取目标文件

data = data.iloc[:, 2:]      #第一列为时间，去除时间列
data_target = data.iloc[:, -1:]     #目标数据
data_dim = len(data.iloc[1, :])
scaler = preprocessing.MinMaxScaler()
if data_dim == 1:
    data_inverse = scaler.fit_transform(np.array(data).reshape(-1, 1))  # 将目标数据变成二维数组，并进行归一化
else:
    data_inverse = scaler.fit_transform(np.array(data))
data = data_inverse
data_length = len(data)
train_set = 0.99
data_train = data[:int(train_set*data_length), :]  # 读取目标数据，第一列记为0：1，后面以此类推, 训练集和验证集，如果是多维输入的话最后一列为目标列数据
data_test = data[int(train_set*data_length):, :]  # 这里把训练集和测试集分开了，也可以换成两个csv文件
#指定任意批次
batch = 80
DATA_TEST = data[300*batch:300*(batch+1), :]
n_feature = data_dim # 输入特征个数
n_hidden = 10
def data_loader(window, length_size, batch_size, data):
    # 构建lstm输入
    seq_len = window  # 模型每次输入序列输入序列长度
    sequence_length = seq_len + length_size  # 序列长度，也就是输入序列的长度+预测序列的长度
    result = []  # 空列表
    for index in range(len(data) - sequence_length):  # 循环次数为数据集的总长度
        result.append(data[index: index + sequence_length])  # 第i行到i+sequence_length
    result = np.array(result)  # 得到样本，样本形式为sequence_length*特征
    x_train = result[:, :-length_size]  # 训练集特征数据
    y_train = result[:, -length_size:, -1]  # 训练集目标数据
    X_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], data_dim))  # 重塑数据形状，保证数据顺利输入模型
    y_train = np.reshape(y_train, (y_train.shape[0], -1))

    X_train, y_train = torch.tensor(X_train).to(torch.float32), torch.tensor(y_train).to(
        torch.float32)  # 将数据转变为tensor张量
    ds = torch.utils.data.TensorDataset(X_train, y_train)
    dataloader = torch.utils.data.DataLoader(ds, batch_size=batch_size,shuffle=True)  # 对训练集数据进行打包，每32个数据进行打包一次，组后一组不足32的自动打包
    return dataloader, X_train, y_train

dataloader_train, X_train, y_train = data_loader(window, length_size, batch_size, data_train)
dataloader_test, X_test, y_test = data_loader(window, length_size, batch_size, data_test)

class LSTM(nn.Module):
    def __init__(self, n_hidden, n_features):
        super().__init__()
        self.rnn = nn.RNN(input_size=n_features, hidden_size=n_hidden, num_layers=2, batch_first=True)
        self.fc = nn.Linear(in_features=n_hidden, out_features=length_size)

    def forward(self, x):
        o = self.rnn(x)[0][:, -1, :]
        o = self.fc(o)
        o = o.squeeze()
        return o

def model_test(path_URL):
    net = LSTM(n_features=n_feature,  n_hidden=n_hidden)
    net.load_state_dict(torch.load(path_URL))  # 加载训练好的模型
    net.eval()
    if length_size == 1:
        pred = net(X_test).unsqueeze(1)
    else:
        pred = net(X_test)
    pred = pred.detach().cpu()
    true = y_test.detach().cpu()
    y_data_test_inverse = scaler.fit_transform(np.array(data_target).reshape(-1, 1))
    pred_uninverse = scaler.inverse_transform(pred[:, -1:])  # 如果是多步预测，取预测结果的最后一列
    true_uninverse = scaler.inverse_transform(true[:, -1:])

    return true_uninverse, pred_uninverse





@app.route('/rnnTrain', methods=['POST'])
def rnnTrain():
    params = request.json
    learning_rate = params['learning_rate']
    epochs = params['epochs']
    n_hidden = params['n_hidden']
    n_features = params['n_features']
    length_size = params['length_size']

    print("Received parameters:")
    print("Learning rate:", learning_rate)
    print("Epochs:", epochs)
    print("Hidden size:", n_hidden)
    print("Number of features:", n_features)
    print("Length size:", length_size)

    net = LSTM(n_features=n_features, n_hidden=n_hidden)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    # 训练过程
    LOSS = []
    iteration = 0
    for epoch in range(epochs):
        for i, (datapoints, labels) in enumerate(dataloader_train):
            optimizer.zero_grad()
            if length_size == 1:
                preds = net(datapoints).unsqueeze(1)
            else:
                preds = net(datapoints)
            loss = criterion(preds, labels)
            LOSS.append(loss.detach().cpu().numpy())
            loss.backward()
            optimizer.step()

            iteration += 1
            if iteration % 100 == 0:
                print("Epoch {}, Iteration {}, Loss: {:.6f}".format(epoch, iteration, loss.item()))

    # 将模型保存到内存中
    buffer = io.BytesIO()
    torch.save(net.state_dict(), buffer)
    buffer.seek(0)

    print("Training completed. Sending trained model.")

    return send_file(buffer, mimetype='application/octet-stream', as_attachment=True, download_name='trained_model.pt')



""" 
RNN
"""
@app.route('/RNN', methods=['GET'])
def rnn():
    path_URL = request.args.get('path')
    true, pred = model_test(path_URL)
    result_finally = np.concatenate((true, pred), axis=1)
    df = pd.DataFrame(result_finally, columns=['real', 'pred'])
    time = np.arange(len(result_finally))
    plt.figure(figsize=(12, 6))
    plt.plot(time, result_finally[:, 0], color='#1f77b4', linestyle='-', linewidth=2, label='true')
    plt.plot(time, result_finally[:, 1], color='#ff7f0e', linestyle='--', linewidth=2, label='pred')
    plt.title('RNN Prediction Results', fontsize=16)
    plt.xlabel('Time', fontsize=14)
    plt.ylabel('Value', fontsize=14)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # 保存图像到静态目录
    filename = 'rnn_prediction.png'
    filepath = os.path.join(app.static_folder, filename)
    plt.savefig(filepath)

    # 返回图片的访问地址
    image_url = url_for('static', filename=filename)
    plt.close()

    return jsonify({'image_url': image_url})

    
    
@app.route('/upload_model', methods=['POST'])
def upload_model():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    # 指定上传文件的保存路径
    upload_dir = './uploaded_models'
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)
    
    # 保存上传的文件到指定路径
    file_path = os.path.join(upload_dir, file.filename)
    file.save(file_path)
    
    return jsonify({'success': 'File uploaded successfully', 'file_path': file_path})


if __name__ == '__main__':
    app.run(debug=True)