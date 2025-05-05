# 间歇过程质量预测系统 - 后端服务

## 项目概述
基于Flask框架开发的间歇过程质量预测系统后端服务，提供RESTful API接口支持前端数据展示和预测功能,前端地址在https://github.com/garcey6/quality-prediction-frontend.git

## 项目结构
flask-back-end/
├── app/                          # 主应用目录
│   ├── init .py                  # 应用初始化
│   ├── api/                      # 路由定义
│   ├── data/                     # 训练数据
│   ├── models/                   # 学习模型
│   ├── static/                   # 静态资源
│   └── uploads/                  # 配置文件
│   └── run.py                    # 启动脚本
├── requirements.txt              # Python依赖包
├── .gitignore                    # Git忽略文件
└── README.md                     # 项目说明


## 环境要求
- Python 3.8+
- Flask 2.0+
- MySQL 5.7+ 或 SQLite3

## 安装指南
1. 克隆项目仓库
2. 创建虚拟环境：
   ```bash
   python -m venv venv
3. 激活虚拟环境：
   Windows:
   ```bash
   venv\Scripts\activate
   ```
   Mac/Linux:
   ```bash
   source venv/bin/activate
   ```
4. 安装依赖包：
   ```bash
   pip install -r requirements.txt
   ```

## 运行项目
运行run.py文件即可