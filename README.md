# MNIST 数字识别项目

基于深度学习的手写数字识别项目，使用卷积神经网络实现 99%+ 的识别准确率。

### 👤 项目信息
- **姓名**：佴悦
- **学号**：112311170312
- **GitHub**：https://github.com/nnsvsy/3CI.git
- **部署网址**：https://112311170312naiyue.onrender.com/

## 📁 项目结构

```
├── code/              # 代码目录
│   ├── app.py         # Flask Web 应用
│   ├── train_tf_cnn.py # CNN 模型训练脚本
│   ├── requirements.txt # Python 依赖
│   ├── Procfile       # Render 部署配置
│   └── templates/     # HTML 模板
├── report/            # 实验报告
│   └── experiment_report.md
├── results/           # 实验结果
│   ├── submission.csv # Kaggle 提交文件
│   └── training_logs/ # 训练日志
└── README.md          # 项目说明
```

## 🚀 快速开始

### 环境要求
- Python 3.8+
- TensorFlow 2.15+
- Flask 2.3+

### 安装依赖
```bash
pip install -r code/requirements.txt
```

### 训练模型
```bash
python code/train_tf_cnn.py
```

### 启动 Web 服务
```bash
python code/app.py
```

### 部署到 Render
1. 将代码推送到 GitHub
2. 在 Render 创建 Web Service
3. 设置构建命令：`pip install -r requirements.txt`
4. 设置启动命令：`gunicorn app:app`

## 📊 实验结果

| 指标 | 数值 |
|------|------|
| 训练准确率 | 99.9%+ |
| 验证准确率 | 99.2%+ |
| 测试样本数 | 28,000 |

## 📝 提交说明规范

- `feat`: 添加新功能
- `fix`: 修复 bug
- `docs`: 更新文档
- `data`: 更新数据
- `model`: 模型更新

## 📄 License

MIT License