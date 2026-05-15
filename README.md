# MNIST 手写数字识别

基于深度学习的手写数字识别系统，使用卷积神经网络实现 99%+ 的识别准确率。

## 👤 项目信息

| 项目 | 内容 |
|------|------|
| 姓名 | 佴悦 |
| 学号 | 112311170312 |
| 姓名学号组合 | naiyue112311170312 |
| 班级 | 计算机科学与技术 |
| 实验名称 | MNIST 手写数字识别 |

## 📁 项目结构

```
112311170312naiyue/
├── code/              # 代码目录
│   ├── app.py         # Flask Web 应用
│   ├── train_tf_cnn.py # CNN 模型训练
│   ├── requirements.txt # Python 依赖
│   ├── Procfile       # Render 部署配置
│   └── templates/     # HTML 模板
├── report/            # 实验报告
│   └── experiment_report.md
├── results/           # 实验结果
│   └── submission.csv # Kaggle 提交文件
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

## 📊 实验结果

| 指标 | 数值 |
|------|------|
| 训练准确率 | 99.9%+ |
| 验证准确率 | 99.2%+ |
| 测试样本数 | 28,000 |

## 🔗 部署信息

- **GitHub 仓库**: https://github.com/nnsvsy/3CI.git
- **部署网址**: https://112311170312naiyue.onrender.com/

## 📝 提交说明规范

- `feat`: 添加新功能
- `fix`: 修复 bug
- `docs`: 更新文档
- `data`: 更新数据
- `model`: 模型更新

## 📄 License

MIT License