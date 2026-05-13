# MNIST 数字识别实验报告

## 实验目标
使用深度学习模型实现手写数字识别，目标准确率达到 99%+

## 实验方法
1. **数据集**：Kaggle MNIST 数据集（42000 训练样本，28000 测试样本）
2. **模型**：卷积神经网络（CNN）
3. **框架**：TensorFlow/Keras

## 模型架构
```
Input (28x28x1)
├─ Conv2D(32) + BatchNorm + ReLU
├─ Conv2D(32) + BatchNorm + ReLU
├─ MaxPooling2D(2x2) + Dropout(0.25)
├─ Conv2D(64) + BatchNorm + ReLU
├─ Conv2D(64) + BatchNorm + ReLU
├─ MaxPooling2D(2x2) + Dropout(0.25)
├─ Conv2D(128) + BatchNorm + ReLU
├─ MaxPooling2D(2x2) + Dropout(0.25)
├─ Flatten
├─ Dense(256) + BatchNorm + ReLU + Dropout(0.5)
└─ Dense(10, softmax)
```

## 实验结果
- **训练准确率**：99.9%+
- **验证准确率**：99.2%+
- **测试集预测**：已生成 submission.csv

## 文件结构
```
code/          # 代码文件
report/        # 实验报告
results/       # 结果文件
README.md      # 仓库说明
```

## 运行说明
```bash
# 训练模型
python code/train_tf_cnn.py

# 启动 Web 服务
python code/app.py
```

## 结论
成功实现了高准确率的 MNIST 数字识别模型，可直接部署为 Web 应用。