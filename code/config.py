# 项目部署配置文件
# 姓名：佴悦
# 学号：112311170312

# GitHub 仓库地址
GITHUB_REPO = "https://github.com/nnsvsy/3CI.git"

# Render 部署网址
RENDER_URL = "https://112311170312naiyue.onrender.com/"

# 项目信息
PROJECT_NAME = "MNIST 数字识别"
MODEL_ACCURACY = "99.2%+"

# 部署配置
BUILD_COMMAND = "pip install -r code/requirements.txt"
START_COMMAND = "gunicorn code.app:app"