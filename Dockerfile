# 使用官方的 Python 基础镜像
FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

# 将当前目录的内容复制到工作目录中
COPY . /app

# 安装 Python 依赖
RUN pip install --no-cache-dir -r requirements.txt

# 暴露应用程序运行的端口（假设 Flask 应用运行在 9003 端口）
EXPOSE 9003

# 设置环境变量
ENV FLASK_APP=src/ocrServer.py

# 启动命令
CMD ["flask", "run", "--host=0.0.0.0", "--port=9003"]