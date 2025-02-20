# 选择 Python 3.9 作为基础镜像
FROM python:3.9

# 设置工作目录
WORKDIR /app

# 复制应用代码
COPY app /app
COPY requirements.txt /app/

# 安装 Python 依赖
RUN pip install --no-cache-dir -r requirements.txt

# 创建必要的文件夹
RUN mkdir -p uploads outputs

# 运行 FastAPI 服务器
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]