#!/bin/bash

# 启动 Redis 服务器
echo "Starting Redis server..."
redis-server --daemonize yes

# 检查 Redis 是否启动成功
if [ $? -eq 0 ]; then
    echo "Redis server started successfully."
else
    echo "Failed to start Redis server."
    exit 1
fi

# 启动 Celery worker
echo "Starting Celery worker..."
celery -A src.server.tasks worker --loglevel=info --detach

# 检查 Celery worker 是否启动成功
if [ $? -eq 0 ]; then
    echo "Celery worker started successfully."
else
    echo "Failed to start Celery worker."
    exit 1
fi

# 启动 FastAPI 应用
echo "Starting FastAPI application..."
uvicorn src.server.server:app --host 0.0.0.0 --port 8000 --reload &

# 检查 FastAPI 应用是否启动成功
if [ $? -eq 0 ]; then
    echo "FastAPI application started successfully."
else
    echo "Failed to start FastAPI application."
    exit 1
fi

echo "All services started successfully."