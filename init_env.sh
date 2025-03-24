#!/bin/bash

# 检查虚拟环境是否已存在
if [ ! -d "image2cad" ]; then
    echo "创建虚拟环境..."
    python3 -m venv image2cad
else
    echo "虚拟环境已存在，跳过创建步骤"
fi

# 激活虚拟环境
echo "激活虚拟环境..."
source image2cad/bin/activate

# 安装依赖
echo "安装依赖..."
pip3 install -r requirements.txt

echo "环境初始化完成"
