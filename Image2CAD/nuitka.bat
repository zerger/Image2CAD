@echo off
REM 设置 Python 解释器路径
set PYTHON_PATH=E:\ProgramData\miniconda3\python.exe

REM 设置 Nuitka 命令
set NUITKA_CMD=%PYTHON_PATH% -m nuitka

REM 编译 image2cad.py 并包含 centerline 包
%NUITKA_CMD% --module --include-package=Centerline image2cad.py

REM 完成提示
echo 编译完成！
pause