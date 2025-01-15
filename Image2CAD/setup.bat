@echo off
@REM  将控制台字符集切换为 UTF-8。
chcp 65001 >nul
REM 设置 Tesseract 的安装路径
set TESSERACT_DIR=E:\Program Files\Tesseract-OCR

REM 设置 PyInstaller 的工作目录
set WORK_DIR=%~dp0

REM 打包的主脚本
set MAIN_SCRIPT=image2cad.py

REM 输出的 EXE 名称
set OUTPUT_NAME=image2cad.exe

REM 定义资源路径（DLL、tesseract.exe 和 tessdata）
set ADD_DLLS=%TESSERACT_DIR%\*.dll;.
set ADD_TESSERACT=%TESSERACT_DIR%\tesseract.exe;.
set ADD_TESSDATA_ENG=%TESSERACT_DIR%\tessdata\eng.traineddata;.\tessdata
set ADD_TESSDATA_CHI=%TESSERACT_DIR%\tessdata\chi_sim.traineddata;.\tessdata
set ADD_TESSDATA_OSD=%TESSERACT_DIR%\tessdata\osd.traineddata;.\tessdata

REM 打包命令
pyinstaller --onefile --name %OUTPUT_NAME% ^
--add-data "%ADD_DLLS%" ^
--add-data "%ADD_TESSERACT%" ^
--add-data "%ADD_TESSDATA_ENG%" ^
--add-data "%ADD_TESSDATA_CHI%" ^
--add-data "%ADD_TESSDATA_OSD%" ^
%MAIN_SCRIPT%

REM 打包完成提示
if %ERRORLEVEL% equ 0 (
    echo 打包完成！
) else (
    echo 打包失败，请检查错误日志。
)

pause
