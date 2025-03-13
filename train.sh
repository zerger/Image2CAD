@echo off
setlocal

:: 配置 Tesseract 相关路径
set TESSDATA_PREFIX=C:\Program Files\Tesseract-OCR
set TRAINING_DIR=%CD%\tesseract_train
set FONT_NAME=SimSun
set LANG=custom_chinese

:: 创建训练目录
mkdir %TRAINING_DIR%
cd %TRAINING_DIR%

:: 1. 生成训练文本（可以手动创建一个文本文件）
echo 你好，世界！OCR 训练示例。 > train_text.txt

:: 2. 生成训练图片
text2image --text train_text.txt --outputbase train_font --font "%FONT_NAME%" --fonts_dir "C:\Windows\Fonts" --ptsize 32 

:: 3. 生成 .box 文件（字符框标注）
tesseract train_font.tif train_font box.train

:: 4. 生成 lstmf 特征数据
tesseract train_font.tif train_font lstm.train

:: 5. 训练 LSTM
lstmtraining --model_output %LANG%.traineddata --traineddata "%TESSDATA_PREFIX%\tessdata\chi_sim.traineddata" --train_listfile train_font.lstmf --max_iterations 400

:: 6. 复制到 tessdata 目录
copy %LANG%.traineddata "%TESSDATA_PREFIX%\tessdata\"

echo 训练完成！
pause
