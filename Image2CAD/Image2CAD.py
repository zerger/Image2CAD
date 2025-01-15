# -*- coding: utf-8 -*-
"""
@author: Aditya Intwala

 Copyright (C) 2016, Aditya Intwala.

 Licensed under the Apache License 2.0. See LICENSE file in the project root for full license information.
"""

import cv2
import time
import os
import sys
import numpy as np

global make_dir_root, timestr

def is_background_white(image):
    # 获取图像的高度和宽度
    height, width, _ = image.shape
    
    # 获取图像四个角的颜色
    corners = [
        image[0, 0],  # 左上角
        image[0, width - 1],  # 右上角
        image[height - 1, 0],  # 左下角
        image[height - 1, width - 1]  # 右下角
    ]
    
    # 计算四个角的平均颜色
    average_color = np.mean(corners, axis=0)
    # 判断背景颜色是否接近白色
    return np.all(average_color > 200)

def invert_image_colors(image):
    return cv2.bitwise_not(image)

def main(argv1):

    img_path = argv1
    #img_path = "..\\TestData\\1.png"
    img_path = os.path.abspath(img_path)

    dir = os.path.dirname(img_path)
    os.chdir(dir)
    currentDir = os.getcwd()
    
    from Core.Features.Texts.TextsFeature import TextsFeature
    from Core.Features.Arrowheads.ArrowHeadsFeature import ArrowHeadsFeature
    from Core.Features.Circles.CirclesFeature import CirclesFeature
    from Core.Features.LineSegments.LineSegmentsFeature import LineSegmentsFeature
    from Core.Features.Cognition.DimensionalLinesFeature import DimensionalLinesFeature
    from Core.Features.Cognition.Cognition import Cognition
    from Core.Features.Cognition.SupportLinesFeature import SupportLinesFeature
    from Core.Features.FeatureManager import FeatureManager
    from Core.Utils.Eraser import Eraser
    from Core.Utils.I2CWriter import I2CWriter
    from Core.Utils.DXFWriter import DXFWriter
    
    timestr = time.strftime("%Y%m%d-%H%M%S")
    Start_time = time.strftime("%H hr - %M min - %S sec")
    print("Image2CAD Script Started at " + Start_time + "...")
    base = os.path.basename(img_path)
    folder_name = os.path.splitext(base)[0]
    print("Image Loaded: " + img_path + "...")
    print("Making Required Directory...")
    make_dir_Output = r"./Output"
    if not os.path.exists(make_dir_Output):
        os.mkdir(make_dir_Output)
    make_dir_folder = r"./Output/"+folder_name
    make_dir_root = r"./Output/"+folder_name+r"/"+timestr
    if not os.path.exists(make_dir_folder):
        os.mkdir(make_dir_folder)
    os.mkdir(make_dir_root)
        
    print("Initializing Feature Manager...")
    FM = FeatureManager()
    FM._ImagePath = img_path
    FM._RootDirectory = make_dir_root
    img = cv2.imread(FM._ImagePath)
    
    # 判断背景是否为白色，如果不是则进行反色处理
    if not is_background_white(img):
        img = invert_image_colors(img)
        
    FM._ImageOriginal = img
    FM._ImageCleaned = img.copy()
    FM._ImageDetectedDimensionalText = img.copy()
    FM._ImageDetectedCircle = img.copy()
    Erased_Img = img.copy()
    
    AD_Time = time.strftime("%H hr - %M min - %S sec")
    print("Arrow Detection Started at " + AD_Time + "...")
    BB_Arrows, Arrow_Img = ArrowHeadsFeature.Detect(FM, 35, 70)
    FM._DetectedArrowHead = BB_Arrows
    FM._ImageDetectedArrow = Arrow_Img
    print("Arrow Detection Complete...")
    
    # 定义一个窗口名称
    WINDOW_NAME = "测试窗口"    
    # 初始化窗口（可选：指定窗口大小或其他属性）
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    # 显示图像并刷新内容
    cv2.imshow(WINDOW_NAME, FM._ImageDetectedArrow)
    cv2.setWindowTitle(WINDOW_NAME, "Detected Arrows")
    cv2.waitKey(0)  # 设置 5秒的延迟以刷新内容   
    # cv2.imshow("Detected Arrows", FM._ImageDetectedArrow)
    # cv2.waitKey(0)
    
    for i in BB_Arrows:
        P1 = i._BoundingBoxP1
        P2 = i._BoundingBoxP2
        Erased_Img = Eraser.EraseBox(FM._ImageCleaned, P1, P2)
    FM._ImageCleaned = Erased_Img
    
    DL_Time = time.strftime("%H hr - %M min - %S sec")
    print("Dimensional Line Detection Started at " + DL_Time + "...")    
    segments, DimensionalLine_Img = DimensionalLinesFeature.Detect(FM)
    FM._ImageDetectedDimensionalLine = DimensionalLine_Img
    FM._DetectedDimensionalLine = segments
    print("Dimensional Line Detection Complete...")  
    cv2.imshow(WINDOW_NAME, FM._ImageDetectedDimensionalLine)
    cv2.setWindowTitle(WINDOW_NAME, "Detected Dimensional Lines")
    cv2.waitKey(0)

    for j in segments:
      for i in j._Leaders:
            P1 = i.startPoint
            P2 = i.endPoint
            Erased_Img = Eraser.EraseLine(FM._ImageCleaned, P1, P2)
    FM._ImageCleaned = Erased_Img
    
    print("Correlating ArrowHead Direction...")
    Cognition.ArrowHeadDirection(FM)
    print("Correlating ArrowHead Direction Complete...")
    
    TE_Time = time.strftime("%H hr - %M min - %S sec")
    print("Text Area Extraction Started at " + TE_Time + "...")
    ExtractedTextArea, TextArea_Img = TextsFeature.Detect(FM)
    FM._ImageDetectedDimensionalText = TextArea_Img
    FM._DetectedDimensionalText = ExtractedTextArea
    print("Text Area Extraction Complete...") 
    cv2.imshow(WINDOW_NAME, FM._ImageDetectedDimensionalText)
    cv2.setWindowTitle(WINDOW_NAME, "Detected Text Area")
    cv2.waitKey(0)

    for i in ExtractedTextArea:
        P1 = i._TextBoxP1
        P2 = i._TextBoxP2
        Erased_Img = Eraser.EraseBox(FM._ImageCleaned, P1, P2)
    FM._ImageCleaned = Erased_Img
    
    DC_Time = time.strftime("%H hr - %M min - %S sec")
    print("Correlation of Dimensions Started at " + DC_Time + "...")
    Dimension_correlate = Cognition.ProximityCorrelation(FM)
    FM._DetectedDimension = Dimension_correlate
    print("Correlation of Dimensions complete...")
         
    LD_Time = time.strftime("%H hr - %M min - %S sec")
    print("Line Detection Started at " + LD_Time + "...")
    segments, DetectedLine_Img = LineSegmentsFeature.Detect(FM)
    FM._DetectedLine = segments
    FM._ImageDetectedLine = DetectedLine_Img
    print("Line Detection Complete...")    
    cv2.imshow(WINDOW_NAME, FM._ImageDetectedLine)
    cv2.setWindowTitle(WINDOW_NAME, "Detected Lines")
    cv2.waitKey(0)

    print("Correlation of Support Lines Started...")
    SupportLinesFeature.Detect(FM)
    print("Correlation of Support Lines Complete...")
    
    print("Correction of Broken Ends Stage 1 Started...")
    Cognition.CorrectEnds(FM)
    print("Correction of Broken Ends Stage 1 Complete...")

    print("Correction of Broken Ends Stage 2 Started...")
    Cognition.JoinLineSegmentsWithinProximityTolerance(FM) 
    print("Correction of Broken Ends Stage 2 Complete...")
           
    for i in segments:
        for ls in i:
            P1 = ls.startPoint
            P2 = ls.endPoint
            Erased_Img = Eraser.EraseLine(FM._ImageCleaned, P1, P2)
    FM._ImageCleaned = Erased_Img
    
    print("Entity Correlation Started...")
    Cognition.EntityCorrelation(FM)
    print("Entity Correlation Complete...")
        
    CD_Time = time.strftime("%H hr - %M min - %S sec")
    print("Circle Detection Started at " + CD_Time + "...")
    detectedcircle, DetectedCircle_Img = CirclesFeature.Detect(FM)
    FM._ImageDetectedCircle = DetectedCircle_Img
    FM._DetectedCircle = detectedcircle
    print("Circle Detection Complete...")   
    cv2.imshow(WINDOW_NAME, FM._ImageDetectedCircle)
    cv2.setWindowTitle(WINDOW_NAME, "Detected circles")
    cv2.waitKey(0)

    for i in detectedcircle:
        center = i._centre
        radius = i._radius
        Erased_Img = Eraser.EraseCircle(FM._ImageCleaned, center, radius)
    FM._ImageCleaned = Erased_Img
    
    print("Exporting Extracted Data to I2C File...")
    I2CWriter.Write(FM)
    print("Exporting Complete...")
    
    print("Exporting Extracted Data to DXF File...")
    DXFWriter.Write(FM)
    print("Exporting Complete...")
    
    print("Image2CAD Script Execution Complete...")


#if __name__ == "__main__":
#    main("Debug")
    
if __name__ == "__main__":
    main(sys.argv[1])


