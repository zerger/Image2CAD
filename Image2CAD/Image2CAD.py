import cv2
import time
import os
import sys
import numpy as np
import builtins
from PIL import ImageFont, Image, ImageTk

# 替换全局 print 函数
original_print = builtins.print

def patched_print(*args, **kwargs):
    kwargs.setdefault('flush', True)
    original_print(*args, **kwargs)

builtins.print = patched_print

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
    from Core.Utils.ShowImage import ShowImage
    
    # 加载微软雅黑字体
    font_path = "C:/Windows/Fonts/msyh.ttc"  # 微软雅黑字体路径
    font = ImageFont.truetype(font_path, 32)
    
    timestr = time.strftime("%Y%m%d-%H%M")
    Start_time = time.strftime("%H hr - %M min - %S sec")
    print("Image2CAD 启动： " + Start_time + "...")
    base = os.path.basename(img_path)
    folder_name = os.path.splitext(base)[0]
    print("加载图片: " + img_path + "...")
    print("创建目录...")
    make_dir_Output = r"./Output"
    if not os.path.exists(make_dir_Output):
        os.mkdir(make_dir_Output)
    make_dir_folder = r"./Output/"+folder_name
    make_dir_root = r"./Output/"+folder_name+r"/"+timestr
    if not os.path.exists(make_dir_folder):
        os.mkdir(make_dir_folder)
    os.mkdir(make_dir_root)
        
    print("初始化特征管理器...")
    FM = FeatureManager()
    FM._ImagePath = img_path
    FM._RootDirectory = make_dir_root
    img = cv2.imread(FM._ImagePath)
    
    # 判断背景是否为白色，如果不是则进行反色处理
    if not is_background_white(img):
        img = invert_image_colors(img)
    # 放大图像
    scale_factor = 1  # 放大因子
    img = cv2.resize(img, None, fx=scale_factor, fy=scale_factor) 
       
    FM._ImageOriginal = img
    FM._ImageCleaned = img.copy()
    FM._ImageDetectedDimensionalText = img.copy()
    FM._ImageDetectedCircle = img.copy()
    Erased_Img = img.copy()
    
    AD_Time = time.strftime("%H hr - %M min - %S sec")
    print("开始标注尺寸箭头检测 " + AD_Time + "...")
    BB_Arrows, Arrow_Img = ArrowHeadsFeature.Detect(FM, 35, 70)
    FM._DetectedArrowHead = BB_Arrows
    FM._ImageDetectedArrow = Arrow_Img
    print("完成标注尺寸箭头检测...")
       
    ShowImage.show_image(FM._ImageDetectedArrow, "标注尺寸箭头检测")    
    
    for i in BB_Arrows:
        P1 = i._BoundingBoxP1
        P2 = i._BoundingBoxP2
        Erased_Img = Eraser.EraseBox(FM._ImageCleaned, P1, P2)
    FM._ImageCleaned = Erased_Img
    
    DL_Time = time.strftime("%H hr - %M min - %S sec")
    print("开始标注尺寸线检测 " + DL_Time + "...")    
    segments, DimensionalLine_Img = DimensionalLinesFeature.Detect(FM)
    FM._ImageDetectedDimensionalLine = DimensionalLine_Img
    FM._DetectedDimensionalLine = segments
    print("完成标注尺寸线检测...")     
    ShowImage.show_image(FM._ImageDetectedDimensionalLine, str("标注尺寸线检测"))   

    for j in segments:
      for i in j._Leaders:
            P1 = i.startPoint
            P2 = i.endPoint
            Erased_Img = Eraser.EraseLine(FM._ImageCleaned, P1, P2)
    FM._ImageCleaned = Erased_Img
    
    print("开始关联标注尺寸界线方向...")
    Cognition.ArrowHeadDirection(FM)
    print("完成关联标注尺寸界线方向...")
    
    TE_Time = time.strftime("%H hr - %M min - %S sec")
    print("开始文本区域提取 " + TE_Time + "...")
    ExtractedTextArea, TextArea_Img = TextsFeature.Detect(FM)
    FM._ImageDetectedDimensionalText = TextArea_Img
    FM._DetectedDimensionalText = ExtractedTextArea
    print("完成文本区域提取...")   
    ShowImage.show_image(FM._ImageDetectedDimensionalText, "文本区域检测")   

    for i in ExtractedTextArea:
        P1 = i._TextBoxP1
        P2 = i._TextBoxP2
        Erased_Img = Eraser.EraseBox(FM._ImageCleaned, P1, P2)
    FM._ImageCleaned = Erased_Img
    
    DC_Time = time.strftime("%H hr - %M min - %S sec")
    print("开始关联标注 " + DC_Time + "...")
    Dimension_correlate = Cognition.ProximityCorrelation(FM)
    FM._DetectedDimension = Dimension_correlate
    print("完成联标注...")
         
    LD_Time = time.strftime("%H hr - %M min - %S sec")
    print("开始直线检测 " + LD_Time + "...")
    segments, DetectedLine_Img = LineSegmentsFeature.Detect(FM)
    FM._DetectedLine = segments
    FM._ImageDetectedLine = DetectedLine_Img
    print("完成直线检测...")       
    ShowImage.show_image(FM._ImageDetectedLine, "直线检测")   

    print("开始支撑线的关联...")
    SupportLinesFeature.Detect(FM)
    print("完成支撑线的关联...")
    
    print("开始断裂端的校正一阶段...")
    Cognition.CorrectEnds(FM)
    print("完成断裂端的校正一阶段...")

    print("开始断裂端的校正二阶段...")
    Cognition.JoinLineSegmentsWithinProximityTolerance(FM) 
    print("完成断裂端的校正二阶段...")
           
    for i in segments:
        for ls in i:
            P1 = ls.startPoint
            P2 = ls.endPoint
            Erased_Img = Eraser.EraseLine(FM._ImageCleaned, P1, P2)
    FM._ImageCleaned = Erased_Img
    
    print("开始实体关联...")
    Cognition.EntityCorrelation(FM)
    print("完成实体关联...")
        
    CD_Time = time.strftime("%H hr - %M min - %S sec")
    print("开始圆检测 " + CD_Time + "...")
    detectedcircle, DetectedCircle_Img = CirclesFeature.Detect(FM)
    FM._ImageDetectedCircle = DetectedCircle_Img
    FM._DetectedCircle = detectedcircle
    print("完成圆检测...")      
    ShowImage.show_image(FM._ImageDetectedCircle, "圆检测")   

    for i in detectedcircle:
        center = i._centre
        radius = i._radius
        Erased_Img = Eraser.EraseCircle(FM._ImageCleaned, center, radius)
    FM._ImageCleaned = Erased_Img
    
    print("开始导出提取数据到I2C文件...")
    I2CWriter.Write(FM)
    print("完成导出提取数据到I2C文件...")
    
    print("E开始导出提取数据到dxf文件...")
    DXFWriter.Write(FM)
    print("完成导出提取数据到dxf文件...")
    
    print("Image2CAD执行完成...")


#if __name__ == "__main__":
#    main("Debug")
    
if __name__ == "__main__":
    main(sys.argv[1])


