# Tkinter 显示图像
import tkinter as tk
from PIL import Image, ImageTk
import cv2
class ImageViewer:
    def __init__(self, image, title, max_width=1080, max_height=768):
        self.root = tk.Tk()
        self.root.title(title)

        # 转换 OpenCV 图像为 Pillow 图像
        self.image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # BGR -> RGB
        self.image_pil = Image.fromarray(self.image_rgb)

        # 获取图像的原始大小
        self.img_width, self.img_height = self.image_pil.size

        # 设置最大显示区域尺寸（窗口大小）
        self.max_width = max_width
        self.max_height = max_height

        # 当前缩放比例初始为 1.0
        self.scale_factor = 1.0

        # 当前平移位置
        self.offset_x = 0
        self.offset_y = 0

        # 创建 Canvas 组件用于显示图像
        self.canvas = tk.Canvas(self.root, width=self.max_width, height=self.max_height)
        self.canvas.pack()

        # 设置初始缩放比例，使图像适应Canvas大小
        self.initial_scale()

        # 计算初始图像显示区域并更新显示
        self.update_image()

        # 绑定鼠标滚轮事件（缩放）
        self.root.bind("<MouseWheel>", self.on_mouse_wheel)

        # 绑定鼠标拖动事件（平移）
        self.canvas.bind("<ButtonPress-1>", self.on_drag_start)
        self.canvas.bind("<B1-Motion>", self.on_drag_motion)
        # 绑定双击事件（全显）
        self.canvas.bind("<Double-1>", self.on_double_click)

    def initial_scale(self):
        """计算图像初始缩放比例，使其适应 Canvas 大小"""
        scale_x = self.max_width / self.img_width
        scale_y = self.max_height / self.img_height
        self.scale_factor = min(scale_x, scale_y)  # 选择最小比例来适应窗口

    def update_image(self):
        """根据当前平移位置和图像区域更新显示的图像部分"""
        # 根据当前缩放比例计算图像新的尺寸
        new_width = int(self.img_width * self.scale_factor)
        new_height = int(self.img_height * self.scale_factor)

        # 使用 PIL 的 resize 方法调整图像大小
        resized_image = self.image_pil.resize((new_width, new_height), Image.Resampling.NEAREST)

        # 限制平移偏移量，避免超出图像边界
        self.offset_x = max(0, min(self.offset_x, new_width - self.max_width))
        self.offset_y = max(0, min(self.offset_y, new_height - self.max_height))

        # 根据当前缩放和偏移量计算显示区域的左上角和右下角
        left = self.offset_x
        top = self.offset_y
        right = left + self.max_width
        bottom = top + self.max_height

        # 裁剪图像的特定区域
        cropped_image = resized_image.crop((left, top, right, bottom))

        # 转换为 Tkinter 可用的图像格式
        self.photo = ImageTk.PhotoImage(cropped_image)

        # 清空 Canvas 上的图像
        self.canvas.delete("all")

        # 在 Canvas 上显示裁剪后的图像
        self.canvas.create_image(0, 0, image=self.photo, anchor="nw")

    def on_mouse_wheel(self, event):
        """处理鼠标滚轮事件以缩放图像"""
        if event.delta > 0:  # 向上滚动（放大）
            self.scale_factor *= 1.1
        elif event.delta < 0:  # 向下滚动（缩小）
            self.scale_factor /= 1.1

        self.root.after(100, self.update_image)

    def on_drag_start(self, event):
        """鼠标按下时记录拖动起点"""
        self.drag_start_x = event.x
        self.drag_start_y = event.y

    def on_drag_motion(self, event):
        """鼠标拖动时移动图像"""
        # 计算平移距离
        dx = event.x - self.drag_start_x
        dy = event.y - self.drag_start_y

        # 更新平移偏移量
        self.offset_x -= dx
        self.offset_y -= dy

        self.root.after(100, self.update_image)

        # 更新拖动起点
        self.drag_start_x = event.x
        self.drag_start_y = event.y
    
    def on_double_click(self, event):
        """处理双击事件以全显图像"""
        # 计算全显比例
        scale_x = self.max_width / self.img_width
        scale_y = self.max_height / self.img_height
        self.scale_factor = min(scale_x, scale_y)  # 选择最小比例来适应窗口
        
        self.root.after(100, self.update_image)

    def show(self):
        """显示图像"""
        self.root.mainloop()

class ShowImage():
    global USE_OPENCV, window_created
    # 选择 GUI 库的开关
    USE_OPENCV = False  # 切换为 False 使用 Tkinter

    # OpenCV 显示图像
    # 全局变量，记录窗口是否已创建
    window_created = False
    @staticmethod
    def show_image_opencv(image, title):
        global window_created
        # 定义一个窗口名称
        WINDOW_NAME = "测试窗口"    

        # 检查窗口是否已经创建
        if not window_created:
            # 初始化窗口（只在第一次显示时创建）
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)    
            window_created = True
        
        # 显示图像并更新窗口标题
        cv2.imshow(WINDOW_NAME, image)
        cv2.setWindowTitle(WINDOW_NAME, title)  # 更新窗口标题

        # 等待按键
        cv2.waitKey(0)  # 等待按键来关闭窗口
        # 关闭窗口
        cv2.destroyAllWindows()
        
    @staticmethod
    def show_image_tkinter(image, title, max_width=1080, max_height=768):
        viewer = ImageViewer(image, title, max_width, max_height)
        viewer.show()
        
    # 显示图像（根据 GUI 后端切换）
    @staticmethod
    def show_image(image, title):
        if USE_OPENCV:
            ShowImage.show_image_opencv(image, title)
        else:
            ShowImage.show_image_tkinter(image, title)