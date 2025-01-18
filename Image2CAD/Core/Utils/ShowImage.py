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

        # 默认缩放比例为 1
        self.scale_factor = 1.0

        # 创建 Canvas 组件用于显示图像
        self.canvas = tk.Canvas(self.root, width=self.max_width, height=self.max_height)
        self.canvas.pack()

        # 当前平移位置
        self.offset_x = 0
        self.offset_y = 0

        # 创建初始缩放后的图像
        self.update_image()

        # 绑定鼠标滚轮事件（缩放）
        self.root.bind("<MouseWheel>", self.on_mouse_wheel)

        # 绑定鼠标拖动事件（平移）
        self.canvas.bind("<ButtonPress-1>", self.on_drag_start)
        self.canvas.bind("<B1-Motion>", self.on_drag_motion)

    def update_image(self):
        """根据当前缩放比例更新图像"""
        # 计算新的图像尺寸
        new_width = int(self.img_width * self.scale_factor)
        new_height = int(self.img_height * self.scale_factor)

        # 计算最大缩放比例，确保图像尺寸不超过 Canvas 的大小
        max_scale_x = self.max_width / self.img_width
        max_scale_y = self.max_height / self.img_height
        max_scale = min(max_scale_x, max_scale_y)

        # 确保缩放后的图像不会超出 Canvas
        if self.scale_factor > max_scale:
            self.scale_factor = max_scale

        # 更新图像尺寸
        new_width = int(self.img_width * self.scale_factor)
        new_height = int(self.img_height * self.scale_factor)

        # 调整图像大小
        resized_image = self.image_pil.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # 转换为 Tkinter 可用的图像格式
        self.photo = ImageTk.PhotoImage(resized_image)

        # 清空 Canvas 上的图像
        self.canvas.delete("all")

        # 更新 Canvas 上的图像，确保图像居中
        center_x = self.max_width // 2
        center_y = self.max_height // 2
        image_x = center_x + self.offset_x
        image_y = center_y + self.offset_y

        # 打印调试信息
        # print(f"Image size: {new_width}x{new_height}")
        # print(f"Canvas center: ({center_x}, {center_y})")
        # print(f"Image position: ({image_x}, {image_y})")

        # 显示图像
        self.canvas.create_image(image_x, image_y, image=self.photo)


    def on_mouse_wheel(self, event):
        """处理鼠标滚轮事件以缩放图像"""
        if event.delta > 0:  # 向上滚动（放大）
            self.scale_factor *= 1.1
        elif event.delta < 0:  # 向下滚动（缩小）
            self.scale_factor /= 1.1

        # 更新图像并刷新显示
        self.update_image()

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
        self.offset_x += dx
        self.offset_y += dy

        # 更新图像位置
        self.update_image()

        # 更新拖动起点
        self.drag_start_x = event.x
        self.drag_start_y = event.y

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
        viewer = ImageViewer(image, 'Image Viewer', max_width, max_height)
        viewer.show()
        
    # 显示图像（根据 GUI 后端切换）
    @staticmethod
    def show_image(image, title):
        if USE_OPENCV:
            ShowImage.show_image_opencv(image, title)
        else:
            ShowImage.show_image_tkinter(image, title)