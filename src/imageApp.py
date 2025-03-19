# -*- coding: utf-8 -*-

import cv2
import numpy as np
import argparse
from tkinter import Tk, Button, Canvas, PhotoImage, Scale, HORIZONTAL, font
from PIL import Image, ImageTk
import threading
from ocrProcess import OCRProcess 
import uuid
from pathlib import Path
import tempfile

class ImageApp:
    def __init__(self, root, image_path):
        self.root = root
        self.root.title("Image Viewer with OCR")
        
        # Load image
        self.image_path = image_path
        self.original_image = cv2.imread(image_path)
        self.display_image = self.original_image.copy()
        self.photo = None
        
        # Create canvas
        self.canvas = Canvas(root)
        self.canvas.pack(fill="both", expand=True)  # Allow canvas to expand
        
        # Add buttons
        self.btn_frame = Button(root, text="框选OCR", command=self.recognize_text)
        self.btn_frame.pack(side="bottom")
        
        # Bind events
        self.canvas.bind("<ButtonPress-3>", self.start_pan)
        self.canvas.bind("<B3-Motion>", self.pan_image)
        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel)  # Add mouse wheel zoom
        self.root.bind("<Configure>", self.on_resize)  # Bind window resize event

        # Initialize variables
        self.rect = None
        self.start_x = None
        self.start_y = None
        self.rect_id = None
        self.pan_start_x = None
        self.pan_start_y = None
        self.offset_x = 0
        self.offset_y = 0
        self.zoom_factor = 1.0
        self.resize_timer = None
        self.text_positions = []
        
        self.update_image()

    def on_resize(self, event):
        # Cancel any existing timer
        if self.resize_timer:
            self.root.after_cancel(self.resize_timer)
        # Set a new timer to update the image after 200ms
        self.resize_timer = self.root.after(200, self.update_image)

    def update_image(self):
        # Apply zoom and offset
        height, width = self.original_image.shape[:2]
        new_size = (int(width * self.zoom_factor), int(height * self.zoom_factor))
        resized_image = cv2.resize(self.original_image, new_size)
        
        # Crop the image based on offset
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        x1 = int(min(max(self.offset_x, 0), new_size[0] - canvas_width))
        y1 = int(min(max(self.offset_y, 0), new_size[1] - canvas_height))
        x2 = int(x1 + canvas_width)
        y2 = int(y1 + canvas_height)
        cropped_image = resized_image[y1:y2, x1:x2]
        
        # Convert image to PhotoImage
        self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)))
        self.canvas.create_image(0, 0, image=self.photo, anchor="nw")

        # Redraw rectangles and text
        for text, x, y, width, height, angle in self.text_positions:
            # Adjust coordinates based on zoom and offset
            x = int((x - self.offset_x) * self.zoom_factor)
            y = int((y - self.offset_y) * self.zoom_factor)
            width = int(width * self.zoom_factor)
            height = int(height * self.zoom_factor)
            
            # Convert y coordinate from bottom-left origin to top-left origin
            y = canvas_height - y - height
            
            box = (x, y, x + width, y + height)
            self.canvas.create_rectangle(box[0], box[1], box[2], box[3], outline='green')
            
            # Calculate font size based on bounding box height
            font_size = max(10, int(height * 0.8))  # Ensure a minimum font size
            text_font = font.Font(family="Helvetica", size=font_size, weight="bold")
            self.canvas.create_text(box[0], box[1] - 10, text=text, fill='green', anchor='nw', font=text_font)

    def start_pan(self, event):
        # Save pan start position
        self.pan_start_x = event.x
        self.pan_start_y = event.y

    def pan_image(self, event):
        # Calculate offset
        dx = event.x - self.pan_start_x
        dy = event.y - self.pan_start_y
        self.offset_x -= dx / self.zoom_factor  # Adjust for zoom
        self.offset_y -= dy / self.zoom_factor  # Adjust for zoom
        self.pan_start_x = event.x
        self.pan_start_y = event.y
        self.update_image()

    def on_mouse_wheel(self, event):
        # Adjust zoom factor
        if event.delta > 0:
            self.zoom_factor *= 1.1
        else:
            self.zoom_factor /= 1.1
        self.zoom_factor = max(0.1, min(self.zoom_factor, 3.0))  # Limit zoom range
        self.update_image()

    def recognize_text(self):
        # Temporarily bind mouse events for selection
        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)

    def on_button_press(self, event):
        # Save mouse drag start position
        self.start_x = event.x
        self.start_y = event.y
        # Create rectangle if not yet exist
        if not self.rect_id:
            self.rect_id = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, outline='red')

    def on_mouse_drag(self, event):
        # Update rectangle as you drag the mouse
        cur_x, cur_y = (event.x, event.y)
        self.canvas.coords(self.rect_id, self.start_x, self.start_y, cur_x, cur_y)

    def on_button_release(self, event):
        # Finalize rectangle
        self.rect = (self.start_x, self.start_y, event.x, event.y)
        # Unbind mouse events after selection
        self.canvas.unbind("<ButtonPress-1>")
        self.canvas.unbind("<B1-Motion>")
        self.canvas.unbind("<ButtonRelease-1>")
        self.process_selected_region()

    def process_selected_region(self):
        if self.rect:
            x0, y0, x1, y1 = self.rect
            # Adjust coordinates based on zoom and offset
            x0 = int((x0 + self.offset_x) / self.zoom_factor)
            y0 = int((y0 + self.offset_y) / self.zoom_factor)
            x1 = int((x1 + self.offset_x) / self.zoom_factor)
            y1 = int((y1 + self.offset_y) / self.zoom_factor)
            # Extract the selected region
            selected_region = self.original_image[y0:y1, x0:x1]
            
            # Create a temporary file for the selected region
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
                temp_path = temp_file.name
                cv2.imwrite(temp_path, selected_region)
            
            # Initialize OCR processor
            ocr_processor = OCRProcess()
            self.text_positions, _ = ocr_processor.get_ocr_result_rapidOCR(temp_path, scale_factor=5, max_block_size=512, overlap=20)
            
            # Display results (for demonstration, just print)
            print("Recognized text positions:", self.text_positions)
            self.update_image()

            # Remove the rectangle after processing
            if self.rect_id:
                self.canvas.delete(self.rect_id)
                self.rect_id = None

if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description="Image Viewer with OCR")    
    parser.add_argument('image_path', type=str, help='Path to the image file')   
    args = parser.parse_args()
      
    root = Tk()  
    app = ImageApp(root, args.image_path)  
    root.mainloop()