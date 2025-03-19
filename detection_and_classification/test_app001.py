import cv2  
import numpy as np  
from ultralytics import YOLO  
import matplotlib.pyplot as plt  

class VehicleDetector:  
    def __init__(self, model_path='yolov8n.pt'):  
        """  
        Khởi tạo detector với mô hình YOLO  
        
        Args:  
            model_path (str): Đường dẫn đến mô hình pre-trained  
        """  
        self.model = YOLO(model_path)  
        
        # Danh sách các lớp xe được quan tâm  
        self.vehicle_classes = [  
            'car', 'truck', 'bus', 'motorcycle',   
            'bicycle', 'van', 'pickup'  
        ]  
    
    def detect_vehicles(self, image_path, confidence_threshold=0.5):  
        """  
        Phát hiện và nhận diện xe từ hình ảnh  
        
        Args:  
            image_path (str): Đường dẫn ảnh  
            confidence_threshold (float): Ngưỡng tin cậy  
        
        Returns:  
            dict: Thông tin các xe được phát hiện  
        """  
        # Đọc ảnh  
        image = cv2.imread(image_path)  
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
        
        # Dự đoán  
        results = self.model(image_path)[0]  
        
        # Lưu trữ kết quả  
        detected_vehicles = []  
        
        for box in results.boxes:  
            # Lấy thông tin  
            cls = int(box.cls[0])  
            conf = float(box.conf[0])  
            
            # Kiểm tra lớp và độ tin cậy  
            if (self.model.names[cls] in self.vehicle_classes and   
                conf >= confidence_threshold):  
                
                # Thông tin bounding box  
                x1, y1, x2, y2 = map(int, box.xyxy[0])  
                
                detected_vehicles.append({  
                    'class': self.model.names[cls],  
                    'confidence': conf,  
                    'bbox': [x1, y1, x2, y2]  
                })  
        
        return {  
            'image': image_rgb,  
            'vehicles': detected_vehicles  
        }  
    
    def visualize_detection(self, detection_result):  
        """  
        Hiển thị kết quả nhận diện  
        
        Args:  
            detection_result (dict): Kết quả từ detect_vehicles  
        """  
        plt.figure(figsize=(12, 8))  
        plt.imshow(detection_result['image'])  
        
        for vehicle in detection_result['vehicles']:  
            bbox = vehicle['bbox']  
            plt.gca().add_patch(plt.Rectangle(  
                (bbox[0], bbox[1]),   
                bbox[2] - bbox[0],   
                bbox[3] - bbox[1],   
                fill=False,   
                edgecolor='red',   
                linewidth=2  
            ))  
            plt.text(  
                bbox[0], bbox[1] - 10,   
                f"{vehicle['class']} ({vehicle['confidence']:.2f})",   
                color='red'  
            )  
        
        plt.axis('off')  
        plt.tight_layout()  
        plt.show()  

# Sử dụng  
detector = VehicleDetector()  
detection = detector.detect_vehicles(r'data_test\test_image004.jpg')  
detector.visualize_detection(detection)