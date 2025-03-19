import cv2  
import numpy as np  
from ultralytics import YOLO  
import matplotlib.pyplot as plt  
from collections import Counter  

class VehicleDetector:  
    def __init__(self, model_path='yolov8n.pt'):  
        """  
        Khởi tạo detector với mô hình YOLO và cấu hình chi tiết  
        """  
        self.model = YOLO(model_path)  
        
        # Danh mục xe chi tiết với nhóm phân loại  
        self.vehicle_categories = {  
            'personal': ['car', 'sedan', 'suv', 'hatchback', 'coupe'],  
            'commercial': ['truck', 'van', 'pickup', 'bus'],  
            'two_wheel': ['motorcycle', 'bicycle'],  
            'heavy': ['lorry', 'trailer', 'cargo_truck']  
        }  
    
    def detect_and_count_vehicles(self, image_path, confidence_threshold=0.5):  
        """  
        Phát hiện, nhận diện và đếm xe với phân loại chi tiết  
        
        Returns:  
            dict: Thông tin chi tiết về xe  
        """  
        # Đọc ảnh  
        image = cv2.imread(image_path)  
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
        
        # Dự đoán  
        results = self.model(image_path)[0]  
        
        # Lưu trữ kết quả  
        detected_vehicles = []  
        vehicle_counts = {  
            'total': 0,  
            'by_category': {  
                'personal': 0,  
                'commercial': 0,  
                'two_wheel': 0,  
                'heavy': 0  
            },  
            'detailed_count': {}  
        }  
        
        for box in results.boxes:  
            # Lấy thông tin  
            cls = int(box.cls[0])  
            conf = float(box.conf[0])  
            vehicle_class = self.model.names[cls]  
            
            # Kiểm tra và phân loại  
            for category, classes in self.vehicle_categories.items():  
                if vehicle_class.lower() in classes and conf >= confidence_threshold:  
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  
                    
                    detected_vehicles.append({  
                        'class': vehicle_class,  
                        'category': category,  
                        'confidence': conf,  
                        'bbox': [x1, y1, x2, y2]  
                    })  
                    
                    # Cập nhật số lượng  
                    vehicle_counts['total'] += 1  
                    vehicle_counts['by_category'][category] += 1  
                    vehicle_counts['detailed_count'][vehicle_class] = \
                        vehicle_counts['detailed_count'].get(vehicle_class, 0) + 1  
                    
                    break  
        
        return {  
            'image': image_rgb,  
            'vehicles': detected_vehicles,  
            'counts': vehicle_counts  
        }  
    
    def visualize_detection_with_count(self, detection_result):  
        """  
        Hiển thị kết quả nhận diện kèm thống kê  
        """  
        plt.figure(figsize=(16, 10))  
        plt.subplot(1, 2, 1)  
        plt.imshow(detection_result['image'])  
        
        # Vẽ bounding box  
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
        
        plt.title('Nhận Diện Xe')  
        plt.axis('off')  
        
        # Biểu đồ thống kê  
        plt.subplot(1, 2, 2)  
        counts = detection_result['counts']['detailed_count']  
        plt.bar(counts.keys(), counts.values())  
        plt.title('Thống Kê Loại Xe')  
        plt.xticks(rotation=45, ha='right')  
        
        plt.tight_layout()  
        plt.show()  
    
    def generate_traffic_report(self, detection_result):  
        """  
        Tạo báo cáo chi tiết về lưu lượng giao thông  
        """  
        counts = detection_result['counts']  
        
        report = f"""  
        BÁO CÁO THỐNG KÊ PHƯƠNG TIỆN  
        ----------------------------  
        Tổng số phương tiện: {counts['total']}  
        
        Phân loại:  
        - Xe cá nhân: {counts['by_category']['personal']}  
        - Xe thương mại: {counts['by_category']['commercial']}  
        - Xe hai bánh: {counts['by_category']['two_wheel']}  
        - Xe tải nặng: {counts['by_category']['heavy']}  
        
        Chi tiết từng loại:  
        {chr(10).join(f"- {xe}: {số_lượng}" for xe, số_lượng in counts['detailed_count'].items())}  
        """  
        
        return report  

# Sử dụng mẫu  
detector = VehicleDetector()  
detection = detector.detect_and_count_vehicles(r'data_test\test_image004.jpg')  
detector.visualize_detection_with_count(detection)  
print(detector.generate_traffic_report(detection)) 