import cv2  
import numpy as np  
import mediapipe as mp  
import matplotlib.pyplot as plt  
from collections import Counter  

class VehicleDetector:  
    def __init__(self):  
        """  
        Khởi tạo detector sử dụng MediaPipe Object Detection  
        """  
        # Cấu hình MediaPipe Object Detection  
        self.mp_objectron = mp.solutions.objectron  
        self.mp_drawing = mp.solutions.drawing_utils  
        
        # Khởi tạo detector với cấu hình chi tiết  
        self.objectron = self.mp_objectron.Objectron(  
            static_image_mode=True,  
            max_num_objects=10,  
            min_detection_confidence=0.5,  
            model_name='Bicycle'  # Có thể thay đổi: 'Bicycle', 'Cup', 'Chair', 'Camera', 'Shoe'  
        )  
        
        # Danh mục xe chi tiết  
        self.vehicle_categories = {  
            'two_wheel': ['bicycle', 'motorcycle'],  
            'personal': ['car', 'sedan', 'suv'],  
            'commercial': ['truck', 'van', 'bus'],  
            'heavy': ['lorry', 'trailer']  
        }  
    
    def detect_and_count_vehicles(self, image_path):  
        """  
        Phát hiện, nhận diện và đếm xe  
        """  
        # Đọc ảnh  
        image = cv2.imread(image_path)  
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
        
        # Nhận diện đối tượng  
        results = self.objectron.process(image_rgb)  
        
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
        
        # Xử lý kết quả  
        if results.detected_objects:  
            for detected_object in results.detected_objects:  
                # Thông tin vị trí  
                bbox = detected_object.location_data.relative_bounding_box  
                
                # Chuyển đổi tọa độ  
                h, w, _ = image.shape  
                x = int(bbox.xmin * w)  
                y = int(bbox.ymin * h)  
                width = int(bbox.width * w)  
                height = int(bbox.height * h)  
                
                # Phân loại và đếm  
                vehicle_class = 'bicycle'  # Mặc định với model hiện tại  
                
                detected_vehicles.append({  
                    'class': vehicle_class,  
                    'bbox': [x, y, x+width, y+height]  
                })  
                
                vehicle_counts['total'] += 1  
                for category, classes in self.vehicle_categories.items():  
                    if vehicle_class in classes:  
                        vehicle_counts['by_category'][category] += 1  
                        break  
                
                vehicle_counts['detailed_count'][vehicle_class] = \
                    vehicle_counts['detailed_count'].get(vehicle_class, 0) + 1  
        
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
        
        # Ảnh nhận diện  
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
                vehicle['class'],   
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