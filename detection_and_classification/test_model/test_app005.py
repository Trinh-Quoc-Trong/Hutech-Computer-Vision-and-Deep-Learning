import os  
import cv2  
import numpy as np  
import matplotlib.pyplot as plt  

class VehicleDetector:  
    def __init__(self):  
        """  
        Khởi tạo detector với kiểm tra an toàn  
        """  
        # Xác định đúng đường dẫn Haar Cascade  
        self.haar_dir = self._find_haar_directory()  
        
        # Tải cascade files  
        self.car_cascade = self._load_cascade_safely('haarcascade_car.xml')  
        self.bus_cascade = self._load_cascade_safely('haarcascade_bus.xml')  
        
        # Danh mục xe  
        self.vehicle_categories = {  
            'personal': ['car', 'sedan', 'suv'],  
            'commercial': ['truck', 'van', 'bus'],  
            'two_wheel': ['motorcycle', 'bicycle'],  
            'heavy': ['lorry', 'trailer']  
        }  
    
    def _find_haar_directory(self):  
        """  
        Tìm thư mục chứa Haar Cascade files  
        """  
        # Các đường dẫn tiềm năng  
        potential_dirs = [  
            r'C:\Users\{}\AppData\Local\Programs\Python\Python{}\Lib\site-packages\cv2\data'.format(  
                os.getlogin(),   
                '.'.join(map(str, os.sys.version_info[:2]))  
            ),  
            os.path.join(os.path.dirname(cv2.__file__), 'data'),  
            cv2.data.haarcascades  
        ]  
        
        for directory in potential_dirs:  
            if os.path.exists(directory):  
                print(f"Tìm thấy thư mục Haar Cascade: {directory}")  
                return directory  
        
        raise FileNotFoundError("Không tìm thấy thư mục Haar Cascade")  
    
    def _load_cascade_safely(self, filename):  
        """  
        Nạp file cascade một cách an toàn  
        """  
        cascade_path = os.path.join(self.haar_dir, filename)  
        
        if not os.path.exists(cascade_path):  
            print(f"⚠️ Cảnh báo: Không tìm thấy file {filename}")  
            # Trả về cascade rỗng để tránh lỗi  
            return cv2.CascadeClassifier()  
        
        cascade = cv2.CascadeClassifier(cascade_path)  
        
        # Kiểm tra cascade có hợp lệ không  
        if cascade.empty():  
            print(f"❌ Lỗi: Không thể nạp cascade {filename}")  
            return None  
        
        return cascade  
    
    def detect_and_count_vehicles(self, image_path, scale_factor=1.1, min_neighbors=3):  
        """  
        Phát hiện và đếm xe với nhiều biện pháp phòng ngừa  
        """  
        # Kiểm tra tồn tại file  
        if not os.path.exists(image_path):  
            raise FileNotFoundError(f"❌ Không tìm thấy file {image_path}")  
        
        # Đọc ảnh an toàn  
        image = cv2.imread(image_path)  
        
        if image is None:  
            raise ValueError(f"❌ Không thể đọc ảnh từ {image_path}")  
        
        # Chuyển ảnh sang grayscale  
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
        
        # Nhận diện xe  
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
        
        # Nhận diện xe con (nếu có cascade)  
        cars = []  
        if self.car_cascade and not self.car_cascade.empty():  
            cars = self.car_cascade.detectMultiScale(  
                image_gray,   
                scaleFactor=scale_factor,   
                minNeighbors=min_neighbors  
            )  
        
        # Nhận diện xe buýt (nếu có cascade)  
        buses = []  
        if self.bus_cascade and not self.bus_cascade.empty():  
            buses = self.bus_cascade.detectMultiScale(  
                image_gray,   
                scaleFactor=scale_factor,   
                minNeighbors=min_neighbors  
            )  
        
        # Xử lý xe con  
        for (x, y, w, h) in cars:  
            detected_vehicles.append({  
                'class': 'car',  
                'bbox': [x, y, x+w, y+h]  
            })  
            vehicle_counts['total'] += 1  
            vehicle_counts['by_category']['personal'] += 1  
            vehicle_counts['detailed_count']['car'] = vehicle_counts['detailed_count'].get('car', 0) + 1  
        
        # Xử lý xe buýt  
        for (x, y, w, h) in buses:  
            detected_vehicles.append({  
                'class': 'bus',  
                'bbox': [x, y, x+w, y+h]  
            })  
            vehicle_counts['total'] += 1  
            vehicle_counts['by_category']['commercial'] += 1  
            vehicle_counts['detailed_count']['bus'] = vehicle_counts['detailed_count'].get('bus', 0) + 1  
        
        return {  
            'image': image_rgb,  
            'vehicles': detected_vehicles,  
            'counts': vehicle_counts  
        }  
    
    def visualize_detection(self, detection_result):  
        """  
        Hiển thị kết quả nhận diện  
        """  
        plt.figure(figsize=(15, 7))  
        plt.subplot(121)  
        plt.imshow(detection_result['image'])  
        plt.title('Ảnh Gốc')  
        
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
            plt.text(bbox[0], bbox[1]-10, vehicle['class'], color='red')  
        
        # Biểu đồ thống kê  
        plt.subplot(122)  
        counts = detection_result['counts']['detailed_count']  
        plt.bar(list(counts.keys()), list(counts.values()))  
        plt.title('Thống Kê Xe')  
        plt.xticks(rotation=45)  
        
        plt.tight_layout()  
        plt.show()  

# Hàm main để chạy  
def main():  
    try:  
        # Tạo detector  
        detector = VehicleDetector()  
        
        # Đường dẫn ảnh (điều chỉnh cho phù hợp)  
        image_path = r'data_test\test_image004.jpg'  
        
        # Nhận diện xe  
        detection = detector.detect_and_count_vehicles(image_path)  
        
        # Hiển thị kết quả  
        detector.visualize_detection(detection)  
        
        # In thống kê  
        print("\n🚗 Thống Kê Phương Tiện:")  
        for key, value in detection['counts']['detailed_count'].items():  
            print(f"- {key.upper()}: {value}")  
    
    except Exception as e:  
        print(f"❌ Lỗi: {e}")  

# Chạy chính  
if __name__ == "__main__":  
    main() 