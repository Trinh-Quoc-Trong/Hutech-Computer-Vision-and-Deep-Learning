import torch  
import torchvision  
from torchvision.models.detection import FasterRCNN  
from torchvision.models.detection.rpn import AnchorGenerator  
import cv2  
import numpy as np  
import matplotlib.pyplot as plt  
from collections import Counter  

class VehicleDetector:  
    def __init__(self):  
        """  
        Khởi tạo detector sử dụng Faster R-CNN  
        """  
        # Kiểm tra GPU  
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
        print(f"🖥️ Đang sử dụng: {self.device}")  
        
        # Load mô hình Faster R-CNN  
        self.model = self._load_faster_rcnn()  
        self.model.to(self.device)  
        self.model.eval()  
        
        # Danh mục xe chi tiết  
        self.vehicle_categories = {  
            'personal': ['car', 'sedan', 'suv', 'hatchback'],  
            'commercial': ['truck', 'van', 'bus', 'pickup'],  
            'two_wheel': ['motorcycle', 'bicycle'],  
            'heavy': ['lorry', 'trailer']  
        }  
    
    def _load_faster_rcnn(self):  
        """  
        Tải mô hình Faster R-CNN  
        """  
        # Load backbone ResNet50  
        backbone = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)  
        backbone = torch.nn.Sequential(*list(backbone.children())[:-2])  
        backbone.out_channels = 2048  
        
        # Tạo anchor generator  
        anchor_generator = AnchorGenerator(  
            sizes=((32, 64, 128, 256, 512),),  
            aspect_ratios=((0.5, 1.0, 2.0),) * 5  
        )  
        
        # Tạo mô hình Faster R-CNN  
        model = FasterRCNN(  
            backbone,  
            num_classes=91,  # COCO dataset có 90 classes + background  
            rpn_anchor_generator=anchor_generator  
        )  
        
        return model  
    
    def detect_and_count_vehicles(self, image_path, confidence_threshold=0.5):  
        """  
        Phát hiện và đếm xe với Faster R-CNN  
        """  
        try:  
            # Đọc ảnh  
            image = cv2.imread(image_path)  
            if image is None:  
                raise ValueError(f"❌ Không thể đọc ảnh từ {image_path}")  
            
            # Chuyển đổi ảnh sang tensor  
            image_tensor = self._preprocess_image(image)  
            
            # Dự đoán  
            with torch.no_grad():  
                predictions = self.model(image_tensor)  
            
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
            for box, label, score in zip(  
                predictions[0]['boxes'],  
                predictions[0]['labels'],  
                predictions[0]['scores']  
            ):  
                if score >= confidence_threshold:  
                    label_name = self._get_label_name(label.item())  
                    
                    # Phân loại  
                    for category, classes in self.vehicle_categories.items():  
                        if label_name.lower() in classes:  
                            detected_vehicles.append({  
                                'class': label_name,  
                                'category': category,  
                                'confidence': score.item(),  
                                'bbox': box.cpu().numpy().astype(int)  
                            })  
                            
                            # Cập nhật số lượng  
                            vehicle_counts['total'] += 1  
                            vehicle_counts['by_category'][category] += 1  
                            vehicle_counts['detailed_count'][label_name] = \
                                vehicle_counts['detailed_count'].get(label_name, 0) + 1  
                            break  
            
            return {  
                'image': cv2.cvtColor(image, cv2.COLOR_BGR2RGB),  
                'vehicles': detected_vehicles,  
                'counts': vehicle_counts  
            }  
        
        except Exception as e:  
            print(f"❌ Lỗi nhận diện: {e}")  
            return None  
    
    def _preprocess_image(self, image):  
        """  
        Tiền xử lý ảnh trước khi đưa vào mô hình  
        """  
        # Chuyển ảnh sang tensor  
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
        image = image.astype(np.float32) / 255.0  
        image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)  
        return image.to(self.device)  
    
    def _get_label_name(self, label_id):  
        """  
        Lấy tên class từ ID  
        """  
        # Sử dụng COCO dataset labels  
        coco_labels = [  
            'unlabeled', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',   
            'traffic light', 'fire hydrant', 'street sign', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',   
            'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'hat', 'backpack', 'umbrella',   
            'shoe', 'eye glasses', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',   
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'plate',   
            'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli',   
            'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'mirror',   
            'dining table', 'window', 'desk', 'toilet', 'door', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',   
            'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'blender', 'book', 'clock', 'vase',   
            'scissors', 'teddy bear', 'hair drier', 'toothbrush'  
        ]  
        return coco_labels[label_id] if label_id < len(coco_labels) else 'unknown'  
    
    def visualize_detection(self, detection_result):  
        """  
        Trực quan hóa kết quả nhận diện  
        """  
        if detection_result is None:  
            print("❌ Không có dữ liệu để hiển thị")  
            return  
        
        plt.figure(figsize=(16, 7))  
        
        # Ảnh nhận diện  
        plt.subplot(121)  
        plt.imshow(detection_result['image'])  
        plt.title('Nhận Diện Xe')  
        
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
                bbox[0], bbox[1]-10,   
                f"{vehicle['class']} ({vehicle['confidence']:.2f})",   
                color='red'  
            )  
        
        # Biểu đồ thống kê  
        plt.subplot(122)  
        counts = detection_result['counts']['detailed_count']  
        plt.bar(counts.keys(), counts.values())  
        plt.title('Thống Kê Loại Xe')  
        plt.xticks(rotation=45)  
        
        plt.tight_layout()  
        plt.show()  
    
    def generate_traffic_report(self, detection_result):  
        """  
        Tạo báo cáo chi tiết  
        """  
        if detection_result is None:  
            return "❌ Không có dữ liệu báo cáo"  
        
        counts = detection_result['counts']  
        
        report = f"""  
        🚗 BÁO CÁO THỐNG KÊ PHƯƠNG TIỆN  
        -------------------------------  
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

# Hàm main để chạy  
def main():  
    try:  
        # Tạo detector  
        detector = VehicleDetector()  
        
        # Đường dẫn ảnh   
        image_path = 'data_test/test_image001.jpg'  
        
        # Nhận diện xe  
        detection = detector.detect_and_count_vehicles(image_path)  
        
        # Hiển thị kết quả  
        detector.visualize_detection(detection)  
        
        # In báo cáo  
        print(detector.generate_traffic_report(detection))  
    
    except Exception as e:  
        print(f"❌ Lỗi chương trình: {e}")  

# Chạy chính  
if __name__ == "__main__":  
    main()  