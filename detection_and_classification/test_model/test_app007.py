import tensorflow as tf  
import tensorflow_hub as hub  
import numpy as np  
import cv2  

# Tải mô hình SSD từ TensorFlow Hub  
model_url = "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2"  
model = hub.load(model_url)  

# Hàm để nhận diện và phân loại phương tiện  
def detect_vehicles(image_path):  
    # Đọc ảnh  
    image = cv2.imread(image_path)  
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
    image_resized = cv2.resize(image_rgb, (320, 320))  # SSD yêu cầu kích thước 320x320  
    input_tensor = tf.convert_to_tensor(image_resized)  
    input_tensor = input_tensor[tf.newaxis, ...]  

    # Áp dụng mô hình  
    detections = model(input_tensor)  

    # Lấy kết quả  
    boxes = detections['detection_boxes'].numpy()  
    scores = detections['detection_scores'].numpy()  
    classes = detections['detection_classes'].numpy()  

    # Hiển thị kết quả  
    for i in range(len(scores[0])):  
        if scores[0][i] > 0.5:  # Ngưỡng tin cậy  
            box = boxes[0][i]  
            class_id = int(classes[0][i])  
            label = f"Class {class_id}, Score: {scores[0][i]:.2f}"  
            ymin, xmin, ymax, xmax = box  
            xmin = int(xmin * image.shape[1])  
            xmax = int(xmax * image.shape[1])  
            ymin = int(ymin * image.shape[0])  
            ymax = int(ymax * image.shape[0])  
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)  
            cv2.putText(image, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  

    # Hiển thị ảnh kết quả  
    cv2.imshow("Vehicle Detection", image)  
    cv2.waitKey(0)  
    cv2.destroyAllWindows()  

# Thực hiện nhận diện  
detect_vehicles("test_model/test_image001.jpg")  

