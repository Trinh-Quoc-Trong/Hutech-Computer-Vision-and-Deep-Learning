import torch
import torchvision.transforms as transforms
from PIL import Image
import argparse
import os

# Import lớp mô hình từ file huấn luyện
# Đảm bảo train_cnn.py nằm cùng thư mục hoặc trong PYTHONPATH
from train_cnn import SimpleCNN, device # Import model và device từ script kia

# Định nghĩa các phép biến đổi cho ảnh đầu vào (GIỐNG HỆT validation transform)
image_size = (128, 128)
predict_transforms = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Ánh xạ index sang tên lớp (Lấy từ lúc huấn luyện, thường là {'Cat': 0, 'Dog': 1})
# Bạn có thể lấy tự động nếu lưu class_to_idx lúc huấn luyện,
# hoặc đặt thủ công nếu biết chắc chắn
# Dựa vào output lúc chạy train_cnn.py: Classes found: ['Cat', 'Dog']
class_names = ['Cat', 'Dog']

def predict_image(model, image_path, transform, device):
    """Nạp ảnh, biến đổi và dự đoán lớp."""
    try:
        # Mở ảnh
        img = Image.open(image_path).convert('RGB') # Đảm bảo ảnh là RGB
    except Exception as e:
        print(f"Error opening or converting image {image_path}: {e}")
        return None

    # Áp dụng transform
    img_t = transform(img)
    # Thêm chiều batch (mô hình cần input dạng [batch_size, channels, height, width])
    batch_t = torch.unsqueeze(img_t, 0)

    # Chuyển tensor sang device (CPU/GPU)
    batch_t = batch_t.to(device)

    # Đặt model ở chế độ đánh giá
    model.eval()

    # Thực hiện dự đoán (không cần tính gradient)
    with torch.no_grad():
        outputs = model(batch_t)

    # Lấy index của lớp có xác suất cao nhất
    _, predicted_idx = torch.max(outputs, 1)

    # Chuyển index sang tên lớp
    predicted_class = class_names[predicted_idx.item()]

    return predicted_class

if __name__ == "__main__":
    # --- Thiết lập Argument Parser --- 
    parser = argparse.ArgumentParser(description='Predict cat or dog from an image using a trained CNN model.')
    parser.add_argument('image_path', type=str, help='Path to the input image file.')
    parser.add_argument('--model_path', type=str, default='cat_dog_cnn_model.pth', help='Path to the trained model file (.pth).')
    args = parser.parse_args()

    # Kiểm tra file ảnh tồn tại
    if not os.path.exists(args.image_path):
        print(f"Error: Image file not found at {args.image_path}")
        exit()

    # Kiểm tra file model tồn tại
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        exit()

    # --- Nạp mô hình --- 
    # Khởi tạo kiến trúc mô hình (số lớp phải khớp lúc huấn luyện)
    # Ở đây num_classes=2 vì có 'Cat' và 'Dog'
    num_classes = len(class_names)
    loaded_model = SimpleCNN(num_classes=num_classes)

    # Nạp trọng số đã lưu
    # Cần map location để chạy được trên CPU nếu model được huấn luyện trên GPU
    map_location = torch.device('cpu') if device.type == 'cpu' else None
    try:
        loaded_model.load_state_dict(torch.load(args.model_path, map_location=map_location))
    except Exception as e:
        print(f"Error loading model state_dict from {args.model_path}: {e}")
        exit()

    # Chuyển model sang device
    loaded_model = loaded_model.to(device)

    print(f"Model {args.model_path} loaded successfully on {device}.")

    # --- Thực hiện dự đoán --- 
    prediction = predict_image(loaded_model, args.image_path, predict_transforms, device)

    # --- In kết quả --- 
    if prediction:
        print(f"\nPrediction for {args.image_path}: {prediction}")
