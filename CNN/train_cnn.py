import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
import os
from PIL import Image

# Kiểm tra xem CUDA có sẵn không và chọn thiết bị
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 1. Chuẩn bị dữ liệu
data_dir = 'PetImages'
image_size = (128, 128) # Kích thước ảnh đầu vào cho mô hình
batch_size = 32

# --- Hàm kiểm tra tính hợp lệ của ảnh ---
def is_valid_image(path):
    try:
        img = Image.open(path)
        img.verify() # Kiểm tra xem file có phải ảnh hợp lệ không
        # Kiểm tra xem ảnh có thể load được pixel data không (quan trọng với dataset này)
        # và kiểm tra ảnh có đúng 3 kênh màu (RGB) không
        img = Image.open(path) # Mở lại sau khi verify
        if img.mode != 'RGB':
            # print(f"Skipping non-RGB image: {path}") # Tạm ẩn print để đỡ rối console
            return False
        # Thử tải dữ liệu pixel
        img.load()
        return True
    except (IOError, SyntaxError, OSError) as e:
        # print(f"Skipping corrupted image {path}: {e}") # Tạm ẩn print để đỡ rối console
        return False

# --- Định nghĩa các phép biến đổi ảnh ---
# Resize, chuyển thành Tensor, và chuẩn hóa
data_transforms = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size), # Crop giữa ảnh để đảm bảo kích thước
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # Giá trị chuẩn hóa phổ biến cho ImageNet
])

# 2. Xây dựng mô hình CNN
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2): # num_classes=2 vì có chó và mèo
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1), # Input 3 kênh (RGB), output 16 kênh
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # Giảm kích thước còn 64x64

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # Giảm kích thước còn 32x32

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # Giảm kích thước còn 16x16
        )
        # Tính toán kích thước đầu vào cho lớp fully connected
        # Sau 3 lớp MaxPool2d (stride=2), kích thước ảnh là 128 / (2*2*2) = 16
        # Số kênh đầu ra của lớp conv cuối là 64
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5), # Dropout để giảm overfitting
            nn.Linear(512, num_classes) # Lớp output với số lượng class
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Hàm chính để thực thi
def main():
    print(f"Using device: {device}")

    # --- Tải dữ liệu sử dụng ImageFolder và hàm kiểm tra ảnh ---
    # ImageFolder sẽ tự động gán nhãn dựa trên tên thư mục con (Cat, Dog)
    print("Loading dataset...")
    full_dataset = datasets.ImageFolder(
        data_dir,
        transform=data_transforms,
        is_valid_file=is_valid_image # Sử dụng hàm kiểm tra ảnh
    )

    # In ra số lượng ảnh hợp lệ và các lớp tìm thấy
    print(f"Total valid images found: {len(full_dataset)}")
    if not full_dataset.classes:
        print("Error: No classes found. Check the 'PetImages' directory structure.")
        return
    print(f"Classes found: {full_dataset.classes}")
    class_to_idx = full_dataset.class_to_idx
    print(f"Class to index mapping: {class_to_idx}")

    if len(full_dataset) == 0:
        print("Error: No valid images were loaded. Check image files and paths.")
        return

    # --- Chia dữ liệu thành tập huấn luyện và tập kiểm tra (validation) ---
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    # Đảm bảo val_size không âm nếu dataset quá nhỏ
    if val_size < 0:
        val_size = 0
        train_size = len(full_dataset)

    if train_size == 0:
        print("Error: Training set has size 0. Not enough data.")
        return

    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")

    # --- Tạo DataLoaders ---
    # Sử dụng num_workers=0 trên Windows nếu gặp lỗi liên quan đến multiprocessing
    # Hoặc đảm bảo code chạy trong if __name__ == '__main__'
    num_workers = 4 if device.type == 'cuda' else 0 # Dùng 0 worker nếu là CPU để tránh lỗi
    print(f"Using {num_workers} workers for data loading.")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True if device.type == 'cuda' else False)
    # Chỉ tạo val_loader nếu val_dataset có dữ liệu
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True if device.type == 'cuda' else False) if val_size > 0 else None

    # Khởi tạo mô hình và chuyển sang thiết bị (GPU/CPU)
    model = SimpleCNN(num_classes=len(full_dataset.classes)).to(device)
    print("\nModel architecture:")
    print(model) # In cấu trúc mô hình

    # 3. Định nghĩa hàm mất mát và trình tối ưu hóa
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 4. Huấn luyện mô hình
    num_epochs = 10 # Số lượng epochs (có thể điều chỉnh)

    print("\nStarting training...")

    for epoch in range(num_epochs):
        # --- Training Phase ---
        model.train() # Đặt mô hình ở chế độ huấn luyện
        running_loss = 0.0
        running_corrects = 0
        total_train = 0

        for i, batch_data in enumerate(train_loader):
            # Kiểm tra xem batch_data có đúng định dạng không
            if not isinstance(batch_data, (list, tuple)) or len(batch_data) != 2:
                print(f"Warning: Skipping invalid batch data at index {i}")
                continue

            inputs, labels = batch_data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Zero a gradients tham số
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            # Backward pass và optimize
            loss.backward()
            optimizer.step()

            # Thống kê
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            total_train += inputs.size(0)

            # In tiến trình mỗi 100 batches
            if (i + 1) % 100 == 0:
                batch_loss = loss.item()
                batch_acc = torch.sum(preds == labels.data).double() / inputs.size(0)
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}/{len(train_loader)}], Train Loss: {batch_loss:.4f}, Train Acc: {batch_acc:.4f}")

        # Kiểm tra total_train để tránh chia cho 0
        if total_train == 0:
            print(f"Warning: Epoch {epoch+1} had no training data processed.")
            epoch_train_loss = 0.0
            epoch_train_acc = 0.0
        else:
            epoch_train_loss = running_loss / total_train
            epoch_train_acc = running_corrects.double() / total_train

        # --- Validation Phase ---
        epoch_val_loss = 0.0
        epoch_val_acc = 0.0
        if val_loader: # Chỉ chạy validation nếu có val_loader
            model.eval() # Đặt mô hình ở chế độ đánh giá
            val_loss = 0.0
            val_corrects = 0
            total_val = 0

            with torch.no_grad(): # Không tính gradient trong phase validation
                for inputs, labels in val_loader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item() * inputs.size(0)
                    val_corrects += torch.sum(preds == labels.data)
                    total_val += inputs.size(0)

            # Kiểm tra total_val để tránh chia cho 0
            if total_val > 0:
                epoch_val_loss = val_loss / total_val
                epoch_val_acc = val_corrects.double() / total_val
            else:
                 print(f"Warning: Epoch {epoch+1} had no validation data processed.")


        print(f"\nEpoch [{epoch+1}/{num_epochs}] Summary:")
        print(f"  Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.4f}")
        if val_loader:
            print(f"  Valid Loss: {epoch_val_loss:.4f} | Valid Acc: {epoch_val_acc:.4f}")
        print("-" * 30)


    print("\nTraining finished.")

    # 5. Lưu mô hình đã huấn luyện (tùy chọn)
    model_save_path = "cat_dog_cnn_model.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

# Bảo vệ điểm vào chính của chương trình
if __name__ == '__main__':
    # Dòng này đôi khi cần thiết cho multiprocessing trên Windows khi đóng gói thành file exe
    # from multiprocessing import freeze_support
    # freeze_support()
    main()