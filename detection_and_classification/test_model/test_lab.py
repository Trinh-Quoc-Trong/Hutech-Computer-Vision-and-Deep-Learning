from PIL import Image  

# Đặt đường dẫn đến ảnh cần hiển thị  
# image_path = "test_lab_image001.jpg"  # Thay thế bằng đường dẫn thực tế  
image_path = "test_model/test_lab_image001.jpg"  

# Mở ảnh  
try:  
    img = Image.open(image_path)  
    img.show()  # Hiển thị ảnh  
except FileNotFoundError:  
    print("Không tìm thấy tệp ảnh ở đường dẫn đã chỉ định.")  
except Exception as e:  
    print(f"Đã có lỗi xảy ra: {e}")  