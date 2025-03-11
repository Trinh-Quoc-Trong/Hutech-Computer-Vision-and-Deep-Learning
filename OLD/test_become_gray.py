from PIL import Image  

def compare_gray_levels(image1_path, image2_path):  
    # Mở hình ảnh  
    image1 = Image.open(image1_path).convert('L')  # Chuyển đổi sang ảnh xám  
    image2 = Image.open(image2_path).convert('L')  # Chuyển đổi sang ảnh xám  
    
    # Tính toán mức xám trung bình  
    gray_level1 = sum(image1.getdata()) / (image1.size[0] * image1.size[1])  
    gray_level2 = sum(image2.getdata()) / (image2.size[0] * image2.size[1])  
    
    # Trả về sự khác biệt  
    return abs(gray_level1 - gray_level2)  

# Ví dụ sử dụng  
difference = compare_gray_levels('image1.jpg', 'image2.jpg')  
print(f"Sự khác biệt mức xám: {difference}")  