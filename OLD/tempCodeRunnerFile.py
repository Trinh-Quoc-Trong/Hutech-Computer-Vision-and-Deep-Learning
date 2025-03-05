import cv2  
import numpy as np  
import matplotlib.pyplot as plt  

def gaussian_kernel(size, sigma):  
    """Tạo kernel Gaussian với kích thước và độ lệch chuẩn cho trước."""  
    kernel = np.zeros((size, size))  
    center = size // 2  
    for i in range(size):  
        for j in range(size):  
            x = i - center  
            y = j - center  
            kernel[i, j] = np.exp(-(x**2 + y**2) / (2 * sigma**2))  
    kernel /= kernel.sum()  # Chuẩn hóa kernel  
    return kernel  

def apply_gaussian_filter(image, kernel):  
    """Áp dụng bộ lọc Gaussian lên ảnh."""  
    height, width, channels = image.shape  
    kernel_size = kernel.shape[0]  
    pad = kernel_size // 2  
    padded_image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REFLECT)  
    smoothed_image = np.zeros_like(image)  

    for c in range(channels):  
        for i in range(height):  
            for j in range(width):  
                region = padded_image[i:i+kernel_size, j:j+kernel_size, c]  
                smoothed_image[i, j, c] = np.sum(region * kernel)  

    return smoothed_image  

# Đọc ảnh từ file  
image = cv2.imread('pepper.png')  

# Kiểm tra xem ảnh có được đọc thành công không  
if image is None:  
    print("Không thể đọc ảnh. Vui lòng kiểm tra lại đường dẫn.")  
else:  
    kernel_size = 15  
    sigma = 3  
    kernel = gaussian_kernel(kernel_size, sigma)  

    smoothed_image = apply_gaussian_filter(image, kernel)  

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
    smoothed_image_rgb = cv2.cvtColor(smoothed_image, cv2.COLOR_BGR2RGB)  

    plt.figure(figsize=(10, 5))  

    plt.subplot(1, 2, 1)  
    plt.title('Original Image')  
    plt.imshow(image_rgb)  
    plt.axis('off')  

    plt.subplot(1, 2, 2)  
    plt.title('Smoothed Image')  
    plt.imshow(smoothed_image_rgb)  
    plt.axis('off')  

    plt.show() 