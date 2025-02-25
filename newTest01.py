import cv2  
import numpy as np  
import matplotlib.pyplot as plt  
from PIL import Image  

my_path = "test_g_img_001.jpg"  

img = cv2.imread(my_path)  

if img is None:  
    raise FileNotFoundError(f"Không thể mở được ảnh từ {my_path}")  

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  

# Tách các kênh R, G, B từ ảnh  
r, g, b = cv2.split(img_rgb)  

# Hàm cân bằng histogram cho từng kênh  
def equalize_histogram(channel):  
    return cv2.equalizeHist(channel)  

# Cân bằng histogram cho từng kênh  
r_eq = equalize_histogram(r)  
g_eq = equalize_histogram(g)  
b_eq = equalize_histogram(b)  

# Kết hợp lại thành ảnh RGB sau khi cân bằng  
img_eq = cv2.merge((r_eq, g_eq, b_eq))  

fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(10,10))
ax1.imshow(img_rgb)
ax2.imshow(img_eq)
ax3.plot(np.arange(256), cv2.calcHist([r], [0], None, [256], [0, 256]), color="red", label="R")  
ax3.plot(np.arange(256), cv2.calcHist([g], [0], None, [256], [0, 256]), color="green", label="G")  
ax3.plot(np.arange(256), cv2.calcHist([b], [0], None, [256], [0, 256]), color="blue", label="B")  
ax3.set_title("Histogram Đường - Ảnh Gốc")  
ax3.legend()  

ax4.plot(np.arange(256), cv2.calcHist([r_eq], [0], None, [256], [0, 256]), color="red", label="R")  
ax4.plot(np.arange(256), cv2.calcHist([g_eq], [0], None, [256], [0, 256]), color="green", label="G")  
ax4.plot(np.arange(256), cv2.calcHist([b_eq], [0], None, [256], [0, 256]), color="blue", label="B")  
ax4.set_title("Histogram Đường - Ảnh Sau Cân Bằng")  
ax4.legend()  

plt.show()