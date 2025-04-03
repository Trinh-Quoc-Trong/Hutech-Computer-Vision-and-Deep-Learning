import tensorflow as tf
from tensorflow.keras.datasets import cifar10
import numpy as np
import matplotlib.pyplot as plt

# Tải dữ liệu CIFAR-10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# In thông tin về bộ dữ liệu
print("Kích thước tập training:", x_train.shape)
print("Kích thước tập test:", x_test.shape)
print("\nSố lượng mẫu training:", len(x_train))
print("Số lượng mẫu test:", len(x_test))

# Danh sách các lớp
class_names = ['Máy bay', 'Xe hơi', 'Chim', 'Mèo', 'Chó', 'Hươu', 'Ếch', 'Ngựa', 'Tàu', 'Xe tải']

# Hiển thị một số ảnh mẫu
plt.figure(figsize=(10, 10))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(x_train[i])
    plt.title(class_names[y_train[i][0]])
    plt.axis('off')
plt.show()

# In thông tin về một ảnh
print("\nThông tin về một ảnh:")
print("Kích thước:", x_train[0].shape)
print("Giá trị pixel (min, max):", np.min(x_train[0]), np.max(x_train[0]))
print("Lớp:", class_names[y_train[0][0]]) 