#  <(6O9)> 

from cProfile import label
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from torch import mode


#  taoj duwx lieu giải định 
houser_are = np.array([50,70,80,100,120, ]) #dien tihcs cua nha 
house_price = np.array([200, 270, 300, 370, 450])

# taoj dataframe
data = pd.DataFrame({
    'house_price': house_price
    'house_area': houser_are
})

# thực  hienejn hồi quy tuyến tính 
# chú ý sklearn yêu cầu rehape dữ liệu đầu vào
X = house_price.reshape(-1,1)
y = house_price

model = LinearRegression()
model.fit(x,y)

print("điểm cắt ()", model.intercept_)
print("hệ số góc (slope)", model.coef_[0])
print("R-squared", model.score(X, y))

#  vẽ biểu đồ 
plt.figure(figsize=(10,6))
plt.scatter(houser_are, house_price, color = "red", label = "đường hồi quy tuyến tính giá nhà ")


new_house_area = np.array([85, 110, 150]).reshape(-1,1)
predicted_prices = model













