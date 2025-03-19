

import os
import urllib.request

def download_cascade_files():
    """
    Tải các file Haar Cascade
    """
    # Danh sách các file cascade phổ biến
    cascades = {
        'car': 'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_car.xml',
        'bus': 'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_bus.xml',
        'frontalface': 'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml'
    }
    
    # Thư mục lưu trữ
    cascade_dir = os.path.join(os.getcwd(), 'haarcascades')
    os.makedirs(cascade_dir, exist_ok=True)
    
    # Tải từng file
    for name, url in cascades.items():
        file_path = os.path.join(cascade_dir, f'haarcascade_{name}.xml')
        
        try:
            print(f"🔄 Đang tải {name} cascade...")
            urllib.request.urlretrieve(url, file_path)
            print(f"✅ Tải thành công {name} cascade")
        except Exception as e:
            print(f"❌ Lỗi tải {name} cascade: {e}")
    
    print(f"\n📂 Các file cascade đã được lưu tại: {cascade_dir}")
    return cascade_dir

# Chạy script
if __name__ == "__main__":
    download_dir = download_cascade_files()
