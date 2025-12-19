# CNN-model-for-quickdraw

## Giới thiệu
Đây là một model CNN đơn giản dùng để nhận dạng hình vẽ tay với kích thước 28×28 pixel.
Model được huấn luyện trên bộ dữ liệu **QuickDraw** với 10 nhãn:

`cat`, `dog`, `donut`, `apple`, `banana`, `cloud`, `clock`, `book`, `chair`, `face`

## Dataset
Dữ liệu được lấy từ QuickDraw dưới dạng `.npy`. Bạn có thể tải trực tiếp từ các đường link sau:

- https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/cat.npy  
- https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/dog.npy  
- https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/donut.npy  
- https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/apple.npy  
- https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/banana.npy  
- https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/cloud.npy  
- https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/clock.npy  
- https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/book.npy  
- https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/chair.npy  
- https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/face.npy  

## Training
Model được train trong file `main.py`.

```bash
python main.py
