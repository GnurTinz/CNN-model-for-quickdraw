import tensorflow as tf
from tensorflow.keras import models, layers
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


classes = [
    "cat", "dog","donut", "apple", "banana",
    "cloud", "clock", "book", "chair", "face"
]

X_list = []
y_list = []

for idx, cls in enumerate(classes):
    data = np.load(f"{cls}.npy")
    X_list.append(data)
    y_list.append(np.full(len(data), idx))

x = np.concatenate(X_list)
y = np.concatenate(y_list)
#chuẩn hoá dữ liệu
x = x.reshape(-1, 28, 28, 1)
x = x / 255.0

#chia dữ liệu
x_train, x_val, y_train, y_val = train_test_split(
    x, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = models.Sequential([
    layers.Conv2D(32,(3,3), activation = 'relu', input_shape = (28,28,1)),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(64,(3,3), activation = 'relu'),
    layers.MaxPooling2D((2,2)),

    layers.Flatten(),
    layers.Dense(128, activation = 'relu'),
    layers.Dense(10, activation = 'softmax')

])

model.compile(
    optimizer = 'adam',
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy']
)

history = model.fit(
    x_train, y_train,
    epochs = 5,
    batch_size = 128,
    validation_split = 0.1
)
model.save('quickdraw_cnn.h5')
test_loss, test_acc = model.evaluate(x_val, y_val)
print(f'accuracy: {test_acc}')
