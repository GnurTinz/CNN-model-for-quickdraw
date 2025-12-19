from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

model = load_model('quickdraw_cnn.h5')
classes = [
    "cat", "dog","donut", "apple", "banana",
    "cloud", "clock", "book", "chair", "face"
]

img = Image.open("my_draw.png").convert("L")
img = img.resize((28, 28))
img = np.array(img)

plt.imshow(img, cmap="gray")
plt.title("Your drawing")
plt.axis("off")
plt.show()

img = img / 255.0
img = img.reshape(1, 28, 28, 1)

pred = model.predict(img)
label = np.argmax(pred)

print("Model predicts:", classes[label])
