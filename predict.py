import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# ---------------------------
# LOAD MODEL
# ---------------------------
model = load_model("fruit_model.h5")

# ---------------------------
# LOAD IMAGE
# ---------------------------
img_path = r"C:\Fruits\apple_PNG12503.png"  # path to your test image

img = image.load_img(img_path, target_size=(128,128))
img_array = image.img_to_array(img) / 255.0  # scale pixel values
img_array = np.expand_dims(img_array, axis=0)  # batch dimension

# ---------------------------
# 
# ---------------------------
pred_probs = model.predict(img_array)
pred_class = np.argmax(pred_probs)  # returns 0,1,2
class_names = ['Apple','Banana','Watermelon']
print("Predicted class:", class_names[pred_class])


# ---------------------------
# SHOW IMAGE
# ---------------------------
plt.imshow(img)
plt.title(f"Prediction: {class_names[pred_class]}")
plt.axis('off')
plt.show()
