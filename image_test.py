from PIL import Image
import base64
import pickle
import numpy as np

img_path = '000.jpg'
image = Image.open(img_path)
w, h = image.size  # Resize to a standard size if needed

ser1_image = pickle.dumps(np.array(image))
ser1_image = base64.b64encode(ser1_image).decode('utf-8')
print('Base64 encoded image size in bytes:', len(ser1_image))

image = image.resize((w // 2, h // 2))  # Ensure the image is in a consistent size
ser2_image = pickle.dumps(np.array(image))
ser2_image = base64.b64encode(ser2_image).decode('utf-8')
print('Base64 encoded image size in bytes:', (len(ser1_image) - len(ser2_image)) / len(ser1_image) * 100)