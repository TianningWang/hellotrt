from tensorflow.keras import datasets
import numpy as np
import cv2
from PIL import Image
from matplotlib.pyplot import imsave

save_path = "/home/tiwang/code/trt/mnist/data/"

(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

for i in range(256):
    # im = Image.fromarray(np.uint8(x_train[i]))
    save_name = str(save_path + str(y_train[i]) + str("-") + str(i) + '.jpg')
    print(save_name)
    imsave(save_name, x_train[i])

