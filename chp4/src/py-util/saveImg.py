# writen by chatgpt
import tensorflow as tf
from tensorflow import keras
import numpy as np
import os

# 加载MNIST数据集
mnist = keras.datasets.mnist
(train_images, train_labels), _ = mnist.load_data()

# 随机选择256张图像
random_indices = np.random.choice(len(train_images), 256, replace=False)
random_images = train_images[random_indices]
random_labels = train_labels[random_indices]

# 创建保存图像的文件夹（如果不存在）
save_dir = 'mnist_images'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 保存图像到本地
for i in range(len(random_images)):
    image = random_images[i]
    label = random_labels[i]
    image_path = os.path.join(save_dir, f'{label}_{i}.png')
    tf.keras.preprocessing.image.save_img(image_path, image)

print("图像保存完成！")
