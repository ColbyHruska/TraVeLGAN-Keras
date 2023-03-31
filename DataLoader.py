import numpy as np
import glob
import matplotlib.pyplot as plt
import cv2
from keras.preprocessing.image import ImageDataGenerator
import os


class DataLoader:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.x_dir = "data/trainX/"
        self.y_dir = "data/trainY/"
        self.x_size = len(os.listdir(self.x_dir))
        self.y_size = len(os.listdir(self.y_dir))
        self.datagen = ImageDataGenerator(
            rotation_range=180,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            rescale=1. / 255,
            channel_shift_range=40,
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode='nearest')

    def _load_image(self, path, aug):
        img = cv2.imread(path)
        if aug:
            img = next(self.datagen.flow(img, batch_size=1))[0]
            img = (img * 2.) - 1.
        else:
            img = (img[0] - 127.5) / 127.5
        return img

    def load_batch(self, aug=True):
        random_y = np.random.randint(0, self.y_size, size=self.batch_size)
        y = [self._load_image(f"{self.y_dir}{i}.png", aug) for i in random_y]

        random_x = np.random.randint(0, self.x_size, size=self.batch_size)
        x = [self._load_image(f"{self.x_dir}{i}.png", aug) for i in random_x]

        return np.array(x), np.array(y)


if __name__ == '__main__':
    dl = DataLoader(80, 4)
    a = dl.load_batch()
    print()
