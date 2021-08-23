import numpy as np
import glob
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os


class DataLoader:
    def __init__(self, size, batch_size, dataset='apples2oranges'):
        self.batch_size = batch_size
        self.size = (size, size)
        if dataset == 'apples2oranges':
            self.ROOT = '/media/bonilla/HDD_2TB_basura/databases/Apples2Oranges/Apples2Oranges'
            self.X = glob.glob(os.path.join(self.ROOT, 'trainA', '*.jpg'))
            self.Y = glob.glob(os.path.join(self.ROOT, 'trainB', '*.jpg'))
        if dataset == 'dogs2cats':
            self.ROOT = '/media/bonilla/My Book/DogsCats/data/train'
            self.X = glob.glob(os.path.join(self.ROOT, 'dog*'))
            self.Y = glob.glob(os.path.join(self.ROOT, 'cat*'))
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
        self.n = min(len(self.X), len(self.Y)) * 10

    def _load_image(self, path, aug):
        img = cv2.imread(path)
        img = cv2.resize(img, self.size)[np.newaxis, :, :, ::-1].astype('float32')
        if aug:
            img = next(self.datagen.flow(img, batch_size=1))[0]
            img = (img * 2.) - 1.
        else:
            img = (img[0] - 127.5) / 127.5
        return img

    def load_batch(self, aug=True):
        random_y = np.random.choice(self.Y, size=self.batch_size)
        y = [self._load_image(i, aug) for i in random_y]

        random_x = np.random.choice(self.X, size=self.batch_size)
        x = [self._load_image(i, aug) for i in random_x]

        return np.array(x), np.array(y)


if __name__ == '__main__':
    dl = DataLoader(80, 4)
    a = dl.load_batch()
    print()
