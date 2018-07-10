from utils.augmentation import deform
import numpy as np


def affine_transform(points=20, distort=3):
    def func(x, y):
        return deform(x, y, points, distort)
    return func


def random_flip():
    def func(x, y):
        rand_flip = np.random.randint(low=0, high=2)
        if(rand_flip == 0):
            x = np.flip(x, 0)
            y = np.flip(y, 0)
        return x, y
    return func


def random_rotate():
    def func(x, y):
        rand_rotate = np.random.randint(low=0, high=4)
        for _ in range(rand_rotate):
            x = np.rot90(x)
            y = np.rot90(y)
        return x, y
    return func


def random_illum(perc=0.25):
    def func(x, y):
        ifactor = 1 + np.random.uniform(perc-1, 1-perc)
        x *= ifactor
        return x, y
    return func
