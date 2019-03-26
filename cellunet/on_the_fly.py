from utils.augmentation import deform
import numpy as np
from skimage.exposure import adjust_gamma
import SimpleITK as sitk
from tfutils import imread
import os
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


def gamma_correct(gamma=0.15, gain=0.15):
    def func(x, y):
        _gamma = np.random.uniform(1-gamma, 1+gamma)
        _gain = np.random.uniform(1-gain, 1+gain)
        x = adjust_gamma(x, _gamma, _gain)
        return x, y
    return func


def _histogram_matching(img, previmg, BINS=500, QUANT=2, THRES=False):
    simg = sitk.GetImageFromArray(img)
    spimg = sitk.GetImageFromArray(previmg)
    fil = sitk.HistogramMatchingImageFilter()
    fil.SetNumberOfHistogramLevels(BINS)
    fil.SetNumberOfMatchPoints(QUANT)
    fil.SetThresholdAtMeanIntensity(THRES)
    filimg = fil.Execute(simg, spimg)
    return sitk.GetArrayFromImage(filimg)



def histo_match(ref='ref', bins=256, quant=5):
    refimages = [imread(os.path.join(ref, i)).astype(np.float32) for i in os.listdir(ref)]
    def func(x, y):
        matchpts = np.random.randint(quant)+1
        refimg = refimages[np.random.randint(len(refimages))]
        x = _histogram_matching(x.astype(np.float32), refimg, bins, matchpts, False)
        return x, y
    return func