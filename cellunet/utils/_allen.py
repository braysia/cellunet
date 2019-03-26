from itertools import combinations_with_replacement
import numpy as np
from PIL import Image
from scipy.ndimage import convolve
from scipy.ndimage.filters import gaussian_filter
from skimage.exposure import equalize_adapthist
from skimage.measure import label


def filter_in_60ways(image):
    features = []
    for si in (1, 2, 4, 8):
        im = resize_lanczos(image, si)
        im = resize_upscale(im, image)
        features.append(lawtexture(im))
    features = [i for j in features for i in j]
    return np.dstack(features)


def resize_lanczos(img, scale=2):
    if scale==1:
        return img
    if not scale==1:
        I = Image.fromarray(img)
        resizedI = I.resize((int(img.shape[1]/scale), int(img.shape[0]/scale)), Image.ANTIALIAS)
        return np.array(resizedI)


def resize_upscale(img, refimg):
    if img.shape == refimg.shape:
        return img
    if not img.shape == refimg.shape:
        I = Image.fromarray(img)
        resizedI = I.resize((refimg.shape[1], refimg.shape[0]))
        return np.array(resizedI)


def lawtexture(image):
    """partially adopted from http://www.ncbi.nlm.nih.gov/pmc/articles/PMC3849478/
    """
    L5 = np.array([1, 4, 6, 4, 1])
    E5 = np.array([-1, -2, 0, 2, 1])
    W5 = np.array([-1, 2, 0, -2, 1])
    S5 = np.array([-1, 0, 2, 0, -1])
    R5 = np.array([1, -4, 6, -4, 1])
    law_gen = combinations_with_replacement((L5, E5, W5, S5, R5), r=2)
    laws = [i for i in law_gen]

    features = []
    for i in laws:
        kernel = np.outer(i[0], i[1])
        if (kernel==kernel.T).all():
            features.append(convolve(image, kernel))
        if not (kernel==kernel.T).all():  # rotationally invariant
            im1 = convolve(image, kernel)
            im2 = convolve(image, kernel.T)
            features.append((im1+im2)/2)
    gauss_features = [gaussian_filter(i, 1.5) for i in features]
    return gauss_features


def normalize_vis(img):
    img = img - img.min()
    img = img/img.max()
    return equalize_adapthist(img)


def _calc_equal_weights(features):
    ap = []
    for i in np.unique(features):
        ap.append((features == i).sum())
    frac = np.array([(np.sum(ap) - i)/np.sum(ap) for i in ap])
    prob_2d = np.zeros(features.shape)
    for i in np.unique(features):
        prob_2d[features == i] = frac[i]
    return prob_2d

  
def subsample_data(inputs, outputs, num=10000):
    weights = _calc_equal_weights(outputs)
    idx = np.random.choice(np.arange(len(outputs)), num, p=weights/weights.sum())
    return inputs[idx], outputs[idx]
