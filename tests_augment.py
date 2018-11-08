import numpy as np
from tfutils import parse_image_files
from tfutils import normalize, conv_labels2dto3d, make_outputdir
from patches import extract_patches, pick_coords, pick_coords_list, extract_patch_list, _extract_patches, PatchDataGeneratorList
from tfutils import imread
from on_the_fly import affine_transform, random_flip, random_rotate, random_illum
import tifffile as tiff
from os.path import join


def train(image_list, labels_list, output, patchsize=256, nsamples=10):
    li_image, li_labels = [], []
    for image_path, labels_path in zip(image_list, labels_list):
        image, labels = imread(image_path), imread(labels_path).astype(np.uint8)
        image = normalize(image)
        if image.ndim == 2:
            image = np.expand_dims(image, -1)
        elif image.ndim == 3:
            image = np.moveaxis(image, 0, -1)
        li_image.append(image)
        li_labels.append(labels)

    ecoords = pick_coords_list(nsamples, li_labels, patchsize, patchsize)
    li_labels = [conv_labels2dto3d(lb) for lb in li_labels]
    x_tests, y_tests = extract_patch_list(li_image, li_labels, ecoords, patchsize, patchsize)

    from train import augment_pipe
    make_outputdir(output)

    for i in range(nsamples):
        x, y = x_tests[i], y_tests[i]
        for aug in augment_pipe:
            x, y = aug(x, y)
        tiff.imsave(join(output, 'x{0:04}.tif'.format(i)), x.astype(np.float32))
        tiff.imsave(join(output, 'y{0:04}.tif'.format(i)), y.astype(np.uint16))




def _parse_command_line_args():
    """
    image:  Path to a tif or png file (e.g. data/nuc0.png).
            To pass multiple image files (size can be varied), use syntax like
            "-i im0.tif / im1.tif / im2.tif", and pass the same number of labels.
    labels: (e.g. data/labels0.tif)
    n:      A number of pixels for training. Use a large number (like 1,000,000)
    batch:  Typically 128-512?
    """

    import argparse
    parser = argparse.ArgumentParser(description='predict')
    parser.add_argument('-i', '--image', help='image file path', nargs="*")
    parser.add_argument('-l', '--labels', help='labels file path', nargs="*")
    parser.add_argument('-o', '--output', default='.', help='output directory')
    parser.add_argument('-n', '--nsamples', type=int, default=10, help='number of samples')
    parser.add_argument('-p', '--patch', type=int, default=256,
                        help='pixel size of image patches. make it odd')
    return parser.parse_args()


def _main():
    args = _parse_command_line_args()
    images = parse_image_files(args.image)[0]
    labels = parse_image_files(args.labels)[0]
    train(images, labels, args.output, args.patch, args.nsamples)

if __name__ == "__main__":
    _main()
