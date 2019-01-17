"""
keras==2.0.0, 2.2 not working
"""

from __future__ import division, print_function
import os
import numpy as np

import utils.model_builder
from utils.objectives import weighted_crossentropy
import utils.metrics
import keras.backend
from keras import callbacks
import keras.layers
import keras.models
import keras.optimizers

from tfutils import imread, pad_image
from patches import extract_patches, pick_coords, pick_coords_list, extract_patch_list, _extract_patches, PatchDataGeneratorList
from tfutils import make_outputdir, normalize, conv_labels2dto3d
from os.path import join
from tfutils import parse_image_files
from on_the_fly import affine_transform, random_flip, random_rotate, random_illum


FRAC_TEST = 0.1
augment_pipe = [affine_transform(points=10, distort=2), random_flip(),
                random_rotate(), random_illum(perc=0.25)]


def define_callbacks(output):
    csv_logger = callbacks.CSVLogger(join(output, 'training.log'))
    # earlystop = callbacks.EarlyStopping(monitor='loss', patience=2)
    fpath = join(output, 'weights.{epoch:02d}-{loss:.2f}-{categorical_accuracy:.2f}.hdf5')
    cp_cb = callbacks.ModelCheckpoint(filepath=fpath, monitor='loss', save_best_only=True)
    return [csv_logger, cp_cb]


def train(image_list, labels_list, output, patchsize=256, nsteps=100,
          batch_size=16, nepochs=10, weights=None, loss_weights=[1.0, 1.0, 10.0]):

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
    num_colors = li_image[0].shape[-1]

    num_tests = int(nsteps * batch_size * FRAC_TEST)
    ecoords = pick_coords_list(nsteps * batch_size, li_labels, patchsize, patchsize)
    li_labels = [conv_labels2dto3d(lb) for lb in li_labels]
    ecoords_tests, ecoords_train = ecoords[:num_tests], ecoords[num_tests:]
    x_tests, y_tests = extract_patch_list(li_image, li_labels, ecoords_tests, patchsize, patchsize)

    model = utils.model_builder.get_model(patchsize, patchsize, num_colors, activation=None)
    loss = lambda y_true, y_pred: weighted_crossentropy(y_true, y_pred, loss_weights)
    metrics = [keras.metrics.categorical_accuracy,
               utils.metrics.channel_recall(channel=0, name="background_recall"),
               utils.metrics.channel_precision(channel=0, name="background_precision"),
               utils.metrics.channel_recall(channel=1, name="interior_recall"),
               utils.metrics.channel_precision(channel=1, name="interior_precision"),
               utils.metrics.channel_recall(channel=2, name="boundary_recall"),
               utils.metrics.channel_precision(channel=2, name="boundary_precision"),
               ]
    if weights is not None:
        model.load_weights(weights)
    optimizer = keras.optimizers.RMSprop(lr=1e-4)
    model.compile(loss=loss, metrics=metrics, optimizer=optimizer)
    # model.summary()

    make_outputdir(output)
    callbacks = define_callbacks(output)

    datagen = PatchDataGeneratorList(augment_pipe)

    history = model.fit_generator(
        generator=datagen.flow(li_image, li_labels, ecoords_train, patchsize, patchsize, batch_size=batch_size, shuffle=True),
        steps_per_epoch=nsteps,
        epochs=nepochs,
        validation_data=(x_tests, y_tests),
        # validation_steps=len(ecoords_train)/batch_size,
        validation_steps=len(ecoords_train),
        callbacks=callbacks,
        verbose=1
    )

    # score = model.evaluate(x_tests, y_tests, batch_size=batch_size)
    rec = dict(zip(model.metrics_names, [history.history[i] for i in model.metrics_names]))
    np.savez(join(output, 'records.npz'), **rec)

    json_string = model.to_json()
    open(join(output, 'cnn_model.json'), 'w').write(json_string)
    model.save_weights(join(output, 'cnn_model_weights.hdf5'))
    yaml_string = model.to_yaml()
    open(join(output, 'cnn_model.yaml'), 'w').write(yaml_string)


def _parse_command_line_args():
    """
    image:  Path to a tif or png file (e.g. data/nuc0.png).
            To pass multiple image files (size can be varied), use syntax like
            "-i im0.tif / im1.tif / im2.tif", and pass the same number of labels.
    labels: (e.g. data/labels0.tif)
    nsteps: A number of steps per epoch. A total patches passed to a model will be
            batch * nsteps.
    batch:  Typically 10-32? This will affect a memory usage.
    """

    import argparse
    parser = argparse.ArgumentParser(description='predict')
    parser.add_argument('-i', '--image', help='image file path', nargs="*")
    parser.add_argument('-l', '--labels', help='labels file path', nargs="*")
    parser.add_argument('-o', '--output', default='.', help='output directory')
    parser.add_argument('-n', '--nsteps', type=int, default=100, help='number of steps')
    parser.add_argument('-b', '--batch', type=int, default=16)
    parser.add_argument('-e', '--epoch', type=int, default=50)
    parser.add_argument('-p', '--patch', type=int, default=256,
                        help='pixel size of image patches. make it divisible by 8')
    parser.add_argument('-w', '--weights', help='hdf5 weight file path')
    parser.add_argument('-q', '--loss', type=float, nargs='+', action='append',
                        help='Background, Interior, Border weight for loss function')
    return parser.parse_args()


def _main():
    args = _parse_command_line_args()
    if args.loss is None:
        args.loss = [[1., 1., 1.]]
    images = parse_image_files(args.image)[0]
    labels = parse_image_files(args.labels)[0]
    train(images, labels, args.output, args.patch,
          args.nsteps, args.batch, args.epoch, args.weights,
          args.loss[0])


if __name__ == "__main__":
    _main()
