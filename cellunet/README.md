# CellUNet

Runs a semantic segmentation of microscopy images with a U-Net based deep
learning architecture and on-the-fly data augmentation. 

Example usage:
```
python train.py -i data/nuc0.png -l data/labels0.tif -o output -n 100 -e 5 -p 256
python predict.py -i data/nuc1.png -w output/cnn_model_weights.hdf5 -o output
```

In brief, the model accepts as training data a series of images segmented into 
regions, with each region being given 1 of 3 possible labels. In the training
data (where the model is trying to learn how to segment and identify cell nuclei), 
label 0 corresponds to the background, label 1 to the boundary, and label 2 to 
the interior of cell nuclei. 

A few functions in utils are adopted from https://github.com/carpenterlab/unet4nuclei (checking license).  

Tested with keras (2.0.0), tensorflow (1.8.0).
For GPU, use Cuda 9.0, tensorflow (1.8.0), tensorflow-gpu (1.8.0), keras (2.0.0)
Avoid Keras==2.2.

### Parameters

#### Training

- `i` - image file path
- `l` - label file path
- `o` - output directory
- `n` - number of steps (default 100)
- `b` - number of batches (default 16)
- `e` - number of epochs (default 50)
- `p` - pixel size of image patches, has to be divisible by 8 (default 256)
- `w` - hdf5 weight file path
- `q` - weights for loss function (default 1.0, 1.0, 1.0)

#### Prediction

- `i` - image file path
- `w` - hdf5 weight file path 
- `o` - output directory
- `c` - number of color channels (default 1) -- this *must* match the number of color channels in the training data

### Example Commands


#### Getting started
```
python train.py -i data/nuc0.png -l data/labels0.tif
```

#### Training with flexible weights (10x weights on Border)
```
python train.py -i data/nuc0.png -l data/labels0.tif -q 1 1 10 -o output
```

#### Resume training
```
python train.py -i data/nuc0.png -l data/labels0.tif -w data/weights.tests.hdf5
```

#### Training using multiple images
```
python train.py -i data/nuc0.png / data/nuc0.png -l data/labels0.tif / data/labels1.tif
```

#### Training using multiple color channels
Images provided may have any number of color channels; note that in the training
data nuc_0.png and nuc_1.png have only 1 color channel while composite_nuc.tif has
2 color channels.
```
python train.py -i data/composite_nuc.tif -l data/labels0.tif -o output -n 100 -e 5 -p 256 -c 2
python predict.py -i data/composite_nuc.tif -w output/cnn_model_weights.hdf5 -o output -c 2
```

#### Prediction with trained weights
```
python predict.py -i data/nuc0.png -w data/cnn_model_weights.hdf5
```
