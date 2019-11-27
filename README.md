## Diversity-Generated Image Inpainting with Style Extraction
[Preprints](#)
### Introduction:
We proposed SEGAN, a novel diversity-generated image inpainting adversarial network with a newly designed style extractor for diversity image inpainting tasks. For a single input image with missing regions, our model can generate numerous diverse results with plausible content. Experiments on various datasets have shown that our results are diverse and natural, especially for images with large missing areas. After the paper is published, a detailed description of the method can be found in our [paper].
<p align='center'>  
  <img src='https://user-images.githubusercontent.com/1743048/50673917-aac15080-0faf-11e9-9100-ef10864087c8.png' width='870'/>
</p>
Examples of the inpainting results of our method on a face, leaf, and rainforest image (the missing regions are shown in white). The left is the masked input image, while the right is the diverse and plausible direct output of our trained model without any postprocessing.

## Prerequisites
- Python 3
- Tensorflow1.10
- NVIDIA GPU + CUDA cuDNN

## Installation
- Clone this repo:
```bash
git clone https://github.com/vivitsai/SEGAN.git
```
- Install Tensorflow from https://tensorflow.google.cn/

## Datasets
We use [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html), [Agricultural Disease] and [MauFlex](http://didt.inictel-uni.edu.pe/dataset/MauFlex_Dataset.rar,) datasets. To train a model on the full dataset, download datasets from official websites. 

After downloading, run [`flist.py`](flist.py) to generate train, test and validation set file lists. 

## Getting Started
Download the pre-trained models using the following links and copy them under `./checkpoints` directory.

[CelebA](#) (Released after publication)

Alternatively, you can run the following script to automatically download the pre-trained models:
```bash
bash ./download_model.sh
```

### 1) Training
To train the model, create a `config.yaml` file similar to the [example config file](https://github.com/knazeri/edge-connect/blob/master/config.yml.example) and copy it under your checkpoints directory. Read the [configuration](#model-configuration) guide for more information on model configuration.

EdgeConnect is trained in three stages: 1) training the edge model, 2) training the inpaint model and 3) training the joint model. To train the model:
```bash
python train.py --model [stage] --checkpoints [path to checkpoints]
```

For example to train the edge model on Places2 dataset under `./checkpoints/places2` directory:
```bash
python train.py --model 1 --checkpoints ./checkpoints/places2
```

Convergence of the model differs from dataset to dataset. For example Places2 dataset converges in one of two epochs, while smaller datasets like CelebA require almost 40 epochs to converge. You can set the number of training iterations by changing `MAX_ITERS` value in the configuration file.

### 2) Testing
To test the model, create a `config.yaml` file similar to the [example config file](setting.yml.example) and copy it under your checkpoints directory. Read the [configuration](#model-configuration) guide for more information on model configuration.

You can test the model on all three stages: 1) edge model, 2) inpaint model and 3) joint model. In each case, you need to provide an input image (image with a mask) and a grayscale mask file. Please make sure that the mask file covers the entire mask region in the input image. To test the model:
```bash
python test.py \
  --model [stage] \
  --checkpoints [path to checkpoints] \
  --input [path to input directory or file] \
  --mask [path to masks directory or mask file] \
  --output [path to the output directory]
```

We provide some test examples under `./examples` directory. Please download the [pre-trained models](#getting-started) and run:
```bash
python test.py \
  --checkpoints ./checkpoints/places2 
  --input ./examples/places2/images 
  --mask ./examples/places2/masks
  --output ./checkpoints/results
```
This script will inpaint all images in `./examples/places2/images` using their corresponding masks in `./examples/places2/mask` directory and saves the results in `./checkpoints/results` directory. By default `test.py` script is run on stage 3 (`--model=3`).

### 3) Evaluating
To evaluate the model, you need to first run the model in [test mode](#testing) against your validation set and save the results on disk. We provide a utility [`./scripts/metrics.py`](scripts/metrics.py) to evaluate the model using PSNR, SSIM and Mean Absolute Error:

```bash
python ./metrics.py --data-path [path to validation set] --output-path [path to model output]
```

To measure the Fréchet Inception Distance (FID score) run [`./scripts/fid_score.py`](scripts/fid_score.py). We utilize the PyTorch implementation of FID [from here](https://github.com/mseitzer/pytorch-fid) which uses the pretrained weights from PyTorch's Inception model.

```bash
python ./fid_score.py --path [path to validation, path to model output] --gpu [GPU id to use]
```


### Model Configuration

The model configuration is stored in a [`config.yaml`](setting.yml.example) file under your checkpoints directory. The following tables provide the documentation for all the options available in the configuration file:


#### Loading Train, Test and Validation Sets Configurations

Option          | Description
----------------| -----------
TRAIN_FLIST     | text file containing training set files list
VAL_FLIST       | text file containing validation set files list
TEST_FLIST      | text file containing test set files list


#### Training Mode Configurations

Option                 |Default| Description
-----------------------|-------|------------
LR                     | 0.0001| learning rate
D2G_LR                 | 0.1   | discriminator/generator learning rate ratio
BETA1                  | 0.0   | adam optimizer beta1
BETA2                  | 0.9   | adam optimizer beta2
BATCH_SIZE             | 8     | input batch size 
INPUT_SIZE             | 256   | input image size for training. (0 for original size)
SIGMA                  | 2     | standard deviation of the Gaussian filter used in Canny edge detector </br>(0: random, -1: no edge)
MAX_ITERS              | 2e6   | maximum number of iterations to train the model
EDGE_THRESHOLD         | 0.5   | edge detection threshold (0-1)
L1_LOSS_WEIGHT         | 1     | l1 loss weight
FM_LOSS_WEIGHT         | 10    | feature-matching loss weight
STYLE_LOSS_WEIGHT      | 1     | style loss weight
CONTENT_LOSS_WEIGHT    | 1     | perceptual loss weight
INPAINT_ADV_LOSS_WEIGHT| 0.01  | adversarial loss weight
GAN_LOSS               | nsgan | **nsgan**: non-saturating gan, **lsgan**: least squares GAN, **hinge**: hinge loss GAN
GAN_POOL_SIZE          | 0     | fake images pool size
SAVE_INTERVAL          | 1000  | how many iterations to wait before saving model (0: never)
EVAL_INTERVAL          | 0     | how many iterations to wait before evaluating the model (0: never)
LOG_INTERVAL           | 10    | how many iterations to wait before logging training loss (0: never)
SAMPLE_INTERVAL        | 1000  | how many iterations to wait before saving sample (0: never)
SAMPLE_SIZE            | 12    | number of images to sample on each samling interval

## License
Licensed under a [Creative Commons Attribution-NonCommercial 4.0 International](https://creativecommons.org/licenses/by-nc/4.0/).

Except where otherwise noted, this content is published under a [CC BY-NC](https://creativecommons.org/licenses/by-nc/4.0/) license, which means that you can copy, remix, transform and build upon the content as long as you do not use the material for commercial purposes and give appropriate credit and provide a link to the license.
