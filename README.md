## PiiGAN: Generative Adversarial Networks for Pluralistic Image Inpainting
[Code and model will be available after this paper is published](#)
### Introduction:
We proposed PiiGAN, a novel diversity-generated image inpainting adversarial network with a newly designed style extractor for diversity image inpainting tasks. For a single input image with missing regions, our model can generate numerous diverse results with plausible content. Experiments on various datasets have shown that our results are diverse and natural, especially for images with large missing areas. Detailed description of the method can be found in our paper.
<p align='center'>  
  <img src='https://raw.githubusercontent.com/vivitsai/SEGAN/master/examples/case1.jpg' width='870'/>
</p>
Examples of the inpainting results of our method on a face, leaf, and rainforest image (the missing regions are shown in white). The left is the masked input image, while the right is the diverse and plausible direct output of our trained model without any postprocessing.

## Prerequisites
- Python 3.
- Tensorflow (tested on Tensorflow-gpu 1.10.0).
- NVIDIA GPU + CUDA cuDNN

## Installation
- Clone this repository:
```bash
git clone https://github.com/vivitsai/PiiGAN.git
```
- Install Python3 form https://www.python.org/
- Install Tensorflow from https://tensorflow.google.cn/

## Datasets
We use [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html), [PlantVillage](#) and [MauFlex](http://didt.inictel-uni.edu.pe/dataset/MauFlex_Dataset.rar,) datasets. To train a model on the full dataset, download datasets from official websites. 

After downloading, run [`flist.py`](flist.py) to generate train and validation set file lists. 
```bash
python ./flist.py
```

## Getting Started
Download the pre-trained models using the following links and copy them under `./data_models` directory.

[CelebA](#)

Alternatively, you can run the following script to automatically download the pre-trained models:
```bash
bash ./download_model.sh
```

### 1) Training
To train the model, create a `setting.yaml` file similar to the [example config file](https://github.com/vivitsai/PiiGAN/blob/master/setting.yml.example) and copy it under your root directory. 

To train the model:
```bash
python train.py
```

Convergence of the model differs from dataset to dataset. For example MauFlex dataset converges in one of 10 epochs, while larger datasets like CelebA require almost 20 epochs to converge. You can set the number of training iterations by changing MAX_ITERS value in the configuration file.

### 2) Testing

We provide some test examples under `./examples` directory. Please download the [pre-trained models](#getting-started) and run:
```bash
python test.py \
  --checkpoint_dir ./data_models/celeba 
  --image ./examples/celeba/images 
  --mask ./examples/mask.png
  --output ./examples/output.png
```
This script will inpaint all images in `./examples/celeba/images` using their corresponding masks mask.png and saves the results in output.png. 

### 3) Evaluating
To evaluate the model, you need to first run the model in [test mode](#testing) against your validation set and save the results on disk. We provide a utility [`./evaluate.py`](evaluate.py) to evaluate the model using PSNR, SSIM:

```bash
python ./evaluate.py --data-path [path to validation set] --output-path [path to model output] 
```

To measure the Fréchet Inception Distance (FID score) run [`./fid_score.py`](fid_score.py). We utilize the PyTorch implementation of FID [from here](https://github.com/mseitzer/pytorch-fid) which uses the pretrained weights from PyTorch's Inception model.


```bash
python ./fid_score.py --path [path to validation, path to model output] --gpu [GPU id to use]
```


### Load training and validation set in directory `./data_flist`


Option          | Description
----------------| -----------
train_shuffled.flist            | text file containing training set files list
validation_shuffled.flist       | text file containing validation set files list



## License
Licensed under a [Creative Commons Attribution-NonCommercial 4.0 International](https://creativecommons.org/licenses/by-nc/4.0/).

The software is for educaitonal and academic research purpose only.
