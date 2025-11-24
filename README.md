# Evaluating GAN Inversion Approached for StyleGAN2-ADA
## Project Descripton
An evaluation of the GAN inversion performance of two optimisation based inversion approaches and one encoder based inversion approach. For the optimisation based approaches i evaluated StyleGAN2-ada's built in projector and [Image2StyleGAN](https://arxiv.org/abs/1904.03189). For the encoder based approach i evaluated [Encoder4Editing](https://arxiv.org/abs/2102.02766)

## Implementation of Approaches
Image2StyleGAN was built from scratch in python to be compatible with the StyleGAN2 architecture and Pytorch checkpoints. My implementation can be found under `scripts/image2stylegan`. For those looking for a Tensorflow implementation i would recommend this [repo](https://github.com/abhijitpal1247/Image2StyleGAN) by Abhijit Pal.

Encoder4Editing was cloned from omertov's (an author of e4e) [repo](https://github.com/omertov/encoder4editing). It can be found under `repos/encoder4editing`. 

StyleGAN2-ada was cloned from NVLabs [repo](https://github.com/NVlabs/stylegan2-ada-pytorch). For this experiment I used nvidia's pretrained ffhq network weights `--network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl`.

## The Dataset
I used the FFHQ dataset which can be sourced from NVLabs [repo](https://github.com/NVlabs/ffhq-dataset) or [kaggle](https://www.kaggle.com/datasets/arnaud58/flickrfaceshq-dataset-ffhq).

The dataset on Kaggle consists of 52,000 high-quality PNG images at 512×512 resolution with significant variation in terms of age, gender, ethinicity and image background. It also has good coverage of accessories like eyewear, hats and jewellery. 

For the inversion i used image `00000.jpg` from the ffhq dataset.

