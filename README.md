# Sketch-to-Image Generation with conditional Generative Adversarial Networks

## Overview

This repository explores conditional Generative Adversarial Networks (cGANs) for translating hand-drawn sketches into photorealistic color photographs. Generating realistic images from human sketches is difficult because sketches are sparse, imprecise, and oversimplified. Models must retain artist intent while deviating to allow for accurate texture, color, and depth realism.

The project investigates the primary research question: **Can cGANs preserve and use conditioning information from sparse sketch inputs to synthesize realistic images?**

To answer this, we implement and compare two approaches:
1. A baseline, classic pix2pix cGAN implemented using TensorFlow and Keras
2. An improved model, inspired by SketchyGAN, implemented in PyTorch and incorporating the following improvements:
   - Masked Residual Unit (MRU) blocks
   - Training data augmentation
   - Holistically-nested edge detection (HED) maps
   - Composite loss function

### Features & Improvements

- **pix2pix baseline**: Paired image-to-image translation utilizing a standard U-Net generator with skip connections and a PatchGAN discriminator.
- **MRU blocks**: Inject a downsampled version of the original sketch at each down/upsampling layer to preserve structural information.
- **Multi-factor Loss Function**: Combines adversarial loss, $L_1$ supervised loss, perceptual loss (VGG16 feature matching), diversity loss, auxiliary classification loss.
- **Edge Map Augmentation**: Integrates HED edge maps generated from target photos to augment sketch inputs during training.

## Loss Function

The improved SketchyGAN model is trained using the competing composite loss functions:
- Generator: $\mathcal{L}(G)=\mathcal{L}\_{GAN}(G)-\mathcal{L}\_{ac}(G)+\mathcal{L}\_{sup}(G)+\mathcal{L}\_{p}(G)+\mathcal{L}\_{div}(G)$
- Discriminator: $\mathcal{L}(D)=\mathcal{L}\_{GAN}(D,G)+\mathcal{L}\_{AC}(D)$

### Objective Function Descriptions

*Adversarial Loss*:

$$\mathcal{L}\_{GAN}(D,G)=\mathbb{E}\_{y\sim P\_{image}}[\log D(Y)]+\mathbb{E}\_{y\sim P\_{sketch},z\sim P\_{z}}[\log (1-D(G(x,z)))]$$

- Discriminator wants to maximize the entire function
- Generator wants to minimize the second expected value term, which is $\mathcal{L}\_{GAN}(G)$

*Auxiliary classification loss*:

$$\mathcal{L}\_{ac}(D)=\mathbb{E}[\log P(C=c|y)]$$

- Discriminator wants to correctly predict image class label $c$
- Generator wants discriminator to predict its images as class label $c$

*L1 supervised loss*:

$$\mathcal{L}\_{sup}(G)=\vert G(x,z)-y\vert\_1$$

- Difference between generated output and target image
- Generator wants to minimize

*Perceptual loss*:

$$\mathcal{L}\_{p}(G)=\sum\_{i}\lambda\_{p}\vert\phi\_{i}(G(x,z))-\phi\_{i}(y)\vert\_{1}$$

- Since $L\_{1}$ loss tends to create blurry images, perceptual loss uses Inception-V4 image classification network, looking at feature maps instead of raw pixel similarity
- Minimizes distance between generated and target feature maps

*Diversity loss*:

$$\mathcal{L}\_{div}(G)=\lambda\_{div}\vert G(x,z\_{1}-G(x,z\_{2})\vert\_{1}$$

- Maximizes distance between generated output over two different random noise vectors
- Encourages diverse output instead of learning to copy target image

## Model Outputs

### pix2pix baseline:

<img width="587" height="191" alt="image" src="https://github.com/user-attachments/assets/7e408022-4a4b-4706-82aa-ec4fb8f5c8f1" />
<img width="587" height="191" alt="image" src="https://github.com/user-attachments/assets/a176ccbd-618c-4afe-9886-c01c842daebf" />
<img width="587" height="191" alt="image" src="https://github.com/user-attachments/assets/ac7fa8df-2531-45a3-9c90-d6dbb8d1499e" />
<img width="587" height="191" alt="image" src="https://github.com/user-attachments/assets/9a331d10-329b-49b3-91ab-49dc0dc6cafa" />
<img width="587" height="191" alt="image" src="https://github.com/user-attachments/assets/bda37dbb-09c6-48d2-aae4-70dedef5c0a7" />

### Improved SketchyGAN model:

<img width="558" height="766" alt="image" src="https://github.com/user-attachments/assets/c08ba0cf-0efe-4f67-b2dc-5e9c784f2971" />

## Experimental Results

| Model | Inception Score (IS) &uarr; | Fréchet Inception Distance (FID) &darr; |
| --- | --- | --- |
| pix2pix baseline | 1.0 | >500 |
| Improved SketchyGAN model | 1.3 | 433 |

## Key Findings

1. **Geometric & Structural Preservation**: The inclusion of MRU blocks allowed the network to learn low-level geometric mappings. Coarse structural elements were consistently observed (general balloon shape, vertical orientation).
2. **Lack of fine detail**: High-frequency details and textures were lost (sharp edges, object-specific features). Outputs were generally blurry and indistinct.
3. **Color inconsistency & artifacts**: Poorly localized and diffused color distributions.
4. **Mode Collapse**: Similar structural patterns observed, indicating low generator diversity. A method of dynamic loss function weight adjustments could be implemented to increase the presence of the diversity loss term to combat this.

## Conclusion

The improved model performed slightly better than the pix2pix baseline model. Outputs did not suffer from extreme overfitting as the baseline model did, rather producing a more general, similar set of images. This suggests the improved model was learning fundamental characteristics of hot air balloons instead of specific image features.

The lack of training data (~500 images) and compute limitations (Google Colab unpaid tier) severely hindered the model's ability to sufficiently learn in order to generate accurate images. Given enough training epochs, these models would display better results.
