# Generative-Adversarial-Network-GAN-for-Handwritten-Digit-Generation-using-PyTorch
A deep learning project that implements a Generative Adversarial Network (GAN) using PyTorch to generate realistic handwritten digit images from random noise. The project demonstrates adversarial learning through the interaction of a Generator and a Discriminator trained simultaneously on the MNIST dataset.

# Overview

Generative Adversarial Networks (GANs) are one of the most influential breakthroughs in generative artificial intelligence. Introduced by Ian Goodfellow et al. in 2014, GANs consist of two neural networks that compete against each other during training.

The Generator learns to create realistic images from random noise, while the Discriminator learns to distinguish between real images from the dataset and fake images produced by the Generator.

Through this adversarial process, the Generator gradually improves until it can produce highly realistic handwritten digits that closely resemble the original MNIST dataset.

This project demonstrates the complete implementation of a Vanilla GAN using PyTorch and provides a strong foundation for more advanced GAN architectures such as DCGAN, CGAN, WGAN, StyleGAN, and CycleGAN.

---

# Project Objectives

The objectives of this project were to:

- Build a Vanilla Generative Adversarial Network (GAN) using PyTorch.
- Implement both the Generator and Discriminator neural networks.
- Train the Generator to create realistic handwritten digits from random noise.
- Train the Discriminator to distinguish between real and generated images.
- Understand adversarial learning and minimax optimization.
- Generate realistic handwritten digits after training.
- Save trained Generator and Discriminator models for inference.

---

# Dataset

## MNIST Handwritten Digit Dataset

The project uses the MNIST handwritten digit dataset provided by TorchVision.

### Dataset Information

| Property | Value |
|----------|------:|
| Training Images | 60,000 |
| Test Images | 10,000 |
| Classes | 10 |
| Image Size | 28 × 28 |
| Image Channels | 1 (Grayscale) |
| Pixel Range | Normalized to [-1, 1] |
| Source | TorchVision |

The dataset is downloaded automatically during training.

---

# Data Preprocessing

The following preprocessing steps were applied:

- Converted images into PyTorch tensors.
- Normalized pixel values to the range `[-1, 1]`.
- Created shuffled training batches using DataLoader.
- Loaded images for adversarial training.

---

# Model Architecture

The GAN consists of two neural networks.

---

# Generator

The Generator receives a random latent vector and transforms it into a handwritten digit image.

```text
Random Noise (z)

↓

Fully Connected Layers

↓

Batch Normalization

↓

LeakyReLU

↓

Output Layer

↓

Generated Image (28 × 28)
```

The Generator attempts to fool the Discriminator into classifying generated images as real.

---

# Discriminator

The Discriminator receives either a real image or a generated image.

Its objective is binary classification.

```text
Input Image

↓

Fully Connected Layers

↓

LeakyReLU

↓

Sigmoid

↓

Probability

↓

Real or Fake
```

---

# Adversarial Training

The Generator and Discriminator compete during training.

```text
Random Noise

↓

Generator

↓

Generated Image

↓

Discriminator

↓

Fake Probability


Real Image

↓

Discriminator

↓

Real Probability
```

The Generator minimizes its ability to be detected.

The Discriminator maximizes its ability to correctly classify images.

This competition drives both networks to improve continuously.

---

# Loss Functions

The project uses Binary Cross Entropy (BCE) loss.

---

## Generator Loss

The Generator attempts to fool the Discriminator.

```text
Generator Loss

=

BCE(

Discriminator(Fake Image),

Real Label

)
```

---

## Discriminator Loss

The Discriminator learns from both real and fake images.

```text
Discriminator Loss

=

BCE(Real Images)

+

BCE(Fake Images)
```

---

# Technologies Used

- Python
- PyTorch
- TorchVision
- NumPy
- Matplotlib
- Pillow
- tqdm

---

# Training Configuration

| Parameter | Value |
|-----------|------:|
| Optimizer | Adam |
| Learning Rate | 0.0002 |
| Batch Size | 128 |
| Epochs | 50 |
| Latent Dimension | 100 |
| Loss Function | Binary Cross Entropy |
| Optimizer Betas | (0.5, 0.999) |
| Device | CPU / GPU (CUDA Supported) |

---

# Training Pipeline

```text
Random Noise

↓

Generator

↓

Generated Image

↓

Discriminator

↓

Generator Loss

↓

Backpropagation


Real Image

↓

Discriminator

↓

Discriminator Loss

↓

Backpropagation

↓

Update Generator

↓

Update Discriminator
```

---

# What I Implemented

The project includes the following features.

- Vanilla GAN architecture
- Generator neural network
- Discriminator neural network
- Adversarial training
- Binary Cross Entropy loss
- Adam optimizer
- Random latent vector sampling
- Model checkpoint saving
- Automatic dataset download
- GPU support
- Generated image visualization
- Training progress monitoring
- Image generation after training
- Modular project organization

---

# Model Output

The trained Generator produces handwritten digits directly from random noise.

Example workflow:

```text
Random Noise

↓

Generator

↓

Generated Handwritten Digit
```

Unlike autoencoders and VAEs, GANs do not require an input image during inference.

Generation begins entirely from randomly sampled latent vectors.

---

# Performance Summary

After training, the Generator successfully learned the underlying distribution of handwritten digits.

The trained model was able to:

- Generate realistic handwritten digits.
- Produce unique digit variations.
- Learn meaningful image structures.
- Fool the Discriminator with increasingly realistic images.
- Improve image quality throughout training.

Meanwhile, the Discriminator learned to distinguish between real and generated samples, resulting in stable adversarial learning.

---

# Adversarial Learning

GAN training is fundamentally different from supervised learning.

The Generator and Discriminator improve together.

```text
Generator

↓

Produces Fake Images

↓

Discriminator

↓

Detects Fake Images

↓

Generator Improves

↓

Better Fake Images

↓

Discriminator Improves

↓

Repeat
```

This adversarial process enables the Generator to create increasingly realistic outputs.

---

# Applications

Generative Adversarial Networks are widely used in:

- Image generation
- Face generation
- Image synthesis
- Data augmentation
- Medical image generation
- Art generation
- Image restoration
- Super-resolution
- Style transfer
- Deepfake research
- Synthetic dataset generation

---

# Relationship to Modern Generative AI

GANs transformed the field of generative modeling and inspired many subsequent architectures.

Techniques introduced through GAN research contributed to:

- DCGAN
- Conditional GAN (CGAN)
- Pix2Pix
- CycleGAN
- StyleGAN
- StyleGAN2
- ESRGAN
- BigGAN

Although diffusion models currently dominate image generation benchmarks, GANs remain one of the most important milestones in modern computer vision and generative AI.

---

# Skills Demonstrated

This project demonstrates practical experience in:

- Deep Learning
- PyTorch
- Generative Adversarial Networks
- Adversarial Training
- Computer Vision
- Image Generation
- Generator Design
- Discriminator Design
- Binary Classification
- Latent Space Sampling
- Model Training
- GPU Computing
- Model Checkpointing

---

# Repository Structure

```text
GAN-MNIST/
│
├── config.py
├── dataset.py
├── model.py
├── loss.py
├── train.py
├── generate.py
├── inference.py
├── utils.py
├── requirements.txt
├── README.md
│
├── checkpoints/
│   ├── generator_best.pth
│   ├── discriminator_best.pth
│   ├── generator_last.pth
│   └── discriminator_last.pth
│
├── samples/
│   ├── epoch_001.png
│   ├── epoch_002.png
│   └── ...
│
└── results/
    ├── loss_curve.png
    └── generated_grid.png
```

---

# Installation

Clone the repository.

```bash
git clone https://github.com/yourusername/GAN-MNIST.git
```

Install dependencies.

```bash
pip install -r requirements.txt
```

---

# Usage

Train the model.

```bash
python train.py
```

Generate handwritten digits.

```bash
python generate.py
```

Run inference using the trained Generator.

```bash
python inference.py
```

---

# Future Improvements

Potential extensions include:

- Deep Convolutional GAN (DCGAN)
- Conditional GAN (CGAN)
- Wasserstein GAN (WGAN)
- WGAN with Gradient Penalty
- Progressive GAN
- StyleGAN
- StyleGAN2
- Spectral Normalization GAN
- Self-Attention GAN
- Higher-resolution image generation
- CelebA Dataset
- CIFAR-10 Dataset

---

# Project Outcome

This project successfully implemented a Vanilla Generative Adversarial Network capable of learning the distribution of handwritten digits through adversarial training. The Generator transformed random latent vectors into realistic handwritten digit images, while the Discriminator learned to distinguish real images from generated samples.

Through continuous competition between both networks, the Generator progressively improved its ability to synthesize convincing handwritten digits. This project establishes a strong foundation for advanced adversarial models such as DCGAN, Conditional GAN, WGAN, StyleGAN, and image-to-image translation architectures, making it an essential milestone in learning modern generative AI and computer vision.
