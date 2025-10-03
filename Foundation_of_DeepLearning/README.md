# 🧠 Foundations of Deep Learning – Super-Resolution and Restoration of images (Image-to-Image task) with U-Net

This project was developed as part of the *Foundations of Deep Learning* course, and focuses on **super-resolution and restoration of images**. 
The goal is to train a deep learning model that can transform low-resolution, degraded images into high-resolution, clean counterparts.

---

💡 If you’re interested in an **upgrade** of this project, you can find it here:  
[📄 `Link`](https://github.com/kugogt/SRRGAN-U-Net-PatchGan)

---

### Dataset & Pre-Process
The model was trained on the **IAPR TC-12** dataset.
The core of the project involves training a model on pairs of images: a high-resolution (HR) original and a synthetically degraded low-resolution (LR) version. The model learns to reverse the degradation process, effectively "upscaling" and "cleaning" the input.

The degradation process included:
- **Downsampling (2x)**
- **Gaussian Blur**
- **Gaussian Noise**

### Data Pipeline
A `tf.data` pipeline was implemented to load and processes, "on-the-fly", the images needed for each training batch directly into memory, allowing for scalable and memory-efficient training.

### Models
Two variants of the **U-Net** architecture were developed:
- `Base U-Net`: built strictly following the structure and techniques studied during the course, serving as a solid reference implementation. It uses MaxPooling for downsampling and UpSampling2D for upsampling.
- `Advanced U-Net`: extended with architectural improvements and design choices that go beyond what was implemented in class:
  - LeakyRelu; 
  - Strided Convolution;
  - Conv2DTranspose;
  - BatchNormalization; 
  - Residual Skip Connections.

### Loss Functions
The models were trained using:
- **MAE loss**: A standard pixel-wise Mean Absolute Error loss.
- **composite loss**:   composite loss function defined as MAE + α * Perceptual Loss. The perceptual component was calculated using features extracted from a pre-trained VGG19 network to better capture high-level textural and structural similarities.

### Experimentation
Several training runs were conducted to compare performance across different configurations.

<img width="1989" height="331" alt="__results___44_0" src="https://github.com/user-attachments/assets/6a76d1ce-a2f6-4c04-b631-c554925b2270" />
