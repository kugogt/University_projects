# 📡 Recovery and Information Extraction from Degraded Sources

This folder contains a collection of three distinct projects. The methodologies span across 1D biomedical signal processing and advanced 2D computer vision techniques.

**Authors**
* Cristina Papi
* Marco Rosato

---

### Module 1: Heart Rate Extraction from a Noisy PPG Signal

This project focuses on extracting accurate Heart Rate (BPM) measurements from Photoplethysmogram (PPG) signals corrupted by varying levels of synthetic noise.

**Dataset:** BIDMC PPG and Respiration dataset (PhysioNet) sampled at 125 Hz.

#### Challenge & Solution:
In real-world scenarios, PPG signals are subject to thermal and electronic noise, and heart rates are highly dynamic due to physiological factors. The main challenge was to accurately estimate the BPM under extreme noise conditions (from a clean 20 dB SNR down to a heavily degraded 0 dB SNR).

To solve this, we implemented a segmented signal processing pipeline:
* **Filtering:** We utilized a **Butterworth Band-pass filter (0.5 - 4 Hz)**. This specific filter was chosen because it offers a smooth response in the passband, suppressing noise without distorting the physiological morphology of the heartbeats.
* **Estimation Algorithms:** We compared two distinct approaches:
  * *Time-Domain (Peaks Detection):* Calculates the Inter-Beat Interval (IBI) by tracking systolic peaks. Highly sensitive to noise.
  * *Frequency-Domain (FFT):* Uses the Fast Fourier Transform to find the dominant frequency peak within the physiological range. 

**Results:** The FFT combined with the Butterworth Band-pass filter proved to be significantly more robust against White Gaussian Noise compared to time-domain peak detection, maintaining a low Mean Absolute Error (MAE) even at 0 dB SNR.

---

### Module 2: Image Dehazing via Dark Channel Prior (DCP)

This module tackles the problem of recovering a clear image from a hazy/foggy photograph by estimating the parameters of the Atmospheric Scattering Model: $I(x) = J(x)t(x) + A(1-t(x))$.

**Dataset:** Synthetic Objective Testing Set (SOTS) from RESIDE (492 outdoor image pairs).

#### Challenge & Solution:
A major issue with traditional dehazing is the creation of halo effects and visual artifacts, particularly in bright sky regions where the Dark Channel Prior assumption fails.

To solve this, we implemented this recovery pipeline:
* **Dark Channel Prior (DCP):** We extracted the dark channel using local patches to ensure robustness to noise and texture.
* **Atmospheric Light ($A$) & Transmission Map ($t(x)$):** We estimated the ambient illumination using the top 0.1% brightest pixels in the dark channel. 
* **Guided Filtering:** To prevent halo effects and artifact generation from the raw transmission map, we applied an OpenCV **Guided Filter** to smooth homogeneous regions while strictly preserving sharp edges.
* **Adaptive Constraint:** A transition luminance mask was implemented to specifically protect sky regions from over-enhancement.

**Results:** The final scene recovery, coupled with a slight Gamma and Saturation enhancement, achieved a **PSNR of 23.81** and an **SSIM of 0.9266**, outperforming the baseline hazy inputs.

---

### Module 3: Blind Image Super-Resolution & Restoration (SRRGAN)

This project focuses on an `img2img` task: reconstructing high-resolution (HR) images from low-resolution (LR) inputs that have been subjected to complex, randomized degradation pipelines.

**Dataset:** DF2K (3,450 training images) and BSDS100 for testing.

#### Challenge & Solution:
Unlike standard super-resolution tasks, "blind" restoration assumes the degradation process is unknown. The challenge is to train a model that doesn't just reduce pixel error (which often leads to blurry images) but actually hallucinates realistic, high-frequency textures.

To solve this, we implemented a Generative Adversarial Network (GAN) architecture and a multi-stage training pipeline:

**Project Workflow:**
1. **Dynamic Degradation Pipeline:** During training, images were randomly degraded on-the-fly using combinations of blur (Gaussian/Box), downsampling (x2), noise (Gaussian/Poisson), and JPEG compression.
2. **Model Architectures:**
   * *Generator:* A U-Net with a "long" bottleneck, integrating Squeeze-and-Excitation, Spatial Attention, PixelShuffle, and Global Residual connections.
   * *Discriminator:* A PatchGAN utilizing Spectral Normalization (without activation functions).
3. **Multi-Stage Training & Loss Functions:**
   * *Stage 1 (Pre-training):* Trained solely on **MAE Loss** to establish a baseline pixel-wise accuracy.
   * *Stage 2 (Perceptual Refinement):* Introduced **Perceptual Loss** using a pre-trained VGG19 network to capture high-level textural features.
   * *Stage 3 (Adversarial Fine-tuning):* Integrated **LSGAN Loss** (treating the discriminator as a regression task) to push the generator to produce photorealistic textures.

**Results:** While the baseline U-Net (MAE) achieved the best pixel-wise scores, the **SRRGAN** model achieved the best perceptual quality, scoring the lowest **LPIPS (0.3676)**, effectively restoring sharp, realistic details in heavily degraded patches (e.g., animal fur, text, and foliage).
