# gpu-dso-stacker

> A high-performance DSO (Deep Sky Object) stacker with hardware acceleration

---

## 🛠️ Technology Stack

- **C** - Core implementation
- **CUDA** - GPU acceleration
- **Python** - High-level interface

---

## 🔄 Pipeline

0. **Input**
   - A csv file is going to be provided as program argument.
   - First column is going to be file_path. The file paths of input fits files are going to be there.
   - Second column is goingto be is_reference. Only one row should contain 1 for this column, and all others are going to be 0

1. **Preprocessing(GPU)**  
   Initial image preparation and calibration
   - Debayering(in case of color images) - VNG debayering

2. **Star Detection & Center-of-Mass Calculation**  
   Identify stars and fit point spread functions
   1. Convolution with the moffat kernel to detect star pixels(GPU)
   2. leave only pixels over threshold(ex. >3*sigma) (GPU)
   3. Find weighted center of mass (CPU)
   4. Get list of star center coordinates (CPU)

3. **Transform Computation(CPU)**  
   Calculate alignment using RANSAC
   - Calculate the homography matrix

4. **Image Transformation(GPU)**  
   Apply Lanczos interpolation to align with reference frame
   - Code already implemented

5. **Integration(GPU)**  
   Combine images using kappa-sigma clipping
   - We are going to use kappa-sigma clipping on mini-batches, not the whole dataset. Divide the images into batches of M images, keep M images(after transformation) on GPU, and do kappa-sigma clipping on the M images, and integrate them all into the final image.

---

## 📋 Roadmap

### Week 1

**Seungwon**
- Create preprocessing, star detection, transform computation skeleton

**Seungmin**
- Lanczos interpolation & integration kernel implementation

### Week 2

**Seungwon**
- Homography matrix calculation

**Seungmin**
- Interpolation & integration test
