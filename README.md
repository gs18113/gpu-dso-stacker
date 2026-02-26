# gpu-dso-stacker

> A high-performance DSO (Deep Sky Object) stacker with hardware acceleration

---

## 🛠️ Technology Stack

- **C** - Core implementation
- **CUDA** - GPU acceleration
- **Python** - High-level interface

---

## 🔄 Pipeline

1. **Preprocessing**  
   Initial image preparation and calibration

2. **Star Detection & PSF Fitting**  
   Identify stars and fit point spread functions

3. **Transform Computation**  
   Calculate alignment using RANSAC

4. **Image Transformation**  
   Apply Lanczos interpolation to align with reference frame

5. **Integration**  
   Combine images using kappa-sigma clipping

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
