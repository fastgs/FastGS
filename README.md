<div align="center">
<h1>FastGS: Training 3D Gaussian Splatting in 100 Seconds</h1> 

[🌐 Homepage](https://fastgs.github.io/) | [📄 Paper](https://arxiv.org/abs/2511.04283)

</div>

<p align="center">
    <img src="assets/teaser_fastgs.jpg" width="800px"/>
</p>

## 🚀 What Makes FastGS Special?

FastGS is a **general acceleration framework** that supercharges 3D Gaussian Splatting training while maintaining Comparable rendering quality. Our method stands out with:

- **⚡ Blazing Fast Training**: Achieve SOTA results within **100 seconds**. **3.32× faster** than DashGaussian on Mip-NeRF 360 dataset. **15.45× acceleration** vs vanilla 3DGS on Deep Blending.
- **⚡ High fidelity**: Comparable rendering quality with SOTA methods
- **🎯 Easy Integration**: Seamlessly integrates with various backbones (Vanilla 3DGS, Scaffold-GS, Mip-splatting, etc.)
- **🛠️ Multi-Task Ready**: Proven effective across dynamic scenes, surface reconstruction, sparse-view, large-scale, and SLAM tasks
- **💡 Memory-Efficient**: Low GPU Memory requirements make it accessible for various hardware setups
- **🔧 Easy Deployment**: Simple post-training tool for feedforward 3DGS that works out-of-the-box

## 📢 Latest Updates
### 🎯 Coming Soon
- **[2025.11.30]** 🔥 **Code Release**: Clean implementation of FastGS core framework  - stay tuned! 🔭
- **[2025.12.31]** 🎯 **Multi-Task Expansion**:
  - Dynamic scenes Reconstruction: [Deformable-3D-Gaussians](https://github.com/ingra14m/Deformable-3D-Gaussians)
  - Autonomus Driving scene: [street_gaussians](https://github.com/zju3dv/street_gaussians)
  - Surface reconstruction: [PGSR](https://github.com/zju3dv/PGSR)
  - Sparse-view Reconstruction: [DropGaussian](https://github.com/DCVL-3D/DropGaussian_release)
  - Large-scale Reconstruction: [OctreeGS](https://github.com/city-super/Octree-GS/tree/main)
  - SLAM: [Photo-SLAM](https://github.com/HuajianUP/Photo-SLAM)
- **[2025.12.31]** 🔌 **Backbone Enhancing**: popular 3DGS variants ([Vanilla 3DGS](https://github.com/graphdeco-inria/gaussian-splatting), [Scaffold-GS](https://github.com/city-super/Scaffold-GS), [Mip-splatting](https://github.com/autonomousvision/mip-splatting))


## 🎯 Quick Facts

| Feature | FastGS | Previous Methods |
|---------|---------|---------------------|
| Training Time | **100 seconds** | 5-30 minutes |
| Gaussian Efficiency | ✅ **Strict Control** | ❌ Redundant Growth |
| Memory Usage | ✅ **Low Footprint** | ❌ High Demand |
| Task Versatility | ✅ **6 Domains** | ❌ Limited Scope |

## Setup
```bash
conda env create --file environment.yml
conda activate FastGS
```

## Usage
Clone the repository
```bash
git clone https://github.com/fastgs/FastGS.git --recursive
```

### Dataset Directory Structure
Please organize the dataset in the following structure:
```bash
datasets/
├── mipnerf360/
│ ├── bicycle/
│ ├── flowers/ 
│ ├── ... 
├── db/
│ ├── playroom/
│ ├── ... 
├── tanksandtemples/
│ ├── truck/
│ ├── ... 
```

## Training
```bash
bash train_base.sh
```
or
```bash
bash train_big.sh
```


## Citation
If you find this repo useful, please cite:
```
@article{ren2025fastgs,
  title={FastGS: Training 3D Gaussian Splatting in 100 Seconds},
  author={Ren, Shiwei and Wen, Tianci and Fang, Yongchun and Lu, Biao},
  journal={arXiv preprint arXiv:2511.04283},
  year={2025}
}

```

---

<div align="center">

**⭐ Star this repo to get notified when we release the code!**

*FastGS: Training 3D Gaussian Splatting in 100 Seconds*

</div>

---

*Note: This is a preview README. Full documentation and code examples will be available upon release.*
