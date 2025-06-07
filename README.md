# <p align=center>`MRI super-resolution reconstruction using efficient diffusion probabilistic model with residual shifting`</p> # 


[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT) [![Open Access](https://img.shields.io/badge/PMB%20Paper-Open%20Access%20%7C%20Click%20to%20Read-8a2be2)](https://doi.org/10.1088/1361-6560/ade049)



:fire::fire:**Res-SRDiff** is a deep learning framework designed to robustly restore high-resolution pelvic T2w MRI and ultra-high field brain T1 maps using an efficient probabilistic diffusion model.

- Our paper on arXiv: [MRI super-resolution reconstruction using efficient diffusion probabilistic model with residual shifting](https://arxiv.org/abs/2503.01576) :heart:


## 🔍 Diffusion Process

The following diagram illustrates the diffusion process used in this project:


<p align="center"> <img src="./figures/diffusion_processes_v3.jpg" alt="Hyper-parameters" width="1000"/> </p>



## Getting Started

### Prerequisites

- Python (>=3.12)
- PyTorch (>=2.5)
- NVIDIA CUDA (for GPU acceleration)
- Additional dependencies as listed in `requirements.txt`

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/mosaf/Res-SRDiff.git
   cd Res-SRDiff

2. **Install dependencies:**

    ```bash
    conda env update --file environment.yml --prune

## Running the Code

To run the project, modify the parameters in the `main.py` file and execute the `main.py` script:

```bash
python main.py
```

## ⚙️ Model Hyper-parameters


The diagram below visualizes the key hyper-parameters used in this model:
<p align="center"> <img src="./figures/hyperparameters_v2.svg" alt="Hyper-parameters" width="1000"/> </p>


## 📚 Citation

[//]: # (      author={Mojtaba Safari and Shansong Wang and Zach Eidex and Richard Qiu and Chih-Wei Chang and David S. Yu and Xiaofeng Yang},)

If you find **Res-SRDiff** useful for your research or project, please consider citing our work:

```

@article{10.1088/1361-6560/ade049,
	author={Safari, Mojtaba and Wang, Shansong and Eidex, Zach and Li, Qiang and Qiu, Richard L J and Middlebrooks, Erik H and Yu, David S and Yang, Xiaofeng},
	title={MRI super-resolution reconstruction using efficient diffusion probabilistic model with residual shifting},
	journal={Physics in Medicine & Biology},
	url={http://iopscience.iop.org/article/10.1088/1361-6560/ade049},
	year={2025},
	abstract={Objective: Magnetic resonance imaging (MRI) is essential in clinical and research contexts, providing exceptional soft-tissue contrast. However, prolonged acquisition times often lead to patient discomfort and motion artifacts. Diffusion-based deep learning super-resolution (SR) techniques reconstruct high-resolution (HR) images from low-resolution (LR) pairs, but they involve extensive sampling steps, limiting real-time application. To overcome these issues, this study introduces a residual error-shifting mechanism markedly reducing sampling steps while maintaining vital anatomical details, thereby accelerating MRI reconstruction. Approach: We developed Res-SRDiff, a novel diffusion-based SR framework incorporating residual error shifting into the forward diffusion process. This integration aligns the degraded HR and LR distributions, enabling efficient HR image reconstruction. We evaluated Res-SRDiff using ultra-high-field brain T1 MP2RAGE maps and T2-weighted prostate images, benchmarking it against Bicubic, Pix2pix, CycleGAN, SPSR, I2SB, and TM-DDPM methods. Quantitative assessments employed peak signal-to-noise ratio (PSNR), structural similarity index (SSIM), gradient magnitude similarity deviation (GMSD), and learned perceptual image patch similarity (LPIPS). Additionally, we qualitatively and quantitatively assessed the proposed framework’s individual components through an ablation study and conducted a Likert-based image quality evaluation. Main results: Res-SRDiff significantly surpassed most comparison methods regarding PSNR, SSIM, and GMSD for both datasets, with statistically significant improvements (p-values≪0.05). The model achieved high-fidelity image reconstruction using only four sampling steps, drastically reducing computation time to under one second per slice. In contrast, traditional methods like TM-DDPM and I2SB required approximately 20 and 38 seconds per slice, respectively. Qualitative analysis showed Res-SRDiff effectively preserved fine anatomical details and lesion morphologies. The Likert study indicated that our method received the highest scores, 4.14±0.77(brain) and 4.80±0.40(prostate). Significance:Res-SRDiff demonstrates efficiency and accuracy, markedly improving computational speed and image quality. Incorporating residual error shifting into diffusion-based SR facilitates rapid, robust HR image reconstruction, enhancing clinical MRI workflow and advancing medical imaging research. Code available at https://github.com/mosaf/Res-SRDiff}
}
```


## Acknowledgments

- This project is based on [Original Repository Name](https://github.com/zsyOAOA/ResShift) :heart:.
