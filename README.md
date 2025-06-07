# <p align=center>`MRI super-resolution reconstruction using efficient diffusion probabilistic model with residual shifting`</p> # 


[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT) [![Open Access](https://img.shields.io/badge/PMB%20Paper-Open%20Access%20%7C%20Click%20to%20Read-8a2be2)](https://doi.org/10.1088/1361-6560/ade049)



:fire::fire:**Res-SRDiff** is a deep learning framework designed to robustly restore high-resolution pelvic T2w MRI and ultra-high field brain T1 maps using an efficient probabilistic diffusion model.

- Our paper on Physics in Medicine and Biology: [MRI super-resolution reconstruction using efficient diffusion probabilistic model with residual shifting](https://doi.org/10.1088/1361-6560/ade049) :heart:


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
	journal={Physics in Medicine \& Biology},
	url={http://iopscience.iop.org/article/10.1088/1361-6560/ade049},
	doi={https://doi.org/10.1088/1361-6560/ade049},
	year={2025}
}
```


## Acknowledgments

- This project is based on [ResShift](https://github.com/zsyOAOA/ResShift) :heart:.
