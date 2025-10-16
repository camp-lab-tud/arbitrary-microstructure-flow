[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)
![Python](https://img.shields.io/badge/python-3.10.12-blue.svg)
![Repo Size](https://img.shields.io/github/repo-size/camp-lab-tud/arbitrary-microstructure-flow)
[![Scc Count Badge](https://sloc.xyz/github/camp-lab-tud/arbitrary-microstructure-flow)](https://github.com/camp-lab-tud/arbitrary-microstructure-flow)

[![DOI](https://zenodo.org/badge/DOI/10.1016/j.compositesa.2025.109337.svg)](https://doi.org/10.1016/j.compositesa.2025.109337)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.16940478.svg)](https://doi.org/10.5281/zenodo.16940478)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17306446.svg)](https://doi.org/10.5281/zenodo.17306446)


# Predicting flow in arbitrary fibrous microstructures

## 📖 Overview
This repository contains the source code for the ML model and sliding-window technique described in our paper: 

[Jimmy Gaspard Jean](https://camp-lab.org/members/jimmy-jean.html),
[Guillaume Broggi](https://camp-lab.org/members/guillaume-broggi.html),
[Baris Caglar](https://camp-lab.org/members/baris-caglar.html) (2025).<br>
[**An image-based deep learning framework for flow field prediction in arbitrary-sized fibrous microstructures**](https://doi.org/10.1016/j.compositesa.2025.109337).<br>
Composites Part A: Applied Science and Manufacturing, 109337


### BibTeX
```
@article{jean2025deep,
  title={An image-based deep learning framework for flow field prediction in arbitrary-sized fibrous microstructures},
  author={Jean, Jimmy Gaspard and Caglar, Baris and Broggi, Guillaume},
  journal={Composites Part A: Applied Science and Manufacturing},
  pages={109337},
  year={2025},
  publisher={Elsevier}
}
```

### Abstract
Numerical simulations are commonly used to predict resin flow in fibrous reinforcements but exhibit a trade-off between accuracy and computational cost. As an alternative, machine learning (ML) based models pose as a potential tool to accelerate or replace such costly simulations. This work proposes an open-source image-based deep learning framework to estimate the permeability of unidirectional microstructures in arbitrarily sized domains. This presents a scalable step towards estimating the permeability of large meso-domains. First, we present two robust and accurate surrogate models capable of predicting microstructure velocity and pressure fields with varying physical dimensions, fiber diameter, and volume fraction. These predictions achieve 5% error on the training set and 8% error on unseen microstructures. Secondly, based on those predicted flow fields, we infer the permeability of the microstructures with respectively 4% and 6% deviation for the training and validation sets. Third, opposed to previous works limited to microstructures with a fixed aspect ratio, we propose a so-called sliding window procedure, based on physics-based principles to predict the resin velocity and pressure field in microstructures with different aspect ratios. The method is validated against high-fidelity numerical simulations, and its predictive performance and computational efficiency are confirmed with μ-CT scans of real microstructures. Finally, the presented code and surrogate model are open-sourced to promote further exploration by the scientific community.


<img src="figs/graphic_abstract.png">


## 🗂️ Setup

Assuming that [Python 3.10.12](https://www.python.org/downloads/release/python-31012/) is installed on your computer, create a virtual environment in the main folder by running:

    python -m venv ${ENV}
`${ENV}` (e.g., `.venv`) is the environment's name.

Activate the created environment by running (on Linux):

    source .venv/bin/activate


Install the required dependencies:

    pip install -r requirements.txt



## 🚀 Inference
For inference on the validation split of the [dataset](https://doi.org/10.5281/zenodo.16940478) with a [pre-trained model](https://doi.org/10.5281/zenodo.17306446) , run

    python eval.py --folder ${FOLDER} --split 'valid'
in which `${FOLDER}` refers to the folder containing the trained model weights.


## 💻 Training
To train from scratch, run:

(for the velocity field)

    python train.py --root-dir ${ROOT-DIR} --predictor-type 'velocity' --in-channels 1 --out-channels 2

(for the pressure field)

    python train.py --root-dir ${ROOT-DIR} --predictor-type 'pressure' --in-channels 2 --out-channels 1 --distance-transform ''

`${ROOT-DIR}` refers to the dataset directory.

