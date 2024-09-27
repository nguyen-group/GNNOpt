# GNNOpt
Universal Ensemble-Embedding Graph Neural Network for Direct Prediction of Optical Spectra from Crystal Structures

<img src="https://github.com/nguyen-group/GNNOpt/assets/46996256/a8aa00ed-5637-494b-9149-c6852a0a58dc" width="600">

# Requirement
GNNOpt requires the packages as follows: 
- `torch`: an open-source machine learning library with strong GPU acceleration.
- `jupyterlab`: a web-based interactive development environment for notebooks, code, and data.
- `e3nn`: a modular PyTorch framework for Euclidean neural networks.
- `ase`: a set of tools and Python modules for setting up, manipulating, running, visualizing, and analyzing atomistic.  
- `seaborn`: a Python data visualization library based on matplotlib.
- `pandas`: a Python library used to work with data sets.
- `scipy`: an open-source software for mathematics, science, and engineering.
- `scikit-learn`: A set of python modules for machine learning and data mining
- `mendeleev`: A package for accessing various properties of elements in the periodic table of elements.

Example to install requirements with conda for CPU:
```md
$ conda create -n torch python=3.9
$ conda activate torch
$ pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
$ pip3 install torch-cluster torch-scatter torch-sparse torch-spline-conv -f https://pytorch-geometric.com/whl/torch-2.3.0+cu121.html
$ pip3 install torch-geometric
$ pip3 install jupyterlab ase e3nn pandas seaborn scipy scikit-learn mendeleev 
```
For GPU, you might need install as following:
$ pip3 install torch-cluster torch-scatter torch-sparse torch-spline-conv

# Directory Description

```md
GNNOpt
├── data
│   └── absorption_mp_data.pkl: a dataset of the dielectric function of 944 materials, which is obtained from Material Project and saved in pickle format.
├── utils
│   ├── utils_data.py: define the function related to the dataset.
│   ├── utils_model.py: define the function related to the model.
│   └── utils_plot.py: define the function related to plotting results.
├── model
│   ├── idx_train_240406.txt: list of the order of the training set.
│   ├── idx_vaild_240406.txt: list of the order of the validation set.
│   ├── idx_test_240406.txt: list of the order of the testing set.
│   ├── model_eps1_240406.torch: the trained model for the real part of the dielectric function using the dataset order in idx_***_240406.txt.
│   ├── model_eps2_240406.torch: the trained model for the imaginary part of the dielectric function using the dataset order in idx_***_240406.txt.
│   ├── model_alpha_240406.torch: the trained model for absorption coefficient using the dataset order in idx_***_240406.txt.
│   ├── model_n_240406.torch: the trained model for the real part of the refractive index using the dataset order in idx_***_240406.txt.
│   ├── model_k_240406.torch: the trained model for the imaginary part of the refractive index using the dataset order in idx_***_240406.txt.
│   └── model_R_240406.torch: the trained model for the reflectance using the dataset order in idx_***_240406.txt.
├── gnnopt-eps1.ipynb: the GNNOpt code to predict the real part of the dielectric function from the crystal structure.
├── gnnopt-eps2.ipynb: the GNNOpt code to predict the imaginary part of the dielectric function from the crystal structure.
├── gnnopt-alpha.ipynb: the GNNOpt code to predict the absorption coefficient from the crystal structure.
├── gnnopt-n.ipynb: the GNNOpt code to predict the real part of the refractive index from the crystal structure.
├── gnnopt-k.ipynb: the GNNOpt code to predict the imaginary part of the refractive index from the crystal structure.
└── gnnopt-R.ipynb: the GNNOpt code to predict the reflectance from the crystal structure.
```
# How to run
Step 1: Download the GNNOpt package:

    git clone https://github.com/nguyen-group/GNNOpt.git

Step 2: Go to the source code in the GNNOpt directory to run the program:

    cd GNNOpt
    jupyter-lab gnnopt-***.ipynb
(***: is the optical spectra that you want to run; for example, `jupyter-lab gnnopt-alpha.ipynb` for the absorption coefficient)

Note: In `gnnopt-***.ipynb` files, we used the trained model in the folder `GNNOpt/model` to save time. If you want to train the model on your computer, please uncomment at line:

`# train(model, opt, dataloader_train, dataloader_valid, loss_fn, loss_fn_mae, run_name, max_iter=100, scheduler=scheduler, device=device)`

# References and citing
The detail GNNOpt is described in our pre-print:
> N. T. Hung, R. Okabe,  A. Chotrattanapituk and M. Li, Ensemble-embedding graph neural network for direct prediction of optical spectra from crystal structure, arXiv:2406.16654
> 
> https://arxiv.org/abs/2406.16654

# Contributors
- Nguyen Tuan Hung (Tohoku University, Japan)
- Ryotaro Okabe (MIT, USA)
- Abhijatmedhi Chotrattanapituk (MIT, USA)
- Mingda Li (MIT, USA)
