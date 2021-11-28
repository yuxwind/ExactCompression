# Exact Compression
[Exact-Compression](https://proceedings.neurips.cc/paper/2021/file/e35d7a5768c4b85b4780384d55dc3620-Paper.pdf) is an approach to compress a rectifier network while exactly preserving its underlying functionality with respect to a given input domain if some of its neurons are stable.

<p align="center" width="100%">
    <img width="66%" src="figures/2021_11_NeurIPS_Thumbnail.png"> 
</p>

## Implementation

### Requirements
1. Python 3
2. [Pytorch](http://pytorch.org)
3. Torchvision
4. Cuda 8.0 or higher
5. [Gurobi 7.51](https://www.gurobi.com/downloads/gurobi-optimizer-eula/)


### Installation

We assume that cuda-8.0 is present on your machine.

#### Python

```bash
conda create -n lossless python=3.6.3
conda activate lossless

pip install -r requirements.txt
```

#### Gurobi
Gurobi is not available on the public PyPI server. The following command instructs the Python interpreter to download gurobipy from the private PyPI server

```bash
python -m pip install -i https://pypi.gurobi.com gurobipy
```

a. Visit Gurobi Website and request for an academic license [here](https://www.gurobi.com/academia/academic-program-and-licenses/)

c. Then, type the following

```bash
/opt/gurobi_server900/linux64/bin/grbgetkey xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
```

The ```grbgetkey``` program will prompt you to store the license file on your machine.


### Directory structure
We need to make some extra directories to store the dataset models
```bash
cd $PROJECT_DIR
# For storing train datasets
mkdir data
# This directory stores the models of the training in its sub-directories
mkdir -p model_dir  
```

The directory structure should look like this
```bash
./LLC-NeurIPS-21-code/
├── data
│   ├── cifar-100-python
│   └── MNIST
├── model_dir
│   ├── CIFAR100-rgb
│   ├── CIFAR10-rgb
│   └── MNIST
│   ...

```

### Training the models with a specific regularisation
```bash
python train_fcnn.py --arch fcnn2d --save-dir model_dir/CIFAR10-rgb/dnn_CIFAR10-rgb_400-400_0.000175_0003 --l1 0.000175 --dataset CIFAR10-rgb --eval-stable
```

### Get stably active/inactive neurons with the proposed new approach 
This will invoke the Gurobi optimizer to list out all stable neurons in a file ```stable_neurons.npy``` inside the respective model directory.
```bash
python get_activation_patterns.py -b --input model_dir/CIFAR10-rgb/dnn_CIFAR10-rgb_400-400_0.000175_0003/weights.dat --formulation network --time_limit 10800 --dataset CIFAR10-rgb --preprocess_all_samples
```

### Get stably active/inactive neurons with the baseline old approach

```bash
python get_activation_patterns.py -b --input model_dir/CIFAR10-rgb/dnn_CIFAR10-rgb_400-400_0.000175_0003/weights.dat --formulation neuron --time_limit 10800 --dataset CIFAR10-rgb
```

### Get a compressed model from the stable neuron file
This will generate a new checkpoint file ```pruned_checkpoint_120.tar ``` of the updated weights and biases after compressing the network according to  ```stable_neurons.npy``` inside the specified model directory. 
```bash
python prune_network.py model_dir/CIFAR10-rgb/dnn_CIFAR10-rgb_400-400_0.000175_0003
```

### Evaluate the uncompressed model
```bash
python train_fcnn.py --arch fcnn_prune --resume model_dir/CIFAR10-rgb/dnn_CIFAR10-rgb_400-400_0.000175_0003/pruned_checkpoint_120.tar -e --dataset CIFAR10-rgb

```

### Evaluate the compressed model
```bash
python train_fcnn.py --arch fcnn2d --resume model_dir/CIFAR10-rgb/dnn_CIFAR10-rgb_400-400_0.000175_0003/checkpoint_120.tar -e --dataset CIFAR10-rgb
```

We provided the script 'llc2e.py' with the config of different combination of the netwrok architecure, regularization. Please feel free to play around it.

Our main contribution is the new approach to get the stable neurons for a linear network. We thus provided a few models to make it easy to play with pruning networks.
```
model_dir
├── CIFAR100-rgb
│   └── dnn_CIFAR100-rgb_400-400_7.500000000000001e-05_0001
├── CIFAR10-rgb
│   └── dnn_CIFAR10-rgb_400-400_0.000175_0003
└── MNIST
    ├── dnn_MNIST_100-100_0.0_0000
    └── dnn_MNIST_200-200_0.0_0001
```    
## Citing Exact-Compression
Please cite Exact-Compression in your publication if it helps your research:

```
@article{serra2021scaling,
  title={Scaling Up Exact Neural Network Compression by ReLU Stability},
  author={Serra, Thiago and Yu, Xin and Kumar, Abhinav and Ramalingam, Srikumar},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  year={2021}
}
```
