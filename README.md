# Introduction 
This repository contains the source code inspired from the source code of the paper 
**The Hammer and the Nut: Is Bilevel Optimization Really Needed to Poison Linear Classifiers?**, 
by Antonio Emanuele Cinà, Sebastiano Vascon, Ambra Demontis, Battista Biggio, Fabio Roli and Marcello Pelillo,
accepted at IJCNN 2021.
In this study, in the concept of data privacy and security lecture at Koç University,
we extended beta-poisoning attacks on non-linear classifiers and compare 
the results with baseline attacks (e.g label flipping, clean label poisoning) on MNIST and CIFAR-10 datasets.



# Installation 
Please use the command 
```bash 
$ conda env create -f environment.yml
```
to create the conda environment `beta_poison`. 
Then you need to activate the conda env by running `conda activate beta_poison`

# Experiments
The file `run_experiments.sh` contains a list of commands to replicate the experiments and results
proposed in our paper. Output files are, by default, saved in "IJCNN_Experiments/" dir.

### MNIST Pairs
Experiments for MNIST Pairs (4 vs. 0 and 9 vs. 8):
```bash
$ ./run_experiments.sh mnist_bin
```

### CIFAR-10 Pairs
Experiments for CIFAR-10 Pairs (frog vs. ship and frog vs. horse):
```bash
$ ./run_experiments.sh cifar_bin
```

### MNIST TRIPLET
Experiments for MNIST triplet are obtained with:
```bash
$ ./run_experiments.sh mnist_triplet
```

### Ablation study
Ablation study evaluates the effect of k (number of prototypes) during the optimization procedure.
To run it use:

```bash
$ ./run_experiments.sh mnist_ablation # ablation on mnist
$ ./run_experiments.sh cifar_ablation # ablation on cifar
```

## Contact

* berdem21@ku.edu.tr

