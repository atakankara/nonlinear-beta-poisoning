# Introduction 
This repository contains the source code for the paper **Beta Poisoning Attacks Against Machine Learning Models: Extensions, Limitations, and Defenses**, by Atakan Kara, Nursena Köprücü, M. Emre Gürsoy, accepted at IEEE TPS 2022: IEEE International Conference on Trust, Privacy and Security in Intelligent Systems, and Applications. It is cloned from the original implementation of beta poisoning, which you can find [here](https://github.com/Cinofix/beta_poisoning).




# Installation 
Please use the command 
```bash 
$ conda env create -f environment.yml
```
to create the conda environment `beta_poison`. 
Then you need to activate the conda env by running `conda activate beta_poison`

# Experiments
The file `run_experiments.sh` contains a list of commands to replicate the experiments and results
proposed in our paper. Output files are, by default, saved in "TPS_experiments/" dir.

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
* akara18 [at] ku.edu.tr
* berdem21 [at] ku.edu.tr
* nkoprucu16 [at] ku.edu.tr
