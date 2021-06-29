# The Bures Metric for Generative Adversarial Networks
This repository contains the code used in the experiments for the paper "The Bures Metric for Generative Adversarial Networks".
It contains complete implementations of the following GANs in Tensorflow 2:

 - BuresGAN  
 - AlternatingBuresGAN 
 - FrobeniusGAN
 - Vanilla GAN [1]
 - VEEGAN [2]
 - MDGAN (Both variants) [3]
 - UnrolledGAN [4]
 - GDPPGAN [5]
 - WGAN-GP [6]
 - PacGAN [7]

## Requirements
Experiments have been run on given versions but it is often not required to run with the exact same package versions.
- Tensorflow 2.2.0
  - tensorflow-probability   0.10.0
  - tensorflow-hub           0.8.0
  - tensorflow-gan           2.0.0
- tqdm 4.41.1
- matplotlib               3.2.1

## Usage
The scripts used in the **evaluations** experiments are the following:

 - evaluate_synthetic.py
 - evaluate_cifar10.py 
 - evaluate_cifar100.py 
 - evaluate_stackedMNIST.py
 - evaluate_stl10.py

The synthetic script is run without parameters, the other scripts are used with default settings as follows:
	
    python evaluate_cifar10.py [GAN]

With [GAN] being one of 'AlternatingBures', 'Bures', 'GAN', 'VEEGAN', 'MDGANv1', 'MDGANv2', 'UnrolledGAN', 'GDPPGAN', 'WGANGP'

The scripts used in the **timing** experiments are the following:
 - timing_cifar10.py 
 - timing_cifar100.py 
 - timing_stackedMNIST.py
 - timing_stl10.py

These scripts do not take any arguments and can be called in the following manner:

    python timing_cifar10.py
    
The script *sample_saved_models.py* was used to create the figures containing the GAN samples.
The script *train_mnist_cnn.py* was used to train the MNIST CNN model used in the evaluation of stacked MNIST to calculate the number of modes and KL-divergence.

The **data** directory contains the pretrained MNIST CNN model used for the evaluation of stacked MNIST. It also contains the stacked MNIST dataset that was used in the experiments. All other data sets are downloaded automatically when running the scripts.

## ResNet Experiments
The scripts used in these experiments can be found in the **resnet** directory. 
The code is based on the original architecture used in [6]. As such, the code is written in Tensorflow 1.
Requirements:
 - tensorflow-gpu 	1.15
 - scikit-learn 0.22
 - imageio		3.2.1
 - scipy		1.5.2

These scripts do not take any arguments and can be called in the following manner:

	python buresgan_stl_resnet.py
	
The code for the CIFAR-10 experiments require the CIFAR-10 files to be downloade and placed in the **resnet/data** directory.
This data set can be found here: http://www.cs.toronto.edu/~kriz/cifar.html
The STL-10 data set will be downloaded automatically.


## References

[1] Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative Adversarial Nets. In Advances in Neural Information Processing Systems 27, 2014.

[2] Akash Srivastava, Lazar Valkov, Chris Russell, Michael U Gutmann, and Charles Sutton. Veegan: Reducing Mode Collapse in GANs using Implicit Variational Learning. In Advances in Neural Information Processing Systems 30, 2017.

[3] Tong Che, Yanran Li, Athul Paul Jacob, Yoshua Bengio, and Wenjie Li. Mode Regularized Generative Adversarial Networks. In Proceedings of the International Conference on Learning Representations (ICLR), 2017.

[4] Luke Metz, Ben Poole, David Pfau, and Jascha Sohl-Dickstein. Unrolled Generative Adversarial Networks. In Proceedings of the International Conference on Learning Representations(ICLR), 2017.

[5] Mohamed Elfeki, Camille Couprie, Morgane Riviere, and Mohamed Elhoseiny. GDPP: Learning Diverse Generations using Determinantal Point Processes. In Proceedings of the 36th International Conference on Machine Learning (ICML), 2019.

[6] Ishaan Gulrajani, Faruk Ahmed, Martin Arjovsky, Vincent Dumoulin, and Aaron C Courville. Improved training of Wasserstein GANs. In Advances in neural information processing systems 31, 2017.

[7] Zinan Lin, Ashish Khetan, Giulia Fanti, and Sewoong Oh. Pacgan: The power of two samples in generative adversarial networks. In Advances in Neural Information Processing Systems 31, 2018.