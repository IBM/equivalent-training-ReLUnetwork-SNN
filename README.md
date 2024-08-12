# High-performance deep spiking neural networks with 0.3 spikes per neuron

This repository contains code material for the publication: Stanojevic, A., Woźniak, S., Bellec, G., Cherubini, G., Pantazi, A., &amp; Gerstner, W., High-performance deep spiking neural networks with 0.3 spikes per neuron, 
Nature Communications 15, 6793 (2024). https://www.nature.com/articles/s41467-024-51110-5

Deep spiking neural networks (SNNs) offer the promise of low-power artificial intelligence. However, training deep spiking neural networks (SNNs) with backpropagation has been harder than with artificial neural networks (ANNs), which is puzzling given that recent theoretical results provide exact mapping algorithms from ReLU to time-to-first-spike (TTFS) SNNs. After identifying the issue of vanishing-and-exploding gradient we derive a generic solution for the network initialization and SNN parameterization which enables that the an energy-efficient SNN can be trained as robustly as its ANN counterpart. The SNN training is coded in Python and Tensorflow.

## Usage
This repository contains code which trains single-spike TTFS neural networks for different benchmark datasets such that they achieve same performance as corresponding ReLU networks.

To use the code please create an anaconda environment using the configuration file (on Linux platform):
```console
foo@bar:~$ conda env create -f environment.yml
```

Please consult the comments in the source code of `main.py` for the usage options.

## Examples

Below are a few selected examples of various scenarios for quick verification using 1 epoch only (and execution time on a GPU).

### Training from scratch: SNN on MNIST 

Training SNN FC2 on MNIST from scratch (~21 s):
```
python main.py --model_type=SNN --model_name=FC2_example_train --data_name=MNIST --epochs=1
``` 
Training SNN VGG16 on MNIST from scratch (~19 min)
```
python main.py --model_type=SNN --model_name=VGG_example_train --data_name=MNIST --epochs=1
```

### Fine-tuning: SNN fine-tuning from mapped ReLU trained on MNIST

Training ReLU FC2 on MNIST from scratch and saving (~16 s):
```
python main.py --model_type=ReLU --model_name=FC2_example --data_name=MNIST --save=True --epochs=1
``` 
Fine-tuning SNN FC2 on MNIST (~21 s):
```
python main.py --model_type=SNN --model_name=FC2_example --data_name=MNIST --load=True --epochs=1
```

### Fine-tuning: SNN fine-tuning from ReLU pretrained on CIFAR10

Download VGG16 CIFAR10 pretrained weights `cifar10vgg.h5` from [https://github.com/geifmany/cifar-vgg](https://github.com/geifmany/cifar-vgg) to your `logging_dir` location (`./logs/` by deafult)

Test pretrained ReLU VGG16 and preprocess the model for mapping to SNN (~3 min):
```
python main.py --model_type=ReLU --model_name=VGG_BN_example --data_name=CIFAR10 --load=cifar10vgg.h5 --save=True --epochs=0
```

Fine-tune SNN VGG16 on CIFAR10 (~12 min):
```
python main.py --model_type=SNN --lr=1e-6 --model_name=VGG_BN_example --data_name=CIFAR10 --load=True --epochs=1
```

## Feedback
If you have feedback or want to contribute to the code base, please feel free to open Issues or Pull Requests via Git directly.
