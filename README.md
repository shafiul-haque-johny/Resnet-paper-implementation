# Resnet-paper-implementation  
Residual Networks (ResNet) – Deep Learning: code collected from:https://www.geeksforgeeks.org/ 
After the first CNN-based architecture (AlexNet) that win the ImageNet 2012 competition, Every subsequent winning architecture uses more layers in a deep neural network to reduce the error rate. This works for less number of layers, but when we increase the number of layers, there is a common problem in deep learning associated with that called Vanishing/Exploding gradient. This causes the gradient to become 0 or too large. Thus when we increases number of layers, the training and test error rate also increases.

In the above plot, we can observe that a 56-layer CNN gives more error rate on both training and testing dataset than a 20-layer CNN architecture, If this was the result of over fitting, then we should have lower training error in 56-layer CNN but then it also has higher training error. After analyzing more on error rate the authors were able to reach conclusion that it is caused by vanishing/exploding gradient.

ResNet, which was proposed in 2015 by researchers at Microsoft Research introduced a new architecture called Residual Network.
Residual Block:
In order to solve the problem of the vanishing/exploding gradient, this architecture introduced the concept called Residual Network. In this network we use a technique called skip connections . The skip connection skips training from a few layers and connects directly to the output.

The approach behind this network is instead of layers learn the underlying mapping, we allow network fit the residual mapping. So, instead of say H(x), initial mapping, let the network fit, F(x) := H(x) – x which gives H(x) := F(x) + x.
The advantage of adding this type of skip connection is because if any layer hurt the performance of architecture then it will be skipped by regularization. So, this results in training very deep neural network without the problems caused by vanishing/exploding gradient.  The authors of the paper experimented on 100-1000 layers on CIFAR-10 dataset.

There is a similar approach called “highway networks”, these networks also uses skip connection. Similar to LSTM these skip connections also uses parametric gates. These gates determine how much information passes through the skip connection. This architecture however  has not provide accuracy better than ResNet architecture.

Network Architecture:

This network uses a 34-layer plain network architecture inspired by VGG-19 in which then the shortcut connection is added. These shortcut connections then convert the architecture into residual network. 

Implementation:

Using the Tensorflow and Keras API, we can design ResNet architecture (including Residual Blocks) from scratch. Below is the implementation of different ResNet architecture. For this implementation we use CIFAR-10 dataset. This dataset contains 60, 000 32×32 color images in 10 different classes (airplanes, cars, birds, cats, deer, dogs, frogs, horses, ships, and trucks) etc. This datasets can be assessed  from keras.datasets API function.

First, we import the keras module and its APIs. These APIs help in building architecture of the ResNet model.

Results & Conclusion:

On the ImageNet dataset,  the authors uses a 152-layers ResNet, which is 8 times more deep than VGG19 but still have less parameters. An ensemble of these ResNets generated an error of only 3.7% on ImageNet test set, the result which won ILSVRC 2015 competition. On COCO object detection dataset, it also generates a 28% relative improvement due to its very deep representation.
