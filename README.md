# CS236 Final Project

## Generative Editing for Adversarial Attacks

Given the wide adoption of deep learning models, it is important to be aware on how these networks can be fooled to produce strange and potentially dangerous behaviors. An adversarial example is an instance with small, intentional feature perturbations that cause a machine learning model to make a false prediction. So far, there are many ways to generate such examples. Our goal is to explore a variant on adversarial image generation, using a generative network to produce an arbitrary quantity of edited images after training against a single classifier. We specify the fooling label which makes this a targeted attack. We compare this approach to other adversarial methods, and explore potential architectures for this adversarial generator.

## Methodology
* Generator takes in an image I as input, sized for input to a given classifier C.
* Generator outputs an image I' of the same size, perturbed such that C believes it to be from a target class t.
* Generator architecture uses a series of convolutional layers, and an optional residual mode in which the input image is added to the output to create a ResNet-style block network.
* Training process is white-box, relies on access to classifier for backpropagation.

## Results - Visualizations

![Generated Images](https://github.com/matei-armanasu/CS-236-Final-Project/blob/main/visuals-29-Nov/generated115.png)
