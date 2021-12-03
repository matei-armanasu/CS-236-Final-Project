# CS236 Final Project

## Generative Editing for Adversarial Attacks

Given the wide adoption of deep learning models, it is important to be aware on how these networks can be fooled to produce strange and potentially dangerous behaviors. An adversarial example is an instance with small, intentional feature perturbations that cause a machine learning model to make a false prediction. So far, there are many ways to generate such examples. Our goal is to explore a variant on adversarial image generation, using a generative network to produce an arbitrary quantity of edited images after training against a single classifier. We specify the fooling label which makes this a targeted attack. We compare this approach to other adversarial methods, and explore potential architectures for this adversarial generator.

## Methodology
* Generator takes in an image I as input, sized for input to a given classifier C.
* Generator outputs an image I' of the same size, perturbed such that C believes it to be from a target class t.
* Generator architecture uses a series of convolutional layers, and an optional residual mode in which the input image is added to the output to create a ResNet-style block network.
* Training process is white-box, relies on access to classifier for backpropagation.
* <img src="https://render.githubusercontent.com/render/math?math=e^{i \pi} = -1">
* 
* ![Loss functions (for generator $\mathcal{G}$
, classifier $\mathcal{C}$, image $i$, target class $c$, tuning parameter $\alpha$, and Categorical Cross-Entropy loss $CCE(
    \cdot,\cdot)$): $L_{MSE} = \frac{\alpha}{64\times64\times3}||i-\mathcal{G}(i)||_2^2$, $L_{Adv.}= CCE(c,\mathcal{C}(\mathcal{G}(i)))$](https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+Loss+functions+%28for+generator+%24%5Cmathcal%7BG%7D%24%0A%2C+classifier+%24%5Cmathcal%7BC%7D%24%2C+image+%24i%24%2C+target+class+%24c%24%2C+tuning+parameter+%24%5Calpha%24%2C+and+Categorical+Cross-Entropy+loss+%24CCE%28%0A++++%5Ccdot%2C%5Ccdot%29%24%29%3A+%24L_%7BMSE%7D+%3D+%5Cfrac%7B%5Calpha%7D%7B64%5Ctimes64%5Ctimes3%7D%7C%7Ci-%5Cmathcal%7BG%7D%28i%29%7C%7C_2%5E2%24%2C+%24L_%7BAdv.%7D%3D+CCE%28c%2C%5Cmathcal%7BC%7D%28%5Cmathcal%7BG%7D%28i%29%29%29%24)
