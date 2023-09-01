# Deep-Learning-on-PDE
A summary of Yucheng's work, supervised by Pro.Bin Han, sponsored by CSC 

Solving Second-order Elliptic Interface problems with Mixed Boundary Conditions

# Best Results
1. Poisson equation with Dirichlet and Neumann boundary condition and one closed interface

solution:
![](https://github.com/1417573837/Deep-Learning-on-PDE/blob/main/Best%20Results/3_1%20VDL%20v5.5/U2D%20400.jpg)
solution's gradient (of x):
![](https://github.com/1417573837/Deep-Learning-on-PDE/blob/main/Best%20Results/3_1%20VDL%20v5.5/Ux2D%20400.jpg)
3D plot:
![](https://github.com/1417573837/Deep-Learning-on-PDE/blob/main/Best%20Results/3_1%20VDL%20v5.5/U3DPred200.jpg)

2. Poisson equation with Dirichlet boundary condition and cross interface
solution:
![](https://github.com/1417573837/Deep-Learning-on-PDE/blob/main/Best%20Results/CrossE5Simplified%20VDLV2%20SAIS%20v2.3/U2D%20200.jpg)
solution's gradient (of x):
![](https://github.com/1417573837/Deep-Learning-on-PDE/blob/main/Best%20Results/CrossE5Simplified%20VDLV2%20v3.4/Ux2D%200.jpg)
3D plot:
![](https://github.com/1417573837/Deep-Learning-on-PDE/blob/main/Best%20Results/CrossE5Simplified%20VDLV2%20v3.4/U3DPred0.jpg)


# Methodology
## Physics-informed Neural Network(PINN)
The main parts of PINN: network structure, pre/post-processing, sampling method, optimizer. Free to switch components for each part.

## Domain Decomposition
For interface problems, it's wise to separate the whole domain into subdomains by the interface. 

The advantage is you don't need to use limit to calculate the jump across the interface. The solution on each subdomain is continuous, which is easy for a neural network to express.

Another choice to deal with an interface is cusp-capturing.


## Hard-constraints
A post-processing technique that assures the final output satisfies boundary conditions.
For details, see https://arxiv.org/pdf/2102.04626.pdf

## Cusp-capturing
Another choice for interface problems. Since a neural network is infinitely continuous, when regarded as a function, they introduced a function with a nondifferentiable point, such as abs(x), to be one of the input variables of the neural network. In this way, the neural network's output has some nondifferentiable points. This method can be easily extended to make the output even discontinuous.
For details, see https://arxiv.org/pdf/2210.08424.pdf

## Failure-informed Adaptive Sampling
Add randomly sampled points from areas where their residuals are large.
For details, see https://arxiv.org/pdf/2210.00279.pdf

## Vector Dense Layer
My idea, showed some effect. 

Allowing to include arbitrary activations with arbitrary dimension input and output. 

How it works:

Suppose you give p activations, p is an integer. For simplicity, let's say activations are all n-dim to m-dim.

Suppose your input given by the last dense layer is an N-dim vector. Suppose N is exactly divided by n.

We split the N-dim vector into N/n parts of n-dim subvector.

Then, for each subvector, apply p activations to get p * m-dim outputs.

Altogether is N / n * p * m-dimension output vector.

The benefit of it is adding a high dimension twist to neural networks, and taking advantage of each activation capturing different characteristics.

## Psudo-gradient
My idea, but doesn't work well. I assumed that the neural network is hard to fit the solution while the gradient of the neural network (using auto-derivation) fits the gradients of the solution. I tried to express solutions and gradients separately, using another network to express the gradient of the solution.

## Code Structure
There are several parts:



|File name | Description|
|---|---|
|main.py | Assemble each part, including model definition and training control|
|PDE/PDExx.py| The definition of PDE, including equations, boundaries, etc.|
|plotting.py| Customized plot settings and fine control|
|conditional_print.py| Just for fun, add some visual effects to the result reports in the console. Losses values will be green if smaller than the last one, green and underlined if smallest in history, otherwise red.|
|VectorDenseLayerV2.py| Defines the Vector Dense Layer|
|SAIS.py|Perform self-adaptive Importance Sampling|
|DataGenerate| Generate training points for PDE. You may adjust it for different PDE problems.|

## To whom may want to go further:
Try levenberg-marquardt optimizor, since I was not very sure if I used it correctly.

You may use pytorch instead of TensorFlow, since pytorch is the mainstream in the PINN area.

You may try DeepXDE library, which is more convenient.

PINN is very hopeful.

## Acknowledgement
Thanks to the guidance from Pro.Bin Han, the support from Dr.Qiwei Feng, and the scholarship from the Chinese Scholarship Council.


