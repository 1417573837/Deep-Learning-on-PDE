# Deep-Learning-on-PDE
A summary of Yucheng's work, supervised by Pro.Bin Han, sponsored by CSC 

Solving Second-order Elliptic Interface problems with Mixed Boundary Conditions

# Methodology
## Physics-informed Neural Network(PINN)
The main parts of PINN: network structure, pre/post-processing, sampling method, optimizer. Free to swich componets for each part.

## Domain Decomposition
For interface problems, it's wise to seperate the whole domain into subdomains by interface. 

e.g.: for the following circle interface, seperate into inner subdomain and outer subdomain.
![][]

## Hard-constraints

## Cusp-capturing

## Failure-informed Adaptive Sampling

## Psudo-gradient
