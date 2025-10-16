# pinn4bhoc: PINN for Black Hole Photon Orbit Calculator

## Introduction
This module can  be used to train a Physics-Informed Neural Network (PINN) [1, 2] to solve the following nonlinear ordinary differential equation (ODE):

```math
\overset{\textstyle\cdot\cdot}{u}  \: + \: u - \: 3 \: \frac{u^2}{2}  = \: 0 ,
```

which describes the orbit of photons in a Schwarzschild spacetime about a spherically symmetric body of mass $M$.

### Notation
The variable of interest here is
```math
  u = \: \frac{r_s }{ r},
```

where $r_s$, the Schwarzschild radius, is defined as $r_s = \: \frac{2 G M}{c^2},$ where $G$ is Newton's gravitational constant and $c$ is the speed of light in vacuum.


If $C$ is the proper circumference of a circle centered at the center of mass,
in a Schwarzschild spacetime, the radial coordinate is *defined by* $r \equiv \frac{C}{2\pi}$ and differs from the proper radial distance.


The overdot here ($\overset{\textstyle\cdot\cdot}{u}$) indicates differentiation with respect to $\phi$, the azimuthal angle in a spherical polar coordinate system, $(r, \theta, \phi)$. Here $\theta$ is set to $\pi \, / \, 2$ without loss of generality.


The initial conditions are
```math
u\,(0) &= u_0 \\
\overset{\textstyle\cdot}{u}\,(0) &= v_0.
```

### Approach
The ODE is solved using a PINN following the approach in [3]. The neural network is described by the function $g_\beta(\phi; u_0, v_0),$ where $\beta$ are the network's trainable weights.  
  
We use the following Ansatz from the theory of connections (ToC) [4] that incorporates the initial conditions explicitly:

```math
    u(\phi; u_0, v_0)  &= u_0 + g_\beta(\phi; u_0, v_0) - g_\beta(0; u_0, v_0) + \phi \left[ v_0 - \dot{g}_\beta(0; u_0, v_0) \right], \\[1ex]
    \dot{u}(\phi; u_0, v_0) &= v_0 + \dot{g}_\beta(\phi; u_0, v_0) - \dot{g}_\beta(0; u_0, v_0),
```

### References
[1] B. Moseley, [Deep Learning in Scientific Computing (2023)](https://camlab.ethz.ch/teaching/deep-learning-in-scientific-computing-2023.html), ETH Zürich, Computational and Applied Mathematics Laboratory (CAMLab)  
[2] S. Cuomo *et al*., *Scientific Machine Learning through Physics-Informed Neural Networks: Where we are and What's next*, [arXiv:2201.05624](https://doi.org/10.48550/arXiv.2201.05624)  
[3] Aditi S. Krishnapriyan, Amir Gholami, Shandian Zhe, Robert M. Kirby, Michael W. Mahoney, *Characterizing possible failure modes in physics-informed neural networks*, NIPS'21: Proceedings of the 35th International Conference on Neural Information Processing Systems; [arXiv:2109.01050](https://arxiv.org/abs/2109.01050)  
[4] D. Mortari, *The Theory of Connections: Connecting Points*, Mathematics, vol. 5, no. 57, 2017.


