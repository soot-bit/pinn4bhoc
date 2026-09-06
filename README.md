<div align="center">

<pre style="font-size: 5pt; line-height: 1;">
██████╗ ██╗███╗   ██╗███╗   ██╗██╗  ██╗██████╗ ██╗  ██╗ ██████╗  ██████╗
██╔══██╗██║████╗  ██║████╗  ██║██║  ██║██╔══██╗██║  ██║██╔═══██╗██╔════╝
██████╔╝██║██╔██╗ ██║██╔██╗ ██║███████║██████╔╝███████║██║   ██║██║     
██╔═══╝ ██║██║╚██╗██║██║╚██╗██║╚════██║██╔══██╗██╔══██║██║   ██║██║     
██║     ██║██║ ╚████║██║ ╚████║     ██║██████╔╝██║  ██║╚██████╔╝╚██████╗
╚═╝     ╚═╝╚═╝  ╚═══╝╚═╝  ╚═══╝     ╚═╝╚═════╝ ╚═╝  ╚═╝ ╚═════╝  ╚═════╝
</pre>

<h1>pinn4bhoc</h1>

<p><b>Physics-Informed Neural Network for Black-Hole Photon-Orbit Calculation</b></p>

<p>Solving photon orbits in Schwarzschild spacetime with a PINN + Theory of Connections ansatz.</p>

<p>
  <a href="https://www.python.org/downloads/"><img alt="Python" src="https://img.shields.io/badge/Python-3.12%2B-3776AB?style=for-the-badge&logo=python&logoColor=white"></a>
  <a href="https://pytorch.org/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.9%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"></a>
  <a href="https://numpy.org/"><img alt="NumPy" src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white"></a>
  <a href="https://scipy.org/"><img alt="SciPy" src="https://img.shields.io/badge/SciPy-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white"></a>
</p>
</div>


## Introduction
This module can  be used to train a Physics-Informed Neural Network (PINN) [1, 2] to solve the following nonlinear ordinary differential equation (ODE):
```math
\overset{\textstyle\cdot\cdot}{u}  \: + \: u - \: 3 \: \frac{u^2}{2}  =  0 ,
```
which describes the orbit of photons in a Schwarzschild spacetime about a spherically symmetric body of mass $M$.

### Notation
The variable of interest here is
```math
  u = \frac{r_s }{ r},
```
where $r_s$, the Schwarzschild radius, is defined as $r_s =  2 G M  /  c^2,$ where $G$ is Newton's gravitational constant and $c$ is the speed of light in vacuum. If $C$ is the proper circumference of a circle centered at the center of mass,
in a Schwarzschild spacetime, the radial coordinate is *defined by* $r \equiv C  /  (2\pi)$ and differs from the proper radial distance.
The overdot here ($\overset{\textstyle\cdot\cdot}{u}$) indicates differentiation with respect to $\phi$, the azimuthal angle in a spherical polar coordinate system, $(r, \theta, \phi)$. Here $\theta$ is set to $\pi  /  2$ without loss of generality.

The initial conditions are
```math
u\,(0) = u_0
```
and
```math
\overset{\textstyle\cdot}{u}\,(0) = v_0.
```

### Approach
The ODE is solved using a PINN following the approach in [3]. The neural network is described by the function $g_\beta(\phi, u_0, v_0),$ where $\beta$ are the network's trainable weights.  
  
We use the following Ansatz from the theory of connections (ToC) [4] that incorporates the initial conditions explicitly:

```math
    u(\phi, u_0, v_0)  = u_0 + g_\beta(\phi, u_0, v_0) - g_\beta(0, u_0, v_0) + \phi \left[ v_0 - \dot{g}_\beta(0, u_0, v_0) \right],
```
and
```math
    \dot{u}(\phi, u_0, v_0) = v_0 + \dot{g}_\beta(\phi, u_0, v_0) - \dot{g}_\beta(0, u_0, v_0),
```

### References
[1] B. Moseley, [Deep Learning in Scientific Computing (2023)](https://camlab.ethz.ch/teaching/deep-learning-in-scientific-computing-2023.html), ETH Zürich, Computational and Applied Mathematics Laboratory (CAMLab)  
[2] S. Cuomo *et al*., *Scientific Machine Learning through Physics-Informed Neural Networks: Where we are and What's next*, [arXiv:2201.05624](https://doi.org/10.48550/arXiv.2201.05624)  
[3] Aditi S. Krishnapriyan, Amir Gholami, Shandian Zhe, Robert M. Kirby, Michael W. Mahoney, *Characterizing possible failure modes in physics-informed neural networks*, NIPS'21: Proceedings of the 35th International Conference on Neural Information Processing Systems; [arXiv:2109.01050](https://arxiv.org/abs/2109.01050)  
[4] D. Mortari, *The Theory of Connections: Connecting Points*, Mathematics, vol. 5, no. 57, 2017.

## Getting Started

Clone the repository:

```

git clone https://github.com/soot-bit/pinn4bhoc.git
cd pinn4bhoc

```

inside a virtual environment:
Install the project in editable mode (installs dependencies too):

```

pip install -e .

```

Verify the installation:

```

python -c "from pinn4bhoc import nn"

```

## Google Colab installation `pinn4bhoc`
  1. Assign Colab working folder to string `COLAB_FOLDER` in notebook.
  2. Execute the code below in a notebook cell before your imports (see, for example, `01_pinn_training.ipynb`).
```python
COLAB_FOLDER = 'AIMS' # change as needed
GITHUB_USER  = 'soot-bit'
GITHUB_REPO  = 'pinn4bhoc'
GITHUB_FOLDERS = ['pinn4bhoc']
#------------------------------------------------------
MYDRIVE      = '/content/gdrive/MyDrive'
GITHUB_BASE  = 'https://raw.githubusercontent.com'
GITHUB_PATH  = f'{MYDRIVE}/{COLAB_FOLDER}'
#------------------------------------------------------
try:
    from google.colab import drive
    drive.mount('/content/gdrive')
    print('\nGoogle Drive mounted\n')
    IN_COLAB = True
except:
    print('\nRunning locally\n')
    IN_COLAB = False
 
if IN_COLAB:
    %cd {GITHUB_PATH}
    %rm -f {GITHUB_PATH}/clone2colab.ipynb
    !wget -q {GITHUB_BASE}/{GITHUB_USER}/{GITHUB_REPO}/refs/heads/main/clone2colab.ipynb
    %run {GITHUB_PATH}/clone2colab.ipynb
    %ls
```

