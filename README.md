# pinn4bhoc
Physics-informed neural networks (PINN)[1] are a way to solve ordinary and partial differential equations using deep neural networks. For many problems, the domain of every solution is the same. The goal of this project is to devise an algorithm to solve a 2nd-order ordinary differential equation (ODE) when the domain of the solution depends on the solution. We shall investigate the equation describing the orbits of photons in the Schwarzschild metric[2]
\begin{align}
    ds^2 & = f (c dt)^2 - (f^{-1} dr ^2 + r^2 d\theta^2 + r^2 \sin^2\theta \, d\phi^2),
\end{align}
where $f = 1 - u$,  $u = r_s \, / \, r$, $r$ is a radial coordinate, and $r_s = 2 G M / c^2$ is the Schwarzschild radius. The quantities $M$, $G$ and $c$ are the mass of a spherically symmetric object, the Newton gravitational constant and the speed of light in vacuum, respectively. The angular coordinates $\theta$ and $\phi$ are the polar and azimuthal angles, respectively. Because of the conservation of angular momentum, every photon orbit lies in a plane. Therefore, without loss of generality we may set the polar angle $\theta = \pi/2$ in which case $d\theta = 0$. Then the metric becomes
\begin{align}
    ds^2 & = f (c dt)^2 - (f^{-1} dr^2 + r^2 \, d\phi^2) .
\end{align}
