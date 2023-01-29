# HPC-AI - Distributed Linear Algebra

## Geometric multigrid method in 2D

The aim of this project is to implement a multigrid method to solve the 2D Poisson problem on the unit square.

```math
\begin{gather*}
- \Delta u(x,y) +\sigma u(x,y) = f(x,y) \text{ in } \Omega \text{ ,} \quad \sigma \ge 0 \\
u(x,y) = 0 \text{ in } \partial \Omega
\end{gather*}
```

We will also tackle the anisotropic problem.

```math
\begin{gather*}
-\frac{\partial^2(x,y)}{\partial x^2} -\epsilon  \frac{\partial^2(x,y)}{\partial y^2} = f(x,y)) \text{ in } \Omega  \\
u(x,y) = 0 \text{ in } \partial \Omega
\end{gather*}
```

## Python scripts

```tp2.py``` implements multigrid method for Poisson equation in 1D. It was the starting point of the project.

```geometric_multigrid_2D.py``` implements multigrid method for Poisson equation in 2D.

These scripts implement:
- Construction of the Laplacian matrix
- Stationary iterative methods (JOR and SOR)
- Transfer operators between fine and coarse grid
- Convergence speed
- Plots and post-process
