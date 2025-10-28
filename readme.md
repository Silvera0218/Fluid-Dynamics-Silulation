
This repository contains the MATLAB project is focusing on solving steady-state and unsteady-state Navier-Stokes equations for 2D channel flow. The project implements and compares various advanced numerical methods to simulate complex incompressible fluid flow around multiple cylindrical obstacles.

![Vortex Street Animation](assets/unsteady_t11.4.png)
![Vortex Street Animation](assets/vortex.gif)
*The von Kármán vortex street phenomenon, successfully captured in the unsteady simulation at a low viscosity (μ=0.01).*

## Project Overview

This project aims to numerically solve the classic computational fluid dynamics problem of 2D flow in a channel with multiple obstacles using the Finite Element Method (FEM). We begin by generating a high-quality unstructured mesh using the `Distmesh` toolbox and employ Taylor-Hood (P2-P1) elements for spatial discretization to ensure the stability and accuracy of the numerical solution.

The core of the project lies in the implementation and comparison of several iterative strategies for handling the non-linear term in the Navier-Stokes equations, including **Stokes (Picard) Linearization**, **Oseen Linearization**, and **Approximate Newton Linearization**. To enhance algorithmic robustness, we introduce a **backtracking line search with the Armijo condition** as a globalization strategy. Finally, the project is extended to unsteady problems, where a **full Newton system** is solved at each time step, successfully simulating the complete evolution from an initial state to a periodic vortex street.

## Governing Equations and Discretization

### 1. Steady-State Navier-Stokes Equations

#### a) Governing Equations (Strong Form)
The steady-state problem describes the final equilibrium state of the fluid flow. The velocity field **u** and pressure field *p* satisfy the following system of equations:
```math
\begin{aligned}
-\mu \Delta \mathbf{u} + (\mathbf{u} \cdot \nabla)\mathbf{u} + \nabla p &= \mathbf{f} \quad &&\text{(Momentum Conservation)} \\
\nabla \cdot \mathbf{u} &= 0 \quad &&\text{(Mass Conservation / Incompressibility)}
\end{aligned}
```
#### b) Spatial Discretization (Finite Element Weak Form)

By multiplying by test functions **v** and *q* and integrating over the domain Ω, we convert the strong form into a weak form:
```math
\begin{aligned}
(\mu \nabla \mathbf{u}, \nabla \mathbf{v}) + ((\mathbf{u} \cdot \nabla)\mathbf{u}, \mathbf{v}) - (p, \nabla \cdot \mathbf{v}) &= (\mathbf{f}, \mathbf{v}) \quad \forall \mathbf{v} \in V^0 \\
(\nabla \cdot \mathbf{u}, q) &= 0 \quad \forall q \in Q
\end{aligned}
```
where `(a, b)` denotes the L2 inner product `∫_Ω a·b dΩ`. After discretizing with Taylor-Hood (P2-P1) elements, we obtain a large-scale non-linear algebraic system:
```math
\begin{pmatrix}
\mu K + N(U) & B^T \\
B & 0
\end{pmatrix}
\begin{pmatrix}
U \\
P
\end{pmatrix}
=
\begin{pmatrix}
F \\
0
\end{pmatrix}
```

```math
- **K**: Stiffness matrix (from the diffusion term `μΔu`)
- **N(U)**: Non-linear convection matrix (from the convection term `(u·∇)u`)
- **B**: Divergence matrix
- **U, P**: Unknown coefficient vectors for velocity and pressure
```
### 2. Unsteady Navier-Stokes Equations

#### a) Governing Equations (Strong Form)
The unsteady problem describes the evolution of the flow field over time `t`. The governing equations include a time derivative term:
$$
\begin{aligned}
\frac{\partial \mathbf{u}}{\partial t} - \mu \Delta \mathbf{u} + (\mathbf{u} \cdot \nabla)\mathbf{u} + \nabla p &= \mathbf{f} \\
\nabla \cdot \mathbf{u} &= 0
\end{aligned}
$$

#### b) Temporal and Spatial Discretization
We employ a "Method of Lines" approach:
1.  **Temporal Discretization (Implicit Euler Method)**: We discretize the time derivative using the first-order accurate Implicit Euler scheme to solve for `u^n` at time step `n`:
    $$
    \frac{\mathbf{u}^n - \mathbf{u}^{n-1}}{\Delta t} - \mu \Delta \mathbf{u}^n + (\mathbf{u}^n \cdot \nabla)\mathbf{u}^n + \nabla p^n = \mathbf{f}^n
    $$
    Rearranging this yields a steady-state-like non-linear equation to be solved at each time step.

2.  **Spatial Discretization (Finite Element Weak Form)**: Applying the same FEM procedure as in the steady case to the time-discretized equation, we get the algebraic system for each time step `n`:

    ```math
    \begin{pmatrix}
    \frac{1}{\Delta t}M + \mu K + N(U^n) & B^T \\
    B & 0
    \end{pmatrix}
    \begin{pmatrix}
    U^n \\
    P^n
    \end{pmatrix}
    =
    \begin{pmatrix}
    \frac{1}{\Delta t}M U^{n-1} + F^n \\
    0
    \end{pmatrix}
    ```
    - **M**: Mass matrix (from the time derivative term `∂u/∂t`)
    - **U^(n-1)**: The known velocity solution from the previous time step

## Numerical Implementation Details

### 1. Mesh Generation
We use `Distmesh` to generate a high-quality, quasi-uniform triangular mesh, which is then elevated to P2 quadratic elements for the Taylor-Hood formulation.

<img src="assets/p2_mesh.png" width="600">

### 2. Finite Element Assembly Core
The core of our solver involves assembling matrices by integrating products of basis functions and their derivatives over each element.

#### a) Basis Functions (`basis_function.m`)
This crucial utility function computes the values and derivatives of the P2 Lagrange basis functions on a reference triangular element. These values are pre-calculated at the Gauss quadrature points and are the fundamental building blocks for assembling all system matrices.

```matlab
function val = basis_function(p, derivative_order_x, derivative_order_y, gauss_points)
    % For P2 elements (p=2), there are 6 basis functions (Nlpj=6)
    % xi and eta are the coordinates on the reference element
    xi = gauss_points(:,1);
    eta = gauss_points(:,2);
    
    if p == 2
        N1 = (1-xi-eta).*(1-2*xi-2*eta);
        N2 = xi.*(2*xi-1);
        N3 = eta.*(2*eta-1);
        N4 = 4*xi.*(1-xi-eta);
        N5 = 4*xi.*eta;
        N6 = 4*eta.*(1-xi-eta);
        
        if derivative_order_x == 0 && derivative_order_y == 0
            val = [N1, N2, N3, N4, N5, N6];
        elseif derivative_order_x == 1 && derivative_order_y == 0 % d/d(xi)
            dN1_dxi = 4*xi + 4*eta - 3;
            dN2_dxi = 4*xi - 1;
            dN3_dxi = zeros(size(xi));
            dN4_dxi = 4 - 8*xi - 4*eta;
            dN5_dxi = 4*eta;
            dN6_dxi = -4*eta;
            val = [dN1_dxi, dN2_dxi, dN3_dxi, dN4_dxi, dN5_dxi, dN6_dxi];
        elseif derivative_order_x == 0 && derivative_order_y == 1 % d/d(eta)
            dN1_deta = 4*xi + 4*eta - 3;
            dN2_deta = zeros(size(xi));
            dN3_deta = 4*eta - 1;
            dN4_deta = -4*xi;
            dN5_deta = 4*xi;
            dN6_deta = 4 - 4*xi - 8*eta;
            val = [dN1_deta, dN2_deta, dN3_deta, dN4_deta, dN5_deta, dN6_deta];
        else
            error('Derivative order not supported for P2 elements');
        end
    else
        error('Only P2 elements (p=2) are supported');
    end
    val = val'; % Return as Nlpj x Ng matrix
end
```

#### b) Oseen Convection Matrix (`assemble_An1_v`)
This function assembles the matrix for the Oseen term `C(u_k-1; u, v) = ∫_Ω ((u_k-1 · ∇)u) · v dΩ`. The resulting matrix `An1` has entries `(An1)_ij = ∫_Ω ((u_k-1 · ∇)φ_j) φ_i dΩ`.

**Code Snippet: `assemble_An1_v.m`**
```matlab
function An1 = assemble_An1_v(P, T, Pb, Tb, gauss, weight, p, u_k_vec)
    Npb = size(Pb, 1); [Ne, Nlpj] = size(Tb); Ng = size(gauss, 1);
    phi = basis_function(p, 0, 0, gauss);
    dphix = basis_function(p, 1, 0, gauss);
    dphiy = basis_function(p, 0, 1, gauss);
    C_global = sparse(Npb, Npb);

    for ne = 1:Ne % Loop over each element
        nodes_k_pb = Tb(ne, :);
        x1=Pb(nodes_k_pb(1),1); y1=Pb(nodes_k_pb(1),2);
        x2=Pb(nodes_k_pb(2),1); y2=Pb(nodes_k_pb(2),2);
        x3=Pb(nodes_k_pb(3),1); y3=Pb(nodes_k_pb(3),2);
        
        detJ = (x2-x1)*(y3-y1) - (x3-x1)*(y2-y1);
        abs_detJ = abs(detJ);
        invJ = [(y3-y1),-(x3-x1);-(y2-y1),(x2-x1)]/detJ;

        u_k_local_nodes = u_k_vec(nodes_k_pb, :);
        Ck_local = zeros(Nlpj, Nlpj);
        
        for i = 1:Nlpj
            for j = 1:Nlpj
                integrand_value = 0;
                for k = 1:Ng % Loop over Gauss points
                    u_k_at_gauss = u_k_local_nodes' * phi(:,k);
                    
                    dphix_phys = invJ(1,1)*dphix(j,k) + invJ(2,1)*dphiy(j,k);
                    dphiy_phys = invJ(1,2)*dphix(j,k) + invJ(2,2)*dphiy(j,k);
                    
                    uk_dot_grad_phij_k = u_k_at_gauss(1) * dphix_phys + u_k_at_gauss(2) * dphiy_phys;
                    phi_i_k = phi(i, k);
                    
                    integrand_value = integrand_value + abs_detJ * weight(k) * (phi_i_k * uk_dot_grad_phij_k);
                end
                Ck_local(i,j) = integrand_value;
            end
        end
        C_global(nodes_k_pb, nodes_k_pb) = C_global(nodes_k_pb, nodes_k_pb) + Ck_local;
    end
    An1 = blkdiag(C_global, C_global);
end
```

#### c) Newton Convection Jacobian (`assemble_An2_v`)
This function assembles the other part of the convection Jacobian, `∫_Ω ((δu · ∇)u_k-1) · v dΩ`. For `δu = φ_j` and `v = φ_i`, this term couples the velocity components, resulting in a 2x2 block matrix `An2`.

**Code Snippet: `assemble_An2_v.m`**
```matlab
function An2 = assemble_An2(P, T, Pb, Tb, gauss, weight, p, u_k_vec)
    Npb = size(Pb, 1); [Ne, Nlpj] = size(Tb); Ng = size(gauss, 1);
    phi = basis_function(p, 0, 0, gauss);
    dphix = basis_function(p, 1, 0, gauss);
    dphiy = basis_function(p, 0, 1, gauss);
    
    C11_global = sparse(Npb, Npb); C12_global = sparse(Npb, Npb);
    C21_global = sparse(Npb, Npb); C22_global = sparse(Npb, Npb);

    for ne = 1:Ne % Loop over elements
        nodes_k_pb = Tb(ne, :);
        x1=Pb(nodes_k_pb(1),1); y1=Pb(nodes_k_pb(1),2);
        x2=Pb(nodes_k_pb(2),1); y2=Pb(nodes_k_pb(2),2);
        x3=Pb(nodes_k_pb(3),1); y3=Pb(nodes_k_pb(3),2);
        detJ = (x2-x1)*(y3-y1) - (x3-x1)*(y2-y1);
        abs_detJ = abs(detJ);
        invJ = [(y3-y1),-(x3-x1);-(y2-y1),(x2-x1)]/detJ;
        
        u_k_local = u_k_vec(nodes_k_pb, :);
        dphix_phys = invJ(1,1) * dphix + invJ(2,1) * dphiy;
        dphiy_phys = invJ(1,2) * dphix + invJ(2,2) * dphiy;

        duk1_dx_at_gauss = u_k_local(:, 1)' * dphix_phys;
        duk1_dy_at_gauss = u_k_local(:, 1)' * dphiy_phys;
        duk2_dx_at_gauss = u_k_local(:, 2)' * dphix_phys;
        duk2_dy_at_gauss = u_k_local(:, 2)' * dphiy_phys;

        Ck_11 = zeros(Nlpj, Nlpj); Ck_12 = zeros(Nlpj, Nlpj);
        Ck_21 = zeros(Nlpj, Nlpj); Ck_22 = zeros(Nlpj, Nlpj);

        for i = 1:Nlpj
            for j = 1:Nlpj
                int_val_11=0; int_val_12=0; int_val_21=0; int_val_22=0;
                for k = 1:Ng % Loop over Gauss points
                    common_factor = phi(i,k) * phi(j,k) * weight(k) * abs_detJ;
                    int_val_11 = int_val_11 + common_factor * duk1_dx_at_gauss(k);
                    int_val_12 = int_val_12 + common_factor * duk1_dy_at_gauss(k);
                    int_val_21 = int_val_21 + common_factor * duk2_dx_at_gauss(k);
                    int_val_22 = int_val_22 + common_factor * duk2_dy_at_gauss(k);
                end
                Ck_11(i,j) = int_val_11; Ck_12(i,j) = int_val_12;
                Ck_21(i,j) = int_val_21; Ck_22(i,j) = int_val_22;
            end
        end
        C11_global(nodes_k_pb, nodes_k_pb) = C11_global(nodes_k_pb, nodes_k_pb) + Ck_11;
        C12_global(nodes_k_pb, nodes_k_pb) = C12_global(nodes_k_pb, nodes_k_pb) + Ck_12;
        C21_global(nodes_k_pb, nodes_k_pb) = C21_global(nodes_k_pb, nodes_k_pb) + Ck_21;
        C22_global(nodes_k_pb, nodes_k_pb) = C22_global(nodes_k_pb, nodes_k_pb) + Ck_22;
    end
    An2 = [C11_global, C12_global; C21_global, C22_global];
end
```

## How to Run

### Dependencies
- **MATLAB** (R2020a or later)
- **Distmesh Toolbox**: Only required if you need to regenerate the mesh. A pre-generated `domain_mesh.mat` is included in the repository.

### Execution Steps
1.  Clone this repository to your local machine:
    ```bash
    git clone https://github.com/Silvera0218/Fluid-Dynamics-Silulation.git
    ```
2.  Open MATLAB and add the project folder to the MATLAB path.
3.  Run the desired solver script from the MATLAB command window:
    - **Steady-State Oseen Solver**: `main_oseen_ls_dbc;`
    - **Steady-State Newton Solver**: `main_newton_ls_dbc;`
    - **Unsteady Newton Solver**: `main_unsteady_ns_nonlinear_newton;`
    *You can modify parameters like viscosity `mu` inside each script.*

---

## Simulation Results and Analysis

### I. Steady-State Simulations
In steady-state simulations, we solve for the final equilibrium state of the flow field.

#### Oseen Linearization Results
| Velocity Magnitude `|u|` (μ=0.01) | V-Component of Velocity `V` (μ=0.01) |
| :---: | :---: |
| ![Oseen Norm](assets/steady_oseen_norm_mu0.01.png) | ![Oseen V-comp](assets/steady_oseen_v_mu0.01.png) |

#### Newton Linearization Results
| Velocity Magnitude `|u|` (μ=0.01) | Streamlines (μ=0.01) |
| :---: | :---: |
| ![Newton Norm](assets/steady_newton_norm_mu0.01.png) | ![Newton Streamline](assets/steady_newton_streamline_mu0.01.png) |

#### Convergence Comparison (μ=0.1)
| Oseen Convergence History | Newton Convergence History |
| :---: | :---: |
| ![Oseen Convergence](assets/steady_oseen_conv_mu0.1.png) | ![Newton Convergence](assets/steady_newton_conv_mu0.1.png) |

---

### II. Unsteady-State Simulations
For a low viscosity of `μ=0.01`, we successfully captured the classic **von Kármán vortex street**.

#### Formation and Development of the Vortex Street
The flow begins from a quasi-steady state, gradually becomes unstable, and vortices begin to shed alternately from the obstacles, forming a periodic wake.

**t = 1.0 (Initial Development)**
| Velocity Magnitude `|u|` | Streamlines |
| :---: | :---: |
| ![Unsteady t=1 Norm](assets/unsteady_t1_norm.png) | ![Unsteady t=1 Streamline](assets/unsteady_t1_streamline.png) |

**t = 5.0 (Vortex Amplification)**
| Velocity Magnitude `|u|` | Streamlines |
| :---: | :---: |
| ![Unsteady t=5 Norm](assets/unsteady_t5_norm.png) | ![Unsteady t=5 Streamline](assets/unsteady_t5_streamline.png) |

**t = 11.4 (Periodic Vortex Street)**
| Velocity Magnitude `|u|` | Streamlines |
| :---: | :---: |
| ![Unsteady t=11.4 Norm](assets/unsteady_t11.4_norm.png) | ![Unsteady t=11.4 Streamline](assets/unsteady_t11.4_streamline.png) |

## Conclusion
This project successfully established a powerful simulation platform for 2D incompressible fluid dynamics. Through the implementation and comparison of different numerical strategies, we conclude the following:
- For **steady-state problems**, the **Approximate Newton method** combined with a **line search** is the most efficient and robust solution strategy.
- For **unsteady problems**, using a **full Newton's method** within each time step accurately captures the transient evolution of the flow field, enabling the simulation of complex physical phenomena like the von Kármán vortex street.
```
