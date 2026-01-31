# Edge-Optimized Fourier Neural Operator for Poisson Equation

## 1. Context & Domain Background
**Read this first to understand the physical system.**

This project builds a surrogate model for the **2D Poisson Equation**. This equation describes how a physical quantity spreads across a region based on material properties and external sources.

### The Physics Variables
* **$u(x,y)$ — The Output (Field):**
    The physical state we want to predict.
    * *Examples:* Temperature distribution, voltage potential, gas concentration, or pressure.
    * *Visual:* A heatmap where high values might mean "hot" or "high voltage".
* **$a(x,y)$ — The Input (Medium):**
    A map of how easily the quantity flows at each point. This is **heterogeneous**, meaning properties change across the grid.
    * *Examples:* Thermal conductivity (heat), permittivity (electronics), or permeability (fluids).
* **$f(x,y)$ — The Input (Source):**
    A map of external drivers injecting or removing the quantity.
    * *Examples:* A heat source (CPU core), a voltage source, or a gas leak.

**The Goal:** The solver (and our AI model) takes the "Medium" ($a$) and the "Source" ($f$) and calculates the resulting "Field" ($u$).

---

## 2. Overview
**Mission:** Train, compress, and deploy a Fourier Neural Operator (FNO) surrogate for the 2D Poisson / heterogeneous diffusion problem, while keeping it hardware efficient.

**Workflow:**
1.  **Generate Data:** Create synthetic $a(x,y)$ and $f(x,y)$ maps and solve for $u(x,y)$ using a classical numerical solver (Finite Difference).
2.  **Train:** Teach an FNO to predict $u$ from $a$ and $f$ (and optional sparse sensors).
3.  **Compress:** Optimise the model (Quantization/Pruning/Distillation) to fit on memory-constrained edge hardware.
4.  **Deploy:** Export to ONNX/TensorRT and measure real-world latency, energy, and accuracy.

---

## 3. Goals (Machine-Readable)
* **Task:** Learn mapping $F: (a, f, [sensors]) \rightarrow u$ where $u$ is the PDE solution on an $H \times W$ grid.
* **Baseline:** Float32 FNO with **Relative L2 $\le$ 0.05** (config-dependent).
* **Compression:** Produce compressed model (INT8/QAT $\pm$ pruning + distillation) whose **Relative L2 $\le$ float32 + 3–5%**.
* **Deployment:** Model must fit target-device RAM and run faster than the classical solver.
* **Deliverables:** Reproducible scripts for data generation, training, eval, compress, export, and a short edge-run demo.

---

## 4. Problem Definition (Formal)
**PDE:**
$$-\nabla \cdot (a(x,y) \nabla u(x,y)) = f(x,y)$$
* **Domain:** $[0,1]^2$
* **Boundary Conditions:** Dirichlet (e.g., $u=0$ on boundary).

**Inputs ($X$):**
1.  `a(x,y)`: $H \times W$ float array (Diffusion coefficient).
2.  `f(x,y)`: $H \times W$ float array (Source term).
3.  *(Optional)* `S`: Sparse sensor readings $\{(x_i, y_i, u_i)\}$.

**Target ($Y$):**
1.  `u(x,y)`: $H \times W$ float array (True solution).

---

## 5. Data Generation Pipeline
* **Grid:** Default $H=W=64$ (prototype); scale to 128 for later tests.
* **Format:** `.npz` or `.h5`.
* **Splits:** Train (10,000), Val (2,000), Test (2,000). Prototype with 1k/200/200.
* **Recipe:**
    1.  **Sample $a(x,y)$:** Noise $\rightarrow$ Gaussian smooth ($\sigma = 3..8$) $\rightarrow$ Rescale to $[0.1, 3.0]$.
    2.  **Sample $f(x,y)$:** Sum of 1–5 Gaussian bumps (random centers/widths) OR random low-freq Fourier series.
    3.  **Solve $u_{true}$:** Finite-Difference 5-point Laplacian (Sparse Solver `scipy.sparse.linalg.spsolve`).
    4.  **Sensors:** Sample $N$ points, record $(x,y,u_{true})$, add small noise.

---




