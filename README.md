**Repository Overview:**
Quantum-NTE contains research code developed as part of my Ph.D. work on quantum algorithms for neutron transport approximations. The primary focus is the development and analysis of quantum algorithms for computing the k-eigenvalue of discretized neutron diffusion. But, code to determine the complexity of quantumly solving the linear systems resulting from discretization of fixed-source transport approximations is also investigated as well as extensions to the P_N approximation.

This work supported some of the methods released in 

Quantum Algorithms for Heterogeneous PDEs: The Neutron Diffusion Eigenvalue Problem (https://arxiv.org/abs/2604.05098)

as well as in my dissertation work that is still in progress.

**Repository Structure:**
In order of importance, the significant directories are

Quantum-NTE/

├── Fast_Inversion/      Block-encoding methods and scaling analysis for implementation of the operators in transport approximations

├── Classical_Scaling/   Classical solver scaling and convergence studies

├── Elementary_Circuits/ Quantum arithmetic and circuit primitives    

├── PN/                  P_N neutron-transport discretization and analysis

├── QLSS/                Quantum linear-system solver investigations for fixed-source transport approximations

├── QPE/                 Quantum phase estimation implementations and tests

├── Simulations/         Input files for numerical experiments and benchmark calculations

└── Plotting/            Scripts used to generate research figures

## Requirements

Python 3.11.9

Primary packages:
- NumPy
- SciPy
- matplotlib
- Qiskit
- scikit-fem

Install dependencies with:

pip install -r requirements.txt
