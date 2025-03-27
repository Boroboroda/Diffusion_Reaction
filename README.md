# Numerical Simulation of Semiconductor's Diffusion-Reaction Process Based on Deep-Learning Methods
![Project Banner](Pics/Task.png) 

> **Master's Thesis**  
> Friedrich-Alexander-Universität Erlangen-Nürnberg  
> Supervisors: Prof. Daniel Tenbrinck (FAU) & Dr. Christopher Straub (Fraunhofer-IISB)  
> Author: Xuepeng Cheng  
> Janurary 2025  

## 📜 Abstract
In this project, numerical simulation of the diffusion reaction process of nickel (Ni) with silicon carbide (SiC) is realized based on the physical information neural network (PINN) framework. Combining random Fourier feature embedding, self-attention mechanism and multilayer perceptron (MLP), the training loss and L2 error are reduced. The innovative application of KAN (Kolmogorov-Arnold Networks) and its variant Chebyshev-KAN in PINN is also explored.

## 🌟 Key Features
- **PINN framework improvement** <br>
  Deep learning model based on physical constraints for coupled multi-physics field simulation.
=======
- **PINN framework improvement**<br>
 [Reference: PINNs - Raissi](https://github.com/maziarraissi/PINNs)<br>
 Deep learning model based on physical constraints for coupled multi-physics field simulation.

- **Innovative Architecture Design**<br>
 -- **Random Fourier Feature Embedding:** An effective method to reduce the spectral bias of neural networks, obtained by analyzing the neural tangent kernel.<br>
 -- **Self-attention-MLP hybrid structure:** Adjust the weight of the network in the form of a moving average.<br>
 -- **KAN:** The introduction of Kolmogorov-Arnold Networks (KAN) and its variant Chebyshev-KAN, brings new directions.<br>
  [Reference: KAN](https://kindxiaoming.github.io/pykan/intro.html) <br>
  [Reference: ChebyKAN](https://github.com/SynodicMonth/ChebyKAN)<br>

## 🔬 Methodology
### 1. Probelm Modeling
The problem consists of Kirkendall Effect and Darken Theory：<br>
<p align="center">
  <img src="Pics/Kirkendall_Effect.png" alt="Project Banner"style="width:60%;"/>
</p>
$$ J_A(x) = - D_A \frac{\partial n_{A}}{\partial x} + n_{A}v $$
  
