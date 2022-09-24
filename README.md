# DensePacking
### Reinforcement Learning Implementation
#### Experiment Setup
?

### Adaptive shrinking cell (ASC)
ASC is an optimization scheme that can be applied to
generate dense packings of nonspherical particles. Here we list some useful references:

* Dense packings of polyhedra: Platonic and Archimedean solids [Paper](https://journals.aps.org/pre/pdf/10.1103/PhysRevE.80.041104)


### Models
#### Ellipsoid
* In 2004, Donev et al. [30] proposed a simple monoclinic crystal with two ellipsoids of different orientations per unit cell (SM2).
![figure2](https://user-images.githubusercontent.com/72123149/184534832-a22fdb2a-6d26-4572-acbf-9d685ac315bd.png)

* It was only recently that an unusual family of crystalline packings of biaxial ellipsoids was discovered, which are denser than the corresponding SM2 packings for a specific range of aspect ratios (like self-dual ellipsoids with 1.365<\alpha<1.5625.
![figure3](https://user-images.githubusercontent.com/72123149/184534880-ad3ba1bb-8cde-48ab-8ce0-6117c34490bd.png)
* Can denser packing been discovered via via a reinforcenment learning–based strategy?

#### Unit cell
  To construct possible packing, we consider cases with repeating unit cell, which contains N parrtticles. The cell's lattice repetition is governed by the translation vectors, subject to the constraint that no two particle overlap.
![2009 Dense packings of polyhedra, Platonic and Archimedean solids](https://user-images.githubusercontent.com/72123149/184535539-f55f8d2a-f6ab-40bf-ae0a-25727a11426a.jpg)

* The number of particles N is small, typically N < 12.
* The three vectors that span the simulation cell are allowed to vary independently of each other in both their
length and orientation.
* We do not make any assumptions concerning the orientation of the box here. A commonly used choice for variable-box-shape simulations is to have one of the box vectors along the x axis, another vector in the positive part of
the xy-plane, and the third in the z > 0 half space.

### Methodology
#### 1. Adaptive shrinking cell scheme.<br> 
The adaptive shrinking cell
scheme is based on the standard Monte Carlo (MC) method, where
the arbitrarily chosen particle is given random translation and rotation.
The main improvement is that the adaptive shrinking cell
scheme allows for deformation (compression/expansion) of the fundamental
cell, leading to a higher packing density. During the
procedure, a trial is rejected if any two particles overlap; otherwise,
the trial is accepted.
* Initial configurations: random, dilute, and without overlap (sometimes start from certain dilute
packings).
* Particle trial move(translation + rotation ): based on Metropolis acceptance (with no overlap), 1e3 momvements averagely per particle. The probabilitiies (sum=1) of translation and rotation are also controlled variables.
![09b389e014699f100d10cdb8b2e0003](https://user-images.githubusercontent.com/72123149/186204588-045399f1-83c2-4a3d-8759-6d5d0f44a382.png)
* Cell trial move (see Choice.02 in action design space): after the step of particle trial move. All particles will move correspondingly in this procedure. Cell trial move will be accepted when no overlap detected.
* Conduct step2 and step3 repeatedly until the system can be compressed no more.





### Experiment.01
Implemented via pymoo.
<img width="375" alt="image" src="https://user-images.githubusercontent.com/37290277/184924363-f6004a68-0cec-47ed-85e5-0105a24c5de9.png">

<img width="823" alt="image" src="https://user-images.githubusercontent.com/37290277/184924894-fb3d1d07-035c-4b02-8bc8-68bb75e36a0d.png">

