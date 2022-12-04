"""
* Generalization of divide and concur (D-C) to any convex particle
* based on the particle's support function.

* Adapted from "Dense-packing crystal structures of physical tetrahedra".
"""

import numpy as np
from scipy import optimize
from particle.base import Particle

"""
Overlap detection between two given particles.

By the separating plane theorem, K1 and K2 do not overlap if and only if 
a vector u exists such that delta_h(u) = h1(u) + h2(-u) <= 0.We can 
determine if such a vector exists by numerically minimizing delta_h(u)/||u||.

!!!: Particles must be convex ones.
"""
def delta_h_normalized(u: np.array, K1: Particle, K2: Particle):
    """ delta_h(u) / ||u|| """
    y = (K1.support_funcs(u) + K2.support_funcs(-u)) / np.linalg.norm(u)
    return y


def is_overlap(K1: Particle, K2: Particle):
    x0 = np.ones(3)
    u_c = optimize.minimize(lambda x: delta_h_normalized(x, K1, K2), x0, method='SLSQP')
    
    y = delta_h_normalized(u_c, K1, K2)
    if (y <= 0.): return 0
    else: return 1
    

def overlap_measure(K1: Particle, K2: Particle):
    """
    According to the separating plane theorem.
    """
    x0 = np.ones(3)
    u_c = optimize.minimize(lambda x: delta_h_normalized(x, K1, K2), x0, method='SLSQP')
    
    y = delta_h_normalized(u_c, K1, K2)
    return y


"""
Exclusion projection:

If two particles overlap, we must make the minimal change possible to the 
configuration parameters so that h1'(u) + h2'(u) = 0 for some u.
"""
def config_dis(x: np.array, K1: Particle, K2: Particle):
    u, v1, v2 = x[0:3], x[3:6], x[6:9]
    r12 = K1.centroid - K2.centroid
    delta_h = K1.support_func(v1) + K2.support_func(-v2) + np.dot(r12, u)
    
    y = (delta_h**2.)/2. + np.linalg.norm(np.matmul(K1.rot_mat.T, u)-v1)**2. + np.linalg.norm(np.matmul(K2.rot_mat.T, u)-v2)**2.
    y /= np.linalg.norm(u)**2.
    
    return y


def resolve_overlap(x: np.array, K1: Particle, K2: Particle):
    u, v1, v2 = x[0:3], x[3:6], x[6:9]
    r12 = K1.centroid - K2.centroid
    delta_h = K1.support_func(v1) + K2.support_func(-v2) + np.dot(r12, u)
    
    K1.centroid -= u*delta_h/2./(np.linalg.norm(u)**2.)
    K2.centroid += u*delta_h/2./(np.linalg.norm(u)**2.)
    
    col = u.reshape(-1, 1)
    row = (v1 - np.matmul(u, K1.rot_mat)).reshape(1, -1)
    K1.rot_mat += np.matmul(col, row) / (np.linalg.norm(u)**2.)
    row = (v2 - np.matmul(u, K2.rot_mat)).reshape(1, -1)
    K2.rot_mat += np.matmul(col, row) / (np.linalg.norm(u)**2.)


def projection(K1: Particle, K2: Particle):
    """
    make the minimal change possible to the configuration parameters
    """
    x0 = np.ones(9)
    res = optimize.minimize(lambda x: config_dis(x, K1, K2), x0, method='SLSQP')
    
    resolve_overlap(res.x, K1, K2)

