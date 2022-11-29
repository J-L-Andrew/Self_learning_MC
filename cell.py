import numpy as np
from utils import surface_area

class Cell(object):
    def __init__(self, dim, mode):
        self.dim = dim
        # origin of lattice (located in the origin by default)
        self.origin = None
        
        self.lattice = None
        
        # color
        self.color = None

    
    @property
    def volume(self):
        if self.dim == 3:
            v = np.dot(np.cross(self.state.lattice[0], self.state.lattice[1]), self.state.lattice[2])
        elif self.dim == 2:
            v = np.cross(self.state.lattice[0], self.state.lattice[1])
        return np.fabs(v)
    
    @property
    def distortion(self):
        """
        Measure the distortion of the simulation cell, for 3D now only 
        """
        norm = 0.
        for i in range(3):
            norm += np.linalg.norm(self.state.lattice[i])
    
        fun = norm * surface_area(self.state.lattice) / self.volume / 18.
        return fun

    def new_combination(self):
        """
        Replace the cell by an equivalent set of basis vectors, 
        which are shorter and more orthogonal
        """
        new_lattice = self.state.lattice.copy()
        is_terminated = True
        for i in range(3):
            for j in range(3):
                if (j == i): continue

                for k in range(2):
                    lattice = self.state.lattice.copy()
                    lattice[i] = self.state.lattice[i] + (-1)**k * self.state.lattice[j]

                    if surface_area(lattice) < surface_area(new_lattice):
                        is_terminated = False
                        new_lattice = lattice.copy()
        
        self.state.lattice = new_lattice
        return is_terminated

    def Diji_reduction(self):
        """
        Repeat the above procedure.
        """
        if self.distortion > 1.5:
            terminal = False
            iter = 0
            while (not terminal):
                terminal = self.new_combination()
                iter += 1

                # it is prudent to impose a cutoff at roughly 10 iterations
                if (iter > 10): break
      
    def LLL_reduction(self):
        pass


LLL_tiny = 1e-10


def LLL_swap(k: int, kmax: int, b: np.array, mu: np.array, H: np.array, B: np.array, bstar: np.array):
    """
    Adapted from 'Factoring Polynomials with Rational Coefficients'
    See Fig. 1. The reduction algorithm. Step(2)
    
    Parameters
    ----------
    b: lattice
    mu
    """
    
    b[:][k-1], b[:][k] = b[:][k], b[:][k-1]
    H[:][k-1], H[:][k] = H[:][k], H[:][k-1]

    if (k > 1):
        for j in range(k-1):
            mu[k-1][j], mu[k][j] = mu[k][j], mu[k-1][j]
    
    mubar = mu[k][k-1]
    Bbar = B[k] + mubar**2 * B[k-1]
    
    if (np.fabs(Bbar) < LLL_tiny):
        B[k], B[k-1] = B[k-1], B[k]
        bstar[:][k], bstar[:][k-1] = bstar[:][k-1], bstar[:][k]
        
        for i in range(k+1, kmax+1):
            mu[i][k], mu[i][k-1] = mu[i][k-1], mu[i][k]
    elif np.fabs(B[k]) < LLL_tiny and mubar != 0:
        B[k-1] = Bbar
        bstar[:][k-1] *= mubar
        mu[k][k-1] = 1. / mubar
        for i in range(k+1, kmax+1):
            mu[i][k-1] /= mubar
    elif B[k] != 0:
        t = B[k-1] / Bbar
        mu[k][k-1] = mubar * t
        bbar = bstar[:][k-1].copy()
        bstar[:][k-1] = bstar[:][k] + mubar * bbar
        bstar[k] = -mu[k][k-1]*bstar[k] + (B[k]/Bbar) * bbar
        
        B[k] *= t
        B[k-1] = Bbar
        for i in range(k+1, kmax+1):
            t = mu[i][k]
            mu[i][k] = mu[i][k-1] - mubar * t
            mu[i][k-1]  = t + mu[k][k-1] * mu[i][k]
            

def LLL_star(k: int, l: int, b: np.array, mu: np.array, H: np.array):
    if (np.fabs(mu[k][l]) <= 0.5): return b, mu, H
    
    # integer nearest to mu
    r = np.floor(0.5 + mu[k][l])
    b[:][k] -= r * b[:][l]
    H[:][k] -= r * H[:][l]
    
    for j in range(l):
        mu[k][j] -= r*mu[l][j]
    mu[k][l] -= r
    
    return b, mu, H

def LLL(dim: int, lattice: np.array):
    # lattice: (dim, dim)
    LLL_tiny = 1e-10
    
    mu = np.zeros(dim, dim)
    b_star = np.zeros(dim, dim)
    B = np.zeros(dim)

    k = 1
    kmax = 0
    H = np.diag(np.ones(dim))
    b_star[:][0] = lattice[:][0]
    B[0] = np.dot(b_star[:][0], b_star[:][0])
    
    while (k < dim):
        if (k > kmax):
              kmax = k
              b_star[:][k] = lattice[:][k]
              
              for j in range(k):
                  if (np.fabs(B[j]) < LLL_tiny): mu[k][j] = 0.
                  else: mu[k][j] = np.dot(lattice[:][k], b_star[:][j]) / B[j]
                  b_star[:][k] -= mu[k][j]*b_star[:][j]
              B[k] = np.dot(b_star[:][k], b_star[:][k])
        
        while(1):
            LLL_star(k, k-1, lattice, mu)
            if (B[k] < (0.75-mu[k][k-1]**2)*B[k-1]):
                LLL_swap(k, lattice, mu)
                k -= 1
                
                if (k < 1): k = 1
                else: 
                    for t in range(k-1):
                        l = k-2-t
                        LLL_star(k, l, lattice, mu)
                    k += 1
                    break
    
    return lattice, H
    
      
    
    
    
    
    
    
